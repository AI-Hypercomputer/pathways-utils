# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pathways JobSet generator and builder (with Worker Job Config)."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import posixpath
import time
from typing import Any, Mapping, Sequence

import yaml

try:
  import kubernetes
except ImportError as e:
  raise ImportError(
      "GKE utilities require `kubernetes`. "
      "Please install pathwaysutils with GKE support:\n\n"
      "    pip install 'pathwaysutils[gke]'\n"
  ) from e

from kubernetes import client
from kubernetes import config as k8s_config

_logger = logging.getLogger(__name__)

# Core constants.
PATHWAYS_HEAD_JOB_NAME = "pathways-head"
PATHWAYS_WORKER_JOB_NAME = "pathways-worker"

DEFAULT_PATHWAYS_RM_AND_WORKER_IMAGE = (
    "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server"
)
DEFAULT_PATHWAYS_PROXY_IMAGE = (
    "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server"
)

PATHWAYS_PROXY_PORT = 29000
PATHWAYS_RM_PORT = 29001
PATHWAYS_WORKER_PORT = 29005

GKE_NODEPOOL_KEY = "cloud.google.com/gke-nodepool"
CUSTOM_LIBTPU_VOLUME_NAME = "custom-libtpu"

MACHINE_TYPE_TO_TPU_VERSION_MAP = {
    "tpu7x-standard-4t": "tpu7x",
    "tpu7x": "tpu7x",
    "ct6e-standard-4t": "tpuv6e",
    "v6e": "tpuv6e",
    "ct6e-standard-8t": "tpuv6e1t",
    "ct5p-hightpu-4t": "tpuv5",
    "v5p": "tpuv5",
    "ct5lp-hightpu-4t": "tpuv5e",
    "v5e": "tpuv5e",
    "ct5lp-hightpu-8t": "tpuv5e1t",
    "ct4p-hightpu-4t": "tpuv4",
    "v4": "tpuv4",
}

MACHINE_TYPE_TO_GKE_ACCELERATOR_TYPE_MAP = {
    "tpu7x-standard-4t": "tpu7x",
    "tpu7x": "tpu7x",
    "ct6e-standard-4t": "tpu-v6e-slice",
    "v6e": "tpu-v6e-slice",
    "ct6e-standard-8t": "tpu-v6e-slice",
    "ct5p-hightpu-4t": "tpu-v5p-slice",
    "v5p": "tpu-v5p-slice",
    "ct5lp-hightpu-4t": "tpu-v5-lite-podslice",
    "v5e": "tpu-v5-lite-podslice",
    "ct5lp-hightpu-8t": "tpu-v5-lite-podslice",
    "ct4p-hightpu-4t": "tpu-v4-podslice",
    "v4": "tpu-v4-podslice",
}


def _format_image(image: str, default_tag: str) -> str:
  """Appends default_tag if image does not already specify a tag or digest."""
  if "@" in image:
    return image
  last_slash = image.rfind("/")
  if ":" in image[last_slash + 1:]:
    return image
  return f"{image}:{default_tag}"


def _deserialize_dict(
    api_client: Any, data_dict: Mapping[str, Any], klass: Any
) -> Any:
  class FakeResponse:

    def __init__(self, data):
      self.data = data

  return api_client.deserialize(FakeResponse(json.dumps(data_dict)), klass)


def _make_env_var(
    name: str, value: str | client.V1EnvVarSource
) -> client.V1EnvVar:
  """Returns an environment variable with a literal value or a value source."""
  if isinstance(value, client.V1EnvVarSource):
    return client.V1EnvVar(name=name, value_from=value)
  return client.V1EnvVar(name=name, value=value)


def _apply_container_kwargs(
    container: client.V1Container,
    container_kwargs: Mapping[str, Any] | None,
) -> None:
  """Applies caller-provided `client.V1Container` fields to a container.

  `args` and `volume_mounts` are appended to the container's defaults. `env` is
  a mapping of environment variable names to values, either strings or
  `client.V1EnvVarSource`s (e.g., a `field_ref` or `secret_key_ref`); each one
  replaces the default variable with the same name or is added after the
  defaults. Any other field replaces the default, with a warning if the
  container already sets that field.

  Args:
    container: Container to update in place.
    container_kwargs: `client.V1Container` fields to apply.

  Raises:
    ValueError: `container_kwargs` sets `name` or `image`, or a field that
      `client.V1Container` does not have.
    TypeError: `env` is not a mapping, or one of its values is neither a string
      nor a `client.V1EnvVarSource`.
  """
  if not container_kwargs:
    return
  kwargs = dict(container_kwargs)
  for field in ("name", "image"):
    if field in kwargs:
      raise ValueError(
          f"Cannot set {field!r} of the {container.name} container with"
          " container kwargs."
      )
  unknown = sorted(set(kwargs) - set(client.V1Container.openapi_types))
  if unknown:
    raise ValueError(
        f"Unknown client.V1Container fields for {container.name}: {unknown}"
    )

  extra_args = kwargs.pop("args", None)
  if extra_args:
    container.args = list(container.args or []) + list(extra_args)
  extra_volume_mounts = kwargs.pop("volume_mounts", None)
  if extra_volume_mounts:
    container.volume_mounts = list(container.volume_mounts or []) + list(
        extra_volume_mounts
    )

  env = kwargs.pop("env", None) or {}
  if not isinstance(env, Mapping):
    raise TypeError(
        f"env of the {container.name} container must be a mapping of names to"
        f" values, got {type(env).__name__}."
    )
  env = dict(env)
  for var_name, value in env.items():
    if not isinstance(value, (str, client.V1EnvVarSource)):
      raise TypeError(
          f"Environment variable {var_name} of the {container.name} container"
          " must be a str or client.V1EnvVarSource, got"
          f" {type(value).__name__}."
      )
  if env:
    merged_env = []
    for var in container.env or []:
      if var.name in env:
        value = env.pop(var.name)
        _logger.debug(
            "Overriding environment variable %s of %s with %s",
            var.name,
            container.name,
            value,
        )
        merged_env.append(_make_env_var(var.name, value))
      else:
        merged_env.append(var)
    merged_env.extend(_make_env_var(k, v) for k, v in env.items())
    container.env = merged_env

  for field, value in kwargs.items():
    if getattr(container, field) is not None:
      _logger.warning(
          "Replacing the default %s of the %s container.",
          field,
          container.name,
      )
    setattr(container, field, value)


def _merge_pod_spec_kwargs(
    defaults: Mapping[str, Any],
    pod_spec_kwargs: Mapping[str, Any] | None,
    pod_name: str,
) -> dict[str, Any]:
  """Merges caller-provided `client.V1PodSpec` fields into the defaults.

  `volumes` are merged by name: a volume replaces the default volume with the
  same name (with a warning if they differ) or is added after the defaults, so
  that required volumes such as `shared-tmp` are kept. Any other field replaces
  the default.

  Args:
    defaults: Default `client.V1PodSpec` fields.
    pod_spec_kwargs: `client.V1PodSpec` fields to apply.
    pod_name: Name of the pod, for log messages.

  Returns:
    The merged `client.V1PodSpec` fields.
  """
  merged = dict(defaults)
  if not pod_spec_kwargs:
    return merged
  overrides = dict(pod_spec_kwargs)
  extra_volumes = overrides.pop("volumes", None)
  merged.update(overrides)
  if extra_volumes:
    volumes = {v.name: v for v in merged.get("volumes") or []}
    for volume in extra_volumes:
      if volume.name in volumes and volumes[volume.name] != volume:
        _logger.warning(
            "Replacing the default %s volume of the %s pod.",
            volume.name,
            pod_name,
        )
      volumes[volume.name] = volume
    merged["volumes"] = list(volumes.values())
  return merged


def _build_rm_container(
    pathways_dir: str,
    num_slices: int,
    instance_type: str,
    rm_image: str,
    rm_container_kwargs: Mapping[str, Any] | None,
) -> client.V1Container:
  """Builds the `pathways-rm` container.

  Args:
    pathways_dir: GCS path for Pathways scratch space.
    num_slices: Number of slices.
    instance_type: TPU instance type (e.g., "tpuv5:2x2").
    rm_image: Docker image for the Resource Manager, including the tag.
    rm_container_kwargs: Optional `client.V1Container` fields for the container;
      see `PathwaysJobSet.__init__`.

  Returns:
    The `pathways-rm` container.
  """
  rm_args = [
      f"--server_port={PATHWAYS_RM_PORT}",
      f"--gcs_scratch_location={pathways_dir}",
      "--node_type=resource_manager",
      f"--instance_count={num_slices}",
      f"--instance_type={instance_type}",
  ]
  rm_env = [
      client.V1EnvVar(
          name="REPLICATED_JOB_NAME",
          value_from=client.V1EnvVarSource(
              field_ref=client.V1ObjectFieldSelector(
                  field_path="metadata.annotations['jobset.sigs.k8s.io/replicatedjob-name']"
              )
          ),
      ),
      client.V1EnvVar(
          name="JOBSET_NAME",
          value_from=client.V1EnvVarSource(
              field_ref=client.V1ObjectFieldSelector(
                  field_path=(
                      "metadata.annotations['jobset.sigs.k8s.io/jobset-name']"
                  )
              )
          ),
      ),
      client.V1EnvVar(
          name="HOST_ADDRESS",
          value_from=client.V1EnvVarSource(
              field_ref=client.V1ObjectFieldSelector(
                  field_path=(
                      "metadata.labels['jobset.sigs.k8s.io/coordinator']"
                  )
              )
          ),
      ),
      client.V1EnvVar(name="TPU_SKIP_MDS_QUERY", value="true"),
  ]
  rm_container = client.V1Container(
      name="pathways-rm",
      image=rm_image,
      image_pull_policy="Always",
      args=rm_args,
      env=rm_env,
      ports=[
          client.V1ContainerPort(
              container_port=PATHWAYS_RM_PORT, protocol="TCP"
          ),
          client.V1ContainerPort(container_port=29002, protocol="TCP"),
      ],
      resources=client.V1ResourceRequirements(
          limits={"cpu": "8", "memory": "32G"}
      ),
  )
  _apply_container_kwargs(rm_container, rm_container_kwargs)
  return rm_container


def _build_proxy_container(
    pathways_dir: str,
    elastic_slices: int,
    proxy_image: str,
    proxy_container_kwargs: Mapping[str, Any] | None,
) -> client.V1Container:
  """Builds the `pathways-proxy` container.

  Args:
    pathways_dir: GCS path for Pathways scratch space.
    elastic_slices: Number of elastic slices.
    proxy_image: Docker image for the Proxy, including the tag.
    proxy_container_kwargs: Optional `client.V1Container` fields for the
      container; see `PathwaysJobSet.__init__`.

  Returns:
    The `pathways-proxy` container.
  """
  proxy_args = [
      f"--server_port={PATHWAYS_PROXY_PORT}",
      f"--resource_manager_address=$(PATHWAYS_HEAD):{PATHWAYS_RM_PORT}",
      f"--gcs_scratch_location={pathways_dir}",
  ]
  if elastic_slices > 0:
    proxy_args.append(f"--num_elastic_slices={elastic_slices}")

  proxy_env = [
      client.V1EnvVar(
          name="PATHWAYS_HEAD",
          value_from=client.V1EnvVarSource(
              field_ref=client.V1ObjectFieldSelector(
                  field_path=(
                      "metadata.labels['jobset.sigs.k8s.io/coordinator']"
                  )
              )
          ),
      ),
      client.V1EnvVar(
          name="IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS",
          value="true",
      ),
  ]
  proxy_container = client.V1Container(
      name="pathways-proxy",
      image=proxy_image,
      image_pull_policy="Always",
      args=proxy_args,
      env=proxy_env,
      ports=[
          client.V1ContainerPort(
              container_port=PATHWAYS_PROXY_PORT, protocol="TCP"
          )
      ],
      resources=client.V1ResourceRequirements(
          limits={"cpu": "16", "memory": "100G"}
      ),
  )
  _apply_container_kwargs(proxy_container, proxy_container_kwargs)
  return proxy_container


class PathwaysJobSet:
  """JobSet configuration generator for Pathways."""

  def __init__(
      self,
      name: str,
      namespace: str,
      pathways_dir: str,
      tpu_type: str,
      topology: str,
      num_slices: int,
      max_restarts: int = 0,
      max_slice_restarts: int = 1_000_000,
      termination_grace_period_seconds: int | None = None,
      pathways_version: str = "latest",
      jobset_api_version: str = "v1alpha2",
      elastic_slices: int = 0,
      labels: Mapping[str, str] | None = None,
      annotations: Mapping[str, str] | None = None,
      shared_pathways_service: bool = False,
      pathways_rm_and_worker_image: str = DEFAULT_PATHWAYS_RM_AND_WORKER_IMAGE,
      pathways_proxy_image: str = DEFAULT_PATHWAYS_PROXY_IMAGE,
      head_pod_spec_kwargs: Mapping[str, Any] | None = None,
      worker_pod_spec_kwargs: Mapping[str, Any] | None = None,
      rm_container_kwargs: Mapping[str, Any] | None = None,
      proxy_container_kwargs: Mapping[str, Any] | None = None,
      worker_container_kwargs: Mapping[str, Any] | None = None,
  ):
    """Initializes the instance.

    Args:
      name: Name of the JobSet.
      namespace: Namespace of the JobSet.
      pathways_dir: GCS path for Pathways scratch space.
      tpu_type: TPU type (e.g., "v5e").
      topology: TPU topology (e.g., "2x2").
      num_slices: Number of slices.
      max_restarts: Maximum number of restarts for the JobSet.
      max_slice_restarts: Maximum number of slice restarts (defaults to
        1_000_000 in headless and SPS mode).
      termination_grace_period_seconds: Optional termination grace period.
      pathways_version: Version tag for Pathways images.
      jobset_api_version: API version of JobSet.
      elastic_slices: Number of elastic slices.
      labels: Optional labels for the JobSet.
      annotations: Optional annotations for the JobSet.
      shared_pathways_service: Whether to run only RM for Shared Pathways
        Service.
      pathways_rm_and_worker_image: Base Docker image for Resource Manager and
        Worker containers.
      pathways_proxy_image: Base Docker image for Proxy container.
      head_pod_spec_kwargs: Optional keyword arguments passed to
        `client.V1PodSpec` for the head pod (e.g., `node_selector`,
        `priority_class_name`, `host_network`, `dns_policy`). `volumes` are
        merged with the default volumes by name; other fields replace the
        defaults.
      worker_pod_spec_kwargs: Optional keyword arguments passed to
        `client.V1PodSpec` for the worker pods (e.g., `priority_class_name`,
        `host_network`, `dns_policy`, `volumes`). `volumes` are merged with the
        default volumes by name, so the default `shared-tmp` volume is kept
        unless a volume with the same name replaces it; other fields replace the
        defaults.
      rm_container_kwargs: Optional `client.V1Container` fields for the
        `pathways-rm` container. `args` and `volume_mounts` are appended to the
        defaults. `env` maps environment variable names to values (strings or
        `client.V1EnvVarSource`s), which replace the default variables with the
        same name or are added. Other fields (e.g., `resources`) replace the
        defaults, with a warning if the container sets them by default. `name`
        and `image` cannot be set.
      proxy_container_kwargs: Same as `rm_container_kwargs`, for the
        `pathways-proxy` container. Not allowed with `shared_pathways_service`,
        which runs no proxy.
      worker_container_kwargs: Same as `rm_container_kwargs`, for the
        `pathways-worker` container.

    Raises:
      ValueError: An argument is invalid.
    """
    if shared_pathways_service and proxy_container_kwargs:
      raise ValueError(
          "proxy_container_kwargs cannot be used with"
          " shared_pathways_service=True, which runs no proxy."
      )
    self._shared_pathways_service = shared_pathways_service
    self._pathways_rm_and_worker_image = pathways_rm_and_worker_image
    self._pathways_proxy_image = pathways_proxy_image

    self._name = name
    self._namespace = namespace
    self._jobset_api_version = jobset_api_version
    self._max_restarts = max_restarts
    self._worker_replicas = num_slices
    self._labels = dict(labels) if labels else {}
    self._annotations = dict(annotations) if annotations else {}

    if elastic_slices > num_slices:
      raise ValueError(
          f"elastic_slices ({elastic_slices}) cannot be greater than "
          f"num_slices ({num_slices})"
      )
    tpu_version = MACHINE_TYPE_TO_TPU_VERSION_MAP.get(tpu_type.lower())
    if not tpu_version:
      raise ValueError(f"Unsupported TPU type: {tpu_type}")

    gke_accel_type = MACHINE_TYPE_TO_GKE_ACCELERATOR_TYPE_MAP.get(
        tpu_type.lower()
    )

    # Calculate VMs.
    dims = [int(x) for x in topology.split("x")]
    total_chips = math.prod(dims)
    chips_per_vm = 8 if tpu_type.lower().endswith("8t") else 4
    if total_chips < chips_per_vm:
      num_vms = 1
    else:
      num_vms = total_chips // chips_per_vm

    instance_type = f"{tpu_version}:{topology}"
    image_tag = pathways_version

    # Build head template.
    self._head_job_template = self._build_head_job_template(
        pathways_dir=pathways_dir,
        num_slices=num_slices,
        instance_type=instance_type,
        image_tag=image_tag,
        elastic_slices=elastic_slices,
        shared_pathways_service=shared_pathways_service,
        pathways_rm_and_worker_image=pathways_rm_and_worker_image,
        pathways_proxy_image=pathways_proxy_image,
        head_pod_spec_kwargs=head_pod_spec_kwargs,
        rm_container_kwargs=rm_container_kwargs,
        proxy_container_kwargs=proxy_container_kwargs,
    )

    worker_kwargs = (
        dict(worker_pod_spec_kwargs) if worker_pod_spec_kwargs else {}
    )
    if termination_grace_period_seconds is not None:
      worker_kwargs.setdefault(
          "termination_grace_period_seconds", termination_grace_period_seconds
      )

    # Build worker template.
    self._worker_job_template = self._build_worker_job_template(
        pathways_dir=pathways_dir,
        num_vms=num_vms,
        chips_per_vm=chips_per_vm,
        gke_accel_type=gke_accel_type,  # pyrefly: ignore[bad-argument-type]
        topology=topology,
        image_tag=image_tag,
        max_slice_restarts=max_slice_restarts,
        pathways_rm_and_worker_image=pathways_rm_and_worker_image,
        worker_pod_spec_kwargs=worker_kwargs,
        worker_container_kwargs=worker_container_kwargs,
    )

    self._success_policy = None

  @property
  def head_job_template(self) -> client.V1JobTemplateSpec:
    return self._head_job_template

  @property
  def worker_job_template(self) -> client.V1JobTemplateSpec:
    return self._worker_job_template

  @property
  def pathways_rm_and_worker_image(self) -> str:
    return self._pathways_rm_and_worker_image

  @property
  def pathways_proxy_image(self) -> str:
    return self._pathways_proxy_image

  def _build_head_job_template(
      self,
      pathways_dir: str,
      num_slices: int,
      instance_type: str,
      image_tag: str,
      elastic_slices: int,
      shared_pathways_service: bool,
      pathways_rm_and_worker_image: str,
      pathways_proxy_image: str,
      head_pod_spec_kwargs: Mapping[str, Any] | None = None,
      rm_container_kwargs: Mapping[str, Any] | None = None,
      proxy_container_kwargs: Mapping[str, Any] | None = None,
  ) -> client.V1JobTemplateSpec:
    """Builds the head job template for the JobSet.

    Args:
      pathways_dir: GCS path for Pathways scratch space.
      num_slices: Number of slices.
      instance_type: TPU instance type (e.g., "tpuv5:2x2").
      image_tag: Version tag for Pathways images.
      elastic_slices: Number of elastic slices.
      shared_pathways_service: Whether to run only RM for Shared Pathways
        Service.
      pathways_rm_and_worker_image: Base Docker image for Resource Manager.
      pathways_proxy_image: Base Docker image for Proxy container.
      head_pod_spec_kwargs: Optional keyword arguments passed to
        `client.V1PodSpec` for the head pod.
      rm_container_kwargs: Optional `client.V1Container` fields for the
        `pathways-rm` container; see `__init__`.
      proxy_container_kwargs: Optional `client.V1Container` fields for the
        `pathways-proxy` container; see `__init__`.

    Returns:
      The head job template.
    """
    containers = [
        _build_rm_container(
            pathways_dir=pathways_dir,
            num_slices=num_slices,
            instance_type=instance_type,
            rm_image=_format_image(pathways_rm_and_worker_image, image_tag),
            rm_container_kwargs=rm_container_kwargs,
        )
    ]
    if not shared_pathways_service:
      containers.append(
          _build_proxy_container(
              pathways_dir=pathways_dir,
              elastic_slices=elastic_slices,
              proxy_image=_format_image(pathways_proxy_image, image_tag),
              proxy_container_kwargs=proxy_container_kwargs,
          )
      )

    pod_spec_kwargs = _merge_pod_spec_kwargs(
        {
            "containers": containers,
            "restart_policy": "Never",
            "node_selector": {GKE_NODEPOOL_KEY: "cpu-np"},
        },
        head_pod_spec_kwargs,
        PATHWAYS_HEAD_JOB_NAME,
    )

    head_pod_spec = client.V1PodSpec(**pod_spec_kwargs)

    job_annotations = {
        "alpha.jobset.sigs.k8s.io/exclusive-topology": "kubernetes.io/hostname"
    }

    head_job_template = client.V1JobTemplateSpec(
        metadata=client.V1ObjectMeta(annotations=job_annotations),
        spec=client.V1JobSpec(
            backoff_limit=0,
            completion_mode="Indexed",
            completions=1,
            parallelism=1,
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    annotations=job_annotations, labels={}
                ),
                spec=head_pod_spec,
            ),
        ),
    )
    return head_job_template

  def _build_worker_job_template(
      self,
      pathways_dir: str,
      num_vms: int,
      chips_per_vm: int,
      gke_accel_type: str,
      topology: str,
      image_tag: str,
      max_slice_restarts: int,
      pathways_rm_and_worker_image: str,
      worker_pod_spec_kwargs: Mapping[str, Any] | None = None,
      worker_container_kwargs: Mapping[str, Any] | None = None,
  ) -> client.V1JobTemplateSpec:
    """Builds the worker job template for the JobSet."""
    worker_image = _format_image(pathways_rm_and_worker_image, image_tag)

    args = [
        f"--resource_manager_address=$(PATHWAYS_HEAD):{PATHWAYS_RM_PORT}",
        f"--server_port={PATHWAYS_WORKER_PORT}",
        f"--gcs_scratch_location={pathways_dir}",
    ]
    worker_env = [
        client.V1EnvVar(name="TPU_MIN_LOG_LEVEL", value="0"),
        client.V1EnvVar(name="TF_CPP_MIN_LOG_LEVEL", value="0"),
        client.V1EnvVar(name="XCLOUD_ENVIRONMENT", value="GCP"),
        client.V1EnvVar(name="MEGASCALE_GRPC_ENABLE_XOR_TRACER", value="false"),
        client.V1EnvVar(
            name="MEGASCALE_NUM_SLICES",
            value_from=client.V1EnvVarSource(
                field_ref=client.V1ObjectFieldSelector(
                    field_path="metadata.labels['jobset.sigs.k8s.io/replicatedjob-replicas']"
                )
            ),
        ),
        client.V1EnvVar(
            name="JOBSET_NAME",
            value_from=client.V1EnvVarSource(
                field_ref=client.V1ObjectFieldSelector(
                    field_path=(
                        "metadata.annotations['jobset.sigs.k8s.io/jobset-name']"
                    )
                )
            ),
        ),
        client.V1EnvVar(
            name="REPLICATED_JOB_NAME",
            value_from=client.V1EnvVarSource(
                field_ref=client.V1ObjectFieldSelector(
                    field_path="metadata.annotations['jobset.sigs.k8s.io/replicatedjob-name']"
                )
            ),
        ),
        client.V1EnvVar(
            name="MEGASCALE_SLICE_ID",
            value_from=client.V1EnvVarSource(
                field_ref=client.V1ObjectFieldSelector(
                    field_path="metadata.labels['jobset.sigs.k8s.io/job-index']"
                )
            ),
        ),
        client.V1EnvVar(
            name="PATHWAYS_HEAD",
            value_from=client.V1EnvVarSource(
                field_ref=client.V1ObjectFieldSelector(
                    field_path=(
                        "metadata.labels['jobset.sigs.k8s.io/coordinator']"
                    )
                )
            ),
        ),
        client.V1EnvVar(
            name="MEGASCALE_COORDINATOR_ADDRESS",
            value_from=client.V1EnvVarSource(
                field_ref=client.V1ObjectFieldSelector(
                    field_path=(
                        "metadata.labels['jobset.sigs.k8s.io/coordinator']"
                    )
                )
            ),
        ),
    ]

    worker_container = client.V1Container(
        name="pathways-worker",
        image=worker_image,
        image_pull_policy="Always",
        args=args,
        env=worker_env,
        ports=[
            client.V1ContainerPort(
                container_port=PATHWAYS_WORKER_PORT, protocol="TCP"
            ),
            client.V1ContainerPort(container_port=29006, protocol="TCP"),
            client.V1ContainerPort(container_port=8471, protocol="TCP"),
            client.V1ContainerPort(container_port=8080, protocol="TCP"),
        ],
        volume_mounts=[
            client.V1VolumeMount(name="shared-tmp", mount_path="/tmp")
        ],
        resources=client.V1ResourceRequirements(
            limits={"google.com/tpu": str(chips_per_vm)}
        ),
    )
    _apply_container_kwargs(worker_container, worker_container_kwargs)

    node_selector = {
        "cloud.google.com/gke-tpu-accelerator": gke_accel_type,
        "cloud.google.com/gke-tpu-topology": topology,
    }

    backoff_limit = num_vms * max_slice_restarts

    pod_spec_kwargs = _merge_pod_spec_kwargs(
        {
            "containers": [worker_container],
            "node_selector": node_selector,
            "volumes": [
                client.V1Volume(
                    name="shared-tmp",
                    host_path=client.V1HostPathVolumeSource(
                        path="/tmp", type="DirectoryOrCreate"
                    ),
                )
            ],
            "restart_policy": "OnFailure",
        },
        worker_pod_spec_kwargs,
        PATHWAYS_WORKER_JOB_NAME,
    )

    worker_pod_spec = client.V1PodSpec(**pod_spec_kwargs)

    worker_job_template = client.V1JobTemplateSpec(
        metadata=client.V1ObjectMeta(),
        spec=client.V1JobSpec(
            backoff_limit=backoff_limit,
            completion_mode="Indexed",
            completions=num_vms,
            parallelism=num_vms,
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    annotations={
                        "alpha.jobset.sigs.k8s.io/exclusive-topology": (
                            GKE_NODEPOOL_KEY
                        )
                    }
                ),
                spec=worker_pod_spec,
            ),
        ),
    )
    return worker_job_template

  def _filter_matching_containers(
      self,
      containers_param: str | Sequence[str],
      all_containers: list[client.V1Container],
  ) -> list[client.V1Container]:
    """Filters containers matching the containers_param selection."""
    if containers_param == "all":
      return all_containers
    if containers_param == "worker":
      return [c for c in all_containers if c.name == "pathways-worker"]

    filter_names = (
        [containers_param]
        if isinstance(containers_param, str)
        else set(containers_param)
    )
    return [c for c in all_containers if c.name in filter_names]

  def _enable_gcsfuse_annotations(
      self, job_template: client.V1JobTemplateSpec
  ) -> None:
    """Enables gke-gcsfuse/volumes annotation on job and pod metadata."""
    job_metadata = job_template.metadata or client.V1ObjectMeta()
    job_annotations = job_metadata.annotations or {}
    job_annotations["gke-gcsfuse/volumes"] = "true"
    job_metadata.annotations = job_annotations
    job_template.metadata = job_metadata

    pod_metadata = job_template.spec.template.metadata or client.V1ObjectMeta()
    pod_annotations = pod_metadata.annotations or {}
    pod_annotations["gke-gcsfuse/volumes"] = "true"
    pod_metadata.annotations = pod_annotations
    job_template.spec.template.metadata = pod_metadata

  def _add_volume_to_pod_spec(
      self, pod_spec: client.V1PodSpec, volume: client.V1Volume
  ) -> None:
    """Appends volume to pod_spec if not already present."""
    volumes = pod_spec.volumes or []
    if not any(v.name == volume.name for v in volumes):
      volumes.append(volume)
      pod_spec.volumes = volumes

  def add_user_workload(
      self,
      image: str,
      command: Sequence[str] | str,
      name: str = "user-workload",
      **container_kwargs: Any,
  ) -> "PathwaysJobSet":
    """Adds a user workload container to the head pod.

    Args:
      image: Docker image for the user workload.
      command: Command to execute in the container (string or sequence of
        strings).
      name: Container name for the user workload (default: 'user-workload').
      **container_kwargs: Optional `client.V1Container` fields (e.g., `env`,
        `resources`, `volume_mounts`), merged like `rm_container_kwargs` of
        `__init__`. `env` maps names to values, which replace the default JAX
        environment variables with the same name or are added.

    Returns:
      The PathwaysJobSet instance for chaining.
    """
    pod_spec = self._head_job_template.spec.template.spec

    cmd = ["sh", "-c", command] if isinstance(command, str) else list(command)

    default_env = {
        "JAX_PLATFORMS": "proxy",
        "JAX_BACKEND_TARGET": f"grpc://localhost:{PATHWAYS_PROXY_PORT}",
        "IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS": "true",
    }
    user_container = client.V1Container(
        name=name,
        image=image,
        image_pull_policy="Always",
        command=cmd,
        env=[client.V1EnvVar(name=k, value=v) for k, v in default_env.items()],
    )
    _apply_container_kwargs(user_container, container_kwargs)

    init_containers = list(pod_spec.init_containers or [])
    containers = []
    for c in pod_spec.containers:
      if c.name in ("pathways-rm", "pathways-proxy"):
        c.restart_policy = "Always"  # pyrefly: ignore[missing-attribute]
        init_containers.append(c)
      elif c.name != name:
        containers.append(c)
    containers.append(user_container)
    pod_spec.init_containers = init_containers
    pod_spec.containers = containers

    if not self._shared_pathways_service:
      self._success_policy = {
          "operator": "All",
          "targetReplicatedJobs": [PATHWAYS_HEAD_JOB_NAME],
      }

    return self

  def add_colocated_python(
      self,
      image: str,
      shm_mount_path: str = "/tmp/shared-memory",
      shm_size_limit: str | None = None,
      **container_kwargs: Any,
  ) -> "PathwaysJobSet":
    """Adds colocated python sidecar to the worker pods.

    The sidecar runs as a native sidecar (an init container with
    `restartPolicy: Always`), so Kubernetes starts it before `pathways-worker`,
    which connects to it during placement initialization, and stops it when
    `pathways-worker` exits. The sidecar and the `pathways-worker` container
    share a memory-backed volume mounted at `shm_mount_path`, and
    `pathways-worker` is started with
    `--cloud_pathways_sidecar_shm_directory=<shm_mount_path>` so that large
    tensors are transferred through shared host memory.

    Args:
      image: Docker image for the colocated python sidecar.
      shm_mount_path: Mount path of the shared memory volume in both the sidecar
        and the `pathways-worker` container.
      shm_size_limit: Optional size limit of the shared memory volume (e.g.,
        "16Gi"). If `None`, no `sizeLimit` is set: the memory-backed volume
        can grow up to the pod's memory limit (or the node's allocatable
        memory if the pod has no limit), and data written to it counts toward
        the pod's memory usage.
      **container_kwargs: Optional `client.V1Container` fields for the sidecar
        (e.g., `env`, `resources`, `volume_mounts`), merged like
        `rm_container_kwargs` of `__init__`. `env` maps names to values, which
        replace the default sidecar environment variables with the same name or
        are added. `volume_mounts` are appended to the default `shared-tmp` and
        shared memory mounts.

    Returns:
      The PathwaysJobSet instance for chaining.
    """
    pod_spec = self._worker_job_template.spec.template.spec

    # Add shared memory volume if not exists.
    volumes = pod_spec.volumes or []
    shm_volume_name = "shared-memory"
    shm_exists = any(v.name == shm_volume_name for v in volumes)
    if not shm_exists:
      volumes.append(
          client.V1Volume(
              name=shm_volume_name,
              empty_dir=client.V1EmptyDirVolumeSource(
                  medium="Memory", size_limit=shm_size_limit
              ),
          )
      )
      pod_spec.volumes = volumes

    # Add colocated python container.
    colocated_container = client.V1Container(
        name="colocated-python-sidecar",
        image=image,
        image_pull_policy="Always",
        env=[
            client.V1EnvVar(name="GRPC_SERVER_ADDRESS", value="0.0.0.0:50051"),
            client.V1EnvVar(
                name="CLOUD_PATHWAYS_SIDECAR_SHM_DIRECTORY",
                value=shm_mount_path,
            ),
        ],
        ports=[client.V1ContainerPort(container_port=50051)],
        volume_mounts=[
            client.V1VolumeMount(name="shared-tmp", mount_path="/tmp"),
            client.V1VolumeMount(
                name=shm_volume_name, mount_path=shm_mount_path
            ),
        ],
    )
    _apply_container_kwargs(colocated_container, container_kwargs)
    # Run the sidecar as a native sidecar so that it is started before, and
    # stopped after, pathways-worker.
    colocated_container.restart_policy = "Always"  # pyrefly: ignore[missing-attribute]

    init_containers = list(pod_spec.init_containers or [])
    init_containers.append(colocated_container)
    pod_spec.init_containers = init_containers

    # Add volume mount and shared memory flag to pathways-worker.
    for container in pod_spec.containers:
      if container.name == "pathways-worker":
        # The Pathways worker only enables shared memory transfers to the
        # sidecar when this flag is set.
        args = container.args or []
        if not any(
            a.startswith("--cloud_pathways_sidecar_shm_directory=")
            for a in args
        ):
          args.append(
              f"--cloud_pathways_sidecar_shm_directory={shm_mount_path}"
          )
        container.args = args
        volume_mounts = container.volume_mounts or []
        volume_mounts.append(
            client.V1VolumeMount(
                name=shm_volume_name, mount_path=shm_mount_path
            )
        )
        container.volume_mounts = volume_mounts
        # Add env var for shm dir.
        worker_env = container.env or []
        worker_env.append(
            client.V1EnvVar(
                name="CLOUD_PATHWAYS_SIDECAR_SHM_DIRECTORY",
                value=shm_mount_path,
            )
        )
        container.env = worker_env
        break

    return self

  def add_gcsfuse(
      self,
      containers: str | Sequence[str],
      mount_path: str,
      bucket: str,
      read_only: bool = False,
      mount_options: str = "implicit-dirs",
  ) -> "PathwaysJobSet":
    """Adds GCSFuse mount to specified containers."""
    bucket_hash = int(hashlib.md5(bucket.encode()).hexdigest(), 16) % (10**8)
    volume_name = f"gcsfuse-{bucket_hash}"
    volume_attributes = {"bucketName": bucket}
    if mount_options:
      volume_attributes["mountOptions"] = mount_options
    volume = client.V1Volume(
        name=volume_name,
        csi=client.V1CSIVolumeSource(
            driver="gcsfuse.csi.storage.gke.io",
            volume_attributes=volume_attributes,
        ),
    )
    volume_mount = client.V1VolumeMount(
        name=volume_name,
        mount_path=mount_path,
        read_only=read_only,
    )

    for job_template in (self._head_job_template, self._worker_job_template):
      pod_spec = job_template.spec.template.spec
      all_containers = (pod_spec.containers or []) + (
          pod_spec.init_containers or []
      )

      matching = self._filter_matching_containers(containers, all_containers)
      if not matching:
        continue

      self._enable_gcsfuse_annotations(job_template)
      self._add_volume_to_pod_spec(pod_spec, volume)

      for container in matching:
        volume_mounts = container.volume_mounts or []
        volume_mounts.append(volume_mount)
        container.volume_mounts = volume_mounts

    return self

  def with_custom_libtpu(
      self,
      libtpu_path: str,
      containers: str | Sequence[str] = "all",
      mount_host_dir: bool = False,
  ) -> "PathwaysJobSet":
    """Overrides `TPU_LIBRARY_PATH` in the selected Pathways containers.

    Args:
      libtpu_path: Absolute container path to `libtpu.so`.
      containers: Target container selector (e.g. `"all"` or `"worker"`).
      mount_host_dir: If True, mounts the parent directory of `libtpu_path` from
        the host node into matching containers.

    Returns:
      Self for method chaining.

    Raises:
      ValueError: If `libtpu_path` is not a valid absolute file path or has no
        parent subdirectory when `mount_host_dir=True`.
    """
    stripped = libtpu_path.strip()
    if not stripped.startswith("/") or stripped.endswith("/"):
      raise ValueError(
          "libtpu_path must be an absolute container file path (got"
          f" {libtpu_path!r})."
      )
    path = posixpath.normpath(stripped)
    parent_dir = posixpath.dirname(path)
    if mount_host_dir and parent_dir == "/":
      raise ValueError(
          f"mount_host_dir=True requires a subdirectory (got {libtpu_path!r})."
      )

    for job_template in (self._head_job_template, self._worker_job_template):
      pod_spec = job_template.spec.template.spec
      all_containers = (pod_spec.containers or []) + (
          pod_spec.init_containers or []
      )
      matching = self._filter_matching_containers(containers, all_containers)
      if not matching:
        continue

      if mount_host_dir:
        self._add_volume_to_pod_spec(
            pod_spec,
            client.V1Volume(
                name=CUSTOM_LIBTPU_VOLUME_NAME,
                host_path=client.V1HostPathVolumeSource(
                    path=parent_dir, type="DirectoryOrCreate"
                ),
            ),
        )
      for container in matching:
        if mount_host_dir:
          volume_mounts = container.volume_mounts or []
          volume_mounts.append(
              client.V1VolumeMount(
                  name=CUSTOM_LIBTPU_VOLUME_NAME,
                  mount_path=parent_dir,
                  read_only=True,
              )
          )
          container.volume_mounts = volume_mounts
        env = container.env or []
        env.append(client.V1EnvVar(name="TPU_LIBRARY_PATH", value=path))
        container.env = env

    return self

  def _compile_config(self) -> dict[str, Any]:
    """Compiles the JobSet configuration into a dictionary."""
    with client.ApiClient() as api_client:
      serialized_head = api_client.sanitize_for_serialization(
          self._head_job_template
      )
      serialized_worker = api_client.sanitize_for_serialization(
          self._worker_job_template
      )

    head_job = {
        "name": PATHWAYS_HEAD_JOB_NAME,
        "replicas": 1,
        "template": serialized_head,
    }
    worker_job = {
        "name": PATHWAYS_WORKER_JOB_NAME,
        "replicas": self._worker_replicas,
        "template": serialized_worker,
    }

    coordinator = {
        "replicatedJob": PATHWAYS_HEAD_JOB_NAME,
    }

    failure_policy: dict[str, Any] = {
        "restartStrategy": "Recreate",
    }
    if self._max_restarts > 0:
      failure_policy["maxRestarts"] = self._max_restarts

    jobset_config = {
        "apiVersion": f"jobset.x-k8s.io/{self._jobset_api_version}",
        "kind": "JobSet",
        "metadata": {
            "name": self._name,
            "namespace": self._namespace,
        },
        "spec": {
            "startupPolicy": {"startupPolicyOrder": "InOrder"},
            "failurePolicy": failure_policy,
            "network": {
                "enableDNSHostnames": True,
                "publishNotReadyAddresses": True,
            },
            "coordinator": coordinator,
            "replicatedJobs": [head_job, worker_job],
        },
    }
    if self._labels:
      jobset_config["metadata"]["labels"] = self._labels  # pyrefly: ignore[bad-assignment]
    if self._annotations:
      jobset_config["metadata"]["annotations"] = self._annotations  # pyrefly: ignore[bad-assignment]
    if self._success_policy:
      jobset_config["spec"]["successPolicy"] = self._success_policy  # pyrefly: ignore[bad-assignment]

    return jobset_config

  def to_dict(self) -> dict[str, Any]:
    """Returns the JobSet configuration as a dictionary."""
    return self._compile_config()

  def export_yaml(self, filepath: str) -> None:
    """Exports the JobSet configuration to a YAML file."""
    with open(filepath, "w") as f:
      yaml.dump(self.to_dict(), f, default_flow_style=False)

  @classmethod
  def import_yaml(cls, filepath: str) -> "PathwaysJobSet":
    """Imports a JobSet configuration from a YAML file."""
    with open(filepath, "r") as f:
      config = yaml.safe_load(f)

    cls._validate_config(config)

    instance = cls.__new__(cls)
    instance._name = config["metadata"]["name"]
    instance._namespace = config["metadata"].get("namespace", "default")
    api_version_parts = config.get("apiVersion", "").split("/")
    instance._jobset_api_version = (
        api_version_parts[-1] if len(api_version_parts) > 1 else "v1alpha2"
    )
    instance._max_restarts = (
        config["spec"].get("failurePolicy", {}).get("maxRestarts", 0)
    )
    instance._labels = config["metadata"].get("labels", {})
    instance._annotations = config["metadata"].get("annotations", {})

    # Extract replicated jobs and deserialize.
    head_job_template: client.V1JobTemplateSpec | None = None
    worker_job_template: client.V1JobTemplateSpec | None = None

    with client.ApiClient() as api_client:
      for job in config["spec"]["replicatedJobs"]:
        if job["name"] == PATHWAYS_HEAD_JOB_NAME:
          head_job_template = _deserialize_dict(
              api_client, job["template"], client.V1JobTemplateSpec
          )
        elif job["name"] in ("worker", PATHWAYS_WORKER_JOB_NAME):
          worker_job_template = _deserialize_dict(
              api_client, job["template"], client.V1JobTemplateSpec
          )
          instance._worker_replicas = job["replicas"]

    if head_job_template is None:
      raise ValueError(f"Missing head job ({PATHWAYS_HEAD_JOB_NAME}) in config")
    if worker_job_template is None:
      raise ValueError(
          f"Missing worker job ({PATHWAYS_WORKER_JOB_NAME}) in config"
      )

    instance._head_job_template = head_job_template
    instance._worker_job_template = worker_job_template

    rm_image = DEFAULT_PATHWAYS_RM_AND_WORKER_IMAGE
    proxy_image = DEFAULT_PATHWAYS_PROXY_IMAGE
    if (
        head_job_template.spec
        and head_job_template.spec.template
        and head_job_template.spec.template.spec
    ):
      pod_spec = head_job_template.spec.template.spec
      all_containers = (pod_spec.containers or []) + (
          pod_spec.init_containers or []
      )
      for c in all_containers:
        if c.name == "pathways-rm" and c.image:
          rm_image = c.image
        elif c.name == "pathways-proxy" and c.image:
          proxy_image = c.image
    instance._pathways_rm_and_worker_image = rm_image
    instance._pathways_proxy_image = proxy_image

    instance._success_policy = config["spec"].get("successPolicy")
    return instance

  @classmethod
  def _validate_config(cls, config: dict[str, Any]) -> None:
    """Validates that the config is a valid Pathways JobSet."""
    if config.get("kind") != "JobSet":
      raise ValueError("Resource kind is not JobSet")
    jobs = {
        j["name"]: j for j in config.get("spec", {}).get("replicatedJobs", [])
    }
    if "head" not in jobs and PATHWAYS_HEAD_JOB_NAME not in jobs:
      raise ValueError(
          f"Missing head replicated job ('head' or '{PATHWAYS_HEAD_JOB_NAME}')"
      )
    if "worker" not in jobs and PATHWAYS_WORKER_JOB_NAME not in jobs:
      raise ValueError(
          "Missing worker replicated job ('worker' or"
          f" '{PATHWAYS_WORKER_JOB_NAME}')"
      )

  def apply(
      self, recreate: bool = False, field_manager: str = "pathwaysutils"
  ) -> None:
    """Applies the JobSet to the GKE cluster."""

    try:
      k8s_config.load_kube_config()
    except Exception:  # pylint: disable=broad-except
      try:
        k8s_config.load_incluster_config()
      except Exception as e:
        raise RuntimeError("Failed to load Kubernetes configuration") from e

    api = client.CustomObjectsApi()
    group = "jobset.x-k8s.io"
    version = self._jobset_api_version
    plural = "jobsets"

    exists = False
    try:
      api.get_namespaced_custom_object(
          group, version, self._namespace, plural, self._name
      )
      exists = True
    except client.rest.ApiException as e:
      if e.status != 404:
        raise

    if exists:
      if recreate:
        _logger.info(
            "JobSet %s already exists. Deleting it first...", self._name
        )
        api.delete_namespaced_custom_object(
            group, version, self._namespace, plural, self._name
        )

        # Poll for deletion.
        max_retries = 30
        for i in range(max_retries):
          try:
            api.get_namespaced_custom_object(
                group, version, self._namespace, plural, self._name
            )
            _logger.info(
                "Waiting for JobSet %s to be deleted... (%d/%d)",
                self._name,
                i + 1,
                max_retries,
            )
            time.sleep(2)
          except client.rest.ApiException as e:
            if e.status == 404:
              _logger.info("JobSet %s deleted.", self._name)
              break
            raise
        else:
          raise RuntimeError(
              f"Timeout waiting for JobSet {self._name} to be deleted"
          )
      else:
        raise RuntimeError(
            f"JobSet {self._name} already exists. Use recreate=True to"
            " overwrite."
        )

    _logger.info("Creating JobSet %s...", self._name)
    api.create_namespaced_custom_object(
        group=group,
        version=version,
        namespace=self._namespace,
        plural=plural,
        body=self.to_dict(),
        field_manager=field_manager,
    )
    _logger.info("JobSet %s created successfully.", self._name)
