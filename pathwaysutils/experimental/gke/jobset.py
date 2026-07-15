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

import hashlib
import json
import logging
import math
from typing import Any, Mapping, Sequence
from kubernetes import client

# GKE sidecar containers restartPolicy compatibility placeholder.

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


def _deserialize_dict(
    api_client: client.ApiClient, data_dict: Mapping[str, Any], klass: Any
) -> Any:
  class FakeResponse:

    def __init__(self, data):
      self.data = data

  return api_client.deserialize(FakeResponse(json.dumps(data_dict)), klass)


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
      user_pod_template: Mapping[str, Any] | None = None,
      main_container_name: str = "main",
      max_restarts: int = 0,
      max_slice_restarts: int = 0,
      termination_grace_period_seconds: int | None = None,
      pathways_version: str = "latest",
      jobset_api_version: str = "v1alpha2",
      elastic_slices: int = 0,
      labels: Mapping[str, str] | None = None,
      annotations: Mapping[str, str] | None = None,
  ):
    """Initializes the instance.

    Args:
      name: Name of the JobSet.
      namespace: Namespace of the JobSet.
      pathways_dir: GCS path for Pathways scratch space.
      tpu_type: TPU type (e.g., "v5e").
      topology: TPU topology (e.g., "2x2").
      num_slices: Number of slices.
      user_pod_template: Optional user pod template for the head job.
      main_container_name: Name of the main container in user_pod_template.
      max_restarts: Maximum number of restarts for the JobSet.
      max_slice_restarts: Maximum number of slice restarts.
      termination_grace_period_seconds: Optional termination grace period.
      pathways_version: Version tag for Pathways images.
      jobset_api_version: API version of JobSet.
      elastic_slices: Number of elastic slices.
      labels: Optional labels for the JobSet.
      annotations: Optional annotations for the JobSet.
    """
    self._name = name
    self._namespace = namespace
    self._jobset_api_version = jobset_api_version
    self._max_restarts = max_restarts
    self._worker_replicas = num_slices
    self._labels = dict(labels) if labels else {}
    self._annotations = dict(annotations) if annotations else {}

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
        user_pod_template=user_pod_template,
        main_container_name=main_container_name,
        elastic_slices=elastic_slices,
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
        termination_grace_period_seconds=termination_grace_period_seconds,
    )

    self._success_policy = None
    if user_pod_template:
      self._success_policy = {
          "operator": "All",
          "targetReplicatedJobs": [PATHWAYS_HEAD_JOB_NAME],
      }

  def _build_head_job_template(
      self,
      pathways_dir: str,
      num_slices: int,
      instance_type: str,
      image_tag: str,
      user_pod_template: Mapping[str, Any] | None,
      main_container_name: str,
      elastic_slices: int,
  ) -> client.V1JobTemplateSpec:
    """Builds the head job template for the JobSet.

    Args:
      pathways_dir: GCS path for Pathways scratch space.
      num_slices: Number of slices.
      instance_type: TPU instance type (e.g., "tpuv5:2x2").
      image_tag: Version tag for Pathways images.
      user_pod_template: Optional user pod template for the head job.
      main_container_name: Name of the main container in user_pod_template.
      elastic_slices: Number of elastic slices.

    Returns:
      The head job template.
    """
    rm_image = f"{DEFAULT_PATHWAYS_RM_AND_WORKER_IMAGE}:{image_tag}"
    proxy_image = f"{DEFAULT_PATHWAYS_PROXY_IMAGE}:{image_tag}"

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
        )
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

    api_client = client.ApiClient()

    if user_pod_template:
      user_template_obj = _deserialize_dict(
          api_client, user_pod_template, client.V1PodTemplateSpec
      )
      head_pod_spec = user_template_obj.spec
      head_pod_spec.host_network = True
      head_pod_spec.dns_policy = "ClusterFirstWithHostNet"

      rm_container.restart_policy = "Always"  # pyrefly: ignore[missing-attribute]
      proxy_container.restart_policy = "Always"  # pyrefly: ignore[missing-attribute]

      init_containers = head_pod_spec.init_containers or []
      init_containers.extend([rm_container, proxy_container])
      head_pod_spec.init_containers = init_containers

      # Inject JAX env vars into main container.
      jax_env = [
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
          client.V1EnvVar(name="JAX_PLATFORMS", value="proxy"),
          client.V1EnvVar(name="XCLOUD_ENVIRONMENT", value="GCP"),
          client.V1EnvVar(
              name="JAX_BACKEND_TARGET",
              value=f"grpc://$(PATHWAYS_HEAD):{PATHWAYS_PROXY_PORT}",
          ),
      ]
      containers = head_pod_spec.containers or []
      for c in containers:
        if c.name == main_container_name:
          env = c.env or []
          env.extend(jax_env)
          c.env = env
          break
      head_pod_spec.containers = containers

      annotations = user_pod_template.get("metadata", {}).get("annotations", {})
      labels = user_pod_template.get("metadata", {}).get("labels", {})
    else:
      # Headless mode.
      head_pod_spec = client.V1PodSpec(
          host_network=True,
          dns_policy="ClusterFirstWithHostNet",
          containers=[rm_container, proxy_container],
      )
      annotations = {}
      labels = {}

    if not head_pod_spec.restart_policy:
      head_pod_spec.restart_policy = "Never"

    # Default annotations
    job_annotations = {
        "alpha.jobset.sigs.k8s.io/exclusive-topology": "kubernetes.io/hostname"
    }
    job_annotations.update(annotations)

    head_job_template = client.V1JobTemplateSpec(
        metadata=client.V1ObjectMeta(annotations=job_annotations),
        spec=client.V1JobSpec(
            backoff_limit=0,
            completion_mode="Indexed",
            completions=1,
            parallelism=1,
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    annotations=job_annotations, labels=labels
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
      termination_grace_period_seconds: int | None,
  ) -> client.V1JobTemplateSpec:
    worker_image = f"{DEFAULT_PATHWAYS_RM_AND_WORKER_IMAGE}:{image_tag}"

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

    node_selector = {
        "cloud.google.com/gke-tpu-accelerator": gke_accel_type,
        "cloud.google.com/gke-tpu-topology": topology,
    }

    backoff_limit = num_vms * 4
    if max_slice_restarts > 0:
      backoff_limit = num_vms * max_slice_restarts

    worker_pod_spec = client.V1PodSpec(
        containers=[worker_container],
        node_selector=node_selector,
        volumes=[
            client.V1Volume(
                name="shared-tmp",
                host_path=client.V1HostPathVolumeSource(
                    path="/tmp", type="DirectoryOrCreate"
                ),
            )
        ],
        host_network=True,
        dns_policy="ClusterFirstWithHostNet",
        restart_policy="OnFailure",
    )
    if termination_grace_period_seconds is not None:
      worker_pod_spec.termination_grace_period_seconds = (
          termination_grace_period_seconds
      )

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
                            "cloud.google.com/gke-nodepool"
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

  def add_gcsfuse(
      self,
      containers: str | Sequence[str],
      mount_path: str,
      bucket: str,
      read_only: bool = False,
  ) -> "PathwaysJobSet":
    """Adds GCSFuse mount to specified containers."""
    bucket_hash = int(hashlib.md5(bucket.encode()).hexdigest(), 16) % (10**8)
    volume_name = f"gcsfuse-{bucket_hash}"
    volume = client.V1Volume(
        name=volume_name,
        csi=client.V1CSIVolumeSource(
            driver="gcsfuse.csi.storage.gke.io",
            volume_attributes={"bucketName": bucket},
        ),
    )
    volume_mount = client.V1VolumeMount(
        name=volume_name,
        mount_path=mount_path,
        read_only=read_only,
    )

    for job_template in (self._head_job_template, self._worker_job_template):
      pod_spec = job_template.spec.template.spec
      all_containers = (pod_spec.containers or []) + (pod_spec.init_containers or [])

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

  def _compile_config(self) -> dict[str, Any]:
    """Compiles the JobSet configuration into a dictionary."""
    with client.ApiClient() as api_client:
      serialized_head = api_client.sanitize_for_serialization(
          self._head_job_template
      )
      serialized_worker = api_client.sanitize_for_serialization(
          self._worker_job_template
      )

    replicated_jobs = [
        {
            "name": PATHWAYS_HEAD_JOB_NAME,
            "replicas": 1,
            "template": serialized_head,
        },
        {
            "name": PATHWAYS_WORKER_JOB_NAME,
            "replicas": self._worker_replicas,
            "template": serialized_worker,
        },
    ]

    jobset_config = {
        "apiVersion": f"jobset.sigs.k8s.io/{self._jobset_api_version}",
        "kind": "JobSet",
        "metadata": {
            "name": self._name,
            "namespace": self._namespace,
        },
        "spec": {
            "failurePolicy": {"maxRestarts": self._max_restarts},
            "replicatedJobs": replicated_jobs,
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
