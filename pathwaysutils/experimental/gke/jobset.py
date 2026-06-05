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
"""Pathways JobSet generator and builder (Head Job Config)."""

import json
import logging
from typing import Any, Mapping
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
  """Generates JobSet configuration for Pathways (with Head Job Config)."""

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

    # Build minimal worker template (placeholder)
    self._worker_job_template = self._build_minimal_job_template("worker")

    self._success_policy = None
    if user_pod_template:
      self._success_policy = {
          "operator": "All",
          "targetReplicatedJobs": [PATHWAYS_HEAD_JOB_NAME],
      }

  def _build_minimal_job_template(self, role: str) -> client.V1JobTemplateSpec:
    """Builds a minimal job template for a given role."""
    pod_spec = client.V1PodSpec(
        containers=[
            client.V1Container(name=f"placeholder-{role}", image="ubuntu")
        ]
    )
    job_spec = client.V1JobSpec(
        template=client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels={"role": role}), spec=pod_spec
        )
    )
    return client.V1JobTemplateSpec(spec=job_spec)

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

      rm_container.restart_policy = "Always"
      proxy_container.restart_policy = "Always"

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
      jobset_config["metadata"]["labels"] = self._labels
    if self._annotations:
      jobset_config["metadata"]["annotations"] = self._annotations
    if self._success_policy:
      jobset_config["spec"]["successPolicy"] = self._success_policy

    return jobset_config

  def to_dict(self) -> dict[str, Any]:
    """Returns the JobSet configuration as a dictionary."""
    return self._compile_config()
