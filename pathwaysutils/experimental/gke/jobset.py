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
"""Pathways JobSet generator and builder (Skeleton)."""
from typing import Any, Mapping
from kubernetes import client

# Core constants.
PATHWAYS_HEAD_JOB_NAME = "pathways-head"
PATHWAYS_WORKER_JOB_NAME = "pathways-worker"

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


class PathwaysJobSet:
  """Generates JobSet configuration for Pathways (Skeleton)."""

  def __init__(
      self,
      name: str,
      namespace: str,
      tpu_type: str,
      num_slices: int,
      user_pod_template: Mapping[str, Any] | None = None,
      max_restarts: int = 0,
      jobset_api_version: str = "v1alpha2",
      labels: Mapping[str, str] | None = None,
      annotations: Mapping[str, str] | None = None,
  ):
    """Initializes the instance.

    Args:
      name: Name of the JobSet.
      namespace: Namespace of the JobSet.
      tpu_type: TPU type (e.g., "v5e").
      num_slices: Number of slices.
      user_pod_template: Optional user pod template for the head job.
      max_restarts: Maximum number of restarts for the JobSet.
      jobset_api_version: API version of JobSet.
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

    # Build minimal head template (placeholder)
    self._head_job_template = self._build_minimal_job_template("head")

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
