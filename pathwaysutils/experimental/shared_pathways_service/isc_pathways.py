"""Module for connecting to a Pathways server for interactive supercomputing."""

from collections.abc import Iterator, Mapping
import contextlib
import logging
import os
import random
import string
import subprocess
from typing import Any

import jax
import pathwaysutils
from pathwaysutils.experimental.shared_pathways_service import gke_utils
from pathwaysutils.experimental.shared_pathways_service import validators


PROXY_FILEPATH = os.path.join(
    os.path.dirname(__file__), "yamls/pw-proxy.yaml"
)
# TODO(b/459935429): Hardcoding the port and using hostNetwork: true in the
# proxy YAML limits us to one proxy server pod per node. Consider alternative
# networking configurations to allow multiple proxies per node if needed.
PROXY_SERVER_PORT = 29_000

_JAX_PLATFORMS_KEY = "jax_platforms"
_JAX_PLATFORM_PROXY = "proxy"
_JAX_BACKEND_TARGET_KEY = "jax_backend_target"
_JAX_BACKEND_TARGET_HOSTNAME = "grpc://localhost"

_logger = logging.getLogger(__name__)


def _deploy_pathways_proxy_server(
    *, pathways_service: str,
    proxy_job_name: str,
    expected_instances: Mapping[Any, Any],
    gcs_scratch_location: str,
) -> None:
  """Deploys the Pathways proxy pods to the GKE cluster.

  Args:
    pathways_service: The service name and port of the Pathways head.
    proxy_job_name: The name to use for the deployed proxy.
    expected_instances: A dictionary mapping instance types to the number of
      instances.
    gcs_scratch_location: The Google Cloud Storage location to use.

  Raises:
    subprocess.CalledProcessError: If the kubectl command fails.
  """
  try:
    with open(PROXY_FILEPATH, "r") as f:
      yaml_template = f.read()
  except OSError as err:
    raise ValueError("Could not read file: " + PROXY_FILEPATH) from err

  pathways_head_hostname, pathways_head_port = pathways_service.split(":")

  # Take the first instance type and count since we only support a single
  # instance type for now.
  instance_type, count = next(iter(expected_instances.items()))
  instances_str = ",".join(instance_type for _ in range(count))

  template = string.Template(yaml_template)
  substituted_yaml = template.substitute(
      PROXY_JOB_NAME=proxy_job_name,
      PROXY_SERVER_PORT=PROXY_SERVER_PORT,
      PATHWAYS_HEAD_HOSTNAME=pathways_head_hostname,
      PATHWAYS_HEAD_PORT=pathways_head_port,
      EXPECTED_INSTANCES=instances_str,
      GCS_SCRATCH_LOCATION=gcs_scratch_location,
  )

  _logger.info("Deploying Pathways proxy: %s", proxy_job_name)
  gke_utils.deploy_gke_yaml(substituted_yaml)

  _logger.info("Successfully deployed Pathways proxy.")


class _ISCPathways:
  """Class for managing TPUs for interactive supercomputing.

  Attributes:
    cluster: The name of the GKE cluster.
    project: The GCP project ID.
    region: The GCP region.
    bucket: The Google Cloud Storage bucket to use.
    pathways_service: The service name and port of the Pathways head pod.
    expected_tpu_instances: A dictionary mapping TPU machine types to the number
      of instances.
  """

  def __init__(
      self,
      *, cluster: str,
      project: str,
      region: str,
      gcs_bucket: str,
      pathways_service: str,
      expected_tpu_instances: Mapping[Any, Any],
  ):
    """Initializes the TPU manager."""
    self.cluster = cluster
    self.project = project
    self.region = region
    self.bucket = gcs_bucket
    self.pathways_service = pathways_service
    self.expected_tpu_instances = expected_tpu_instances
    suffix = "".join(
        random.choices(string.ascii_lowercase + string.digits, k=5)
    )
    user = os.environ.get("USER", "user")
    self._proxy_job_name = f"isc-proxy-{user}-{suffix}"
    self._port_forward_process = None
    self._proxy_port = None

  def __repr__(self):
    return (
        f"_ISCPathways(cluster='{self.cluster}', project='{self.project}', "
        f"region='{self.region}', bucket='{self.bucket}', "
        f"pathways_service='{self.pathways_service}', "
        f"expected_tpu_instances={self.expected_tpu_instances}, "
        f"_proxy_job_name='{self._proxy_job_name}')"
    )

  def __enter__(self):
    """Enters the context manager, ensuring cluster exists."""
    try:
      _deploy_pathways_proxy_server(
          pathways_service=self.pathways_service,
          proxy_job_name=self._proxy_job_name,
          expected_instances=self.expected_tpu_instances,
          gcs_scratch_location=self.bucket,
      )
      # Print a link to Cloud Logging
      cloud_logging_link = gke_utils.get_log_link(
          cluster=self.cluster,
          project=self.project,
          job_name=self._proxy_job_name,
      )
      _logger.info("View proxy logs in Cloud Logging: %s", cloud_logging_link)

      proxy_pod = gke_utils.wait_for_pod(self._proxy_job_name)
      self._proxy_port, self._port_forward_process = (
          gke_utils.enable_port_forwarding(proxy_pod, PROXY_SERVER_PORT)
      )

      # Update the JAX backend to use the proxy.
      jax.config.update(_JAX_PLATFORMS_KEY, _JAX_PLATFORM_PROXY)
      jax.config.update(
          _JAX_BACKEND_TARGET_KEY,
          f"{_JAX_BACKEND_TARGET_HOSTNAME}:{self._proxy_port}",
      )
      pathwaysutils.initialize()
      _logger.info(
          "Interactive supercomputing proxy client ready for cluster '%s'.",
          self.cluster,
      )
      return self
    except Exception as e:
      _logger.exception("Error setting up Pathways proxy: %r", e)
      # If any part of setup fails after deployment, cleanup.
      self._cleanup()
      raise

  def __exit__(self, exc_type, exc_value, traceback):
    """Exits the context manager."""
    _logger.info("Exiting ISCPathways context.")
    self._cleanup()

  def _cleanup(self):
    """Cleans up resources created by the ISCPathways context."""
    if self._port_forward_process:
      self._port_forward_process.terminate()
      try:
        self._port_forward_process.wait(timeout=10)
      except subprocess.TimeoutExpired as e:
        _logger.exception(
            "Failed to terminate port forwarding process. Not treating as an "
            "error: %r",
            e,
        )

    _logger.info("Deleting Pathways proxy")
    gke_utils.delete_gke_job(self._proxy_job_name)


@contextlib.contextmanager
def connect(
    *, cluster: str,
    project: str,
    region: str,
    gcs_bucket: str,
    pathways_service: str,
    expected_tpu_instances: Mapping[str, int],
) -> Iterator["_ISCPathways"]:
  """Connects to a Pathways server if the cluster exists. If not, creates it.

  Args:
    cluster: The name of the GKE cluster.
    project: The GCP project ID.
    region: The GCP region.
    gcs_bucket: The Google Cloud Storage bucket to use for scratch space.
    pathways_service: The service name and port of the Pathways head pod.
    expected_tpu_instances: A dictionary mapping TPU machine types to the number
      of instances. For example: {"tpuv6e:2x2": 2}

  Yields:
    The Pathways manager.
  """
  validators.validate_pathways_service(pathways_service)
  validators.validate_tpu_instances(expected_tpu_instances)
  gke_utils.fetch_cluster_credentials(
      cluster_name=cluster, project_id=project, location=region
  )
  _logger.info("Starting ISCPathways context.")
  with _ISCPathways(
      cluster=cluster,
      project=project,
      region=region,
      gcs_bucket=gcs_bucket,
      pathways_service=pathways_service,
      expected_tpu_instances=expected_tpu_instances,
  ) as t:
    yield t
