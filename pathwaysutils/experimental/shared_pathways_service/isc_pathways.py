"""Module for connecting to a Pathways server for interactive supercomputing."""

from collections.abc import Iterable, Iterator, Mapping
import contextlib
import dataclasses
import gc
import logging
import os
import random
import string
import subprocess
import threading
import time
from typing import Any

import jax
import jax.extend.backend as jax_backend
import pathwaysutils
from pathwaysutils.experimental.shared_pathways_service import gke_utils
from pathwaysutils.experimental.shared_pathways_service import metrics_collector
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
_JAX_BACKEND_TARGET_HOSTNAME = "grpc://127.0.0.1"
DEFAULT_PROXY_IMAGE = (
    "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:latest"
)

_logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ProxyOptions:
  """Configuration options for the Pathways proxy.

  Attributes:
    use_insecure_credentials: Whether to use insecure gRPC credentials for the
      proxy server.
    xla_flags: A list of XLA flags to pass to the proxy server.
  """
  use_insecure_credentials: bool = False
  xla_flags: list[str] = dataclasses.field(default_factory=list)

  @classmethod
  def from_list(cls, options: Iterable[str] | None) -> "ProxyOptions":
    """Creates a ProxyOptions object from a list of 'key:value' strings."""
    use_insecure = False
    xla_flags = []
    for option in options or []:
      if ":" in option:
        key, value = option.split(":", 1)
        key_strip = key.strip().lower()
        if key_strip == "use_insecure_credentials":
          use_insecure = value.strip().lower() == "true"
        elif key_strip == "xla_flags":
          val_strip = value.strip()
          if (
              val_strip
              and val_strip.startswith(('"', "'"))
              and val_strip.endswith(val_strip[0])
          ):
            val_to_split = val_strip[1:-1]
          else:
            val_to_split = val_strip
          xla_flags = val_to_split.split()

    if xla_flags:
      validators.validate_xla_flags(xla_flags)

    return cls(use_insecure_credentials=use_insecure, xla_flags=xla_flags)


def _deploy_pathways_proxy_server(
    *,
    pathways_service: str,
    proxy_job_name: str,
    expected_instances: Mapping[Any, Any],
    gcs_scratch_location: str,
    proxy_server_image: str,
    proxy_options: ProxyOptions | None = None,
) -> None:
  """Deploys the Pathways proxy pods to the GKE cluster.

  Args:
    pathways_service: The service name and port of the Pathways head.
    proxy_job_name: The name to use for the deployed proxy.
    expected_instances: A dictionary mapping instance types to the number of
      instances.
    gcs_scratch_location: The Google Cloud Storage location to use.
    proxy_server_image: The image to use for the proxy server.
    proxy_options: Configuration options for the Pathways proxy. If not
      provided, no extra options will be used.

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

  proxy_options = proxy_options or ProxyOptions()

  proxy_env_str = ""
  if proxy_options.use_insecure_credentials:
    proxy_env_str = (
        '        - name: IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS\n'
        '          value: "true"\n'
    )

  proxy_args_str = ""
  if proxy_options.xla_flags:
    proxy_args_str = "\n".join(
        f"        - {flag}" for flag in proxy_options.xla_flags
    )
    proxy_args_str = "\n" + proxy_args_str

  template = string.Template(yaml_template)
  substituted_yaml = template.substitute(
      PROXY_JOB_NAME=proxy_job_name,
      PROXY_SERVER_PORT=PROXY_SERVER_PORT,
      PATHWAYS_HEAD_HOSTNAME=pathways_head_hostname,
      PATHWAYS_HEAD_PORT=pathways_head_port,
      EXPECTED_INSTANCES=instances_str,
      GCS_SCRATCH_LOCATION=gcs_scratch_location,
      PROXY_SERVER_IMAGE=proxy_server_image,
      PROXY_ENV=proxy_env_str,
      PROXY_ARGS=proxy_args_str,
  )

  _logger.info("Deploying Pathways proxy: %s", proxy_job_name)
  gke_utils.deploy_gke_yaml(substituted_yaml)

  _logger.info("Successfully deployed Pathways proxy.")


def _wait_for_placement(
    pod_name: str,
    num_slices: int,
    stream_logs_func=gke_utils.stream_pod_logs,
    metrics_collector_inst: Any = None,
    start_time: float | None = None,
    total_chips: int = 0,
) -> None:
  """Waits for the placement to be complete by checking proxy logs."""
  _logger.info("Streaming proxy logs until the placement is complete...")
  with stream_logs_func(pod_name) as log_process:
    keywords = [
        "placement",
        "Signaling to RM",
        "Transition slice",
        "FAILED_PRECONDITION",
    ]
    end_phrase = "unplaced -> placed"
    placement_count = 0

    if not log_process.stdout:
      _logger.error("Log streaming process stdout is empty. Terminating.")
      log_process.terminate()
      _, stderr = log_process.communicate()
      raise RuntimeError(
          "Failed to stream proxy logs: stdout not available.\n"
          f"STDERR: {stderr}"
      )

    for line in log_process.stdout:
      line_lower = line.lower()
      if any(keyword.lower() in line_lower for keyword in keywords):
        _logger.info("Proxy log: %s", line.strip())

      if end_phrase.lower() in line_lower:
        placement_count += 1
        if placement_count < num_slices:
          _logger.info(
              "TPU slice %d/%d placed!",
              placement_count,
              num_slices,
          )
        else:
          _logger.info("TPU placement for %d slice(s) complete!", num_slices)
          metrics_collector_inst.record_active_user(True)
          metrics_collector_inst.record_capacity_in_use(total_chips)
          if start_time:
            duration = time.time() - start_time
            metrics_collector_inst.record_assignment_time(duration)
            metrics_collector_inst.record_successful_request()
          break


def _restore_env_var(key: str, original_value: str | None) -> None:
  """Restores an environment variable to its original value or unsets it."""
  if original_value is None:
    _logger.info("Unsetting environment variable: %s", key)
    os.environ.pop(key, None)
  else:
    _logger.info(
        "Restoring environment variable '%s' to '%s'", key, original_value
    )
    os.environ[key] = original_value


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
    proxy_job_name: The name to use for the deployed proxy.
    proxy_pod_name: The name of the proxy pod, assigned during deployment.
    proxy_server_image: The image to use for the proxy server.
    proxy_options: Configuration options for the Pathways proxy.
    metrics_collector: The metrics collector instance if enabled.
    start_time: The start time of the TPU assignment.
    total_chips: The total number of TPU chips expected across all instances.
  """

  def __init__(
      self,
      *,
      cluster: str,
      project: str,
      region: str,
      gcs_bucket: str,
      pathways_service: str,
      expected_tpu_instances: Mapping[Any, Any],
      proxy_job_name: str,
      proxy_server_image: str,
      proxy_options: ProxyOptions | None = None,
      collect_service_metrics: bool = False,
  ):
    """Initializes the TPU manager."""
    self.cluster = cluster
    self.project = project
    self.region = region
    self.bucket = gcs_bucket
    self.pathways_service = pathways_service
    self.expected_tpu_instances = expected_tpu_instances
    self._proxy_job_name = proxy_job_name
    self.proxy_pod_name: str = ""
    self._port_forward_process = None
    self._proxy_port = None
    self.proxy_server_image = proxy_server_image
    self.proxy_options = proxy_options or ProxyOptions()
    self._old_jax_platforms = None
    raw_collector = (
        metrics_collector.MetricsCollector(self.project)
        if collect_service_metrics
        else None
    )
    self.metrics_collector = metrics_collector.SafeMetricsCollector(
        raw_collector
    )
    self.start_time = None
    self._old_jax_backend_target = None
    self._old_jax_platforms_config = None
    self._old_jax_backend_target_config = None
    self.total_chips = self._get_total_chips()

  def __repr__(self):
    return (
        f"_ISCPathways(cluster='{self.cluster}', project='{self.project}', "
        f"region='{self.region}', bucket='{self.bucket}', "
        f"pathways_service='{self.pathways_service}', "
        f"expected_tpu_instances={self.expected_tpu_instances}, "
        f"_proxy_job_name='{self._proxy_job_name}', "
        f"proxy_options={self.proxy_options})"
    )

  def _get_total_chips(self) -> int:
    """Calculates total chips from expected_tpu_instances."""
    total_chips = 0
    for tpu_type, count in self.expected_tpu_instances.items():
      parts = tpu_type.split(":")
      topology = parts[1]
      dimensions = [int(d) for d in topology.split("x")]
      chips_per_instance = 1
      for d in dimensions:
        chips_per_instance *= d
      total_chips += chips_per_instance * count
    return total_chips

  def __enter__(self):
    """Enters the context manager, ensuring cluster exists."""
    self.metrics_collector.record_requested_capacity(self.total_chips)

    self._old_jax_platforms = os.environ.get(_JAX_PLATFORMS_KEY.upper())
    self._old_jax_backend_target = os.environ.get(
        _JAX_BACKEND_TARGET_KEY.upper()
    )
    self._old_jax_platforms_config = getattr(
        jax.config, _JAX_PLATFORMS_KEY, None
    )
    self._old_jax_backend_target_config = getattr(
        jax.config, _JAX_BACKEND_TARGET_KEY, None
    )

    try:
      self.start_time = time.time()
      _deploy_pathways_proxy_server(
          pathways_service=self.pathways_service,
          proxy_job_name=self._proxy_job_name,
          expected_instances=self.expected_tpu_instances,
          gcs_scratch_location=self.bucket,
          proxy_server_image=self.proxy_server_image,
          proxy_options=self.proxy_options,
      )
      self.metrics_collector.record_user_waiting(True)
      cloud_logging_link = gke_utils.get_log_link(
          cluster=self.cluster,
          project=self.project,
          job_name=self._proxy_job_name,
      )
      _logger.info("View proxy logs in Cloud Logging: %s", cloud_logging_link)

      self.proxy_pod_name = gke_utils.wait_for_pod(self._proxy_job_name)
      self._proxy_port, self._port_forward_process = (
          gke_utils.enable_port_forwarding(
              self.proxy_pod_name, PROXY_SERVER_PORT
          )
      )

      # Update the JAX backend to use the proxy.
      jax_backend_target = f"{_JAX_BACKEND_TARGET_HOSTNAME}:{self._proxy_port}"
      # Update the JAX config for the inline mode of Shared Pathways Service.
      jax.config.update(_JAX_PLATFORMS_KEY, _JAX_PLATFORM_PROXY)
      jax.config.update(_JAX_BACKEND_TARGET_KEY, jax_backend_target)
      # Update the environment variables for the CLI mode of Shared Pathways
      # Service.
      os.environ[_JAX_PLATFORMS_KEY.upper()] = _JAX_PLATFORM_PROXY
      os.environ[_JAX_BACKEND_TARGET_KEY.upper()] = jax_backend_target

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

  def _cleanup(self) -> None:
    """Cleans up resources created by the ISCPathways context."""
    # Clear JAX caches and run garbage collection.
    _logger.info("Starting Pathways proxy cleanup.")
    jax_backend.clear_backends()
    jax.clear_caches()
    gc.collect()
    _logger.info("Cleared JAX caches and ran garbage collection.")

    # Terminate the port forwarding process.
    if self._port_forward_process:
      _logger.info("Terminating port forwarding process...")
      self._port_forward_process.terminate()
      try:
        self._port_forward_process.wait(timeout=10)
      except subprocess.TimeoutExpired as e:
        _logger.exception(
            "Failed to terminate port forwarding process. Not treating as an "
            "error: %r",
            e,
        )

    # Delete the proxy GKE job.
    _logger.info("Deleting Pathways proxy...")
    gke_utils.delete_gke_job(self._proxy_job_name)
    _logger.info("Pathways proxy GKE job deletion complete.")

    # Restore JAX variables.
    _logger.info("Restoring JAX env and config variables...")
    _restore_env_var(_JAX_PLATFORMS_KEY.upper(), self._old_jax_platforms)
    _restore_env_var(
        _JAX_BACKEND_TARGET_KEY.upper(), self._old_jax_backend_target
    )
    jax.config.update(_JAX_PLATFORMS_KEY, self._old_jax_platforms_config)
    jax.config.update(
        _JAX_BACKEND_TARGET_KEY, self._old_jax_backend_target_config
    )
    _logger.info("JAX variables restored.")


@contextlib.contextmanager
def connect(
    *,
    cluster: str,
    project: str,
    region: str,
    gcs_bucket: str,
    pathways_service: str,
    expected_tpu_instances: Mapping[str, int],
    proxy_job_name: str | None = None,
    proxy_server_image: str = DEFAULT_PROXY_IMAGE,
    proxy_options: ProxyOptions | None = None,
    collect_service_metrics: bool = False,
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
    proxy_job_name: The name to use for the deployed proxy. If not provided, a
      random name will be generated.
    proxy_server_image: The proxy server image to use. If not provided, a
      default will be used.
    proxy_options: Configuration options for the Pathways proxy. If not
      provided, no extra options will be used.
    collect_service_metrics: Whether to collect usage metrics for Shared
      Pathways Service.

  Yields:
    The Pathways manager.
  """
  _logger.info("Validating Pathways service and TPU instances...")
  validators.validate_pathways_service(pathways_service)
  validators.validate_tpu_instances(expected_tpu_instances)
  validators.validate_proxy_server_image(proxy_server_image)
  validators.validate_proxy_options(proxy_options)
  _logger.info("Validation complete.")
  gke_utils.fetch_cluster_credentials(
      cluster_name=cluster, project_id=project, location=region
  )
  proxy_job_name = (
      proxy_job_name or f"isc-proxy-{os.environ.get('USER', 'user')}-{''.join(
          random.choices(string.ascii_lowercase + string.digits, k=5)
      )}"
  )

  proxy_options_obj = ProxyOptions.from_list(proxy_options)

  _logger.info("Starting ISCPathways context.")
  with _ISCPathways(
      cluster=cluster,
      project=project,
      region=region,
      gcs_bucket=gcs_bucket,
      pathways_service=pathways_service,
      expected_tpu_instances=expected_tpu_instances,
      proxy_job_name=proxy_job_name,
      proxy_server_image=proxy_server_image,
      proxy_options=proxy_options_obj,
      collect_service_metrics=collect_service_metrics,
  ) as t:
    if t.proxy_pod_name:
      num_slices = sum(t.expected_tpu_instances.values())
      placement_thread = threading.Thread(
          target=_wait_for_placement,
          args=(
              t.proxy_pod_name,
              num_slices,
              gke_utils.stream_pod_logs,
              t.metrics_collector,
              t.start_time,
              t.total_chips,
          ),
          daemon=True,
      )
      placement_thread.start()
    else:
      _logger.warning(
          "proxy_pod_name not set on _ISCPathways instance, skipping background"
          " _wait_for_placement."
      )
    yield t
