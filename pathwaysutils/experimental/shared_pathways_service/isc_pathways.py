"""Module for connecting to a Pathways server for interactive supercomputing."""

import atexit
from collections.abc import Iterable, Iterator, Mapping, Sequence
import contextlib
import dataclasses
import gc
import logging
import os
import random
import re
import signal
import string
import subprocess
import sys
import threading
import time
from typing import Any
import warnings

import jax
import jax.extend.backend as jax_backend
import pathwaysutils
from pathwaysutils.experimental.shared_pathways_service import gke_utils
from pathwaysutils.experimental.shared_pathways_service import metrics_collector
from pathwaysutils.experimental.shared_pathways_service import validators


_CLEANUP_SIGNALS = [signal.SIGTERM, signal.SIGINT]
if hasattr(signal, "SIGHUP"):  # SIGHUP is not available on Windows.
  _CLEANUP_SIGNALS.append(signal.SIGHUP)

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
EPHEMERAL_SIDECAR_CONTAINER_NAME = "ephemeral-python-sidecar"
DEFAULT_SIDECAR_PORT = 50051
EPHEMERAL_SIDECAR_PORT = 50052
DEFAULT_SIDECAR_CONNECT_TIMEOUT = "300s"

_logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ProxyOptions:
  """Configuration options for the Pathways proxy.

  Attributes:
    use_insecure_credentials: Whether to use insecure gRPC credentials for the
      proxy server.
    xla_flags: A list of XLA flags to pass to the proxy server.
    sidecar: Whether to use the worker sidecar or not.
    sidecar_image: Optional custom sidecar image to inject as an ephemeral
      container into placed worker pods.
    sidecar_port: Port used by the external sidecar gRPC server.
    sidecar_connect_timeout: Timeout for connecting to the external sidecar.
  """
  use_insecure_credentials: bool = False
  xla_flags: list[str] = dataclasses.field(default_factory=list)
  sidecar: bool = False
  sidecar_image: str | None = None
  sidecar_port: int = DEFAULT_SIDECAR_PORT
  sidecar_connect_timeout: str = DEFAULT_SIDECAR_CONNECT_TIMEOUT

  @classmethod
  def from_list(cls, options: Iterable[str] | None) -> "ProxyOptions":
    """Creates a ProxyOptions object from a list of 'key:value' strings."""
    use_insecure = False
    use_sidecar = False
    sidecar_image = None
    sidecar_port = DEFAULT_SIDECAR_PORT
    sidecar_connect_timeout = DEFAULT_SIDECAR_CONNECT_TIMEOUT
    xla_flags = []
    for option in options or []:
      if ":" in option:
        key, value = option.split(":", 1)
        key_strip = key.strip().lower()
        if key_strip == "use_insecure_credentials":
          use_insecure = value.strip().lower() == "true"
        elif key_strip == "sidecar":
          use_sidecar = value.strip().lower() == "true"
        elif key_strip == "sidecar_image":
          val_strip = value.strip()
          if val_strip:
            sidecar_image = val_strip
            use_sidecar = True
            sidecar_port = EPHEMERAL_SIDECAR_PORT
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

    return cls(
        use_insecure_credentials=use_insecure,
        xla_flags=xla_flags,
        sidecar=use_sidecar,
        sidecar_image=sidecar_image,
        sidecar_port=sidecar_port,
        sidecar_connect_timeout=sidecar_connect_timeout,
    )


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

  if proxy_options.sidecar or proxy_options.sidecar_image:
    proxy_args_str += "\n        - --sidecar_name=external"
  if proxy_options.sidecar_image:
    proxy_args_str += (
        "\n        -"
        f" --cloud_pathways_external_sidecar_port={proxy_options.sidecar_port}"
        "\n        -"
        " --cloud_pathways_external_sidecar_connect_timeout="
        f"{proxy_options.sidecar_connect_timeout}"
        "\n        - --vmodule=location_map=2"
    )

  escaped_proxy_server_image = (
      proxy_server_image.replace("\\", "\\\\")
      .replace('"', '\\"')
      .replace("\n", "\\n")
  )

  template = string.Template(yaml_template)
  substituted_yaml = template.substitute(
      PROXY_JOB_NAME=proxy_job_name,
      PROXY_SERVER_PORT=PROXY_SERVER_PORT,
      PATHWAYS_HEAD_HOSTNAME=pathways_head_hostname,
      PATHWAYS_HEAD_PORT=pathways_head_port,
      EXPECTED_INSTANCES=instances_str,
      GCS_SCRATCH_LOCATION=gcs_scratch_location,
      PROXY_SERVER_IMAGE=f'"{escaped_proxy_server_image}"',
      PROXY_ENV=proxy_env_str,
      PROXY_ARGS=proxy_args_str,
  )

  _logger.info("Deploying Pathways proxy: %s", proxy_job_name)
  gke_utils.deploy_gke_yaml(substituted_yaml)

  _logger.info("Successfully deployed Pathways proxy.")


def _extract_pod_names_from_log_line(line: str) -> list[str]:
  """Extracts worker Pod names from a pw-proxy log line."""
  pods = []
  # Format 1: Transition slice ... unplaced -> placed on worker pods: [p1, p2]
  match = re.search(r"on worker pods:\s*\[([^\]]+)\]", line, re.IGNORECASE)
  if match:
    for item in match.group(1).split(","):
      pod = item.strip().split(".")[0].split(":")[0]
      if pod:
        pods.append(pod)
  # Format 2: VLOG(2) Sidecar init request resolved_addresses: "pod.jobset:port"
  for addr in re.findall(r'resolved_addresses:\s*"([^"]+)"', line):
    pod = addr.strip().split(".")[0].split(":")[0]
    if pod:
      pods.append(pod)
  return pods


def _inject_ephemeral_sidecars(
    pod_names: Iterable[str],
    sidecar_image: str,
    sidecar_port: int,
) -> None:
  """Injects and waits for the ephemeral sidecar container on placed pods."""
  pods = sorted(set(pod_names))
  if not pods:
    _logger.warning(
        "No placed worker pods found in proxy logs; skipping ephemeral sidecar"
        " injection."
    )
    return
  _logger.info(
      "Injecting ephemeral sidecar image '%s' into %d placed worker pod(s): %s",
      sidecar_image,
      len(pods),
      pods,
  )
  for pod_name in pods:
    gke_utils.inject_ephemeral_sidecar(
        pod_name=pod_name,
        container_name=EPHEMERAL_SIDECAR_CONTAINER_NAME,
        image=sidecar_image,
        port=sidecar_port,
    )
  for pod_name in pods:
    gke_utils.wait_for_ephemeral_container(
        pod_name=pod_name,
        container_name=EPHEMERAL_SIDECAR_CONTAINER_NAME,
    )


def _wait_for_placement(
    log_process: subprocess.Popen[str],
    num_slices: int,
    metrics_collector_inst: Any = None,
    start_time: float | None = None,
    total_chips: int = 0,
    sidecar_image: str | None = None,
    sidecar_port: int = EPHEMERAL_SIDECAR_PORT,
    placed_pods_out: set[str] | None = None,
) -> None:
  """Waits for the placement to be complete by checking proxy logs."""
  _logger.info("Streaming proxy logs until the placement is complete...")
  keywords = [
      "placement",
      "Signaling to RM",
      "Transition slice",
      "FAILED_PRECONDITION",
  ]
  end_phrase = "unplaced -> placed"
  reconfig_phrase = "component reconfiguration after placement changes"
  placement_count = 0
  placed_pods: set[str] = set()

  if not log_process.stdout:
    _logger.error("Log streaming process stdout is empty. Terminating.")
    log_process.terminate()
    _, stderr = log_process.communicate()
    raise RuntimeError(
        "Failed to stream proxy logs: stdout not available.\n"
        f"STDERR: {stderr}"
    )

  def _complete_placement() -> None:
    if placed_pods_out is not None:
      placed_pods_out.update(placed_pods)
    if sidecar_image:
      _inject_ephemeral_sidecars(placed_pods, sidecar_image, sidecar_port)
    if metrics_collector_inst is not None:
      metrics_collector_inst.record_active_user(True)
      metrics_collector_inst.record_capacity_in_use(total_chips)
      if start_time:
        duration = time.time() - start_time
        metrics_collector_inst.record_assignment_time(duration)
        metrics_collector_inst.record_successful_request()

  waiting_for_vlog_pods = False
  for line in log_process.stdout:
    line_lower = line.lower()
    if any(keyword.lower() in line_lower for keyword in keywords):
      _logger.info("Proxy log: %s", line.strip())

    for pod in _extract_pod_names_from_log_line(line):
      placed_pods.add(pod)

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
        if sidecar_image and not placed_pods:
          waiting_for_vlog_pods = True
        else:
          _complete_placement()
          return

    if waiting_for_vlog_pods and reconfig_phrase in line_lower:
      _complete_placement()
      return

  if waiting_for_vlog_pods:
    _complete_placement()


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
    self._log_process = None
    self._proxy_port = None
    self.proxy_server_image = proxy_server_image
    self.proxy_options = proxy_options or ProxyOptions()
    self._old_jax_platforms = None
    if collect_service_metrics:
      raw_collector = metrics_collector.MetricsCollector(
          self.project, self.cluster, self._proxy_job_name
      )
      self.metrics_collector = metrics_collector.SafeMetricsCollector(
          raw_collector
      )
    else:
      self.metrics_collector = metrics_collector.SafeMetricsCollector(None)

    self.start_time = None
    self._old_jax_backend_target = None
    self._old_jax_platforms_config = None
    self._old_jax_backend_target_config = None
    self.total_chips = self._get_total_chips()
    self._placed_worker_pods: set[str] = set()
    self._cleaned_up = False
    self._cleanup_lock = threading.Lock()
    self._original_signal_handlers: dict[signal.Signals, Any] = {}

  def _register_signal_handlers(self) -> None:
    """Registers signal handlers to ensure cleanup on termination."""
    if threading.current_thread() is not threading.main_thread():
      # Python only allows signal handlers to be registered in the main thread.
      return

    def _handle_signal(signum, frame):
      del frame
      _logger.warning(
          "Received signal %d. Triggering Pathways proxy cleanup...", signum
      )
      self._cleanup()
      if signum == signal.SIGINT:
        raise KeyboardInterrupt()
      sys.exit(128 + signum)

    for sig in _CLEANUP_SIGNALS:
      try:
        self._original_signal_handlers[sig] = signal.signal(sig, _handle_signal)
      except (ValueError, OSError) as e:
        _logger.debug("Could not register handler for signal %s: %s", sig, e)

  def _restore_signal_handlers(self) -> None:
    """Restores original signal handlers.

    Signal handlers are global process state. If the handlers aren't restored,
    the new behavior (triggering Pathways cleanup and exiting) persists for the
    remainder of the process's life, even after the _ISCPathways context has
    finished.
    """
    if threading.current_thread() is not threading.main_thread():
      # Only restore signal handlers in the main thread because this is the
      # thread that registered them.
      return

    for sig, original_handler in self._original_signal_handlers.items():
      try:
        signal.signal(sig, original_handler)
      except (ValueError, OSError) as e:
        _logger.debug("Could not restore handler for signal %s: %s", sig, e)
    self._original_signal_handlers.clear()

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
    atexit.register(self._cleanup)
    self._register_signal_handlers()

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
          gke_utils.start_port_forwarding(
              f"pod/{self.proxy_pod_name}",
              PROXY_SERVER_PORT,
          )
      )
      gke_utils.wait_for_port_forwarding(
          self._port_forward_process, self._proxy_port
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

      if self.proxy_options.sidecar_image and self.proxy_pod_name:
        num_slices = sum(self.expected_tpu_instances.values())
        self._log_process = gke_utils.stream_pod_logs(self.proxy_pod_name)
        placement_thread = threading.Thread(
            target=_wait_for_placement,
            args=(
                self._log_process,
                num_slices,
                self.metrics_collector,
                self.start_time,
                self.total_chips,
                self.proxy_options.sidecar_image,
                self.proxy_options.sidecar_port,
                self._placed_worker_pods,
            ),
            daemon=True,
        )
        placement_thread.start()

      pathwaysutils.initialize()
      _logger.info(
          "Interactive supercomputing proxy client ready for cluster '%s'.",
          self.cluster,
      )
      return self
    except BaseException as e:
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
    with self._cleanup_lock:
      # Ensure that cleanup logic only runs once, even if triggered by
      # multiple events (like a signal and a normal context exit).
      if self._cleaned_up:
        return
      self._cleaned_up = True

      atexit.unregister(self._cleanup)
      self._restore_signal_handlers()

      # Clear JAX caches and run garbage collection.
      _logger.info("Starting Pathways proxy cleanup.")
      jax_backend.clear_backends()
      jax.clear_caches()
      gc.collect()
      _logger.info("Cleared JAX caches and ran garbage collection.")

      # Terminate the port forwarding process.
      if self._port_forward_process:
        _logger.info(
            "Terminating port forwarding process (PID %s)...",
            getattr(self._port_forward_process, "pid", "unknown"),
        )
        try:
          gke_utils.terminate_process(
              self._port_forward_process, process_name="Port forwarding"
          )
        finally:
          self._port_forward_process = None

      # Terminate the log streaming process.
      if self._log_process:
        _logger.info(
            "Terminating log streaming process (PID %s)...",
            getattr(self._log_process, "pid", "unknown"),
        )
        try:
          gke_utils.terminate_process(
              self._log_process, process_name="Log streaming"
          )
        finally:
          self._log_process = None

      # Delete the proxy GKE job.
      if self._proxy_job_name:
        _logger.info("Deleting Pathways proxy...")
        try:
          gke_utils.delete_gke_resource("job", self._proxy_job_name)
          _logger.info("Pathways proxy GKE job deletion complete.")
        except Exception as e:  # pylint: disable=broad-exception-caught
          _logger.exception(
              "Failed to delete Pathways proxy GKE job: %r", e
          )

      # Delete placed worker pods that had ephemeral sidecars injected so the
      # JobSet controller recreates clean worker pods.
      if self._placed_worker_pods:
        _logger.info(
            "Deleting %d worker pod(s) with ephemeral sidecar containers: %s",
            len(self._placed_worker_pods),
            sorted(self._placed_worker_pods),
        )
        gke_utils.delete_worker_pods(sorted(self._placed_worker_pods))
        self._placed_worker_pods.clear()

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


def _get_username() -> str:
  """Gets the username from the environment.

  If the username contains an underscore, splits on '_' and uses the first
  portion.

  Returns:
    The sanitized username, or 'user' if unavailable.
  """
  username = os.environ.get("USER", "user")
  if "_" in username:
    username = username.split("_")[0]
  return username or "user"


def _is_current_kube_context(
    *, cluster: str, project: str, location: str
) -> bool:
  """Checks whether the active kube config context points at the cluster.

  Args:
    cluster: The name of the GKE cluster.
    project: The GCP project ID.
    location: The GCP region or zone of the cluster.

  Returns:
    True if the current kube config context already targets the given cluster.
  """
  return gke_utils.get_current_kube_context() == (cluster, project, location)


def _ensure_cluster_credentials(
    *, cluster: str, project: str, location: str
) -> None:
  """Fetches the GKE cluster credentials unless kube config already has them."""
  if _is_current_kube_context(
      cluster=cluster, project=project, location=location
  ):
    _logger.info(
        "The current kube config context already points to cluster '%s' in"
        " project '%s' and location '%s'. Skipping credential fetch.",
        cluster,
        project,
        location,
    )
    return

  gke_utils.fetch_cluster_credentials(
      cluster_name=cluster, project_id=project, location=location
  )


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
    proxy_server_image: str | None = None,
    proxy_options: Sequence[str] | None = None,
    sidecar_image: str | None = None,
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
    proxy_server_image: (Deprecated) The proxy server image to use. If not
      provided, it will be auto-detected from the Pathways service. If the given
      proxy image is incompatible with the Pathways service, it will be replaced
      with the compatible proxy image.
    proxy_options: Configuration options for the Pathways proxy. If not
      provided, no extra options will be used.
    sidecar_image: Optional custom colocated Python sidecar image to inject as
      an ephemeral container into assigned worker pods.
    collect_service_metrics: Whether to collect usage metrics for Shared
      Pathways Service.

  Yields:
    The Pathways manager.
  """
  if proxy_server_image is not None:
    warnings.warn(
        "`proxy_server_image` is deprecated and will be removed in a future"
        " release. The proxy server image is automatically detected from the"
        " Pathways service.",
        DeprecationWarning,
        stacklevel=2,
    )
  _logger.info("Validating Pathways service and TPU instances...")
  validators.validate_pathways_service(pathways_service)
  validators.validate_tpu_instances(expected_tpu_instances)
  validators.validate_proxy_options(proxy_options)
  _ensure_cluster_credentials(
      cluster=cluster, project=project, location=region
  )

  server_image, service_sidecar_image = gke_utils.get_pathways_service_images(
      pathways_service
  )
  compatible_proxy_image = gke_utils.get_compatible_proxy_server_image(
      server_image
  )
  _logger.info(
      "Auto-detected compatible proxy server image: %s", compatible_proxy_image
  )
  if proxy_server_image and proxy_server_image != compatible_proxy_image:
    _logger.warning(
        "The provided proxy image '%s' is incompatible with the service"
        " '%s'. Replacing it with the compatible proxy image '%s'.",
        proxy_server_image,
        pathways_service,
        compatible_proxy_image,
    )
  proxy_server_image = compatible_proxy_image

  proxy_options_obj = ProxyOptions.from_list(proxy_options)
  if sidecar_image:
    proxy_options_obj.sidecar = True
    proxy_options_obj.sidecar_image = sidecar_image
    proxy_options_obj.sidecar_port = EPHEMERAL_SIDECAR_PORT

  effective_sidecar_image = (
      proxy_options_obj.sidecar_image or service_sidecar_image
  )
  if proxy_options_obj.sidecar and effective_sidecar_image:
    validators.validate_sidecar_image_versions(effective_sidecar_image)
  _logger.info("Validation complete.")

  if not proxy_job_name:
    username = _get_username()
    random_suffix = "".join(
        random.choices(string.ascii_lowercase + string.digits, k=5)
    )
    proxy_job_name = f"isc-proxy-{username}-{random_suffix}"

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
      if not proxy_options_obj.sidecar_image:
        num_slices = sum(t.expected_tpu_instances.values())
        t._log_process = gke_utils.stream_pod_logs(t.proxy_pod_name)
        placement_thread = threading.Thread(
            target=_wait_for_placement,
            args=(
                t._log_process,
                num_slices,
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
