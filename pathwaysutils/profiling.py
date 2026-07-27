# Copyright 2024 Google LLC
#
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
"""Profiling Utilities."""

from collections.abc import Mapping
import concurrent.futures
import datetime
import json
import logging
import os
import threading
import time
from typing import Any

import grpc
import jax
from jax import numpy as jnp
from jax.extend import backend
from pathwaysutils import plugin_executable
from tensorflow.core.profiler.protobuf import profiler_service_pb2  # pylint: disable=g-direct-tensorflow-import
from tensorflow.core.profiler.protobuf import profiler_service_pb2_grpc  # pylint: disable=g-direct-tensorflow-import


_logger = logging.getLogger(__name__)


class _ProfileState:
  """Holds the state of an ongoing profiling session.

  Attributes:
    executable: The `plugin_executable.PluginExecutable` instance used for the
      profiling session.
    profile_request: The mapping containing the profile request options.
    lock: A thread lock to protect access to the state.
  """
  executable: plugin_executable.PluginExecutable | None = None
  profile_request: Mapping[str, Any] | None = None
  lock: threading.Lock

  def __init__(self) -> None:
    self.executable = None
    self.profile_request = None
    self.lock = threading.Lock()

  def reset(self) -> None:
    self.executable = None
    self.profile_request = None

  def call_profile_executable(self) -> None:
    """Calls the profiling executable and waits for the result."""
    if self.executable is None:
      raise RuntimeError(
          "_call_profile_executable called with no active executable."
      )
    # If the profile request contains xprofTraceOptions, then we need to pass
    # out_avals and out_shardings to the executable call because the
    # executable will return a future that needs to be resolved. This is true
    # for both starting and stopping a trace.
    if (
        self.profile_request is not None
        and "xprofTraceOptions" in self.profile_request
    ):
      out_avals = [jax.core.ShapedArray((1,), jnp.object_)]
      out_shardings = [
          getattr(
              jax.sharding,
              "make_single_device_sharding",
              jax.sharding.SingleDeviceSharding,
          )(backend.get_default_device())
      ]
    else:
      out_avals = ()
      out_shardings = ()

    _, result_future = self.executable.call(
        out_avals=out_avals, out_shardings=out_shardings
    )
    result_future.result()


_first_profile_start = True
_profile_state = _ProfileState()
_original_start_trace = jax.profiler.start_trace
_original_stop_trace = jax.profiler.stop_trace


def toy_computation() -> None:
  """A toy computation to run before the first profile."""
  x = jax.jit(lambda x: x + 1)(jnp.array(1))
  x.block_until_ready()


def _is_default_profile_options(
    profiler_options: jax.profiler.ProfileOptions,
) -> bool:
  if jax.version.__version_info__ < (0, 9, 2):
    return True

  default_options = jax.profiler.ProfileOptions()
  return (
      profiler_options.host_tracer_level == default_options.host_tracer_level
      and profiler_options.python_tracer_level
      == default_options.python_tracer_level
      and profiler_options.duration_ms == default_options.duration_ms
      and not profiler_options.advanced_configuration
      and not profiler_options.session_id
  )


def _create_profile_request(
    log_dir: os.PathLike[str] | str,
    profiler_options: jax.profiler.ProfileOptions | None = None,
    max_num_hosts: int = 1,
) -> Mapping[str, Any]:
  """Creates a profile request mapping from the given options."""
  profile_request: dict[str, Any] = {
      "traceLocation": str(log_dir),
      "maxNumHosts": max_num_hosts,
  }

  if profiler_options is None or _is_default_profile_options(profiler_options):
    return profile_request

  advanced_config = None
  if profiler_options.advanced_configuration:
    advanced_config = {}
    for k, v in profiler_options.advanced_configuration.items():
      # Convert python dict to tensorflow.ProfileOptions.AdvancedConfigValue
      # json-compatible dict
      if isinstance(v, bool):
        advanced_config[k] = {"boolValue": v}
      elif isinstance(v, int):
        advanced_config[k] = {"int64Value": v}
      elif isinstance(v, str):
        advanced_config[k] = {"stringValue": v}
      else:
        raise ValueError(
            f"Unsupported advanced configuration value type: {type(v)}. "
            "Supported types are bool, int, and str."
        )

  xprof_options: dict[str, Any] = {
      "traceDirectory": str(log_dir),
  }

  if profiler_options.host_tracer_level != 2:
    xprof_options["hostTraceLevel"] = profiler_options.host_tracer_level

  pw_trace_opts: dict[str, Any] = {}
  if profiler_options.python_tracer_level:
    pw_trace_opts["enablePythonTracer"] = bool(
        profiler_options.python_tracer_level
    )

  if advanced_config:
    pw_trace_opts["advancedConfiguration"] = advanced_config

  if pw_trace_opts:
    xprof_options["pwTraceOptions"] = pw_trace_opts

  if profiler_options.session_id:
    xprof_options["traceSessionName"] = profiler_options.session_id

  profile_request["xprofTraceOptions"] = xprof_options

  if profiler_options.duration_ms > 0:
    profile_request["maxDurationSecs"] = profiler_options.duration_ms / 1000.0

  return profile_request


def _start_pathways_trace_from_profile_request(
    profile_request: Mapping[str, Any],
) -> None:
  """Starts a profiler trace on Pathways components from a profile request.

  This will only profile the Pathways components and not the JAX client code.

  Args:
    profile_request: A mapping containing the profile request options.
  """
  with _profile_state.lock:
    global _first_profile_start
    if _first_profile_start:
      _first_profile_start = False
      toy_computation()

    if _profile_state.executable is not None:
      raise RuntimeError(
          "start_trace called while a trace is already being taken!"
      )
    try:
      _profile_state.executable = plugin_executable.PluginExecutable(
          json.dumps({"profileRequest": profile_request})
      )
      _profile_state.profile_request = profile_request
      _profile_state.call_profile_executable()
    except Exception as e:
      _profile_state.reset()
      if (
          "xprofTraceOptions" in profile_request
          and "traceSessionName" in profile_request["xprofTraceOptions"]
      ):
        if "Bad PluginProgram" in str(e):
          raise RuntimeError(
              "Failed to start Pathways trace. The Pathways backend rejected "
              "the request, likely because the running Pathways server images "
              "do not support the trace session ID option. Please ensure you "
              "are running the latest versions of both Pathways server images "
              "and the pathwaysutils library."
          ) from e
      _logger.exception("Failed to start trace")
      raise


def start_trace(
    log_dir: os.PathLike[str] | str,
    *,
    create_perfetto_link: bool = False,
    create_perfetto_trace: bool = False,
    profiler_options: jax.profiler.ProfileOptions | None = None,
    max_num_hosts: int = 1,
) -> None:
  """Starts a profiler trace.

  The trace will capture CPU and TPU activity, including Python
  functions and JAX on-device operations. Use :func:`stop_trace` to end the
  trace and save the results to ``log_dir``.

  The resulting trace can be viewed with TensorBoard. Note that TensorBoard
  doesn't need to be running when collecting the trace.

  Only one trace may be collected at a time. A RuntimeError will be raised if
  :func:`start_trace` is called while another trace is running.

  Args:
    log_dir: The GCS directory to save the profiler trace to (usually the
      TensorBoard log directory), e.g., "gs://my_bucket/profiles".
    create_perfetto_link: A boolean which, if true, creates and prints link to
      the Perfetto trace viewer UI (https://ui.perfetto.dev). The program will
      block until the link is opened and Perfetto loads the trace. This feature
      is experimental for Pathways on Cloud and may not be fully supported.
    create_perfetto_trace: A boolean which, if true, additionally dumps a
      ``perfetto_trace.json.gz`` file that is compatible for upload with the
      Perfetto trace viewer UI (https://ui.perfetto.dev). The file will also be
      generated if ``create_perfetto_link`` is true. This could be useful if you
      want to generate a Perfetto-compatible trace without blocking the process.
      This feature is experimental for Pathways on Cloud and may not be fully
      supported.
    profiler_options: Profiler options to configure the profiler for collection.
    max_num_hosts: An optional integer to limit the number of hosts profiled
      (defaults to 1).
  """
  if not str(log_dir).startswith("gs://"):
    raise ValueError(f"log_dir must be a GCS bucket path, got {log_dir}")

  if create_perfetto_link or create_perfetto_trace:
    _logger.warning(
        "create_perfetto_link and create_perfetto_trace are experimental "
        "features for Pathways on Cloud and may not be fully supported."
    )

  if jax.version.__version_info__ < (0, 9, 2):
    if profiler_options is not None:
      _logger.warning(
          "ProfileOptions are not supported until JAX 0.9.2 and will be omitted. "
          "Some options can be specified via command line flags."
      )
      profiler_options = None
  else:
    if profiler_options is None:
      profiler_options = jax.profiler.ProfileOptions()
    if not profiler_options.session_id:
      profiler_options.session_id = datetime.datetime.now().strftime(
          "%Y_%m_%d_%H_%M_%S"
      )

  profile_request = _create_profile_request(
      log_dir,
      profiler_options,
      max_num_hosts=max_num_hosts,
  )

  _logger.debug("Profile request: %s", profile_request)

  _start_pathways_trace_from_profile_request(profile_request)

  if jax.version.__version_info__ >= (0, 9, 2):
    _original_start_trace(
        log_dir=log_dir,
        create_perfetto_link=create_perfetto_link,
        create_perfetto_trace=create_perfetto_trace,
        profiler_options=profiler_options,
    )
  else:
    _original_start_trace(
        log_dir=log_dir,
        create_perfetto_link=create_perfetto_link,
        create_perfetto_trace=create_perfetto_trace,
    )


def stop_trace() -> None:
  """Stops the currently-running profiler trace."""
  try:
    with _profile_state.lock:
      if _profile_state.executable is None:
        raise RuntimeError("stop_trace called before a trace is being taken!")
      try:
        _profile_state.call_profile_executable()
      finally:
        _profile_state.reset()
  finally:
    _original_stop_trace()


_profiler_server: grpc.Server | None = None
_profiler_server_lock = threading.Lock()


class PathwaysProfilerServicer(
    profiler_service_pb2_grpc.ProfilerServiceServicer
):
  """gRPC servicer for Pathways Profiler Service implementing tensorflow.ProfilerService."""

  def Profile(
      self,
      request: profiler_service_pb2.ProfileRequest,
      context: grpc.ServicerContext,
  ) -> profiler_service_pb2.ProfileResponse:
    del context
    duration_ms = request.duration_ms
    log_dir = request.repository_root
    _logger.info("Received gRPC profile request for %s ms", duration_ms)
    _logger.info("Writing profiling data to %s", log_dir)

    try:
      jax.profiler.start_trace(log_dir)
      time.sleep(duration_ms / 1000.0)
    finally:
      _logger.info("Stopping trace")
      jax.profiler.stop_trace()

    return profiler_service_pb2.ProfileResponse()


def start_server(
    port: int, requires_backend: bool = True, use_alts: bool | None = None
) -> None:
  """Starts the profiling server on port `port`.

  The signature matches `jax.profiler.start_server`, though no handle
  to the server is returned because there is no
  `xla_client.profiler.ProfilerServer` to return.

  Args:
    port: The port to start the server on.
    requires_backend: Unused in Pathways; accepted for parameter parity.
    use_alts: Whether to use ALTS credentials. If None, defaults to checking
      the PATHWAYS_PROFILER_USE_ALTS environment variable (defaulting to True).
  """
  del requires_backend
  global _profiler_server
  with _profiler_server_lock:
    if _profiler_server is not None:
      raise RuntimeError("Only one profiler server can be active at a time.")

    if use_alts is None:
      use_alts = os.environ.get("PATHWAYS_PROFILER_USE_ALTS", "1") == "1"

    _logger.info("Starting JAX pathways profiler gRPC server on port %s", port)
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=2))
    profiler_service_pb2_grpc.add_ProfilerServiceServicer_to_server(
        PathwaysProfilerServicer(), server
    )
    if use_alts:
      server_creds = grpc.alts_server_credentials()
      server.add_secure_port(f"[::]:{port}", server_creds)
    else:
      server.add_insecure_port(f"[::]:{port}")
    server.start()
    _profiler_server = server


def stop_server() -> None:
  """Stops the active profiler server."""
  global _profiler_server
  with _profiler_server_lock:
    if _profiler_server is None:
      raise RuntimeError("No active profiler server.")
    _logger.info("Stopping JAX pathways profiler gRPC server")
    _profiler_server.stop(grace=5.0)
    _profiler_server = None


def collect_profile(
    port: int,
    duration_ms: int,
    host: str,
    log_dir: os.PathLike[str] | str,
    use_alts: bool = True,
) -> bool:
  """Collects a JAX profile and saves it to the specified directory.

  Args:
    port: The port on which the JAX profiler server is running.
    duration_ms: The duration in milliseconds for which to collect the profile.
    host: The host on which the JAX profiler server is running.
    log_dir: The GCS path to save the profile data.
    use_alts: Whether to connect using ALTS credentials.

  Returns:
    True if the profile was collected successfully, False otherwise.

  Raises:
    ValueError: If the log_dir is not a GCS path.
  """
  if not str(log_dir).startswith("gs://"):
    raise ValueError(f"log_dir must be a GCS bucket path, got {log_dir}")

  target = f"{host}:{port}"
  _logger.info("Connecting to profiling server at %s", target)
  try:
    if use_alts:
      creds = grpc.alts_channel_credentials()
      channel = grpc.secure_channel(target, creds)
    else:
      channel = grpc.insecure_channel(target)

    with channel:
      stub = profiler_service_pb2_grpc.ProfilerServiceStub(channel)
      request = profiler_service_pb2.ProfileRequest(
          duration_ms=duration_ms,
          repository_root=str(log_dir),
      )
      timeout = (duration_ms / 1000.0) + 10.0
      _logger.info("Triggering profile for %s ms", duration_ms)
      stub.Profile(request, timeout=timeout)
      _logger.info("Profiling response completed successfully")
      return True
  except grpc.RpcError as e:
    e_call: Any = e
    if e_call.code() == grpc.StatusCode.UNAVAILABLE:
      _logger.error(
          "Failed to connect to the profiling server at %s. "
          "Please verify that the server is running on this port. "
          "Error details: %s",
          target,
          e_call,
      )
    elif e_call.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
      _logger.error(
          "Profiling request timed out. The server might be unresponsive. "
          "Error details: %s",
          e_call,
      )
    else:
      _logger.error(
          "gRPC error occurred while collecting profile. "
          "Error code: %s, details: %s",
          e_call.code(),
          e_call.details(),
      )
    return False

  return True


def monkey_patch_jax() -> None:
  """Monkey patches JAX with Pathways versions of functions.

  The signatures in patched functions should match the original.

  Patched functions are:
  - `jax.profiler.start_trace`
    https://jax.readthedocs.io/en/latest/_autosummary/jax.profiler.start_trace.html
  - `jax.profiler.stop_trace`
    https://jax.readthedocs.io/en/latest/_autosummary/jax.profiler.stop_trace.html
  - `jax.profiler.start_server`
    https://jax.readthedocs.io/en/latest/_autosummary/jax.profiler.start_server.html
  - `jax.profiler.stop_server`
  """

  def start_trace_patch(
      log_dir,
      create_perfetto_link: bool = False,
      create_perfetto_trace: bool = False,
      profiler_options: jax.profiler.ProfileOptions | None = None,
      max_num_hosts: int = 1,
  ) -> None:
    _logger.debug("jax.profile.start_trace patched with pathways' start_trace")
    start_trace(
        log_dir,
        create_perfetto_link=create_perfetto_link,
        create_perfetto_trace=create_perfetto_trace,
        profiler_options=profiler_options,
        max_num_hosts=max_num_hosts,
    )

  jax.profiler.start_trace = start_trace_patch
  jax._src.profiler.start_trace = start_trace_patch  # pylint: disable=protected-access

  def stop_trace_patch() -> None:
    _logger.debug("jax.profile.stop_trace patched with pathways' stop_trace")
    stop_trace()

  jax.profiler.stop_trace = stop_trace_patch
  jax._src.profiler.stop_trace = stop_trace_patch  # pylint: disable=protected-access

  def start_server_patch(port: int, requires_backend: bool = True) -> None:
    _logger.debug(
        "jax.profile.start_server patched with pathways' start_server"
    )
    start_server(port, requires_backend=requires_backend)

  jax.profiler.start_server = start_server_patch  # pyrefly: ignore[bad-assignment]
  jax._src.profiler.start_server = start_server_patch  # pylint: disable=protected-access  # pyrefly: ignore[bad-assignment]

  def stop_server_patch() -> None:
    _logger.debug("jax.profile.stop_server patched with pathways' stop_server")
    stop_server()

  jax.profiler.stop_server = stop_server_patch
  jax._src.profiler.stop_server = stop_server_patch  # pylint: disable=protected-access
