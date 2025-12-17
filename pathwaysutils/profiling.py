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
"""Profiling utilites."""

import dataclasses
import json
import logging
import os
import threading
import time
from typing import Any
import urllib.parse

import fastapi
import jax
from jax import numpy as jnp
from pathwaysutils import plugin_executable
import requests
import uvicorn


_logger = logging.getLogger(__name__)


class _ProfileState:
  executable: plugin_executable.PluginExecutable | None = None
  lock: threading.Lock

  def __init__(self):
    self.executable = None
    self.lock = threading.Lock()

  def reset(self):
    self.executable = None


_first_profile_start = True
_profile_state = _ProfileState()
_original_start_trace = jax.profiler.start_trace
_original_stop_trace = jax.profiler.stop_trace


def toy_computation():
  """A toy computation to run before the first profile."""
  x = jax.jit(lambda x: x + 1)(jnp.array(1))
  x.block_until_ready()


def _create_profile_request(
    log_dir: os.PathLike[str] | str,
) -> dict[str, Any]:
  """Creates a profile request dictionary from the given options."""
  profile_request = {}
  profile_request["traceLocation"] = str(log_dir)

  return profile_request


def _start_pathways_trace_from_profile_request(
    profile_request: dict[str, Any],
) -> None:
  """Starts a profiler trace on Pathways components from a profile request.

  This will only profile the Pathways components and not the JAX client code.

  Args:
    profile_request: A dictionary containing the profile request options.
  """
  with _profile_state.lock:
    global _first_profile_start
    if _first_profile_start:
      _first_profile_start = False
      toy_computation()

    if _profile_state.executable is not None:
      raise ValueError(
          "start_trace called while a trace is already being taken!"
      )
    _profile_state.executable = plugin_executable.PluginExecutable(
        json.dumps({"profileRequest": profile_request})
    )
    try:
      _, result_future = _profile_state.executable.call()
      result_future.result()
    except Exception as e:  # pylint: disable=broad-except
      _logger.exception("Failed to start trace")
      _profile_state.reset()
      raise


def start_trace(
    log_dir: os.PathLike[str] | str,
    *,
    create_perfetto_link: bool = False,
    create_perfetto_trace: bool = False,
    profiler_options: jax.profiler.ProfileOptions | None = None,  # pylint: disable=unused-argument
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
      Options are not currently supported and ignored.
  """
  if not str(log_dir).startswith("gs://"):
    raise ValueError(f"log_dir must be a GCS bucket path, got {log_dir}")

  if create_perfetto_link or create_perfetto_trace:
    _logger.warning(
        "create_perfetto_link and create_perfetto_trace are experimental "
        "features for Pathways on Cloud and may not be fully supported."
    )

  _start_pathways_trace_from_profile_request(_create_profile_request(log_dir))

  _original_start_trace(
      log_dir=log_dir,
      create_perfetto_link=create_perfetto_link,
      create_perfetto_trace=create_perfetto_trace,
  )


def stop_trace():
  """Stops the currently-running profiler trace."""
  try:
    with _profile_state.lock:
      if _profile_state.executable is None:
        raise ValueError("stop_trace called before a trace is being taken!")
      try:
        _, result_future = _profile_state.executable.call()
        result_future.result()
      finally:
        _profile_state.reset()
  finally:
    _original_stop_trace()


_profiler_thread: threading.Thread | None = None


def start_server(port: int):
  """Starts the profiling server on port `port`.

  The signature is slightly different from `jax.profiler.start_server`
  because no handle to the server is returned because there is no
  `xla_client.profiler.ProfilerServer` to return.

  Args:
    port : The port to start the server on.
  """
  def server_loop(port: int):
    _logger.debug("Starting JAX profiler server on port %s", port)
    app = fastapi.FastAPI()

    @dataclasses.dataclass
    class ProfilingConfig:
      duration_ms: int
      repository_path: str

    @app.post("/profiling")
    async def profiling(pc: ProfilingConfig):  # pylint: disable=unused-variable
      _logger.debug("Capturing profiling data for %s ms", pc.duration_ms)
      _logger.debug("Writing profiling data to %s", pc.repository_path)
      jax.profiler.start_trace(pc.repository_path)
      time.sleep(pc.duration_ms / 1e3)
      jax.profiler.stop_trace()
      return {"response": "profiling completed"}

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="debug")

  global _profiler_thread
  if _profiler_thread is not None:
    raise ValueError("Only one profiler server can be active at a time.")

  _profiler_thread = threading.Thread(target=server_loop, args=(port,))
  _profiler_thread.start()


def stop_server():
  """Raises an error if there is not an active profiler server but otherwise does nothing.

  Pathways profiling servers are not stoppable at this time.
  """
  if _profiler_thread is None:
    raise ValueError("No active profiler server.")


def collect_profile(
    port: int,
    duration_ms: int,
    host: str,
    log_dir: os.PathLike[str] | str,
) -> bool:
  """Collects a JAX profile and saves it to the specified directory.

  Args:
    port: The port on which the JAX profiler server is running.
    duration_ms: The duration in milliseconds for which to collect the profile.
    host: The host on which the JAX profiler server is running.
    log_dir: The GCS path to save the profile data.

  Returns:
    True if the profile was collected successfully, False otherwise.

  Raises:
    ValueError: If the log_dir is not a GCS path.
  """
  if not str(log_dir).startswith("gs://"):
    raise ValueError(f"log_dir must be a GCS bucket path, got {log_dir}")

  request_json = {
      "duration_ms": duration_ms,
      "repository_path": log_dir,
  }
  address = urllib.parse.urljoin(f"http://{host}:{port}", "profiling")
  try:
    response = requests.post(address, json=request_json)
    response.raise_for_status()
  except requests.exceptions.RequestException:
    _logger.exception("Failed to collect profiling data")
    return False

  return True


def monkey_patch_jax():
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
      profiler_options: jax.profiler.ProfileOptions | None = None,  # pylint: disable=unused-argument
  ) -> None:
    _logger.debug("jax.profile.start_trace patched with pathways' start_trace")
    return start_trace(
        log_dir,
        create_perfetto_link=create_perfetto_link,
        create_perfetto_trace=create_perfetto_trace,
        profiler_options=profiler_options,
    )

  jax.profiler.start_trace = start_trace_patch
  jax._src.profiler.start_trace = start_trace_patch  # pylint: disable=protected-access

  def stop_trace_patch() -> None:
    _logger.debug("jax.profile.stop_trace patched with pathways' stop_trace")
    return stop_trace()

  jax.profiler.stop_trace = stop_trace_patch
  jax._src.profiler.stop_trace = stop_trace_patch  # pylint: disable=protected-access

  def start_server_patch(port: int):
    _logger.debug(
        "jax.profile.start_server patched with pathways' start_server"
    )
    return start_server(port)

  jax.profiler.start_server = start_server_patch

  def stop_server_patch():
    _logger.debug("jax.profile.stop_server patched with pathways' stop_server")
    return stop_server()

  jax.profiler.stop_server = stop_server_patch
