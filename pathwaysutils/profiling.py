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
import logging
import os
import pathlib
import tempfile
import threading
import time
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


_profile_state = _ProfileState()
_original_start_trace = jax.profiler.start_trace
_original_stop_trace = jax.profiler.stop_trace


def toy_computation():
  """A toy computation to run before the first profile."""
  x = jax.jit(lambda x: x + 1)(jnp.array(1))
  x.block_until_ready()


def start_trace(gcs_bucket: str):
  """Starts a profiler trace."""
  with _profile_state.lock:
    if start_trace._first_profile_start:  # pylint: disable=protected-access, attribute-error
      start_trace._first_profile_start = False  # pylint: disable=protected-access
      toy_computation()

    if _profile_state.executable is not None:
      raise ValueError(
          "start_trace called while a trace is already being taken!"
      )
    _profile_state.executable = plugin_executable.PluginExecutable(
        f"{{profileRequest: {{traceLocation: '{gcs_bucket}'}}}}"
    )
    try:
      _profile_state.executable.call()[1].result()
    except:
      _profile_state.reset()
      raise

  _original_start_trace(gcs_bucket)


start_trace._first_profile_start = True  # pylint: disable=protected-access


def stop_trace():
  """Stops the currently-running profiler trace."""
  with _profile_state.lock:
    if _profile_state.executable is None:
      raise ValueError("stop_trace called before a trace is being taken!")
    try:
      _profile_state.executable.call()[1].result()
    except:
      _profile_state.reset()
      raise
    _profile_state.reset()

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
    log_dir: str,
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
  if not log_dir.startswith("gs://"):
    raise ValueError("log_dir must be a GCS path.")

  json = {
      "duration_ms": duration_ms,
      "repository_path": log_dir,
  }
  address = urllib.parse.urljoin(f"http://{host}:{port}", "profiling")
  try:
    response = requests.post(address, json=json)
    response.raise_for_status()
  except requests.exceptions.RequestException as e:
    _logger.error("Failed to collect profiling data: %s", e)
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
      create_perfetto_link: bool = False,  # pylint: disable=unused-argument
      create_perfetto_trace: bool = False,  # pylint: disable=unused-argument
  ) -> None:
    _logger.debug("jax.profile.start_trace patched with pathways' start_trace")
    return start_trace(log_dir)

  jax.profiler.start_trace = start_trace_patch

  def stop_trace_patch() -> None:
    _logger.debug("jax.profile.stop_trace patched with pathways' stop_trace")
    return stop_trace()

  jax.profiler.stop_trace = stop_trace_patch

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
