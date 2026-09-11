# Copyright 2025 Google LLC
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
"""Initialization functions for Pathways-on-Cloud utilities."""

from collections.abc import Sequence
import concurrent.futures
import datetime
import logging
import os

import jax
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from pathwaysutils import profiling
from pathwaysutils import proxy_backend
from pathwaysutils.persistence import orbax_handler


_logger = logging.getLogger(__name__)
_initialization_count = 0


#  This is a brittle implementation since the platforms value is not necessarily
#  which backend is ultimately selected
def is_pathways_backend_used() -> bool:
  """Returns whether Pathways backend is used.

  This function checks the JAX platforms configuration to determine whether
  Pathways is used. If the platforms configuration contains the string "proxy",
  Pathways is used. This is a brittle implementation since the platforms value
  is not necessarily which backend is ultimately selected or there may be more
  than one platform specified and another may have higher priority.
  """
  return jax.config.jax_platforms and "proxy" in jax.config.jax_platforms


def _is_persistence_enabled() -> bool:
  """Returns whether persistence is enabled.

  This function checks the environment variable ENABLE_PATHWAYS_PERSISTENCE to
  determine whether persistence is enabled. If the variable is set to "1",
  persistence is enabled. If the variable is set to "0" or unset, persistence is
  disabled.

  Returns:
    True if persistence is enabled, False otherwise.
  """
  if "ENABLE_PATHWAYS_PERSISTENCE" in os.environ:
    if os.environ["ENABLE_PATHWAYS_PERSISTENCE"] == "1":
      return True
    if os.environ["ENABLE_PATHWAYS_PERSISTENCE"] == "0":
      return False
    else:
      raise ValueError(
          "ENABLE_PATHWAYS_PERSISTENCE must be set to 1/0 or unset, got: "
          + os.environ["ENABLE_PATHWAYS_PERSISTENCE"]
      )
  return False


def initialize() -> None:
  """Initializes pathwaysutils.

  This function is called by the user to initialize pathwaysutils. It is
  responsible for setting up the logging, profiling, and persistence handlers
  through various monkey patching functions. It is also responsible for
  registering the proxy backend factory.
  """
  global _initialization_count
  _initialization_count += 1

  # Ignoring the second call to initialize() is a temporary measure so that this
  # debug log is not triggered for customers who are following our instructions
  # and using the new initialize() function only once but have already had the
  # legacy initialization triggered.
  if _initialization_count > 1:
    _logger.debug("Already initialized. Ignoring duplicate call.")
    return

  _logger.debug("Starting initialize.")

  if is_pathways_backend_used():
    _logger.debug("Detected Pathways-on-Cloud backend. Applying changes.")
    proxy_backend.register_backend_factory()
    profiling.monkey_patch_jax()
    # TODO: b/365549911 - Remove when OCDBT-compatible
    if _is_persistence_enabled():
      orbax_handler.register_pathways_handlers(
          timeout=datetime.timedelta(hours=1),
          array_metadata_store=array_metadata_store_lib.Store(),
      )

    # Turn off JAX compilation cache because Pathways handles its own
    # compilation cache.
    jax.config.update("jax_enable_compilation_cache", False)

  else:
    _logger.debug(
        "Did not detect Pathways-on-Cloud backend. No changes applied."
    )


def wait_for_devices_ready(
    devices: Sequence[jax.Device] | None = None,
    timeout: float | int | None = None,
) -> None:
  """Waits for the given devices to be ready and available for computation.

  Args:
    devices: The sequence of JAX devices to wait for. If None, defaults to all
      available devices via `jax.devices()`.
    timeout: The maximum number of seconds to wait. If None, there is no timeout
      (waits indefinitely).

  Raises:
    TimeoutError: If the timeout is reached before the devices become ready.
  """
  if devices is None:
    devices = jax.devices()

  if not devices:
    return

  _logger.info(
      "Waiting for %d devices to be ready (timeout=%s).", len(devices), timeout
  )
  fn = lambda x: x + 1
  results = [jax.jit(fn, device=d)(0) for d in devices]
  if timeout is None:
    jax.block_until_ready(results)
  else:
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(jax.block_until_ready, results)
    try:
      future.result(timeout=timeout)
    except concurrent.futures.TimeoutError as e:
      executor.shutdown(wait=False, cancel_futures=True)
      raise TimeoutError(
          f"Timed out waiting for {len(devices)} devices to be ready after"
          f" {timeout} seconds."
      ) from e
    else:
      executor.shutdown(wait=True)

  _logger.info("All %d devices are ready.", len(devices))
