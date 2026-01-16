# Copyright 2026 Google LLC
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
"""Elasticity manager.

This class provides a utility for elastic training. It provides a decorator that
retries a function in case of `jax.errors.JaxRuntimeError` caused by slice down
events. It also provides a utility for waiting for slices to become active.
"""

import collections
from collections.abc import Mapping, Sequence
import logging
import time

import jax
import numpy as np
from pathwaysutils.debug import timing


_logger = logging.getLogger(__name__)

_SIMPLE_EXECUTION_TEST_VALUE = 100
_ELASTIC_DOWN_ERROR_TYPES = frozenset([
    "DATA_LOSS",
])
_ELASTIC_DOWN_ADDITIONAL_ERROR_TYPES = frozenset([
    "DEADLINE_EXCEEDED",
    "NOT_FOUND",
    "INTERNAL",
])


def _plus_one(x: jax.Array) -> jax.Array:
  """Adds one to each element in the array.

  Used to test if a slice is active.

  Args:
    x: The array to add one to.

  Returns:
      The array with one added to each element.
  """
  return x + 1


def _simple_execution(devices: Sequence[jax.Device]) -> jax.Array:
  """Simple execution to test if a slice is active.

  This function is used to test if a slice is active. It executes a simple
  computation on the devices and returns the result. If any of the devices are
  not active, the returned array will fail with a JaxRuntimeError used.

  Simply executing this function is not enough to determine if the slice is
  active. We also need to check the value of the returned array.

  Args:
    devices: The devices to execute on.

  Returns:
    The result of the execution.
  """
  if not devices:
    raise ValueError("No devices")

  test_input = np.zeros(len(devices), dtype=float) + (
      _SIMPLE_EXECUTION_TEST_VALUE - 1
  )

  return jax.pmap(_plus_one, devices=devices)(test_input)


def get_slice_to_devices(
    devices: Sequence[jax.Device],
) -> dict[int, Sequence[jax.Device]]:
  """Returns the mapping from slice index to devices."""
  slice_to_devices = collections.defaultdict(list)
  for d in devices:
    slice_to_devices[d.slice_index].append(d)
  return dict(slice_to_devices)


@timing.timeit
def get_active_slice_indices(
    slice_to_devices: Mapping[int, Sequence[jax.Device]] | None = None,
) -> set[int]:
  """Returns the set of active slices indices.

  Args:
    slice_to_devices: A mapping from slice index to devices. If None,
      `get_slice_to_devices(jax.devices())` is used to gather all available
      devices and group them by slice.

  Returns:
    A set of integers representing the indices of the active slices.
  """
  if slice_to_devices is None:
    _logger.debug("slice_to_devices is None. Getting from jax.devices().")
    slice_to_devices = get_slice_to_devices(tuple(jax.devices()))

  _logger.debug(
      "Getting active slice indices for slices: %s",
      sorted(list(slice_to_devices.keys())),
  )

  active_slice_indices = set()

  results = {
      slice_index: _simple_execution(devices)
      for slice_index, devices in slice_to_devices.items()
  }

  for slice_index, x in results.items():
    _logger.debug("Checking slice_index=%s", slice_index)
    expected = (
        np.zeros(len(slice_to_devices[slice_index]), dtype=float)
        + _SIMPLE_EXECUTION_TEST_VALUE
    )
    try:
      with timing.Timer(f"Checking {slice_index=}"):
        _logger.debug("Blocking until ready for slice_index=%s", slice_index)
        jax.block_until_ready(x)
        _logger.debug("Execution finished for slice_index=%s", slice_index)
        if np.allclose(x, expected):
          active_slice_indices.add(slice_index)
          _logger.debug("slice_index=%s active", slice_index)
        else:
          _logger.error(
              "Error with _simple_execution for slice_index=%s. "
              "This should never happen. Expected: %r, Actual: %r",
              slice_index,
              expected,
              x,
          )
          raise ValueError(
              f"Error with _simple_execution for slice_index={slice_index}."
          )
    except jax.errors.JaxRuntimeError as error:
      _logger.debug(
          "Caught JaxRuntimeError for slice_index=%s: %s", slice_index, error
      )
      if not is_error_due_to_slice_down(error):
        _logger.info("Re-raising error for slice_index=%s", slice_index)
        raise
      _logger.debug("slice_index=%s bad", slice_index)

  _logger.debug("active_slice_indices=%s", active_slice_indices)

  return active_slice_indices


def wait_for_slices(
    slice_count: int,
    poll_interval: float | int = 10,
    timeout: float | int | None = None,
    slice_to_devices: Mapping[int, Sequence[jax.Device]] | None = None,
) -> set[int]:
  """Waits until after at least `slice_count` slices become active.

  Args:
    slice_count: The number of slices to wait for.
    poll_interval: The minimum number of seconds to wait between availability
      checks. If the check takes longer than this, the next check will start
      immediately after the current check completes. Defaults to 10 seconds.
    timeout: The maximum number of seconds to wait. If None, there is no
      timeout.
    slice_to_devices: A mapping from slice index to devices. If None,
      `get_slice_to_devices(jax.devices())` is used.

  Returns:
    The active slice indices

  Raises:
    TimeoutError: If the timeout is reached before the slices become
      active.
  """
  if slice_to_devices is None:
    _logger.debug("slice_to_devices is None. Getting from jax.devices().")
    slice_to_devices = get_slice_to_devices(jax.devices())

  _logger.info(
      "Waiting for %s slices. Poll interval: %s, Timeout: %s",
      slice_count,
      poll_interval,
      timeout,
  )
  start_time = time.time()

  while True:
    check_start_time = time.time()

    _logger.debug("Checking active slices...")
    active_slice_indices = get_active_slice_indices(slice_to_devices)
    if len(active_slice_indices) >= slice_count:
      _logger.info(
          "Sufficient slices active: %s >= %s. Active indices: %s",
          len(active_slice_indices),
          slice_count,
          active_slice_indices,
      )
      return active_slice_indices

    _logger.info(
        "%s slices active. Wanting at least %s. Active indices: %s",
        len(active_slice_indices),
        slice_count,
        active_slice_indices,
    )

    time_to_sleep = max(0, poll_interval - (time.time() - check_start_time))

    if timeout is not None:
      elapsed_time = time.time() - start_time
      if elapsed_time + time_to_sleep >= timeout:
        raise TimeoutError(
            f"Timed out waiting for {slice_count} slices. Only"
            f" {len(active_slice_indices)} active after"
            f" {elapsed_time:.2f} seconds."
            f" Next check would occur after the timeout of {timeout}"
            " seconds."
        )

    if time_to_sleep > 0:
      _logger.debug("Sleeping for %.2f seconds.", time_to_sleep)

      time.sleep(time_to_sleep)


def is_error_due_to_slice_down(error: Exception) -> bool:
  """Returns True if the error is due to slice down.

  The error types that are considered due to slice down are
  jax.errors.JaxRuntimeError with the following error kind in the message:
  - DATA_LOSS
  - DEADLINE_EXCEEDED
  - NOT_FOUND
  - INTERNAL

  Args:
    error: The error to check.
  """
  error_due_to_slice_down = False
  traceback_logging_level = logging.DEBUG

  if isinstance(error, jax.errors.JaxRuntimeError):
    _logger.debug("Checking if JaxRuntimeError is due to slice down: %s", error)
    if any(
        error_type in str(error) for error_type in _ELASTIC_DOWN_ERROR_TYPES
    ):
      _logger.debug(
          "Caught an error due to slice down (matched"
          " _ELASTIC_DOWN_ERROR_TYPES)"
      )

      error_due_to_slice_down = True

    elif any(
        error_type in str(error)
        for error_type in _ELASTIC_DOWN_ADDITIONAL_ERROR_TYPES
    ):
      _logger.warning(
          "Caught an error that may or may not be due to slice down (matched"
          " _ELASTIC_DOWN_ADDITIONAL_ERROR_TYPES). This error will be treated"
          " as due to slice down."
      )
      traceback_logging_level = logging.WARNING

      error_due_to_slice_down = True

  if not error_due_to_slice_down:
    _logger.debug("Caught an error not due to slice down")

  _logger.log(traceback_logging_level, "Error details:", exc_info=True)

  return error_due_to_slice_down
