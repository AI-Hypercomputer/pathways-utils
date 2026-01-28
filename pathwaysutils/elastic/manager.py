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
"""Elasticity manager.

This class provides a utility for elastic training. It provides a decorator that
retries a function in case of `jax.errors.JaxRuntimeError` caused by slice down
events. It also provides a utility for waiting for slices to become active.
"""

import collections
from collections.abc import Mapping, Sequence
import functools
import itertools
import logging
import time
import traceback
from typing import Any

import jax
import numpy as np
from pathwaysutils.debug import timing


_logger = logging.getLogger(__name__)


def _plus_one(x: jax.Array) -> jax.Array:
  """Adds one to each element in the array.

  Used to test if a slice is active.

  Args:
    x: The array to add one to.

  Returns:
      The array with one added to each element.
  """
  return x + 1


class ElasticRuntimeError(RuntimeError):
  """Error raised when elasticity cannot continue."""


class Manager:
  """Utility class for elastic training."""

  _devices: Sequence[jax.Device]
  _total_slice_count: int | None = None
  slice_to_devices: Mapping[int, Sequence[jax.Device]]
  active_slice_indices: set[int]

  _SIMPLE_EXECUTION_TEST_VALUE = 100
  _ELASTIC_DOWN_ERROR_TYPES = [
      "DATA_LOSS",
  ]
  _ELASTIC_DOWN_ADDITIONAL_ERROR_TYPES = [
      "DEADLINE_EXCEEDED",
      "NOT_FOUND",
      "INTERNAL",
  ]

  def __init__(self, devices: Sequence[jax.Device] | None = None) -> None:
    """Initializes the manager.

    Args:
      devices: The devices to use. If None, jax.devices() is used.
    """
    if devices is None:
      devices = jax.devices()
    self.devices = devices

    self.active_slice_indices = self.get_active_slice_indices()

  @property
  def devices(self) -> Sequence[jax.Device]:
    """Returns the devices."""
    return self._devices

  @devices.setter
  def devices(self, devices: Sequence[jax.Device]) -> None:
    """Sets the devices."""
    self._devices = devices

    self.slice_to_devices = collections.defaultdict(list)
    for d in self._devices:
      self.slice_to_devices[d.slice_index].append(d)
    self.slice_to_devices = dict(self.slice_to_devices)

  @property
  def total_slice_count(self) -> int:
    """Returns the total number of slices."""
    if self._total_slice_count is None:
      self._total_slice_count = len(self.slice_to_devices)
    return self._total_slice_count

  def slice_device_count(self, slice_index: int) -> int:
    """Returns the number of devices in a slice."""
    try:
      return len(self.slice_to_devices[slice_index])
    except KeyError as error:
      raise ValueError(
          f"Slice {slice_index=} not found in {self.slice_to_devices=}"
      ) from error

  def is_error_due_to_slice_down(self, error: Exception) -> bool:
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
      if any(
          error_type in str(error)
          for error_type in self._ELASTIC_DOWN_ERROR_TYPES
      ):
        _logger.info("Caught an error due to slice down")

        error_due_to_slice_down = True

      elif any(
          error_type in str(error)
          for error_type in self._ELASTIC_DOWN_ADDITIONAL_ERROR_TYPES
      ):
        _logger.warning(
            "Caught an error due that may or may not be due to slice down. This"
            " error will be treated as due to slice down."
        )
        traceback_logging_level = logging.WARNING

        error_due_to_slice_down = True

    if not error_due_to_slice_down:
      _logger.info("Caught an error not due to slice down")

    _logger.log(
        traceback_logging_level, "\n".join(traceback.format_exception(error))
    )

    return error_due_to_slice_down

  def _simple_execution(self, devices: Sequence[jax.Device]) -> jax.Array:
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
        self._SIMPLE_EXECUTION_TEST_VALUE - 1
    )

    return jax.pmap(_plus_one, devices=devices)(test_input)

  @timing.timeit
  def get_active_slice_indices(self) -> set[int]:
    """Returns the set of active slices indices."""
    active_slice_indices = set()

    results = {
        slice_index: self._simple_execution(devices)
        for slice_index, devices in self.slice_to_devices.items()
    }

    for slice_index, x in results.items():
      _logger.info("Checking slice_index=%s", slice_index)
      expected = (
          np.zeros(self.slice_device_count(slice_index), dtype=float)
          + self._SIMPLE_EXECUTION_TEST_VALUE
      )
      try:
        with timing.Timer(f"Checking {slice_index=}"):
          jax.block_until_ready(x)
          if np.allclose(x, expected):
            active_slice_indices.add(slice_index)
            _logger.info("slice_index=%s good", slice_index)
          else:
            _logger.error(
                "Error with _simple_execution for slice_index=%s. "
                "This should never happen. Expected: %s, Actual: %s",
                slice_index,
                expected,
                x,
            )
            raise ValueError(
                f"Error with _simple_execution for slice_index={slice_index}."
            )
      except jax.errors.JaxRuntimeError as error:
        if not self.is_error_due_to_slice_down(error):
          raise
        _logger.info("slice_index=%s bad", slice_index)

    _logger.info("active_slice_indices=%s", active_slice_indices)

    return active_slice_indices

  @property
  def active_slice_to_devices(self) -> dict[int, Sequence[jax.Device]]:
    """The mapping from a active slice to its devices."""
    return {
        slice_index: self.slice_to_devices[slice_index]
        for slice_index in self.active_slice_indices
    }

  @property
  def active_devices(self) -> list[jax.Device]:
    """Returns the active slice indices."""
    return list(
        itertools.chain.from_iterable(self.active_slice_to_devices.values())
    )

  @property
  def default_device(self) -> jax.Device:
    """Returns the device that should be set to the default device.

    This will be from one of the slices in `active_slice_indices`.
    """
    try:
      return self.slice_to_devices[next(iter(self.active_slice_indices))][0]
    except StopIteration as error:
      raise ValueError("No active slices") from error

  @property
  def active_slice_count(self) -> int:
    """Returns the number of slices."""
    return len(self.active_slice_indices)

  def scale_by_active_slices(self, x: int | float) -> int | float:
    """Scale x by the number of good slices."""
    if isinstance(x, int):
      quotient, remainder = divmod(
          x * self.active_slice_count, self.total_slice_count
      )
      if remainder:
        raise ValueError(
            f"Cannot scale {x=} by good slices because it will result in a "
            f"remainder of {remainder=}."
        )
      return quotient
    elif isinstance(x, float):
      return x * self.active_slice_count / self.total_slice_count
    else:
      raise ValueError(f"Unsupported type: {type(x)=}")

  def wait_for_slices(
      self,
      slice_count: int | None = None,
      poll_interval: float | int = 10,
      timeout: float | int | None = None,
  ) -> set[int]:
    """Waits until after at least `slice_count` slices become active.

    Args:
      slice_count: The number of slices to wait for. If None, waits for all
        slices to become active.
      poll_interval: The minimum number of seconds to wait between availability
        checks. If the check takes longer than this, the next check will start
        immediately after the current check completes. Defaults to 10 seconds.
      timeout: The maximum number of seconds to wait. If None, there is no
        timeout.

    Returns:
      The good slice indices

    Raises:
      TimeoutError: If the timeout is reached before the slices become
        active.
    """
    if slice_count is None:
      slice_count = self.total_slice_count

    start_time = time.time()

    while True:
      check_start_time = time.time()

      active_slice_indices = self.get_active_slice_indices()
      if len(active_slice_indices) >= slice_count:
        _logger.info(
            "%s/%s slices are active",
            len(active_slice_indices),
            self.total_slice_count,
        )
        return active_slice_indices

      _logger.info(
          "%s/%s slices active. Wanting at least %s/%s.",
          len(active_slice_indices),
          self.total_slice_count,
          slice_count,
          self.total_slice_count,
      )

      time_to_sleep = max(0, poll_interval - (time.time() - check_start_time))

      if (
          timeout is not None
          and (elapsed_time := time.time() - start_time) + time_to_sleep
          >= timeout
      ):
        raise TimeoutError(
            f"Timed out waiting for {slice_count} slices. Only"
            f" {len(active_slice_indices)} active after"
            f" {elapsed_time:.2f} seconds."
            f" Next check would occur after the timeout of {timeout}"
            " seconds."
        )

      if time_to_sleep > 0:
        _logger.info("Sleeping for %.2f seconds.", time_to_sleep)

        time.sleep(time_to_sleep)

  def pause_resume(
      self,
      max_retries: int,
      poll_interval: float | int = 10,
      timeout: float | None = None,
  ) -> Any:
    """Retries a function with pause/resume fault tolerance.

    This decorator wraps a function to automatically retry execution in case of
    `jax.errors.JaxRuntimeError` caused by slice down events. It waits for
    active slices before each attempt and cleans up JAX caches on failure.
    The function will not be attempted (or reattempted) until all of the slices
    are active.

    Often, the function will dispatch JAX operations and wait for them to
    complete while creating a log message. If using Python logging, it is
    recommended to set `logging.raiseExceptions=True` to ensure that the
    `jax.errors.JaxRuntimeError` is not silently ignored within the logging
    call.

    Args:
      max_retries: The maximum number of times to retry the function.
      poll_interval: The number of seconds to wait between activity checks.
        Defaults to 10 seconds.
      timeout: The maximum number of seconds to wait for slices to become
        active before each retry attempt. If None, there is no timeout.

    Returns:
      The result of the wrapped function.

    Raises:
      ElasticRuntimeError: If all retry attempts fail.
      Exception: Any other exception raised by the wrapped function that is not
        due to a slice down event.
    """
    def decorator(func):
      @functools.wraps(func)
      def wrapper(*args, **kwargs):
        for retry_index in range(max_retries):
          try:
            _logger.info(
                "Elastic attempt %d out of %d", retry_index + 1, max_retries
            )

            self.wait_for_slices(poll_interval=poll_interval, timeout=timeout)

            return func(*args, **kwargs)
          except jax.errors.JaxRuntimeError as error:
            if not self.is_error_due_to_slice_down(error):
              raise

            try:
              _logger.info("Cleaning up any ongoing traces")
              jax.profiler.stop_trace()
            except (RuntimeError, ValueError) as e:
              _logger.info("No ongoing traces to clean up")
            except Exception:
              _logger.exception("Error cleaning up ongoing traces")
              raise

            jax.clear_caches()
            for array in jax.live_arrays():
              array.delete()
        raise ElasticRuntimeError(
            f"Elastic attempt {max_retries} out of {max_retries} failed."
        )

      return wrapper
    return decorator
