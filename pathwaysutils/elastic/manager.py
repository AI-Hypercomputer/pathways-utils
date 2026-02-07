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

import _thread
from collections.abc import Callable, Mapping, Sequence
import functools
import logging
import threading
from typing import Any, TypeVar

import jax
from pathwaysutils.elastic import elastic


_logger = logging.getLogger(__name__)


class ElasticRuntimeError(RuntimeError):
  """Error raised when elasticity cannot continue."""


class NewSliceAvailableError(RuntimeError):
  """Error raised when a new slice is available."""


_F = TypeVar("_F", bound=Callable[..., Any])


class Manager:
  """Utility class for elastic training."""

  _total_slice_count: int | None = None
  slice_to_devices: Mapping[int, Sequence[jax.Device]]
  active_slice_indices: set[int]

  def __init__(self, devices: Sequence[jax.Device] | None = None) -> None:
    """Initializes the manager.

    Args:
      devices: The devices to use. If None, jax.devices() is used.
    """
    if devices is None:
      devices = jax.devices()
    self.slice_to_devices = elastic.get_slice_to_devices(devices)

    self.all_slice_indices = set(self.slice_to_devices.keys())

    self.active_slice_indices = elastic.get_active_slice_indices(
        slice_to_devices=self.slice_to_devices
    )

  @property
  def total_slice_count(self) -> int:
    """Returns the total number of slices."""
    if self._total_slice_count is None:
      self._total_slice_count = len(self.slice_to_devices)
    return self._total_slice_count

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

  @property
  def inactive_slice_indices(self) -> set[int]:
    """Returns the set of inactive slice indices."""
    return self.all_slice_indices - self.active_slice_indices

  def scale_by_active_slices(self, x: int | float) -> int | float:
    """Scale x by the number of active slices."""
    if isinstance(x, int):
      quotient, remainder = divmod(
          x * self.active_slice_count, self.total_slice_count
      )
      if remainder:
        raise ValueError(
            f"Cannot scale {x=} by active slices because it will result in a "
            f"remainder of {remainder=}."
        )
      return quotient
    elif isinstance(x, float):
      return x * self.active_slice_count / self.total_slice_count
    else:
      raise ValueError(f"Unsupported type: {type(x)=}")

  def _elasticity_retry_decorator(
      self,
      max_retries: int,
      pre_func: Callable[..., Any] | None = None,
  ) -> Callable[[_F], _F]:
    """Retries a function with elasticity fault tolerance."""
    def decorator(func):
      @functools.wraps(func)
      def wrapper(*args, **kwargs):
        for retry_index in range(max_retries):
          try:
            _logger.info(
                "Elastic attempt %d out of %d", retry_index + 1, max_retries
            )
            if pre_func is not None:
              pre_func()

            with jax.default_device(self.default_device):
              return func(*args, **kwargs)
          except (jax.errors.JaxRuntimeError, NewSliceAvailableError) as error:
            if isinstance(
                error, jax.errors.JaxRuntimeError
            ) and not elastic.is_error_due_to_slice_down(error):
              raise

            _logger.info("Slice down event detected. Retrying.")

            try:
              _logger.debug("Cleaning up any ongoing traces")
              jax.profiler.stop_trace()
            except (RuntimeError, ValueError) as e:
              _logger.debug("No ongoing traces to clean up")
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

  def pause_resume(
      self,
      max_retries: int,
      poll_interval: float | int = 10,
      timeout: float | None = None,
  ) -> Callable[[_F], _F]:
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
    def pre_func():
      elastic.wait_for_slices(
          slice_count=self.total_slice_count,
          slice_to_devices=self.slice_to_devices,
          poll_interval=poll_interval,
          timeout=timeout,
      )

    return self._elasticity_retry_decorator(
        max_retries=max_retries, pre_func=pre_func
    )

  def replica_resize(
      self,
      max_resizes: int,
      poll_interval: float = 10,
  ) -> Callable[[_F], _F]:
    """Retries a function with replica/resize fault tolerance.

    Args:
      max_resizes: The maximum number of times to retry the function after
        resizing the replica count.
      poll_interval: The number of seconds to wait between active slice checks.
        Defaults to 10 seconds.

    Returns:
      The result of the wrapped function.

    Raises:
      ElasticRuntimeError: If all retry attempts fail.
      Exception: Any other exception raised by the wrapped function that is not
        due to a slice down event.
    """

    def pre_func():
      elastic.wait_for_slices(
          slice_count=1,
          slice_to_devices=self.slice_to_devices,
          poll_interval=poll_interval,
      )
      self.active_slice_indices = elastic.get_active_slice_indices(
          self.slice_to_devices
      )

    retry_decorator = self._elasticity_retry_decorator(
        max_retries=max_resizes, pre_func=pre_func
    )

    def decorator(func):
      @functools.wraps(func)
      def wrapper(*args, **kwargs):
        stop_event = threading.Event()
        new_slice_event = threading.Event()

        def monitor():
          while not stop_event.wait(poll_interval):
            try:
              if not self.inactive_slice_indices:
                _logger.debug("No inactive slices to check.")
                continue

              _logger.debug(
                  "Checking inactive slices: %s", self.inactive_slice_indices
              )
              inactive_slice_to_devices = {
                  i: self.slice_to_devices[i]
                  for i in self.inactive_slice_indices
              }
              newly_active_indices = elastic.get_active_slice_indices(
                  inactive_slice_to_devices
              )

              if newly_active_indices:
                _logger.info(
                    "New slices found: %s. Interrupting main thread.",
                    newly_active_indices,
                )
                new_slice_event.set()
                _thread.interrupt_main()
                return

              _logger.debug("No new slices found.")
            except Exception:  # pylint: disable=broad-exception-caught
              _logger.exception("Error in monitor thread")

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        try:
          return func(*args, **kwargs)
        except KeyboardInterrupt:
          if new_slice_event.is_set():
            raise NewSliceAvailableError("New slice available") from None
          raise
        finally:
          stop_event.set()
          try:
            monitor_thread.join()
          except KeyboardInterrupt:
            if new_slice_event.is_set():
              raise NewSliceAvailableError("New slice available") from None
            raise

      return retry_decorator(wrapper)

    return decorator
