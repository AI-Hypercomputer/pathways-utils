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

from collections.abc import Callable, Mapping, Sequence, Set
import functools
import logging
import threading
from typing import Any, TypeVar
import warnings

import jax
from pathwaysutils.elastic import elastic


_logger = logging.getLogger(__name__)


class ElasticRuntimeError(RuntimeError):
  """Error raised when elasticity cannot continue."""


class ScaleUpSignalError(Exception):
  """Signals that the workload is ready to scale up.

  This exception should be raised by user code when it detects that new hardware
  is available and it wants to restart computation to make use of it.
  Raising this exception will interrupt the current computation and cause the
  elasticity manager to retry it with an updated slice configuration that
  includes the new hardware.
  """


_F = TypeVar("_F", bound=Callable[..., Any])


def _elastic_event_cleanup() -> None:
  """Cleans up JAX profiles, caches, and live arrays."""
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


class Manager:
  """Utility class for elastic training.

  Attributes:
    slice_to_devices: A mapping from slice index to a sequence of `jax.Device`
      objects for that slice.
    all_slice_indices: A set of all possible slice indices.
    active_slice_indices: A set of indices of the currently active slices.
    new_slice_event: A `threading.Event` that is set when new slices become
      available during replica/resize mode.
  """

  slice_to_devices: Mapping[int, Sequence[jax.Device]]
  all_slice_indices: Set[int]
  active_slice_indices: Set[int]
  new_slice_event: threading.Event

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
    self.new_slice_event = threading.Event()

  @functools.cached_property
  def total_slice_count(self) -> int:
    """The total number of slices."""
    return len(self.slice_to_devices)

  @property
  def default_device(self) -> jax.Device:
    """The device that should be set to the default device.

    This will be from one of the slices in `active_slice_indices`.
    """
    try:
      return self.slice_to_devices[next(iter(self.active_slice_indices))][0]
    except StopIteration as error:
      raise ValueError("No active slices") from error

  @property
  def active_slice_count(self) -> int:
    """The number of active slices."""
    return len(self.active_slice_indices)

  @property
  def inactive_slice_indices(self) -> set[int]:
    """The set of inactive slice indices."""
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

  def _cleanup_on_retry(self):
    """Cleans up JAX caches and traces on retry."""
    try:
      _logger.debug("Cleaning up any ongoing traces")
      jax.profiler.stop_trace()
    except (RuntimeError, ValueError):
      _logger.debug("No ongoing traces to clean up")
    except Exception:  # pylint: disable=broad-exception-caught
      _logger.exception("Error cleaning up ongoing traces")

    jax.clear_caches()
    for array in jax.live_arrays():
      array.delete()

  def _monitor_new_slices(
      self, stop_event: threading.Event, poll_interval: float | int
  ):
    """Monitors for new slices and sets the `new_slice_event` if found."""
    while not stop_event.wait(poll_interval):
      try:
        if not self.inactive_slice_indices:
          _logger.debug("No inactive slices to check.")
          continue

        _logger.debug(
            "Checking inactive slices: %s", self.inactive_slice_indices
        )
        inactive_slice_to_devices = {
            i: self.slice_to_devices[i] for i in self.inactive_slice_indices
        }
        newly_active_indices = elastic.get_active_slice_indices(
            inactive_slice_to_devices
        )

        if newly_active_indices:
          _logger.info(
              "New slices found: %s. Setting new slice event.",
              newly_active_indices,
          )
          self.new_slice_event.set()
          return

        _logger.debug("No new slices found.")
      except Exception:  # pylint: disable=broad-exception-caught
        _logger.exception("Error in monitor thread")

  def elastic_retry(
      self,
      max_retries: int,
      minimum_slice_count: int | None = None,
      poll_interval: float | int = 10,
      timeout: float | None = None,
      pre_callback: Callable[..., Any] | None = None,
      on_elastic_event_callback: Callable[..., Any] | None = None,
  ) -> Callable[[_F], _F]:
    """Retries a function with elasticity fault tolerance.

    This decorator wraps a function to automatically retry execution in case of
    `jax.errors.JaxRuntimeError` caused by slice down events. It waits for
    active slices before each attempt and cleans up JAX caches on failure.

    If `minimum_slice_count` is not met, the function will wait until at least
    `minimum_slice_count` slices are active before execution. If
    `minimum_slice_count` is None, it defaults to the total number of slices
    (i.e., it waits for all slices to be active).

    When `minimum_slice_count` is less than the total number of slices, a
    background thread will monitor for new slices becoming available and trigger
    a retry if they do.

    Often, the function will dispatch JAX operations and wait for them to
    complete while creating a log message. If using Python logging, it is
    recommended to set `logging.raiseExceptions=True` to ensure that the
    `jax.errors.JaxRuntimeError` is not silently ignored within the logging
    call.

    Args:
      max_retries: The maximum number of times to retry the function.
      minimum_slice_count: The minimum number of slices required to run the
        function. If None, defaults to the total number of slices.
      poll_interval: The number of seconds to wait between activity checks.
        Defaults to 10 seconds.
      timeout: The maximum number of seconds to wait for slices to become
        active before each retry attempt. If None, there is no timeout.
      pre_callback: A callback to call before the function is attempted.
      on_elastic_event_callback: A callback to call after an elastic failure
        occurs.

    Returns:
      A decorator that retries the wrapped function.

    Raises:
      ElasticRuntimeError: If all retry attempts fail.
      Exception: Any other exception raised by the wrapped function that is not
        due to a slice down event.
    """
    if minimum_slice_count is None:
      target_slice_count = self.total_slice_count
    else:
      target_slice_count = minimum_slice_count

    if max_retries <= 0:
      raise ValueError("max_retries must be positive.")

    def decorator(func: _F) -> _F:
      @functools.wraps(func)
      def wrapper(*args: Any, **kwargs: Any) -> Any:
        def attempt_execution(retry_index: int):
          _logger.info(
              "Elastic attempt %d out of %d", retry_index + 1, max_retries
          )
          self.active_slice_indices = elastic.wait_for_slices(
              slice_count=target_slice_count,
              slice_to_devices=self.slice_to_devices,
              poll_interval=poll_interval,
              timeout=timeout,
          )
          if pre_callback is not None:
            pre_callback()

          with jax.default_device(self.default_device):
            self.new_slice_event.clear()
            stop_event = threading.Event()

            if target_slice_count < self.total_slice_count:
              monitor_thread = threading.Thread(
                  target=self._monitor_new_slices,
                  args=(stop_event, poll_interval),
                  daemon=True,
              )
              monitor_thread.start()
            else:
              monitor_thread = None

            try:
              return func(*args, **kwargs)
            finally:
              stop_event.set()
              if monitor_thread is not None:
                monitor_thread.join()

        for retry_index in range(max_retries):
          try:
            return attempt_execution(retry_index)
          except ScaleUpSignalError:
            _logger.info("Scale up requested. Retrying.")
            _elastic_event_cleanup()

            if on_elastic_event_callback is not None:
              on_elastic_event_callback()
          except jax.errors.JaxRuntimeError as error:
            if not elastic.is_error_due_to_slice_down(error):
              raise

            if self.new_slice_event.is_set():
              _logger.info(
                  "Slice down event and new slice available detected. Retrying."
              )
            else:
              _logger.info("Slice down event detected. Retrying.")

            _elastic_event_cleanup()

            if on_elastic_event_callback is not None:
              on_elastic_event_callback()
        else:
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
      pre_callback: Callable[..., Any] | None = None,
      on_elastic_event_callback: Callable[..., Any] | None = None,
  ) -> Callable[[_F], _F]:
    """Retries a function with pause/resume fault tolerance.

    DEPRECATED: Use `elastic_retry` instead.

    Args:
      max_retries: The maximum number of times to retry the function.
      poll_interval: The number of seconds to wait between activity checks.
        Defaults to 10 seconds.
      timeout: The maximum number of seconds to wait for slices to become
        active before each retry attempt. If None, there is no timeout.
      pre_callback: A callback to call before the function is attempted.
      on_elastic_event_callback: A callback to call after an elastic failure
        occurs.

    Returns:
      A decorator that retries the wrapped function.
    """
    warnings.warn(
        "`pause_resume` is deprecated. Please use `elastic_retry` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.elastic_retry(
        max_retries=max_retries,
        minimum_slice_count=None,
        poll_interval=poll_interval,
        timeout=timeout,
        pre_callback=pre_callback,
        on_elastic_event_callback=on_elastic_event_callback,
    )

  def replica_resize(
      self,
      max_resizes: int,
      poll_interval: float = 10,
      pre_callback: Callable[..., Any] | None = None,
      on_elastic_event_callback: Callable[..., Any] | None = None,
  ) -> Callable[[_F], _F]:
    """Retries a function with replica/resize fault tolerance.

    DEPRECATED: Use `elastic_retry` instead.

    Args:
      max_resizes: The maximum number of times to retry the function after
        resizing the replica count.
      poll_interval: The number of seconds to wait between active slice checks.
        Defaults to 10 seconds.
      pre_callback: A callback to call before the function is attempted.
      on_elastic_event_callback: A callback to call after an elastic failure
        occurs.

    Returns:
      A decorator that retries the wrapped function.
    """
    warnings.warn(
        "`replica_resize` is deprecated. Please use `elastic_retry` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.elastic_retry(
        max_retries=max_resizes,
        minimum_slice_count=1,
        poll_interval=poll_interval,
        pre_callback=pre_callback,
        on_elastic_event_callback=on_elastic_event_callback,
    )
