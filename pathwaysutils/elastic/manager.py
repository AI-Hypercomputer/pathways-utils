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
from typing import Any, TypeAlias, TypeVar
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
RetryPolicy: TypeAlias = Callable[[int, Exception], bool]


def _elastic_event_cleanup() -> None:
  """Cleans up JAX profiles, caches, and live arrays."""
  try:
    _logger.info("Cleaning up any ongoing traces")
    jax.profiler.stop_trace()
  except (RuntimeError, ValueError):
    _logger.info("No ongoing traces to clean up")
  except Exception:
    _logger.exception("Error cleaning up ongoing traces")
    raise

  jax.clear_caches()
  for array in jax.live_arrays():
    array.delete()


class ElasticRetryLimit:
  """A retry callback that limits the number of attempts."""

  def __init__(self, max_attempts: int):
    if max_attempts <= 0:
      raise ValueError("max_attempts must be positive.")
    self.max_attempts = max_attempts

  def __call__(self, attempt: int, error: Exception) -> bool:
    del error  # Unused
    return attempt < self.max_attempts


class Manager:
  """Utility class for elastic training.

  Attributes:
    slice_to_devices: A mapping from slice index to a sequence of `jax.Device`
      objects for that slice.
    all_slice_indices: A set of all possible slice indices in the allocation.
    active_slice_indices: A set of indices of the slices currently participating
      in the JAX computation mesh.
    inactive_slice_indices: A set of indices of the slices in the allocation
      that are not currently participating in the JAX computation mesh (e.g.
      because they are down or have not joined yet).
    available_inactive_slices: A set of indices of the inactive slices that
      have been detected as up and healthy by the background monitor thread,
      but have not yet been joined to the active JAX computation mesh.
  """

  slice_to_devices: Mapping[int, Sequence[jax.Device]]
  all_slice_indices: Set[int]
  active_slice_indices: Set[int]
  inactive_slice_indices: Set[int]
  available_inactive_slices: Set[int]
  _stop_event: threading.Event | None
  _monitor_thread: threading.Thread | None
  def __init__(self, devices: Sequence[jax.Device] | None = None) -> None:
    """Initializes the manager.

    Args:
      devices: The devices to use. If None, jax.devices() is used.
    """
    if devices is None:
      devices = jax.devices()
    self.slice_to_devices = elastic.get_slice_to_devices(devices)

    self.all_slice_indices = frozenset(self.slice_to_devices.keys())

    self.active_slice_indices = elastic.get_active_slice_indices(
        slice_to_devices=self.slice_to_devices
    )
    self.inactive_slice_indices = self.all_slice_indices - self.active_slice_indices
    self.available_inactive_slices = frozenset()

    self._stop_event = None
    self._monitor_thread = None

  def start_monitoring(self, poll_interval: float | int = 10) -> None:
    """Starts the background monitor thread.

    Args:
      poll_interval: The number of seconds to wait between activity checks.
    """
    if self._monitor_thread is not None and self._monitor_thread.is_alive():
      _logger.warning("Monitor thread is already running.")
      return

    self._stop_event = threading.Event()
    self._monitor_thread = threading.Thread(
        target=self._monitor_new_slices,
        args=(self._stop_event, poll_interval),
        daemon=True,
    )
    self._monitor_thread.start()
    _logger.info("Elastic monitor thread started with interval %s.", poll_interval)

  def close(self) -> None:
    """Stops the background monitor thread."""
    if self._stop_event is not None:
      self._stop_event.set()
    if self._monitor_thread is not None:
      _logger.info("Closing manager, waiting for monitor thread to stop...")
      try:
        self._monitor_thread.join(timeout=5)
        if self._monitor_thread.is_alive():
          _logger.warning(
              "Elastic monitor thread failed to stop within 5s timeout."
          )
      except RuntimeError as e:
        if "cannot join thread" in str(e):
          pass
        else:
          raise
      self._monitor_thread = None
      self._stop_event = None

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
  def new_slice_event(self) -> threading.Event:
    """Deprecated compatibility property for un-updated MaxText code.

    TODO: b/527183831 - Remove this property once MaxText CL 2 is submitted.
    """
    event = threading.Event()
    if self.available_inactive_slices:
      event.set()
    return event

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

  def _check_inactive_slices(self) -> None:
    """Checks inactive slices and updates available_inactive_slices."""
    if not self.inactive_slice_indices:
      _logger.debug("No inactive slices to check.")
      if self.available_inactive_slices:
        self.available_inactive_slices = frozenset()
      return

    _logger.debug(
        "Now checking inactive slices %s", self.inactive_slice_indices
    )
    inactive_slice_to_devices = {
        i: self.slice_to_devices[i] for i in self.inactive_slice_indices
    }
    found_slices = elastic.get_active_slice_indices(
        inactive_slice_to_devices
    )

    _logger.debug(
        "Found available and inactive slices %s", found_slices
    )

    # Filter against active_slice_indices in case the main thread initiated scale-up
    # and claimed these slices while this background health check was running.
    found_slices = found_slices - self.active_slice_indices

    if found_slices != self.available_inactive_slices:
      _logger.info(
          "Newly available but inactive slices %s", found_slices
      )
      self.available_inactive_slices = frozenset(found_slices)

  def _monitor_new_slices(
      self, stop_event: threading.Event, poll_interval: float | int
  ) -> None:
    """Monitors for new slices and updates available_inactive_slices."""
    _logger.info("Elastic monitor thread started.")
    try:
      while not stop_event.wait(poll_interval):
        try:
          self._check_inactive_slices()
        except Exception:  # pylint: disable=broad-exception-caught
          _logger.exception("Error in monitor thread loop")
    except BaseException as e:
      _logger.critical(
          "Catastrophic error in monitor thread, thread is dying!",
          exc_info=True,
      )
      raise
    finally:
      _logger.info("Elastic monitor thread stopped.")

  def elastic_retry(
      self,
      # TODO: b/523384843 - Remove max_retries parameter.
      max_retries: int | None = None,
      minimum_slice_count: int | None = None,
      poll_interval: float | int = 10,
      timeout: float | None = None,
      pre_callback: Callable[..., Any] | None = None,
      on_elastic_event_callback: Callable[..., Any] | None = None,
      retry_policy: RetryPolicy | None = None,
  ) -> Callable[[_F], _F]:
    """Retries a function with elasticity fault tolerance.

    This decorator wraps a function to automatically retry execution in case of
    `jax.errors.JaxRuntimeError` caused by slice down events. It waits for
    `minimum_slice_count` active slices before each attempt and cleans up JAX
    caches on failure.

    If `minimum_slice_count` is not met, the function will wait until at least
    `minimum_slice_count` slices are active before execution. If
    `minimum_slice_count` is None, it defaults to the total number of slices
    (i.e., it waits for all slices to be active).

    When `minimum_slice_count` is less than the total number of slices, a
    background thread will monitor for newly joined inactive slices and populate
    `self.available_inactive_slices`. User code can check this set (e.g. at step
    boundaries) and raise a `ScaleUpSignalError` to gracefully interrupt the
    current execution and trigger a retry with the expanded hardware.

    Often, the function will dispatch JAX operations and wait for them to
    complete while creating a log message. If using Python logging, it is
    recommended to set `logging.raiseExceptions=True` to ensure that the
    `jax.errors.JaxRuntimeError` is not silently ignored within the logging
    call.

    Args:
      max_retries: The maximum number of times to retry the function.
        Deprecated: Use `retry_policy` instead.
        TODO: b/523384843 - Remove max_retries parameter.
      minimum_slice_count: The minimum number of slices required to run the
        function. If None, defaults to the total number of slices.
      poll_interval: The number of seconds to wait between activity checks.
        Defaults to 10 seconds.
      timeout: The maximum number of seconds to wait for slices to become active
        before each retry attempt. If None, there is no timeout.
      pre_callback: A callback to call before the function is attempted.
      on_elastic_event_callback: A callback to call after an elastic failure
        occurs.
      retry_policy: A policy (callable) to determine if a retry should be
        attempted. It accepts the attempt number (1-indexed) and the exception
        that triggered the retry. If it returns False, no more retries are
        attempted. If neither `retry_policy` nor `max_retries` is specified, it
        defaults to unlimited retries.

    Returns:
      A decorator that retries the wrapped function.

    Raises:
      ElasticRuntimeError: If all retry attempts fail.
      Exception: Any other exception raised by the wrapped function that is not
        due to a slice down event.
    """
    target_slice_count = (
        self.total_slice_count
        if minimum_slice_count is None
        else minimum_slice_count
    )

    if max_retries is not None and retry_policy is not None:
      raise ValueError("Cannot specify both max_retries and retry_policy.")

    if retry_policy is None:
      if max_retries is None:
        # Default to unlimited retries if neither parameter is supplied.
        retry_policy = lambda attempt, error: True
      else:
        if max_retries <= 0:
          raise ValueError("max_retries must be positive.")
        retry_policy = ElasticRetryLimit(max_retries)

    def decorator(func: _F) -> _F:
      @functools.wraps(func)
      def wrapper(*args: Any, **kwargs: Any) -> Any:
        self.start_monitoring(poll_interval)

        def attempt_execution(attempt: int) -> Any:
          _logger.info("Elastic attempt %d", attempt)
          self.active_slice_indices = elastic.wait_for_slices(
              slice_count=target_slice_count,
              slice_to_devices=self.slice_to_devices,
              poll_interval=poll_interval,
              timeout=timeout,
          )
          self.inactive_slice_indices = (
              self.all_slice_indices - self.active_slice_indices
          )
          # Reset available_inactive_slices at attempt start since
          # active_slice_indices has just been updated by wait_for_slices.
          self.available_inactive_slices = frozenset()
          if pre_callback is not None:
            pre_callback()

          with jax.default_device(self.default_device):
            return func(*args, **kwargs)

        def handle_scale_up_error(attempt: int, error: ScaleUpSignalError) -> None:
          _logger.info("Scale up requested.")
          _elastic_event_cleanup()
          # Reset available_inactive_slices before retry callback and next attempt.
          self.available_inactive_slices = frozenset()

          if on_elastic_event_callback is not None:
            on_elastic_event_callback()

          if not retry_policy(attempt, error):
            _logger.info("Retry policy rejected retry after ScaleUpSignalError.")
            raise ElasticRuntimeError(
                f"Elastic attempt {attempt} failed."
            ) from error

        def handle_slice_down_error(
            attempt: int, error: jax.errors.JaxRuntimeError
        ) -> None:
          if not elastic.is_error_due_to_slice_down(error):
            raise

          _logger.exception("Elastic event detected")
          _elastic_event_cleanup()
          # Reset available_inactive_slices on slice-down failure before retry.
          self.available_inactive_slices = frozenset()

          if on_elastic_event_callback is not None:
            on_elastic_event_callback()

          if not retry_policy(attempt, error):
            _logger.info(
                "Retry policy rejected retry after jax.errors.JaxRuntimeError."
            )
            raise ElasticRuntimeError(
                f"Elastic attempt {attempt} failed."
            ) from error

        try:
          attempt = 1
          while True:
            try:
              return attempt_execution(attempt)
            except ScaleUpSignalError as error:
              handle_scale_up_error(attempt, error)
            except jax.errors.JaxRuntimeError as error:
              handle_slice_down_error(attempt, error)
            attempt += 1
        finally:
          self.close()

      return wrapper  # pyrefly: ignore[bad-return]

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
      timeout: The maximum number of seconds to wait for slices to become active
        before each retry attempt. If None, there is no timeout.
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
