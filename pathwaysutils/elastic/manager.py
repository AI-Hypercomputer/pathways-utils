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

This class is responsible for managing the elastic training.

It is responsible for:
- Tracking the availability of slices.
- Tracking the number of elastic down events and reshard retries.
- Tracking the snapshots.
- Resharding the snapshots.
- Resharding down if the error is due to slice down.
- Resharding up if it is time to reshard.
- Resharding the snapshot.
"""

import sys
import collections
from collections.abc import Callable, Mapping, Sequence
import copy
import itertools
import logging
import traceback
from typing import Any, TypeAlias

import jax
import numpy as np
from pathwaysutils.debug import timing

PyTree: TypeAlias = Any

_logger = logging.getLogger(__name__)


class ElasticRuntimeError(RuntimeError):
  """Error raised when too many elastic down events or reshard retries occur."""


class Manager:
  """Utility class for elastic training."""
  _devices: Sequence[jax.Device]
  _total_slice_count: int | None = None
  slice_to_devices: Mapping[int, Sequence[jax.Device]]
  snapshot_period: int
  reshard_check_period: int
  max_elastic_down_event_count: int | None
  max_reshard_retry_count: int | None
  elastic_down_event_count: int
  reshard_retry_count: int
  good_slice_indices: set[int]
  # TODO b/407772100 - Support multiple snapshots.
  _snapshot: PyTree

  _SIMPLE_EXECUTION_TEST_VALUE = 100
  _ELASTIC_DOWN_ERROR_TYPES = [
      "DATA_LOSS",
      "NOT_FOUND",
      "INTERNAL",
  ]

  def __init__(
      self,
      devices: Sequence[jax.Device] | None = None,
      reshard_check_period: int = 1,
      snapshot_period: int = 1,
      max_elastic_down_event_count: int | None = None,
      max_reshard_retry_count: int | None = None,
  ) -> None:
    if devices is None:
      devices = jax.devices()
    self.devices = devices

    if reshard_check_period <= 0:
      raise ValueError(
          f"reshard_check_period must be positive: {reshard_check_period=}"
      )
    self.reshard_check_period = reshard_check_period

    if snapshot_period <= 0:
      raise ValueError(f"snapshot_period must be positive: {snapshot_period=}")
    self.snapshot_period = snapshot_period

    if (
        max_elastic_down_event_count is not None
        and max_elastic_down_event_count <= 0
    ):
      raise ValueError(
          "max_elastic_down_event_count must be positive or None:"
          f" {max_elastic_down_event_count=}"
      )
    self.max_elastic_down_event_count = max_elastic_down_event_count

    if max_reshard_retry_count is not None and max_reshard_retry_count <= 0:
      raise ValueError(
          "max_reshard_retry_count must be positive or None:"
          f" {max_reshard_retry_count=}"
      )
    self.max_reshard_retry_count = max_reshard_retry_count

    self.elastic_down_event_count = 0
    self.reshard_retry_count = 0

    self.good_slice_indices = self.get_slice_availability()
    self._snapshot = None

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

  @classmethod
  def _is_error_due_to_slice_down(cls, error: Exception) -> bool:
    """Check if the error is due to slice down."""
    return_value = any(
        error_type in str(error)
        for error_type in cls._ELASTIC_DOWN_ERROR_TYPES
    )
    if return_value:
      _logger.info("Caught an error due to slice down")
    else:
      _logger.info("Caught an error not due to slice down")

    _logger.debug("\n".join(traceback.format_exception(error)))

    return return_value

  @classmethod
  def _simple_execution(cls, devices: Sequence[jax.Device]) -> jax.Array:
    """Simple execution to test if a slice is available.

    This function is used to test if a slice is available. It executes a simple
    computation on the devices and returns the result. If any of the devices are
    not available, the returned array will fail with a JaxRuntimeError used.

    Simply executing this function is not enough to determine if the slice is
    available. We also need to check the value of the returned array.

    Args:
      devices: The devices to execute on.

    Returns:
      The result of the execution.
    """
    if not devices:
      raise ValueError("No devices")

    test_input = np.zeros(len(devices), dtype=float) + (
        cls._SIMPLE_EXECUTION_TEST_VALUE - 1
    )

    return jax.pmap(lambda x: x + 1, devices=devices)(test_input)

  @timing.timeit
  def get_slice_availability(self) -> set[int]:
    """Returns the set of good and bad slices."""
    good_slice_indices = set()

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
            good_slice_indices.add(slice_index)
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
        if not self._is_error_due_to_slice_down(error):
          raise
        _logger.info("slice_index=%s bad", slice_index)

    _logger.info("good_slice_indices=%s", good_slice_indices)

    return good_slice_indices

  def _is_ready_to_reshard(self, step: int) -> bool:
    """Returns if it is time to reshard.

    May update `good_slice_indices`.

    Args:
      step: The current step.
    """
    if step % self.reshard_check_period:
      return False
    if self.good_slice_count >= self.total_slice_count:
      return False

    good_slice_indices = self.get_slice_availability()

    # If any of the existing good slices are no longer good, we cannot reshard.
    if self.good_slice_indices - good_slice_indices:
      return False

    if len(good_slice_indices) == len(self.good_slice_indices):
      return False

    _logger.info("New slice available.")
    _logger.info(
        "Previous good slice indices: self.good_slice_indices=%s",
        self.good_slice_indices,
    )
    _logger.info(
        "Current good slice indices: %s", good_slice_indices
    )

    self.good_slice_indices = good_slice_indices

    return True

  @property
  def good_slice_to_devices(self) -> dict[int, Sequence[jax.Device]]:
    """The mapping from a good slice to its devices."""
    return {
        slice_index: self.slice_to_devices[slice_index]
        for slice_index in self.good_slice_indices
    }

  @property
  def good_devices(self) -> Sequence[jax.Device]:
    """Returns the good data slice indices."""
    return list(
        itertools.chain.from_iterable(self.good_slice_to_devices.values())
    )

  @property
  def default_device(self) -> jax.Device:
    """Returns the device that should be set to the default device."""
    try:
      return self.slice_to_devices[next(iter(self.good_slice_indices))][0]
    except StopIteration as error:
      raise ValueError("No good slices") from error

  @property
  def good_slice_count(self) -> int:
    """Returns the number of slices."""
    return len(self.good_slice_indices)

  def scale_by_good_slices(self, x: int | float) -> int | float:
    """Scale x by the number of good slices."""
    if isinstance(x, int):
      quotient, remainder = divmod(
          x * self.good_slice_count, self.total_slice_count
      )
      if remainder:
        raise ValueError(
            f"Cannot scale {x=} by good slices because it will result in a "
            f"remainder of {remainder=}."
        )
      return quotient
    elif isinstance(x, float):
      return x * self.good_slice_count / self.total_slice_count
    else:
      raise ValueError(f"Unsupported type: {type(x)=}")

  def _slice_down(self, reshard_retry: bool = False) -> None:
    """Function to react to a slice going down.

    This function does two things:
    1. Updates the good slice indices.
    2. Updates the elastic down event count and reshard retry count.

    Args:
      reshard_retry: Whether this is a reshard retry.

    Raises:
      ElasticRuntimeError: If the maximum number of elastic down events or
        reshard retries is reached.
    """
    _logger.info("Slice down")
    self.good_slice_indices = self.get_slice_availability()
    self.elastic_down_event_count += 1
    if reshard_retry:
      self.reshard_retry_count += 1
    else:
      self.reshard_retry_count = 0

    _logger.info(
        "elastic_down_event_count=%s max_elastic_down_event_count=%s",
        self.elastic_down_event_count,
        self.max_elastic_down_event_count,
    )
    if (
        self.max_elastic_down_event_count is not None
        and self.elastic_down_event_count >= self.max_elastic_down_event_count
    ):
      raise ElasticRuntimeError(
          "Max elastic down event count reached:"
          f" {self.max_elastic_down_event_count}"
      )

    _logger.info(
        "self.reshard_retry_count=%s self.max_reshard_retry_count=%s",
        self.reshard_retry_count,
        self.max_reshard_retry_count,
    )
    if (
        self.max_reshard_retry_count is not None
        and self.reshard_retry_count > self.max_reshard_retry_count
    ):
      raise ElasticRuntimeError(
          f"Max reshard retry count reached {self.max_reshard_retry_count=}"
      )

  # TODO: b/407772100 - Support multiple snapshots.
  def pop_snapshot(self) -> tuple[int, PyTree | None, PyTree | None]:
    """Pops next snapshot.

    This function is used to get the next snapshot and remove it from
    the manager. Calls will raise an error if there are no snapshot to pop.

    Returns:
      A tuple of the step and the snapshot.

    Raises:
      ElasticRuntimeError: If there is no snapshot to pop.
    """

    if self._snapshot is None:
      raise ElasticRuntimeError("No snapshot to pop.")

    step, snapshot_jax_arrays, snapshot_controller = (
        self._snapshot.pop(key)
        for key in ["step", "snapshot_jax_arrays", "snapshot_controller"]
    )
    self._snapshot = None

    return step, snapshot_jax_arrays, snapshot_controller

  @staticmethod
  def _get_snapshot_jax_arrays_size(snapshot_jax_arrays: PyTree | None) -> int:
    """Returns the size of a snapshot.

    Args:
      snapshot: The snapshot to get the size of.
    """
    return sum(leaf.nbytes for leaf in jax.tree.leaves(snapshot_jax_arrays))

  @staticmethod
  def _put_snapshot_jax_arrays_on_host(
      snapshot_jax_arrays: PyTree | None,
  ) -> PyTree | None:
    """Puts a copy of the snapshot on the host.

    Args:
      snapshot: The snapshot to move to the host. Must be a PyTree of JAX
        arrays or None.

    Returns:
      A copy of the snapshot on the host.
    """

    sharding_pinned_host = jax.tree.map(
        lambda x: x.sharding.with_memory_kind("pinned_host"), snapshot_jax_arrays
    )
    return jax.device_put(
        snapshot_jax_arrays,
        sharding_pinned_host,
        donate=False,
        may_alias=False,
    )

  @staticmethod
  def _put_snapshot_on_controller(
      snapshot: PyTree | None,
  ) -> PyTree | None:
    return copy.deepcopy(snapshot)

  # TODO: b/407772100 - Support multiple snapshots.
  @timing.timeit
  def maybe_snapshot(
      self,
      step: int,
      snapshot_jax_arrays: PyTree | None = None,
      snapshot_controller: PyTree | None = None,
      force: bool = False,
      block: bool = False,
  ) -> None:
    """Save step and a copy of a snapshot on the host if it is time to save.

    A snapshot is saved if:
    - `force` is True
    - `step` is a multiple of `snapshot_period`

    Args:
      step: The current step.
      snapshot: The snapshot to save. Must be a PyTree of JAX arrays.
      force: If True, save the snapshot regardless of the step.
      block: If True, block until the snapshot is ready.
    """
    if not force and step % self.snapshot_period:
      _logger.info("Not saving a snapshot")
      return

    total_nbytes = self._get_snapshot_jax_arrays_size(snapshot_jax_arrays)

    _logger.info("Saving a snapshot of %s bytes on host", total_nbytes)

    snapshot_jax_arrays_host = self._put_snapshot_jax_arrays_on_host(snapshot_jax_arrays)
    _logger.info("Snapshot dispatched")

    if block:
      jax.block_until_ready(snapshot_jax_arrays_host)
      _logger.info("Snapshot completed")

    snapshot_on_controller = self._put_snapshot_on_controller(snapshot_controller)
    self._snapshot = {
        "step": step,
        "snapshot_jax_arrays": snapshot_jax_arrays_host,
        "snapshot_controller": snapshot_on_controller,
    }

  @timing.timeit
  def get_resharded_snapshot(
      self, mesh: jax.sharding.Mesh
  ) -> tuple[int, PyTree | None, PyTree | None]:
    """Get the resharded snapshot.

    The snapshot on pinned memory is resharded to the new mesh. This snapshot is
    saved to the manager. Then the snapshot is copied from pinned memory to
    device memory and returned.

    Args:
      mesh: The mesh.

    Returns:
      The next step and snapshot resharded to the new mesh.
    """
    step, snapshot_jax_arrays, snapshot_controller = self.pop_snapshot()

    sharding_pinned_host = jax.tree.map(
        lambda x: jax.sharding.NamedSharding(
            mesh, x.sharding.spec, memory_kind="pinned_host"
        ),
        snapshot_jax_arrays,
    )
    resharded_jax_arrays_pinned_host = jax.device_put(
        snapshot_jax_arrays,
        sharding_pinned_host,
        donate=True,
        may_alias=False,
    )

    sharding_device = jax.tree.map(
        lambda x: x.sharding.with_memory_kind("device"),
        resharded_jax_arrays_pinned_host,
    )
    resharded_jax_arrays_device = jax.device_put(
        resharded_jax_arrays_pinned_host,
        sharding_device,
        donate=False,
        may_alias=False,
    )

    snapshot_on_controller = self._put_snapshot_on_controller(snapshot_controller)

    self._snapshot = {
        "step": step,
        "snapshot_jax_arrays": resharded_jax_arrays_pinned_host,
        "snapshot_controller": snapshot_on_controller,
    }

    return step, resharded_jax_arrays_device, snapshot_controller

  @timing.timeit
  def maybe_reshard_down(
      self,
      error: Exception,
      elastic_handler: Callable[..., Any],
      handler_args: tuple[Any, ...] | None = None,
      handler_kwargs: Mapping[str, Any] | None = None,
      reshard_retry: bool = False,
  ) -> Any:
    """Reshards down if the error is due to slice down.

    This should be called after catching an error. This function will check
    to see if the error is from an elastic event due to a lost slice. If so,
    it will call the elastic handler in a loop until success or the max retry
    attempts. If the error is not due to a lost slice, the error will be
    reraised. The return values of the elastic handler are passed through to the
    caller.

    Args:
      error: The error to check.
      elastic_handler: The elastic handler to call.
      handler_args: The args to pass to the elastic handler.
      handler_kwargs: The kwargs to pass to the elastic handler.
      reshard_retry: Whether this is a reshard retry.

    Returns:
      The return value of the elastic handler.

    Raises:
      error: If the error is not due to an elastic event.
      ElasticRuntimeError: If the maximum number of elastic down events or
        reshard retries is reached.
    """
    if handler_args is None:
      handler_args = ()

    if handler_kwargs is None:
      handler_kwargs = {}

    while True:
      if not self._is_error_due_to_slice_down(error):
        _logger.info(
            "Not resharding down because the error is not due to a slice down."
        )
        raise error from error.__cause__

      _logger.info("Resharding down")
      self._slice_down(reshard_retry)

      try:
        handler_return_values = elastic_handler(*handler_args, **handler_kwargs)
        break
      except jax.errors.JaxRuntimeError as e:
        _logger.info("Elastic handler raised an error.")
        error = e
        reshard_retry = True

    _logger.info("Successfully resharded down")
    return handler_return_values

  @timing.timeit
  def maybe_reshard_up(
      self,
      step: int,
      elastic_handler: Callable[..., Any],
      snapshot_jax_arrays: PyTree | None = None,
      snapshot_controller: PyTree | None = None,
      handler_args: tuple[Any, ...] | None = None,
      handler_kwargs: Mapping[str, Any] | None = None,
  ) -> Any:
    """Reshards up if it is time to reshard.

    This function will check to see if it is time to reshard up. If so, it will
    immediately snapshot (if a preexisting snapshot for the current step was not
    already taken) and call the elastic handler. If there is error the elastic
    handler, maybe_reshard_down will be called. If resharding occurs, the
    return values of the elastic handler are passed through to the caller.

    Args:
      step: The current step.
      snapshot: The snapshot to reshard.
      elastic_handler: The elastic handler to call. This function must work for
        both reshard up and reshard down.
      handler_args: The args to pass to the elastic handler.
      handler_kwargs: The kwargs to pass to the elastic handler.

    Returns:
      The return value of the elastic handler.
    """
    if handler_args is None:
      handler_args = ()

    if handler_kwargs is None:
      handler_kwargs = {}

    if not self._is_ready_to_reshard(step):
      _logger.info("Not resharding up since it is not time to reshard.")
      return

    self.maybe_snapshot(
        step=step,
        snapshot_jax_arrays=snapshot_jax_arrays,
        snapshot_controller=snapshot_controller,
        force=True,
        block=True,
    )

    try:
      handler_return_values = elastic_handler(*handler_args, **handler_kwargs)
    except jax.errors.JaxRuntimeError as error:
      _logger.info("Elastic handler failed. Trying again")
      handler_return_values = self.maybe_reshard_down(
          error=error,
          elastic_handler=elastic_handler,
          handler_args=handler_args,
          handler_kwargs=handler_kwargs,
          reshard_retry=True,
      )

    _logger.info("Finished resharding up")
    return handler_return_values
