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
"""Mock fault injection test utility for JAX elastic training.

Simulates device loss and array corruption for testing elasticity logic
without requiring physical pod/node cordoning or deletion.
"""

import functools
from typing import Any, Mapping, Sequence, Set
from unittest import mock
import jax
import jax.core
import jax.extend.core
import numpy as np


def create_mock_devices(
    num_slices: int = 2, devices_per_slice: int = 2, platform: str = "tpu"
) -> tuple[Sequence[jax.Device], dict[int, list[jax.Device]]]:
  """Creates mock devices and slice mappings for OSS testing."""
  devices = []
  slice_to_devices = {i: [] for i in range(num_slices)}
  for slice_idx in range(num_slices):
    for i in range(devices_per_slice):
      dev_id = slice_idx * devices_per_slice + i
      dev = mock.MagicMock(spec=jax.Device)
      dev.id = dev_id
      dev.slice_index = slice_idx
      dev.platform = platform
      dev.device_kind = "Mock Device"
      devices.append(dev)
      slice_to_devices[slice_idx].append(dev)
  return tuple(devices), slice_to_devices


class DeviceLossSimulator:
  """Simulates device loss and sticky array corruption in JAX."""

  def __init__(self):
    self._lost_devices: Set[int] = set()
    self._corrupted_array_ids: Set[int] = set()
    self._array_devices: dict[int, Set[int]] = {}
    self._generation: int = 1
    self._device_lost_history: dict[int, list[int]] = {}
    self._array_generation: dict[int, int] = {}
    self._active = False
    self._patchers: list[Any] = []

  def bind_array_to_devices(
      self, arr: jax.Array | jax.core.Tracer, devices: Sequence[jax.Device]
  ) -> None:
    """Explicitly binds a JAX array to mock or custom devices for OSS testing."""
    if isinstance(arr, jax.Array) and not isinstance(arr, jax.core.Tracer):
      arr_id = id(arr)
      dev_ids = {dev.id for dev in devices}
      self._array_devices[arr_id] = dev_ids
      if arr_id not in self._array_generation:
        self._array_generation[arr_id] = self._generation

  def track_array(self, arr: Any) -> None:
    """Tracks a JAX array generation and device mapping."""
    if isinstance(arr, jax.Array) and not isinstance(arr, jax.core.Tracer):
      arr_id = id(arr)
      if arr_id not in self._array_devices:
        dev_ids = {dev.id for dev in arr.devices()}
        self._array_devices[arr_id] = dev_ids
      else:
        dev_ids = self._array_devices[arr_id]
      if arr_id not in self._array_generation:
        self._array_generation[arr_id] = self._generation
        if all(d_id not in self._lost_devices for d_id in dev_ids):
          self._corrupted_array_ids.discard(arr_id)

  def mark_device_lost(self, device: jax.Device) -> None:
    """Marks a JAX device as lost and records loss generation."""
    self._lost_devices.add(device.id)
    self._generation += 1
    if device.id not in self._device_lost_history:
      self._device_lost_history[device.id] = []
    self._device_lost_history[device.id].append(self._generation)

    for arr_id, dev_ids in list(self._array_devices.items()):
      if device.id in dev_ids:
        self._corrupted_array_ids.add(arr_id)

  def mark_device_connected(self, device: jax.Device) -> None:
    """Marks a JAX device as re-connected."""
    self._lost_devices.discard(device.id)
    self._generation += 1

  def mark_slice_lost(
      self,
      slice_index: int,
      slice_to_devices: Mapping[int, Sequence[jax.Device]],
  ) -> None:
    """Marks all devices in a slice as lost."""
    for dev in slice_to_devices[slice_index]:
      self.mark_device_lost(dev)

  def mark_slice_connected(
      self,
      slice_index: int,
      slice_to_devices: Mapping[int, Sequence[jax.Device]],
  ) -> None:
    """Marks all devices in a slice as connected."""
    for dev in slice_to_devices[slice_index]:
      self.mark_device_connected(dev)

  def is_array_corrupted(self, arr: Any) -> bool:
    """Checks if an array is corrupted due to sticky loss or lost devices."""
    if not isinstance(arr, jax.Array) or isinstance(arr, jax.core.Tracer):
      return False

    self.track_array(arr)
    arr_id = id(arr)

    if arr_id in self._corrupted_array_ids:
      return True

    dev_ids = self._array_devices.get(arr_id, {dev.id for dev in arr.devices()})
    arr_gen = self._array_generation.get(arr_id, self._generation)

    if any(d_id in self._lost_devices for d_id in dev_ids):
      if arr.ndim > 0:
        self._corrupted_array_ids.add(arr_id)
        return True

    for d_id in dev_ids:
      loss_history = self._device_lost_history.get(d_id, [])
      if any(arr_gen <= lost_gen for lost_gen in loss_history):
        if arr.ndim > 0:
          self._corrupted_array_ids.add(arr_id)
          return True

    return False

  def check_and_raise_if_corrupted(self, arr: Any) -> None:
    """Raises JaxRuntimeError with DATA_LOSS if array is corrupted."""
    if self.is_array_corrupted(arr):
      raise jax.errors.JaxRuntimeError(
          "DATA_LOSS: Simulated device loss error accessing array on lost "
          "device"
      )

  def propagate_corruption(self, inputs: Any, outputs: Any) -> Any:
    """Propagates corruption from input PyTree leaves to output PyTree leaves."""
    input_leaves = [
        x
        for x in jax.tree_util.tree_leaves(inputs)
        if isinstance(x, jax.Array) and not isinstance(x, jax.core.Tracer)
    ]
    output_leaves = [
        x
        for x in jax.tree_util.tree_leaves(outputs)
        if isinstance(x, jax.Array) and not isinstance(x, jax.core.Tracer)
    ]

    for leaf in input_leaves + output_leaves:
      self.track_array(leaf)

    for out_leaf in output_leaves:
      out_id = id(out_leaf)
      self._array_generation[out_id] = self._generation

    if any(self.is_array_corrupted(x) for x in input_leaves):
      for out_leaf in output_leaves:
        self._corrupted_array_ids.add(id(out_leaf))
    else:
      for out_leaf in output_leaves:
        self._corrupted_array_ids.discard(id(out_leaf))
    return outputs

  def _patch(self, target: Any, attr: str, wrapper_factory: Any) -> Any:
    """Creates and starts a patcher using a factory receiving the unmocked attribute."""
    orig = getattr(target, attr, None)
    if orig is None:
      return None
    patcher = mock.patch.object(target, attr, wrapper_factory(orig))
    patcher.start()
    self._patchers.append(patcher)
    return patcher

  def start(self) -> None:
    """Starts mocking JAX functions and array accessors."""
    if self._active:
      return
    self._active = True
    sim = self

    def check_arr(orig):
      def wrapper(a, *args, **kwargs):
        sim.check_and_raise_if_corrupted(a)
        return orig(a, *args, **kwargs)

      return wrapper

    def patched_value_factory(orig_value):
      def wrapper(self_arr):
        sim.check_and_raise_if_corrupted(self_arr)
        if isinstance(orig_value, property) and orig_value.fget is not None:
          return orig_value.fget(self_arr)
        return orig_value(self_arr)

      return property(wrapper)

    def patched_jax_block_until_ready_factory(orig_fn):
      def wrapper(x):
        if any(
            sim.is_array_corrupted(leaf)
            for leaf in jax.tree_util.tree_leaves(x)
        ):
          raise jax.errors.JaxRuntimeError(
              "DATA_LOSS: Simulated device loss error accessing array on lost "
              "device"
          )
        return orig_fn(x)

      return wrapper

    def patched_primitive_bind_factory(orig_bind):
      def wrapper(prim_self, *args, **params):
        out = orig_bind(prim_self, *args, **params)
        data_args = [a for a in args if not callable(a)]
        return sim.propagate_corruption(data_args, out)

      return wrapper

    def patched_jit_factory(orig_jit):
      def wrapper(fun, *args, **kwargs):
        jitted_fn = orig_jit(fun, *args, **kwargs)

        @functools.wraps(jitted_fn)
        def inner(*call_args, **call_kwargs):
          out = jitted_fn(*call_args, **call_kwargs)
          data_args = [a for a in call_args if not callable(a)]
          return sim.propagate_corruption(data_args, out)

        return inner

      return wrapper

    patchers = [
        self._patch(np, "asarray", check_arr),
        self._patch(np, "array", check_arr),
        self._patch(
            jax, "block_until_ready", patched_jax_block_until_ready_factory
        ),
        self._patch(
            jax.extend.core.Primitive, "bind", patched_primitive_bind_factory
        ),
        self._patch(jax, "jit", patched_jit_factory),
        self._patch(jax, "pmap", patched_jit_factory),
    ]

    for cls in [jax.Array] + list(jax.Array.__subclasses__()):
      patchers.extend([
          self._patch(cls, "__array__", check_arr),
          self._patch(cls, "_value", patched_value_factory),
          self._patch(cls, "__dlpack__", check_arr),
          self._patch(cls, "block_until_ready", check_arr),
      ])

    self._patchers = [p for p in patchers if p is not None]

  def stop(self) -> None:
    """Restores original JAX methods."""
    if not self._active:
      return
    self._active = False

    for p in reversed(self._patchers):
      p.stop()
    self._patchers.clear()

  def __enter__(self):
    self.start()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.stop()
