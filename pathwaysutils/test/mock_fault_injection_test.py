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
"""Unit tests for DeviceLossSimulator mock fault injection utility."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from pathwaysutils.test import mock_fault_injection


class MockFaultInjectionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.simulator = mock_fault_injection.DeviceLossSimulator()
    self.simulator.start()
    self.devices = jax.devices()

  def tearDown(self):
    self.simulator.stop()
    super().tearDown()

  def test_normal_array_access(self):
    arr = jnp.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(np.asarray(arr), [1.0, 2.0, 3.0])

  def test_device_loss_raises_jax_runtime_error(self):
    dev = self.devices[0]
    arr = jnp.array([1.0, 2.0, 3.0])
    self.simulator.bind_array_to_devices(arr, [dev])

    self.simulator.mark_device_lost(dev)

    with self.assertRaises(jax.errors.JaxRuntimeError) as ctx:
      _ = np.asarray(arr)
    self.assertIn("DATA_LOSS", str(ctx.exception))

  def test_scalar_array_corruption(self):
    """Verifies that scalar arrays (0-d) are correctly marked as corrupted on device loss."""
    dev = self.devices[0]
    scalar_arr = jnp.array(5.0)
    self.simulator.bind_array_to_devices(scalar_arr, [dev])

    self.simulator.mark_device_lost(dev)

    with self.assertRaises(jax.errors.JaxRuntimeError) as ctx:
      _ = np.asarray(scalar_arr)
    self.assertIn("DATA_LOSS", str(ctx.exception))

  def test_sticky_loss_after_reconnection(self):
    """Verifies that an array marked as lost permanently raises DATA_LOSS."""
    dev = self.devices[0]
    arr = jnp.array([1.0, 2.0, 3.0])
    self.simulator.bind_array_to_devices(arr, [dev])

    # Reading before loss works
    _ = np.asarray(arr)

    # Mark device lost and trigger corruption check on array
    self.simulator.mark_device_lost(dev)
    with self.assertRaises(jax.errors.JaxRuntimeError):
      _ = np.asarray(arr)

    # Mark device as connected again
    self.simulator.mark_device_connected(dev)

    # Existing array MUST STILL raise DATA_LOSS forever ("sticky loss")
    with self.assertRaises(jax.errors.JaxRuntimeError) as ctx:
      _ = np.asarray(arr)
    self.assertIn("DATA_LOSS", str(ctx.exception))

    # A newly created array on the re-connected device SHOULD work
    new_arr = jnp.array([4.0, 5.0, 6.0])
    np.testing.assert_array_equal(np.asarray(new_arr), [4.0, 5.0, 6.0])

  def test_dependent_array_corruption_propagation(self):
    """Verifies operations with corrupted input produce corrupted outputs."""
    dev = self.devices[0]
    a = jnp.array([1.0, 2.0])
    b = jnp.array([3.0, 4.0])
    self.simulator.bind_array_to_devices(a, [dev])
    self.simulator.bind_array_to_devices(b, [dev])
    self.simulator.mark_device_lost(dev)

    # Evaluate 'a' to register its corruption
    self.assertTrue(self.simulator.is_array_corrupted(a))

    # Operation where 'a' was an input
    c = a + b

    # 'c' must also be marked corrupted and raise DATA_LOSS
    with self.assertRaises(jax.errors.JaxRuntimeError) as ctx:
      _ = np.asarray(c)
    self.assertIn("DATA_LOSS", str(ctx.exception))

  def test_decorated_jitted_function_propagates_corruption(self):
    """Verifies that @jax.jit with arguments propagates corruption."""

    @jax.jit(static_argnums=(0,))
    def my_jitted_func(static_val, x):
      return x * 2.0

    dev = self.devices[0]

    arr = jnp.array([1.0, 2.0])

    self.simulator.bind_array_to_devices(arr, [dev])
    self.simulator.mark_device_lost(dev)

    out = my_jitted_func(5, arr)
    with self.assertRaises(jax.errors.JaxRuntimeError) as ctx:
      _ = np.asarray(out)
    self.assertIn("DATA_LOSS", str(ctx.exception))

  def test_undecorated_jitted_function_propagates_corruption(self):
    """Verifies that jax.jit(fn) propagates corruption."""

    def my_func(x):
      return x * 2.0

    my_jitted_func = jax.jit(my_func)
    dev = self.devices[0]
    arr = jnp.array([1.0, 2.0])
    self.simulator.bind_array_to_devices(arr, [dev])
    self.simulator.mark_device_lost(dev)

    out = my_jitted_func(arr)
    with self.assertRaises(jax.errors.JaxRuntimeError) as ctx:
      _ = np.asarray(out)
    self.assertIn("DATA_LOSS", str(ctx.exception))

  def test_block_until_ready_raises_data_loss(self):
    dev = self.devices[0]
    arr = jnp.array([1.0, 2.0])
    self.simulator.bind_array_to_devices(arr, [dev])
    self.simulator.mark_device_lost(dev)

    with self.assertRaises(jax.errors.JaxRuntimeError) as ctx:
      arr.block_until_ready()
    self.assertIn("DATA_LOSS", str(ctx.exception))

    with self.assertRaises(jax.errors.JaxRuntimeError) as ctx:
      jax.block_until_ready(arr)
    self.assertIn("DATA_LOSS", str(ctx.exception))

  def test_device_lost_and_reconnected_without_reading(self):
    """Verifies array corruption when device is lost and reconnected without prior reading."""
    dev = self.devices[0]
    arr = jnp.array([1.0, 2.0, 3.0])
    self.simulator.bind_array_to_devices(arr, [dev])

    # Device is lost (arr is untracked prior to mark_device_lost)
    self.simulator.mark_device_lost(dev)

    # Device is reconnected WITHOUT reading arr during the loss period
    self.simulator.mark_device_connected(dev)

    # Reading arr now MUST still raise DATA_LOSS
    with self.assertRaises(jax.errors.JaxRuntimeError) as ctx:
      _ = np.asarray(arr)
    self.assertIn("DATA_LOSS", str(ctx.exception))

  def test_mock_devices_oss_topology(self):
    """Verifies multi-device and multi-slice topology fault injection in OSS JAX using magic mocks."""
    mock_devices, slice_to_devices = mock_fault_injection.create_mock_devices(
        num_slices=2, devices_per_slice=2
    )

    arr_slice0 = jnp.array([1.0, 2.0])
    arr_slice1 = jnp.array([3.0, 4.0])

    # Bind arrays to mock devices in Slice 0 and Slice 1
    self.simulator.bind_array_to_devices(arr_slice0, slice_to_devices[0])
    self.simulator.bind_array_to_devices(arr_slice1, slice_to_devices[1])

    # Simulate Slice 0 failure
    self.simulator.mark_slice_lost(0, slice_to_devices)

    # Array on Slice 0 MUST raise DATA_LOSS
    with self.assertRaises(jax.errors.JaxRuntimeError) as ctx:
      _ = np.asarray(arr_slice0)
    self.assertIn("DATA_LOSS", str(ctx.exception))

    # Array on Slice 1 MUST remain healthy and readable
    np.testing.assert_array_equal(np.asarray(arr_slice1), [3.0, 4.0])


if __name__ == "__main__":
  absltest.main()
