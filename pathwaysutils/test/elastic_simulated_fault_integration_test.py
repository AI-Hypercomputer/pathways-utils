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
"""Integration test for elasticity logic using DeviceLossSimulator.

Simulates device loss and recovery to test slice health checks and fault
handling without cordoning or deleting physical pods.
"""

import os

os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
)

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from pathwaysutils.elastic import elastic
from pathwaysutils.test import mock_fault_injection


class ElasticSimulatedFaultIntegrationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.simulator = mock_fault_injection.DeviceLossSimulator()
    self.simulator.start()
    self.devices = tuple(jax.devices())
    # Each "slice" is made up of one device to simulate multislice failures in
    # single host tests.
    self.slice_to_devices = {i: [dev] for i, dev in enumerate(self.devices)}

  def tearDown(self):
    self.simulator.stop()
    super().tearDown()

  @parameterized.parameters(0, 1)
  def test_simulated_slice_loss_and_recovery(self, target_slice: int):
    """Verifies elasticity slice health detection under simulated faults."""
    all_slices = set(self.slice_to_devices.keys())

    # 1. All slices active initially
    active_slices = elastic.get_active_slice_indices(self.slice_to_devices)
    self.assertEqual(active_slices, all_slices)

    # 2. Simulate loss of target slice
    self.simulator.mark_slice_lost(target_slice, self.slice_to_devices)

    active_after_loss = elastic.get_active_slice_indices(self.slice_to_devices)
    self.assertNotIn(target_slice, active_after_loss)
    self.assertEqual(active_after_loss, all_slices - {target_slice})

    # 3. Simulate recovery of target slice
    self.simulator.mark_slice_connected(target_slice, self.slice_to_devices)

    active_after_recovery = elastic.get_active_slice_indices(
        self.slice_to_devices
    )
    self.assertEqual(active_after_recovery, all_slices)

  def _create_array(self, device_indices: tuple[int, ...]) -> jax.Array:
    if len(device_indices) == 1:
      return jax.device_put(
          jnp.array([1.0, 2.0]), self.devices[device_indices[0]]
      )
    target_devices = [self.devices[i] for i in device_indices]
    mesh = Mesh(np.array(target_devices), ("devices",))
    sharding = NamedSharding(mesh, P("devices"))
    return jax.device_put(jnp.array([1.0, 2.0]), sharding)

  @parameterized.named_parameters(
      dict(
          testcase_name="single_device_on_lost_device",
          device_indices=(0,),
      ),
      dict(
          testcase_name="multi_device_including_lost_device",
          device_indices=(0, 1),
      ),
  )
  def test_array_loss_on_device_failure(self, device_indices: tuple[int, ...]):
    """Verifies arrays residing on failed devices raise slice-down errors."""
    arr = self._create_array(device_indices)

    self.simulator.mark_device_lost(self.devices[0])

    with self.assertRaises(jax.errors.JaxRuntimeError) as ctx:
      _ = arr.block_until_ready()
    self.assertTrue(elastic.is_error_due_to_slice_down(ctx.exception))

  @parameterized.named_parameters(
      dict(
          testcase_name="single_device_on_healthy_device",
          device_indices=(1,),
      ),
      dict(
          testcase_name="multi_device_excluding_lost_device",
          device_indices=(1, 2),
      ),
  )
  def test_array_accessible_on_healthy_devices(
      self, device_indices: tuple[int, ...]
  ):
    """Verifies arrays on healthy devices remain accessible when other devices fail."""
    arr = self._create_array(device_indices)

    self.simulator.mark_device_lost(self.devices[0])

    _ = arr.block_until_ready()
    np.testing.assert_array_equal(np.asarray(arr), [1.0, 2.0])


if __name__ == "__main__":
  absltest.main()
