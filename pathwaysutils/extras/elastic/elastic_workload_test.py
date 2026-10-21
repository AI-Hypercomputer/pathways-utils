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
"""Tests for elastic training workload."""

import json
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import orbax.checkpoint as ocp
from pathwaysutils.elastic import manager
from pathwaysutils.extras import simple_model
from pathwaysutils.extras.elastic import elastic_workload
from pathwaysutils.extras.elastic import gke_demo_bridge
from pathwaysutils.test import mock_fault_injection


def setUpModule():
  jax.config.update("jax_num_cpu_devices", 2)


class _FakeSnapshotter:
  """In-memory fake Snapshotter for testing without Pathways backend."""

  def __init__(self):
    self.saved: dict[int, dict[str, object]] = {}
    self.latest = None

  def save(self, step: int, payload: dict[str, object]) -> None:
    self.saved[step] = payload
    self.latest = mock.MagicMock(step=step)

  def join(self) -> None:
    pass

  def load(
      self, abstract_state: dict[str, object], reset_snapshot_state: bool = True
  ) -> dict[str, object]:
    del abstract_state, reset_snapshot_state
    assert self.latest is not None
    return self.saved[self.latest.step]


class ElasticWorkloadSimulationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.monitor_patcher = mock.patch.object(
        manager.Manager, "start_monitoring", autospec=True
    )
    self.monitor_patcher.start()
    self.addCleanup(self.monitor_patcher.stop)

    self.snapshotter_patcher = mock.patch.object(
        ocp.v1.training.pathways, "Snapshotter", side_effect=_FakeSnapshotter
    )
    self.snapshotter_patcher.start()
    self.addCleanup(self.snapshotter_patcher.stop)

    self.simulator = mock_fault_injection.DeviceLossSimulator()
    self.simulator.start()
    self.addCleanup(self.simulator.stop)

    self.temp_dir = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp_dir.cleanup)

  def _create_mock_elastic_manager(
      self, num_slices: int = 2, devices_per_slice: int = 1
  ) -> tuple[manager.Manager, dict[int, list[jax.Device]]]:
    _, slice_to_devs = mock_fault_injection.create_mock_devices(
        num_slices=num_slices, devices_per_slice=devices_per_slice
    )
    elastic_mgr = manager.Manager(devices=[])
    elastic_mgr.slice_to_devices = slice_to_devs
    elastic_mgr.all_slice_indices = frozenset(slice_to_devs.keys())
    elastic_mgr.active_slice_indices = set(range(num_slices))
    return elastic_mgr, slice_to_devs

  @parameterized.named_parameters(
      (
          "step_snapshot_replica_resize",
          elastic_workload.ElasticityMode.STEP_SNAPSHOT_REPLICA_RESIZE,
          True,
      ),
      (
          "step_snapshot_pause_resume",
          elastic_workload.ElasticityMode.STEP_SNAPSHOT_PAUSE_RESUME,
          False,
      ),
  )
  def test_elasticity_modes_properties(
      self,
      mode: elastic_workload.ElasticityMode,
      expected_replica_resize: bool,
  ):
    self.assertIsInstance(mode, elastic_workload.ElasticityMode)
    self.assertEqual(mode.is_replica_resize, expected_replica_resize)
    self.assertEqual(
        {m.value for m in elastic_workload.ElasticityMode},
        {"step_snapshot_pause_resume", "step_snapshot_replica_resize"},
    )
    for removed in (
        "step_checkpoint_replica_resize",
        "run_checkpoint_replica_resize",
    ):
      with self.assertRaises(ValueError):
        elastic_workload.ElasticityMode(removed)

  def test_simulated_fault_injection_recovery(self):
    elastic_mgr, slice_to_devs = self._create_mock_elastic_manager(
        num_slices=2, devices_per_slice=1
    )
    workload = elastic_workload.ElasticWorkload(
        elastic_manager=elastic_mgr,
        config=simple_model.DEFAULT_CONFIG,
        mode=elastic_workload.ElasticityMode.STEP_SNAPSHOT_REPLICA_RESIZE,
        total_steps=10,
        checkpoint_interval=5,
        snapshot_interval=2,
        checkpoint_dir=self.temp_dir.name,
        simulator=self.simulator,
        simulate_fault_step=3,
        simulate_reconnect_step=6,
        simulate_slice_index=1,
    )

    self.assertEqual(workload.simulate_fault_step, 3)
    self.assertEqual(workload.simulate_reconnect_step, 6)
    self.assertEqual(workload.simulate_slice_index, 1)

    self.simulator.mark_slice_lost(1, slice_to_devs)
    dev_slice1 = slice_to_devs[1][0]
    self.assertIn(dev_slice1.id, self.simulator._lost_devices)

    self.simulator.mark_slice_connected(1, slice_to_devs)
    self.assertNotIn(dev_slice1.id, self.simulator._lost_devices)

  def test_workload_run_with_retry_and_simulator(self):
    elastic_mgr, _ = self._create_mock_elastic_manager(
        num_slices=2, devices_per_slice=1
    )
    workload = elastic_workload.ElasticWorkload(
        elastic_manager=elastic_mgr,
        config=simple_model.DEFAULT_CONFIG,
        mode=elastic_workload.ElasticityMode.STEP_SNAPSHOT_REPLICA_RESIZE,
        total_steps=4,
        checkpoint_interval=2,
        snapshot_interval=1,
        checkpoint_dir=self.temp_dir.name,
        simulator=self.simulator,
        simulate_fault_step=2,
        simulate_slice_index=1,
    )

    step_calls = []

    def mock_step():
      current_step = workload.step
      step_calls.append(current_step)
      if current_step == 2 and len(step_calls) == 2:
        raise jax.errors.JaxRuntimeError(
            "DATA_LOSS: Simulated device loss error accessing array on lost "
            "device"
        )
      workload.step += 1

    workload.initialize = mock.MagicMock()
    workload.run_one_step = mock_step
    workload.pre_callback_restore_and_reconfigure = mock.MagicMock()

    with mock.patch(
        "pathwaysutils.elastic.elastic.wait_for_slices",
        return_value={0, 1},
    ):
      workload.run()

    self.assertEqual(workload.step, 5)
    self.assertIn(2, step_calls)
    self.assertEqual(step_calls.count(2), 2)
    self.assertTrue(workload.pre_callback_restore_and_reconfigure.called)

  def test_workload_control_handler(self):
    elastic_mgr, slice_to_devs = self._create_mock_elastic_manager(
        num_slices=2, devices_per_slice=1
    )
    workload = elastic_workload.ElasticWorkload(
        elastic_manager=elastic_mgr,
        config=simple_model.DEFAULT_CONFIG,
        mode=elastic_workload.ElasticityMode.STEP_SNAPSHOT_REPLICA_RESIZE,
        total_steps=10,
        checkpoint_dir=self.temp_dir.name,
        simulator=self.simulator,
        wait_for_start_signal=True,
    )

    with mock.patch.object(
        elastic_workload.WorkloadControlHandler, "__init__", return_value=None
    ):
      handler = elastic_workload.WorkloadControlHandler(workload)
      handler.workload = workload
      handler.wfile = mock.MagicMock()
      handler.send_response = mock.MagicMock()
      handler.send_header = mock.MagicMock()
      handler.end_headers = mock.MagicMock()

      handler.path = "/status"
      handler.do_GET()
      handler.send_response.assert_called_with(200)
      status_payload = json.loads(
          handler.wfile.write.call_args[0][0].decode("utf-8")
      )
      self.assertEqual(status_payload["step"], 1)
      self.assertEqual(status_payload["total_steps"], 10)
      self.assertEqual(status_payload["mode"], "step_snapshot_replica_resize")
      self.assertFalse(status_payload["running"])

      handler.path = "/disrupt?slice=1"
      handler.do_GET()
      self.assertEqual(workload.pending_disrupt_slice, 1)
      dev_slice1 = slice_to_devs[1][0]
      self.assertIn(dev_slice1.id, self.simulator._lost_devices)

      handler.path = "/reconnect?slice=1"
      handler.do_GET()
      self.assertEqual(workload.pending_reconnect_slice, 1)
      self.assertNotIn(dev_slice1.id, self.simulator._lost_devices)

      handler.path = "/pause"
      handler.do_GET()
      self.assertFalse(workload.start_event.is_set())

      handler.path = "/start"
      handler.do_GET()
      self.assertTrue(workload.start_event.is_set())

      handler.path = "/reset"
      handler.do_GET()
      self.assertTrue(workload.reset_requested)
      self.assertFalse(workload.start_event.is_set())

      handler.path = "/unknown"
      handler.do_GET()
      handler.send_response.assert_called_with(404)

  def test_precompile_and_step_on_cpu_devices(self):
    self.simulator.stop()
    cpu_devs = jax.devices()[:2]
    self.assertLen(cpu_devs, 2)

    d0 = cpu_devs[0]
    d1 = cpu_devs[1]
    elastic_mgr = manager.Manager(devices=[])
    elastic_mgr.slice_to_devices = {0: [d0], 1: [d1]}
    elastic_mgr.all_slice_indices = frozenset({0, 1})
    elastic_mgr.active_slice_indices = {0, 1}

    tiny_cfg = simple_model.ModelConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        head_dim=8,
        num_layers=1,
        per_device_batch_size=1,
        seq_len=4,
    )

    with mock.patch.object(
        elastic_workload.ElasticWorkload,
        "get_slice_devices",
        side_effect=lambda idxs: [
            elastic_mgr.slice_to_devices[i][0] for i in sorted(idxs)
        ],
    ):
      workload = elastic_workload.ElasticWorkload(
          elastic_manager=elastic_mgr,
          config=tiny_cfg,
          mode=elastic_workload.ElasticityMode.STEP_SNAPSHOT_REPLICA_RESIZE,
          total_steps=2,
          checkpoint_interval=1,
          snapshot_interval=1,
          checkpoint_dir=self.temp_dir.name,
          simulator=None,
      )
      workload.initialize()
      self.assertTrue(workload.is_initialized)
      self.assertEqual(workload.current_slice_count, 2)
      self.assertIn(frozenset({0, 1}), workload.compiled_slices)
      self.assertIn(frozenset({0}), workload.compiled_slices)
      self.assertIn(frozenset({1}), workload.compiled_slices)

      with mock.patch("time.sleep", return_value=None):
        workload.run_one_step()
      workload.checkpoint_manager.wait_until_finished()
      self.assertEqual(workload.checkpoint_manager.latest_step(), 1)
      self.assertEqual(workload.step, 2)
      self.assertIsNotNone(workload.prev_loss)

      elastic_mgr.active_slice_indices = {0}
      workload.pre_callback_restore_and_reconfigure()
      self.assertEqual(workload.current_slice_count, 1)
      self.assertEqual(workload.step, 2)
      workload.close()

      # Verify cold start restore from CheckpointManager seeds Snapshotter.
      elastic_mgr.active_slice_indices = {0, 1}
      workload_resumed = elastic_workload.ElasticWorkload(
          elastic_manager=elastic_mgr,
          config=tiny_cfg,
          mode=elastic_workload.ElasticityMode.STEP_SNAPSHOT_REPLICA_RESIZE,
          total_steps=4,
          checkpoint_interval=10,
          snapshot_interval=5,
          checkpoint_dir=self.temp_dir.name,
          simulator=None,
      )
      workload_resumed.initialize()
      self.assertEqual(workload_resumed.step, 2)
      elastic_mgr.active_slice_indices = {0}
      workload_resumed.pre_callback_restore_and_reconfigure()
      self.assertEqual(workload_resumed.current_slice_count, 1)
      self.assertEqual(workload_resumed.step, 2)
      workload_resumed.close()

  def test_default_checkpoint_interval_is_infrequent(self):
    elastic_mgr = mock.MagicMock(spec=manager.Manager)
    workload = elastic_workload.ElasticWorkload(
        elastic_manager=elastic_mgr,
        checkpoint_dir=self.temp_dir.name,
    )
    self.assertEqual(workload.checkpoint_interval, 200)
    self.assertEqual(workload.snapshot_interval, 2)
    self.assertEqual(elastic_workload._CHECKPOINT_INTERVAL.default, 200)
    workload.close()

  def test_bridge_yaml_inference(self):
    cfg = gke_demo_bridge._merge_config_dict(
        gke_demo_bridge.default_config(),
        {
            "jobset_name": "custom-elastic-job",
            "namespace": "ml-team",
            "num_slices": 3,
            "tpu_type": "tpuv5e",
            "topology": "4x8",
            "total_steps": 750,
            "elasticity_mode": "step_snapshot_pause_resume",
            "container_image": "us-docker.pkg.dev/demo/repo/image:v1",
            "gcs_scratch_location": "gs://demo-bucket/scratch",
        },
    )
    ok, err, bundle = gke_demo_bridge._build_yaml_bundle(cfg)
    self.assertTrue(ok, msg=err)
    yaml_file = f"{self.temp_dir.name}/rendered.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
      f.write(bundle)
    inferred = gke_demo_bridge._infer_config_from_yaml(yaml_file)
    self.assertEqual(inferred["jobset_name"], "custom-elastic-job")
    self.assertEqual(inferred["namespace"], "ml-team")
    self.assertEqual(inferred["num_slices"], 3)
    self.assertEqual(inferred["tpu_type"], "tpuv5e")
    self.assertEqual(inferred["topology"], "4x8")
    self.assertEqual(inferred["total_steps"], 750)
    self.assertEqual(inferred["elasticity_mode"], "step_snapshot_pause_resume")


if __name__ == "__main__":
  absltest.main()
