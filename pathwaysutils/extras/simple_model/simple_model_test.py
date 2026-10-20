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
"""Tests for the simple Transformer model package using Flax NNX, Grain, and Snapshotter."""

import os
import tempfile
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax
from jax import numpy as jnp
import numpy as np
from pathwaysutils.elastic import manager
from pathwaysutils.extras.simple_model import compile as compile_lib
from pathwaysutils.extras.simple_model import config as config_lib
from pathwaysutils.extras.simple_model import data as data_lib
from pathwaysutils.extras.simple_model import model as model_lib
from pathwaysutils.extras.simple_model import workload as workload_lib


def setUpModule():
  jax.config.update("jax_num_cpu_devices", 2)


class SimpleModelTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.config = config_lib.ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        head_dim=8,
        num_layers=2,
        per_device_batch_size=2,
        seq_len=8,
    )
    self.temp_dir = tempfile.TemporaryDirectory()

  def tearDown(self):
    self.temp_dir.cleanup()
    super().tearDown()

  def test_config_dimensions_and_mesh(self):
    self.assertEqual(config_lib.parse_topology("4x8"), 32)
    self.assertEqual(config_lib.parse_topology("4x4x4"), 64)
    self.assertEqual(config_lib.parse_topology("invalid"), 1)

    self.assertIn("default", config_lib.CONFIGS)
    self.assertIn("test", config_lib.CONFIGS)
    self.assertIn("small", config_lib.CONFIGS)
    self.assertIn("large", config_lib.CONFIGS)

    cfg = config_lib.DEFAULT_CONFIG
    self.assertEqual(cfg.hidden_size % cfg.num_heads, 0)
    self.assertEqual(cfg.intermediate_size % cfg.num_heads, 0)

  def test_compute_mesh_shape(self):
    # Single slice with 8 chips
    dp, mp = compile_lib.compute_mesh_shape(8, devices_per_slice=8)
    self.assertEqual((dp, mp), (1, 8))

    # Default devices_per_slice=0 falls back to num_devices
    dp, mp = compile_lib.compute_mesh_shape(8, devices_per_slice=0)
    self.assertEqual((dp, mp), (1, 8))

    # Two slices of 8 chips (16 total) -> gcd(16, 8) = 8 -> dp=2, mp=8
    dp, mp = compile_lib.compute_mesh_shape(16, devices_per_slice=8)
    self.assertEqual((dp, mp), (2, 8))

    # Single device
    dp, mp = compile_lib.compute_mesh_shape(1, devices_per_slice=1)
    self.assertEqual((dp, mp), (1, 1))

  def test_geglu_activation(self):
    x = jnp.array([[0.0, 2.0], [1.0, 3.0]], dtype=jnp.float32)
    output = model_lib.geglu(x)
    self.assertEqual(output.shape, (2, 1))
    expected = jax.nn.gelu(x[:, :1]) * x[:, 1:]
    np.testing.assert_allclose(output, expected, rtol=1e-6)
    self.assertEqual(float(output[0, 0]), 0.0)

  def test_model_forward_pass(self):
    rngs = nnx.Rngs(0)
    model = model_lib.SimpleModel(self.config, rngs=rngs)
    for block in model.layers:
      for linear in (
          block.attn_q,
          block.attn_k,
          block.attn_v,
          block.attn_o,
          block.mlp_gate_up,
          block.mlp_down,
      ):
        self.assertFalse(linear.use_bias)
        self.assertIsNone(linear.bias)

    tokens = jnp.zeros((2, self.config.seq_len), dtype=jnp.int32)
    logits = model(tokens)

    expected_shape = (2, self.config.seq_len, self.config.vocab_size)
    self.assertEqual(logits.shape, expected_shape)
    self.assertTrue(jnp.all(jnp.isfinite(logits)))

    model.output_scale.value = jnp.array(2.0, dtype=jnp.float32)
    scaled_logits = model(tokens)
    np.testing.assert_allclose(scaled_logits, logits * 2.0, rtol=1e-5)

  def test_synthetic_token_source(self):
    source = data_lib.SyntheticTokenSource(
        vocab_size=self.config.vocab_size,
        seq_len=self.config.seq_len,
        num_samples=100,
    )
    self.assertLen(source, 100)

    sample_0 = source[0]
    sample_0_repeat = source[0]
    sample_1 = source[1]

    self.assertEqual(sample_0.shape, (self.config.seq_len,))
    self.assertEqual(sample_0.dtype, np.int32)
    np.testing.assert_array_equal(sample_0, sample_0_repeat)
    self.assertFalse(np.array_equal(sample_0, sample_1))

  def test_data_iterator_batching_and_checkpointing(self):
    dataset = data_lib.create_token_dataset(self.config)
    batch_size = 4
    iterator = data_lib.create_data_iterator(
        dataset, global_batch_size=batch_size
    )

    batch_1 = next(iterator)
    self.assertEqual(batch_1.shape, (batch_size, self.config.seq_len))

    checkpoint_state = iterator.get_state()
    batch_2 = next(iterator)
    iterator.close()

    restored_iterator = data_lib.create_data_iterator(
        dataset,
        global_batch_size=batch_size,
        restore_state=checkpoint_state,
    )
    restored_batch_2 = next(restored_iterator)
    restored_iterator.close()

    np.testing.assert_array_equal(batch_2, restored_batch_2)

  def test_compile_and_training_step(self):
    devices = jax.devices()[:2]
    self.assertLen(devices, 2)

    with self.assertRaises(ValueError):
      compile_lib.compile_training_step(
          self.config, devices, mesh_shape=(3, 1)
      )

    # Test mp_size=2 sharding (1x2 mesh)
    graphdef, artifacts = compile_lib.compile_training_step(
        self.config,
        devices,
        devices_per_slice=2,
        learning_rate=0.01,
        mesh_shape=(1, 2),
    )
    self.assertIsNotNone(graphdef)
    self.assertEqual(
        artifacts.global_batch_size, self.config.per_device_batch_size * 2
    )
    self.assertEqual(
        artifacts.param_shardings.embedder.embedding.value.spec,
        jax.sharding.PartitionSpec(None, "mp"),
    )
    self.assertEqual(
        artifacts.param_shardings.layers[0].rmsnorm_1.scale.value.spec,
        jax.sharding.PartitionSpec(),
    )
    self.assertEqual(
        artifacts.param_shardings.output_scale.value.spec,
        jax.sharding.PartitionSpec(),
    )

    dummy_model = model_lib.SimpleModel(self.config, rngs=nnx.Rngs(42))
    _, raw_params = nnx.split(dummy_model)
    params = jax.device_put(raw_params, artifacts.param_shardings)

    tokens = jax.device_put(
        jnp.ones(
            (artifacts.global_batch_size, self.config.seq_len),
            dtype=jnp.int32,
        ),
        artifacts.data_sharding,
    )

    loss, new_params = artifacts.compiled_step_fn(params, tokens)
    self.assertTrue(jnp.isfinite(loss))
    self.assertGreater(float(loss), 0.0)
    self.assertFalse(
        jnp.array_equal(
            params.embedder.embedding.value,
            new_params.embedder.embedding.value,
        )
    )

    # Test mp_size=1 sharding (2x1 mesh) replicates all parameters
    _, dp_artifacts = compile_lib.compile_training_step(
        self.config, devices, mesh_shape=(2, 1)
    )
    self.assertEqual(
        dp_artifacts.param_shardings.embedder.embedding.value.spec,
        jax.sharding.PartitionSpec(),
    )

  def test_workload_defaults_and_snapshotter_init(self):
    workload = workload_lib.Workload(config=self.config)
    self.assertEqual(workload.devices, list(jax.devices()))
    self.assertEqual(workload.default_device, jax.devices()[0])
    self.assertEqual(workload.devices_per_slice, 0)
    self.assertIsNone(workload.snapshotter)

    with self.assertRaises(RuntimeError):
      workload.run_step()

    # Early return when snapshotter is None
    workload.restore_from_snapshot()

    with mock.patch.object(
        workload_lib.ocp.v1.training.pathways, "Snapshotter"
    ) as mock_snap_cls:
      snap_workload = workload_lib.Workload(
          config=self.config,
          devices=jax.devices()[:1],
          snapshot_interval=1,
      )
      mock_snap_cls.assert_called_once_with()
      self.assertEqual(snap_workload.snapshotter, mock_snap_cls.return_value)

  def test_workload_run_and_checkpoint(self):
    devices = jax.devices()[:1]
    with mock.patch.object(
        workload_lib.ocp,
        "CheckpointManager",
        wraps=workload_lib.ocp.CheckpointManager,
    ) as mock_ckpt_cls:
      workload = workload_lib.Workload(
          config=self.config,
          devices=devices,
          devices_per_slice=1,
          learning_rate=0.001,
          total_steps=3,
          checkpoint_interval=2,
          checkpoint_dir=self.temp_dir.name,
      )
      mock_ckpt_cls.assert_called_once()
      self.assertTrue(
          mock_ckpt_cls.call_args.kwargs["options"].enable_async_checkpointing
      )

    try:
      losses = workload.run()
      self.assertLen(losses, 3)
      for loss in losses:
        self.assertTrue(jnp.isfinite(loss))
        self.assertGreater(loss, 0.0)

      ckpt_step_2_path = os.path.join(self.temp_dir.name, "2")
      self.assertTrue(os.path.exists(ckpt_step_2_path))
    finally:
      workload.close()

  def test_snapshotter_integration(self):
    devices = jax.devices()[:2]
    mock_snapshotter = mock.MagicMock()

    workload = workload_lib.Workload(
        config=self.config,
        devices=devices,
        devices_per_slice=2,
        learning_rate=0.001,
        total_steps=2,
        snapshot_interval=1,
        snapshotter=mock_snapshotter,
    )

    try:
      workload.run()
      self.assertEqual(mock_snapshotter.save.call_count, 2)
      self.assertEqual(mock_snapshotter.join.call_count, 1)

      # Verify sharded params were promoted to replicated sharding on save
      self.assertEqual(
          workload.params.embedder.embedding.value.sharding.spec,
          jax.sharding.PartitionSpec(None, "mp"),
      )
      saved_step, saved_payload = mock_snapshotter.save.call_args[0]
      self.assertEqual(saved_step, 2)
      assert workload.artifacts is not None
      for leaf in jax.tree.leaves(saved_payload["params"]):
        self.assertEqual(
            leaf.sharding, workload.artifacts.replicated_sharding
        )

      fake_params = workload.params
      assert workload.data_iterator is not None
      real_grain_st = workload.data_iterator.get_state()
      mock_snapshotter.load.return_value = {
          "params": fake_params,
          "grain_state": real_grain_st,
      }
      workload.restore_from_snapshot()
      mock_snapshotter.load.assert_called_once()
      self.assertEqual(workload.data_iterator.get_state(), real_grain_st)
    finally:
      workload.close()

  def test_elastic_manager_default_device(self):
    mock_mgr = mock.MagicMock(spec=manager.Manager)
    mock_dev0 = mock.MagicMock(spec=jax.Device)
    mock_dev1 = mock.MagicMock(spec=jax.Device)
    mock_mgr.default_device = mock_dev1
    mock_mgr.active_slice_indices = frozenset([1])
    mock_mgr.slice_to_devices = {0: [mock_dev0], 1: [mock_dev1]}

    workload = workload_lib.Workload(
        config=self.config,
        elastic_manager=mock_mgr,
    )
    self.assertEqual(workload.default_device, mock_dev1)
    self.assertEqual(workload.devices, [mock_dev1])
    self.assertEqual(workload.devices_per_slice, 1)

    # Explicit devices_per_slice is preserved when non-zero
    workload_explicit = workload_lib.Workload(
        config=self.config,
        elastic_manager=mock_mgr,
        devices_per_slice=4,
    )
    self.assertEqual(workload_explicit.devices_per_slice, 4)


if __name__ == "__main__":
  absltest.main()
