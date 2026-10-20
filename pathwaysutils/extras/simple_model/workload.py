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
"""Training workload runner with Orbax Checkpointing and Pathways in-memory Snapshotting."""

from collections.abc import Sequence
import logging
import time
from typing import Any

from flax import nnx
import grain.python as grain
import jax
from jax import numpy as jnp
import orbax.checkpoint as ocp
from pathwaysutils.elastic import manager
from pathwaysutils.extras.simple_model import compile as compile_lib
from pathwaysutils.extras.simple_model import config as config_lib
from pathwaysutils.extras.simple_model import data as data_lib
from pathwaysutils.extras.simple_model import model as model_lib

_logger = logging.getLogger(__name__)


class Workload:
  """Training workload coordinator for the simple transformer problem."""

  def __init__(
      self,
      config: config_lib.ModelConfig = config_lib.DEFAULT_CONFIG,
      elastic_manager: manager.Manager | None = None,
      devices: Sequence[jax.Device] | None = None,
      devices_per_slice: int = 0,
      learning_rate: float = 0.0005,
      total_steps: int = 20,
      snapshot_interval: int = 0,
      checkpoint_interval: int = 0,
      checkpoint_dir: str | None = None,
      snapshotter: Any | None = None,
  ):
    self.config = config
    self.elastic_manager = elastic_manager
    if devices is not None:
      self.devices = list(devices)
    elif self.elastic_manager is not None:
      self.devices = [
          d
          for s in self.elastic_manager.active_slice_indices
          for d in self.elastic_manager.slice_to_devices[s]
      ]
    else:
      self.devices = list(jax.devices())

    self.devices_per_slice = devices_per_slice
    if self.devices_per_slice == 0 and self.elastic_manager is not None:
      self.devices_per_slice = len(
          next(iter(self.elastic_manager.slice_to_devices.values()))
      )

    self.learning_rate = learning_rate
    self.total_steps = total_steps
    self.snapshot_interval = snapshot_interval
    self.checkpoint_interval = checkpoint_interval
    self.checkpoint_dir = (
        checkpoint_dir or f"/tmp/simple_model_ckpts_{int(time.time())}"
    )

    self.snapshotter = snapshotter
    if self.snapshot_interval > 0 and self.snapshotter is None:
      self.snapshotter = ocp.v1.training.pathways.Snapshotter()

    self.checkpoint_manager: ocp.CheckpointManager | None = None
    if self.checkpoint_interval > 0:
      self.checkpoint_manager = ocp.CheckpointManager(
          self.checkpoint_dir,
          options=ocp.CheckpointManagerOptions(enable_async_checkpointing=True),
      )

    self.graphdef: nnx.GraphDef | None = None
    self.artifacts: compile_lib.CompiledStepArtifacts | None = None
    self.params: Any = None
    self.step: int = 1
    self.data_dataset = data_lib.create_token_dataset(self.config)
    self.data_iterator: grain.experimental.ElasticIterator | None = None

  @property
  def default_device(self) -> jax.Device:
    """The default device from the elastic manager, or the first device if none."""
    if self.elastic_manager is not None:
      return self.elastic_manager.default_device
    return self.devices[0]

  def initialize(self) -> None:
    """Precompiles step function, initializes parameters, and sets up data iterator."""
    self.graphdef, self.artifacts = compile_lib.compile_training_step(
        self.config,
        self.devices,
        devices_per_slice=self.devices_per_slice,
        learning_rate=self.learning_rate,
    )
    with jax.default_device(self.default_device):
      model = model_lib.SimpleModel(self.config, rngs=nnx.Rngs(42))
      _, raw_params = nnx.split(model)
      self.params = jax.device_put(raw_params, self.artifacts.param_shardings)

    self.data_iterator = data_lib.create_data_iterator(
        self.data_dataset, self.artifacts.global_batch_size
    )
    self.step = 1
    _logger.info(
        "Initialized Workload: %d devices (mesh %s), batch size %d,"
        " seq len %d.",
        len(self.devices),
        self.artifacts.mesh.shape,
        self.artifacts.global_batch_size,
        self.config.seq_len,
    )

  def run_step(self) -> float:
    """Runs a single training step and returns the loss scalar."""
    if self.artifacts is None or self.data_iterator is None:
      raise RuntimeError("Workload must be initialized before running steps.")

    t0 = time.time()
    batch = next(self.data_iterator)
    with jax.default_device(self.default_device):
      tokens = jax.device_put(jnp.asarray(batch), self.artifacts.data_sharding)

    loss, self.params = self.artifacts.compiled_step_fn(self.params, tokens)
    jax.block_until_ready(loss)
    duration = time.time() - t0
    loss_val = float(loss)

    _logger.info(
        "Step %d/%d | Loss: %.4f | Step Time: %.4fs",
        self.step,
        self.total_steps,
        loss_val,
        duration,
    )

    if (
        self.snapshotter is not None
        and self.snapshot_interval > 0
        and self.step % self.snapshot_interval == 0
    ):
      promoted = jax.device_put(self.params, self.artifacts.replicated_sharding)
      grain_st = self.data_iterator.get_state()
      self.snapshotter.save(
          self.step, {"params": promoted, "grain_state": grain_st}
      )
      _logger.info("Saved in-memory snapshot at step %d.", self.step)

    if (
        self.checkpoint_manager is not None
        and self.checkpoint_interval > 0
        and self.step % self.checkpoint_interval == 0
    ):
      grain_st = self.data_iterator.get_state()
      self.checkpoint_manager.save(
          self.step,
          args=ocp.args.StandardSave(
              {"params": self.params, "step": self.step, "grain_state": grain_st}
          ),
      )
      _logger.info(
          "Saved checkpoint at step %d to %s.", self.step, self.checkpoint_dir
      )

    self.step += 1
    return loss_val

  def restore_from_snapshot(self) -> None:
    """Restores parameters and iterator state from the latest in-memory snapshot."""
    if not self.snapshotter or not self.artifacts or not self.data_iterator:
      return
    self.snapshotter.join()
    payload = self.snapshotter.load({
        "params": self.artifacts.abstract_params,
        "grain_state": self.data_iterator.get_state(),
    })
    self.params = payload["params"]
    if "grain_state" in payload:
      self.data_iterator.set_state(payload["grain_state"])
    _logger.info("Restored state from in-memory snapshot.")

  def run(self) -> list[float]:
    """Runs the training loop for total_steps and returns the losses recorded."""
    if self.artifacts is None:
      self.initialize()
    losses = []
    while self.step <= self.total_steps:
      loss = self.run_step()
      losses.append(loss)

    if self.snapshotter is not None:
      self.snapshotter.join()
    if self.checkpoint_manager is not None:
      self.checkpoint_manager.wait_until_finished()
    _logger.info("Training complete. Executed %d steps.", len(losses))
    return losses

  def close(self) -> None:
    """Cleans up resources and closes the data iterator and snapshotter."""
    if self.snapshotter is not None:
      self.snapshotter.join()
    if self.data_iterator is not None:
      self.data_iterator.close()
      self.data_iterator = None
