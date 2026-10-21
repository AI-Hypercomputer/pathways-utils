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
"""Elastic training workload demonstrating Pathways elasticity recovery modes."""

from collections.abc import Sequence
import enum
import functools
import http.server
import json
import logging
import threading
import time
from typing import Any
import urllib.parse

from absl import app
from absl import flags
from flax import nnx
import grain.python as grain
import jax
from jax import numpy as jnp
import orbax.checkpoint as ocp
import pathwaysutils
from pathwaysutils.elastic import elastic
from pathwaysutils.elastic import manager
from pathwaysutils.extras import simple_model
from pathwaysutils.test import mock_fault_injection

_logger = logging.getLogger(__name__)

_CONFIG_NAME = flags.DEFINE_string(
    "config_name",
    "default",
    "Model config preset (default, small, large) or topology string.",
)
_ENABLE_CONTROL_SERVER = flags.DEFINE_boolean(
    "enable_control_server", True, "Enable HTTP control server."
)
_CONTROL_PORT = flags.DEFINE_integer(
    "control_port", 8090, "Port for the HTTP control server."
)
_WAIT_FOR_START_SIGNAL = flags.DEFINE_boolean(
    "wait_for_start_signal", False, "Wait for HTTP /start before running."
)
_ELASTICITY_MODE = flags.DEFINE_string(
    "elasticity_mode",
    "step_snapshot_replica_resize",
    "Elasticity recovery mode.",
)
_TOTAL_STEPS = flags.DEFINE_integer(
    "total_steps", 1000, "Total training steps."
)
_CHECKPOINT_INTERVAL = flags.DEFINE_integer(
    "checkpoint_interval", 20, "Checkpoint step interval."
)
_SNAPSHOT_INTERVAL = flags.DEFINE_integer(
    "snapshot_interval", 2, "Snapshot step interval."
)
_CHECKPOINT_DIR = flags.DEFINE_string(
    "checkpoint_dir", "", "GCS or local directory for Orbax checkpoints."
)
_SIMULATE_FAULTS = flags.DEFINE_boolean(
    "simulate_faults", False, "Enable simulated fault injection."
)
_SIMULATE_FAULT_STEP = flags.DEFINE_integer(
    "simulate_fault_step", None, "Step for simulated slice loss."
)
_SIMULATE_RECONNECT_STEP = flags.DEFINE_integer(
    "simulate_reconnect_step", None, "Step for simulated slice reconnect."
)
_SIMULATE_SLICE_INDEX = flags.DEFINE_integer(
    "simulate_slice_index", 1, "Slice index to disconnect/reconnect."
)


class ElasticityMode(enum.Enum):
  """Supported recovery strategies for elastic training."""

  STEP_SNAPSHOT_PAUSE_RESUME = "step_snapshot_pause_resume"
  STEP_SNAPSHOT_REPLICA_RESIZE = "step_snapshot_replica_resize"

  @property
  def is_replica_resize(self) -> bool:
    return self == ElasticityMode.STEP_SNAPSHOT_REPLICA_RESIZE


class WorkloadControlHandler(http.server.BaseHTTPRequestHandler):
  """HTTP request handler for in-workload signaling and status telemetry."""

  def __init__(self, workload: Any, *args: Any, **kwargs: Any):
    self.workload = workload
    super().__init__(*args, **kwargs)

  def log_message(self, format: str, *args: Any) -> None:
    del format, args

  def _send_json(self, data: dict[str, Any], code: int = 200) -> None:
    body = json.dumps(data).encode("utf-8")
    self.send_response(code)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def do_GET(self) -> None:  # pylint: disable=invalid-name
    """Handles HTTP GET requests for status and workload control actions."""
    p = urllib.parse.urlparse(self.path)
    w, em = self.workload, self.workload.elastic_manager
    if p.path == "/status":
      loss_val = float(w.prev_loss) if w.prev_loss is not None else None
      self._send_json({
          "step": w.step,
          "total_steps": w.total_steps,
          "mode": w.mode.value,
          "active_slices": sorted(list(em.active_slice_indices)),
          "inactive_slices": sorted(list(em.inactive_slice_indices)),
          "available_inactive_slices": sorted(
              list(em.available_inactive_slices)
          ),
          "last_loss": loss_val,
          "running": w.start_event.is_set(),
          "initialized": w.is_initialized,
      })
    elif p.path in ("/start", "/pause", "/reset"):
      action = p.path.lstrip("/")
      if action == "start":
        w.start_event.set()
        _logger.warning(">>> [CONTROL] TRAINING STARTED signal received! <<<")
      elif action == "pause":
        w.start_event.clear()
        _logger.warning(">>> [CONTROL] TRAINING PAUSED signal received! <<<")
      elif action == "reset":
        if w.start_event.is_set() or w.step > 1:
          w.reset_requested = True
        w.start_event.clear()
        _logger.warning(">>> [CONTROL] RESET signal received! <<<")
      self._send_json(
          {"status": "ok", "action": action, "running": w.start_event.is_set()}
      )
    elif p.path in ("/disrupt", "/reconnect"):
      action = p.path.lstrip("/")
      s_idx = int(urllib.parse.parse_qs(p.query).get("slice", ["1"])[0])
      if action == "disrupt":
        w.pending_disrupt_slice = s_idx
        if w.simulator and not w.start_event.is_set():
          w.simulator.mark_slice_lost(s_idx, em.slice_to_devices)
      else:
        w.pending_reconnect_slice = s_idx
        if w.simulator and not w.start_event.is_set():
          w.simulator.mark_slice_connected(s_idx, em.slice_to_devices)
      self._send_json({"status": "ok", "action": action, "slice": s_idx})
    else:
      self._send_json({"error": "not found"}, code=404)


class ElasticWorkload:
  """Elastic training workload orchestrating execution, pre-compilation, and recovery."""

  def __init__(
      self,
      elastic_manager: manager.Manager,
      config: simple_model.ModelConfig = simple_model.DEFAULT_CONFIG,
      mode: ElasticityMode = ElasticityMode.STEP_SNAPSHOT_REPLICA_RESIZE,
      total_steps: int = 1000,
      checkpoint_interval: int = 20,
      snapshot_interval: int = 2,
      checkpoint_dir: str | None = None,
      simulator: mock_fault_injection.DeviceLossSimulator | None = None,
      simulate_fault_step: int | None = None,
      simulate_reconnect_step: int | None = None,
      simulate_slice_index: int = 1,
      enable_control_server: bool = False,
      control_port: int = 8090,
      wait_for_start_signal: bool = False,
  ):
    self.elastic_manager = elastic_manager
    self.config = config
    self.mode = mode
    self.total_steps = total_steps
    self.checkpoint_interval = checkpoint_interval
    self.snapshot_interval = snapshot_interval
    self.simulator = simulator
    self.simulate_slice_index = simulate_slice_index
    self.simulate_fault_step = simulate_fault_step
    self.simulate_reconnect_step = simulate_reconnect_step
    self.pending_disrupt_slice: int | None = None
    self.pending_reconnect_slice: int | None = None
    self.is_initialized = False
    self.reset_requested = False
    self.start_event = threading.Event()
    if not wait_for_start_signal:
      self.start_event.set()
    self.control_server: http.server.HTTPServer | None = None
    self.control_thread: threading.Thread | None = None
    if enable_control_server:
      self.control_server = http.server.HTTPServer(
          ("", control_port), functools.partial(WorkloadControlHandler, self)
      )
      self.control_thread = threading.Thread(
          target=self.control_server.serve_forever, daemon=True
      )
      self.control_thread.start()
      _logger.info("HTTP control server started on port %d.", control_port)

    self.checkpoint_dir = (
        checkpoint_dir or f"/tmp/checkpoints_{int(time.time())}"
    )
    self.checkpoint_manager = ocp.CheckpointManager(
        self.checkpoint_dir,
        options=ocp.CheckpointManagerOptions(enable_async_checkpointing=True),
    )
    self.snapshotter = ocp.v1.training.pathways.Snapshotter()
    self.graphdef: nnx.GraphDef | None = None
    self.params: Any = None
    self.step: int = 1
    self.prev_loss: jax.Array | None = None
    self.prev_start_time: float | None = None
    self.compiled_slices: dict[
        frozenset[int], simple_model.CompiledStepArtifacts
    ] = {}
    self.current_slice_count: int = 0
    self.current_artifacts: simple_model.CompiledStepArtifacts | None = None
    self.raw_dataset = simple_model.create_token_dataset(self.config)
    self.data_iterator: grain.experimental.ElasticIterator | None = None
    self.last_saved_grain_state: dict[str, Any] | None = None

  def close(self) -> None:
    """Stops the HTTP control server if running."""
    if self.control_server is not None:
      self.control_server.shutdown()
      self.control_server.server_close()
      self.control_server = None
    if self.control_thread is not None:
      self.control_thread.join(timeout=2)
      self.control_thread = None

  def _promote_to_replicated(
      self, state: Any, repl: jax.sharding.NamedSharding
  ) -> Any:
    def _promote(x):
      if isinstance(x, jax.Array) and (
          not isinstance(x.sharding, jax.sharding.NamedSharding)
          or x.sharding.mesh != repl.mesh
      ):
        return jax.device_put(x, repl)
      return x

    return jax.tree.map(_promote, state)

  def precompile_step_functions(self) -> None:
    """Pre-compiles sharded step executables for active slice subsets."""
    total_slices = self.elastic_manager.total_slice_count
    target_subsets = [frozenset(range(total_slices))]
    if self.mode.is_replica_resize:
      all_idx = frozenset(range(total_slices))
      target_subsets.extend(frozenset({s}) for s in range(total_slices))
      target_subsets.extend(
          all_idx - {s} for s in range(total_slices) if all_idx - {s}
      )
      target_subsets = list(dict.fromkeys(target_subsets))

    for active_idx in target_subsets:
      num_slices = len(active_idx)
      sorted_devs = self.get_slice_devices(active_idx)
      devices_per_slice = len(sorted_devs) // num_slices
      self.graphdef, artifacts = simple_model.compile_training_step(
          self.config,
          sorted_devs,
          devices_per_slice=devices_per_slice,
          mesh_shape=(num_slices, devices_per_slice),
      )
      self.compiled_slices[active_idx] = artifacts

  def get_slice_devices(
      self, slice_indices: set[int] | frozenset[int]
  ) -> list[jax.Device]:
    """Returns devices for the given slice indices ordered by slice and coords."""
    devices: list[jax.Device] = []
    for idx in slice_indices:
      devices.extend(self.elastic_manager.slice_to_devices[idx])
    return sorted(devices, key=lambda d: (d.slice_index, d.coords))

  def reconfigure_mesh_and_executables(self) -> None:
    """Switches current artifacts to match active slices."""
    active = frozenset(self.elastic_manager.active_slice_indices)
    self.current_slice_count = len(active)
    self.current_artifacts = self.compiled_slices[active]
    _logger.info(
        "Stage 2: Activating pre-compiled %d-slice executable %s (batch=%d).",
        self.current_slice_count,
        set(active),
        self.current_artifacts.global_batch_size,
    )

  def init_or_reconfigure_data_iterator(
      self, restore_state: dict[str, Any] | None = None
  ) -> None:
    """Initializes or reconfigures the Grain iterator for the active batch size."""
    if self.data_iterator is not None:
      self.data_iterator.close()
    assert self.current_artifacts is not None
    self.data_iterator = simple_model.create_data_iterator(
        self.raw_dataset,
        self.current_artifacts.global_batch_size,
        restore_state=restore_state,
    )
    if restore_state is not None:
      self.last_saved_grain_state = restore_state

  def _reset_model_params(self, save_snapshot: bool = True) -> None:
    """Re-initializes model parameters and Grain iterator state."""
    artifacts = self.current_artifacts
    assert artifacts is not None
    with jax.default_device(self.elastic_manager.default_device):
      _, self.params = nnx.split(
          simple_model.SimpleModel(self.config, rngs=nnx.Rngs(42))
      )
      self.params = jax.device_put(self.params, artifacts.param_shardings)
      self.step, self.prev_loss, self.prev_start_time = 1, None, None
      self.init_or_reconfigure_data_iterator()
      self.last_saved_grain_state = (
          self.data_iterator.get_state() if self.data_iterator else None
      )
      if save_snapshot:
        promoted = self._promote_to_replicated(
            self.params, artifacts.replicated_sharding
        )
        self.snapshotter.save(
            0, {"params": promoted, "grain_state": self.last_saved_grain_state}
        )
        self.snapshotter.join()

  def initialize(self) -> None:
    """Pre-compiles executables and initializes or restores training state."""
    self.precompile_step_functions()
    if (
        self.elastic_manager.active_slice_count
        < self.elastic_manager.total_slice_count
    ):
      self.elastic_manager.active_slice_indices = elastic.wait_for_slices(
          slice_count=self.elastic_manager.total_slice_count,
          slice_to_devices=self.elastic_manager.slice_to_devices,
          poll_interval=5,
      )
    self.reconfigure_mesh_and_executables()
    artifacts = self.current_artifacts
    assert artifacts is not None

    latest = self.checkpoint_manager.latest_step()
    if latest is not None:
      _logger.info("Found GCS checkpoint at step %d, restoring...", latest)
      self.init_or_reconfigure_data_iterator()
      assert self.data_iterator is not None
      abstract_state = {
          "params": artifacts.abstract_params,
          "step": 0,
          "grain_state": self.data_iterator.get_state(),
      }
      restored = self.checkpoint_manager.restore(
          latest, args=ocp.args.StandardRestore(abstract_state)
      )
      self.params = restored["params"]
      self.step = latest + 1
      self.last_saved_grain_state = restored["grain_state"]
      self.init_or_reconfigure_data_iterator(
          restore_state=self.last_saved_grain_state
      )
      promoted = self._promote_to_replicated(
          self.params, artifacts.replicated_sharding
      )
      self.snapshotter.save(
          latest,
          {"params": promoted, "grain_state": self.last_saved_grain_state},
      )
      self.snapshotter.join()
    else:
      _logger.info("Initializing fresh Flax NNX model parameters.")
      self._reset_model_params(save_snapshot=True)

    self.save_backup()
    self.is_initialized = True
    _logger.warning(
        ">>> [WORKLOAD READY] Pre-compilation finished (%d slices). <<<",
        self.current_slice_count,
    )

  def save_backup(self) -> None:
    """Saves periodic in-memory snapshots and GCS checkpoints."""
    artifacts = self.current_artifacts
    if artifacts is None:
      return
    grain_st = self.data_iterator.get_state() if self.data_iterator else None

    if self.snapshot_interval > 0 and self.step % self.snapshot_interval == 0:
      promoted = self._promote_to_replicated(
          self.params, artifacts.replicated_sharding
      )
      self.snapshotter.save(
          self.step, {"params": promoted, "grain_state": grain_st}
      )
      if self.simulator is not None:
        self.snapshotter.join()
      self.last_saved_grain_state = grain_st

    if (
        self.checkpoint_interval > 0
        and self.step % self.checkpoint_interval == 0
    ):
      self.checkpoint_manager.save(
          self.step,
          args=ocp.args.StandardSave({
              "params": self.params,
              "step": self.step,
              "grain_state": grain_st,
          }),
      )
      self.last_saved_grain_state = grain_st
      _logger.info("Saved GCS checkpoint at step %d.", self.step)

  def restore_state(self) -> None:
    """Restores model parameters and Grain iterator state after an elastic event."""
    artifacts = self.current_artifacts
    assert artifacts is not None
    _logger.info(
        "Stage 3: Restoring model parameters and Grain iterator state..."
    )

    self.snapshotter.join()
    payload = self.snapshotter.load(
        {
            "params": artifacts.abstract_params,
            "grain_state": self.last_saved_grain_state,
        },
        reset_snapshot_state=True,
    )
    self.params = payload["params"]
    self.last_saved_grain_state = payload["grain_state"]
    assert self.snapshotter.latest is not None
    self.step = self.snapshotter.latest.step + 1

    self.params = jax.device_put(self.params, artifacts.param_shardings)
    jax.block_until_ready(self.params)
    self.init_or_reconfigure_data_iterator(
        restore_state=self.last_saved_grain_state
    )
    _logger.info("Restore complete. Resuming training at step %d.", self.step)

  def pre_callback_restore_and_reconfigure(self) -> None:
    """Reconfigures the mesh and restores state before an elastic retry."""
    if not self.is_initialized:
      return
    _logger.warning(
        ">>> [ELASTIC RECOVERY TRIGGERED] Active slices: %s <<<",
        self.elastic_manager.active_slice_indices,
    )
    _logger.info(
        "Stage 1: Active slices detected: %s",
        frozenset(self.elastic_manager.active_slice_indices),
    )
    if self.mode.is_replica_resize:
      self.reconfigure_mesh_and_executables()
    self.restore_state()
    self.prev_loss, self.prev_start_time = None, None
    _logger.warning(
        ">>> [ELASTIC RECOVERY COMPLETE] Resuming on %d slice(s). <<<",
        self.current_slice_count,
    )

  def _handle_injected_faults(self) -> None:
    """Applies pending fault-injection or reconnect actions for the step."""
    em, sim = self.elastic_manager, self.simulator
    if not sim:
      return
    disrupt_idx = self.pending_disrupt_slice
    if disrupt_idx is None and self.step == self.simulate_fault_step:
      disrupt_idx = self.simulate_slice_index
    self.pending_disrupt_slice = None
    if disrupt_idx is not None:
      _logger.warning(
          ">>> [FAULT INJECTION] INTERRUPTING SLICE %d AT STEP %d <<<",
          disrupt_idx,
          self.step,
      )
      sim.mark_slice_lost(disrupt_idx, em.slice_to_devices)
      for leaf in jax.tree_util.tree_leaves(self.params):
        sim.check_and_raise_if_corrupted(leaf)

    reconnect_idx = self.pending_reconnect_slice
    if reconnect_idx is None and self.step == self.simulate_reconnect_step:
      reconnect_idx = self.simulate_slice_index
    self.pending_reconnect_slice = None
    if reconnect_idx is not None:
      _logger.warning(
          ">>> [SCALE UP] Reconnecting slice %d at step %d <<<",
          reconnect_idx,
          self.step,
      )
      sim.mark_slice_connected(reconnect_idx, em.slice_to_devices)
      em._check_inactive_slices()  # pylint: disable=protected-access

  def run_one_step(self) -> None:
    """Runs one training step, saves backups, and checks for scale-up signals."""
    artifacts = self.current_artifacts
    if artifacts is None or self.data_iterator is None:
      raise RuntimeError("Workload must be initialized before stepping.")

    t0 = time.time()
    with jax.default_device(self.elastic_manager.default_device):
      tokens = jax.device_put(
          jnp.asarray(next(self.data_iterator)), artifacts.data_sharding
      )

    loss, self.params = artifacts.compiled_step_fn(self.params, tokens)
    jax.block_until_ready(loss)
    time.sleep(0.5)
    self.prev_loss, self.prev_start_time = loss, t0
    _logger.info(
        "Step %d | Active Slices: %d | Loss: %.4f | Step Time: %.4fs",
        self.step,
        self.current_slice_count,
        float(loss),
        time.time() - t0,
    )

    self.save_backup()
    self._handle_injected_faults()
    if (
        self.mode.is_replica_resize
        and self.elastic_manager.available_inactive_slices
    ):
      _logger.warning(
          ">>> [SCALE UP DETECTED] Slices available: %s. Initiating resize!"
          " <<<",
          self.elastic_manager.available_inactive_slices,
      )
      raise manager.ScaleUpSignalError("Scale up requested.")
    self.step += 1

  def run(self) -> None:
    """Executes the elastic training loop until total_steps is reached."""
    _logger.info(
        "Starting Elastic Workload | Mode: %s | Config: %s | Steps: %d",
        self.mode.value,
        self.config,
        self.total_steps,
    )

    @self.elastic_manager.elastic_retry(
        minimum_slice_count=1 if self.mode.is_replica_resize else None,
        poll_interval=5,
        pre_callback=self.pre_callback_restore_and_reconfigure,
    )
    def run_workload():
      if not self.is_initialized:
        self.initialize()
      if not self.start_event.is_set():
        _logger.warning(">>> [WORKLOAD READY] Waiting for Start signal... <<<")
        self.start_event.wait()
      while self.step <= self.total_steps:
        if self.reset_requested:
          self.reset_requested = False
          if self.simulator is not None:
            for s in range(self.elastic_manager.total_slice_count):
              self.simulator.mark_slice_connected(
                  s, self.elastic_manager.slice_to_devices
              )
          self.elastic_manager.active_slice_indices = set(
              self.elastic_manager.all_slice_indices
          )
          self.reconfigure_mesh_and_executables()
          self._reset_model_params(save_snapshot=True)
          _logger.warning(
              ">>> [WORKLOAD READY] Workload reset to Step 0. Waiting for Start"
              " signal... <<<"
          )
          self.start_event.wait()
        if not self.start_event.is_set():
          _logger.warning(
              ">>> [TRAINING PAUSED] Waiting for resume signal... <<<"
          )
          self.start_event.wait()
          self.prev_start_time = time.time()
        self.run_one_step()

    run_workload()
    if self.prev_loss is not None:
      jax.block_until_ready(self.prev_loss)
    self.checkpoint_manager.wait_until_finished()
    _logger.info("Training finished successfully at step %d.", self.step - 1)


def main(argv: Sequence[str]) -> None:
  del argv
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
  )

  cfg = _CONFIG_NAME.value
  if cfg in simple_model.CONFIGS:
    model_config = simple_model.CONFIGS[cfg]
  elif ":" in cfg:
    model_config = simple_model.DEFAULT_CONFIG
  else:
    raise ValueError(
        f"Unknown config_name: {cfg}. Available:"
        f" {list(simple_model.CONFIGS.keys())}"
    )

  mode = ElasticityMode(_ELASTICITY_MODE.value)
  elastic_mgr = manager.Manager()
  enable_ctrl = _ENABLE_CONTROL_SERVER.value

  sim = None
  if _SIMULATE_FAULTS.value or enable_ctrl:
    sim = mock_fault_injection.DeviceLossSimulator()
    sim.start()

  workload = ElasticWorkload(
      elastic_manager=elastic_mgr,
      config=model_config,
      mode=mode,
      total_steps=_TOTAL_STEPS.value,
      checkpoint_interval=_CHECKPOINT_INTERVAL.value,
      snapshot_interval=_SNAPSHOT_INTERVAL.value,
      checkpoint_dir=_CHECKPOINT_DIR.value or None,
      simulator=sim,
      simulate_fault_step=(
          _SIMULATE_FAULT_STEP.value if _SIMULATE_FAULTS.value else None
      ),
      simulate_reconnect_step=(
          _SIMULATE_RECONNECT_STEP.value if _SIMULATE_FAULTS.value else None
      ),
      simulate_slice_index=_SIMULATE_SLICE_INDEX.value,
      enable_control_server=enable_ctrl,
      control_port=_CONTROL_PORT.value,
      wait_for_start_signal=_WAIT_FOR_START_SIGNAL.value,
  )

  try:
    workload.run()
  finally:
    workload.close()
    if sim is not None:
      sim.stop()
    elastic_mgr.close()


if __name__ == "__main__":
  pathwaysutils.initialize()
  app.run(main)
