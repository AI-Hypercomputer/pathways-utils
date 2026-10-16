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
"""CLI entry point for the simple Transformer training workload."""

from collections.abc import Sequence
import logging

from absl import app
from absl import flags
import pathwaysutils
from pathwaysutils.extras.simple_model.config import DEFAULT_CONFIG
from pathwaysutils.extras.simple_model.config import parse_topology
from pathwaysutils.extras.simple_model.workload import Workload

_TOPOLOGY = flags.DEFINE_string(
    "topology", "", "Optional TPU slice topology (e.g., 4x8, 4x4x4)."
)
_DEVICES_PER_SLICE = flags.DEFINE_integer(
    "devices_per_slice", 0, "Devices per slice (inferred from topology if set)."
)
_TOTAL_STEPS = flags.DEFINE_integer("total_steps", 20, "Total steps.")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 0.0005, "Learning rate.")
_SNAPSHOT_INTERVAL = flags.DEFINE_integer(
    "snapshot_interval",
    0,
    "Interval for in-memory snapshots (0 to disable).",
)
_CHECKPOINT_INTERVAL = flags.DEFINE_integer(
    "checkpoint_interval", 0, "Interval for checkpoints (0 to disable)."
)
_CHECKPOINT_DIR = flags.DEFINE_string(
    "checkpoint_dir", "", "Checkpoint directory."
)


def main(argv: Sequence[str]) -> None:
  del argv
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
  )

  devices_per_slice = (
      parse_topology(_TOPOLOGY.value)
      if _TOPOLOGY.value
      else _DEVICES_PER_SLICE.value
  )

  workload = Workload(
      config=DEFAULT_CONFIG,
      devices_per_slice=devices_per_slice,
      total_steps=_TOTAL_STEPS.value,
      learning_rate=_LEARNING_RATE.value,
      snapshot_interval=_SNAPSHOT_INTERVAL.value,
      checkpoint_interval=_CHECKPOINT_INTERVAL.value,
      checkpoint_dir=_CHECKPOINT_DIR.value or None,
  )

  try:
    workload.run()
  finally:
    workload.close()


if __name__ == "__main__":
  pathwaysutils.initialize()
  app.run(main)
