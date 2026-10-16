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
"""Single universal configuration for the simple Transformer model."""

import dataclasses
import math


@dataclasses.dataclass(frozen=True)
class ModelConfig:
  """Universal Transformer configuration that works for any hardware topology.

  The dimensions (hidden size 512, intermediate size 2048, 8 heads) are powers
  of 2 that divide evenly across any TPU slice size (from 1 to 512 chips).
  Global batch size scales linearly with the number of devices.
  """
  vocab_size: int = 4096
  hidden_size: int = 512
  intermediate_size: int = 2048
  num_heads: int = 8
  head_dim: int = 64
  num_layers: int = 2
  per_device_batch_size: int = 1
  seq_len: int = 16


DEFAULT_CONFIG = ModelConfig()

CONFIGS: dict[str, ModelConfig] = {
    "default": DEFAULT_CONFIG,
    "test": DEFAULT_CONFIG,
    "small": DEFAULT_CONFIG,
    "large": ModelConfig(
        vocab_size=32768,
        hidden_size=2048,
        intermediate_size=8192,
        num_heads=32,
        head_dim=64,
        num_layers=8,
    ),
}


def parse_topology(topology: str) -> int:
  """Returns the total number of chips in a topology string like '4x8' or '4x4x4'."""
  dims = [int(x) for x in topology.strip().lower().split("x") if x.isdigit()]
  return math.prod(dims) if dims else 1
