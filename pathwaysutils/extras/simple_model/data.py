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
"""Synthetic token data source and Grain elastic iteration."""

from typing import Any

import grain.python as grain
import numpy as np
from pathwaysutils.extras.simple_model import config as config_lib


class SyntheticTokenSource(grain.RandomAccessDataSource):
  """Grain RandomAccessDataSource providing deterministic synthetic integer tokens."""

  def __init__(
      self, vocab_size: int, seq_len: int, num_samples: int = 10_000_000
  ):
    self.vocab_size = vocab_size
    self.seq_len = seq_len
    self.num_samples = num_samples

  def __len__(self) -> int:
    return self.num_samples

  def __getitem__(self, record_key: int) -> np.ndarray:
    return np.random.RandomState(record_key % 2_147_483_647).randint(
        0, self.vocab_size, size=(self.seq_len,), dtype=np.int32
    )


def create_token_dataset(config: config_lib.ModelConfig) -> grain.MapDataset:
  """Creates a Grain MapDataset from the synthetic token data source."""
  source = SyntheticTokenSource(
      vocab_size=config.vocab_size, seq_len=config.seq_len
  )
  return grain.MapDataset.source(source)


def create_data_iterator(
    dataset: grain.MapDataset,
    global_batch_size: int,
    restore_state: dict[str, Any] | None = None,
) -> grain.experimental.ElasticIterator:
  """Creates a Grain ElasticIterator with optional state restoration."""
  it = grain.experimental.ElasticIterator(
      ds=dataset,
      global_batch_size=global_batch_size,
      shard_options=grain.NoSharding(),
  )
  if restore_state is not None:
    it.set_state(restore_state)
  return it
