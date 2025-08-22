# Copyright 2025 Google LLC
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
"""A simulated manager for elastic training.

This module provides a simulated manager for elastic training. It can be used
to test elastic training without needing to actually trigger elastic events.
Instead, the user can control which slices are available at what times by
calling `update_good_slice_indices`.
"""

import logging
from typing import Sequence

import jax
from pathwaysutils.debug import timing
from pathwaysutils.elastic import manager


_logger = logging.getLogger(__name__)


class SimulatedManager(manager.Manager):
  """An elastic manager with settable slice availability.

  This class can be used to modify which slices are marked as available by
  overloading the `get_slice_availability` function.
  """

  _simulated_good_slice_indices: set[int]

  def __init__(
      self,
      devices: Sequence[jax.Device],
      reshard_check_period: int = 1,
      snapshot_period: int = 1,
      max_elastic_down_event_count: int | None = None,
      max_reshard_retry_count: int | None = None,
  ) -> None:
    """Initializes the simulated manager.

    Args:
      devices: The devices to use. If None, jax.devices() is used.
      reshard_check_period: The number of steps between reshard checks after a
        slice down event has occurred.
      snapshot_period: The number of steps between snapshots.
      max_elastic_down_event_count: The maximum number of elastic down events.
        If None, there is no limit.
      max_reshard_retry_count: The maximum number of consequetive reshard
        retries. If None, there is no limit.
    """
    self._simulated_good_slice_indices = set(d.slice_index for d in devices)

    super().__init__(
        devices,
        snapshot_period,
        reshard_check_period,
        max_elastic_down_event_count,
        max_reshard_retry_count,
    )

  def update_good_slice_indices(self, good_slice_indices: set[int]) -> None:
    """Sets the good slice indices.

    Subsequent calls to `get_slice_availability` will return these indices.

    Args:
      good_slice_indices: The simulated good slice indices.
    """
    self._simulated_good_slice_indices = good_slice_indices
    _logger.debug(
        "Updated: simumlated_good_slice_indices=%s",
        self._simulated_good_slice_indices,
    )

  @timing.timeit
  def get_slice_availability(self) -> set[int]:
    """Returns the set of good slice indices.

    Returns:
      The set of good slice indices from the last call to
      update_good_slice_indices. Returns an empty set if
      update_good_slice_indices has not been called.
    """
    good_slice_indices = self._simulated_good_slice_indices

    _logger.debug("good_slice_indices=%s", good_slice_indices)

    return good_slice_indices
