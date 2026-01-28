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
Instead, the user can control which slices are active at what times by
calling `update_active_slice_indices`.
"""

import logging
from typing import Sequence

import jax
from pathwaysutils.debug import timing
from pathwaysutils.elastic import manager


_logger = logging.getLogger(__name__)


class SimulatedManager(manager.Manager):
  """An elastic manager with settable slice activity.

  This class can be used to modify which slices are marked as active by
  overloading the `get_active_slice_indices` function.
  """

  _simulated_active_slice_indices: set[int]

  def __init__(self, devices: Sequence[jax.Device]) -> None:
    """Initializes the simulated manager.

    Args:
      devices: The devices to use. If None, jax.devices() is used.
    """
    self._simulated_active_slice_indices = set(d.slice_index for d in devices)

    super().__init__(devices)

  def update_active_slice_indices(self, active_slice_indices: set[int]) -> None:
    """Sets the active slice indices.

    Subsequent calls to `get_active_slice_indices` will return these indices.

    Args:
      active_slice_indices: The simulated active slice indices.
    """
    self._simulated_active_slice_indices = active_slice_indices
    _logger.debug(
        "Updated: simumlated_active_slice_indices=%s",
        self._simulated_active_slice_indices,
    )

  @timing.timeit
  def get_active_slice_indices(self) -> set[int]:
    """Returns the set of active slice indices.

    Returns:
      The set of active slice indices from the last call to
      update_active_slice_indices. Returns an empty set if
      update_active_slice_indices has not been called.
    """
    active_slice_indices = self._simulated_active_slice_indices

    _logger.debug("active_slice_indices=%s", active_slice_indices)

    return active_slice_indices
