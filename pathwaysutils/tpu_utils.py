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
"""TPU topology utilities."""

import math


def parse_topology(topology: str) -> int:
  """Returns the total number of chips in a topology string like '4x8' or '4x4x4'."""
  stripped = topology.strip().lower()
  if not stripped:
    raise ValueError(f"Invalid topology: {topology!r}")
  try:
    dims = [int(x) for x in stripped.split("x")]
  except ValueError as e:
    raise ValueError(f"Invalid topology: {topology!r}") from e
  if any(d <= 0 for d in dims):
    raise ValueError(
        f"Topology dimensions must be positive integers, got {topology!r}"
    )
  return math.prod(dims)
