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

TWISTED_SUFFIX = "twisted"
UNTWISTED_SUFFIX = "untwisted"
_VALID_TOPOLOGY_SUFFIXES = frozenset({TWISTED_SUFFIX, UNTWISTED_SUFFIX})


def parse_topology_dimensions(
    topology: str,
) -> tuple[tuple[int, ...], str | None]:
  """Parses a topology string into its integer dimensions and twist suffix.

  Args:
    topology: Topology string such as '4x8', '4x4x4', or '4x4x8_twisted'.

  Returns:
    A tuple `(dimensions, suffix)` where `dimensions` is a tuple of positive
    integers and `suffix` is `'twisted'`, `'untwisted'`, or `None`.

  Raises:
    ValueError: If the topology string is malformed, has non-positive
      dimensions, or specifies a twist suffix on a non-3D topology.
  """
  stripped = topology.strip().lower()
  if not stripped:
    raise ValueError(f"Invalid topology: {topology!r}")
  base = stripped
  suffix: str | None = None
  if "_" in stripped:
    base, suffix = stripped.split("_", 1)
    if suffix not in _VALID_TOPOLOGY_SUFFIXES:
      raise ValueError(f"Invalid topology: {topology!r}")
  try:
    dims = tuple(int(x) for x in base.split("x"))
  except ValueError as e:
    raise ValueError(f"Invalid topology: {topology!r}") from e
  if any(d <= 0 for d in dims):
    raise ValueError(
        f"Topology dimensions must be positive integers, got {topology!r}"
    )
  if suffix is not None and len(dims) != 3:
    raise ValueError(
        f"Invalid topology {topology!r}: suffix {suffix!r} requires a 3D"
        " topology"
    )
  return dims, suffix


def parse_topology(topology: str) -> int:
  """Returns the total number of chips in a topology string."""
  dims, _ = parse_topology_dimensions(topology)
  return math.prod(dims)
