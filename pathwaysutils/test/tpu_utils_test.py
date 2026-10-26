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
"""Tests for TPU topology utilities."""

from absl.testing import absltest
from absl.testing import parameterized
from pathwaysutils import tpu_utils


class TpuUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("2d", "4x8", 32),
      ("3d", "4x4x4", 64),
      ("uppercase_and_whitespace", "  2X4  ", 8),
  )
  def test_parse_topology_valid(self, topology: str, expected_chips: int):
    self.assertEqual(tpu_utils.parse_topology(topology), expected_chips)

  @parameterized.named_parameters(
      ("empty", "", "Invalid topology"),
      ("whitespace", "   ", "Invalid topology"),
      ("non_numeric", "2xabc", "Invalid topology"),
      ("zero_dim", "0x4", "Topology dimensions must be positive"),
      ("negative_dim", "-2x4", "Topology dimensions must be positive"),
  )
  def test_parse_topology_invalid(self, topology: str, expected_error: str):
    with self.assertRaisesRegex(ValueError, expected_error):
      tpu_utils.parse_topology(topology)


if __name__ == "__main__":
  absltest.main()
