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
      ("3d_twisted", "4x4x8_twisted", 128),
      ("3d_untwisted", "4x4x8_untwisted", 128),
      ("3d_class2_twisted", "4x8x8_twisted", 256),
      ("uppercase_and_whitespace", "  2X4  ", 8),
      ("uppercase_twisted", "  4X4X8_TWISTED  ", 128),
  )
  def test_parse_topology_valid(self, topology: str, expected_chips: int):
    self.assertEqual(tpu_utils.parse_topology(topology), expected_chips)

  @parameterized.named_parameters(
      ("2d", "4x8", (4, 8), None),
      ("3d", "4x4x4", (4, 4, 4), None),
      ("twisted", "4x4x8_twisted", (4, 4, 8), "twisted"),
      ("untwisted", "4x4x8_untwisted", (4, 4, 8), "untwisted"),
  )
  def test_parse_topology_dimensions(
      self,
      topology: str,
      expected_dims: tuple[int, ...],
      expected_suffix: str | None,
  ):
    self.assertEqual(
        tpu_utils.parse_topology_dimensions(topology),
        (expected_dims, expected_suffix),
    )

  @parameterized.named_parameters(
      ("empty", "", "Invalid topology"),
      ("whitespace", "   ", "Invalid topology"),
      ("non_numeric", "2xabc", "Invalid topology"),
      ("zero_dim", "0x4", "Topology dimensions must be positive"),
      ("negative_dim", "-2x4", "Topology dimensions must be positive"),
      ("unknown_suffix", "4x4x8_foo", "Invalid topology"),
      ("extra_suffix", "4x4x8_twisted_extra", "Invalid topology"),
      ("suffix_only", "_twisted", "Invalid topology"),
      ("2d_twisted", "4x8_twisted", "requires a 3D topology"),
      ("2d_untwisted", "4x8_untwisted", "requires a 3D topology"),
  )
  def test_parse_topology_invalid(self, topology: str, expected_error: str):
    with self.assertRaisesRegex(ValueError, expected_error):
      tpu_utils.parse_topology(topology)


if __name__ == "__main__":
  absltest.main()
