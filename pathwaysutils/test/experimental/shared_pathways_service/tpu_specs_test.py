"""Tests for tpu_specs.py."""

from absl.testing import absltest
from absl.testing import parameterized
from pathwaysutils.experimental.shared_pathways_service import tpu_specs


class TpuSpecsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("v5e", "v5e", 4, "tpu-v5-lite-podslice", "tpuv5e"),
      ("v5p", "v5p", 4, "tpu-v5p-slice", "tpuv5"),
      ("v6e", "v6e", 4, "tpu-v6e-slice", "tpuv6e"),
      ("tpu7x", "tpu7x", 4, "tpu7x", "tpu7x"),
  )
  def test_get_tpu_config(
      self, tpu_type, expected_chips, expected_label, expected_prefix
  ):
    config = tpu_specs.get_tpu_config(tpu_type)
    self.assertEqual(config.chips_per_vm, expected_chips)
    self.assertEqual(config.accelerator_label, expected_label)
    self.assertEqual(config.instance_prefix, expected_prefix)

  def test_get_tpu_config_invalid(self):
    with self.assertRaises(ValueError):
      tpu_specs.get_tpu_config("invalid_tpu")

  @parameterized.named_parameters(
      ("v5e_single", "4x4", 4, 4),
      ("v5e_large", "4x8", 4, 8),
      ("v5e_multi", "8x8", 4, 16),
      ("3d_topology", "2x2x2", 4, 2),
  )
  def test_calculate_vms_per_slice(self, topology, chips_per_vm, expected_vms):
    self.assertEqual(
        tpu_specs.calculate_vms_per_slice(topology, chips_per_vm), expected_vms
    )

  def test_calculate_vms_per_slice_invalid_format(self):
    with self.assertRaisesRegex(ValueError, "Invalid topology format"):
      tpu_specs.calculate_vms_per_slice("invalid", 4)

  def test_calculate_vms_per_slice_indivisible(self):
    with self.assertRaisesRegex(ValueError, "is not divisible by chips_per_vm"):
      tpu_specs.calculate_vms_per_slice("2x1", 4)

  @parameterized.named_parameters(
      ("v5e", "tpuv5e:4x8", ("v5e", "4x8")),
      ("v5p", "tpuv5:8x8", ("v5p", "8x8")),
      ("v6e", "tpuv6e:4x4", ("v6e", "4x4")),
      ("tpu7x", "tpu7x:4x4", ("tpu7x", "4x4")),
  )
  def test_parse_tpu_type_string(self, tpu_type_str, expected):
    self.assertEqual(tpu_specs.parse_tpu_type_string(tpu_type_str), expected)

  def test_parse_tpu_type_string_invalid_format(self):
    with self.assertRaisesRegex(ValueError, "Invalid tpu_type string"):
      tpu_specs.parse_tpu_type_string("tpuv5e-4x8")

  def test_parse_tpu_type_string_invalid_prefix(self):
    with self.assertRaisesRegex(ValueError, "Unsupported TPU prefix"):
      tpu_specs.parse_tpu_type_string("invalid_prefix:4x8")

  def test_get_tpu_params(self):
    params = tpu_specs.get_tpu_params("v5e", "4x8")
    expected = {
        "ACCELERATOR_LABEL": "tpu-v5-lite-podslice",
        "TOPOLOGY": "4x8",
        "VMS_PER_SLICE": "8",
        "CHIPS_PER_VM": "4",
        "INSTANCE_TYPE": "tpuv5e:4x8",
    }
    self.assertEqual(params, expected)


if __name__ == "__main__":
  absltest.main()
