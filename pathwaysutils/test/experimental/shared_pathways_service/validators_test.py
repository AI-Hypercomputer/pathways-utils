"""Tests for validation functions for the Shared Pathways service."""

from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized

from pathwaysutils.experimental.shared_pathways_service import validators


class ValidatorsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="simple_service", service="test-service:1234"),
      dict(
          testcase_name="complex_hostname",
          service="pathways-cluster-pathways-head-0-0.pathways-cluster:8080",
      ),
  )
  def test_validate_pathways_service_success(self, service):
    """Tests that valid pathways service strings pass validation."""
    validators.validate_pathways_service(service)

  @parameterized.named_parameters(
      dict(
          testcase_name="missing_port",
          service="test-service",
          expected_regex=(
              "pathways_service=test-service is not in the expected format of"
          ),
      ),
      dict(
          testcase_name="empty_port",
          service="test-service:",
          expected_regex=(
              "pathways_service=test-service: contains an empty string for the"
              " service port."
          ),
      ),
      dict(
          testcase_name="empty_service_name",
          service=":1234",
          expected_regex=(
              "pathways_service=:1234 contains an empty string for the service"
              " name."
          ),
      ),
      dict(
          testcase_name="non_numeric_port",
          service="test-service:port",
          expected_regex=(
              "pathways_service=test-service:port contains a non-numeric"
              " service port."
          ),
      ),
      dict(
          testcase_name="empty_string",
          service="",
          expected_regex="No Pathways service found.",
      ),
      dict(
          testcase_name="too_many_parts",
          service="test-service:1234:5678",
          expected_regex=("pathways_service=test-service:1234:5678 is not in"
                          " the expected format of"),
      ),
  )
  def test_validate_pathways_service_failure(self, service, expected_regex):
    """Tests that invalid pathways service strings raise a ValueError."""
    with self.assertRaisesRegex(ValueError, expected_regex):
      validators.validate_pathways_service(service)

  @parameterized.named_parameters(
      dict(testcase_name="tpuv6e_2x2", instance_dict={"tpuv6e:2x2": 4}),
      dict(testcase_name="tpuv6e_2x4", instance_dict={"tpuv6e:2x4": 1}),
      dict(testcase_name="tpuv6e_2x2x2", instance_dict={"tpuv6e:2x2x2": 1}),
      dict(testcase_name="tpuv6e_4x4", instance_dict={"tpuv6e:4x4": 1}),
      dict(testcase_name="tpuv5e_2x2", instance_dict={"tpuv5e:2x2": 4}),
      dict(testcase_name="tpuv5e_4x4", instance_dict={"tpuv5e:4x4": 1}),
      dict(testcase_name="tpuv5_2x2x2", instance_dict={"tpuv5:2x2x2": 1}),
      dict(testcase_name="tpuv5_2x2x4", instance_dict={"tpuv5:2x2x4": 1}),
  )
  def test_validate_tpu_instances_success(self, instance_dict):
    """Tests that valid instance lists pass validation."""
    validators.validate_tpu_instances(instance_dict)

  @parameterized.named_parameters(
      dict(
          testcase_name="tpuv6e_8",
          instance_dict={"tpuv6e-8": 1},
          expected_regex="Unrecognized instance format: tpuv6e-8.",
      ),
      dict(
          testcase_name="tpuv6e_4",
          instance_dict={"tpuv6e-4": 1},
          expected_regex="Unrecognized instance format: tpuv6e-4.",
      ),
      dict(
          testcase_name="tpuv6e_16",
          instance_dict={"tpuv6e-16": 1},
          expected_regex="Unrecognized instance format: tpuv6e-16.",
      ),
      dict(
          testcase_name="invalid_format",
          instance_dict={"invalid-format": 1},
          expected_regex="Unrecognized instance format: invalid-format.",
      ),
      dict(
          testcase_name="ct5lp-hightpu-16t_4x4",
          instance_dict={"ct5lp-hightpu-16t:4x4": 1},
          expected_regex="Unrecognized instance format: ct5lp-hightpu-16t:4x4.",
      ),
      dict(
          testcase_name="ct5lp_hightpu_16t_2x2",
          instance_dict={"ct5lp-hightpu-16t:2x2": 1},
          expected_regex="Unrecognized instance format: ct5lp-hightpu-16t:2x2.",
      ),
      dict(
          testcase_name="tpuv5p_2x4x4",
          instance_dict={"tpuv5p:2x4x4": 1},
          expected_regex="Unrecognized instance format: tpuv5p:2x4x4.",
      ),
      dict(
          testcase_name="ct5lp_8",
          instance_dict={"ct5lp-8": 1},
          expected_regex="Unrecognized instance format: ct5lp-8.",
      ),
      dict(
          testcase_name="ct5l_8",
          instance_dict={"ct5l-8": 1},
          expected_regex="Unrecognized instance format: ct5l-8.",
      ),
      dict(
          testcase_name="ct5p_2x2",
          instance_dict={"ct5p:2x2": 1},
          expected_regex="Unrecognized instance format: ct5p:2x2.",
      ),
      dict(
          testcase_name="ct5p_4",
          instance_dict={"ct5p-4": 1},
          expected_regex="Unrecognized instance format: ct5p-4.",
      ),
      dict(
          testcase_name="ct7e_8",
          instance_dict={"ct7e-8": 1},
          expected_regex="Unrecognized instance format: ct7e-8.",
      ),
      dict(
          testcase_name="tpuv7e-8",
          instance_dict={"tpuv7e-8": 1},
          expected_regex="Unrecognized instance format: tpuv7e-8.",
      ),
      dict(
          testcase_name="ct5lp_1x1x",
          instance_dict={"ct5lp:1x1x": 1},
          expected_regex="Unrecognized instance format: ct5lp:1x1x.",
      ),
      dict(
          testcase_name="ct5lp_axb",
          instance_dict={"ct5lp:axb": 1},
          expected_regex="Unrecognized instance format: ct5lp:axb.",
      ),
      dict(
          testcase_name="foo_bar",
          instance_dict={"foo-bar": 1},
          expected_regex="Unrecognized instance format: foo-bar.",
      ),
      dict(
          testcase_name="empty_dict",
          instance_dict={},
          expected_regex="No instances found.",
      ),
      dict(
          testcase_name="empty_key",
          instance_dict={"": 2},
          expected_regex=(
              r"expected_tpu_instances=\{\'\'\: 2\} contains an empty string"
              r" for an instance name."
          ),
      ),
      dict(
          testcase_name="tpuv6e_1_dim",
          instance_dict={"tpuv6e:1": 1},
          expected_regex="Unrecognized instance format: tpuv6e:1.",
      ),
      dict(
          testcase_name="tpuv6e_4_dims",
          instance_dict={"tpuv6e:1x2x3x4": 1},
          expected_regex="Unrecognized instance format: tpuv6e:1x2x3x4.",
      ),
      dict(
          testcase_name="multiple_keys",
          instance_dict={"tpuv6e:2x2": 2, "v5e-16": 1},
          expected_regex="Only one machine type is supported at this time.",
      ),
  )
  def test_validate_tpu_instances_failure(
      self, instance_dict, expected_regex
  ):
    """Tests that invalid TPU instance dictionaries raise a ValueError."""
    with self.assertRaisesRegex(ValueError, expected_regex):
      validators.validate_tpu_instances(instance_dict)

  @parameterized.named_parameters(
      dict(testcase_name="empty_list", options=[]),
      dict(testcase_name="none", options=None),
      dict(testcase_name="valid_options", options=["key1:val1", "key2:val2"]),
      dict(
          testcase_name="with_xla_flags",
          options=['xla_flags:"--flag1 --flag2"'],
      ),
  )
  def test_validate_proxy_options_success(self, options):
    validators.validate_proxy_options(options)

  @parameterized.named_parameters(
      dict(
          testcase_name="no_colon",
          options=["invalid_option"],
          expected_regex='--proxy_options must be in the format "key:value".',
      ),
      dict(
          testcase_name="empty_key",
          options=[":value"],
          expected_regex='--proxy_options must be in the format "key:value".',
      ),
      dict(
          testcase_name="empty_value",
          options=["key:"],
          expected_regex='--proxy_options must be in the format "key:value".',
      ),
  )
  def test_validate_proxy_options_failure(self, options, expected_regex):
    with self.assertRaisesRegex(flags.ValidationError, expected_regex):
      validators.validate_proxy_options(options)

  @parameterized.named_parameters(
      dict(testcase_name="empty_list", xla_flags=[]),
      dict(testcase_name="none", xla_flags=None),
      dict(
          testcase_name="valid_flags",
          xla_flags=[
              "--xla_tpu_scoped_vmem_limit_kib=98304",
              "--xla_tpu_use_minor_sharding_for_major_trivial_input=true",
          ],
      ),
  )
  def test_validate_xla_flags_success(self, xla_flags):
    validators.validate_xla_flags(xla_flags)

  @parameterized.named_parameters(
      dict(
          testcase_name="invalid_prefix",
          xla_flags=["--not_xla_flag"],
          expected_regex="XLA flag '--not_xla_flag' must start with '--xla_'.",
      ),
  )
  def test_validate_xla_flags_failure(self, xla_flags, expected_regex):
    with self.assertRaisesRegex(flags.ValidationError, expected_regex):
      validators.validate_xla_flags(xla_flags)

  def test_validate_sidecar_image_versions_success(self):
    mock_sys_info = mock.Mock()
    mock_sys_info.major = 3
    mock_sys_info.minor = 12
    mock_sys_info.micro = 8
    with mock.patch("sys.version_info", mock_sys_info), mock.patch(
        "jax.__version__", "0.10.0"
    ):
      # Python 3.12 (matches 3.12.8), JAX 0.10.0 (matches 0.10.0)
      validators.validate_sidecar_image_versions(
          "us-docker.pkg.dev/.../sidecar:20260423-python_3.12-jax_0.10.0"
      )

      # Python omitted, JAX 0.10 (matches 0.10.0)
      validators.validate_sidecar_image_versions(
          "us-docker.pkg.dev/.../sidecar:20260423-jax_0.10"
      )

      # No version info
      validators.validate_sidecar_image_versions(
          "us-docker.pkg.dev/.../sidecar:latest"
      )

  def test_validate_sidecar_image_versions_python_mismatch(self):
    mock_sys_info = mock.Mock()
    mock_sys_info.major = 3
    mock_sys_info.minor = 11
    mock_sys_info.micro = 5
    with mock.patch("sys.version_info", mock_sys_info), mock.patch(
        "jax.__version__", "0.10.0"
    ):
      with self.assertRaisesRegex(ValueError, "Python version mismatch"):
        validators.validate_sidecar_image_versions(
            "us-docker.pkg.dev/.../sidecar:20260423-python_3.12-jax_0.10.0"
        )

  def test_validate_sidecar_image_versions_jax_mismatch(self):
    mock_sys_info = mock.Mock()
    mock_sys_info.major = 3
    mock_sys_info.minor = 12
    mock_sys_info.micro = 8
    with mock.patch("sys.version_info", mock_sys_info), mock.patch(
        "jax.__version__", "0.9.0"
    ):
      with self.assertRaisesRegex(ValueError, "JAX version mismatch"):
        validators.validate_sidecar_image_versions(
            "us-docker.pkg.dev/.../sidecar:20260423-python_3.12-jax_0.10.0"
        )


if __name__ == "__main__":
  absltest.main()
