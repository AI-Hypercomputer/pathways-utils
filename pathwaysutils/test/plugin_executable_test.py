"""Tests related to the PluginExecutable class.

These should not exercise a specific feature that uses side channel, but rather
the general logic of the class.
"""
import jax
from pathwaysutils import plugin_executable
from absl.testing import absltest

PluginExecutable = plugin_executable.PluginExecutable
XlaRuntimeError = jax.errors.JaxRuntimeError


class PluginExecutableTest(absltest.TestCase):

  def setUp(self):
    jax.config.update("jax_platforms", "cpu")
    super().setUp()

  def test_bad_json_program(self):
    with self.assertRaisesRegex(XlaRuntimeError, "INVALID_ARGUMENT"):
      PluginExecutable('{"printTextRequest":{"badParamName":"foo"}}')

  def test_bad_program(self):
    with self.assertRaisesRegex(XlaRuntimeError, "INVALID_ARGUMENT"):
      PluginExecutable("this is not json")

if __name__ == "__main__":
  absltest.main()
