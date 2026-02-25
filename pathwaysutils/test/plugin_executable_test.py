"""Tests related to the PluginExecutable class.

These should not exercise a specific feature that uses side channel, but rather
the general logic of the class.
"""
from absl.testing import absltest
import jax
from pathwaysutils import plugin_executable

PluginExecutable = plugin_executable.PluginExecutable
XlaRuntimeError = jax.errors.JaxRuntimeError


class PluginExecutableTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    orig_jax_platforms = getattr(jax.config, "jax_platforms", None)
    self.addCleanup(jax.config.update, "jax_platforms", orig_jax_platforms)

    jax.config.update("jax_platforms", "cpu")

  def test_bad_json_program(self):
    with self.assertRaisesRegex(XlaRuntimeError, "INVALID_ARGUMENT"):
      PluginExecutable('{"printTextRequest":{"badParamName":"foo"}}')

  def test_bad_program(self):
    with self.assertRaisesRegex(XlaRuntimeError, "INVALID_ARGUMENT"):
      PluginExecutable("this is not json")

if __name__ == "__main__":
  absltest.main()
