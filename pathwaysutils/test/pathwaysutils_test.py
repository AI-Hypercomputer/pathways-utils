import os
import jax
import pathwaysutils
from absl.testing import absltest


class PathwaysutilsTest(absltest.TestCase):

  def test_is_pathways_used(self):
    for platform in ["", "cpu", "tpu", "gpu", "cpu,tpu,gpu"]:
      jax.config.update("jax_platforms", platform)
      self.assertFalse(pathwaysutils._is_pathways_used())
    for platform in ["proxy", "proxy,cpu", "cpu,proxy", "tpu,cpu,proxy,gpu"]:
      jax.config.update("jax_platforms", platform)
      self.assertTrue(pathwaysutils._is_pathways_used())

  def test_persistence_enabled(self):
    os.environ["ENABLE_PATHWAYS_PERSISTENCE"] = "1"
    self.assertTrue(pathwaysutils._is_persistence_enabled())

    os.environ["ENABLE_PATHWAYS_PERSISTENCE"] = "0"
    self.assertFalse(pathwaysutils._is_persistence_enabled())

    os.environ["ENABLE_PATHWAYS_PERSISTENCE"] = ""
    self.assertRaises(ValueError, pathwaysutils._is_persistence_enabled)

    del os.environ["ENABLE_PATHWAYS_PERSISTENCE"]
    self.assertFalse(pathwaysutils._is_persistence_enabled())


if __name__ == "__main__":
  absltest.main()
