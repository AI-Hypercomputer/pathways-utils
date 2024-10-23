# Copyright 2024 Google LLC
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
