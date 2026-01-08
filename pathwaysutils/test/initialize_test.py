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
from pathwaysutils import _initialize

from google3.testing.pybase import googletest
from google3.testing.pybase import parameterized


class InitializeTest(parameterized.TestCase):

  def test_first_initialize(self):
    jax.config.update("jax_platforms", "proxy")
    _initialize._initialization_count = 0

    with self.assertLogs(_initialize._logger, level="DEBUG") as logs:
      _initialize.initialize()

    self.assertLen(logs.output, 2)
    self.assertIn("Starting initialize.", logs.output[0])
    self.assertIn(
        "Detected Pathways-on-Cloud backend. Applying changes.", logs.output[1]
    )

  @parameterized.named_parameters(
      ("initialization_count 1", 1),
      ("initialization_count 2", 2),
      ("initialization_count 5", 5),
      ("initialization_count 1000", 1000),
  )
  def test_initialize_more_than_once(self, initialization_count):
    _initialize._initialization_count = initialization_count

    with self.assertLogs(_initialize._logger, level="DEBUG") as logs:
      _initialize.initialize()

    self.assertLen(logs.output, 1)
    self.assertIn(
        "Already initialized. Ignoring duplicate call.", logs.output[0]
    )

  @parameterized.named_parameters(
      ("empty", ""),
      ("cpu", "cpu"),
      ("tpu", "tpu"),
      ("gpu", "gpu"),
      ("cpu,tpu,gpu", "cpu,tpu,gpu"),
  )
  def test_not_is_pathways_backend_used(self, platform: str):
    jax.config.update("jax_platforms", platform)
    self.assertFalse(_initialize.is_pathways_backend_used())

  @parameterized.named_parameters(
      ("proxy", "proxy"),
      ("proxy,cpu", "proxy,cpu"),
      ("cpu,proxy", "cpu,proxy"),
      ("tpu,cpu,proxy,gpu", "tpu,cpu,proxy,gpu"),
  )
  def test_is_pathways_backend_used(self, platform: str):
    jax.config.update("jax_platforms", platform)
    self.assertTrue(_initialize.is_pathways_backend_used())

  def test_persistence_enabled(self):
    os.environ["ENABLE_PATHWAYS_PERSISTENCE"] = "1"
    self.assertTrue(_initialize._is_persistence_enabled())

    os.environ["ENABLE_PATHWAYS_PERSISTENCE"] = "0"
    self.assertFalse(_initialize._is_persistence_enabled())

    os.environ["ENABLE_PATHWAYS_PERSISTENCE"] = ""
    self.assertRaises(ValueError, _initialize._is_persistence_enabled)

    del os.environ["ENABLE_PATHWAYS_PERSISTENCE"]
    self.assertFalse(_initialize._is_persistence_enabled())


if __name__ == "__main__":
  googletest.main()
