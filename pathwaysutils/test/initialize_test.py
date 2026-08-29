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
import time
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import pathwaysutils
from pathwaysutils import _initialize


class InitializeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    orig_jax_platforms = getattr(jax.config, "jax_platforms", None)
    self.addCleanup(jax.config.update, "jax_platforms", orig_jax_platforms)

    self._orig_environ = os.environ.copy()
    self.addCleanup(self._restore_environ)

  def _restore_environ(self):
    os.environ.clear()
    os.environ.update(self._orig_environ)

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

  def test_wait_for_devices_ready_default(self):
    # Should execute without errors on default devices.
    pathwaysutils.wait_for_devices_ready()

  def test_wait_for_devices_ready_explicit_devices(self):
    devices = jax.devices()[:1]
    pathwaysutils.wait_for_devices_ready(devices)

  def test_wait_for_devices_ready_empty(self):
    pathwaysutils.wait_for_devices_ready([])

  def test_wait_for_devices_ready_calls_jit_and_block_until_ready(self):
    mock_dev1 = mock.create_autospec(jax.Device, instance=True)
    mock_dev2 = mock.create_autospec(jax.Device, instance=True)
    mock_devices = [mock_dev1, mock_dev2]

    mock_jit_fn = mock.MagicMock(return_value="result")
    mock_jit = self.enter_context(
        mock.patch.object(jax, "jit", return_value=mock_jit_fn)
    )
    mock_block = self.enter_context(mock.patch.object(jax, "block_until_ready"))

    pathwaysutils.wait_for_devices_ready(mock_devices)

    self.assertEqual(mock_jit.call_count, 2)
    mock_jit.assert_any_call(mock.ANY, device=mock_dev1)
    mock_jit.assert_any_call(mock.ANY, device=mock_dev2)
    self.assertIs(
        mock_jit.call_args_list[0][0][0], mock_jit.call_args_list[1][0][0]
    )
    mock_block.assert_called_once_with(["result", "result"])

  def test_wait_for_devices_ready_logs(self):
    with self.assertLogs(_initialize._logger, level="INFO") as logs:
      pathwaysutils.wait_for_devices_ready(jax.devices()[:1], timeout=10)
    self.assertLen(logs.output, 2)
    self.assertIn(
        "Waiting for 1 devices to be ready (timeout=10).", logs.output[0]
    )
    self.assertIn("All 1 devices are ready.", logs.output[1])

  def test_wait_for_devices_ready_with_timeout_success(self):
    pathwaysutils.wait_for_devices_ready(timeout=60)

  def test_wait_for_devices_ready_with_timeout_exceeded(self):
    def slow_block_until_ready(results):
      time.sleep(1)
      return results

    self.enter_context(
        mock.patch.object(
            jax, "block_until_ready", side_effect=slow_block_until_ready
        )
    )
    with self.assertRaises(TimeoutError):
      pathwaysutils.wait_for_devices_ready(timeout=0.01)


if __name__ == "__main__":
  absltest.main()
