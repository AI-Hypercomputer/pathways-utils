# Copyright 2025 Google LLC
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
"""Watchdog tests."""

import logging
import sys
import threading
import traceback
from unittest import mock

from pathwaysutils.debug import watchdog

from absl.testing import absltest
from absl.testing import parameterized


class WatchdogTest(parameterized.TestCase):
  def test_watchdog_start_join(self):
    with (
        mock.patch.object(
            threading.Thread,
            "start",
            autospec=True,
        ) as mock_start,
        mock.patch.object(threading.Thread, "join", autospec=True) as mock_join,
    ):
      with watchdog.watchdog(timeout=1):
        mock_start.assert_called_once()
        mock_join.assert_not_called()

    mock_start.assert_called_once()
    mock_join.assert_called_once()

  @parameterized.named_parameters([
      (
          "thread 1",
          1,
          [
              "DEBUG:pathwaysutils.debug.watchdog:Thread: 1",
              "DEBUG:pathwaysutils.debug.watchdog:examplestack1",
          ],
      ),
      (
          "thread 2",
          2,
          [
              "DEBUG:pathwaysutils.debug.watchdog:Thread: 2",
              "DEBUG:pathwaysutils.debug.watchdog:examplestack2",
          ],
      ),
      (
          "thread 3",
          3,
          [
              "DEBUG:pathwaysutils.debug.watchdog:Thread: 3",
              "DEBUG:pathwaysutils.debug.watchdog:",
          ],
      ),
  ])
  def test_log_thread_strack_succes(self, thread_ident, expected_log_output):
    with (
        mock.patch.object(
            sys,
            "_current_frames",
            return_value={1: ["example", "stack1"], 2: ["example", "stack2"]},
            autospec=True,
        ),
        mock.patch.object(
            traceback,
            "format_stack",
            side_effect=lambda stack_str_list: stack_str_list,
            autospec=True,
        ),
    ):
      mock_thread = mock.create_autospec(threading.Thread, instance=True)
      mock_thread.ident = thread_ident

      with self.assertLogs(watchdog._logger, logging.DEBUG) as log_output:
        watchdog._log_thread_stack(mock_thread)

    self.assertEqual(log_output.output, expected_log_output)


if __name__ == "__main__":
  absltest.main()
