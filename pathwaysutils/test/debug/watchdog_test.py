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
import time
import traceback
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from pathwaysutils.debug import watchdog


class WatchdogTest(parameterized.TestCase):

  @parameterized.parameters([
      "test",
      "loop",
      "initialization",
  ])
  def test_watchdog_name(self, watchdog_name):
    mock_thread_cls = self.enter_context(
        mock.patch.object(threading, "Thread", autospec=True)
    )

    with watchdog.watchdog(name=watchdog_name, timeout=1):
      pass

    mock_thread_cls.assert_called_once_with(name=watchdog_name, target=mock.ANY)

  def test_watchdog_start_join(self):
    mock_start = self.enter_context(
        mock.patch.object(threading.Thread, "start", autospec=True)
    )
    mock_join = self.enter_context(
        mock.patch.object(threading.Thread, "join", autospec=True)
    )

    with watchdog.watchdog(name="test", timeout=1):
      mock_start.assert_called_once()
      mock_join.assert_not_called()

    mock_start.assert_called_once()
    mock_join.assert_called_once()

  @parameterized.named_parameters(
      dict(
          testcase_name="thread_1",
          thread_ident=1,
          expected_log_output=[
              "DEBUG:pathwaysutils.debug.watchdog:Thread: 1",
              "DEBUG:pathwaysutils.debug.watchdog:examplestack1",
          ],
      ),
      dict(
          testcase_name="thread_2",
          thread_ident=2,
          expected_log_output=[
              "DEBUG:pathwaysutils.debug.watchdog:Thread: 2",
              "DEBUG:pathwaysutils.debug.watchdog:examplestack2",
          ],
      ),
      dict(
          testcase_name="thread_3",
          thread_ident=3,
          expected_log_output=[
              "DEBUG:pathwaysutils.debug.watchdog:Thread: 3",
              "DEBUG:pathwaysutils.debug.watchdog:",
          ],
      ),
  )
  def test_log_thread_strack_succes(self, thread_ident, expected_log_output):
    self.enter_context(
        mock.patch.object(
            sys,
            "_current_frames",
            return_value={1: ["example", "stack1"], 2: ["example", "stack2"]},
            autospec=True,
        )
    )
    self.enter_context(
        mock.patch.object(
            traceback,
            "format_stack",
            side_effect=lambda stack_str_list: stack_str_list,
            autospec=True,
        )
    )

    mock_thread = mock.create_autospec(threading.Thread, instance=True)
    mock_thread.ident = thread_ident

    with self.assertLogs(watchdog._logger, logging.DEBUG) as log_output:
      watchdog._log_thread_stack(mock_thread)

    self.assertEqual(log_output.output, expected_log_output)

  @parameterized.named_parameters(
      dict(
          testcase_name="test_logs_1",
          name_arg="test_logs_1",
          timeout=1,
          repeat=False,
          expected_log_messages=[
              (
                  "Registering 'test_logs_1' watchdog with timeout 1 seconds"
                  " and repeat=False"
              ),
              "Deregistering 'test_logs_1' watchdog",
          ],
      ),
      dict(
          testcase_name="test_logs_2",
          name_arg="test_logs_2",
          timeout=2,
          repeat=True,
          expected_log_messages=[
              (
                  "Registering 'test_logs_2' watchdog with timeout 2 seconds"
                  " and repeat=True"
              ),
              "Deregistering 'test_logs_2' watchdog",
          ],
      ),
  )
  def test_watchdog_logs(
      self,
      name_arg: str,
      timeout: float,
      repeat: bool,
      expected_log_messages: list[str],
  ):
    # Test registration and deregistration logs
    with self.assertLogs(watchdog._logger, logging.DEBUG) as log_output:
      with watchdog.watchdog(name=name_arg, timeout=timeout, repeat=repeat):
        pass

    output_messages = [record.getMessage() for record in log_output.records]
    self.assertEqual(output_messages, expected_log_messages)

  def test_watchdog_timeout_logs(self):
    self.enter_context(
        mock.patch.object(
            threading,
            "enumerate",
            return_value=[],
            autospec=True,
        )
    )

    # Test timeout log
    with self.assertLogs(watchdog._logger, logging.DEBUG) as log_output:
      with watchdog.watchdog(name="test_logs_timeout", timeout=0.01):
        time.sleep(0.02)

    output_messages = [record.getMessage() for record in log_output.records]
    self.assertIn(
        "Registering 'test_logs_timeout' watchdog with timeout 0.01 seconds and"
        " repeat=True",
        output_messages,
    )

    self.assertIn(
        "'test_logs_timeout' watchdog thread stack dump every 0.01 seconds."
        " Count: 0",
        output_messages,
    )


if __name__ == "__main__":
  absltest.main()
