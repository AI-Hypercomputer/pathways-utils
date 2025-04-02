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
"""Timing tests."""

import logging
import time
from unittest import mock

from pathwaysutils.debug import timing

from absl.testing import absltest
from absl.testing import parameterized


class TimingTest(parameterized.TestCase):

  def test_timer_context_manager(self):
    with mock.patch.object(
        time,
        "time",
        side_effect=[1, 8.9],
        autospec=True,
    ):
      with timing.Timer("test_timer") as timer:
        pass

    self.assertEqual(timer.name, "test_timer")
    self.assertEqual(timer.start, 1)
    self.assertEqual(timer.stop, 8.9)
    self.assertEqual(timer.duration, 7.9)
    self.assertEqual(str(timer), "test_timer elapsed 7.9000 seconds.")

  def test_timeit_log(self):

    @timing.timeit
    def my_function():
      pass

    with mock.patch.object(
        time,
        "time",
        side_effect=[1, 8.9, 0],  # Third time is used for logging.
        autospec=True,
    ):
      with self.assertLogs(timing._logger, logging.DEBUG) as log_output:
        my_function()

    self.assertEqual(
        log_output.output,
        [
            "DEBUG:pathwaysutils.debug.timing:my_function"
            " elapsed 7.9000 seconds."
        ],
    )

  def test_timeit_return_value(self):

    @timing.timeit
    def my_function():
      return "test"

    self.assertEqual(my_function(), "test")

  def test_timeit_exception(self):

    @timing.timeit
    def my_function():
      raise ValueError("test")

    with self.assertRaises(ValueError):
      my_function()


if __name__ == "__main__":
  absltest.main()
