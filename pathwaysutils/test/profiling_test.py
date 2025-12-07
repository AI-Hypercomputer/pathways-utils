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

import json
import logging
from unittest import mock

import jax
from pathwaysutils import profiling
import requests

from absl.testing import absltest
from absl.testing import parameterized


class ProfilingTest(parameterized.TestCase):
  """Tests for Pathways on Cloud profiling."""

  def setUp(self):
    super().setUp()
    self.mock_post = self.enter_context(
        mock.patch.object(requests, "post", autospec=True)
    )
    profiling._profile_state.reset()
    profiling._first_profile_start = True
    profiling._profiler_thread = None
    self.mock_plugin_executable_cls = self.enter_context(
        mock.patch.object(
            profiling.plugin_executable, "PluginExecutable", autospec=True
        )
    )
    self.mock_plugin_executable_cls.return_value.call.return_value = (
        mock.MagicMock(),
        mock.MagicMock(),
    )
    self.mock_toy_computation = self.enter_context(
        mock.patch.object(profiling, "toy_computation", autospec=True)
    )
    self.mock_original_start_trace = self.enter_context(
        mock.patch.object(profiling, "_original_start_trace", autospec=True)
    )
    self.mock_original_stop_trace = self.enter_context(
        mock.patch.object(profiling, "_original_stop_trace", autospec=True)
    )

  @parameterized.parameters(8000, 1234)
  def test_collect_profile_port(self, port):
    result = profiling.collect_profile(
        port=port,
        duration_ms=1000,
        host="127.0.0.1",
        log_dir="gs://test_bucket/test_dir",
    )

    self.assertTrue(result)
    self.mock_post.assert_called_once_with(
        f"http://127.0.0.1:{port}/profiling",
        json={
            "duration_ms": 1000,
            "repository_path": "gs://test_bucket/test_dir",
        },
    )

  @parameterized.parameters(1000, 1234)
  def test_collect_profile_duration_ms(self, duration_ms):
    result = profiling.collect_profile(
        port=8000,
        duration_ms=duration_ms,
        host="127.0.0.1",
        log_dir="gs://test_bucket/test_dir",
    )

    self.assertTrue(result)
    self.mock_post.assert_called_once_with(
        "http://127.0.0.1:8000/profiling",
        json={
            "duration_ms": duration_ms,
            "repository_path": "gs://test_bucket/test_dir",
        },
    )

  @parameterized.parameters("127.0.0.1", "localhost", "192.168.1.1")
  def test_collect_profile_host(self, host):
    result = profiling.collect_profile(
        port=8000,
        duration_ms=1000,
        host=host,
        log_dir="gs://test_bucket/test_dir",
    )

    self.assertTrue(result)
    self.mock_post.assert_called_once_with(
        f"http://{host}:8000/profiling",
        json={
            "duration_ms": 1000,
            "repository_path": "gs://test_bucket/test_dir",
        },
    )

  @parameterized.parameters(
      "gs://test_bucket/test_log_dir",
      "gs://test_bucket2",
      "gs://test_bucket3/test/log/dir",
  )
  def test_collect_profile_log_dir(self, log_dir):
    result = profiling.collect_profile(
        port=8000, duration_ms=1000, host="127.0.0.1", log_dir=log_dir
    )

    self.assertTrue(result)
    self.mock_post.assert_called_once_with(
        "http://127.0.0.1:8000/profiling",
        json={
            "duration_ms": 1000,
            "repository_path": log_dir,
        },
    )

  @parameterized.parameters("/logs/test_log_dir", "relative_path/my_log_dir")
  def test_collect_profile_log_dir_error(self, log_dir):
    with self.assertRaises(ValueError):
      profiling.collect_profile(
          port=8000, duration_ms=1000, host="127.0.0.1", log_dir=log_dir
      )

  @parameterized.parameters(
      requests.exceptions.ConnectionError("Connection error"),
      requests.exceptions.Timeout("Timeout"),
      requests.exceptions.TooManyRedirects("Too many redirects"),
      requests.exceptions.RequestException("Request exception"),
      requests.exceptions.HTTPError("HTTP error"),
  )
  def test_collect_profile_request_error(self, exception):
    self.mock_post.side_effect = exception

    with self.assertLogs(profiling._logger, level=logging.ERROR) as logs:
      result = profiling.collect_profile(
          port=8000,
          duration_ms=1000,
          host="127.0.0.1",
          log_dir="gs://test_bucket/test_dir",
      )

    self.assertLen(logs.output, 1)
    self.assertIn("Failed to collect profiling data", logs.output[0])
    self.assertIn(str(exception), logs.output[0])
    self.assertFalse(result)
    self.mock_post.assert_called_once()

  def test_collect_profile_success(self):
    mock_response = mock.Mock()
    mock_response.raise_for_status.return_value = None
    self.mock_post.return_value = mock_response

    result = profiling.collect_profile(
        port=8000,
        duration_ms=1000,
        host="127.0.0.1",
        log_dir="gs://test_bucket/test_dir",
    )

    self.assertTrue(result)
    self.mock_post.assert_called_once()
    mock_response.raise_for_status.assert_called_once()

  @parameterized.parameters(
      "/logs/test_log_dir",
      "relative_path/my_log_dir",
      "cns://test_bucket/test_dir",
      "not_a_gcs_path",
  )
  def test_start_trace_log_dir_error(self, log_dir):
    with self.assertRaisesRegex(
        ValueError, "log_dir must be a GCS bucket path"
    ):
      profiling.start_trace(log_dir)

  def test_lock_is_always_released(self):
    # Lock should be released after successful start_trace and stop_trace.
    profiling.start_trace("gs://test_bucket/test_dir")
    self.assertFalse(profiling._profile_state.lock.locked())
    profiling.stop_trace()
    self.assertFalse(profiling._profile_state.lock.locked())

    # Lock should be released if start_trace fails.
    mock_result = (
        self.mock_plugin_executable_cls.return_value.call.return_value[1]
    )
    mock_result.result.side_effect = RuntimeError("start failed")
    with self.assertRaisesRegex(RuntimeError, "start failed"):
      profiling.start_trace("gs://test_bucket/test_dir2")
    self.assertFalse(profiling._profile_state.lock.locked())
    mock_result.result.side_effect = None

    # Lock should be released if stop_trace fails.
    profiling.start_trace("gs://test_bucket/test_dir3")
    self.assertFalse(profiling._profile_state.lock.locked())
    mock_result = (
        self.mock_plugin_executable_cls.return_value.call.return_value[1]
    )
    mock_result.result.side_effect = RuntimeError("stop failed")
    with self.assertRaisesRegex(RuntimeError, "stop failed"):
      profiling.stop_trace()
    self.assertFalse(profiling._profile_state.lock.locked())
    mock_result.result.side_effect = None

  def test_start_trace_success(self):
    profiling.start_trace("gs://test_bucket/test_dir")

    self.mock_toy_computation.assert_called_once()
    self.mock_plugin_executable_cls.assert_called_once_with(
        json.dumps(
            {"profileRequest": {"traceLocation": "gs://test_bucket/test_dir"}}
        )
    )
    self.mock_plugin_executable_cls.return_value.call.assert_called_once()
    self.mock_original_start_trace.assert_called_once_with(
        log_dir="gs://test_bucket/test_dir",
        create_perfetto_link=False,
        create_perfetto_trace=False,
        profiler_options=None,
    )
    self.assertIsNotNone(profiling._profile_state.executable)

  def test_start_trace_no_toy_computation_second_time(self):
    profiling.start_trace("gs://test_bucket/test_dir")
    profiling.stop_trace()

    self.mock_toy_computation.assert_called_once()
    self.mock_original_start_trace.assert_called_once()

    # Reset mock and call again
    self.mock_toy_computation.reset_mock()
    self.mock_original_start_trace.reset_mock()
    profiling.start_trace("gs://test_bucket/test_dir2")

    self.mock_toy_computation.assert_not_called()
    self.mock_original_start_trace.assert_called_once()

  def test_start_trace_while_running_error(self):
    profiling.start_trace("gs://test_bucket/test_dir")
    with self.assertRaisesRegex(ValueError, "trace is already being taken"):
      profiling.start_trace("gs://test_bucket/test_dir2")

  @parameterized.named_parameters(
      dict(
          testcase_name="host_trace_only",
          host_tracer_level=1,
          start_timestamp_ns=0,
          duration_ms=0,
          expected_profile_request={
              "traceLocation": "gs://test_bucket/test_dir",
              "xprofTraceOptions": {
                  "hostTraceLevel": 1,
              },
          },
      ),
      dict(
          testcase_name="all_options",
          host_tracer_level=2,
          start_timestamp_ns=123,
          duration_ms=456,
          expected_profile_request={
              "traceLocation": "gs://test_bucket/test_dir",
              "xprofTraceOptions": {
                  "hostTraceLevel": 2,
                  "traceOptions": {
                      "profilingStartTimeNs": 123,
                      "profilingDurationMs": 456,
                  },
              },
          },
      ),
  )
  def test_start_trace_with_profiler_options(
      self,
      host_tracer_level,
      start_timestamp_ns,
      duration_ms,
      expected_profile_request,
  ):
    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = host_tracer_level
    options.start_timestamp_ns = start_timestamp_ns
    options.duration_ms = duration_ms
    profiling.start_trace("gs://test_bucket/test_dir", profiler_options=options)
    self.mock_original_start_trace.assert_called_once_with(
        log_dir="gs://test_bucket/test_dir",
        create_perfetto_link=False,
        create_perfetto_trace=False,
        profiler_options=options,
    )
    self.mock_plugin_executable_cls.assert_called_once_with(
        json.dumps({"profileRequest": expected_profile_request})
    )

  def test_stop_trace_success(self):
    profiling.start_trace("gs://test_bucket/test_dir")
    # call() is called once in start_trace, and once in stop_trace.
    self.mock_plugin_executable_cls.return_value.call.assert_called_once()

    profiling.stop_trace()

    self.assertEqual(
        self.mock_plugin_executable_cls.return_value.call.call_count, 2
    )
    self.mock_original_stop_trace.assert_called_once()
    self.assertIsNone(profiling._profile_state.executable)

  def test_stop_trace_before_start_error(self):
    with self.assertRaisesRegex(
        ValueError, "stop_trace called before a trace is being taken!"
    ):
      profiling.stop_trace()

  def test_start_server_starts_thread(self):
    mock_thread = self.enter_context(
        mock.patch.object(profiling.threading, "Thread", autospec=True)
    )
    profiling.start_server(9000)
    mock_thread.assert_called_once_with(target=mock.ANY, args=(9000,))
    mock_thread.return_value.start.assert_called_once()
    self.assertIsNotNone(profiling._profiler_thread)

  def test_start_server_twice_raises_error(self):
    self.enter_context(
        mock.patch.object(profiling.threading, "Thread", autospec=True)
    )
    profiling.start_server(9000)
    with self.assertRaisesRegex(
        ValueError, "Only one profiler server can be active"
    ):
      profiling.start_server(9001)

  def test_stop_server_no_server_raises_error(self):
    with self.assertRaisesRegex(ValueError, "No active profiler server"):
      profiling.stop_server()

  def test_stop_server_does_nothing_if_server_exists(self):
    self.enter_context(
        mock.patch.object(profiling.threading, "Thread", autospec=True)
    )
    profiling.start_server(9000)
    profiling.stop_server()  # Should not raise

  def test_monkey_patch_jax(self):
    original_jax_start_trace = jax.profiler.start_trace
    original_jax_stop_trace = jax.profiler.stop_trace
    original_jax_start_server = jax.profiler.start_server
    original_jax_stop_server = jax.profiler.stop_server

    profiling.monkey_patch_jax()

    self.assertNotEqual(jax.profiler.start_trace, original_jax_start_trace)
    self.assertNotEqual(jax.profiler.stop_trace, original_jax_stop_trace)
    self.assertNotEqual(jax.profiler.start_server, original_jax_start_server)
    self.assertNotEqual(jax.profiler.stop_trace, original_jax_stop_trace)

    with mock.patch.object(
        profiling, "start_trace", autospec=True
    ) as mock_pw_start_trace:
      jax.profiler.start_trace("gs://bucket/dir")
      mock_pw_start_trace.assert_called_once_with("gs://bucket/dir")

    with mock.patch.object(
        profiling, "stop_trace", autospec=True
    ) as mock_pw_stop_trace:
      jax.profiler.stop_trace()
      mock_pw_stop_trace.assert_called_once()

    with mock.patch.object(
        profiling, "start_server", autospec=True
    ) as mock_pw_start_server:
      jax.profiler.start_server(1234)
      mock_pw_start_server.assert_called_once_with(1234)

    with mock.patch.object(
        profiling, "stop_server", autospec=True
    ) as mock_pw_stop_server:
      jax.profiler.stop_server()
      mock_pw_stop_server.assert_called_once()

    # Restore original jax functions
    jax.profiler.start_trace = original_jax_start_trace
    jax.profiler.stop_trace = original_jax_stop_trace
    jax.profiler.start_server = original_jax_start_server
    jax.profiler.stop_server = original_jax_stop_server


class CreateProfileRequestTest(parameterized.TestCase):

  def test_create_profile_request_no_options(self):
    request = profiling._create_profile_request("gs://bucket/dir", None)
    self.assertEqual(request, {"traceLocation": "gs://bucket/dir"})

  def test_create_profile_request_with_options_none_zero(self):
    options = mock.create_autospec(jax.profiler.ProfileOptions, instance=True)
    options.host_tracer_level = 0
    options.start_timestamp_ns = None
    options.duration_ms = None
    request = profiling._create_profile_request("gs://bucket/dir", options)
    self.assertEqual(
        request,
        {
            "traceLocation": "gs://bucket/dir",
            "xprofTraceOptions": {"hostTraceLevel": 0},
        },
    )

  def test_create_profile_request_with_options_all_set(self):
    options = mock.create_autospec(jax.profiler.ProfileOptions, instance=True)
    options.host_tracer_level = 2
    options.start_timestamp_ns = 12345
    options.duration_ms = 5000
    request = profiling._create_profile_request("gs://bucket/dir", options)
    self.assertEqual(
        request,
        {
            "traceLocation": "gs://bucket/dir",
            "xprofTraceOptions": {
                "hostTraceLevel": 2,
                "traceOptions": {
                    "profilingStartTimeNs": 12345,
                    "profilingDurationMs": 5000,
                },
            },
        },
    )


if __name__ == "__main__":
  absltest.main()
