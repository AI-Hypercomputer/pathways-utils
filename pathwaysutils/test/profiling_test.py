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
from typing import Any
import os
from unittest import mock
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from pathwaysutils import profiling
import requests


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
    self.mock_datetime = self.enter_context(
        mock.patch.object(profiling.datetime, "datetime", autospec=True)
    )
    self.mock_datetime.now.return_value.strftime.return_value = (
        "2026_06_04_05_29_33"
    )

  def _get_expected_profile_request(
      self,
      trace_location: str,
      max_num_hosts: int = 1,
      session_id: str = "2026_06_04_05_29_33",
  ) -> dict[str, Any]:
    if jax.version.__version_info__ >= (0, 9, 2):
      return {
          "profileRequest": {
              "traceLocation": trace_location,
              "maxNumHosts": max_num_hosts,
              "xprofTraceOptions": {
                  "traceDirectory": trace_location,
                  "pwTraceOptions": {
                      "enablePythonTracer": True,
                  },
                  "traceSessionName": session_id,
              },
          }
      }
    else:
      return {
          "profileRequest": {
              "traceLocation": trace_location,
              "maxNumHosts": max_num_hosts,
          }
      }

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
        headers={},
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
        headers={},
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
        headers={},
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
        headers={},
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
    with self.assertRaisesRegex(ValueError, "Path must be a GCS path"):
      profiling.start_trace(log_dir)

  def test_lock_released_on_success(self):
    """Tests lock release after successful start and stop trace.

    Verifies that the profiling lock is released after both a successful
    `start_trace` and `stop_trace` calls.
    """
    profiling.start_trace("gs://test_bucket/test_dir")
    self.assertFalse(profiling._profile_state.lock.locked())
    profiling.stop_trace()
    self.assertFalse(profiling._profile_state.lock.locked())

  def test_lock_released_on_start_failure(self):
    """Tests that the lock is released if start_trace fails."""
    mock_result = (
        self.mock_plugin_executable_cls.return_value.call.return_value[1]
    )
    mock_result.result.side_effect = RuntimeError("start failed")
    with (
        self.assertRaisesRegex(RuntimeError, "start failed"),
        mock.patch.object(profiling._logger, "exception"),
    ):
      profiling.start_trace("gs://test_bucket/test_dir2")
    self.assertFalse(profiling._profile_state.lock.locked())

  def test_lock_released_on_stop_failure(self):
    """Tests that the lock is released if stop_trace fails."""
    profiling.start_trace("gs://test_bucket/test_dir3")
    self.assertFalse(profiling._profile_state.lock.locked())
    mock_result_fail = mock.MagicMock()
    mock_result_fail.result.side_effect = RuntimeError("stop failed")
    self.mock_plugin_executable_cls.return_value.call.return_value = (
        mock.MagicMock(),
        mock_result_fail,
    )
    self.mock_plugin_executable_cls.return_value.call.side_effect = None
    with self.assertRaisesRegex(RuntimeError, "stop failed"):
      profiling.stop_trace()
    self.assertFalse(profiling._profile_state.lock.locked())

  def test_start_trace_success(self):
    profiling.start_trace("gs://test_bucket/test_dir")

    self.mock_toy_computation.assert_called_once()
    expected_request = self._get_expected_profile_request(
        "gs://test_bucket/test_dir", max_num_hosts=1
    )
    self.mock_plugin_executable_cls.assert_called_once_with(
        json.dumps(expected_request)
    )
    self.mock_plugin_executable_cls.return_value.call.assert_called_once()
    self.mock_original_start_trace.assert_called_once()
    call_args = self.mock_original_start_trace.call_args[1]
    self.assertEqual(call_args["log_dir"], "gs://test_bucket/test_dir")
    self.assertFalse(call_args["create_perfetto_link"])
    self.assertFalse(call_args["create_perfetto_trace"])
    if jax.version.__version_info__ >= (0, 9, 2):
      self.assertEqual(
          call_args["profiler_options"].session_id, "2026_06_04_05_29_33"
      )
    self.assertIsNotNone(profiling._profile_state.executable)

  def test_start_trace_with_max_num_hosts(self):
    profiling.start_trace("gs://test_bucket/test_dir", max_num_hosts=10)

    self.mock_toy_computation.assert_called_once()
    expected_request = self._get_expected_profile_request(
        "gs://test_bucket/test_dir", max_num_hosts=10
    )
    self.mock_plugin_executable_cls.assert_called_once_with(
        json.dumps(expected_request)
    )
    self.mock_plugin_executable_cls.return_value.call.assert_called_once()
    self.mock_original_start_trace.assert_called_once()
    call_args = self.mock_original_start_trace.call_args[1]
    self.assertEqual(call_args["log_dir"], "gs://test_bucket/test_dir")
    self.assertFalse(call_args["create_perfetto_link"])
    self.assertFalse(call_args["create_perfetto_trace"])
    if jax.version.__version_info__ >= (0, 9, 2):
      self.assertEqual(
          call_args["profiler_options"].session_id, "2026_06_04_05_29_33"
      )

  @absltest.skipIf(
      jax.version.__version_info__ < (0, 9, 2),
      "ProfileOptions requires JAX 0.9.2 or newer",
  )
  def test_start_trace_with_session_id_in_options(self):
    options = jax.profiler.ProfileOptions()
    options.session_id = "options_session"
    profiling.start_trace("gs://test_bucket/test_dir", profiler_options=options)

    expected_request = self._get_expected_profile_request(
        "gs://test_bucket/test_dir",
        max_num_hosts=1,
        session_id="options_session",
    )
    self.mock_plugin_executable_cls.assert_called_once_with(
        json.dumps(expected_request)
    )
    self.assertEqual(options.session_id, "options_session")
    self.mock_original_start_trace.assert_called_once()
    call_args = self.mock_original_start_trace.call_args[1]
    self.assertEqual(call_args["log_dir"], "gs://test_bucket/test_dir")
    self.assertFalse(call_args["create_perfetto_link"])
    self.assertFalse(call_args["create_perfetto_trace"])
    self.assertEqual(
        call_args["profiler_options"].session_id, "options_session"
    )

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
    with self.assertRaisesRegex(RuntimeError, "trace is already being taken"):
      profiling.start_trace("gs://test_bucket/test_dir2")

  def test_stop_trace_success(self):
    profiling.start_trace("gs://test_bucket/test_dir")
    # call() is called once in start_trace, and once in stop_trace.
    with self.subTest("call_in_start_trace"):
      self.mock_plugin_executable_cls.return_value.call.assert_called_once()

    profiling.stop_trace()

    with self.subTest("call_count_after_stop_trace"):
      self.assertEqual(
          self.mock_plugin_executable_cls.return_value.call.call_count, 2
      )
    with self.subTest("original_stop_trace_called"):
      self.mock_original_stop_trace.assert_called_once()
    with self.subTest("executable_is_none"):
      self.assertIsNone(profiling._profile_state.executable)

  @absltest.skipIf(
      jax.version.__version_info__ < (0, 9, 2),
      "ProfileOptions requires JAX 0.9.2 or newer",
  )
  def test_stop_trace_with_xprof_options_passes_out_avals(self):
    options = jax.profiler.ProfileOptions()
    options.duration_ms = 2000

    request = profiling._create_profile_request(
        "gs://test_bucket/test_dir", options
    )
    profiling._profile_state.profile_request = request
    profiling._profile_state.executable = (
        self.mock_plugin_executable_cls.return_value
    )
    self.addCleanup(profiling._profile_state.reset)

    profiling.stop_trace()

    with self.subTest("plugin_executable_called"):
      self.mock_plugin_executable_cls.return_value.call.assert_called_once()
      _, kwargs = self.mock_plugin_executable_cls.return_value.call.call_args
      self.assertIn("out_avals", kwargs)
      self.assertIn("out_shardings", kwargs)

    with self.subTest("out_avals_properties"):
      _, kwargs = self.mock_plugin_executable_cls.return_value.call.call_args
      self.assertLen(kwargs["out_avals"], 1)
      (out_aval,) = kwargs["out_avals"]
      self.assertEqual(out_aval.shape, (1,))
      self.assertEqual(out_aval.dtype, jnp.object_)

  def test_stop_trace_before_start_warns_and_returns(self):
    with mock.patch.object(profiling._logger, "warning") as mock_warn:
      profiling.stop_trace()
      mock_warn.assert_called_once_with(
          "stop_trace called before a trace was started; ignoring."
      )

  def test_start_server_starts_thread(self):
    mock_thread = self.enter_context(
        mock.patch.object(profiling.threading, "Thread", autospec=True)
    )
    profiling.start_server(9000)
    mock_thread.assert_called_once_with(
        target=mock.ANY, args=(9000, "0.0.0.0", None)
    )
    mock_thread.return_value.start.assert_called_once()
    self.assertIsNotNone(profiling._profiler_thread)

  @parameterized.named_parameters(
      dict(testcase_name="unset", env_host=None, expected_host="0.0.0.0"),
      dict(testcase_name="empty", env_host="", expected_host=""),
      dict(
          testcase_name="all_ipv4",
          env_host="0.0.0.0",
          expected_host="0.0.0.0",
      ),
      dict(
          testcase_name="localhost",
          env_host="127.0.0.1",
          expected_host="127.0.0.1",
      ),
      dict(
          testcase_name="public_ip",
          env_host="192.15.2.4",
          expected_host="192.15.2.4",
      ),
      dict(
          testcase_name="private_ip",
          env_host="10.0.0.3",
          expected_host="10.0.0.3",
      ),
      dict(testcase_name="all_ipv6", env_host="[::]", expected_host="[::]"),
  )
  def test_start_server_host_env_var(
      self, env_host: str | None, expected_host: str
  ):
    mock_thread = self.enter_context(
        mock.patch.object(profiling.threading, "Thread", autospec=True)
    )
    env = dict(profiling.os.environ)
    if env_host is not None:
      env["PATHWAYS_PROFILING_SERVER_HOST"] = env_host
    else:
      env.pop("PATHWAYS_PROFILING_SERVER_HOST", None)

    with mock.patch.dict(profiling.os.environ, env, clear=True):
      profiling.start_server(9000)

    mock_thread.assert_called_once_with(
        target=mock.ANY, args=(9000, expected_host, None)
    )

  def test_start_server_twice_raises_error(self):
    self.enter_context(
        mock.patch.object(profiling.threading, "Thread", autospec=True)
    )
    profiling.start_server(9000)
    with self.assertRaisesRegex(
        RuntimeError, "Only one profiler server can be active"
    ):
      profiling.start_server(9001)

  def test_stop_server_no_server_raises_error(self):
    with self.assertRaisesRegex(RuntimeError, "No active profiler server"):
      profiling.stop_server()

  def test_stop_server_does_nothing_if_server_exists(self):
    self.enter_context(
        mock.patch.object(profiling.threading, "Thread", autospec=True)
    )
    profiling.start_server(9000)
    profiling.stop_server()  # Should not raise

  def _setup_monkey_patch(self):
    """Saves originals, applies monkey patch, and sets up mocks."""
    targets = [
        (jax.profiler, "start_trace"),
        (jax.profiler, "stop_trace"),
        (jax.profiler, "start_server"),
        (jax.profiler, "stop_server"),
        (jax._src.profiler, "start_trace"),
        (jax._src.profiler, "stop_trace"),
        (jax._src.profiler, "start_server"),
        (jax._src.profiler, "stop_server"),
    ]
    original_jax_funcs = {}
    for module, func_name in targets:
      original_func = getattr(module, func_name)
      original_jax_funcs[(module, func_name)] = original_func
      self.addCleanup(setattr, module, func_name, original_func)

    profiling.monkey_patch_jax()

    for module, func_name in targets:
      self.assertNotEqual(
          getattr(module, func_name),
          original_jax_funcs[(module, func_name)],
      )

    mocks = {
        "start_trace": self.enter_context(
            mock.patch.object(profiling, "start_trace", autospec=True)
        ),
        "stop_trace": self.enter_context(
            mock.patch.object(profiling, "stop_trace", autospec=True)
        ),
        "start_server": self.enter_context(
            mock.patch.object(profiling, "start_server", autospec=True)
        ),
        "stop_server": self.enter_context(
            mock.patch.object(profiling, "stop_server", autospec=True)
        ),
    }
    return mocks

  @parameterized.named_parameters(
      dict(testcase_name="jax_profiler", profiler_module=jax.profiler),
      dict(testcase_name="jax_src_profiler", profiler_module=jax._src.profiler),
  )
  def test_monkey_patched_start_trace(self, profiler_module):
    mocks = self._setup_monkey_patch()

    profiler_module.start_trace("gs://bucket/dir")

    mocks["start_trace"].assert_called_once_with(
        "gs://bucket/dir",
        create_perfetto_link=False,
        create_perfetto_trace=False,
        profiler_options=None,
        max_num_hosts=1,
    )

  @parameterized.named_parameters(
      dict(testcase_name="jax_profiler", profiler_module=jax.profiler),
      dict(testcase_name="jax_src_profiler", profiler_module=jax._src.profiler),
  )
  def test_monkey_patched_start_trace_with_max_num_hosts(self, profiler_module):
    mocks = self._setup_monkey_patch()

    profiler_module.start_trace("gs://bucket/dir", max_num_hosts=3)

    mocks["start_trace"].assert_called_once_with(
        "gs://bucket/dir",
        create_perfetto_link=False,
        create_perfetto_trace=False,
        profiler_options=None,
        max_num_hosts=3,
    )

  @parameterized.named_parameters(
      dict(testcase_name="jax_profiler", profiler_module=jax.profiler),
      dict(testcase_name="jax_src_profiler", profiler_module=jax._src.profiler),
  )
  def test_monkey_patched_stop_trace(self, profiler_module):
    mocks = self._setup_monkey_patch()

    profiler_module.stop_trace()

    mocks["stop_trace"].assert_called_once()

  @parameterized.named_parameters(
      dict(testcase_name="jax_profiler", profiler_module=jax.profiler),
      dict(testcase_name="jax_src_profiler", profiler_module=jax._src.profiler),
  )
  def test_monkey_patched_start_server(self, profiler_module):
    mocks = self._setup_monkey_patch()

    profiler_module.start_server(1234, requires_backend=False)

    mocks["start_server"].assert_called_once_with(
        1234,
        requires_backend=False,
    )

  @parameterized.named_parameters(
      dict(testcase_name="jax_profiler", profiler_module=jax.profiler),
      dict(testcase_name="jax_src_profiler", profiler_module=jax._src.profiler),
  )
  def test_monkey_patched_stop_server(self, profiler_module):
    mocks = self._setup_monkey_patch()

    profiler_module.stop_server()

    mocks["stop_server"].assert_called_once()

  @parameterized.parameters(None, jax.profiler.ProfileOptions())
  def test_create_profile_request_default_options(self, profiler_options):
    request = profiling._create_profile_request(
        "gs://bucket/dir", profiler_options=profiler_options
    )
    self.assertEqual(
        request,
        {
            "traceLocation": "gs://bucket/dir",
            "maxNumHosts": 1,
        },
    )

  def test_create_profile_request_with_max_num_hosts(self):
    request = profiling._create_profile_request(
        "gs://bucket/dir", max_num_hosts=5
    )
    self.assertEqual(
        request,
        {
            "traceLocation": "gs://bucket/dir",
            "maxNumHosts": 5,
        },
    )

  @absltest.skipIf(
      jax.version.__version_info__ < (0, 9, 2),
      "ProfileOptions requires JAX 0.9.2 or newer",
  )
  def test_create_profile_request_with_options(self):
    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = 2
    options.python_tracer_level = 1
    options.duration_ms = 2000
    options.start_timestamp_ns = 123456789
    options.session_id = "test_session"
    options.advanced_configuration = {
        "tpu_num_chips_to_profile_per_task": 3,
        "tpu_num_sparse_core_tiles_to_trace": 5,
        "tpu_trace_mode": "TRACE_COMPUTE",
        "tpu_num_sparse_cores_to_trace": 1,
        "tpu_enable_flag": True,
    }

    request = profiling._create_profile_request(
        "gs://bucket/dir", profiler_options=options
    )
    self.assertEqual(
        request,
        {
            "traceLocation": "gs://bucket/dir",
            "maxDurationSecs": 2.0,
            "maxNumHosts": 1,
            "xprofTraceOptions": {
                "traceDirectory": "gs://bucket/dir",
                "traceSessionName": "test_session",
                "pwTraceOptions": {
                    "enablePythonTracer": True,
                    "advancedConfiguration": {
                        "tpu_num_chips_to_profile_per_task": {"int64Value": 3},
                        "tpu_num_sparse_core_tiles_to_trace": {"int64Value": 5},
                        "tpu_trace_mode": {"stringValue": "TRACE_COMPUTE"},
                        "tpu_num_sparse_cores_to_trace": {"int64Value": 1},
                        "tpu_enable_flag": {"boolValue": True},
                    },
                },
            },
        },
    )

  @absltest.skipIf(
      jax.version.__version_info__ < (0, 9, 2),
      "ProfileOptions requires JAX 0.9.2 or newer",
  )
  @parameterized.parameters(
      ({"traceLocation": "gs://test_bucket/test_dir"},),
      (
          {
              "traceLocation": "gs://test_bucket/test_dir",
              "blockUntilStart": True,
              "maxDurationSecs": 10.0,
              "devices": {"deviceIds": [1, 2]},
              "includeResourceManagers": True,
              "maxNumHosts": 5,
              "xprofTraceOptions": {
                  "blockUntilStart": True,
                  "traceDirectory": "gs://test_bucket/test_dir",
              },
          },
      ),
      (
          {
              "traceLocation": "gs://bucket/dir",
              "xprofTraceOptions": {
                  "hostTraceLevel": 0,
                  "traceOptions": {
                      "traceMode": "TRACE_COMPUTE",
                      "numSparseCoresToTrace": 1,
                      "numSparseCoreTilesToTrace": 2,
                      "numChipsToProfilePerTask": 3,
                      "powerTraceLevel": 4,
                      "enableFwThrottleEvent": True,
                      "enableFwPowerLevelEvent": True,
                      "enableFwThermalEvent": True,
                  },
                  "traceDirectory": "gs://bucket/dir",
              },
          },
      ),
  )
  def test_start_pathways_trace_from_profile_request(self, profile_request):
    profiling._start_pathways_trace_from_profile_request(profile_request)

    self.mock_toy_computation.assert_called_once()
    self.mock_plugin_executable_cls.assert_called_once_with(
        json.dumps({"profileRequest": profile_request})
    )
    self.mock_plugin_executable_cls.return_value.call.assert_called_once()
    self.mock_original_start_trace.assert_not_called()
    self.assertIsNotNone(profiling._profile_state.executable)

  def test_original_stop_trace_called_on_stop_failure(self):
    """Tests that original_stop_trace is called if pathways stop_trace fails."""
    profiling.start_trace("gs://test_bucket/test_dir")
    self.assertFalse(profiling._profile_state.lock.locked())
    self.mock_plugin_executable_cls.return_value.call.side_effect = (
        RuntimeError("stop failed")
    )
    with self.assertRaisesRegex(RuntimeError, "stop failed"):
      profiling.stop_trace()
    self.mock_original_stop_trace.assert_called_once()

  def test_jax_profiler_trace_calls_patched_functions(self):
    mocks = self._setup_monkey_patch()

    with jax.profiler.trace("gs://bucket/dir"):
      pass

    mocks["start_trace"].assert_called_once()
    mocks["stop_trace"].assert_called_once()

  @absltest.skipIf(
      jax.version.__version_info__ < (0, 9, 2),
      "ProfileOptions requires JAX 0.9.2 or newer",
  )
  def test_is_default_profile_options_with_session_id(self):
    options = jax.profiler.ProfileOptions()
    options.session_id = "test_session"
    self.assertFalse(profiling._is_default_profile_options(options))

  @absltest.skipIf(
      jax.version.__version_info__ < (0, 9, 2),
      "ProfileOptions requires JAX 0.9.2 or newer",
  )
  def test_start_trace_compatibility_error(self):
    self.mock_plugin_executable_cls.side_effect = RuntimeError(
        "Bad PluginProgram"
    )

    options = jax.profiler.ProfileOptions()
    options.session_id = "test_session"

    with self.assertRaisesRegex(
        RuntimeError,
        "likely because the running Pathways server images do not support the"
        " trace session ID option",
    ):
      profiling.start_trace(
          "gs://test_bucket/test_dir", profiler_options=options
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="allowed_bucket",
          log_dir="gs://bucket1/dir",
      ),
  )
  def test_validate_gcs_bucket_env_var_allowed(self, log_dir: str):
    with mock.patch.dict(
        profiling.os.environ,
        {"PATHWAYS_PROFILING_ALLOWED_GCS_BUCKETS": "gs://bucket1,gs://bucket2"},
    ):
      profiling._validate_gcs_bucket(log_dir)

  @parameterized.named_parameters(
      dict(
          testcase_name="disallowed_bucket",
          log_dir="gs://bucket3/dir",
      ),
  )
  def test_validate_gcs_bucket_env_var_disallowed(self, log_dir: str):
    with mock.patch.dict(
        profiling.os.environ,
        {"PATHWAYS_PROFILING_ALLOWED_GCS_BUCKETS": "gs://bucket1,gs://bucket2"},
    ):
      with self.assertRaisesRegex(ValueError, "is not in allowed buckets list"):
        profiling._validate_gcs_bucket(log_dir)

  def test_start_trace_rollback_on_original_failure(self):
    self.mock_original_start_trace.side_effect = RuntimeError(
        "original start trace error"
    )
    with self.assertRaisesRegex(RuntimeError, "original start trace error"):
      profiling.start_trace("gs://test_bucket/test_dir")

    self.assertIsNone(profiling._profile_state.executable)
    self.assertFalse(profiling._profile_state.lock.locked())

  def test_collect_profile_without_auth_token(self):
    env = dict(profiling.os.environ)
    env.pop("PATHWAYS_PROFILING_AUTH_TOKEN", None)

    with mock.patch.dict(profiling.os.environ, env, clear=True):
      result = profiling.collect_profile(
          port=8000,
          duration_ms=1000,
          host="127.0.0.1",
          log_dir="gs://test_bucket/test_dir",
      )

      self.assertTrue(result)
      self.mock_post.assert_called_once_with(
          "http://127.0.0.1:8000/profiling",
          json={
              "duration_ms": 1000,
              "repository_path": "gs://test_bucket/test_dir",
          },
          headers={},
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="valid_token",
          env_token="secret_token",
          expected_result=True,
          http_error=None,
      ),
      dict(
          testcase_name="http_error",
          env_token="wrong_token",
          expected_result=False,
          http_error=requests.exceptions.HTTPError(
              "401 Client Error: Unauthorized"
          ),
      ),
  )
  def test_collect_profile_with_auth_token(
      self,
      env_token: str,
      expected_result: bool,
      http_error: Exception | None,
  ):
    env = dict(profiling.os.environ)
    env["PATHWAYS_PROFILING_AUTH_TOKEN"] = env_token

    if http_error:
      self.mock_post.return_value.raise_for_status.side_effect = http_error

    with mock.patch.dict(profiling.os.environ, env, clear=True):
      result = profiling.collect_profile(
          port=8000,
          duration_ms=1000,
          host="127.0.0.1",
          log_dir="gs://test_bucket/test_dir",
      )

      self.assertEqual(result, expected_result)
      self.mock_post.assert_called_once_with(
          "http://127.0.0.1:8000/profiling",
          json={
              "duration_ms": 1000,
              "repository_path": "gs://test_bucket/test_dir",
          },
          headers={"Authorization": f"Bearer {env_token}"},
      )

  def _get_server_app(self) -> Any:
    with (
        mock.patch.object(profiling.threading, "Thread") as mock_thread,
        mock.patch.object(profiling.uvicorn, "run") as mock_uvicorn,
    ):
      profiling.start_server(9000)
      server_loop_fn = mock_thread.call_args[1]["target"]
      args = mock_thread.call_args[1]["args"]
      server_loop_fn(*args)
      return mock_uvicorn.call_args[0][0]

  @parameterized.named_parameters(
      dict(
          testcase_name="valid_token",
          server_token="secret_token",
          request_headers={"Authorization": "Bearer secret_token"},
          expected_status=200,
          expected_detail_substring=None,
      ),
      dict(
          testcase_name="missing_token",
          server_token="secret_token",
          request_headers=None,
          expected_status=401,
          expected_detail_substring=(
              "Unauthorized: invalid or missing authentication token"
          ),
      ),
      dict(
          testcase_name="wrong_token",
          server_token="secret_token",
          request_headers={"Authorization": "Bearer invalid_token"},
          expected_status=401,
          expected_detail_substring=(
              "Unauthorized: invalid or missing authentication token"
          ),
      ),
      dict(
          testcase_name="token_when_not_needed",
          server_token=None,
          request_headers={"Authorization": "Bearer unneeded_token"},
          expected_status=200,
          expected_detail_substring=None,
      ),
      dict(
          testcase_name="no_token_when_not_needed",
          server_token=None,
          request_headers=None,
          expected_status=200,
          expected_detail_substring=None,
      ),
  )
  @unittest.skipIf(
      os.environ.get("GITHUB_ACTIONS") == "true",
      "Skipping FastAPI server test in GitHub CI",
  )
  def test_server_auth(
      self,
      server_token: str | None,
      request_headers: dict[str, str] | None,
      expected_status: int,
      expected_detail_substring: str | None,
  ):
    from fastapi import testclient
    env = dict(profiling.os.environ)
    if server_token is not None:
      env["PATHWAYS_PROFILING_AUTH_TOKEN"] = server_token
    else:
      env.pop("PATHWAYS_PROFILING_AUTH_TOKEN", None)

    with (
        mock.patch.dict(profiling.os.environ, env, clear=True),
        mock.patch.object(profiling, "start_trace"),
        mock.patch.object(profiling, "stop_trace"),
        mock.patch.object(profiling.asyncio, "sleep"),
    ):
      app = self._get_server_app()
      client = testclient.TestClient(app)
      response = client.post(
          "/profiling",
          json={"duration_ms": 100, "repository_path": "gs://test_bucket/dir"},
          headers=request_headers,
      )
      self.assertEqual(response.status_code, expected_status)
      if expected_status == 200:
        self.assertEqual(response.json(), {"response": "profiling completed"})
      if expected_detail_substring:
        self.assertIn(expected_detail_substring, response.json()["detail"])


if __name__ == "__main__":
  absltest.main()
