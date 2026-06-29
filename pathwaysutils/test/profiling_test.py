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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import grpc
import jax
from jax import numpy as jnp
from pathwaysutils import profiling
from tensorflow.core.profiler.protobuf import profiler_service_pb2  # pylint: disable=g-direct-tensorflow-import
from tensorflow.core.profiler.protobuf import profiler_service_pb2_grpc  # pylint: disable=g-direct-tensorflow-import


class _MockRpcError(grpc.RpcError):

  def __init__(self, code, details=""):
    self._code = code
    self._details = details

  def code(self):
    return self._code

  def details(self):
    return self._details


class ProfilingTest(parameterized.TestCase):
  """Tests for Pathways on Cloud profiling."""

  def setUp(self):
    super().setUp()
    # Mock grpc channel and server
    self.mock_secure_channel = self.enter_context(
        mock.patch.object(grpc, "secure_channel", autospec=True)
    )
    self.mock_grpc_server = self.enter_context(
        mock.patch.object(grpc, "server", autospec=True)
    )
    self.mock_alts_server_creds = self.enter_context(
        mock.patch.object(grpc, "alts_server_credentials", autospec=True)
    )
    self.mock_alts_channel_creds = self.enter_context(
        mock.patch.object(grpc, "alts_channel_credentials", autospec=True)
    )

    # Mock the gRPC stub
    self.mock_stub_cls = self.enter_context(
        mock.patch.object(
            profiler_service_pb2_grpc,
            "ProfilerServiceStub",
        )
    )
    self.mock_stub = self.mock_stub_cls.return_value

    profiling._profile_state.reset()
    profiling._first_profile_start = True
    profiling._profiler_server = None

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
    self.mock_stub.Profile.return_value = profiler_service_pb2.ProfileResponse()
    result = profiling.collect_profile(
        port=port,
        duration_ms=1000,
        host="127.0.0.1",
        log_dir="gs://test_bucket/test_dir",
    )

    self.assertTrue(result)
    self.mock_stub.Profile.assert_called_once_with(
        profiler_service_pb2.ProfileRequest(
            duration_ms=1000,
            repository_root="gs://test_bucket/test_dir",
        ),
        timeout=11.0,
    )

  @parameterized.parameters(1000, 1234)
  def test_collect_profile_duration_ms(self, duration_ms):
    self.mock_stub.Profile.return_value = profiler_service_pb2.ProfileResponse()
    result = profiling.collect_profile(
        port=8000,
        duration_ms=duration_ms,
        host="127.0.0.1",
        log_dir="gs://test_bucket/test_dir",
    )

    self.assertTrue(result)
    self.mock_stub.Profile.assert_called_once_with(
        profiler_service_pb2.ProfileRequest(
            duration_ms=duration_ms,
            repository_root="gs://test_bucket/test_dir",
        ),
        timeout=(duration_ms / 1000.0) + 10.0,
    )

  @parameterized.parameters("127.0.0.1", "localhost", "192.168.1.1")
  def test_collect_profile_host(self, host):
    self.mock_stub.Profile.return_value = profiler_service_pb2.ProfileResponse()
    result = profiling.collect_profile(
        port=8000,
        duration_ms=1000,
        host=host,
        log_dir="gs://test_bucket/test_dir",
    )

    self.assertTrue(result)
    self.mock_stub.Profile.assert_called_once_with(
        profiler_service_pb2.ProfileRequest(
            duration_ms=1000,
            repository_root="gs://test_bucket/test_dir",
        ),
        timeout=11.0,
    )

  @parameterized.parameters(
      "gs://test_bucket/test_log_dir",
      "gs://test_bucket2",
      "gs://test_bucket3/test/log/dir",
  )
  def test_collect_profile_log_dir(self, log_dir):
    self.mock_stub.Profile.return_value = profiler_service_pb2.ProfileResponse()
    result = profiling.collect_profile(
        port=8000, duration_ms=1000, host="127.0.0.1", log_dir=log_dir
    )

    self.assertTrue(result)
    self.mock_stub.Profile.assert_called_once_with(
        profiler_service_pb2.ProfileRequest(
            duration_ms=1000,
            repository_root=log_dir,
        ),
        timeout=11.0,
    )

  @parameterized.parameters("/logs/test_log_dir", "relative_path/my_log_dir")
  def test_collect_profile_log_dir_error(self, log_dir):
    with self.assertRaises(ValueError):
      profiling.collect_profile(
          port=8000, duration_ms=1000, host="127.0.0.1", log_dir=log_dir
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="unavailable",
          status_code=grpc.StatusCode.UNAVAILABLE,
          expected_log=(
              "Failed to connect to the profiling server at 127.0.0.1:8000."
          ),
      ),
      dict(
          testcase_name="timeout",
          status_code=grpc.StatusCode.DEADLINE_EXCEEDED,
          expected_log=(
              "Profiling request timed out. The server might be unresponsive."
          ),
      ),
      dict(
          testcase_name="other_error",
          status_code=grpc.StatusCode.INTERNAL,
          expected_log=(
              "gRPC error occurred while collecting profile. "
              "Error code: StatusCode.INTERNAL"
          ),
      ),
  )
  def test_collect_profile_request_error(self, status_code, expected_log):
    self.mock_stub.Profile.side_effect = _MockRpcError(
        status_code, "Error details"
    )

    with self.assertLogs(profiling._logger, level=logging.ERROR) as logs:
      result = profiling.collect_profile(
          port=8000,
          duration_ms=1000,
          host="127.0.0.1",
          log_dir="gs://test_bucket/test_dir",
      )

    self.assertLen(logs.output, 1)
    self.assertIn(expected_log, logs.output[0])
    self.assertFalse(result)
    self.mock_stub.Profile.assert_called_once()

  def test_collect_profile_use_alts(self):
    self.mock_stub.Profile.return_value = profiler_service_pb2.ProfileResponse()
    result = profiling.collect_profile(
        port=8000,
        duration_ms=1000,
        host="127.0.0.1",
        log_dir="gs://test_bucket/test_dir",
        use_alts=True,
    )
    self.assertTrue(result)
    self.mock_secure_channel.assert_called_once_with(
        "127.0.0.1:8000", self.mock_alts_channel_creds.return_value
    )
    self.mock_stub.Profile.assert_called_once()

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
    with self.assertRaisesRegex(
        RuntimeError, "start failed"
    ), mock.patch.object(profiling._logger, "exception"):
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
    profiling.start_trace(
        "gs://test_bucket/test_dir", profiler_options=options
    )

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

  def test_stop_trace_before_start_error(self):
    with self.assertRaisesRegex(
        RuntimeError, "stop_trace called before a trace is being taken!"
    ):
      profiling.stop_trace()

  def test_start_server_starts_grpc_server(self):
    mock_server = self.mock_grpc_server.return_value
    profiling.start_server(9000)

    self.mock_grpc_server.assert_called_once()
    mock_server.add_secure_port.assert_called_once_with(
        "[::]:9000", self.mock_alts_server_creds.return_value
    )
    mock_server.start.assert_called_once()
    self.assertEqual(profiling._profiler_server, mock_server)

  def test_start_server_twice_raises_error(self):
    profiling.start_server(9000)
    with self.assertRaisesRegex(
        RuntimeError, "Only one profiler server can be active"
    ):
      profiling.start_server(9001)

  def test_stop_server_no_server_raises_error(self):
    with self.assertRaisesRegex(RuntimeError, "No active profiler server"):
      profiling.stop_server()

  def test_stop_server_stops_server(self):
    mock_server = self.mock_grpc_server.return_value
    profiling.start_server(9000)
    profiling.stop_server()
    mock_server.stop.assert_called_once_with(grace=5.0)
    self.assertIsNone(profiling._profiler_server)

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

    mocks["start_server"].assert_called_once_with(1234, requires_backend=False)

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
                        "tpu_num_chips_to_profile_per_task": {
                            "int64Value": 3
                        },
                        "tpu_num_sparse_core_tiles_to_trace": {
                            "int64Value": 5
                        },
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

  def test_servicer_profile_success(self):
    with (
        mock.patch.object(
            jax.profiler, "start_trace", autospec=True
        ) as mock_start_trace,
        mock.patch.object(
            jax.profiler, "stop_trace", autospec=True
        ) as mock_stop_trace,
    ):
      servicer = profiling.PathwaysProfilerServicer()
      request = profiler_service_pb2.ProfileRequest(
          duration_ms=100, repository_root="gs://test_bucket/test_dir"
      )
      mock_context = mock.create_autospec(grpc.ServicerContext, instance=True)

      response = servicer.Profile(request, mock_context)

      self.assertIsInstance(response, profiler_service_pb2.ProfileResponse)
      mock_start_trace.assert_called_once_with("gs://test_bucket/test_dir")
      mock_stop_trace.assert_called_once()

  def test_servicer_profile_failure(self):
    with (
        mock.patch.object(
            jax.profiler, "start_trace", autospec=True
        ) as mock_start_trace,
        mock.patch.object(
            jax.profiler, "stop_trace", autospec=True
        ) as mock_stop_trace,
    ):
      servicer = profiling.PathwaysProfilerServicer()
      request = profiler_service_pb2.ProfileRequest(
          duration_ms=100, repository_root="gs://test_bucket/test_dir"
      )
      mock_context = mock.create_autospec(grpc.ServicerContext, instance=True)
      mock_start_trace.side_effect = RuntimeError("start failed")

      with self.assertRaisesRegex(RuntimeError, "start failed"):
        servicer.Profile(request, mock_context)

      mock_stop_trace.assert_called_once()


if __name__ == "__main__":
  absltest.main()
