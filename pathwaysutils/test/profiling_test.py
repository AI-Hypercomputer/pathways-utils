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

from unittest import mock

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

  @parameterized.parameters(8000, 1234)
  def test_collect_profile_port(self, port):
    profiling.collect_profile(
        port=port,
        duration_ms=1000,
        host="127.0.0.1",
        log_dir="gs://test_bucket/test_dir",
    )

    self.mock_post.assert_called_once_with(
        f"http://127.0.0.1:{port}/profiling",
        json={
            "duration_ms": 1000,
            "repository_path": "gs://test_bucket/test_dir",
        },
    )

  @parameterized.parameters(1000, 1234)
  def test_collect_profile_duration_ms(self, duration_ms):
    profiling.collect_profile(
        port=8000,
        duration_ms=duration_ms,
        host="127.0.0.1",
        log_dir="gs://test_bucket/test_dir",
    )

    self.mock_post.assert_called_once_with(
        "http://127.0.0.1:8000/profiling",
        json={
            "duration_ms": duration_ms,
            "repository_path": "gs://test_bucket/test_dir",
        },
    )

  @parameterized.parameters("127.0.0.1", "localhost", "192.168.1.1")
  def test_collect_profile_host(self, host):
    profiling.collect_profile(
        port=8000,
        duration_ms=1000,
        host=host,
        log_dir="gs://test_bucket/test_dir",
    )

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
    profiling.collect_profile(
        port=8000, duration_ms=1000, host="127.0.0.1", log_dir=log_dir
    )

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
      requests.exceptions.ConnectionError,
      requests.exceptions.Timeout,
      requests.exceptions.TooManyRedirects,
      requests.exceptions.RequestException,
      requests.exceptions.HTTPError,
  )
  def test_collect_profile_request_error(self, exception_type):
    self.mock_post.side_effect = exception_type

    result = profiling.collect_profile(
        port=8000,
        duration_ms=1000,
        host="127.0.0.1",
        log_dir="gs://test_bucket/test_dir",
    )

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


if __name__ == "__main__":
  absltest.main()
