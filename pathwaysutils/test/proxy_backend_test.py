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
"""Tests for the proxy backend module."""

import os
from unittest import mock

from absl.testing import absltest
import jax
from jax.extend import backend
from jax.extend.backend import ifrt_proxy
from pathwaysutils import proxy_backend


class ProxyBackendTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    orig_jax_platforms = getattr(jax.config, "jax_platforms", None)
    orig_jax_backend_target = getattr(jax.config, "jax_backend_target", None)

    self.addCleanup(jax.config.update, "jax_platforms", orig_jax_platforms)
    self.addCleanup(
        jax.config.update, "jax_backend_target", orig_jax_backend_target
    )
    self.addCleanup(backend.clear_backends)

    jax.config.update("jax_platforms", "proxy")
    jax.config.update("jax_backend_target", "grpc://localhost:12345")
    backend.clear_backends()

  @absltest.skip("b/408025233")
  def test_no_proxy_backend_registration_raises_error(self):
    self.assertRaises(RuntimeError, backend.backends)

  def test_proxy_backend_registration(self):
    self.enter_context(
        mock.patch.object(
            ifrt_proxy,
            "get_client",
            return_value=mock.MagicMock(),
        )
    )
    proxy_backend.register_backend_factory()
    self.assertIn("proxy", backend.backends())

  def test_proxy_backend_registration_with_timeout(self):
    mock_get_client = self.enter_context(
        mock.patch.object(
            ifrt_proxy,
            "get_client",
            return_value=mock.MagicMock(),
        )
    )
    self.enter_context(
        mock.patch.dict(
            os.environ, {"PATHWAYS_PROXY_CONNECTION_TIMEOUT_SECS": "42"}
        )
    )
    proxy_backend.register_backend_factory()
    self.assertIn("proxy", backend.backends())
    mock_get_client.assert_called_once()
    args, _ = mock_get_client.call_args
    self.assertEqual(args[0], "grpc://localhost:12345")
    options = args[1]
    self.assertEqual(options.connection_timeout_in_seconds, 42)

  def test_proxy_backend_registration_without_timeout(self):
    mock_get_client = self.enter_context(
        mock.patch.object(
            ifrt_proxy,
            "get_client",
            return_value=mock.MagicMock(),
        )
    )
    self.enter_context(mock.patch.dict(os.environ))
    os.environ.pop("PATHWAYS_PROXY_CONNECTION_TIMEOUT_SECS", None)

    proxy_backend.register_backend_factory()
    self.assertIn("proxy", backend.backends())
    mock_get_client.assert_called_once()
    args, _ = mock_get_client.call_args
    self.assertEqual(args[0], "grpc://localhost:12345")
    options = args[1]
    self.assertNotEqual(options.connection_timeout_in_seconds, 42)


if __name__ == "__main__":
  absltest.main()
