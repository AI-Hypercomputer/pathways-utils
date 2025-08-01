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

from unittest import mock

import jax
from jax.extend import backend
from jax.lib.xla_extension import ifrt_proxy
from pathwaysutils import proxy_backend


from absl.testing import absltest


class ProxyBackendTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
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


if __name__ == "__main__":
  absltest.main()
