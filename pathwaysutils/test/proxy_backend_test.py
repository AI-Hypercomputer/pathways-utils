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
import sys # Added for sys.modules manipulation
from jax.extend import backend
# Note: We don't import ifrt_proxy directly from jax.lib.xla_extension here anymore in the tests
# as we want to test the dynamic import logic within proxy_backend.py
from pathwaysutils import proxy_backend

from absl.testing import absltest


class ProxyBackendTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Ensure a clean slate for jax config and backend states for each test.
    jax.config.update("jax_platforms", "") # Reset platforms
    jax.config.update("jax_backend_target", "grpc://localhost:12345")
    backend.clear_backends()
    # Unload the proxy_backend module to ensure its import logic is re-evaluated for each test case
    # This is crucial for testing the try-except import block.
    if "pathwaysutils.proxy_backend" in sys.modules:
      del sys.modules["pathwaysutils.proxy_backend"]

  def tearDown(self):
    super().tearDown()
    # Reload the module to leave it in a predictable state for other tests, if any.
    # and reset jax configurations
    if "pathwaysutils.proxy_backend" not in sys.modules:
        # Import it using a known path for cleanup if it was deleted.
        # This step might need adjustment based on how tests are structured overall.
        from pathwaysutils import proxy_backend as pb_module
        globals()['proxy_backend'] = pb_module
    jax.config.update("jax_platforms", "")
    jax.config.update("jax_backend_target", "")
    backend.clear_backends()


  @absltest.skip("b/408025233")
  def test_no_proxy_backend_registration_raises_error(self):
    # This test might need re-evaluation based on how proxy_backend is loaded now.
    # For now, we ensure proxy_backend is loaded before testing.
    from pathwaysutils import proxy_backend as pb_reloaded_for_test
    self.assertRaises(RuntimeError, backend.backends)

  def test_proxy_backend_registration_fallback_path(self):
    """Tests registration when ifrt_proxy is loaded from the old path."""
    # Simulate that the new path (jax._src.lib._jax) is not available
    with mock.patch.dict(sys.modules, {"jax._src.lib": None}):
        if "pathwaysutils.proxy_backend" in sys.modules: # Ensure it's reloaded
            del sys.modules["pathwaysutils.proxy_backend"]
        from pathwaysutils import proxy_backend as pb_fallback

        # Mock jax.lib.xla_extension.ifrt_proxy for the fallback path
        mock_ifrt_proxy_old = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"jax.lib.xla_extension.ifrt_proxy": mock_ifrt_proxy_old}):
            # Ensure the proxy_backend module uses this mocked old ifrt_proxy
            # We need to make sure that when proxy_backend.ifrt_proxy is accessed, it's our mock
            # This is tricky because the import happens at module load time.
            # The deletion of proxy_backend from sys.modules in setUp helps here.

            # Re-import proxy_backend so it picks up the mocked jax.lib.xla_extension.ifrt_proxy
            if "pathwaysutils.proxy_backend" in sys.modules:
                del sys.modules["pathwaysutils.proxy_backend"]
            from pathwaysutils import proxy_backend as pb_reloaded

            # Check if the fallback was used by verifying which mock is present
            # This requires ensuring pb_reloaded.ifrt_proxy IS mock_ifrt_proxy_old
            # However, direct access like that might be tricky if it's not explicitly assigned.
            # Instead, we mock the get_client on the expected ifrt_proxy object.

            # To ensure the correct ifrt_proxy is mocked, we patch it where it's defined by the fallback.
            # The actual ifrt_proxy object is obtained after module reload, then its get_client is mocked.
            if "pathwaysutils.proxy_backend" in sys.modules: # Ensure it's reloaded under these mocks
                del sys.modules["pathwaysutils.proxy_backend"]
            from pathwaysutils import proxy_backend as pb_final_fallback

            # pb_final_fallback.ifrt_proxy should now point to the one from jax.lib.xla_extension
            # due to the outer mocks. We mock its get_client method.
            with mock.patch.object(pb_final_fallback.ifrt_proxy, "get_client", return_value=mock.MagicMock()) as mock_get_client_old:
                jax.config.update("jax_platforms", "proxy") # Set for this test
                pb_final_fallback.register_backend_factory()
                self.assertIn("proxy", backend.backends())
                mock_get_client_old.assert_called_once()
        # Clean up jax_platforms config
        jax.config.update("jax_platforms", "")
        backend.clear_backends()


  def test_proxy_backend_registration_new_path(self):
    """Tests registration when ifrt_proxy is loaded from the new path."""
    mock_ifrt_proxy_new = mock.MagicMock()
    mock_jax_src_lib = mock.MagicMock()
    mock_jax_src_lib._jax.ifrt_proxy = mock_ifrt_proxy_new

    with mock.patch.dict(sys.modules, {"jax._src.lib": mock_jax_src_lib}):
        # Ensure the module is reloaded to pick up the new mock
        if "pathwaysutils.proxy_backend" in sys.modules:
            del sys.modules["pathwaysutils.proxy_backend"]
        from pathwaysutils import proxy_backend as pb_new_path

        # pb_new_path.ifrt_proxy should be mock_ifrt_proxy_new due to the outer mocks.
        # We mock its get_client method.
        with mock.patch.object(pb_new_path.ifrt_proxy, "get_client", return_value=mock.MagicMock()) as mock_get_client_new:
            jax.config.update("jax_platforms", "proxy") # Set for this test
            pb_new_path.register_backend_factory()
            self.assertIn("proxy", backend.backends())
            mock_get_client_new.assert_called_once() # Check the mock on the instance

        # Clean up jax_platforms config
        jax.config.update("jax_platforms", "")
        backend.clear_backends()

  # Keep the original test_proxy_backend_registration but adapt it or ensure it still makes sense.
  # For now, let's assume the dynamic loading tests cover the core logic sufficiently.
  # We might need to remove or adjust the original test_proxy_backend_registration
  # if it becomes redundant or conflicts with the module reloading strategy.

  # The original test_proxy_backend_registration might fail due to module reloading.
  # Let's adapt it to also reload the module to ensure it uses a fresh import.
  def test_proxy_backend_registration_original_adapted(self):
    # This test will now effectively test the fallback path by default if jax._src.lib is not mocked.
    if "pathwaysutils.proxy_backend" in sys.modules:
        del sys.modules["pathwaysutils.proxy_backend"]

    # Mock the old path directly as this test doesn't distinguish new/old explicitly
    # but relies on the default import behavior (which would be fallback if new path fails)
    with mock.patch("jax.lib.xla_extension.ifrt_proxy.get_client", return_value=mock.MagicMock()) as mock_get_client:
        # Reload the module under the mock
        from pathwaysutils import proxy_backend as pb_adapted
        jax.config.update("jax_platforms", "proxy")
        pb_adapted.register_backend_factory()
        self.assertIn("proxy", backend.backends())
        mock_get_client.assert_called_once()
        jax.config.update("jax_platforms", "")
        backend.clear_backends()


if __name__ == "__main__":
  absltest.main()
