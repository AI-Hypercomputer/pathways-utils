"""Tests for remote_ide."""

import argparse
import os
from unittest import mock

from absl.testing import absltest
from pathwaysutils.experimental.remote_ide import remote_ide

_ORIGINAL_LOAD_SCRIPT = remote_ide.load_script


class RemoteIdeTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Mock methods that interact with Kubernetes/subprocesses
    self.mock_find_pod = self.enter_context(
        mock.patch.object(remote_ide, "find_pod", return_value="dummy-pod")
    )
    self.mock_process = self.enter_context(
        mock.patch("multiprocessing.Process")
    )
    self.mock_load_script = self.enter_context(
        mock.patch.object(
            remote_ide, "load_script", return_value=["bash", "-c", "echo"]
        )
    )
    self.mock_load_k8s_config = self.enter_context(
        mock.patch.object(remote_ide, "load_k8s_config")
    )
    self.mock_core_v1_api = self.enter_context(
        mock.patch.object(remote_ide.client, "CoreV1Api")
    )
    self.mock_stream = self.enter_context(
        mock.patch(
            "pathwaysutils.experimental.remote_ide.remote_ide.stream.stream"
        )
    )
    # Mock stream response to exit loop immediately
    self.mock_stream_instance = mock.Mock()
    self.mock_stream_instance.is_open.return_value = False
    self.mock_stream.return_value = self.mock_stream_instance

  @mock.patch.dict(os.environ, {"USER": "testuser"})
  def test_main_default_pathways(self):
    mock_args = argparse.Namespace(
        mode="jupyter",
        session="testuser",
        port=8888,
        bucket="",
        check_active_session=False,
        sps=False,
        sps_image="my-image",
        instance_type="c4-standard-192",
    )
    with mock.patch.object(remote_ide, "get_args", return_value=mock_args):
      remote_ide.main()

    self.mock_find_pod.assert_called_once_with(
        "testuser-pathways-head", exit_on_fail=True
    )
    self.mock_stream.assert_called_once()
    kwargs = self.mock_stream.call_args[1]
    self.assertEqual(kwargs["container"], "jax-tpu")

  @mock.patch.dict(os.environ, {"USER": "testuser"})
  def test_main_sps_default_workload(self):
    mock_args = argparse.Namespace(
        mode="jupyter",
        session="testuser",
        port=8888,
        bucket="",
        check_active_session=False,
        sps=True,
        sps_image="my-image",
        instance_type="c4-standard-192",
    )
    with mock.patch.object(remote_ide, "get_args", return_value=mock_args):
      remote_ide.main()

    self.mock_find_pod.assert_called_once_with(
        "testuser", exit_on_fail=False
    )
    self.mock_stream.assert_called_once()
    kwargs = self.mock_stream.call_args[1]
    self.assertEqual(kwargs["container"], "sps-remote-jax")

  @mock.patch.dict(os.environ, {"USER": "testuser"})
  def test_main_sps_custom_workload(self):
    mock_args = argparse.Namespace(
        mode="jupyter",
        session="my-custom-proxy",
        port=8888,
        bucket="",
        check_active_session=False,
        sps=True,
        sps_image="my-image",
        instance_type="c4-standard-192",
    )
    with mock.patch.object(remote_ide, "get_args", return_value=mock_args):
      remote_ide.main()

    self.mock_find_pod.assert_called_once_with(
        "my-custom-proxy", exit_on_fail=False
    )

  @mock.patch.dict(os.environ, {"USER": "testuser"})
  def test_main_active_session_skip_setup(self):
    mock_args = argparse.Namespace(
        mode="vscode",
        session="testuser",
        port=8888,
        bucket="",
        check_active_session=True,
        sps=False,
        sps_image="my-image",
        instance_type="c4-standard-192",
    )
    mock_is_port_active = self.enter_context(
        mock.patch.object(remote_ide, "is_port_active", return_value=True)
    )

    sleep_calls = []
    def mock_sleep_fn(seconds):
      sleep_calls.append(seconds)
      if len(sleep_calls) > 1:
        raise KeyboardInterrupt()

    # Use a dummy loop exit side effect to avoid infinite sleep
    with mock.patch.object(
        remote_ide, "get_args", return_value=mock_args
    ), mock.patch("time.sleep", side_effect=mock_sleep_fn):
      remote_ide.main()

    mock_is_port_active.assert_called_once_with(
        "dummy-pod", 8888, "jax-tpu"
    )
    # Since skip_setup is True, load_script and stream should NOT be called
    self.mock_load_script.assert_not_called()
    self.mock_stream.assert_not_called()

  @mock.patch.dict(os.environ, {"USER": "testuser"})
  def test_main_sps_pod_deployment(self):
    mock_args = argparse.Namespace(
        mode="jupyter",
        session="testuser",
        port=8888,
        bucket="",
        check_active_session=False,
        sps=True,
        sps_image="my-container-image",
        instance_type="c4-standard-192",
    )
    mock_deploy_yaml = self.enter_context(
        mock.patch.object(remote_ide, "deploy_yaml")
    )
    mock_sleep = self.enter_context(mock.patch("time.sleep"))

    # We want find_pod to return None on direct check, None on the first poll check,
    # and then "my-deployed-sps-pod" on the second poll check.
    self.mock_find_pod.side_effect = [None, None, "my-deployed-sps-pod"]

    with mock.patch.object(remote_ide, "get_args", return_value=mock_args):
      remote_ide.main()

    expected_yaml_path = os.path.join(
        remote_ide.CURRENT_PATH,
        "..",
        "shared_pathways_service",
        "yamls",
        "sps-pod.yaml",
    )
    mock_deploy_yaml.assert_called_once_with(
        expected_yaml_path,
        "testuser",
        "my-container-image",
        8888,
        "c4-standard-192",
    )
    self.mock_find_pod.assert_has_calls([
        mock.call("testuser", exit_on_fail=False),
        mock.call("testuser", exit_on_fail=False),
    ])
    mock_sleep.assert_has_calls([mock.call(2), mock.call(1)])

  @mock.patch.dict(os.environ, {"USER": "testuser"})
  def test_main_sps_default_yaml_deployment(self):
    mock_args = argparse.Namespace(
        mode="jupyter",
        session="testuser",
        port=8888,
        bucket="",
        check_active_session=False,
        sps=True,
        sps_image="my-default-image",
        instance_type="cpu-test-type",
    )
    mock_deploy_yaml = self.enter_context(
        mock.patch.object(remote_ide, "deploy_yaml")
    )
    self.enter_context(mock.patch("time.sleep"))

    self.mock_find_pod.side_effect = [None, "my-default-sps-pod"]

    with mock.patch.object(remote_ide, "get_args", return_value=mock_args):
      remote_ide.main()

    expected_default_yaml_path = os.path.join(
        remote_ide.CURRENT_PATH,
        "..",
        "shared_pathways_service",
        "yamls",
        "sps-pod.yaml",
    )
    mock_deploy_yaml.assert_called_once_with(
        expected_default_yaml_path,
        "testuser",
        "my-default-image",
        8888,
        "cpu-test-type",
    )

  def test_deploy_yaml_success(self):
    mock_open_func = mock.mock_open(
        read_data=(
            "name: {SESSION}, image: {IMAGE}, "
            "port: {PORT}, instance_type: {INSTANCE_TYPE}"
        )
    )
    mock_deploy_gke_yaml = self.enter_context(
        mock.patch.object(remote_ide.gke_utils, "deploy_gke_yaml")
    )

    with mock.patch("builtins.open", mock_open_func):
      remote_ide.deploy_yaml(
          "dummy.yaml",
          "my-session",
          "my-image",
          8888,
          "c4-standard-192",
      )

    mock_open_func.assert_called_once_with("dummy.yaml", "r")
    mock_deploy_gke_yaml.assert_called_once_with(
        (
            "name: my-session, image:"
            " my-image, port: 8888, instance_type: c4-standard-192"
        ),
        action="apply",
    )

  def test_get_args_sps_subcommand(self):
    test_argv = [
        "remote_ide.py",
        "sps",
        "--image",
        "custom-sps-image",
        "--instance-type",
        "custom-type",
    ]
    with mock.patch("sys.argv", test_argv):
      args = remote_ide.get_args()
      self.assertTrue(args.sps)
      self.assertEqual(args.command, "sps")
      self.assertEqual(args.sps_image, "custom-sps-image")
      self.assertEqual(args.instance_type, "custom-type")

  def test_get_args_normal_pathways(self):
    test_argv = ["remote_ide.py", "-s", "my-workload-head", "-P", "9999"]
    with mock.patch("sys.argv", test_argv):
      args = remote_ide.get_args()
      self.assertFalse(args.sps)
      self.assertEqual(args.session, "my-workload-head")
      self.assertEqual(args.port, 9999)

  @mock.patch("os.path.exists")
  @mock.patch("builtins.open")
  def test_load_script_injects_bash_aliases(self, mock_open, mock_exists):
    mock_exists.side_effect = lambda path: path.endswith(".bash_aliases")
    mock_script_file = mock.mock_open(
        read_data="port: {PORT}, aliases: {BASH_ALIASES_BASE64}"
    ).return_value
    mock_aliases_file = mock.mock_open(
        read_data=b"alias gst='git status'"
    ).return_value
    mock_open.side_effect = [mock_script_file, mock_aliases_file]

    res = _ORIGINAL_LOAD_SCRIPT(
        "dummy_script.sh", 8888, "my-bucket", "my-workload"
    )
    expected_base64 = "YWxpYXMgZ3N0PSdnaXQgc3RhdHVzJw=="
    self.assertEqual(
        res, ["/bin/bash", "-c", f"port: 8888, aliases: {expected_base64}"]
    )

  @mock.patch("os.path.exists")
  @mock.patch("builtins.open")
  def test_load_script_no_bash_aliases(self, mock_open, mock_exists):
    mock_exists.return_value = False
    mock_open.return_value = mock.mock_open(
        read_data="port: {PORT}, aliases: {BASH_ALIASES_BASE64}"
    ).return_value

    res = _ORIGINAL_LOAD_SCRIPT(
        "dummy_script.sh", 8888, "my-bucket", "my-workload"
    )
    self.assertEqual(res, ["/bin/bash", "-c", "port: 8888, aliases: "])


if __name__ == "__main__":
  absltest.main()
