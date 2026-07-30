import os
import random
import string
from unittest import mock

from absl import app
from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from pathwaysutils.experimental.shared_pathways_service import gke_utils
from pathwaysutils.experimental.shared_pathways_service import start_vscode_on_cpu_np


class StartVSCodeOnCPUNPTest(absltest.TestCase):


  @flagsaver.flagsaver(
      namespace="my-ns",
      image="my-image",
      password="password123",
      instance_type="cpu-type",
  )
  def test_prepare_deployment_yaml_success(self):
    template_content = (
        "name: ${NAME}\n"
        "namespace: ${NAMESPACE}\n"
        "image: ${IMAGE}\n"
        "password: ${PASSWORD}\n"
        "node: ${INSTANCE_TYPE}\n"
        "service: ${SERVICE_NAME}\n"
        "port: ${PORT}"
    )
    self.enter_context(
        mock.patch(
            "builtins.open",
            new_callable=mock.mock_open,
            read_data=template_content,
        )
    )
    yaml_content = start_vscode_on_cpu_np._prepare_deployment_yaml(
        service_name="test-service", remote_port=9090
    )
    self.assertEqual(
        yaml_content,
        "name: test-service\n"
        "namespace: my-ns\n"
        "image: my-image\n"
        "password: password123\n"
        "node: cpu-type\n"
        "service: test-service\n"
        "port: 9090",
    )

  def test_prepare_deployment_yaml_missing_file_raises_value_error(self):
    self.enter_context(mock.patch("builtins.open", side_effect=OSError()))
    with self.assertRaisesRegex(ValueError, "Could not read template file:"):
      start_vscode_on_cpu_np._prepare_deployment_yaml(
          service_name="test-service", remote_port=9090
      )

  @flagsaver.flagsaver(namespace="my-ns")
  def test_deploy_vscode_success(self):
    mock_deploy = self.enter_context(
        mock.patch.object(gke_utils, "deploy_gke_yaml", autospec=True)
    )
    mock_wait_dep = self.enter_context(
        mock.patch.object(gke_utils, "wait_for_deployment", autospec=True)
    )
    mock_wait_svc = self.enter_context(
        mock.patch.object(gke_utils, "wait_for_service_ip", autospec=True)
    )
    mock_wait_svc.return_value = "10.0.0.1"

    start_vscode_on_cpu_np._deploy_vscode("test-service", "dummy-yaml")

    mock_deploy.assert_called_once_with("dummy-yaml", action="create")
    mock_wait_dep.assert_called_once_with("test-service", "my-ns")
    mock_wait_svc.assert_called_once_with("test-service", "my-ns")

  @flagsaver.flagsaver(namespace="my-ns")
  def test_deploy_vscode_service_ip_failure(self):
    mock_deploy = self.enter_context(
        mock.patch.object(gke_utils, "deploy_gke_yaml", autospec=True)
    )
    mock_wait_dep = self.enter_context(
        mock.patch.object(gke_utils, "wait_for_deployment", autospec=True)
    )
    mock_wait_svc = self.enter_context(
        mock.patch.object(gke_utils, "wait_for_service_ip", autospec=True)
    )
    mock_wait_svc.side_effect = RuntimeError("Service IP timeout")

    with self.assertLogs(level="WARNING") as log_capture:
      start_vscode_on_cpu_np._deploy_vscode("test-service", "dummy-yaml")

    mock_deploy.assert_called_once_with("dummy-yaml", action="create")
    mock_wait_dep.assert_called_once_with("test-service", "my-ns")
    mock_wait_svc.assert_called_once_with("test-service", "my-ns")
    self.assertTrue(
        any(
            "Could not get service IP" in record.message
            for record in log_capture.records
        )
    )

  @flagsaver.flagsaver(namespace="my-ns")
  def test_start_port_forwarding_keyboard_interrupt(self):
    mock_enable = self.enter_context(
        mock.patch.object(
            gke_utils, "enable_port_forwarding", autospec=True
        )
    )
    mock_process = mock.Mock()
    mock_enable.return_value = (8080, mock_process)
    mock_sleep = self.enter_context(
        mock.patch("time.sleep", side_effect=KeyboardInterrupt())
    )

    start_vscode_on_cpu_np._start_port_forwarding("test-service", 9090)

    mock_enable.assert_called_once_with(
        remote_server="svc/test-service",
        server_port=9090,
        namespace="my-ns",
    )
    mock_sleep.assert_called_once_with(1)
    mock_process.terminate.assert_called_once()
    mock_process.wait.assert_called_once()

  @flagsaver.flagsaver(namespace="my-ns")
  def test_start_port_forwarding_other_exception(self):
    mock_enable = self.enter_context(
        mock.patch.object(
            gke_utils, "enable_port_forwarding", autospec=True
        )
    )
    mock_process = mock.Mock()
    mock_enable.return_value = (8080, mock_process)
    mock_sleep = self.enter_context(
        mock.patch("time.sleep", side_effect=Exception("Forwarding crash"))
    )

    start_vscode_on_cpu_np._start_port_forwarding("test-service", 9090)

    mock_enable.assert_called_once_with(
        remote_server="svc/test-service",
        server_port=9090,
        namespace="my-ns",
    )
    mock_sleep.assert_called_once_with(1)
    mock_process.terminate.assert_called_once()
    mock_process.wait.assert_called_once()

  def test_cleanup_gke_resources_success(self):
    mock_delete_resource = self.enter_context(
        mock.patch.object(gke_utils, "delete_gke_resource", autospec=True)
    )

    start_vscode_on_cpu_np._cleanup_gke_resources("test-service", "my-ns")

    mock_delete_resource.assert_has_calls([
        mock.call("deployment", "test-service", "my-ns"),
        mock.call("service", "test-service", "my-ns"),
    ])

  def test_cleanup_gke_resources_ignores_exceptions(self):
    mock_delete_resource = self.enter_context(
        mock.patch.object(gke_utils, "delete_gke_resource", autospec=True)
    )
    mock_delete_resource.side_effect = Exception("delete fail")

    # Should not raise an exception
    start_vscode_on_cpu_np._cleanup_gke_resources("test-service", "my-ns")

    mock_delete_resource.assert_has_calls([
        mock.call("deployment", "test-service", "my-ns"),
        mock.call("service", "test-service", "my-ns"),
    ])

  def test_main_too_many_args(self):
    with self.assertRaises(app.UsageError):
      start_vscode_on_cpu_np.main(["script_name", "extra_arg"])

  @flagsaver.flagsaver(dry_run=True, name="my-vscode")
  @mock.patch.dict(os.environ, {"USER": "testuser"})
  def test_main_dry_run(self):
    mock_prepare = self.enter_context(
        mock.patch.object(
            start_vscode_on_cpu_np, "_prepare_deployment_yaml", autospec=True
        )
    )
    mock_prepare.return_value = "dummy-yaml"
    mock_deploy = self.enter_context(
        mock.patch.object(
            start_vscode_on_cpu_np, "_deploy_vscode", autospec=True
        )
    )
    mock_cleanup = self.enter_context(
        mock.patch.object(
            start_vscode_on_cpu_np, "_cleanup_gke_resources", autospec=True
        )
    )

    mock_choices = self.enter_context(
        mock.patch.object(random, "choices", autospec=True)
    )
    mock_choices.return_value = ["a", "b", "c", "d"]

    start_vscode_on_cpu_np.main(["script_name"])

    mock_prepare.assert_called_once_with("my-vscode-testuser-abcd", 8080)
    mock_choices.assert_called_once_with(
        string.ascii_lowercase + string.digits, k=4
    )
    mock_deploy.assert_not_called()
    mock_cleanup.assert_not_called()

  @flagsaver.flagsaver(dry_run=False, name="my-vscode", namespace="my-ns")
  @mock.patch.dict(os.environ, {"USER": "testuser"})
  def test_main_real_run_success(self):
    mock_fetch = self.enter_context(
        mock.patch.object(gke_utils, "fetch_cluster_credentials", autospec=True)
    )
    mock_prepare = self.enter_context(
        mock.patch.object(
            start_vscode_on_cpu_np, "_prepare_deployment_yaml", autospec=True
        )
    )
    mock_prepare.return_value = "dummy-yaml"
    mock_deploy = self.enter_context(
        mock.patch.object(
            start_vscode_on_cpu_np, "_deploy_vscode", autospec=True
        )
    )
    mock_forward = self.enter_context(
        mock.patch.object(
            start_vscode_on_cpu_np, "_start_port_forwarding", autospec=True
        )
    )
    mock_cleanup = self.enter_context(
        mock.patch.object(
            start_vscode_on_cpu_np, "_cleanup_gke_resources", autospec=True
        )
    )

    mock_choices = self.enter_context(
        mock.patch.object(random, "choices", autospec=True)
    )
    mock_choices.return_value = ["a", "b", "c", "d"]

    start_vscode_on_cpu_np.main(["script_name"])

    expected_service_name = "my-vscode-testuser-abcd"
    mock_prepare.assert_called_once_with(expected_service_name, 8080)
    mock_choices.assert_called_once_with(
        string.ascii_lowercase + string.digits, k=4
    )
    mock_deploy.assert_called_once_with(expected_service_name, "dummy-yaml")
    mock_forward.assert_called_once_with(expected_service_name, 8080)
    mock_cleanup.assert_called_once_with(expected_service_name, "my-ns")
    mock_fetch.assert_called_once_with(
        cluster_name="dummy-cluster",
        project_id="dummy-project",
        location="dummy-region",
    )

  @flagsaver.flagsaver(dry_run=False, name="my-vscode", namespace="my-ns")
  @mock.patch.dict(os.environ, {"USER": "testuser"})
  def test_main_real_run_exception_still_cleans_up(self):
    mock_fetch = self.enter_context(
        mock.patch.object(gke_utils, "fetch_cluster_credentials", autospec=True)
    )
    mock_prepare = self.enter_context(
        mock.patch.object(
            start_vscode_on_cpu_np, "_prepare_deployment_yaml", autospec=True
        )
    )
    mock_prepare.return_value = "dummy-yaml"
    mock_deploy = self.enter_context(
        mock.patch.object(
            start_vscode_on_cpu_np, "_deploy_vscode", autospec=True
        )
    )
    mock_deploy.side_effect = RuntimeError("Deploy failed")
    mock_forward = self.enter_context(
        mock.patch.object(
            start_vscode_on_cpu_np, "_start_port_forwarding", autospec=True
        )
    )
    mock_cleanup = self.enter_context(
        mock.patch.object(
            start_vscode_on_cpu_np, "_cleanup_gke_resources", autospec=True
        )
    )

    mock_choices = self.enter_context(
        mock.patch.object(random, "choices", autospec=True)
    )
    mock_choices.return_value = ["a", "b", "c", "d"]

    with self.assertRaises(RuntimeError):
      start_vscode_on_cpu_np.main(["script_name"])

    expected_service_name = "my-vscode-testuser-abcd"
    mock_prepare.assert_called_once()
    mock_choices.assert_called_once_with(
        string.ascii_lowercase + string.digits, k=4
    )
    mock_deploy.assert_called_once_with(expected_service_name, "dummy-yaml")
    mock_forward.assert_not_called()
    mock_cleanup.assert_called_once_with(expected_service_name, "my-ns")
    mock_fetch.assert_called_once_with(
        cluster_name="dummy-cluster",
        project_id="dummy-project",
        location="dummy-region",
    )


if __name__ == "__main__":
  FLAGS = flags.FLAGS
  FLAGS.cluster = "dummy-cluster"
  FLAGS.project = "dummy-project"
  FLAGS.region = "dummy-region"
  absltest.main()
