"""Tests for gke_utils.py.
"""

import io
import socket
import subprocess
from typing import Any
from unittest import mock

from absl.testing import absltest
from kubernetes import client
from kubernetes import config as k8s_config
from pathwaysutils.experimental.shared_pathways_service import gke_utils
import portpicker


class GKEUtilsTest(absltest.TestCase):
  """Tests for gke_utils.py."""

  def setUp(self):
    super().setUp()
    gke_utils._init_k8s_config.cache_clear()
    gke_utils._get_k8s_core_api.cache_clear()
    gke_utils._get_k8s_custom_objects_api.cache_clear()

  def _make_jobset(
      self,
      name: str = "my-jobset",
      replicated_jobs: list[dict[str, Any]] | None = None,
  ) -> dict[str, Any]:
    return {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "metadata": {
            "name": name,
            "namespace": "default",
        },
        "spec": {
            "replicatedJobs": replicated_jobs or [],
        },
    }

  def _make_replicated_job(
      self,
      name: str = "pathways-worker",
      containers: list[dict[str, Any]] | None = None,
      init_containers: list[dict[str, Any]] | None = None,
  ) -> dict[str, Any]:
    return {
        "name": name,
        "template": {
            "spec": {
                "template": {
                    "spec": {
                        "containers": containers or [],
                        "initContainers": init_containers or [],
                    }
                }
            }
        },
    }

  def test_get_k8s_core_api_load_kube_config_success(self):
    mock_load = self.enter_context(
        mock.patch.object(k8s_config, "load_kube_config", autospec=True)
    )
    api = gke_utils._get_k8s_core_api()
    self.assertIsInstance(api, client.CoreV1Api)
    mock_load.assert_called_once()

  def test_get_k8s_core_api_caches_result(self):
    mock_load = self.enter_context(
        mock.patch.object(k8s_config, "load_kube_config", autospec=True)
    )
    api1 = gke_utils._get_k8s_core_api()
    api2 = gke_utils._get_k8s_core_api()
    self.assertIs(api1, api2)
    mock_load.assert_called_once()

  def test_get_k8s_core_api_incluster_fallback(self):
    self.enter_context(
        mock.patch.object(
            k8s_config,
            "load_kube_config",
            side_effect=Exception("Failed to load kubeconfig"),
        )
    )
    mock_incluster = self.enter_context(
        mock.patch.object(k8s_config, "load_incluster_config", autospec=True)
    )
    api = gke_utils._get_k8s_core_api()
    self.assertIsInstance(api, client.CoreV1Api)
    mock_incluster.assert_called_once()

  def test_get_k8s_core_api_failure_raises(self):
    self.enter_context(
        mock.patch.object(
            k8s_config,
            "load_kube_config",
            side_effect=Exception("Failed to load kubeconfig"),
        )
    )
    self.enter_context(
        mock.patch.object(
            k8s_config,
            "load_incluster_config",
            side_effect=Exception("Failed in cluster"),
        )
    )
    with self.assertRaises(RuntimeError):
      gke_utils._get_k8s_core_api()

  def test_get_k8s_custom_objects_api_caches_result(self):
    mock_load = self.enter_context(
        mock.patch.object(k8s_config, "load_kube_config", autospec=True)
    )
    api1 = gke_utils._get_k8s_custom_objects_api()
    api2 = gke_utils._get_k8s_custom_objects_api()
    self.assertIs(api1, api2)
    self.assertIsInstance(api1, client.CustomObjectsApi)
    mock_load.assert_called_once()

  def test_fetch_cluster_credentials_success(self):
    """Tests that fetch_cluster_credentials calls gcloud with the correct arguments."""
    mock_run = self.enter_context(
        mock.patch.object(subprocess, "run", autospec=True)
    )
    gke_utils.fetch_cluster_credentials(
        cluster_name="test-cluster",
        project_id="test-project",
        location="test-zone",
    )
    mock_run.assert_called_once_with(
        [
            "gcloud",
            "container",
            "clusters",
            "get-credentials",
            "--location=test-zone",
            "--project=test-project",
            "--dns-endpoint",
            "--",
            "test-cluster",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

  def test_fetch_cluster_credentials_failure(self):
    """Tests that fetch_cluster_credentials raises an error when gcloud fails."""
    mock_run = self.enter_context(
        mock.patch.object(subprocess, "run", autospec=True)
    )
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd="gcloud", stderr="error"
    )
    with self.assertRaises(subprocess.CalledProcessError):
      gke_utils.fetch_cluster_credentials(
          cluster_name="test-cluster",
          project_id="test-project",
          location="test-zone",
      )

  def test_validate_k8s_name_valid(self):
    gke_utils._validate_k8s_name("valid-name-123")
    gke_utils._validate_k8s_name("a")
    gke_utils._validate_k8s_name("a-b")

  def test_validate_k8s_name_invalid(self):
    with self.assertRaises(ValueError):
      gke_utils._validate_k8s_name("-invalid")
    with self.assertRaises(ValueError):
      gke_utils._validate_k8s_name("invalid-")
    with self.assertRaises(ValueError):
      gke_utils._validate_k8s_name("Invalid")
    with self.assertRaises(ValueError):
      gke_utils._validate_k8s_name("invalid_name")
    with self.assertRaises(ValueError):
      gke_utils._validate_k8s_name("invalid.name")

  def test_deploy_gke_yaml_success(self):
    mock_run = self.enter_context(
        mock.patch.object(subprocess, "run", autospec=True)
    )
    test_yaml = "apiVersion: v1\nkind: Pod\nmetadata:\n  name: test"
    gke_utils.deploy_gke_yaml(test_yaml)
    mock_run.assert_called_once_with(
        ["kubectl", "apply", "-f", "-"],
        input=test_yaml,
        check=True,
        capture_output=True,
        text=True,
    )

  def test_deploy_gke_yaml_failure(self):
    mock_run = self.enter_context(
        mock.patch.object(subprocess, "run", autospec=True)
    )
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd="kubectl apply", stderr="error"
    )
    with self.assertRaises(subprocess.CalledProcessError):
      gke_utils.deploy_gke_yaml("test_yaml")

  def test_deploy_gke_yaml_create_success(self):
    mock_run = self.enter_context(
        mock.patch.object(subprocess, "run", autospec=True)
    )
    test_yaml = "apiVersion: v1\nkind: Pod\nmetadata:\n  name: test"
    gke_utils.deploy_gke_yaml(test_yaml, action="create")
    mock_run.assert_called_once_with(
        ["kubectl", "create", "-f", "-"],
        input=test_yaml,
        check=True,
        capture_output=True,
        text=True,
    )

  def test_delete_gke_resource_success(self):
    mock_run = self.enter_context(
        mock.patch.object(subprocess, "run", autospec=True)
    )
    gke_utils.delete_gke_resource("deployment", "test-deploy", "test-ns")
    mock_run.assert_called_once_with(
        [
            "kubectl",
            "delete",
            "deployment",
            "-n",
            "test-ns",
            "--ignore-not-found",
            "--",
            "test-deploy",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

  def test_delete_gke_resource_failure(self):
    mock_run = self.enter_context(
        mock.patch.object(subprocess, "run", autospec=True)
    )
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd="kubectl delete", stderr="error"
    )
    with self.assertRaises(subprocess.CalledProcessError):
      gke_utils.delete_gke_resource("deployment", "test-deploy", "test-ns")

  def test_get_pod_from_job_success(self):
    """Tests that get_pod_from_job returns the pod name on success."""
    mock_run = self.enter_context(
        mock.patch.object(
            subprocess,
            "run",
            autospec=True,
        )
    )
    mock_get_pods_result = subprocess.CompletedProcess(
        args=["kubectl", "get", "pods"],
        returncode=0,
        stdout="pod/test-pod-123\n",
    )
    mock_run.return_value = mock_get_pods_result

    pod_name = gke_utils.get_pod_from_job("test-proxy-job")

    self.assertEqual(pod_name, "test-pod-123")
    mock_run.assert_called_once()
    self.assertIn("get", mock_run.call_args[0][0])
    self.assertIn("job-name=test-proxy-job", mock_run.call_args[0][0])

  def test_get_pod_from_job_failure(self):
    """Tests that get_pod_from_job raises an error if kubectl fails."""
    mock_run = self.enter_context(
        mock.patch.object(
            subprocess,
            "run",
            autospec=True,
        )
    )
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd="kubectl get pods", stderr="error"
    )

    with self.assertRaises(subprocess.CalledProcessError):
      gke_utils.get_pod_from_job("test-proxy-job")

  def test_check_pod_ready_success(self):
    """Tests that check_pod_ready returns the pod name on success."""
    mock_run = self.enter_context(
        mock.patch.object(
            subprocess,
            "run",
            autospec=True,
        )
    )
    mock_wait_success_result = subprocess.CompletedProcess(
        args=["kubectl", "wait"],
        returncode=0,
    )
    mock_run.return_value = mock_wait_success_result

    pod_name = gke_utils.check_pod_ready("test-pod-123")

    self.assertEqual(pod_name, "test-pod-123")
    mock_run.assert_called_once_with(
        [
            "kubectl",
            "wait",
            "--for=condition=Ready",
            "--timeout=30s",
            "--",
            "pod/test-pod-123",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

  def test_check_pod_ready_failure(self):
    """Tests that check_pod_ready raises a RuntimeError if kubectl wait fails."""
    mock_run = self.enter_context(
        mock.patch.object(
            subprocess,
            "run",
            autospec=True,
        )
    )
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd="kubectl wait", stderr="error"
    )

    with self.assertRaisesRegex(
        RuntimeError, "Pod did not become ready: error."
    ):
      gke_utils.check_pod_ready("test-pod-123")

  def test_get_log_link(self):
    cluster = "test-cluster"
    project = "test-project"
    job_name = "test-job"
    log_link = gke_utils.get_log_link(
        cluster=cluster, project=project, job_name=job_name
    )
    self.assertEqual(
        log_link,
        r"https://console.cloud.google.com/logs/query;query=resource.type%3D"
        r"%22k8s_container%22%0Aresource.labels.cluster_name%3D"
        "%22test-cluster%22%0Aresource.labels.namespace_name%3D"
        "%22default%22%0Alabels.k8s-pod%2Fjob-name%3A%22test-job%22;"
        "duration=PT1H?project=test-project",
    )

  def test_wait_for_pod_success(self):
    """Tests that wait_for_pod returns the pod name on success."""
    mock_run = self.enter_context(
        mock.patch.object(
            subprocess,
            "run",
            autospec=True,
        )
    )
    mock_get_pods_result = subprocess.CompletedProcess(
        args=["kubectl", "get", "pods"],
        returncode=0,
        stdout="pod/test-pod-123\n",
    )
    mock_wait_success_result = subprocess.CompletedProcess(
        args=["kubectl", "wait"],
        returncode=0,
    )
    mock_run.side_effect = [mock_get_pods_result, mock_wait_success_result]

    pod_name = gke_utils.wait_for_pod("test-proxy-job")

    self.assertEqual(pod_name, "test-pod-123")
    self.assertEqual(mock_run.call_count, 2)
    self.assertIn("get", mock_run.call_args_list[0].args[0])
    self.assertIn("wait", mock_run.call_args_list[1].args[0])

  def test_wait_for_pod_get_pods_fails(self):
    """Tests that wait_for_pod raises an error if 'get pods' fails."""
    mock_run = self.enter_context(
        mock.patch.object(
            subprocess,
            "run",
            autospec=True,
        )
    )
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd="kubectl get pods", stderr="error"
    )

    with self.assertRaises(subprocess.CalledProcessError):
      gke_utils.wait_for_pod("test-proxy-job")
    self.assertEqual(mock_run.call_count, 1)

  def test_wait_for_pod_wait_fails(self):
    """Tests that wait_for_pod raises a RuntimeError if 'wait' times out."""
    mock_run = self.enter_context(
        mock.patch.object(
            subprocess,
            "run",
            autospec=True,
        )
    )
    mock_get_pods_result = subprocess.CompletedProcess(
        args=["kubectl", "get", "pods"],
        returncode=0,
        stdout="pod/test-pod-123\n",
    )
    mock_run.side_effect = [
        mock_get_pods_result,
        subprocess.TimeoutExpired(cmd="kubectl wait", timeout=30),
    ]

    with self.assertRaises(RuntimeError):
      gke_utils.wait_for_pod("test-proxy-job")
    self.assertEqual(mock_run.call_count, 2)

  def test_enable_port_forwarding_success(self):
    """Tests successful port forwarding."""
    # Arrange
    mock_create_connection = self.enter_context(
        mock.patch.object(socket, "create_connection", autospec=True)
    )
    mock_popen = self.enter_context(
        mock.patch.object(
            subprocess,
            "Popen",
            autospec=True,
        )
    )
    pod_name = "test-pod-123"
    mock_pick_port = self.enter_context(
        mock.patch.object(portpicker, "pick_unused_port", autospec=True)
    )
    mock_pick_port.return_value = 29007
    mock_process = mock_popen.return_value
    mock_process.stdout = io.StringIO(
        "Forwarding from 127.0.0.1:29007 -> 8080\n"
    )

    # Act
    port, process = gke_utils.enable_port_forwarding(pod_name, 8080)

    # Assert
    self.assertEqual(port, 29007)
    self.assertIs(process, mock_process)
    mock_popen.assert_called_once_with(
        [
            "kubectl",
            "port-forward",
            "-n",
            "default",
            "--address",
            "localhost",
            "--",
            "test-pod-123",
            "29007:8080",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    mock_process.terminate.assert_not_called()
    mock_create_connection.assert_called_once_with(
        ("localhost", 29007), timeout=30
    )

  def test_enable_port_forwarding_timeout_raises_error(self):
    """Tests that a timeout in port forwarding raises a RuntimeError."""
    # Arrange
    mock_create_connection = self.enter_context(
        mock.patch.object(socket, "create_connection", autospec=True)
    )
    mock_popen = self.enter_context(
        mock.patch.object(
            subprocess,
            "Popen",
            autospec=True,
        )
    )
    pod_name = "test-pod-123"
    mock_pick_port = self.enter_context(
        mock.patch.object(portpicker, "pick_unused_port", autospec=True)
    )
    mock_pick_port.return_value = 29007
    mock_process = mock_popen.return_value
    mock_process.stdout = io.StringIO("")  # Empty output simulates a timeout
    mock_process.communicate.return_value = ("", "")
    mock_process.poll.return_value = None
    def terminate_effect():
      mock_process.poll.return_value = 1
    mock_process.terminate.side_effect = terminate_effect
    mock_popen.return_value = mock_process

    # Act & Assert
    with self.assertRaises(RuntimeError):
      gke_utils.enable_port_forwarding(pod_name, 8080)

    mock_create_connection.assert_not_called()
    mock_process.terminate.assert_called_once()

  def test_enable_port_forwarding_socket_error_raises_error(self):
    """Tests that a socket error during connection check raises an error."""
    # Arrange
    mock_create_connection = self.enter_context(
        mock.patch.object(socket, "create_connection", autospec=True)
    )
    mock_popen = self.enter_context(
        mock.patch.object(
            subprocess,
            "Popen",
            autospec=True,
        )
    )
    pod_name = "test-pod-123"
    mock_pick_port = self.enter_context(
        mock.patch.object(portpicker, "pick_unused_port", autospec=True)
    )
    mock_pick_port.return_value = 29007
    mock_process = mock_popen.return_value
    mock_process.stdout = io.StringIO(
        "Forwarding from 127.0.0.1:29007 -> 8080\n"
    )
    mock_process.poll.return_value = None
    mock_popen.return_value = mock_process
    mock_create_connection.side_effect = OSError("Connection failed")

    # Act & Assert
    with self.assertRaises(OSError):
      gke_utils.enable_port_forwarding(pod_name, 8080)

    mock_process.terminate.assert_called_once()

  def test_stream_pod_logs_success(self):
    mock_popen = self.enter_context(
        mock.patch.object(subprocess, "Popen", autospec=True)
    )
    mock_process = mock_popen.return_value

    process = gke_utils.stream_pod_logs("test-pod-123")

    self.assertIs(process, mock_process)
    mock_popen.assert_called_once_with(
        ["kubectl", "logs", "-f", "--", "pod/test-pod-123"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

  def test_stream_pod_logs_failure(self):
    mock_popen = self.enter_context(
        mock.patch.object(subprocess, "Popen", autospec=True)
    )
    mock_popen.side_effect = Exception("error")

    with self.assertRaisesRegex(Exception, "error"):
      gke_utils.stream_pod_logs("test-pod-123")

  def test_deploy_gke_yaml_invalid_action(self):
    with self.assertRaisesRegex(ValueError, "Invalid kubectl action:"):
      gke_utils.deploy_gke_yaml("test_yaml", action="invalid_action")

  def test_get_pod_from_job_invalid_format_empty(self):
    mock_run = self.enter_context(
        mock.patch.object(subprocess, "run", autospec=True)
    )
    mock_run.return_value = subprocess.CompletedProcess(
        args=["kubectl", "get", "pods"],
        returncode=0,
        stdout="\n",
    )
    with self.assertRaisesRegex(RuntimeError, "Failed to get pod name. Expected format:"):
      gke_utils.get_pod_from_job("test-job")

  def test_get_pod_from_job_invalid_format_no_prefix(self):
    mock_run = self.enter_context(
        mock.patch.object(subprocess, "run", autospec=True)
    )
    mock_run.return_value = subprocess.CompletedProcess(
        args=["kubectl", "get", "pods"],
        returncode=0,
        stdout="test-pod-123\n",
    )
    with self.assertRaisesRegex(RuntimeError, "Failed to get pod name. Expected format:"):
      gke_utils.get_pod_from_job("test-job")

  def test_get_pod_from_job_invalid_format_too_many_slashes(self):
    mock_run = self.enter_context(
        mock.patch.object(subprocess, "run", autospec=True)
    )
    mock_run.return_value = subprocess.CompletedProcess(
        args=["kubectl", "get", "pods"],
        returncode=0,
        stdout="pod/test-pod/extra\n",
    )
    with self.assertRaisesRegex(RuntimeError, "Failed to get pod name. Expected format:"):
      gke_utils.get_pod_from_job("test-job")

  def test_test_remote_connection_success(self):
    mock_create = self.enter_context(
        mock.patch.object(socket, "create_connection", autospec=True)
    )
    gke_utils._test_remote_connection(8080)
    mock_create.assert_called_once_with(("localhost", 8080), timeout=30)

  def test_test_remote_connection_timeout(self):
    mock_create = self.enter_context(
        mock.patch.object(socket, "create_connection", autospec=True)
    )
    mock_create.side_effect = socket.timeout("timeout error")
    with self.assertRaisesRegex(RuntimeError, "Could not connect to the pod."):
      gke_utils._test_remote_connection(8080)

  def test_test_remote_connection_refused(self):
    mock_create = self.enter_context(
        mock.patch.object(socket, "create_connection", autospec=True)
    )
    mock_create.side_effect = ConnectionRefusedError("connection refused")
    with self.assertRaisesRegex(RuntimeError, "Could not connect to the pod."):
      gke_utils._test_remote_connection(8080)

  def test_enable_port_forwarding_pick_port_fails(self):
    self.enter_context(
        mock.patch.object(portpicker, "pick_unused_port", side_effect=ValueError("pick failed"))
    )
    with self.assertRaisesRegex(ValueError, "pick failed"):
      gke_utils.enable_port_forwarding("test-pod", 8080)

  def test_enable_port_forwarding_popen_fails(self):
    self.enter_context(
        mock.patch.object(portpicker, "pick_unused_port", return_value=12345)
    )
    self.enter_context(
        mock.patch.object(subprocess, "Popen", side_effect=OSError("Popen failed"))
    )
    with self.assertRaisesRegex(OSError, "Popen failed"):
      gke_utils.enable_port_forwarding("test-pod", 8080)

  def test_enable_port_forwarding_stdout_none(self):
    self.enter_context(
        mock.patch.object(portpicker, "pick_unused_port", return_value=12345)
    )
    mock_popen = self.enter_context(
        mock.patch.object(subprocess, "Popen", autospec=True)
    )
    mock_process = mock_popen.return_value
    mock_process.stdout = None
    mock_process.communicate.return_value = ("stdout", "stderr_out")
    
    with self.assertRaisesRegex(RuntimeError, "Failed to start port forwarding: stdout not available.\nSTDERR: stderr_out"):
      gke_utils.enable_port_forwarding("test-pod", 8080)
    mock_process.terminate.assert_called_once()
    mock_process.communicate.assert_called_once()

  def test_wait_for_deployment_success(self):
    mock_run = self.enter_context(
        mock.patch.object(subprocess, "run", autospec=True)
    )
    gke_utils.wait_for_deployment("my-deploy", "my-ns")
    mock_run.assert_called_once_with(
        [
            "kubectl",
            "rollout",
            "status",
            "deployment/my-deploy",
            "-n",
            "my-ns",
            "--timeout=300s",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

  def test_wait_for_deployment_failure(self):
    mock_run = self.enter_context(
        mock.patch.object(subprocess, "run", autospec=True)
    )
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd="kubectl rollout", stderr="rollout failed"
    )
    with self.assertRaisesRegex(RuntimeError, "Deployment did not become ready: rollout failed"):
      gke_utils.wait_for_deployment("my-deploy", "my-ns")

  def test_wait_for_service_ip_success_first_try(self):
    mock_run = self.enter_context(
        mock.patch.object(subprocess, "run", autospec=True)
    )
    mock_run.return_value = subprocess.CompletedProcess(
        args=["kubectl", "get", "svc"],
        returncode=0,
        stdout="1.2.3.4\n",
    )
    ip = gke_utils.wait_for_service_ip("my-svc", "my-ns", timeout=10)
    self.assertEqual(ip, "1.2.3.4")
    mock_run.assert_called_once_with(
        [
            "kubectl",
            "get",
            "svc",
            "my-svc",
            "-n",
            "my-ns",
            "-o",
            "jsonpath={.status.loadBalancer.ingress[0].ip}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

  def test_wait_for_service_ip_success_third_try(self):
    mock_run = self.enter_context(
        mock.patch.object(subprocess, "run", autospec=True)
    )
    mock_run.side_effect = [
        subprocess.CompletedProcess(args=[], returncode=0, stdout=""),
        subprocess.CompletedProcess(args=[], returncode=0, stdout=""),
        subprocess.CompletedProcess(args=[], returncode=0, stdout="1.2.3.4"),
    ]
    mock_sleep = self.enter_context(
        mock.patch("time.sleep", autospec=True)
    )
    ip = gke_utils.wait_for_service_ip("my-svc", "my-ns", timeout=10)
    self.assertEqual(ip, "1.2.3.4")
    self.assertEqual(mock_run.call_count, 3)
    self.assertEqual(mock_sleep.call_count, 2)

  def test_wait_for_service_ip_timeout(self):
    mock_run = self.enter_context(
        mock.patch.object(subprocess, "run", autospec=True)
    )
    mock_run.return_value = subprocess.CompletedProcess(
        args=["kubectl", "get", "svc"],
        returncode=0,
        stdout="",
    )
    self.enter_context(mock.patch("time.sleep", autospec=True))
    self.enter_context(
        mock.patch("time.time", side_effect=[0, 0, 4, 4, 10])
    )
    with self.assertRaisesRegex(
        RuntimeError, "Timeout waiting for service IP for my-svc"
    ):
      gke_utils.wait_for_service_ip("my-svc", "my-ns", timeout=5)
    self.assertGreaterEqual(mock_run.call_count, 2)

  def test_pick_unused_local_port(self):
    mock_pick = self.enter_context(
        mock.patch.object(portpicker, "pick_unused_port", return_value=12345)
    )
    port = gke_utils.pick_unused_local_port()
    self.assertEqual(port, 12345)
    mock_pick.assert_called_once()

  def test_is_local_port_free(self):
    mock_check = self.enter_context(
        mock.patch.object(portpicker, "is_port_free", return_value=True)
    )
    is_free = gke_utils.is_local_port_free(12345)
    self.assertTrue(is_free)
    mock_check.assert_called_once_with(12345)

  def test_enable_port_forwarding_with_slash_success(self):
    """Tests successful port forwarding when remote_server contains a slash."""
    self.enter_context(
        mock.patch.object(socket, "create_connection", autospec=True)
    )
    mock_popen = self.enter_context(
        mock.patch.object(
            subprocess,
            "Popen",
            autospec=True,
        )
    )
    mock_pick_port = self.enter_context(
        mock.patch.object(portpicker, "pick_unused_port", autospec=True)
    )
    mock_pick_port.return_value = 29007
    mock_process = mock_popen.return_value
    mock_process.stdout = io.StringIO(
        "Forwarding from 127.0.0.1:29007 -> 8080\n"
    )

    port, process = gke_utils.enable_port_forwarding("svc/test-service", 8080)

    self.assertEqual(port, 29007)
    self.assertIs(process, mock_process)
    mock_popen.assert_called_once_with(
        [
            "kubectl",
            "port-forward",
            "-n",
            "default",
            "--address",
            "localhost",
            "--",
            "svc/test-service",
            "29007:8080",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

  def test_enable_port_forwarding_invalid_format(self):
    with self.assertRaises(ValueError):
      gke_utils.enable_port_forwarding("invalid/svc/name", 8080)

  def test_enable_port_forwarding_invalid_name(self):
    with self.assertRaises(ValueError):
      gke_utils.enable_port_forwarding("svc/invalid_name", 8080)

  def test_delete_gke_resource_invalid_params(self):
    with self.assertRaises(ValueError):
      gke_utils.delete_gke_resource("invalid_type", "name-123", "namespace")
    with self.assertRaises(ValueError):
      gke_utils.delete_gke_resource("deployment", "invalid_name", "namespace")
    with self.assertRaises(ValueError):
      gke_utils.delete_gke_resource("deployment", "name-123",
                                    "invalid_namespace")

  def test_get_pathways_service_images_success(self):
    mock_custom_api = mock.MagicMock(spec=client.CustomObjectsApi)
    self.enter_context(
        mock.patch.object(
            gke_utils,
            "_get_k8s_custom_objects_api",
            return_value=mock_custom_api,
        )
    )
    mock_custom_api.get_namespaced_custom_object.return_value = (
        self._make_jobset(
            replicated_jobs=[
                self._make_replicated_job(
                    name="pathways-worker",
                    containers=[
                        {
                            "name": "pathways-worker",
                            "image": "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:jax-0.9.0",
                        },
                        {
                            "name": "colocated-python-sidecar",
                            "image": (
                                "us-docker.pkg.dev/cloud-tpu-v2-images/"
                                "pathways-colocated-python/sidecar:"
                                "20260423-python_3.12-jax_0.10.0"
                            ),
                        },
                    ],
                )
            ]
        )
    )
    server_img, sidecar_img = gke_utils.get_pathways_service_images(
        pathways_service="my-jobset-pathways-head-0-0:8080"
    )
    self.assertEqual(
        server_img,
        "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:jax-0.9.0",
    )
    self.assertEqual(
        sidecar_img,
        (
            "us-docker.pkg.dev/cloud-tpu-v2-images/pathways-colocated-python/"
            "sidecar:20260423-python_3.12-jax_0.10.0"
        ),
    )
    mock_custom_api.get_namespaced_custom_object.assert_called_once_with(
        group="jobset.x-k8s.io",
        version="v1alpha2",
        namespace="default",
        plural="jobsets",
        name="my-jobset",
    )

  def test_get_pathways_service_images_without_sidecar(self):
    mock_custom_api = mock.MagicMock(spec=client.CustomObjectsApi)
    self.enter_context(
        mock.patch.object(
            gke_utils,
            "_get_k8s_custom_objects_api",
            return_value=mock_custom_api,
        )
    )
    mock_custom_api.get_namespaced_custom_object.return_value = (
        self._make_jobset(
            replicated_jobs=[
                self._make_replicated_job(
                    name="pathways-worker",
                    containers=[
                        {
                            "name": "pathways-worker",
                            "image": "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:jax-0.9.0",
                        }
                    ],
                )
            ]
        )
    )
    server_img, sidecar_img = gke_utils.get_pathways_service_images(
        pathways_service="my-jobset-pathways-head-0-0:8080"
    )
    self.assertEqual(
        server_img,
        "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:jax-0.9.0",
    )
    self.assertIsNone(sidecar_img)

  def test_get_pathways_service_images_sidecar_in_init_containers(self):
    mock_custom_api = mock.MagicMock(spec=client.CustomObjectsApi)
    self.enter_context(
        mock.patch.object(
            gke_utils,
            "_get_k8s_custom_objects_api",
            return_value=mock_custom_api,
        )
    )
    mock_custom_api.get_namespaced_custom_object.return_value = (
        self._make_jobset(
            replicated_jobs=[
                self._make_replicated_job(
                    name="pathways-worker",
                    containers=[
                        {
                            "name": "pathways-worker",
                            "image": "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:jax-0.9.0",
                        }
                    ],
                    init_containers=[
                        {
                            "name": "colocated-python-sidecar",
                            "image": "sidecar-image-url",
                        }
                    ],
                )
            ]
        )
    )
    server_img, sidecar_img = gke_utils.get_pathways_service_images(
        pathways_service="my-jobset-pathways-head-0-0:8080"
    )
    self.assertEqual(
        server_img,
        "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:jax-0.9.0",
    )
    self.assertEqual(sidecar_img, "sidecar-image-url")

  def test_get_pathways_service_images_custom_namespace(self):
    mock_custom_api = mock.MagicMock(spec=client.CustomObjectsApi)
    self.enter_context(
        mock.patch.object(
            gke_utils,
            "_get_k8s_custom_objects_api",
            return_value=mock_custom_api,
        )
    )
    mock_custom_api.get_namespaced_custom_object.return_value = (
        self._make_jobset(
            replicated_jobs=[
                self._make_replicated_job(
                    name="pathways-worker",
                    containers=[
                        {
                            "name": "pathways-worker",
                            "image": "server-image",
                        }
                    ],
                )
            ]
        )
    )
    gke_utils.get_pathways_service_images(
        pathways_service="my-jobset-pathways-head-0-0:8080",
        namespace="my-custom-ns",
    )
    mock_custom_api.get_namespaced_custom_object.assert_called_once_with(
        group="jobset.x-k8s.io",
        version="v1alpha2",
        namespace="my-custom-ns",
        plural="jobsets",
        name="my-jobset",
    )

  def test_get_pathways_service_images_invalid_namespace(self):
    with self.assertRaises(ValueError):
      gke_utils.get_pathways_service_images(
          pathways_service="my-jobset-pathways-head-0-0:8080",
          namespace="invalid namespace!",
      )

  def test_get_pathways_service_images_no_pathways_head_in_hostname(self):
    with self.assertRaises(ValueError):
      gke_utils.get_pathways_service_images(pathways_service="my-jobset:8080")

  def test_get_pathways_service_images_missing_server_image_raises(self):
    mock_custom_api = mock.MagicMock(spec=client.CustomObjectsApi)
    self.enter_context(
        mock.patch.object(
            gke_utils,
            "_get_k8s_custom_objects_api",
            return_value=mock_custom_api,
        )
    )
    mock_custom_api.get_namespaced_custom_object.return_value = (
        self._make_jobset(replicated_jobs=[])
    )
    with self.assertRaises(RuntimeError):
      gke_utils.get_pathways_service_images(
          pathways_service="my-jobset-pathways-head-0-0:8080"
      )

  def test_get_pathways_service_images_api_error_raises(self):
    mock_custom_api = mock.MagicMock(spec=client.CustomObjectsApi)
    self.enter_context(
        mock.patch.object(
            gke_utils,
            "_get_k8s_custom_objects_api",
            return_value=mock_custom_api,
        )
    )
    mock_custom_api.get_namespaced_custom_object.side_effect = (
        client.rest.ApiException(status=500, reason="API error")
    )
    with self.assertRaises(Exception):
      gke_utils.get_pathways_service_images(
          pathways_service="my-jobset-pathways-head-0-0:8080"
      )

  def test_get_compatible_proxy_server_image(self):
    test_cases = [
        (
            "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:jax-0.9.0",
            "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:jax-0.9.0",
        ),
        (
            "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:latest",
            "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:latest",
        ),
        (
            "us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/unsanitized_server:staging",
            "us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/unsanitized_proxy_server:staging",
        ),
        (
            "us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/sanitized_server:nightly",
            "us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/sanitized_proxy_server:nightly",
        ),
        (
            "us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/akshu/server:latest",
            "us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/akshu/proxy_server:latest",
        ),
        (
            "us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/akshu/unsanitized_server:latest",
            "us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/akshu/unsanitized_proxy_server:latest",
        ),
        (
            "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server",
            "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server",
        ),
        (
            "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server@sha256:12345",
            "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server@sha256:12345",
        ),
        (
            "custom.registry.com/pathways/my_custom_image:1.0",
            "custom.registry.com/pathways/my_custom_image:1.0",
        ),
        ("", ""),
    ]
    for server_img, expected_proxy_img in test_cases:
      with self.subTest(server_img=server_img):
        self.assertEqual(
            gke_utils.get_compatible_proxy_server_image(server_img),
            expected_proxy_img,
        )

  def test_start_port_forwarding_success(self):
    mock_popen = self.enter_context(
        mock.patch.object(subprocess, "Popen", autospec=True)
    )
    mock_pick_port = self.enter_context(
        mock.patch.object(portpicker, "pick_unused_port", autospec=True)
    )
    mock_pick_port.return_value = 29007
    mock_process = mock_popen.return_value

    port, process = gke_utils.start_port_forwarding("test-pod-123", 8080)

    self.assertEqual(port, 29007)
    self.assertEqual(process, mock_process)
    mock_popen.assert_called_once_with(
        [
            "kubectl",
            "port-forward",
            "-n",
            "default",
            "--address",
            "localhost",
            "--",
            "test-pod-123",
            "29007:8080",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

  def test_wait_for_port_forwarding_success(self):
    mock_create_connection = self.enter_context(
        mock.patch.object(socket, "create_connection", autospec=True)
    )
    mock_process = mock.create_autospec(subprocess.Popen, instance=True)
    mock_process.stdout = io.StringIO(
        "Forwarding from 127.0.0.1:29007 -> 8080\n"
    )

    gke_utils.wait_for_port_forwarding(mock_process, 29007)

    mock_create_connection.assert_called_once_with(
        ("localhost", 29007), timeout=30
    )
    mock_process.terminate.assert_not_called()

  def test_enable_port_forwarding_interrupted_terminates_process(self):
    mock_create_connection = self.enter_context(
        mock.patch.object(socket, "create_connection", autospec=True)
    )
    mock_popen = self.enter_context(
        mock.patch.object(subprocess, "Popen", autospec=True)
    )
    mock_pick_port = self.enter_context(
        mock.patch.object(portpicker, "pick_unused_port", autospec=True)
    )
    mock_pick_port.return_value = 29007
    mock_process = mock_popen.return_value
    mock_process.stdout = io.StringIO(
        "Forwarding from 127.0.0.1:29007 -> 8080\n"
    )
    mock_create_connection.side_effect = KeyboardInterrupt()

    with self.assertRaises(KeyboardInterrupt):
      gke_utils.enable_port_forwarding("test-pod-123", 8080)

    mock_process.terminate.assert_called_once()

  def test_terminate_process_graceful_termination(self):
    mock_proc = mock.create_autospec(subprocess.Popen, instance=True)
    mock_proc.pid = 12345

    gke_utils.terminate_process(mock_proc, timeout=5)

    mock_proc.terminate.assert_called_once()
    mock_proc.wait.assert_called_once_with(timeout=5)
    mock_proc.kill.assert_not_called()

  def test_terminate_process_timeout_falls_back_to_kill(self):
    mock_proc = mock.create_autospec(subprocess.Popen, instance=True)
    mock_proc.pid = 12345
    mock_proc.wait.side_effect = [
        subprocess.TimeoutExpired(cmd="test", timeout=5),
        0,
    ]

    gke_utils.terminate_process(mock_proc, timeout=5)

    mock_proc.terminate.assert_called_once()
    mock_proc.kill.assert_called_once()
    self.assertEqual(mock_proc.wait.call_count, 2)
    mock_proc.wait.assert_has_calls(
        [mock.call(timeout=5), mock.call(timeout=5)]
    )

  def test_terminate_process_kill_timeout_does_not_raise(self):
    mock_proc = mock.create_autospec(subprocess.Popen, instance=True)
    mock_proc.pid = 12345
    mock_proc.wait.side_effect = [
        subprocess.TimeoutExpired(cmd="test", timeout=5),
        subprocess.TimeoutExpired(cmd="test", timeout=5),
    ]

    gke_utils.terminate_process(mock_proc, timeout=5)

    mock_proc.terminate.assert_called_once()
    mock_proc.kill.assert_called_once()
    self.assertEqual(mock_proc.wait.call_count, 2)

  def test_terminate_process_handles_process_lookup_error(self):
    mock_proc = mock.create_autospec(subprocess.Popen, instance=True)
    mock_proc.pid = 12345
    mock_proc.terminate.side_effect = ProcessLookupError()

    gke_utils.terminate_process(mock_proc, timeout=5)

    mock_proc.terminate.assert_called_once()
    mock_proc.kill.assert_not_called()

  def test_terminate_process_none_is_noop(self):
    gke_utils.terminate_process(None)

  def test_terminate_process_custom_timeout(self):
    mock_proc = mock.create_autospec(subprocess.Popen, instance=True)
    mock_proc.pid = 12345

    gke_utils.terminate_process(mock_proc, timeout=10)

    mock_proc.terminate.assert_called_once()
    mock_proc.wait.assert_called_once_with(timeout=10)


if __name__ == "__main__":
  absltest.main()

