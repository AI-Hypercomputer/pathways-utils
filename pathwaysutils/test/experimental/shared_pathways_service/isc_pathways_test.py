"""Tests for the ISCPathways class.
"""
import io
import os
import signal
import subprocess
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from pathwaysutils.experimental.shared_pathways_service import isc_pathways



class ISCPathwaysTest(parameterized.TestCase):
  """Tests for the ISCPathways class."""

  def setUp(self):
    super().setUp()
    self.mock_stream_pod_logs = self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "stream_pod_logs", autospec=True
        )
    )

  def test_wait_for_placement_success(self):
    """Tests that _wait_for_placement correctly processes logs."""
    mock_process = mock.create_autospec(subprocess.Popen, instance=True)
    mock_process.stdout = io.StringIO(
        "Some log\nPlacement info\nTransition slice\nSignaling to RM\nunplaced"
        " -> placed\n"
    )
    mock_metrics_collector = mock.Mock()

    isc_pathways._wait_for_placement(
        mock_process,
        num_slices=1,
        metrics_collector_inst=mock_metrics_collector,
    )

    mock_metrics_collector.record_active_user.assert_called_once_with(True)

  def test_wait_for_placement_empty_stdout_raises(self):
    """Tests that _wait_for_placement terminates and raises if stdout is None."""
    mock_process = mock.create_autospec(subprocess.Popen, instance=True)
    mock_process.stdout = None
    mock_process.communicate.return_value = ("", "error details")

    with self.assertRaises(RuntimeError):
      isc_pathways._wait_for_placement(
          mock_process,
          num_slices=1,
          metrics_collector_inst=mock.Mock(),
      )
    mock_process.terminate.assert_called_once()

  def test_wait_for_placement_reports_metrics(self):
    """Tests that _wait_for_placement reports metrics on success."""
    mock_process = mock.create_autospec(subprocess.Popen, instance=True)
    mock_process.stdout = io.StringIO("Some log\nunplaced -> placed\n")

    mock_metrics_collector = mock.Mock()
    start_time = 100.0

    with mock.patch("time.time", return_value=150.0):
      isc_pathways._wait_for_placement(
          mock_process,
          num_slices=1,
          metrics_collector_inst=mock_metrics_collector,
          start_time=start_time,
      )

    mock_metrics_collector.record_assignment_time.assert_called_once_with(50.0)
    mock_metrics_collector.record_successful_request.assert_called_once()

  def test_deploy_pathways_proxy_server_success(self):
    mock_deploy_gke_yaml = self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "deploy_gke_yaml", autospec=True
        )
    )
    pathways_service = "test-service:8080"
    proxy_name = "test-proxy"
    expected_instances = {"tpuv6e:2x2": 2}
    gcs_bucket = "test-bucket"

    isc_pathways._deploy_pathways_proxy_server(
        pathways_service=pathways_service,
        proxy_job_name=proxy_name,
        expected_instances=expected_instances,
        gcs_scratch_location=gcs_bucket,
        proxy_server_image="test-image:latest",
        proxy_options=isc_pathways.ProxyOptions(use_insecure_credentials=False),
    )

    mock_deploy_gke_yaml.assert_called_once()
    substituted_yaml = mock_deploy_gke_yaml.call_args[0][0]
    self.assertIn("name: test-proxy", substituted_yaml)
    self.assertIn(
        "--resource_manager_address=test-service:8080", substituted_yaml
    )
    self.assertIn("--gcs_scratch_location=test-bucket", substituted_yaml)
    self.assertIn("--virtual_slices=tpuv6e:2x2,tpuv6e:2x2", substituted_yaml)
    self.assertIn('image: "test-image:latest"', substituted_yaml)
    # Extract the env section and check that it doesn't contain any - name:
    # entries.
    env_section = substituted_yaml.split("env:\n")[1].split("ports:")[0]
    self.assertNotIn("- name:", env_section)

  def test_deploy_pathways_proxy_server_image_escaping(self):
    mock_deploy_gke_yaml = self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "deploy_gke_yaml", autospec=True
        )
    )
    pathways_service = "test-service:8080"
    proxy_name = "test-proxy"
    expected_instances = {"tpuv6e:2x2": 1}
    gcs_bucket = "test-bucket"
    malicious_image = 'gcr.io/image:latest"\n- command: [pwn]\\\n'

    isc_pathways._deploy_pathways_proxy_server(
        pathways_service=pathways_service,
        proxy_job_name=proxy_name,
        expected_instances=expected_instances,
        gcs_scratch_location=gcs_bucket,
        proxy_server_image=malicious_image,
        proxy_options=isc_pathways.ProxyOptions(use_insecure_credentials=False),
    )

    mock_deploy_gke_yaml.assert_called_once()
    substituted_yaml = mock_deploy_gke_yaml.call_args[0][0]
    self.assertIn(
        'image: "gcr.io/image:latest\\"\\n- command: [pwn]\\\\\\n"',
        substituted_yaml,
    )


  def test_deploy_pathways_proxy_server_with_insecure_credentials_success(self):
    mock_deploy_gke_yaml = self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "deploy_gke_yaml", autospec=True
        )
    )
    pathways_service = "test-service:8080"
    proxy_name = "test-proxy"
    expected_instances = {"tpuv6e:2x2": 2}
    gcs_bucket = "test-bucket"

    isc_pathways._deploy_pathways_proxy_server(
        pathways_service=pathways_service,
        proxy_job_name=proxy_name,
        expected_instances=expected_instances,
        gcs_scratch_location=gcs_bucket,
        proxy_server_image="test-image:latest",
        proxy_options=isc_pathways.ProxyOptions(use_insecure_credentials=True),
    )

    mock_deploy_gke_yaml.assert_called_once()
    substituted_yaml = mock_deploy_gke_yaml.call_args[0][0]
    self.assertIn("env:", substituted_yaml)
    self.assertIn(
        "- name: IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS", substituted_yaml
    )
    self.assertIn('value: "true"', substituted_yaml)

  def test_deploy_pathways_proxy_server_with_xla_flags_success(self):
    mock_deploy_gke_yaml = self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "deploy_gke_yaml", autospec=True
        )
    )
    pathways_service = "test-service:8080"
    proxy_name = "test-proxy"
    expected_instances = {"tpuv6e:2x2": 1}
    gcs_bucket = "test-bucket"
    isc_pathways._deploy_pathways_proxy_server(
        pathways_service=pathways_service,
        proxy_job_name=proxy_name,
        expected_instances=expected_instances,
        gcs_scratch_location=gcs_bucket,
        proxy_server_image="test-image:latest",
        proxy_options=isc_pathways.ProxyOptions(
            xla_flags=["--xla_flag1", "--xla_flag2"]
        ),
    )
    mock_deploy_gke_yaml.assert_called_once()
    substituted_yaml = mock_deploy_gke_yaml.call_args[0][0]
    self.assertIn("- --xla_flag1", substituted_yaml)
    self.assertIn("- --xla_flag2", substituted_yaml)

  def test_deploy_pathways_proxy_server_with_sidecar_success(self):
    mock_deploy_gke_yaml = self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "deploy_gke_yaml", autospec=True
        )
    )
    pathways_service = "test-service:8080"
    proxy_name = "test-proxy"
    expected_instances = {"tpuv6e:2x2": 1}
    gcs_bucket = "test-bucket"
    isc_pathways._deploy_pathways_proxy_server(
        pathways_service=pathways_service,
        proxy_job_name=proxy_name,
        expected_instances=expected_instances,
        gcs_scratch_location=gcs_bucket,
        proxy_server_image="test-image:latest",
        proxy_options=isc_pathways.ProxyOptions(sidecar=True),
    )
    mock_deploy_gke_yaml.assert_called_once()
    substituted_yaml = mock_deploy_gke_yaml.call_args[0][0]
    self.assertIn("- --sidecar_name=external", substituted_yaml)

  def test_proxy_options_from_list(self):
    """Tests ProxyOptions.from_list with varied input formats."""
    # Standard valid input
    self.assertTrue(
        isc_pathways.ProxyOptions.from_list(
            ["use_insecure_credentials:true"]
        ).use_insecure_credentials
    )
    # Case sensitivity and whitespace
    self.assertTrue(
        isc_pathways.ProxyOptions.from_list(
            [" USE_INSECURE_CREDENTIALS : True "]
        ).use_insecure_credentials
    )
    # Valid false
    self.assertFalse(
        isc_pathways.ProxyOptions.from_list(
            ["use_insecure_credentials:false"]
        ).use_insecure_credentials
    )
    # Empty and None
    self.assertFalse(
        isc_pathways.ProxyOptions.from_list([]).use_insecure_credentials
    )
    self.assertFalse(
        isc_pathways.ProxyOptions.from_list(None).use_insecure_credentials
    )
    # Invalid formats and unknown keys
    self.assertFalse(
        isc_pathways.ProxyOptions.from_list(
            ["invalid_format", "unknown_key:value"]
        ).use_insecure_credentials
    )
    # Invalid value for known key
    self.assertFalse(
        isc_pathways.ProxyOptions.from_list(
            ["use_insecure_credentials:maybe"]
        ).use_insecure_credentials
    )

  def test_proxy_options_from_list_with_sidecar(self):
    """Tests ProxyOptions.from_list parsing the sidecar option."""
    self.assertTrue(
        isc_pathways.ProxyOptions.from_list(["sidecar:true"]).sidecar
    )
    self.assertTrue(
        isc_pathways.ProxyOptions.from_list([" SIDECAR : True "]).sidecar
    )
    self.assertFalse(
        isc_pathways.ProxyOptions.from_list(["sidecar:false"]).sidecar
    )
    self.assertFalse(
        isc_pathways.ProxyOptions.from_list([]).sidecar
    )
    self.assertFalse(
        isc_pathways.ProxyOptions.from_list(None).sidecar
    )

  @parameterized.named_parameters(
      (
          "standard_valid",
          ['xla_flags:"--xla_flag1 --xla_flag2"'],
          ["--xla_flag1", "--xla_flag2"],
      ),
      ("single_flag", ['xla_flags:"--xla_flag1"'], ["--xla_flag1"]),
      (
          "no_quotes",
          ["xla_flags:--xla_flag1 --xla_flag2"],
          ["--xla_flag1", "--xla_flag2"],
      ),
  )
  def test_proxy_options_from_list_with_xla_flags(
      self, input_list, expected_flags
  ):
    options = isc_pathways.ProxyOptions.from_list(input_list)
    self.assertEqual(options.xla_flags, expected_flags)

  def test_proxy_options_from_list_with_xla_flags_failure(self):
    with self.assertRaisesRegex(
        flags.ValidationError, "must start with '--xla_'"
    ):
      isc_pathways.ProxyOptions.from_list(["xla_flags:--not_xla_flag"])

  def test_deploy_pathways_proxy_server_failure(self):
    mock_deploy_gke_yaml = self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "deploy_gke_yaml", autospec=True
        )
    )

    mock_deploy_gke_yaml.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd="kubectl", stderr="error"
    )

    with self.assertRaises(subprocess.CalledProcessError):
      isc_pathways._deploy_pathways_proxy_server(
          pathways_service="service:1234",
          proxy_job_name="proxy",
          expected_instances={"tpuv6e:2x2": 1},
          gcs_scratch_location="bucket",
          proxy_server_image="test-image:latest",
      )

  def test_deploy_pathways_proxy_server_file_not_found(self):
    """Tests ValueError when PROXY_FILEPATH is not found.

    Ensures that _deploy_pathways_proxy_server raises a ValueError if the
    PROXY_FILEPATH cannot be read.
    """
    mock_open = self.enter_context(
        mock.patch("builtins.open", autospec=True)
    )
    mock_open.side_effect = OSError("File not found")

    with self.assertRaisesRegex(ValueError, "Could not read file:"):
      isc_pathways._deploy_pathways_proxy_server(
          pathways_service="service:1234",
          proxy_job_name="proxy",
          expected_instances={"tpuv6e:2x2": 1},
          gcs_scratch_location="bucket",
          proxy_server_image="test-image:latest",
      )

    # Ensure open was called with the expected file path.
    mock_open.assert_called_once_with(isc_pathways.PROXY_FILEPATH, "r")

  def test_isc_pathways_pod_wait_failure_raises_error(self):
    """Tests ISCPathways raises an error if pods do not become ready."""
    # Arrange
    mock_deploy = self.enter_context(
        mock.patch.object(
            isc_pathways, "_deploy_pathways_proxy_server",
            autospec=True,
        )
    )
    mock_run = self.enter_context(mock.patch("subprocess.run", autospec=True))
    self.enter_context(
        mock.patch("urllib.parse.quote", return_value="encoded_filter")
    )
    # Simulate 'kubectl get' returning a pod, 'kubectl wait' timing out,
    # then 'kubectl delete' succeeding.
    mock_get_pods_result = subprocess.CompletedProcess(
        args=["kubectl", "get", "pods"],
        returncode=0,
        stdout="pod/test-pod-123\n",
    )
    mock_delete_result = subprocess.CompletedProcess(
        args=["kubectl", "delete"],
        returncode=0,
    )
    mock_run.side_effect = [
        # First call for 'get'
        mock_get_pods_result,
        # Second call for 'wait'
        subprocess.TimeoutExpired(cmd="kubectl wait", timeout=30),
        # Third call for 'delete'
        mock_delete_result,
    ]

    # Act & Assert
    with self.assertRaises(RuntimeError) as context:
      with isc_pathways._ISCPathways(
          cluster="test-cluster",
          project="test-project",
          region="test-region",
          gcs_bucket="test-bucket",
          pathways_service="test-service:1234",
          expected_tpu_instances={"tpuv5:4x4x4": 1},
          proxy_job_name="test-proxy",
          proxy_server_image="test-image:latest",
      ):
        self.fail(
            "ISCPathways context should not be entered because we expect "
            "a RuntimeError to be raised."
        )

    self.assertIn("Pod did not become ready", str(context.exception))
    mock_deploy.assert_called_once()

    # Check that subprocess.run was called for get, wait, and delete.
    self.assertEqual(mock_run.call_count, 3)
    self.assertIn("get", mock_run.call_args_list[0][0][0])
    self.assertIn("wait", mock_run.call_args_list[1][0][0])
    self.assertIn("delete", mock_run.call_args_list[2][0][0])

  def test_isc_pathways(self):
    """Tests the full lifecycle of ISCPathways."""
    # Arrange
    self.enter_context(
        mock.patch(
            "pathwaysutils.experimental.shared_pathways_service.gke_utils.socket.create_connection",
            autospec=True,
        )
    )
    mock_pick_unused_port = self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils.portpicker, "pick_unused_port", autospec=True
        )
    )
    mock_deploy = self.enter_context(
        mock.patch.object(
            isc_pathways, "_deploy_pathways_proxy_server",
            autospec=True,
        )
    )
    mock_run = self.enter_context(mock.patch("subprocess.run", autospec=True))
    mock_popen = self.enter_context(
        mock.patch("subprocess.Popen", autospec=True)
    )
    mock_random = self.enter_context(
        mock.patch(
            "pathwaysutils.experimental.shared_pathways_service.isc_pathways.random",
            autospec=True,
        )
    )
    self.enter_context(
        mock.patch.dict(
            os.environ,
            {
                "USER": "testuser",
                "JAX_PLATFORMS": "original_platform",
                "JAX_BACKEND_TARGET": "original_target",
            },
        )
    )
    # Mock jax.config
    mock_jax_config = self.enter_context(
        mock.patch(
            "pathwaysutils.experimental.shared_pathways_service.isc_pathways.jax.config"
        )
    )
    mock_jax_config.jax_platforms = "original_platform_config"
    mock_jax_config.jax_backend_target = "original_target_config"
    mock_jax_config_update = mock_jax_config.update
    mock_clear_backends = self.enter_context(
        mock.patch(
            "pathwaysutils.experimental.shared_pathways_service.isc_pathways.jax_backend.clear_backends",
            autospec=True,
        )
    )
    mock_clear_caches = self.enter_context(
        mock.patch(
            "pathwaysutils.experimental.shared_pathways_service.isc_pathways.jax.clear_caches",
            autospec=True,
        )
    )
    mock_gc_collect = self.enter_context(
        mock.patch(
            "pathwaysutils.experimental.shared_pathways_service.isc_pathways.gc.collect",
            autospec=True,
        )
    )
    mock_metrics_collector = self.enter_context(
        mock.patch.object(
            isc_pathways.metrics_collector, "MetricsCollector", autospec=True
        )
    )
    mock_collector_instance = mock_metrics_collector.return_value
    mock_random.choices.return_value = list("abcde")
    mock_random.randint.return_value = 29005

    # Mock for 'kubectl wait' and 'kubectl delete'
    mock_success_result = subprocess.CompletedProcess(
        args=["kubectl"],
        returncode=0,
        stdout="",
    )

    # Mock for 'kubectl get pods'
    mock_get_pods_result = subprocess.CompletedProcess(
        args=["kubectl", "get", "pods"],
        returncode=0,
        stdout="pod/test-pod-123\n",
    )

    mock_run.side_effect = [
        mock_get_pods_result,  # For 'get_pod_from_job'
        mock_success_result,  # For 'check_pod_ready'
        mock_success_result,  # For 'kubectl delete job' in __exit__
    ]

    mock_pick_unused_port.return_value = 29007

    mock_process = mock_popen.return_value
    mock_process.stdout = io.StringIO(
        f"Forwarding from 127.0.0.1:{mock_pick_unused_port.return_value} ->"
        " 8080\n"
    )
    proxy_job_name = "test-proxy"

    # Act
    with isc_pathways._ISCPathways(
        cluster="test-cluster",
        project="test-project",
        region="test-region",
        gcs_bucket="test-bucket",
        pathways_service="test-service:1234",
        expected_tpu_instances={"tpuv5:4x4x4": 1},
        proxy_job_name=proxy_job_name,
        proxy_server_image="test-image:latest",
        proxy_options=isc_pathways.ProxyOptions(use_insecure_credentials=False),
        collect_service_metrics=True,
    ):
      # Assertions inside the context
      self.assertEqual(
          os.environ["JAX_PLATFORMS"],
          isc_pathways._JAX_PLATFORM_PROXY,
      )
      self.assertEqual(
          os.environ["JAX_BACKEND_TARGET"], "grpc://127.0.0.1:29007",
      )

      expected_jax_calls = [
          mock.call("jax_platforms", "proxy"),
          mock.call("jax_backend_target", "grpc://127.0.0.1:29007"),
      ]
      mock_jax_config_update.assert_has_calls(
          expected_jax_calls, any_order=True
      )
      self.assertEqual(
          os.environ.get("JAX_PLATFORMS"),
          isc_pathways._JAX_PLATFORM_PROXY,
      )
      self.assertEqual(
          os.environ.get("JAX_BACKEND_TARGET"),
          f"{isc_pathways._JAX_BACKEND_TARGET_HOSTNAME}:{mock_pick_unused_port.return_value}",
      )

    # Assertions outside the context (cleanup)
    self.assertEqual(
        os.environ.get("JAX_PLATFORMS"), "original_platform"
    )
    self.assertEqual(
        os.environ.get("JAX_BACKEND_TARGET"), "original_target"
    )

    restoration_calls = [
        mock.call("jax_platforms", "original_platform_config"),
        mock.call("jax_backend_target", "original_target_config"),
    ]
    mock_jax_config_update.assert_has_calls(restoration_calls, any_order=True)

    mock_clear_backends.assert_called_once()
    mock_deploy.assert_called_once()
    mock_clear_caches.assert_called_once()
    mock_gc_collect.assert_called_once()
    mock_metrics_collector.assert_called_once_with(
        "test-project", "test-cluster", "test-proxy"
    )
    mock_collector_instance.record_active_user.assert_not_called()
    mock_collector_instance.record_requested_capacity.assert_called_once_with(
        64
    )
    mock_run.assert_called_with(
        [
            "kubectl",
            "delete",
            "job",
            "-n",
            "default",
            "--ignore-not-found",
            "--",
            proxy_job_name,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

  def test_connect_success(self):
    """Tests that connect calls the dependencies and yields the manager."""
    # Arrange
    self.enter_context(mock.patch.dict(os.environ, {"USER": "testuser"}))
    mock_random = self.enter_context(
        mock.patch(
            "pathwaysutils.experimental.shared_pathways_service.isc_pathways.random",
            autospec=True,
        )
    )
    mock_random.choices.return_value = list("abcde")
    expected_proxy_job_name = "isc-proxy-testuser-abcde"
    mock_validate_tpu = self.enter_context(
        mock.patch.object(
            isc_pathways.validators, "validate_tpu_instances", autospec=True
        )
    )
    mock_fetch_creds = self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "fetch_cluster_credentials", autospec=True
        )
    )
    mock_get_images = self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "get_pathways_service_images", autospec=True
        )
    )
    mock_get_images.return_value = (
        "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:latest",
        None,
    )
    mock_isc_pathways = self.enter_context(
        mock.patch.object(isc_pathways, "_ISCPathways", autospec=True)
    )
    mock_thread = self.enter_context(
        mock.patch("threading.Thread", autospec=True)
    )

    cluster = "test-cluster"
    project = "test-project"
    region = "test-region"
    bucket = "test-bucket"
    pathways_service = "test-service-pathways-head:1234"
    expected_instances = {"tpuv5:4x4x4": 1}

    mock_manager_instance = (
        mock_isc_pathways.return_value.__enter__.return_value
    )
    mock_manager_instance.proxy_pod_name = "test-pod-123"
    mock_manager_instance.expected_tpu_instances = expected_instances
    mock_manager_instance.metrics_collector = mock.Mock()
    mock_manager_instance.start_time = 100.0

    # Act
    with isc_pathways.connect(
        cluster=cluster,
        project=project,
        region=region,
        gcs_bucket=bucket,
        pathways_service=pathways_service,
        expected_tpu_instances=expected_instances,
    ) as tm:
      # Assert
      mock_validate_tpu.assert_called_once_with(expected_instances)
      mock_fetch_creds.assert_called_once_with(
          cluster_name=cluster, project_id=project, location=region
      )
      mock_isc_pathways.assert_called_once_with(
          cluster=cluster,
          project=project,
          region=region,
          gcs_bucket=bucket,
          pathways_service=pathways_service,
          expected_tpu_instances=expected_instances,
          proxy_job_name=expected_proxy_job_name,
          proxy_server_image=isc_pathways.DEFAULT_PROXY_IMAGE,
          proxy_options=isc_pathways.ProxyOptions(),
          collect_service_metrics=False,
      )
      self.assertIs(tm, mock_manager_instance)

      # Verify log process captured and thread start
      self.mock_stream_pod_logs.assert_called_once_with("test-pod-123")
      self.assertIs(
          mock_manager_instance._log_process,
          self.mock_stream_pod_logs.return_value,
      )
      mock_thread.assert_called_once_with(
          target=isc_pathways._wait_for_placement,
          args=(
              self.mock_stream_pod_logs.return_value,
              1,
              mock_manager_instance.metrics_collector,
              mock_manager_instance.start_time,
              mock_manager_instance.total_chips,
          ),
          daemon=True,
      )
      mock_thread.return_value.start.assert_called_once()

  def test_connect_with_non_existent_cluster_raises_error(self):
    """Tests that connect raises an error if the cluster doesn't exist."""
    mock_fetch_creds = self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "fetch_cluster_credentials", autospec=True
        )
    )
    mock_fetch_creds.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd="gcloud", stderr="cluster not found"
    )
    with self.assertRaises(subprocess.CalledProcessError):
      with isc_pathways.connect(
          cluster="non-existent-cluster",
          project="test-project",
          region="test-zone",
          gcs_bucket="test-bucket",
          pathways_service="test-service:1234",
          expected_tpu_instances={"tpuv6e:2x2": 1},
      ):
        self.fail("ISCPathways context should not be entered.")

  def test_connect_with_tpuv3_raises_error(self):
    """Tests that connect raises an error for tpuv3 configurations."""
    with self.assertRaisesRegex(
        ValueError,
        "Unrecognized instance format: tpuv3:4x4.",
    ):
      with isc_pathways.connect(
          cluster="test-cluster",
          project="test-project",
          region="test-zone",
          gcs_bucket="test-bucket",
          pathways_service="test-service:1234",
          expected_tpu_instances={
              "tpuv3:4x4": 1
          },
      ):
        self.fail("ISCPathways context should not be entered.")

  def test_connect_passes_collect_service_metrics(self):
    """Tests that connect passes collect_service_metrics to _ISCPathways."""
    self.enter_context(mock.patch.dict(os.environ, {"USER": "testuser"}))
    self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "fetch_cluster_credentials", autospec=True
        )
    )
    mock_get_images = self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "get_pathways_service_images", autospec=True
        )
    )
    mock_get_images.return_value = (
        "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:latest",
        None,
    )
    mock_isc_pathways = self.enter_context(
        mock.patch.object(isc_pathways, "_ISCPathways", autospec=True)
    )
    self.enter_context(mock.patch("threading.Thread", autospec=True))
    self.enter_context(
        mock.patch.object(
            isc_pathways.validators, "validate_tpu_instances", autospec=True
        )
    )

    mock_manager_instance = (
        mock_isc_pathways.return_value.__enter__.return_value
    )
    mock_manager_instance.proxy_pod_name = "test-pod-123"
    mock_manager_instance.expected_tpu_instances = {"tpuv6e:2x2": 1}

    with isc_pathways.connect(
        cluster="test-cluster",
        project="test-project",
        region="test-region",
        gcs_bucket="test-bucket",
        pathways_service="test-service-pathways-head:1234",
        expected_tpu_instances={"tpuv6e:2x2": 1},
        collect_service_metrics=True,
    ):
      pass

    mock_isc_pathways.assert_called_once()
    _, kwargs = mock_isc_pathways.call_args
    self.assertTrue(kwargs["collect_service_metrics"])

  def test_connect_with_sidecar_validation_success(self):
    self.enter_context(mock.patch.dict(os.environ, {"USER": "testuser"}))
    self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "fetch_cluster_credentials", autospec=True
        )
    )
    mock_get_images = self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "get_pathways_service_images", autospec=True
        )
    )
    mock_get_images.return_value = (
        "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:latest",
        "us-docker.pkg.dev/.../sidecar:20260423-python_3.12-jax_0.10.0",
    )
    mock_validate_versions = self.enter_context(
        mock.patch.object(
            isc_pathways.validators,
            "validate_sidecar_image_versions",
            autospec=True,
        )
    )
    mock_isc_pathways = self.enter_context(
        mock.patch.object(isc_pathways, "_ISCPathways", autospec=True)
    )
    self.enter_context(mock.patch("threading.Thread", autospec=True))

    mock_manager_instance = (
        mock_isc_pathways.return_value.__enter__.return_value
    )
    mock_manager_instance.proxy_pod_name = "test-pod-123"
    mock_manager_instance.expected_tpu_instances = {"tpuv6e:2x2": 1}

    with isc_pathways.connect(
        cluster="test-cluster",
        project="test-project",
        region="test-region",
        gcs_bucket="test-bucket",
        pathways_service="test-service-pathways-head:1234",
        expected_tpu_instances={"tpuv6e:2x2": 1},
        proxy_options=["sidecar:true"],
    ):
      pass

    mock_get_images.assert_called_once_with(
        "test-service-pathways-head:1234"
    )
    mock_validate_versions.assert_called_once_with(
        "us-docker.pkg.dev/.../sidecar:20260423-python_3.12-jax_0.10.0"
    )

  def test_connect_with_sidecar_validation_mismatch_raises_error(self):
    self.enter_context(mock.patch.dict(os.environ, {"USER": "testuser"}))
    self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "fetch_cluster_credentials", autospec=True
        )
    )
    mock_get_images = self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "get_pathways_service_images", autospec=True
        )
    )
    mock_get_images.return_value = (
        "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:latest",
        "us-docker.pkg.dev/.../sidecar:20260423-python_3.12-jax_0.10.0",
    )
    mock_validate_versions = self.enter_context(
        mock.patch.object(
            isc_pathways.validators,
            "validate_sidecar_image_versions",
            autospec=True,
            side_effect=ValueError("Python version mismatch"),
        )
    )

    with self.assertRaisesRegex(ValueError, "Python version mismatch"):
      with isc_pathways.connect(
          cluster="test-cluster",
          project="test-project",
          region="test-region",
          gcs_bucket="test-bucket",
          pathways_service="test-service-pathways-head:1234",
          expected_tpu_instances={"tpuv6e:2x2": 1},
          proxy_options=["sidecar:true"],
      ):
        pass

    mock_get_images.assert_called_once_with(
        "test-service-pathways-head:1234"
    )
    mock_validate_versions.assert_called_once_with(
        "us-docker.pkg.dev/.../sidecar:20260423-python_3.12-jax_0.10.0"
    )

  @parameterized.named_parameters(
      ("no_underscore", "testuser", "testuser"),
      ("single_underscore", "test_user", "test"),
      ("multiple_underscores", "akshu_google_com", "akshu"),
      ("trailing_underscore", "user_", "user"),
  )
  def test_get_username(self, env_user, expected_username):
    with mock.patch.dict(os.environ, {"USER": env_user}):
      self.assertEqual(isc_pathways._get_username(), expected_username)

  def test_get_username_unset_returns_default(self):
    with mock.patch.dict(os.environ, clear=True):
      self.assertEqual(isc_pathways._get_username(), "user")

  def test_get_username_empty_returns_default(self):
    with mock.patch.dict(os.environ, {"USER": ""}):
      self.assertEqual(isc_pathways._get_username(), "user")

  def test_connect_with_underscore_username(self):
    """Tests that connect uses the portion before '_' for proxy job name."""
    self.enter_context(
        mock.patch.dict(os.environ, {"USER": "akshu_google_com"})
    )
    mock_random = self.enter_context(
        mock.patch(
            "pathwaysutils.experimental.shared_pathways_service.isc_pathways.random",
            autospec=True,
        )
    )
    mock_random.choices.return_value = list("abcde")
    expected_proxy_job_name = "isc-proxy-akshu-abcde"
    self.enter_context(
        mock.patch.object(
            isc_pathways.validators, "validate_tpu_instances", autospec=True
        )
    )
    self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "fetch_cluster_credentials", autospec=True
        )
    )
    mock_get_images = self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "get_pathways_service_images", autospec=True
        )
    )
    mock_get_images.return_value = (
        "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:latest",
        None,
    )
    mock_isc_pathways = self.enter_context(
        mock.patch.object(isc_pathways, "_ISCPathways", autospec=True)
    )
    self.enter_context(mock.patch("threading.Thread", autospec=True))

    mock_manager_instance = (
        mock_isc_pathways.return_value.__enter__.return_value
    )
    mock_manager_instance.proxy_pod_name = "test-pod-123"
    mock_manager_instance.expected_tpu_instances = {"tpuv6e:2x2": 1}

    with isc_pathways.connect(
        cluster="test-cluster",
        project="test-project",
        region="test-region",
        gcs_bucket="test-bucket",
        pathways_service="test-service-pathways-head:1234",
        expected_tpu_instances={"tpuv6e:2x2": 1},
    ):
      pass

    mock_isc_pathways.assert_called_once()
    _, kwargs = mock_isc_pathways.call_args
    self.assertEqual(kwargs["proxy_job_name"], expected_proxy_job_name)

  def test_connect_with_explicit_proxy_job_name(self):
    """Tests that connect uses explicitly provided proxy_job_name."""
    self.enter_context(mock.patch.dict(os.environ, {"USER": "testuser"}))
    self.enter_context(
        mock.patch.object(
            isc_pathways.validators, "validate_tpu_instances", autospec=True
        )
    )
    self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "fetch_cluster_credentials", autospec=True
        )
    )
    mock_get_images = self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "get_pathways_service_images", autospec=True
        )
    )
    mock_get_images.return_value = (
        "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:latest",
        None,
    )
    mock_isc_pathways = self.enter_context(
        mock.patch.object(isc_pathways, "_ISCPathways", autospec=True)
    )
    self.enter_context(mock.patch("threading.Thread", autospec=True))

    mock_manager_instance = (
        mock_isc_pathways.return_value.__enter__.return_value
    )
    mock_manager_instance.proxy_pod_name = "test-pod-123"
    mock_manager_instance.expected_tpu_instances = {"tpuv6e:2x2": 1}

    with isc_pathways.connect(
        cluster="test-cluster",
        project="test-project",
        region="test-region",
        gcs_bucket="test-bucket",
        pathways_service="test-service-pathways-head:1234",
        expected_tpu_instances={"tpuv6e:2x2": 1},
        proxy_job_name="custom-proxy-job",
    ):
      pass

    mock_isc_pathways.assert_called_once()
    _, kwargs = mock_isc_pathways.call_args
    self.assertEqual(kwargs["proxy_job_name"], "custom-proxy-job")

  def test_cleanup_idempotent(self):
    manager = isc_pathways._ISCPathways(
        cluster="test-cluster",
        project="test-project",
        region="test-region",
        gcs_bucket="test-bucket",
        pathways_service="test-service:1234",
        expected_tpu_instances={"tpuv6e:2x2": 1},
        proxy_job_name="test-proxy",
        proxy_server_image="test-image",
    )
    mock_delete = self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "delete_gke_resource", autospec=True
        )
    )
    manager._cleanup()
    mock_delete.assert_called_once_with("job", "test-proxy")
    # Second cleanup should be a no-op
    manager._cleanup()
    mock_delete.assert_called_once_with("job", "test-proxy")

  def test_enter_base_exception_triggers_cleanup(self):
    manager = isc_pathways._ISCPathways(
        cluster="test-cluster",
        project="test-project",
        region="test-region",
        gcs_bucket="test-bucket",
        pathways_service="test-service:1234",
        expected_tpu_instances={"tpuv6e:2x2": 1},
        proxy_job_name="test-proxy",
        proxy_server_image="test-image",
    )
    self.enter_context(
        mock.patch.object(
            isc_pathways,
            "_deploy_pathways_proxy_server",
            side_effect=KeyboardInterrupt(),
        )
    )
    mock_cleanup = self.enter_context(
        mock.patch.object(manager, "_cleanup", wraps=manager._cleanup)
    )

    with self.assertRaises(KeyboardInterrupt):
      manager.__enter__()

    mock_cleanup.assert_called_once()

  def test_signal_handler_triggers_cleanup_and_exits(self):
    manager = isc_pathways._ISCPathways(
        cluster="test-cluster",
        project="test-project",
        region="test-region",
        gcs_bucket="test-bucket",
        pathways_service="test-service:1234",
        expected_tpu_instances={"tpuv6e:2x2": 1},
        proxy_job_name="test-proxy",
        proxy_server_image="test-image",
    )
    mock_cleanup = self.enter_context(
        mock.patch.object(manager, "_cleanup", autospec=True)
    )

    manager._register_signal_handlers()
    try:
      self.assertIn(signal.SIGTERM, manager._original_signal_handlers)
      sigterm_handler = signal.getsignal(signal.SIGTERM)
      self.assertTrue(callable(sigterm_handler))
      assert callable(sigterm_handler)
      with mock.patch("sys.exit") as mock_exit:
        sigterm_handler(signal.SIGTERM, None)
        mock_cleanup.assert_called_once()
        mock_exit.assert_called_once_with(128 + signal.SIGTERM)
    finally:
      manager._restore_signal_handlers()

  def test_signal_handler_sigint_raises_keyboard_interrupt(self):
    manager = isc_pathways._ISCPathways(
        cluster="test-cluster",
        project="test-project",
        region="test-region",
        gcs_bucket="test-bucket",
        pathways_service="test-service:1234",
        expected_tpu_instances={"tpuv6e:2x2": 1},
        proxy_job_name="test-proxy",
        proxy_server_image="test-image",
    )
    mock_cleanup = self.enter_context(
        mock.patch.object(manager, "_cleanup", autospec=True)
    )

    manager._register_signal_handlers()
    try:
      self.assertIn(signal.SIGINT, manager._original_signal_handlers)
      sigint_handler = signal.getsignal(signal.SIGINT)
      self.assertTrue(callable(sigint_handler))
      assert callable(sigint_handler)
      with self.assertRaises(KeyboardInterrupt):
        sigint_handler(signal.SIGINT, None)
      mock_cleanup.assert_called_once()
    finally:
      manager._restore_signal_handlers()

  def test_enter_port_forward_interrupted_triggers_cleanup(self):
    manager = isc_pathways._ISCPathways(
        cluster="test-cluster",
        project="test-project",
        region="test-region",
        gcs_bucket="test-bucket",
        pathways_service="test-service:1234",
        expected_tpu_instances={"tpuv6e:2x2": 1},
        proxy_job_name="test-proxy",
        proxy_server_image="test-image",
    )
    self.enter_context(
        mock.patch.object(
            isc_pathways,
            "_deploy_pathways_proxy_server",
            autospec=True,
        )
    )
    self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils,
            "wait_for_pod",
            return_value="test-pod-123",
            autospec=True,
        )
    )
    mock_pf_proc = mock.create_autospec(subprocess.Popen, instance=True)
    self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils,
            "start_port_forwarding",
            return_value=(29007, mock_pf_proc),
            autospec=True,
        )
    )
    self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils,
            "wait_for_port_forwarding",
            side_effect=KeyboardInterrupt(),
            autospec=True,
        )
    )
    mock_delete = self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "delete_gke_resource", autospec=True
        )
    )
    mock_terminate = self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "terminate_process", autospec=True
        )
    )
    with self.assertRaises(KeyboardInterrupt):
      manager.__enter__()
    mock_delete.assert_called_once_with("job", "test-proxy")
    mock_terminate.assert_called_once_with(
        mock_pf_proc, process_name="Port forwarding"
    )

  def test_cleanup_terminates_port_forward_and_log_processes(self):
    manager = isc_pathways._ISCPathways(
        cluster="test-cluster",
        project="test-project",
        region="test-region",
        gcs_bucket="test-bucket",
        pathways_service="test-service:1234",
        expected_tpu_instances={"tpuv6e:2x2": 1},
        proxy_job_name="test-proxy",
        proxy_server_image="test-image",
    )
    mock_pf_proc = mock.create_autospec(subprocess.Popen, instance=True)
    mock_log_proc = mock.create_autospec(subprocess.Popen, instance=True)
    manager._port_forward_process = mock_pf_proc
    manager._log_process = mock_log_proc

    mock_terminate = self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "terminate_process", autospec=True
        )
    )
    self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "delete_gke_resource", autospec=True
        )
    )

    manager._cleanup()

    mock_terminate.assert_has_calls([
        mock.call(mock_pf_proc, process_name="Port forwarding"),
        mock.call(mock_log_proc, process_name="Log streaming"),
    ])
    self.assertIsNone(manager._port_forward_process)
    self.assertIsNone(manager._log_process)

  def test_connect_auto_detect_proxy_image_success(self):
    """Tests that connect auto-detects compatible proxy image when not provided."""
    self.enter_context(mock.patch.dict(os.environ, {"USER": "testuser"}))
    self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "fetch_cluster_credentials", autospec=True
        )
    )
    mock_get_images = self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "get_pathways_service_images", autospec=True
        )
    )
    mock_get_images.return_value = (
        "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:jax-0.9.0",
        None,
    )
    mock_isc_pathways = self.enter_context(
        mock.patch.object(isc_pathways, "_ISCPathways", autospec=True)
    )
    self.enter_context(mock.patch("threading.Thread", autospec=True))

    with isc_pathways.connect(
        cluster="test-cluster",
        project="test-project",
        region="test-region",
        gcs_bucket="test-bucket",
        pathways_service="test-service-pathways-head:1234",
        expected_tpu_instances={"tpuv6e:2x2": 1},
    ):
      pass

    mock_isc_pathways.assert_called_once()
    _, kwargs = mock_isc_pathways.call_args
    self.assertEqual(
        kwargs["proxy_server_image"],
        "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:jax-0.9.0",
    )

  def test_connect_incompatible_proxy_image_replaced_with_warning(self):
    """Tests that an incompatible proxy image is replaced with compatible one and logs a warning."""
    self.enter_context(mock.patch.dict(os.environ, {"USER": "testuser"}))
    self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "fetch_cluster_credentials", autospec=True
        )
    )
    mock_get_images = self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "get_pathways_service_images", autospec=True
        )
    )
    mock_get_images.return_value = (
        "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:jax-0.9.0",
        None,
    )
    mock_isc_pathways = self.enter_context(
        mock.patch.object(isc_pathways, "_ISCPathways", autospec=True)
    )
    self.enter_context(mock.patch("threading.Thread", autospec=True))

    with self.assertLogs(isc_pathways._logger, level="WARNING") as log_cm:
      with isc_pathways.connect(
          cluster="test-cluster",
          project="test-project",
          region="test-region",
          gcs_bucket="test-bucket",
          pathways_service="test-service-pathways-head:1234",
          expected_tpu_instances={"tpuv6e:2x2": 1},
          proxy_server_image="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:latest",
      ):
        pass

    self.assertTrue(any("incompatible" in msg for msg in log_cm.output))
    mock_isc_pathways.assert_called_once()
    _, kwargs = mock_isc_pathways.call_args
    self.assertEqual(
        kwargs["proxy_server_image"],
        "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:jax-0.9.0",
    )

  def test_connect_compatible_proxy_image_no_warning(self):
    """Tests that a compatible proxy image is used without replacement warning."""
    self.enter_context(mock.patch.dict(os.environ, {"USER": "testuser"}))
    self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "fetch_cluster_credentials", autospec=True
        )
    )
    mock_get_images = self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "get_pathways_service_images", autospec=True
        )
    )
    mock_get_images.return_value = (
        "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:jax-0.9.0",
        None,
    )
    mock_isc_pathways = self.enter_context(
        mock.patch.object(isc_pathways, "_ISCPathways", autospec=True)
    )
    self.enter_context(mock.patch("threading.Thread", autospec=True))

    with isc_pathways.connect(
        cluster="test-cluster",
        project="test-project",
        region="test-region",
        gcs_bucket="test-bucket",
        pathways_service="test-service-pathways-head:1234",
        expected_tpu_instances={"tpuv6e:2x2": 1},
        proxy_server_image="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:jax-0.9.0",
    ):
      pass

    mock_isc_pathways.assert_called_once()
    _, kwargs = mock_isc_pathways.call_args
    self.assertEqual(
        kwargs["proxy_server_image"],
        "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:jax-0.9.0",
    )

  def test_connect_autodetect_fails_fallback_to_default(self):
    """Tests that connect raises an error when auto-detection fails."""
    self.enter_context(mock.patch.dict(os.environ, {"USER": "testuser"}))
    self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "fetch_cluster_credentials", autospec=True
        )
    )
    mock_get_images = self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "get_pathways_service_images", autospec=True
        )
    )
    mock_get_images.side_effect = RuntimeError("Failed to get server image")

    with self.assertRaises(RuntimeError):
      with isc_pathways.connect(
          cluster="test-cluster",
          project="test-project",
          region="test-region",
          gcs_bucket="test-bucket",
          pathways_service="test-service-pathways-head:1234",
          expected_tpu_instances={"tpuv6e:2x2": 1},
      ):
        pass

  def test_connect_proxy_server_image_deprecation_warning(self):
    """Tests that passing proxy_server_image emits a DeprecationWarning."""
    self.enter_context(mock.patch.dict(os.environ, {"USER": "testuser"}))
    self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "fetch_cluster_credentials", autospec=True
        )
    )
    mock_get_images = self.enter_context(
        mock.patch.object(
            isc_pathways.gke_utils, "get_pathways_service_images", autospec=True
        )
    )
    mock_get_images.return_value = (
        "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:jax-0.9.0",
        None,
    )
    self.enter_context(
        mock.patch.object(isc_pathways, "_ISCPathways", autospec=True)
    )
    self.enter_context(mock.patch("threading.Thread", autospec=True))

    with self.assertWarns(DeprecationWarning):
      with isc_pathways.connect(
          cluster="test-cluster",
          project="test-project",
          region="test-region",
          gcs_bucket="test-bucket",
          pathways_service="test-service-pathways-head:1234",
          expected_tpu_instances={"tpuv6e:2x2": 1},
          proxy_server_image="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:jax-0.9.0",
      ):
        pass


if __name__ == "__main__":
  absltest.main()
