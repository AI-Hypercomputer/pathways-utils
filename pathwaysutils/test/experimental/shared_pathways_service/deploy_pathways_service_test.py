"""Unit tests for the deploy_pathways_service script."""

from unittest import mock
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from pathwaysutils.experimental.shared_pathways_service import deploy_pathways_service
from pathwaysutils.experimental.shared_pathways_service import gke_utils


class DeployPathwaysServiceTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="v5p",
          tpu_type="v5p",
          expected_machine_type="ct5p-hightpu-4t",
      ),
      dict(
          testcase_name="v5e",
          tpu_type="v5e",
          expected_machine_type="ct5lp-hightpu-4t",
      ),
      dict(
          testcase_name="v6e",
          tpu_type="v6e",
          expected_machine_type="ct6e-standard-4t",
      ),
      dict(
          testcase_name="tpu7x",
          tpu_type="tpu7x",
          expected_machine_type="tpu7x-standard-4t",
      ),
  )
  def test_get_tpu_config_valid(self, tpu_type, expected_machine_type):
    config = deploy_pathways_service.get_tpu_config(tpu_type)
    self.assertEqual(config.machine_type, expected_machine_type)

  @parameterized.named_parameters(
      dict(
          testcase_name="invalid_tpu_type",
          tpu_type="invalid",
      ),
      dict(
          testcase_name="empty_tpu_type",
          tpu_type="",
      ),
      dict(
          testcase_name="whitespace_tpu_type",
          tpu_type="  ",
      ),
      dict(
          testcase_name="v5_tpu_type",
          tpu_type="v5",
      ),
  )
  def test_get_tpu_config_invalid(self, tpu_type):
    with self.assertRaises(ValueError):
      deploy_pathways_service.get_tpu_config(tpu_type)

  def test_calculate_vms_per_slice_valid(self):
    vms = deploy_pathways_service.calculate_vms_per_slice("4x8", 4)
    self.assertEqual(vms, 8)

  def test_calculate_vms_per_slice_invalid_format(self):
    with self.assertRaises(ValueError):
      deploy_pathways_service.calculate_vms_per_slice("4x8x", 4)

  def test_calculate_vms_per_slice_not_divisible(self):
    with self.assertRaises(ValueError):
      deploy_pathways_service.calculate_vms_per_slice("4x8", 5)

  @mock.patch.object(gke_utils, "get_current_cluster_and_project")
  @mock.patch("pathwaysutils.experimental.shared_pathways_service.deploy_pathways_service.jobset.PathwaysJobSet")
  def test_run_deployment(self, mock_jobset_cls, mock_detect):
    mock_detect.return_value = ("test-cluster", "test-project")
    mock_jobset = mock_jobset_cls.return_value
    mock_jobset.to_dict.return_value = {"metadata": {"name": "test-jobset"}}

    # Mock head and worker job templates for mutation
    mock_head_job = mock.MagicMock()
    mock_head_job.spec.template.spec.containers = [
        mock.MagicMock(name="pathways-rm")
    ]
    mock_head_job.spec.template.spec.containers[0].name = "pathways-rm"

    mock_worker_job = mock.MagicMock()
    mock_worker_job.spec.template.spec.containers = [
        mock.MagicMock(name="pathways-worker")
    ]
    mock_worker_job.spec.template.spec.containers[0].name = "pathways-worker"
    mock_worker_job.spec.template.spec.containers[0].args = []
    
    # Sidecar will be added by add_colocated_python, but we mock it here as if it was added
    mock_sidecar = mock.MagicMock(name="colocated-python-sidecar")
    mock_sidecar.name = "colocated-python-sidecar"
    mock_sidecar.env = []
    mock_worker_job.spec.template.spec.init_containers = [mock_sidecar]

    mock_jobset.head_job_template = mock_head_job
    mock_jobset.worker_job_template = mock_worker_job

    mock_deploy = mock.MagicMock()

    deploy_pathways_service.run_deployment(
        tpu_type="v5e",
        topology="4x8",
        num_slices=2,
        jobset_name="test-jobset",
        gcs_bucket="test-bucket",
        server_image="custom-server-image",
        sidecar_image="custom-sidecar-image",
        dry_run=False,
        deploy_func=mock_deploy,
    )

    # Verify PathwaysJobSet was instantiated correctly
    mock_jobset_cls.assert_called_once_with(
        name="test-jobset",
        namespace="default",
        pathways_dir="test-bucket",
        tpu_type="v5e",
        topology="4x8",
        num_slices=2,
        shared_pathways_service=True,
        max_slice_restarts=1000000,
    )

    # Verify colocated python was added with correct image and SHM path
    mock_jobset.add_colocated_python.assert_called_once_with(
        image="custom-sidecar-image",
        shm_mount_path="/tmp/sidecar_dir",
    )

    # Verify server images were mutated
    self.assertEqual(mock_head_job.spec.template.spec.containers[0].image, "custom-server-image")
    self.assertEqual(mock_worker_job.spec.template.spec.containers[0].image, "custom-server-image")

    # Verify extra logging env vars were added to sidecar
    self.assertTrue(any(e.name == "LOGLEVEL" and e.value == "DEBUG" for e in mock_sidecar.env))

    # Verify arg was added to pathways-worker
    self.assertIn(
        "--cloud_pathways_sidecar_shm_directory=/tmp/sidecar_dir",
        mock_worker_job.spec.template.spec.containers[0].args,
    )

    # Verify deploy_func was called with the dict
    mock_deploy.assert_called_once_with({"metadata": {"name": "test-jobset"}})

  @mock.patch.object(gke_utils, "get_current_cluster_and_project")
  def test_run_deployment_worker_backoff_limit(self, mock_detect):
    mock_detect.return_value = ("test-cluster", "test-project")
    captured_config = {}

    def capture_deploy(config):
      nonlocal captured_config
      captured_config = config

    deploy_pathways_service.run_deployment(
        tpu_type="v5e",
        topology="4x8",
        num_slices=2,
        jobset_name="test-jobset",
        gcs_bucket="gs://test-bucket",
        server_image=(
            "us-docker.pkg.dev/test-project/test-repo/server:test-tag"
        ),
        sidecar_image=(
            "us-docker.pkg.dev/test-project/test-repo/sidecar:test-tag"
        ),
        dry_run=False,
        deploy_func=capture_deploy,
    )

    replicated_jobs = captured_config["spec"]["replicatedJobs"]
    worker_job = next(
        j for j in replicated_jobs if j["name"] == "pathways-worker"
    )
    worker_backoff = worker_job["template"]["spec"]["backoffLimit"]

    # Verify worker backoff limit is set to a large value
    self.assertGreaterEqual(worker_backoff, 1000000)

  @mock.patch.object(gke_utils, "get_log_link")
  @mock.patch.object(gke_utils, "get_current_cluster_and_project")
  def test_run_deployment_cloud_logging_link(
      self, mock_detect, mock_get_log_link
  ):
    mock_detect.return_value = ("test-cluster", "test-project")
    mock_get_log_link.return_value = (
        "https://console.cloud.google.com/logs/query;dummy"
    )
    mock_deploy = mock.MagicMock()

    deploy_pathways_service.run_deployment(
        tpu_type="v5e",
        topology="4x8",
        num_slices=2,
        jobset_name="test-jobset",
        gcs_bucket="gs://test-bucket",
        server_image="server-image",
        sidecar_image="sidecar-image",
        dry_run=False,
        deploy_func=mock_deploy,
    )

    mock_detect.assert_called_once()
    mock_get_log_link.assert_called_once()
    _, kwargs = mock_get_log_link.call_args
    self.assertEqual(kwargs["cluster"], "test-cluster")
    self.assertEqual(kwargs["project"], "test-project")
    self.assertEqual(kwargs["job_name"], "test-jobset")
    self.assertIsNotNone(kwargs["start_time"])
    self.assertIsNotNone(kwargs["end_time"])

  @parameterized.named_parameters(
      dict(testcase_name="neither", cluster=None, project=None),
      dict(testcase_name="no_cluster", cluster=None, project="test-project"),
      dict(testcase_name="no_project", cluster="test-cluster", project=None),
  )
  @mock.patch.object(gke_utils, "get_log_link")
  @mock.patch.object(gke_utils, "get_current_cluster_and_project")
  def test_run_deployment_missing_cluster_or_project_raises(
      self, mock_detect, mock_get_log_link, cluster, project
  ):
    mock_detect.return_value = (cluster, project)
    mock_deploy = mock.MagicMock()

    with self.assertRaises(ValueError):
      deploy_pathways_service.run_deployment(
          tpu_type="v5e",
          topology="4x8",
          num_slices=2,
          jobset_name="test-jobset",
          gcs_bucket="gs://test-bucket",
          server_image="server-image",
          sidecar_image="sidecar-image",
          dry_run=False,
          deploy_func=mock_deploy,
      )

    mock_detect.assert_called_once()
    mock_get_log_link.assert_not_called()
    mock_deploy.assert_not_called()

  @mock.patch.object(gke_utils, "get_log_link")
  def test_run_deployment_dry_run_no_log_link(self, mock_get_log_link):
    mock_deploy = mock.MagicMock()

    deploy_pathways_service.run_deployment(
        tpu_type="v5e",
        topology="4x8",
        num_slices=2,
        jobset_name="test-jobset",
        gcs_bucket="gs://test-bucket",
        server_image="server-image",
        sidecar_image="sidecar-image",
        dry_run=True,
        deploy_func=mock_deploy,
    )

    mock_get_log_link.assert_not_called()
    mock_deploy.assert_not_called()


if __name__ == "__main__":
  FLAGS = flags.FLAGS
  FLAGS.jobset_name = "dummy"
  FLAGS.jax_version = "dummy"
  FLAGS.tpu_type = "v5e"
  FLAGS.topology = "4x8"
  FLAGS.num_slices = 2
  FLAGS.gcs_bucket = "dummy"
  absltest.main()
