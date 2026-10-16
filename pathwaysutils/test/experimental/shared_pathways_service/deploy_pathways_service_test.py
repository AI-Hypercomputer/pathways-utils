"""Unit tests for the deploy_pathways_service script."""

import datetime
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
        max_restarts=30,
        max_slice_restarts=1000000,
        pathways_rm_and_worker_image="custom-server-image",
    )

    # Verify colocated python was added with correct image, SHM path, and
    # verbose logging env vars.
    mock_jobset.add_colocated_python.assert_called_once_with(
        image="custom-sidecar-image",
        shm_mount_path="/tmp/sidecar_dir",
        env=deploy_pathways_service._SIDECAR_DEBUG_ENV,
    )
    self.assertEqual(
        deploy_pathways_service._SIDECAR_DEBUG_ENV["LOGLEVEL"], "DEBUG"
    )

    # Verify deploy_func was called with the dict
    mock_deploy.assert_called_once_with({"metadata": {"name": "test-jobset"}})

  @parameterized.named_parameters(
      dict(testcase_name="none", sidecar_image=None),
      dict(testcase_name="empty", sidecar_image=""),
  )
  @mock.patch.object(gke_utils, "get_current_cluster_and_project")
  def test_run_deployment_without_sidecar_image_omits_sidecar(
      self, mock_detect, sidecar_image
  ):
    mock_detect.return_value = ("test-cluster", "test-project")
    mock_deploy = mock.MagicMock()

    deploy_pathways_service.run_deployment(
        tpu_type="v5e",
        topology="4x8",
        num_slices=2,
        jobset_name="test-jobset",
        gcs_bucket="gs://test-bucket",
        server_image="server-image",
        sidecar_image=sidecar_image,
        dry_run=False,
        deploy_func=mock_deploy,
    )

    mock_deploy.assert_called_once()
    jobset_config = mock_deploy.call_args.args[0]
    worker_job = next(
        j
        for j in jobset_config["spec"]["replicatedJobs"]
        if j["name"] == "pathways-worker"
    )
    worker_pod_spec = worker_job["template"]["spec"]["template"]["spec"]
    containers = worker_pod_spec.get("containers", []) + worker_pod_spec.get(
        "initContainers", []
    )
    self.assertNotIn(
        "colocated-python-sidecar", [c["name"] for c in containers]
    )
    worker_container = next(
        c for c in containers if c["name"] == "pathways-worker"
    )
    self.assertEmpty([
        arg
        for arg in worker_container.get("args", [])
        if arg.startswith("--cloud_pathways_sidecar_shm_directory")
    ])

  @mock.patch.object(deploy_pathways_service, "run_deployment", autospec=True)
  def test_main_deploys_without_sidecar_by_default(self, mock_run_deployment):
    deploy_pathways_service.main(["deploy_pathways_service"])

    mock_run_deployment.assert_called_once()
    self.assertIsNone(mock_run_deployment.call_args.kwargs["sidecar_image"])

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

  @mock.patch.object(gke_utils, "get_current_cluster_and_project")
  def test_run_deployment_max_restarts(self, mock_detect):
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
        max_restarts=3,
    )

    self.assertEqual(
        captured_config["spec"]["failurePolicy"],
        {"restartStrategy": "Recreate", "maxRestarts": 3},
    )

  def test_run_deployment_colocated_python_sidecar(self):
    captured_config = {}

    def capture_yaml_dump(config, *args, **kwargs):
      del args, kwargs
      captured_config.update(config)
      return ""

    with mock.patch.object(
        deploy_pathways_service.yaml, "dump", side_effect=capture_yaml_dump
    ):
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
          dry_run=True,
      )

    worker_job = next(
        j
        for j in captured_config["spec"]["replicatedJobs"]
        if j["name"] == "pathways-worker"
    )
    worker_pod_spec = worker_job["template"]["spec"]["template"]["spec"]
    containers = {c["name"]: c for c in worker_pod_spec["containers"]}
    init_containers = {c["name"]: c for c in worker_pod_spec["initContainers"]}
    # The sidecar runs as a native sidecar so it starts before pathways-worker.
    self.assertNotIn("colocated-python-sidecar", containers)
    self.assertIn("colocated-python-sidecar", init_containers)
    self.assertEqual(
        containers["pathways-worker"]["image"],
        "us-docker.pkg.dev/test-project/test-repo/server:test-tag",
    )
    self.assertIn(
        "--cloud_pathways_sidecar_shm_directory=/tmp/sidecar_dir",
        containers["pathways-worker"]["args"],
    )
    sidecar_env = {
        e["name"]: e["value"]
        for e in init_containers["colocated-python-sidecar"]["env"]
    }
    self.assertEqual(
        sidecar_env["CLOUD_PATHWAYS_SIDECAR_SHM_DIRECTORY"], "/tmp/sidecar_dir"
    )
    self.assertEqual(sidecar_env["LOGLEVEL"], "DEBUG")

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
    start_time = datetime.datetime.fromisoformat(
        kwargs["start_time"].replace("Z", "+00:00")
    )
    end_time = datetime.datetime.fromisoformat(
        kwargs["end_time"].replace("Z", "+00:00")
    )
    self.assertEqual(end_time - start_time, gke_utils.LOG_LINK_WINDOW)

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
