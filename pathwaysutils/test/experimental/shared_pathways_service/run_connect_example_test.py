from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
import numpy as np


class RunConnectExampleTest(absltest.TestCase):
  """Tests the logic from the run_connect_example.py script."""

  @flagsaver.flagsaver(
      cluster="random-cluster-name",
      project="random-project-id",
      region="random-region",
      gcs_bucket="random-bucket",
      pathways_service="random-pathways-service:1234",
      tpu_type="tpuv6e:2x2",
      tpu_count=2,
  )
  def test_run_connect_example_main(self):
    """Tests that the main function calls connect and executes the logic."""
    # Import inside the test to avoid flag parsing errors on module load.
    from pathwaysutils.experimental.shared_pathways_service import run_connect_example

    mock_connect = self.enter_context(
        mock.patch.object(
            run_connect_example.isc_pathways, "connect", autospec=True
        )
    )
    mock_pprint = self.enter_context(mock.patch("pprint.pprint", autospec=True))

    run_connect_example.main(["unused_argv"])

    mock_connect.assert_called_once_with(
        cluster="random-cluster-name",
        project="random-project-id",
        region="random-region",
        gcs_bucket="random-bucket",
        pathways_service="random-pathways-service:1234",
        expected_tpu_instances={"tpuv6e:2x2": 2},
        proxy_job_name=None,
        proxy_server_image=(
            "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:latest"
        ),
        proxy_options=None,
        collect_service_metrics=False,
    )
    self.assertEqual(mock_pprint.call_count, 2)
    np.testing.assert_array_equal(
        mock_pprint.call_args_list[0].args[0], np.zeros(5)
    )
    np.testing.assert_array_equal(
        mock_pprint.call_args_list[1].args[0], np.ones(5)
    )

  @flagsaver.flagsaver(
      cluster="random-cluster-name",
      project="random-project-id",
      region="random-region",
      gcs_bucket="random-bucket",
      pathways_service="random-pathways-service:1234",
      tpu_type="tpuv6e:2x2",
      tpu_count=2,
      proxy_job_name="test-job-name",
      proxy_server_image="test-image",
      proxy_options=["use_insecure_credentials:true"],
  )
  def test_run_connect_example_main_with_optional_flags(self):
    """Tests that main passes optional flags to connect."""
    # Import inside the test to avoid flag parsing errors on module load.
    from pathwaysutils.experimental.shared_pathways_service import run_connect_example

    mock_connect = self.enter_context(
        mock.patch.object(
            run_connect_example.isc_pathways, "connect", autospec=True
        )
    )
    self.enter_context(mock.patch("pprint.pprint", autospec=True))

    run_connect_example.main(["unused_argv"])

    mock_connect.assert_called_once_with(
        cluster="random-cluster-name",
        project="random-project-id",
        region="random-region",
        gcs_bucket="random-bucket",
        pathways_service="random-pathways-service:1234",
        expected_tpu_instances={"tpuv6e:2x2": 2},
        proxy_job_name="test-job-name",
        proxy_server_image="test-image",
        proxy_options=["use_insecure_credentials:true"],
        collect_service_metrics=False,
    )

  @flagsaver.flagsaver(
      cluster="random-cluster-name",
      project="random-project-id",
      region="random-region",
      gcs_bucket="random-bucket",
      pathways_service="random-pathways-service:1234",
      tpu_type="tpuv6e:2x2",
      tpu_count=2,
      collect_service_metrics=True,
  )
  def test_run_connect_example_main_with_metrics_enabled(self):
    """Tests that main passes collect_service_metrics flag to connect."""
    from pathwaysutils.experimental.shared_pathways_service import run_connect_example

    mock_connect = self.enter_context(
        mock.patch.object(
            run_connect_example.isc_pathways, "connect", autospec=True
        )
    )
    self.enter_context(mock.patch("pprint.pprint", autospec=True))

    run_connect_example.main(["unused_argv"])

    mock_connect.assert_called_once_with(
        cluster="random-cluster-name",
        project="random-project-id",
        region="random-region",
        gcs_bucket="random-bucket",
        pathways_service="random-pathways-service:1234",
        expected_tpu_instances={"tpuv6e:2x2": 2},
        proxy_job_name=None,
        proxy_server_image=(
            "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:latest"
        ),
        proxy_options=None,
        collect_service_metrics=True,
    )


if __name__ == "__main__":
  absltest.main()
