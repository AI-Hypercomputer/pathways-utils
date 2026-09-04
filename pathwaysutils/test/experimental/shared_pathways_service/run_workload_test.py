import contextlib
import subprocess
import sys
import types
from typing import Any, Generator
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from pathwaysutils.experimental.shared_pathways_service import run_workload


class FakeConnect:
  """A fake connection manager that tracks entry and exit."""

  def __init__(self, **kwargs: Any):
    self.kwargs = kwargs
    self.entered = False
    self.exited = False

  def __enter__(self) -> "FakeConnect":
    self.entered = True
    return self

  def __exit__(
      self,
      exc_type: type[BaseException] | None,
      exc_val: BaseException | None,
      exc_tb: types.TracebackType | None,
  ) -> None:
    self.exited = True


class RunTpuWorkloadTest(absltest.TestCase):

  def test_run_workload_success(self):
    fake_instances = []

    @contextlib.contextmanager
    def fake_connect_fn(**kwargs: Any) -> Generator[FakeConnect, None, None]:
      fake = FakeConnect(**kwargs)
      fake_instances.append(fake)
      with fake as f:
        yield f

    mock_popen = self.enter_context(
        mock.patch.object(subprocess, "Popen", autospec=True)
    )
    mock_proc = mock_popen.return_value
    mock_proc.wait.return_value = 0

    run_workload.run_command(
        cluster="test-cluster",
        project="test-project",
        region="test-region",
        gcs_bucket="test-bucket",
        pathways_service="test-service:1234",
        tpu_type="tpuv6e:4x8",
        tpu_count=1,
        command="echo hello",
        connect_fn=fake_connect_fn,
    )

    with self.subTest("Connect function called correctly"):
      self.assertLen(fake_instances, 1)
      fake = fake_instances[0]
      self.assertEqual(fake.kwargs["cluster"], "test-cluster")
      self.assertEqual(fake.kwargs["expected_tpu_instances"], {"tpuv6e:4x8": 1})

    with self.subTest("Context manager lifecycle"):
      fake = fake_instances[0]
      self.assertTrue(fake.entered)
      self.assertTrue(fake.exited)

    with self.subTest("Command executed"):
      mock_popen.assert_called_once_with(
          ["echo", "hello"], env=mock.ANY
      )
      mock_proc.wait.assert_called_once()

  def test_run_command_runs_command_inside_context(self):
    """Verifies that the command is executed while the connection is active."""
    connection_active_during_run = False

    @contextlib.contextmanager
    def fake_connect_fn(**kwargs: Any) -> Generator[None, None, None]:
      del kwargs
      yield None

    def mock_wait_side_effect(*args: Any, **kwargs: Any) -> int:
      nonlocal connection_active_during_run
      connection_active_during_run = True
      return 0

    mock_popen = self.enter_context(
        mock.patch.object(subprocess, "Popen", autospec=True)
    )
    mock_proc = mock_popen.return_value
    mock_proc.wait.side_effect = mock_wait_side_effect

    run_workload.run_command(
        cluster="test-cluster",
        project="test-project",
        region="test-region",
        gcs_bucket="test-bucket",
        pathways_service="test-service:1234",
        tpu_type="tpuv6e:4x8",
        tpu_count=1,
        command="echo hello",
        connect_fn=fake_connect_fn,
    )

    self.assertTrue(connection_active_during_run)

  def test_run_command_error(self):

    @contextlib.contextmanager
    def fake_connect_fn(**kwargs: Any) -> Generator[None, None, None]:
      del kwargs
      yield None

    mock_popen = self.enter_context(
        mock.patch.object(subprocess, "Popen", autospec=True)
    )
    mock_proc = mock_popen.return_value
    mock_proc.wait.return_value = 1

    with self.assertRaises(subprocess.CalledProcessError):
      run_workload.run_command(
          cluster="test-cluster",
          project="test-project",
          region="test-region",
          gcs_bucket="test-bucket",
          pathways_service="test-service:1234",
          tpu_type="tpuv6e:4x8",
          tpu_count=1,
          command="false",
          connect_fn=fake_connect_fn,
      )

  def test_run_command_interrupted_terminates_subprocess(self):

    @contextlib.contextmanager
    def fake_connect_fn(**kwargs: Any) -> Generator[None, None, None]:
      del kwargs
      yield None

    mock_popen = self.enter_context(
        mock.patch.object(subprocess, "Popen", autospec=True)
    )
    mock_proc = mock_popen.return_value
    mock_proc.wait.side_effect = KeyboardInterrupt()

    with self.assertRaises(KeyboardInterrupt):
      run_workload.run_command(
          cluster="test-cluster",
          project="test-project",
          region="test-region",
          gcs_bucket="test-bucket",
          pathways_service="test-service:1234",
          tpu_type="tpuv6e:4x8",
          tpu_count=1,
          command="sleep 100",
          connect_fn=fake_connect_fn,
      )

    mock_proc.terminate.assert_called_once()

  @flagsaver.flagsaver(
      cluster="test-cluster",
      project="test-project",
      region="test-region",
      gcs_bucket="test-bucket",
      pathways_service="test-service:1234",
      tpu_type="tpuv6e:4x8",
      tpu_count=1,
      command="echo hello",
  )
  def test_main_calls_run_command(self):
    with mock.patch.object(
        run_workload, "run_command", autospec=True
    ) as mock_run_command:
      run_workload.main(["unused_argv"])
      mock_run_command.assert_called_once_with(
          cluster="test-cluster",
          project="test-project",
          region="test-region",
          gcs_bucket="test-bucket",
          pathways_service="test-service:1234",
          tpu_type="tpuv6e:4x8",
          tpu_count=1,
          command="echo hello",
          proxy_server_image="",
          proxy_options=[],
          collect_service_metrics=False,
      )

  @flagsaver.flagsaver(
      cluster="test-cluster",
      project="test-project",
      region="test-region",
      gcs_bucket="test-bucket",
      pathways_service="test-service:1234",
      tpu_type="tpuv6e:4x8",
      tpu_count=1,
      command="echo hello",
      collect_service_metrics=True,
  )
  def test_main_calls_run_command_with_metrics_enabled(self):
    with mock.patch.object(
        run_workload, "run_command", autospec=True
    ) as mock_run_command:
      run_workload.main(["unused_argv"])
      mock_run_command.assert_called_once_with(
          cluster="test-cluster",
          project="test-project",
          region="test-region",
          gcs_bucket="test-bucket",
          pathways_service="test-service:1234",
          tpu_type="tpuv6e:4x8",
          tpu_count=1,
          command="echo hello",
          proxy_server_image="",
          proxy_options=[],
          collect_service_metrics=True,
      )


if __name__ == "__main__":
  # Provide dummy values for required flags to satisfy absl verification.
  # These are overridden by tests as needed using flagsaver or direct arguments.
  sys.argv.extend([
      "--cluster=dummy",
      "--project=dummy",
      "--region=dummy",
      "--gcs_bucket=dummy",
      "--pathways_service=dummy:1234",
      "--command=dummy",
  ])
  absltest.main()
