"""Unit tests for the MetricsCollector class."""

from unittest import mock
from absl.testing import absltest
from pathwaysutils.experimental.shared_pathways_service import metrics_collector


class MetricsCollectorTest(absltest.TestCase):

  @mock.patch(
      "pathwaysutils.experimental.shared_pathways_service.metrics_collector.monitoring_v3.MetricServiceClient"
  )
  def test_record_active_user(self, mock_client_class):
    mock_client = mock.Mock()
    mock_client_class.return_value = mock_client

    collector = metrics_collector.MetricsCollector(
        "gcp_project", "gke_cluster", "proxy_job"
    )

    collector.record_active_user(True)
    collector.flush()

    mock_client.create_time_series.assert_called_once()
    _, kwargs = mock_client.create_time_series.call_args
    self.assertEqual(kwargs["name"], "projects/gcp_project")
    time_series = kwargs["time_series"][0]
    self.assertEqual(
        time_series.metric.type,
        "custom.googleapis.com/shared_pathways_service/num_active_users",
    )
    self.assertEqual(time_series.points[0].value.int64_value, 1)
    self.assertEqual(
        time_series.metric.labels["cluster_name"], "gke_cluster"
    )
    self.assertEqual(time_series.metric.labels["job_name"], "proxy_job")

  @mock.patch(
      "pathwaysutils.experimental.shared_pathways_service.metrics_collector.monitoring_v3.MetricServiceClient"
  )
  def test_record_capacity_in_use(self, mock_client_class):
    mock_client = mock.Mock()
    mock_client_class.return_value = mock_client

    collector = metrics_collector.MetricsCollector("gcp_project", "gke_cluster")
    collector.record_capacity_in_use(8)
    collector.flush()

    mock_client.create_time_series.assert_called_once()
    _, kwargs = mock_client.create_time_series.call_args
    time_series = kwargs["time_series"][0]
    self.assertEqual(
        time_series.metric.type,
        "custom.googleapis.com/shared_pathways_service/capacity_in_use",
    )
    self.assertEqual(time_series.points[0].value.int64_value, 8)

  @mock.patch(
      "pathwaysutils.experimental.shared_pathways_service.metrics_collector.monitoring_v3.MetricServiceClient"
  )
  def test_record_assignment_time(self, mock_client_class):
    mock_client = mock.Mock()
    mock_client_class.return_value = mock_client

    collector = metrics_collector.MetricsCollector("gcp_project", "gke_cluster")
    collector.record_assignment_time(12.5)
    collector.flush()

    mock_client.create_time_series.assert_called_once()
    _, kwargs = mock_client.create_time_series.call_args
    time_series = kwargs["time_series"][0]
    self.assertEqual(
        time_series.metric.type,
        "custom.googleapis.com/shared_pathways_service/assignment_time",
    )
    self.assertEqual(time_series.points[0].value.double_value, 12.5)

  @mock.patch(
      "pathwaysutils.experimental.shared_pathways_service.metrics_collector.monitoring_v3.MetricServiceClient"
  )
  def test_record_successful_request(self, mock_client_class):
    mock_client = mock.Mock()
    mock_client_class.return_value = mock_client

    collector = metrics_collector.MetricsCollector("gcp_project", "gke_cluster")
    collector.record_successful_request()
    collector.flush()

    mock_client.create_time_series.assert_called_once()
    _, kwargs = mock_client.create_time_series.call_args
    time_series = kwargs["time_series"][0]
    self.assertEqual(
        time_series.metric.type,
        "custom.googleapis.com/shared_pathways_service/num_successful_reqs",
    )
    self.assertEqual(time_series.points[0].value.int64_value, 1)

  @mock.patch(
      "pathwaysutils.experimental.shared_pathways_service.metrics_collector.monitoring_v3.MetricServiceClient"
  )
  def test_record_user_waiting(self, mock_client_class):
    mock_client = mock.Mock()
    mock_client_class.return_value = mock_client

    collector = metrics_collector.MetricsCollector("gcp_project", "gke_cluster")
    collector.record_user_waiting(True)
    collector.flush()

    mock_client.create_time_series.assert_called_once()
    _, kwargs = mock_client.create_time_series.call_args
    time_series = kwargs["time_series"][0]
    self.assertEqual(
        time_series.metric.type,
        "custom.googleapis.com/shared_pathways_service/num_users_waiting",
    )
    self.assertEqual(time_series.points[0].value.int64_value, 1)

  @mock.patch(
      "pathwaysutils.experimental.shared_pathways_service.metrics_collector.monitoring_v3.MetricServiceClient"
  )
  def test_record_requested_capacity(self, mock_client_class):
    mock_client = mock.Mock()
    mock_client_class.return_value = mock_client

    collector = metrics_collector.MetricsCollector("gcp_project", "gke_cluster")
    collector.record_requested_capacity(64)
    collector.flush()

    mock_client.create_time_series.assert_called_once()
    _, kwargs = mock_client.create_time_series.call_args
    time_series = kwargs["time_series"][0]
    self.assertEqual(
        time_series.metric.type,
        "custom.googleapis.com/shared_pathways_service/requested_capacity",
    )
    self.assertEqual(time_series.points[0].value.int64_value, 64)

  @mock.patch(
      "pathwaysutils.experimental.shared_pathways_service.metrics_collector.monitoring_v3.MetricServiceClient"
  )
  def test_initialize_descriptors(self, mock_client_class):
    mock_client = mock.Mock()
    mock_client.get_metric_descriptor.side_effect = (
        metrics_collector.exceptions.NotFound("not found")
    )
    mock_client_class.return_value = mock_client

    _ = metrics_collector.MetricsCollector("gcp_project", "gke_cluster")

    self.assertEqual(mock_client.create_metric_descriptor.call_count, 6)

    # Verify units for each descriptor
    calls = mock_client.create_metric_descriptor.call_args_list

    call_args = [c.kwargs["metric_descriptor"] for c in calls]

    def find_descriptor(name):
      for desc in call_args:
        if desc["type"].endswith(f"/{name}"):
          return desc
      return None

    self.assertEqual(find_descriptor("num_active_users")["unit"], "1")
    self.assertEqual(find_descriptor("capacity_in_use")["unit"], "chips")
    self.assertEqual(find_descriptor("assignment_time")["unit"], "s")
    self.assertEqual(find_descriptor("num_successful_reqs")["unit"], "1")
    self.assertEqual(find_descriptor("num_users_waiting")["unit"], "1")
    self.assertEqual(find_descriptor("requested_capacity")["unit"], "chips")

  @mock.patch(
      "pathwaysutils.experimental.shared_pathways_service.metrics_collector.monitoring_v3.MetricServiceClient"
  )
  def test_initialize_descriptors_already_exists(self, mock_client_class):
    mock_client = mock.Mock()
    # By default, get_metric_descriptor returns a mock without raising,
    # meaning the metric already exists.
    mock_client_class.return_value = mock_client

    _ = metrics_collector.MetricsCollector("gcp_project", "gke_cluster")

    self.assertEqual(mock_client.get_metric_descriptor.call_count, 6)
    self.assertEqual(mock_client.create_metric_descriptor.call_count, 0)

  @mock.patch(
      "pathwaysutils.experimental.shared_pathways_service.metrics_collector.monitoring_v3.MetricServiceClient"
  )
  def test_buffer_queue_and_throttle(self, mock_client_class):
    mock_client = mock.Mock()
    mock_client_class.return_value = mock_client

    collector = metrics_collector.MetricsCollector("gcp_project", "gke_cluster")

    # Send 2 states back-to-back
    collector.record_user_waiting(True)
    collector.record_user_waiting(False)

    # Queue should hold both
    self.assertLen(collector._buffer["num_users_waiting"], 2)

    # First flush sends [0]
    collector.flush()
    self.assertEqual(mock_client.create_time_series.call_count, 1)
    self.assertLen(collector._buffer["num_users_waiting"], 1)

    # Immediate second flush should do NOTHING due to 10.5s limit
    collector.flush()
    self.assertEqual(mock_client.create_time_series.call_count, 1)
    self.assertLen(collector._buffer["num_users_waiting"], 1)

    # Shift time forward by 11s and flush -> sends remaining [1]
    with mock.patch(
        "time.time",
        return_value=collector._last_sent_time["num_users_waiting"] + 11.0,
    ):
      collector.flush()

    self.assertEqual(mock_client.create_time_series.call_count, 2)
    self.assertNotIn("num_users_waiting", collector._buffer)

  @mock.patch(
      "pathwaysutils.experimental.shared_pathways_service.metrics_collector.monitoring_v3.MetricServiceClient"
  )
  def test_shutdown_exhausts_queue(self, mock_client_class):
    mock_client = mock.Mock()
    mock_client_class.return_value = mock_client
    collector = metrics_collector.MetricsCollector("gcp_project", "gke_cluster")
    collector.record_capacity_in_use(16)
    collector.record_capacity_in_use(0)

    current_time = [1000.0]

    def mock_sleep_func(seconds):
      current_time[0] += seconds

    def mock_time_func():
      return current_time[0]

    with mock.patch("time.sleep", side_effect=mock_sleep_func) as mock_sleep:
      with mock.patch("time.time", side_effect=mock_time_func):
        collector._shutdown()

    self.assertEqual(mock_client.create_time_series.call_count, 2)
    mock_sleep.assert_called()


if __name__ == "__main__":
  absltest.main()
