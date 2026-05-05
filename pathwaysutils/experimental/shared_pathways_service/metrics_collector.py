"""Metrics collector for Shared Pathways Service."""

import atexit
import logging
import threading
import time
from typing import Any, Dict
import uuid

try:
  # pylint: disable=g-import-not-at-top
  from google.api_core import exceptions
  from google.cloud import monitoring_v3
except ImportError:
  pass

_logger = logging.getLogger(__name__)


METRIC_PREFIX = "custom.googleapis.com/shared_pathways_service/"

_METRIC_NUM_ACTIVE_USERS = "num_active_users"
_METRIC_CAPACITY_IN_USE = "capacity_in_use"
_METRIC_ASSIGNMENT_TIME = "assignment_time"
_METRIC_NUM_SUCCESSFUL_REQS = "num_successful_reqs"
_METRIC_NUM_USERS_WAITING = "num_users_waiting"
_METRIC_REQUESTED_CAPACITY = "requested_capacity"
_METRIC_DESCRIPTORS = [
    {
        "name": _METRIC_NUM_ACTIVE_USERS,
        "description": "Number of active users at any given time",
        "value_type": "INT64",
        "unit": "1",
    },
    {
        "name": _METRIC_CAPACITY_IN_USE,
        "description": "Number of chips that are actively running workloads",
        "value_type": "INT64",
        "unit": "chips",
        "display_name": "Capacity (chips) in use",
    },
    {
        "name": _METRIC_ASSIGNMENT_TIME,
        "description": "Time to assign slice(s) to an incoming client",
        "value_type": "DOUBLE",
        "unit": "s",
        "display_name": "Capacity assignment time",
    },
    {
        "name": _METRIC_NUM_SUCCESSFUL_REQS,
        "description": (
            "Number of user requests that got capacity assignment successfully"
        ),
        "value_type": "INT64",
        "unit": "1",
        "display_name": "Successful capacity assignment requests",
    },
    {
        "name": _METRIC_NUM_USERS_WAITING,
        "description": "Number of users waiting for capacity",
        "value_type": "INT64",
        "unit": "1",
        "display_name": "Users waiting",
    },
    {
        "name": _METRIC_REQUESTED_CAPACITY,
        "description": "Number of chips requested by an incoming client",
        "value_type": "INT64",
        "unit": "chips",
        "display_name": "Requested capacity (chips)",
    },
]


class MetricsCollector:
  """Collects usage metrics for Shared Pathways Service and reports to Cloud Monitoring."""

  def __init__(self, project_id: str):
    self.project_id = project_id
    self.client = monitoring_v3.MetricServiceClient()
    self.project_name = f"projects/{self.project_id}"
    self._lock = threading.Lock()
    self._buffer: Dict[str, list[tuple[Any, str, Dict[str, str] | None]]] = {}
    self._last_sent_time: Dict[str, float] = {}
    self._instance_id = str(uuid.uuid4())
    self._running = True
    for descriptor in _METRIC_DESCRIPTORS:
      self._create_metric_descriptor(**descriptor)
    self._flusher_thread = threading.Thread(
        target=self._flush_loop, daemon=True
    )
    self._flusher_thread.start()
    atexit.register(self._shutdown)
    _logger.info("Metrics collection initialized.")

  def _create_time_series_object(
      self,
      metric_type: str,
      value: Any,
      value_type: str,
      metric_labels: Dict[str, str] | None = None,
      resource_type: str = "global",
      resource_labels: Dict[str, str] | None = None,
  ) -> Any:
    """Creates a TimeSeries object for a single metric."""
    # Using Any for return type to avoid failing when monitoring_v3 is not
    # available.
    series = monitoring_v3.TimeSeries()
    series.metric.type = METRIC_PREFIX + metric_type
    series.resource.type = resource_type
    if resource_labels:
      series.resource.labels.update(resource_labels)
    if metric_labels:
      series.metric.labels.update(metric_labels)

    now = time.time()
    seconds = int(now)
    nanos = int((now - seconds) * 10**9)

    point = monitoring_v3.Point(
        interval=monitoring_v3.TimeInterval(
            end_time={"seconds": seconds, "nanos": nanos}
        ),
        value=monitoring_v3.TypedValue(**{value_type: value}),
    )
    series.points.append(point)
    return series

  def _flush_loop(self):
    """Runs continuously to flush the metrics buffer."""
    while self._running:
      self.flush()
      time.sleep(1)

  def flush(self):
    """Sends any eligible buffered metrics to Cloud Monitoring."""
    with self._lock:
      now = time.time()
      to_send = []
      for metric_type, queue in list(self._buffer.items()):
        if not queue:
          del self._buffer[metric_type]
          continue
        last_time = self._last_sent_time.get(metric_type, 0)
        # Add a slight cushion (10.5s) to prevent sub-second drift errors.
        if now - last_time >= 10.5 or last_time == 0:
          item = queue.pop(0)
          to_send.append((metric_type, *item))
          self._last_sent_time[metric_type] = now
          if not queue:
            del self._buffer[metric_type]

    for metric_type, value, value_type, metric_labels in to_send:
      self._transmit(metric_type, value, value_type, metric_labels)

  def _shutdown(self):
    """Synchronously drains the final state of the queue before exiting."""
    self._running = False
    while True:
      with self._lock:
        if not any(self._buffer.values()):
          break
        # Wait for the window to open for at least one item
        now = time.time()
        min_wait = 0.0
        for metric_type, queue in self._buffer.items():
          if queue:
            last_time = self._last_sent_time.get(metric_type, 0)
            wait_needed = 10.5 - (now - last_time)
            if wait_needed > 0:
              min_wait = max(min_wait, wait_needed)
      if min_wait > 0:
        _logger.info(
            "Waiting %.1fs for Cloud Monitoring sampling window...", min_wait
        )
        time.sleep(min_wait)
      self.flush()

  def _send_metric(
      self,
      metric_type: str,
      value: Any,
      value_type: str,
      metric_labels: Dict[str, str] | None = None,
  ):
    """Queues a single metric in the buffer."""
    default_labels = {"client_instance_id": self._instance_id}
    if metric_labels:
      default_labels.update(metric_labels)
    _logger.info(
        "Buffering metric %s: %s",
        metric_type,
        (value, value_type, default_labels),
    )
    with self._lock:
      if metric_type not in self._buffer:
        self._buffer[metric_type] = []
      _logger.info(
          "Successfully buffered metric %s: %s",
          metric_type,
          (value, value_type, default_labels),
      )
      self._buffer[metric_type].append((value, value_type, default_labels))

  def _transmit(
      self,
      metric_type: str,
      value: Any,
      value_type: str,
      metric_labels: Dict[str, str] | None = None,
  ):
    """Physically transmits a TimeSeries to Cloud Monitoring."""
    series = self._create_time_series_object(
        metric_type, value, value_type, metric_labels
    )
    try:
      self.client.create_time_series(
          name=self.project_name, time_series=[series]
      )
      _logger.info("Sent metric %s: %s", metric_type, value)
    except exceptions.GoogleAPICallError as e:
      _logger.warning("Failed to send metric %s: %s", metric_type, e)

  def _create_metric_descriptor(
      self,
      name: str,
      description: str,
      value_type: str,
      unit: str,
      metric_kind: str = "GAUGE",
      display_name: str | None = None,
  ):
    """Creates a metric descriptor if not already present."""
    metric_type = METRIC_PREFIX + name
    display_name = display_name or name

    try:
      self.client.create_metric_descriptor(
          name=f"projects/{self.project_id}",
          metric_descriptor={
              "type": metric_type,
              "metric_kind": metric_kind,
              "value_type": value_type,
              "description": description,
              "display_name": display_name,
              "unit": unit,
              "labels": [{
                  "key": "client_instance_id",
                  "value_type": "STRING",
                  "description": "Unique execution identifier",
              }],
          },
      )
      _logger.info("Created metric descriptor: %s", metric_type)
    except exceptions.AlreadyExists:
      _logger.debug("Metric descriptor %s already exists.", metric_type)

  def record_active_user(self, is_active: bool):
    """Records the number of active users (1 for active, 0 for inactive)."""
    self._send_metric(
        _METRIC_NUM_ACTIVE_USERS, 1 if is_active else 0, "int64_value"
    )

  def record_capacity_in_use(self, chips: int):
    """Records the number of chips in use."""
    self._send_metric(_METRIC_CAPACITY_IN_USE, chips, "int64_value")

  def record_requested_capacity(self, chips: int):
    """Records the number of chips requested by the client."""
    self._send_metric(_METRIC_REQUESTED_CAPACITY, chips, "int64_value")

  def record_assignment_time(self, duration_seconds: float):
    """Records the time taken to assign slices."""
    self._send_metric(_METRIC_ASSIGNMENT_TIME, duration_seconds, "double_value")

  def record_successful_request(self):
    """Records a successful request."""
    self._send_metric(_METRIC_NUM_SUCCESSFUL_REQS, 1, "int64_value")

  def record_user_waiting(self, is_waiting: bool):
    """Records a user waiting for capacity."""
    self._send_metric(
        _METRIC_NUM_USERS_WAITING, 1 if is_waiting else 0, "int64_value"
    )


class SafeMetricsCollector:
  """Wrapper for MetricsCollector that safely absorbs calls when metrics are disabled."""

  def __init__(self, collector: MetricsCollector | None):
    self._collector = collector

  def __getattr__(self, name: str):
    if self._collector is None:
      return lambda *args, **kwargs: None
    return getattr(self._collector, name)
