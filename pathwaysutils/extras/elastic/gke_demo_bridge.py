# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GKE Bridge Server for Pathways Elasticity Live Webapp Demo.

This server bridges between the interactive web application frontend and a
configurable Kubernetes cluster and Pathways JobSet workload.

It provides:
- Static file hosting on port 48259 (gke_elasticity_demo.html and
  model_comparison.html), bound to 127.0.0.1 by default
- Real-time Server-Sent Events (SSE) telemetry stream parsing JAX driver logs
- Live workload and cluster configuration API (/api/config) with preset support
- Dynamic ConfigMap and JobSet YAML generation written to /tmp for the user to
  inspect and apply manually with kubectl
- Per-session token authentication and same-origin request validation
- Start, Pause, and Reset training signaling to head pod port 8090
- Dual-mode disruption and reconnect signaling:
    1. Genuine Kubernetes preemption (kubectl cordon + drain / uncordon)
    2. In-workload signaling (HTTP control RPC to head pod port 8090)
- Cluster and JobSet status monitoring and cleanup
- Cumulative Goodput tracking: G = forward progress time / total run time so far
"""

import argparse
import collections
from http import server
import json
import logging
import math
import os
import queue
import re
import secrets
import socket
import socketserver
import subprocess
import tempfile
import threading
import time
from typing import Any
import urllib.parse
import webbrowser

_logger = logging.getLogger("gke_demo_bridge")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIMPLE_MODEL_DIR = os.path.normpath(
    os.path.join(BASE_DIR, "..", "simple_model")
)
PATHWAYSUTILS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", ".."))
ORBAX_DIR = os.path.normpath(os.path.join(PATHWAYSUTILS_DIR, "..", "orbax"))
ALLOWED_CONFIGMAP_ROOTS = (
    os.path.realpath(PATHWAYSUTILS_DIR),
    os.path.realpath(ORBAX_DIR),
)
WORKLOAD_PY_PATH = os.path.join(BASE_DIR, "elastic_workload.py")
FAULT_INJECTION_PY_PATH = os.path.join(
    PATHWAYSUTILS_DIR, "test", "mock_fault_injection.py"
)
MANAGER_PY_PATH = os.path.join(PATHWAYSUTILS_DIR, "elastic", "manager.py")
SNAPSHOTTER_PY_PATH = os.path.normpath(
    os.path.join(
        ORBAX_DIR,
        "checkpoint",
        "experimental",
        "v1",
        "_src",
        "training",
        "pathways",
        "snapshotter.py",
    )
)
JOBSET_YAML_PATH = os.path.join(
    tempfile.gettempdir(), "gke_elastic_workload.yaml"
)
PORT = 48259

_SESSION_TOKEN = secrets.token_urlsafe(32)
_ALLOWED_HOSTS: set[str] = {
    "localhost",
    "127.0.0.1",
    "::1",
    "[::1]",
}
_SAFE_K8S_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:@/-]{0,252}$")
_SAFE_FILENAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{1,128}$")
_SAFE_CONTROL_PATH_RE = re.compile(
    r"^/(start|pause|reset|status|(disrupt|reconnect)\?slice=\d+)$"
)

DEFAULT_CLUSTER_CONTEXT = (
    "gke_cloud-tpu-multipod-dev_us-south1_lukebaumann-pw-v5e-32"
)
DEFAULT_NAMESPACE = "default"
DEFAULT_JOBSET_NAME = "lb-elastic"
DEFAULT_KUEUE_QUEUE = "multislice-queue"
DEFAULT_CONTAINER_IMAGE = (
    "gcr.io/cloud-tpu-multipod-dev/lukebaumann_model_lite_test"
)
DEFAULT_SERVER_IMAGE = "us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/unsanitized_server@sha256:b155d72fe1bc17f1fb2aa692e4ea517e22401c775d50926d7b656b38c024acd6"
DEFAULT_PROXY_IMAGE = "us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/unsanitized_proxy_server@sha256:6dcf83cee6caf21ec019acea7fc9b91af32cb41bdde6e4cc889e5288b7ac03f2"
DEFAULT_GCS_SCRATCH = (
    "gs://sujinesh-cloud-tpu-multipod-dev-us/lukebaumann-model-lite-v5e/"
)
INIT_OVERHEAD_SEC = 2.5

SIMPLE_MODEL_MODULES = (
    "__init__.py",
    "config.py",
    "model.py",
    "data.py",
    "compile.py",
    "workload.py",
    "simple_model.py",
)

TPU_ACCELERATOR_MAP = {
    "tpuv5e": "tpu-v5-lite-podslice",
    "v5e": "tpu-v5-lite-podslice",
    "tpuv6e": "tpu-v6e-slice",
    "v6e": "tpu-v6e-slice",
    "tpuv5p": "tpu-v5p-slice",
    "v5p": "tpu-v5p-slice",
    "tpuv7x": "tpu7x",
    "tpu7x": "tpu7x",
    "v7x": "tpu7x",
}

TPU_INSTANCE_PREFIX_MAP = {
    "tpuv5e": "tpuv5e",
    "v5e": "tpuv5e",
    "tpuv6e": "tpuv6e",
    "v6e": "tpuv6e",
    "tpuv5p": "tpuv5p",
    "v5p": "tpuv5p",
    "tpuv7x": "tpuv7x",
    "tpu7x": "tpuv7x",
    "v7x": "tpuv7x",
}


def compute_topology_specs(tpu_type: str, topology: str) -> dict[str, Any]:
  """Computes accelerator selector, instance_type, chips_per_slice, chips_per_vm, and vms_per_slice."""
  norm_key = tpu_type.strip().lower()
  accelerator = TPU_ACCELERATOR_MAP.get(norm_key, "tpu-v5-lite-podslice")
  instance_prefix = TPU_INSTANCE_PREFIX_MAP.get(norm_key, "tpuv5e")
  clean_topo = topology.strip() or "4x8"
  dims = [int(x) for x in clean_topo.split("x") if x.strip().isdigit()]
  chips_per_slice = math.prod(dims) if dims else 32
  chips_per_vm = min(4, chips_per_slice)
  vms_per_slice = max(1, chips_per_slice // chips_per_vm)
  return {
      "accelerator": accelerator,
      "instance_type": f"{instance_prefix}:{clean_topo}",
      "topology": clean_topo,
      "chips_per_slice": chips_per_slice,
      "chips_per_vm": chips_per_vm,
      "vms_per_slice": vms_per_slice,
  }


def default_config() -> dict[str, Any]:
  """Returns the default workload and cluster configuration."""
  topo_specs = compute_topology_specs("tpuv5e", "4x8")
  default_cm_files = [
      {"filename": "elastic_workload.py", "host_path": WORKLOAD_PY_PATH},
  ]
  for fname, hpath in (
      ("mock_fault_injection.py", FAULT_INJECTION_PY_PATH),
      ("manager.py", MANAGER_PY_PATH),
      ("snapshotter.py", SNAPSHOTTER_PY_PATH),
  ):
    if os.path.isfile(hpath):
      default_cm_files.append({"filename": fname, "host_path": hpath})
  return {
      "cluster_context": DEFAULT_CLUSTER_CONTEXT,
      "namespace": DEFAULT_NAMESPACE,
      "jobset_name": DEFAULT_JOBSET_NAME,
      "kueue_queue_name": DEFAULT_KUEUE_QUEUE,
      "num_slices": 2,
      "standby_slices": 2,
      "tpu_type": "tpuv5e",
      "topology": "4x8",
      "chips_per_slice": topo_specs["chips_per_slice"],
      "vms_per_slice": topo_specs["vms_per_slice"],
      "accelerator_label": topo_specs["accelerator"],
      "total_steps": 1000,
      "elasticity_mode": "step_snapshot_replica_resize",
      "gcs_scratch_location": DEFAULT_GCS_SCRATCH,
      "container_image": DEFAULT_CONTAINER_IMAGE,
      "server_image": DEFAULT_SERVER_IMAGE,
      "proxy_image": DEFAULT_PROXY_IMAGE,
      "cpu_nodepool": "cpu-np",
      "workload_preset": "builtin_elastic_demo",
      "custom_command": (
          "python3 -m pathwaysutils.extras.elastic.elastic_workload \\\n"
          "  --config_name=default \\\n"
          "  --total_steps=1000 \\\n"
          "  --elasticity_mode=step_snapshot_replica_resize \\\n"
          "  --simulate_faults=true \\\n"
          "  --enable_control_server=true \\\n"
          "  --control_port=8090 \\\n"
          "  --wait_for_start_signal=true"
      ),
      "configmap_files": default_cm_files,
  }


def _validate_k8s_id(value: str, field_name: str) -> str:
  """Validates a Kubernetes identifier to prevent kubectl flag injection."""
  cleaned = str(value).strip()
  if not cleaned or cleaned.startswith("-") or not _SAFE_K8S_ID_RE.match(
      cleaned
  ):
    raise ValueError(f"Invalid {field_name}: {value!r}")
  return cleaned


def _extract_hostname(host_header: str) -> str:
  """Extracts the lowercase hostname from an HTTP Host header."""
  raw = host_header.strip().lower()
  if not raw:
    return ""
  if raw.startswith("["):
    end = raw.find("]")
    return raw[: end + 1] if end != -1 else raw
  return raw.split(":", 1)[0]


def _is_valid_host_header(host_header: str) -> bool:
  """Checks whether the HTTP Host header matches an allowed local/server host."""
  hostname = _extract_hostname(host_header)
  return bool(hostname) and hostname in _ALLOWED_HOSTS


def _is_valid_origin_header(origin_header: str, host_header: str) -> bool:
  """Checks whether the HTTP Origin header matches the server Host header."""
  if not origin_header:
    return True
  parsed = urllib.parse.urlparse(origin_header)
  origin_host = (parsed.hostname or "").lower()
  if ":" in origin_host and not origin_host.startswith("["):
    origin_host = f"[{origin_host}]"
  target_host = _extract_hostname(host_header)
  return bool(origin_host) and origin_host == target_host and (
      origin_host in _ALLOWED_HOSTS
  )


def _is_valid_session_token(token_header: str) -> bool:
  """Verifies the per-session CSRF token header in constant time."""
  if not token_header:
    return False
  return secrets.compare_digest(token_header, _SESSION_TOKEN)


def _kubectl_cmd(
    *args: str,
    context: str = "",
    namespace: str = "",
    include_ns: bool = True,
) -> list[str]:
  """Builds a kubectl command scoped to the given context and optional namespace."""
  cmd = ["kubectl"]
  if context:
    cmd.append(f"--context={_validate_k8s_id(context, 'cluster_context')}")
  if include_ns and namespace:
    cmd.extend(["-n", _validate_k8s_id(namespace, "namespace")])
  cmd.extend(args)
  return cmd


def _get_kubeconfig_contexts() -> list[str]:
  """Returns list of available kubeconfig context names."""
  proc = subprocess.run(
      ["kubectl", "config", "get-contexts", "-o", "name"],
      capture_output=True,
      text=True,
      check=False,
  )
  if proc.returncode != 0:
    return [DEFAULT_CLUSTER_CONTEXT]
  contexts = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
  return contexts if contexts else [DEFAULT_CLUSTER_CONTEXT]


def _send_workload_control_rpc(
    head_pod: str,
    endpoint_path: str,
    context: str = "",
    namespace: str = "",
) -> tuple[bool, str]:
  """Sends an HTTP GET request to the head pod's in-workload control server on port 8090."""
  if not head_pod:
    return False, "Head pod not running."
  try:
    safe_pod = _validate_k8s_id(head_pod, "head_pod")
  except ValueError as e:
    return False, str(e)
  if not _SAFE_CONTROL_PATH_RE.match(endpoint_path):
    return False, f"Disallowed control endpoint path: {endpoint_path!r}"
  exec_cmd = _kubectl_cmd(
      "exec",
      safe_pod,
      "-c",
      "jax-tpu",
      "--",
      "python3",
      "-c",
      (
          "import urllib.request; resp ="
          f" urllib.request.urlopen('http://localhost:8090{endpoint_path}',"
          " timeout=5); print(resp.read().decode())"
      ),
      context=context,
      namespace=namespace,
      include_ns=True,
  )
  res = subprocess.run(exec_cmd, capture_output=True, text=True, check=False)
  if res.returncode != 0:
    return False, res.stderr.strip() or res.stdout.strip()
  return True, res.stdout.strip()


def _uncordon_all_tpu_nodes(
    cordoned_nodes: dict[int, str], context: str = ""
) -> None:
  """Uncordons any cordoned TPU nodes tracked in state or marked unschedulable in GKE."""
  for node in list(cordoned_nodes.values()):
    if node:
      try:
        safe_node = _validate_k8s_id(node, "node_name")
      except ValueError:
        continue
      subprocess.run(
          _kubectl_cmd(
              "uncordon", safe_node, context=context, include_ns=False
          ),
          capture_output=True,
          text=True,
          check=False,
      )
  cordoned_nodes.clear()

  cmd_nodes = _kubectl_cmd(
      "get",
      "nodes",
      "--selector=cloud.google.com/gke-tpu-accelerator",
      "-o",
      "jsonpath={.items[?(@.spec.unschedulable==true)].metadata.name}",
      context=context,
      include_ns=False,
  )
  proc_nodes = subprocess.run(
      cmd_nodes, capture_output=True, text=True, check=False
  )
  for node in proc_nodes.stdout.strip().split():
    if node:
      try:
        safe_node = _validate_k8s_id(node, "node_name")
      except ValueError:
        continue
      subprocess.run(
          _kubectl_cmd(
              "uncordon", safe_node, context=context, include_ns=False
          ),
          capture_output=True,
          text=True,
          check=False,
      )


def _merge_config_dict(
    base_cfg: dict[str, Any], updates: dict[str, Any]
) -> dict[str, Any]:
  """Merges updates onto base_cfg and recomputes topology fields."""
  cfg = json.loads(json.dumps(base_cfg))
  for key in (
      "cluster_context",
      "namespace",
      "jobset_name",
      "kueue_queue_name",
      "tpu_type",
      "topology",
      "elasticity_mode",
      "gcs_scratch_location",
      "container_image",
      "server_image",
      "proxy_image",
      "cpu_nodepool",
      "workload_preset",
      "custom_command",
  ):
    if key in updates and updates[key] is not None:
      cfg[key] = str(updates[key]).strip()

  for int_key, min_val, max_val in (
      ("num_slices", 1, 16),
      ("standby_slices", 0, 16),
      ("total_steps", 1, 1000000),
  ):
    if int_key in updates and updates[int_key] is not None:
      try:
        val = int(updates[int_key])
        cfg[int_key] = max(min_val, min(max_val, val))
      except (ValueError, TypeError):
        pass

  if "configmap_files" in updates and isinstance(
      updates["configmap_files"], list
  ):
    cleaned_files = []
    for entry in updates["configmap_files"]:
      if isinstance(entry, dict):
        fname = str(entry.get("filename", "")).strip()
        hpath = str(entry.get("host_path", "")).strip()
        if fname and hpath:
          cleaned_files.append({"filename": fname, "host_path": hpath})
    cfg["configmap_files"] = cleaned_files

  topo_specs = compute_topology_specs(cfg["tpu_type"], cfg["topology"])
  cfg["topology"] = topo_specs["topology"]
  cfg["chips_per_slice"] = topo_specs["chips_per_slice"]
  cfg["vms_per_slice"] = topo_specs["vms_per_slice"]
  cfg["accelerator_label"] = topo_specs["accelerator"]
  return cfg


class BridgeState:
  """Maintains configuration, state, log buffer, telemetry, and SSE subscriber queues."""

  def __init__(self):
    self.lock = threading.Lock()
    self.config: dict[str, Any] = default_config()
    self.subscribers: list[queue.Queue[dict[str, Any]]] = []
    self.log_buffer: collections.deque[dict[str, Any]] = collections.deque(
        maxlen=400
    )
    self.loss_history: list[dict[str, Any]] = []
    self.goodput_history: list[dict[str, Any]] = []

    self.jobset_status: str = "IDLE"
    self.launch_timestamp: float | None = None
    self.head_pod_name: str = ""
    self.head_pod_phase: str = "Unknown"
    self.worker_pods: list[dict[str, Any]] = []

    self.training_running: bool = False
    self.workload_ready: bool = False
    self.auto_start_when_ready: bool = False

    self.current_step: int = 0
    self.total_steps: int = self.config["total_steps"]
    self.total_slices: int = self.config["num_slices"]
    self.standby_slices: int = self.config["standby_slices"]
    self.active_slices: list[int] = list(range(self.total_slices))
    self.inactive_slices: list[int] = []

    self.start_wall_time: float | None = None
    self.last_step_wall_time: float | None = None
    self.forward_progress_time: float = 0.0
    self.current_goodput: float = 0.0

    self.last_loss: float | None = None
    self.current_event: str = "Idle"
    self.stopped: bool = False
    self.cordoned_nodes: dict[int, str] = {}

  def get_config(self) -> dict[str, Any]:
    with self.lock:
      return json.loads(json.dumps(self.config))

  def update_config(self, updates: dict[str, Any]) -> dict[str, Any]:
    """Updates cluster and workload settings and returns the updated config."""
    with self.lock:
      self.config = _merge_config_dict(self.config, updates)
      self.total_steps = self.config["total_steps"]
      self.total_slices = self.config["num_slices"]
      self.standby_slices = self.config["standby_slices"]
      if not self.training_running and self.jobset_status in (
          "IDLE",
          "READY",
          "STARTING",
      ):
        self.active_slices = list(range(self.total_slices))
        self.inactive_slices = []

      return json.loads(json.dumps(self.config))

  def register_subscriber(self, q: queue.Queue[dict[str, Any]]) -> None:
    """Adds an SSE client queue to the broadcast list."""
    with self.lock:
      self.subscribers.append(q)

  def unregister_subscriber(self, q: queue.Queue[dict[str, Any]]) -> None:
    """Removes an SSE client queue from the broadcast list."""
    with self.lock:
      if q in self.subscribers:
        self.subscribers.remove(q)

  def broadcast(self, event: dict[str, Any]) -> None:
    with self.lock:
      dead_queues = []
      for q in self.subscribers:
        try:
          q.put_nowait(event)
        except queue.Full:
          dead_queues.append(q)
      for dq in dead_queues:
        if dq in self.subscribers:
          self.subscribers.remove(dq)

  def record_log(self, line: str) -> None:
    log_entry = {
        "type": "log",
        "line": line,
        "timestamp": time.time(),
    }
    with self.lock:
      self.log_buffer.append(log_entry)
    self.broadcast(log_entry)

  def reset_job_state(self, auto_start: bool = False) -> None:
    """Resets telemetry and job state for a newly launched JobSet."""
    with self.lock:
      self.loss_history.clear()
      self.goodput_history.clear()
      self.log_buffer.clear()
      self.current_step = 0
      self.total_slices = self.config["num_slices"]
      self.standby_slices = self.config["standby_slices"]
      self.total_steps = self.config["total_steps"]
      self.active_slices = list(range(self.total_slices))
      self.inactive_slices = []
      self.start_wall_time = None
      self.last_step_wall_time = None
      self.forward_progress_time = 0.0
      self.current_goodput = 0.0
      self.last_loss = None
      self.training_running = False
      self.workload_ready = False
      self.auto_start_when_ready = auto_start
      self.jobset_status = "STARTING"
      self.launch_timestamp = time.time()
      self.current_event = (
          "JobSet launched, initializing pods and pre-compiling..."
      )
      cordoned_copy = dict(self.cordoned_nodes)
      ctx = self.config["cluster_context"]
      self.cordoned_nodes.clear()
    _uncordon_all_tpu_nodes(cordoned_copy, context=ctx)

  def clear_for_delete(self) -> None:
    """Clears state and uncordons nodes when a JobSet is deleted."""
    with self.lock:
      self.loss_history.clear()
      self.goodput_history.clear()
      self.log_buffer.clear()
      self.current_step = 0
      self.total_slices = self.config["num_slices"]
      self.standby_slices = self.config["standby_slices"]
      self.active_slices = list(range(self.total_slices))
      self.inactive_slices = []
      self.start_wall_time = None
      self.last_step_wall_time = None
      self.forward_progress_time = 0.0
      self.current_goodput = 0.0
      self.last_loss = None
      self.training_running = False
      self.workload_ready = False
      self.auto_start_when_ready = False
      self.head_pod_name = ""
      self.head_pod_phase = "Unknown"
      self.worker_pods = []
      self.jobset_status = "IDLE"
      self.launch_timestamp = None
      self.current_event = "Workload deleted from GKE cluster"
      cordoned_copy = dict(self.cordoned_nodes)
      ctx = self.config["cluster_context"]
      self.cordoned_nodes.clear()
    _uncordon_all_tpu_nodes(cordoned_copy, context=ctx)

  def reset_telemetry(self) -> None:
    """Clears loss and goodput curves and resets step counter to zero."""
    with self.lock:
      self.loss_history.clear()
      self.goodput_history.clear()
      self.current_step = 0
      self.total_slices = self.config["num_slices"]
      self.standby_slices = self.config["standby_slices"]
      self.active_slices = list(range(self.total_slices))
      self.inactive_slices = []
      self.start_wall_time = None
      self.last_step_wall_time = None
      self.forward_progress_time = 0.0
      self.current_goodput = 0.0
      self.last_loss = None
      self.training_running = False
      if self.workload_ready:
        self.jobset_status = "READY"
        self.current_event = (
            "Workload reset to Step 0. Ready to start training."
        )
      cordoned_copy = dict(self.cordoned_nodes)
      ctx = self.config["cluster_context"]
      self.cordoned_nodes.clear()
    _uncordon_all_tpu_nodes(cordoned_copy, context=ctx)

  def get_snapshot(self) -> dict[str, Any]:
    with self.lock:
      return {
          "cluster": self.config["cluster_context"],
          "namespace": self.config["namespace"],
          "jobset": self.config["jobset_name"],
          "config": json.loads(json.dumps(self.config)),
          "jobset_status": self.jobset_status,
          "training_running": self.training_running,
          "workload_ready": self.workload_ready,
          "head_pod_name": self.head_pod_name,
          "head_pod_phase": self.head_pod_phase,
          "worker_pods": list(self.worker_pods),
          "current_step": self.current_step,
          "total_steps": self.total_steps,
          "active_slices": list(self.active_slices),
          "inactive_slices": list(self.inactive_slices),
          "total_slices": self.total_slices,
          "standby_slices": self.standby_slices,
          "current_goodput": round(self.current_goodput, 2),
          "last_loss": self.last_loss,
          "current_event": self.current_event,
          "loss_history": list(self.loss_history[-300:]),
          "goodput_history": list(self.goodput_history[-300:]),
          "recent_logs": list(self.log_buffer),
      }


state = BridgeState()


class ClusterPoller(threading.Thread):
  """Background thread polling Kubernetes JobSet and Pod states."""

  def __init__(self, bridge_state: BridgeState):
    super().__init__(daemon=True)
    self.bridge_state = bridge_state

  def run(self) -> None:
    _logger.info("Cluster poller started.")
    while not self.bridge_state.stopped:
      try:
        self._poll_cluster()
        self._tick_recovery_goodput()
      except (OSError, subprocess.SubprocessError, ValueError, KeyError) as e:
        _logger.debug("Cluster poll error: %s", e)
      time.sleep(1.5)

  def _tick_recovery_goodput(self) -> None:
    """Updates declining goodput during fault disruption or recovery pauses."""
    now = time.time()
    broadcast_payload = None
    with self.bridge_state.lock:
      if (
          self.bridge_state.start_wall_time is not None
          and self.bridge_state.jobset_status
          in ("DISRUPTED", "RECOVERING", "SCALING_UP")
      ):
        total_elapsed = max(0.1, now - self.bridge_state.start_wall_time)
        goodput = min(
            100.0,
            max(
                0.0,
                (self.bridge_state.forward_progress_time / total_elapsed)
                * 100.0,
            ),
        )
        self.bridge_state.current_goodput = goodput
        self.bridge_state.goodput_history.append({
            "time": round(total_elapsed - INIT_OVERHEAD_SEC, 2),
            "goodput": round(goodput, 2),
            "step": self.bridge_state.current_step,
            "active_slices": len(self.bridge_state.active_slices),
        })
        broadcast_payload = {
            "type": "goodput_tick",
            "step": self.bridge_state.current_step,
            "active_slices": len(self.bridge_state.active_slices),
            "goodput": round(goodput, 2),
            "elapsed_time": round(
                max(0.0, total_elapsed - INIT_OVERHEAD_SEC), 2
            ),
            "timestamp": now,
        }
    if broadcast_payload is not None:
      self.bridge_state.broadcast(broadcast_payload)

  def _poll_cluster(self) -> None:
    """Fetches pod phases and container statuses from GKE via kubectl."""
    with self.bridge_state.lock:
      ctx = self.bridge_state.config["cluster_context"]
      ns = self.bridge_state.config["namespace"]
      js_name = self.bridge_state.config["jobset_name"]

    cmd = _kubectl_cmd(
        "get",
        "pods",
        "-l",
        f"jobset.sigs.k8s.io/jobset-name={js_name}",
        "-o",
        "json",
        context=ctx,
        namespace=ns,
        include_ns=True,
    )
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
      return

    try:
      data = json.loads(proc.stdout)
    except json.JSONDecodeError:
      return

    items = data.get("items", [])
    head_name = ""
    head_phase = "Unknown"
    workers = []

    for item in items:
      meta = item.get("metadata", {})
      pname = meta.get("name", "")
      labels = meta.get("labels", {})
      rep_job = labels.get("jobset.sigs.k8s.io/replicatedjob-name", "")
      job_idx = labels.get("jobset.sigs.k8s.io/job-index", "")
      status = item.get("status", {})
      phase = status.get("phase", "Unknown")

      if rep_job == "pathways-head":
        head_name = pname
        head_phase = phase
      elif rep_job == "worker":
        slice_idx = int(job_idx) if job_idx.isdigit() else 0
        container_statuses = status.get("containerStatuses", [])
        ready = False
        restarts = 0
        for cs in container_statuses:
          if cs.get("name") == "pathways-worker":
            ready = cs.get("ready", False)
            restarts = cs.get("restartCount", 0)
        workers.append({
            "name": pname,
            "slice": slice_idx,
            "phase": phase,
            "ready": ready,
            "restarts": restarts,
            "node": item.get("spec", {}).get("nodeName", ""),
        })

    now = time.time()
    detected_new_launch = False
    with self.bridge_state.lock:
      prev_head_pod = self.bridge_state.head_pod_name
      self.bridge_state.head_pod_name = head_name
      self.bridge_state.head_pod_phase = head_phase
      self.bridge_state.worker_pods = sorted(workers, key=lambda w: w["name"])
      if not items:
        starting_expired = (
            self.bridge_state.jobset_status == "STARTING"
            and self.bridge_state.launch_timestamp is not None
            and (now - self.bridge_state.launch_timestamp) > 15.0
        )
        if (
            self.bridge_state.jobset_status not in ("IDLE", "STARTING")
            or starting_expired
        ):
          self.bridge_state.jobset_status = "IDLE"
          self.bridge_state.training_running = False
          self.bridge_state.workload_ready = False
          self.bridge_state.current_event = "No active JobSet running"
      elif (
          head_phase == "Failed" and self.bridge_state.jobset_status != "FAILED"
      ):
        self.bridge_state.jobset_status = "FAILED"
        self.bridge_state.training_running = False
        self.bridge_state.current_event = "Head pod exited with error"
      elif head_name and head_phase in ("Pending", "Running"):
        is_new_pod = bool(prev_head_pod and head_name != prev_head_pod)
        if is_new_pod or self.bridge_state.jobset_status in (
            "IDLE",
            "FAILED",
            "COMPLETED",
        ):
          self.bridge_state.loss_history.clear()
          self.bridge_state.goodput_history.clear()
          self.bridge_state.log_buffer.clear()
          self.bridge_state.current_step = 0
          self.bridge_state.start_wall_time = None
          self.bridge_state.last_step_wall_time = None
          self.bridge_state.forward_progress_time = 0.0
          self.bridge_state.current_goodput = 0.0
          self.bridge_state.last_loss = None
          self.bridge_state.training_running = False
          self.bridge_state.workload_ready = False
          self.bridge_state.jobset_status = "STARTING"
          self.bridge_state.launch_timestamp = now
          self.bridge_state.current_event = (
              "Detected JobSet pods on GKE, initializing and pre-compiling..."
          )
          detected_new_launch = True

    if detected_new_launch:
      self.bridge_state.broadcast(
          {"type": "launched", "snapshot": self.bridge_state.get_snapshot()}
      )


class LogTailer(threading.Thread):
  """Background thread streaming kubectl logs from the JAX head pod container."""

  def __init__(self, bridge_state: BridgeState):
    super().__init__(daemon=True)
    self.bridge_state = bridge_state
    self.active_process: subprocess.Popen[str] | None = None
    self.finished_pods: set[str] = set()

  def stop_current_tail(self) -> None:
    """Terminates the active kubectl logs process if one is running."""
    proc = self.active_process
    if proc is not None:
      try:
        proc.terminate()
      except OSError:
        pass

  def run(self) -> None:
    _logger.info("Log tailer monitor started.")
    while not self.bridge_state.stopped:
      with self.bridge_state.lock:
        head_pod = self.bridge_state.head_pod_name
        head_phase = self.bridge_state.head_pod_phase
        ctx = self.bridge_state.config["cluster_context"]
        ns = self.bridge_state.config["namespace"]

      pod_key = f"{ctx}/{ns}/{head_pod}"
      if (
          head_pod
          and head_phase == "Running"
          and pod_key not in self.finished_pods
      ):
        self._tail_logs(head_pod, ctx, ns, pod_key)
      else:
        time.sleep(1.5)

  def _tail_logs(self, pod_name: str, ctx: str, ns: str, pod_key: str) -> None:
    """Streams and parses stdout from the head pod jax-tpu container."""
    cmd = _kubectl_cmd(
        "logs",
        "-f",
        pod_name,
        "-c",
        "jax-tpu",
        context=ctx,
        namespace=ns,
        include_ns=True,
    )
    _logger.info(
        "Starting log tail process on pod %s (context=%s, ns=%s)...",
        pod_name,
        ctx,
        ns,
    )
    try:
      self.active_process = subprocess.Popen(
          cmd,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          text=True,
          bufsize=1,
      )
    except OSError as e:
      _logger.warning("Failed to launch kubectl logs: %s", e)
      time.sleep(3.0)
      return

    step_regex = re.compile(
        r"Step\s+(\d+)\s+\|\s+Active\s+Slices:\s+(\d+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+Step\s+Time:\s+([\d.]+)s"
    )
    disrupt_regex = re.compile(
        r"INTERRUPTING\s+SLICE\s+(\d+)\s+AT\s+STEP\s+(\d+)"
    )
    stage1_regex = re.compile(
        r"Stage\s+1:\s+Active\s+slices\s+detected:\s+(frozenset\(\{[^}]*\}\)|\S+)"
    )
    stage2_regex = re.compile(
        r"Stage\s+2:\s+Activating\s+pre-compiled\s+(\d+)-slice"
    )

    stdout = self.active_process.stdout
    if stdout is None:
      return

    for line in iter(stdout.readline, ""):
      clean_line = line.rstrip()
      if clean_line:
        self.bridge_state.record_log(clean_line)
        self._parse_line(
            clean_line,
            pod_name,
            ctx,
            ns,
            step_regex,
            disrupt_regex,
            stage1_regex,
            stage2_regex,
        )

    self.active_process.wait()
    self.active_process = None
    if self.bridge_state.head_pod_phase in ("Succeeded", "Failed", "Completed"):
      self.finished_pods.add(pod_key)
    time.sleep(1.5)

  def _parse_line(
      self,
      line: str,
      pod_name: str,
      ctx: str,
      ns: str,
      step_re: re.Pattern[str],
      disrupt_re: re.Pattern[str],
      stage1_re: re.Pattern[str],
      stage2_re: re.Pattern[str],
  ) -> None:
    """Parses a single JAX driver log line and updates bridge telemetry."""
    now = time.time()

    if "WORKLOAD READY" in line:
      if "Workload reset to Step 0" in line:
        self.bridge_state.reset_telemetry()
      auto_start = False
      with self.bridge_state.lock:
        self.bridge_state.workload_ready = True
        self.bridge_state.training_running = False
        self.bridge_state.active_slices = list(
            range(self.bridge_state.total_slices)
        )
        self.bridge_state.inactive_slices = []
        self.bridge_state.jobset_status = "READY"
        self.bridge_state.current_event = (
            "Pre-compilation complete. Ready to Start Training!"
        )
        active_copy = list(self.bridge_state.active_slices)
        total_s = self.bridge_state.total_slices
        if self.bridge_state.auto_start_when_ready:
          self.bridge_state.auto_start_when_ready = False
          auto_start = True

      self.bridge_state.broadcast({
          "type": "event",
          "event": "workload_ready",
          "active_slices": active_copy,
          "message": (
              f"Pre-compilation complete ({total_s} slices ready). Waiting for"
              " Start Training signal..."
          ),
          "timestamp": now,
      })
      if auto_start:
        threading.Thread(
            target=_send_workload_control_rpc,
            args=(pod_name, "/start", ctx, ns),
            daemon=True,
        ).start()
      return

    if "TRAINING STARTED" in line or "TRAINING RESUMED" in line:
      with self.bridge_state.lock:
        self.bridge_state.workload_ready = True
        self.bridge_state.training_running = True
        self.bridge_state.jobset_status = "RUNNING"
        total_s = self.bridge_state.total_slices
        self.bridge_state.current_event = (
            f"Training actively on {len(self.bridge_state.active_slices)}"
            " slices"
        )
        if self.bridge_state.start_wall_time is None:
          self.bridge_state.start_wall_time = now - INIT_OVERHEAD_SEC
          self.bridge_state.goodput_history.append({
              "time": 0.0,
              "goodput": 0.0,
              "step": 0,
              "active_slices": total_s,
          })
        self.bridge_state.last_step_wall_time = now

      self.bridge_state.broadcast({
          "type": "event",
          "event": "training_started",
          "message": (
              "Start Training signal received! Executing training steps..."
          ),
          "timestamp": now,
      })
      return

    if "TRAINING PAUSED" in line:
      with self.bridge_state.lock:
        self.bridge_state.training_running = False
        self.bridge_state.jobset_status = "PAUSED"
        self.bridge_state.last_step_wall_time = None
        self.bridge_state.current_event = "Training paused by user"

      self.bridge_state.broadcast({
          "type": "event",
          "event": "training_paused",
          "message": "Training paused by user.",
          "timestamp": now,
      })
      return

    m_step = step_re.search(line)
    if m_step:
      step = int(m_step.group(1))
      active_slices = int(m_step.group(2))
      loss = float(m_step.group(3))
      step_time = float(m_step.group(4))

      with self.bridge_state.lock:
        self.bridge_state.workload_ready = True
        self.bridge_state.training_running = True
        if self.bridge_state.start_wall_time is None:
          self.bridge_state.start_wall_time = now - INIT_OVERHEAD_SEC
          self.bridge_state.goodput_history.append({
              "time": 0.0,
              "goodput": 0.0,
              "step": 0,
              "active_slices": active_slices,
          })

        self.bridge_state.current_step = step
        self.bridge_state.last_loss = loss
        total_s = self.bridge_state.total_slices
        if active_slices >= total_s:
          if self.bridge_state.jobset_status not in (
              "DISRUPTED",
              "RECOVERING",
          ):
            self.bridge_state.active_slices = list(range(total_s))
            self.bridge_state.inactive_slices = []
            self.bridge_state.jobset_status = "RUNNING"
            self.bridge_state.current_event = (
                f"Training actively on {active_slices} slices"
            )
        else:
          if len(self.bridge_state.active_slices) != active_slices:
            surviving = [
                i
                for i in range(total_s)
                if i not in self.bridge_state.inactive_slices
            ][:active_slices]
            self.bridge_state.active_slices = surviving
            self.bridge_state.inactive_slices = [
                i for i in range(total_s) if i not in surviving
            ]
          if self.bridge_state.jobset_status != "SCALING_UP":
            self.bridge_state.jobset_status = "DEGRADED"
            self.bridge_state.current_event = (
                f"Degraded execution on {active_slices} slice(s)"
            )

        if (
            self.bridge_state.last_step_wall_time is not None
            and (now - self.bridge_state.last_step_wall_time) < 1.5
        ):
          progress_contribution = now - self.bridge_state.last_step_wall_time
        else:
          progress_contribution = min(step_time, 0.7)

        self.bridge_state.last_step_wall_time = now
        self.bridge_state.forward_progress_time += progress_contribution

        total_elapsed = max(0.1, now - self.bridge_state.start_wall_time)
        elapsed_since_start = max(0.0, total_elapsed - INIT_OVERHEAD_SEC)
        goodput = min(
            100.0,
            max(
                0.0,
                (self.bridge_state.forward_progress_time / total_elapsed)
                * 100.0,
            ),
        )
        self.bridge_state.current_goodput = goodput

        self.bridge_state.loss_history.append({
            "step": step,
            "loss": loss,
            "time": round(elapsed_since_start, 2),
        })
        self.bridge_state.goodput_history.append({
            "time": round(elapsed_since_start, 2),
            "goodput": round(goodput, 2),
            "step": step,
            "active_slices": active_slices,
        })

      self.bridge_state.broadcast({
          "type": "step",
          "step": step,
          "active_slices": active_slices,
          "total_slices": self.bridge_state.total_slices,
          "loss": loss,
          "step_time": step_time,
          "goodput": round(goodput, 2),
          "elapsed_time": round(elapsed_since_start, 2),
          "timestamp": now,
      })
      return

    m_disrupt = disrupt_re.search(line)
    if m_disrupt:
      s_idx = int(m_disrupt.group(1))
      step_idx = int(m_disrupt.group(2))
      with self.bridge_state.lock:
        self.bridge_state.jobset_status = "DISRUPTED"
        self.bridge_state.active_slices = [
            i for i in range(self.bridge_state.total_slices) if i != s_idx
        ]
        self.bridge_state.inactive_slices = [s_idx]
        self.bridge_state.last_step_wall_time = None
        self.bridge_state.current_event = (
            f"Slice {s_idx} interrupted at step {step_idx}!"
        )

      self.bridge_state.broadcast({
          "type": "event",
          "event": "disrupted",
          "slice": s_idx,
          "step": step_idx,
          "message": f"Slice {s_idx} interrupted at step {step_idx}!",
          "timestamp": now,
      })
      return

    if "Elastic event detected" in line:
      should_emit = False
      disrupted_slice = 1
      with self.bridge_state.lock:
        if self.bridge_state.inactive_slices:
          disrupted_slice = self.bridge_state.inactive_slices[0]
        elif self.bridge_state.total_slices > 1:
          disrupted_slice = self.bridge_state.total_slices - 1
        else:
          disrupted_slice = 0
        if (
            self.bridge_state.workload_ready
            and self.bridge_state.jobset_status
            not in ("DISRUPTED", "RECOVERING", "SCALING_UP")
        ):
          self.bridge_state.jobset_status = "DISRUPTED"
          self.bridge_state.active_slices = [
              i
              for i in range(self.bridge_state.total_slices)
              if i != disrupted_slice
          ]
          self.bridge_state.inactive_slices = [disrupted_slice]
          self.bridge_state.last_step_wall_time = None
          self.bridge_state.current_event = (
              f"DATA_LOSS detected on Slice {disrupted_slice}! Initiating"
              " elastic recovery..."
          )
          should_emit = True
      if should_emit:
        self.bridge_state.broadcast({
            "type": "event",
            "event": "disrupted",
            "slice": disrupted_slice,
            "step": self.bridge_state.current_step,
            "message": (
                f"JAX DATA_LOSS detected on Slice {disrupted_slice}!"
                " Initiating elastic recovery..."
            ),
            "timestamp": now,
        })
      return

    if "ELASTIC RECOVERY TRIGGERED" in line:
      with self.bridge_state.lock:
        self.bridge_state.jobset_status = "RECOVERING"
        self.bridge_state.last_step_wall_time = None
        self.bridge_state.current_event = (
            "Elastic recovery triggered - reconfiguring mesh"
        )

      self.bridge_state.broadcast({
          "type": "event",
          "event": "recovery_started",
          "message": (
              "Elastic recovery triggered - cleaning JAX caches and"
              " discovering slices"
          ),
          "timestamp": now,
      })
      return

    m_stage1 = stage1_re.search(line)
    if m_stage1:
      slices_str = m_stage1.group(1)
      detected_indices = [int(x) for x in re.findall(r"\d+", slices_str)]
      with self.bridge_state.lock:
        if detected_indices:
          self.bridge_state.active_slices = detected_indices
          self.bridge_state.inactive_slices = [
              i
              for i in range(self.bridge_state.total_slices)
              if i not in detected_indices
          ]
        self.bridge_state.current_event = (
            f"Stage 3: Health check detected active slices {m_stage1.group(1)}"
        )
      self.bridge_state.broadcast({
          "type": "event",
          "event": "stage1",
          "active_slices": detected_indices,
          "message": (
              f"Topology probe verified active slices {m_stage1.group(1)}"
          ),
          "timestamp": now,
      })
      return

    m_stage2 = stage2_re.search(line)
    if m_stage2:
      s_count = int(m_stage2.group(1))
      with self.bridge_state.lock:
        self.bridge_state.current_event = (
            f"Stage 4: Activating pre-compiled {s_count}-slice executable"
        )
      self.bridge_state.broadcast({
          "type": "event",
          "event": "stage2",
          "slice_count": s_count,
          "message": f"Activated pre-compiled {s_count}-slice mesh executable",
          "timestamp": now,
      })
      return

    if "Stage 3: Restoring model parameters" in line:
      with self.bridge_state.lock:
        self.bridge_state.current_event = (
            "Stage 5: Restoring model parameters and Grain iterator"
        )
      self.bridge_state.broadcast({
          "type": "event",
          "event": "stage3",
          "message": (
              "Restoring weights from host-pinned snapshot & Grain elastic"
              " iterator"
          ),
          "timestamp": now,
      })
      return

    if "ELASTIC RECOVERY COMPLETE" in line:
      with self.bridge_state.lock:
        active_cnt = len(self.bridge_state.active_slices)
        total_s = self.bridge_state.total_slices
        self.bridge_state.jobset_status = (
            "RUNNING" if active_cnt >= total_s else "DEGRADED"
        )
        self.bridge_state.last_step_wall_time = now
        self.bridge_state.current_event = (
            f"Recovery complete! Resuming training on {active_cnt} slice(s)"
        )
        active_copy = list(self.bridge_state.active_slices)
      self.bridge_state.broadcast({
          "type": "event",
          "event": "recovery_complete",
          "active_slices": active_copy,
          "message": (
              "Recovery complete - Forward training resumed on"
              f" {len(active_copy)} slice(s)!"
          ),
          "timestamp": now,
      })
      return

    if "SCALE UP DETECTED" in line:
      with self.bridge_state.lock:
        total_s = self.bridge_state.total_slices
        self.bridge_state.jobset_status = "SCALING_UP"
        self.bridge_state.active_slices = list(range(total_s))
        self.bridge_state.inactive_slices = []
        self.bridge_state.last_step_wall_time = None
        self.bridge_state.current_event = (
            "Scale-up detected: Reconnected slice rejoining mesh!"
        )
      self.bridge_state.broadcast({
          "type": "event",
          "event": "scale_up_detected",
          "message": (
              f"Scale-up detected - Gracefully resizing mesh to {total_s}"
              " slices!"
          ),
          "timestamp": now,
      })
      return

    if "Training finished successfully" in line:
      with self.bridge_state.lock:
        self.bridge_state.jobset_status = "COMPLETED"
        self.bridge_state.training_running = False
        self.bridge_state.current_event = "Training run completed successfully!"
      self.bridge_state.broadcast({
          "type": "event",
          "event": "finished",
          "message": "Training finished successfully!",
          "timestamp": now,
      })


tailer_instance: LogTailer | None = None


def _build_head_startup_script(cfg: dict[str, Any], instance_type: str) -> str:
  """Generates the bash entrypoint script for the head pod container."""
  preset = cfg.get("workload_preset", "builtin_elastic_demo")
  cm_files = cfg.get("configmap_files", [])
  file_names = {entry.get("filename", "") for entry in cm_files}

  lines = [
      'echo "=== Head Pod Starting $(date) ==="',
  ]

  if preset == "builtin_elastic_demo":
    lines.extend([
        "pip install --no-cache-dir grain",
        'echo "Grain package installed successfully"',
        (
            "pip install --no-cache-dir --no-deps"
            ' "git+https://github.com/google/orbax.git@test_996639160#subdirectory=checkpoint"'
        ),
        (
            'echo "Orbax checkpoint installed from branch test_996639160'
            ' successfully"'
        ),
    ])
    if "snapshotter.py" in file_names:
      lines.extend([
          (
              'ORBAX_PW_DIR=$(python3 -c "import'
              " orbax.checkpoint.experimental.v1._src.training.pathways as p,"
              ' os; print(os.path.dirname(p.__file__))")'
          ),
          'cp /app/workload/snapshotter.py "${ORBAX_PW_DIR}/snapshotter.py"',
          (
              'echo "Injected local snapshotter.py into ${ORBAX_PW_DIR}'
              ' successfully"'
          ),
      ])
    lines.extend([
        (
            'PW_DIR=$(python3 -c "import pathwaysutils, os;'
            ' print(os.path.dirname(pathwaysutils.__file__))")'
        ),
        'echo "Found installed pathwaysutils at ${PW_DIR}"',
        'mkdir -p "${PW_DIR}/extras/simple_model" "${PW_DIR}/extras/elastic"',
        (
            'touch "${PW_DIR}/extras/__init__.py"'
            ' "${PW_DIR}/extras/elastic/__init__.py"'
        ),
        (
            'for f in /app/workload/sm_*.py; do [ -f "$f" ] && cp "$f"'
            ' "${PW_DIR}/extras/simple_model/$(basename "$f" | sed'
            " 's/^sm_//')\"; done"
        ),
    ])
    if "elastic_workload.py" in file_names:
      lines.extend([
          'cp /app/workload/elastic_workload.py "${PW_DIR}/extras/elastic/"',
      ])
    if "mock_fault_injection.py" in file_names:
      lines.extend([
          'mkdir -p "${PW_DIR}/test"',
          'cp /app/workload/mock_fault_injection.py "${PW_DIR}/test/"',
          'touch "${PW_DIR}/test/__init__.py"',
      ])
    if "manager.py" in file_names:
      lines.append('cp /app/workload/manager.py "${PW_DIR}/elastic/manager.py"')

    workload_cmd = (
        "python3 -m pathwaysutils.extras.elastic.elastic_workload \\\n"
        f"  --config_name={instance_type} \\\n"
        f"  --total_steps={cfg['total_steps']} \\\n"
        f"  --elasticity_mode={cfg['elasticity_mode']} \\\n"
        "  --simulate_faults=true \\\n"
        "  --enable_control_server=true \\\n"
        "  --control_port=8090 \\\n"
        "  --wait_for_start_signal=true"
    )
  else:
    if cm_files:
      lines.extend([
          'export PYTHONPATH="/app/workload:${PYTHONPATH}"',
          'echo "Mounted custom ConfigMap files in /app/workload:"',
          "ls -la /app/workload",
      ])
    workload_cmd = cfg.get("custom_command", "").strip()
    if not workload_cmd:
      workload_cmd = "python3 -m pathwaysutils.extras.elastic.elastic_workload"

  lines.extend([
      "_sigterm() { kill -SIGTERM $PID 2>/dev/null; }",
      "trap _sigterm SIGTERM",
      f"{workload_cmd} &",
      "PID=$!",
      "while kill -0 $PID 2>/dev/null; do",
      "  sleep 2",
      "done",
      "wait $PID",
      "EXIT_CODE=$?",
      'echo "=== Workload Finished $(date) with exit code $EXIT_CODE ==="',
      "exit $EXIT_CODE",
  ])
  return "\n".join(lines) + "\n"


def _build_jobset_manifest(
    cfg: dict[str, Any], has_configmap: bool
) -> dict[str, Any]:
  """Builds the Kubernetes JobSet manifest dict from the current configuration."""
  topo_specs = compute_topology_specs(cfg["tpu_type"], cfg["topology"])
  instance_type = topo_specs["instance_type"]
  accelerator = topo_specs["accelerator"]
  topology = topo_specs["topology"]
  vms_per_slice = topo_specs["vms_per_slice"]
  chips_per_vm = topo_specs["chips_per_vm"]
  num_slices = int(cfg["num_slices"])
  gcs_scratch = cfg["gcs_scratch_location"]
  server_image = cfg["server_image"]
  proxy_image = cfg["proxy_image"]
  container_image = cfg["container_image"]

  head_volume_mounts = [{"mountPath": "/tmp", "name": "shared-tmp"}]
  head_volumes: list[dict[str, Any]] = [{
      "hostPath": {"path": "/tmp", "type": "DirectoryOrCreate"},
      "name": "shared-tmp",
  }]
  if has_configmap:
    head_volume_mounts.insert(
        0, {"mountPath": "/app/workload", "name": "workload-code"}
    )
    head_volumes.insert(
        0,
        {
            "name": "workload-code",
            "configMap": {"name": "elastic-workload-code"},
        },
    )

  head_node_selector: dict[str, str] = {}
  if cfg.get("cpu_nodepool"):
    head_node_selector["cloud.google.com/gke-nodepool"] = cfg["cpu_nodepool"]

  head_pod_spec: dict[str, Any] = {
      "containers": [{
          "command": [
              "bash",
              "-c",
              _build_head_startup_script(cfg, instance_type),
          ],
          "env": [
              {
                  "name": "PATHWAYS_HEAD",
                  "valueFrom": {
                      "fieldRef": {
                          "fieldPath": (
                              "metadata.labels['jobset.sigs.k8s.io/coordinator']"
                          )
                      }
                  },
              },
              {"name": "JAX_PLATFORMS", "value": "proxy"},
              {"name": "XCLOUD_ENVIRONMENT", "value": "GCP"},
              {"name": "TPU_VMODULE", "value": "real_program_continuator=1"},
              {"name": "ENABLE_PATHWAYS_PERSISTENCE", "value": "1"},
              {"name": "ENABLE_PERSISTENCE_API", "value": "1"},
              {"name": "ENABLE_PJRT_COMPATIBILITY", "value": "true"},
              {
                  "name": "JAX_BACKEND_TARGET",
                  "value": "grpc://$(PATHWAYS_HEAD):29000",
              },
          ],
          "image": container_image,
          "imagePullPolicy": "Always",
          "name": "jax-tpu",
          "ports": [{"containerPort": 8090, "protocol": "TCP"}],
          "resources": {"limits": {"cpu": "24", "memory": "100G"}},
          "securityContext": {"privileged": True},
          "volumeMounts": head_volume_mounts,
      }],
      "dnsPolicy": "ClusterFirstWithHostNet",
      "hostNetwork": True,
      "initContainers": [
          {
              "args": [
                  "--server_port=29001",
                  f"--gcs_scratch_location={gcs_scratch}",
                  "--node_type=resource_manager",
                  f"--instance_count={num_slices}",
                  f"--instance_type={instance_type}",
                  "--xla_tpu_use_enhanced_launch_barrier=true",
              ],
              "env": [
                  {
                      "name": "REPLICATED_JOB_NAME",
                      "valueFrom": {
                          "fieldRef": {
                              "fieldPath": (
                                  "metadata.annotations['jobset.sigs.k8s.io/replicatedjob-name']"
                              )
                          }
                      },
                  },
                  {
                      "name": "JOBSET_NAME",
                      "valueFrom": {
                          "fieldRef": {
                              "fieldPath": (
                                  "metadata.annotations['jobset.sigs.k8s.io/jobset-name']"
                              )
                          }
                      },
                  },
                  {
                      "name": "HOST_ADDRESS",
                      "valueFrom": {
                          "fieldRef": {
                              "fieldPath": (
                                  "metadata.labels['jobset.sigs.k8s.io/coordinator']"
                              )
                          }
                      },
                  },
                  {"name": "TPU_SKIP_MDS_QUERY", "value": "true"},
              ],
              "image": server_image,
              "imagePullPolicy": "Always",
              "name": "pathways-rm",
              "ports": [
                  {"containerPort": 29001, "protocol": "TCP"},
                  {"containerPort": 29002, "protocol": "TCP"},
              ],
              "resources": {"limits": {"cpu": "8", "memory": "32G"}},
              "restartPolicy": "Always",
          },
          {
              "args": [
                  "--server_port=29000",
                  "--resource_manager_address=$(PATHWAYS_HEAD):29001",
                  f"--gcs_scratch_location={gcs_scratch}",
                  f"--num_elastic_slices={num_slices}",
                  "--xla_tpu_use_enhanced_launch_barrier=true",
              ],
              "env": [{
                  "name": "PATHWAYS_HEAD",
                  "valueFrom": {
                      "fieldRef": {
                          "fieldPath": (
                              "metadata.labels['jobset.sigs.k8s.io/coordinator']"
                          )
                      }
                  },
              }],
              "image": proxy_image,
              "imagePullPolicy": "Always",
              "name": "pathways-proxy",
              "ports": [{"containerPort": 29000, "protocol": "TCP"}],
              "resources": {"limits": {"cpu": "16", "memory": "100G"}},
              "restartPolicy": "Always",
          },
      ],
      "restartPolicy": "Never",
      "volumes": head_volumes,
  }
  if head_node_selector:
    head_pod_spec["nodeSelector"] = head_node_selector

  metadata: dict[str, Any] = {
      "name": cfg["jobset_name"],
      "namespace": cfg["namespace"],
  }
  if cfg.get("kueue_queue_name"):
    metadata["labels"] = {"kueue.x-k8s.io/queue-name": cfg["kueue_queue_name"]}

  return {
      "apiVersion": "jobset.x-k8s.io/v1alpha2",
      "kind": "JobSet",
      "metadata": metadata,
      "spec": {
          "coordinator": {"replicatedJob": "pathways-head"},
          "failurePolicy": {"restartStrategy": "Recreate"},
          "network": {
              "enableDNSHostnames": True,
              "publishNotReadyAddresses": True,
          },
          "replicatedJobs": [
              {
                  "name": "pathways-head",
                  "replicas": 1,
                  "template": {
                      "metadata": {
                          "annotations": {
                              "alpha.jobset.sigs.k8s.io/exclusive-topology": (
                                  "kubernetes.io/hostname"
                              )
                          }
                      },
                      "spec": {
                          "backoffLimit": 0,
                          "completionMode": "Indexed",
                          "completions": 1,
                          "parallelism": 1,
                          "template": {
                              "metadata": {
                                  "annotations": {
                                      "alpha.jobset.sigs.k8s.io/exclusive-topology": (
                                          "kubernetes.io/hostname"
                                      )
                                  },
                                  "labels": {
                                      "kueue.x-k8s.io/podset": "pathways-head"
                                  },
                              },
                              "spec": head_pod_spec,
                          },
                      },
                  },
              },
              {
                  "name": "worker",
                  "replicas": num_slices,
                  "template": {
                      "metadata": {},
                      "spec": {
                          "backoffLimit": 32000000,
                          "completionMode": "Indexed",
                          "completions": vms_per_slice,
                          "parallelism": vms_per_slice,
                          "template": {
                              "metadata": {
                                  "annotations": {
                                      "alpha.jobset.sigs.k8s.io/exclusive-topology": (
                                          "cloud.google.com/gke-nodepool"
                                      )
                                  },
                                  "labels": {"kueue.x-k8s.io/podset": "worker"},
                              },
                              "spec": {
                                  "containers": [{
                                      "args": [
                                          "--resource_manager_address=$(PATHWAYS_HEAD):29001",
                                          "--server_port=29005",
                                          f"--gcs_scratch_location={gcs_scratch}",
                                          "--xla_tpu_use_enhanced_launch_barrier=true",
                                      ],
                                      "env": [
                                          {
                                              "name": "TPU_MIN_LOG_LEVEL",
                                              "value": "0",
                                          },
                                          {
                                              "name": "TF_CPP_MIN_LOG_LEVEL",
                                              "value": "0",
                                          },
                                          {
                                              "name": "XCLOUD_ENVIRONMENT",
                                              "value": "GCP",
                                          },
                                          {
                                              "name": (
                                                  "MEGASCALE_GRPC_ENABLE_XOR_TRACER"
                                              ),
                                              "value": "false",
                                          },
                                          {
                                              "name": "MEGASCALE_NUM_SLICES",
                                              "valueFrom": {
                                                  "fieldRef": {
                                                      "fieldPath": (
                                                          "metadata.labels['jobset.sigs.k8s.io/replicatedjob-replicas']"
                                                      )
                                                  }
                                              },
                                          },
                                          {
                                              "name": "JOBSET_NAME",
                                              "valueFrom": {
                                                  "fieldRef": {
                                                      "fieldPath": (
                                                          "metadata.annotations['jobset.sigs.k8s.io/jobset-name']"
                                                      )
                                                  }
                                              },
                                          },
                                          {
                                              "name": "REPLICATED_JOB_NAME",
                                              "valueFrom": {
                                                  "fieldRef": {
                                                      "fieldPath": (
                                                          "metadata.annotations['jobset.sigs.k8s.io/replicatedjob-name']"
                                                      )
                                                  }
                                              },
                                          },
                                          {
                                              "name": "MEGASCALE_SLICE_ID",
                                              "valueFrom": {
                                                  "fieldRef": {
                                                      "fieldPath": (
                                                          "metadata.labels['jobset.sigs.k8s.io/job-index']"
                                                      )
                                                  }
                                              },
                                          },
                                          {
                                              "name": "PATHWAYS_HEAD",
                                              "valueFrom": {
                                                  "fieldRef": {
                                                      "fieldPath": (
                                                          "metadata.labels['jobset.sigs.k8s.io/coordinator']"
                                                      )
                                                  }
                                              },
                                          },
                                          {
                                              "name": (
                                                  "MEGASCALE_COORDINATOR_ADDRESS"
                                              ),
                                              "valueFrom": {
                                                  "fieldRef": {
                                                      "fieldPath": (
                                                          "metadata.labels['jobset.sigs.k8s.io/coordinator']"
                                                      )
                                                  }
                                              },
                                          },
                                      ],
                                      "image": server_image,
                                      "imagePullPolicy": "Always",
                                      "name": "pathways-worker",
                                      "ports": [
                                          {
                                              "containerPort": 29005,
                                              "protocol": "TCP",
                                          },
                                          {
                                              "containerPort": 29006,
                                              "protocol": "TCP",
                                          },
                                          {
                                              "containerPort": 8471,
                                              "protocol": "TCP",
                                          },
                                          {
                                              "containerPort": 8080,
                                              "protocol": "TCP",
                                          },
                                      ],
                                      "resources": {
                                          "limits": {
                                              "google.com/tpu": str(
                                                  chips_per_vm
                                              )
                                          }
                                      },
                                      "volumeMounts": [{
                                          "mountPath": "/tmp",
                                          "name": "shared-tmp",
                                      }],
                                  }],
                                  "dnsPolicy": "ClusterFirstWithHostNet",
                                  "hostNetwork": True,
                                  "nodeSelector": {
                                      "cloud.google.com/gke-tpu-accelerator": (
                                          accelerator
                                      ),
                                      "cloud.google.com/gke-tpu-topology": (
                                          topology
                                      ),
                                  },
                                  "restartPolicy": "OnFailure",
                                  "volumes": [{
                                      "hostPath": {
                                          "path": "/tmp",
                                          "type": "DirectoryOrCreate",
                                      },
                                      "name": "shared-tmp",
                                  }],
                              },
                          },
                      },
                  },
              },
          ],
          "startupPolicy": {"startupPolicyOrder": "InOrder"},
          "successPolicy": {
              "operator": "All",
              "targetReplicatedJobs": ["pathways-head"],
          },
      },
  }


_YAML_SAFE_STR_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_./-]*$")
_YAML_RESERVED = frozenset(
    {"true", "false", "null", "yes", "no", "on", "off", "~"}
)


def _yaml_scalar(value: Any) -> str:
  """Formats a scalar value as valid Kubernetes YAML 1.2."""
  if value is None:
    return "null"
  if isinstance(value, bool):
    return "true" if value else "false"
  if isinstance(value, (int, float)):
    return str(value)
  s = str(value)
  if not s or s.lower() in _YAML_RESERVED or not _YAML_SAFE_STR_RE.match(s):
    return json.dumps(s)
  return s


def _manifest_to_yaml(obj: Any, indent: int = 0) -> str:
  """Serializes a nested manifest dict or list into clean Kubernetes YAML."""
  spaces = " " * indent
  if isinstance(obj, dict):
    if not obj:
      return f"{spaces}{{}}"
    lines: list[str] = []
    for k, v in obj.items():
      key_str = _yaml_scalar(str(k))
      if isinstance(v, dict):
        if not v:
          lines.append(f"{spaces}{key_str}: {{}}")
        else:
          lines.append(f"{spaces}{key_str}:")
          lines.append(_manifest_to_yaml(v, indent + 2))
      elif isinstance(v, list):
        if not v:
          lines.append(f"{spaces}{key_str}: []")
        else:
          lines.append(f"{spaces}{key_str}:")
          lines.append(_manifest_to_yaml(v, indent + 2))
      elif isinstance(v, str) and "\n" in v:
        lines.append(f"{spaces}{key_str}: |-")
        for subline in v.rstrip("\n").split("\n"):
          lines.append(f"{' ' * (indent + 2)}{subline}")
      else:
        lines.append(f"{spaces}{key_str}: {_yaml_scalar(v)}")
    return "\n".join(lines)

  if isinstance(obj, list):
    if not obj:
      return f"{spaces}[]"
    lines = []
    for item in obj:
      if isinstance(item, dict) and item:
        rendered = _manifest_to_yaml(item, indent + 2)
        first_nl = rendered.find("\n")
        if first_nl == -1:
          lines.append(f"{spaces}- {rendered.lstrip()}")
        else:
          first_line = rendered[:first_nl].lstrip()
          rest = rendered[first_nl + 1 :]
          lines.append(f"{spaces}- {first_line}\n{rest}")
      elif isinstance(item, list):
        lines.append(f"{spaces}-")
        lines.append(_manifest_to_yaml(item, indent + 2))
      elif isinstance(item, str) and "\n" in item:
        lines.append(f"{spaces}- |-")
        for subline in item.rstrip("\n").split("\n"):
          lines.append(f"{' ' * (indent + 2)}{subline}")
      else:
        lines.append(f"{spaces}- {_yaml_scalar(item)}")
    return "\n".join(lines)

  return f"{spaces}{_yaml_scalar(obj)}"


def _resolve_allowed_configmap_files(
    cfg: dict[str, Any],
) -> tuple[list[tuple[str, str]], str | None]:
  """Validates ConfigMap entries and returns (filename, real_path) pairs."""
  cm_files = list(cfg.get("configmap_files", []))
  if cfg.get("workload_preset") == "builtin_elastic_demo" and os.path.isdir(
      SIMPLE_MODEL_DIR
  ):
    existing_names = {entry.get("filename", "") for entry in cm_files}
    for mod_name in SIMPLE_MODEL_MODULES:
      sm_key = f"sm_{mod_name}"
      sm_path = os.path.join(SIMPLE_MODEL_DIR, mod_name)
      if sm_key not in existing_names and os.path.isfile(sm_path):
        cm_files.append({"filename": sm_key, "host_path": sm_path})

  resolved: list[tuple[str, str]] = []
  for item in cm_files:
    fname = str(item.get("filename", "")).strip()
    hpath = str(item.get("host_path", "")).strip()
    if not fname or not hpath:
      continue
    if not _SAFE_FILENAME_RE.match(fname):
      return [], f"Invalid ConfigMap filename: {fname!r}"
    real_path = os.path.realpath(hpath)
    is_allowed_root = any(
        os.path.commonpath([real_path, root]) == root
        for root in ALLOWED_CONFIGMAP_ROOTS
        if os.path.exists(root)
    )
    if not is_allowed_root or not real_path.endswith(".py"):
      return (
          [],
          (
              "ConfigMap path must be a .py file inside pathwaysutils or"
              f" orbax: {hpath}"
          ),
      )
    if not os.path.isfile(real_path):
      return [], f"ConfigMap local file not found: {hpath}"
    resolved.append((fname, real_path))
  return resolved, None


def _build_configmap_manifest(
    cfg: dict[str, Any], resolved_files: list[tuple[str, str]]
) -> dict[str, Any] | None:
  """Builds a Kubernetes ConfigMap dict from validated local Python files."""
  if not resolved_files:
    return None
  data: dict[str, str] = {}
  for fname, real_path in resolved_files:
    with open(real_path, "r", encoding="utf-8") as f:
      data[fname] = f.read()
  return {
      "apiVersion": "v1",
      "kind": "ConfigMap",
      "metadata": {
          "name": "elastic-workload-code",
          "namespace": cfg["namespace"],
      },
      "data": data,
  }


def _build_kubectl_apply_cmd(
    cfg: dict[str, Any], yaml_path: str = JOBSET_YAML_PATH
) -> str:
  """Returns the terminal kubectl apply command for the generated YAML file."""
  ctx = cfg.get("cluster_context", "")
  ns = cfg.get("namespace", "default")
  parts = ["kubectl"]
  if ctx:
    parts.append(f"--context={ctx}")
  if ns:
    parts.extend(["-n", ns])
  parts.extend(["apply", "-f", yaml_path])
  return " ".join(parts)


def _build_yaml_preview(cfg: dict[str, Any]) -> str:
  """Builds the Kubernetes JobSet YAML string for the given configuration."""
  cm_files = list(cfg.get("configmap_files", []))
  has_configmap = bool(cm_files) or (
      cfg.get("workload_preset") == "builtin_elastic_demo"
      and os.path.isdir(SIMPLE_MODEL_DIR)
  )
  manifest = _build_jobset_manifest(cfg, has_configmap=has_configmap)
  return _manifest_to_yaml(manifest) + "\n"


def _build_yaml_bundle(
    cfg: dict[str, Any], jobset_yaml_override: str | None = None
) -> tuple[bool, str, str]:
  """Builds the full deployable YAML bundle (ConfigMap + JobSet)."""
  resolved_files, err = _resolve_allowed_configmap_files(cfg)
  if err is not None:
    return False, err, ""
  cm_manifest = _build_configmap_manifest(cfg, resolved_files)
  if jobset_yaml_override and jobset_yaml_override.strip():
    jobset_yaml = jobset_yaml_override.rstrip() + "\n"
  else:
    js_manifest = _build_jobset_manifest(
        cfg, has_configmap=cm_manifest is not None
    )
    jobset_yaml = _manifest_to_yaml(js_manifest) + "\n"

  if cm_manifest is not None:
    cm_yaml = _manifest_to_yaml(cm_manifest) + "\n"
    bundle = f"{cm_yaml}---\n{jobset_yaml}"
  else:
    bundle = jobset_yaml
  return True, "", bundle


def _write_workload_yaml_bundle(
    cfg: dict[str, Any], jobset_yaml_override: str | None = None
) -> tuple[bool, str, str]:
  """Writes the deployable YAML bundle to JOBSET_YAML_PATH with 0o600 permissions."""
  for key in ("cluster_context", "namespace", "jobset_name"):
    val = cfg.get(key, "")
    if val:
      try:
        _validate_k8s_id(val, key)
      except ValueError as e:
        return False, str(e), ""
  for opt_key in ("kueue_queue_name", "cpu_nodepool"):
    opt_val = cfg.get(opt_key, "")
    if opt_val:
      try:
        _validate_k8s_id(opt_val, opt_key)
      except ValueError as e:
        return False, str(e), ""

  ok, err, bundle = _build_yaml_bundle(
      cfg, jobset_yaml_override=jobset_yaml_override
  )
  if not ok:
    return False, err, ""

  fd = os.open(JOBSET_YAML_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
  with os.fdopen(fd, "w", encoding="utf-8") as f:
    f.write(bundle)

  apply_cmd = _build_kubectl_apply_cmd(cfg)
  msg = (
      f"Wrote YAML to {JOBSET_YAML_PATH}. Run '{apply_cmd}' in your terminal to"
      " launch."
  )
  return True, msg, bundle


class DemoRequestHandler(server.SimpleHTTPRequestHandler):
  """HTTP request handler supporting static file serving and JSON/SSE API routes."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, directory=BASE_DIR, **kwargs)

  def log_message(self, format: str, *args: Any) -> None:
    is_error_code = (
        len(args) >= 2 and str(args[1]).isdigit() and int(args[1]) >= 400
    )
    if "/api/" in self.path or is_error_code:
      super().log_message(format, *args)

  def _serve_demo_html(self) -> None:
    """Serves gke_elasticity_demo.html with the per-session CSRF token injected."""
    html_path = os.path.join(BASE_DIR, "gke_elasticity_demo.html")
    try:
      with open(html_path, "r", encoding="utf-8") as f:
        html_text = f.read()
    except OSError:
      self.send_error(404, "Demo HTML file not found")
      return
    token_script = (
        "<script>window.__DEMO_SESSION_TOKEN__ ="
        f" {json.dumps(_SESSION_TOKEN)};</script>\n</head>"
    )
    if "</head>" in html_text:
      html_text = html_text.replace("</head>", token_script, 1)
    body = html_text.encode("utf-8")
    self.send_response(200)
    self.send_header("Content-Type", "text/html; charset=utf-8")
    self.send_header("Content-Length", str(len(body)))
    self.send_header("Cache-Control", "no-store")
    self.end_headers()
    self.wfile.write(body)

  def do_GET(self) -> None:
    host_hdr = self.headers.get("Host", "")
    if not _is_valid_host_header(host_hdr):
      self._send_json({"error": "forbidden host header"}, status=403)
      return

    parsed = urllib.parse.urlparse(self.path)
    if parsed.path == "/api/cluster/status":
      self._send_json(state.get_snapshot())
    elif parsed.path == "/api/config":
      cfg = state.get_config()
      _, _, bundle = _build_yaml_bundle(cfg)
      self._send_json({
          "config": cfg,
          "available_contexts": _get_kubeconfig_contexts(),
          "yaml_preview": _build_yaml_preview(cfg),
          "yaml_bundle": bundle,
          "yaml_path": JOBSET_YAML_PATH,
          "kubectl_apply_cmd": _build_kubectl_apply_cmd(cfg),
          "snapshot": state.get_snapshot(),
      })
    elif parsed.path == "/api/telemetry/stream":
      self._handle_sse()
    elif parsed.path in ("/", "/demo", "/gke", "/gke_elasticity_demo.html"):
      self._serve_demo_html()
    else:
      super().do_GET()

  def do_POST(self) -> None:  # pylint: disable=invalid-name
    """Routes HTTP POST requests to configuration, lifecycle, and signal handlers."""
    host_hdr = self.headers.get("Host", "")
    origin_hdr = self.headers.get("Origin", "")
    token_hdr = self.headers.get("X-Demo-Token", "")
    if not _is_valid_host_header(host_hdr):
      self._send_json({"error": "forbidden host header"}, status=403)
      return
    if not _is_valid_origin_header(origin_hdr, host_hdr):
      self._send_json({"error": "forbidden cross-origin request"}, status=403)
      return
    if not _is_valid_session_token(token_hdr):
      self._send_json(
          {"error": "missing or invalid X-Demo-Token header"}, status=403
      )
      return

    parsed = urllib.parse.urlparse(self.path)
    content_len = int(self.headers.get("Content-Length", 0))
    body = (
        self.rfile.read(content_len).decode("utf-8")
        if content_len > 0
        else "{}"
    )
    try:
      payload = json.loads(body)
    except json.JSONDecodeError:
      payload = {}

    if parsed.path == "/api/config":
      self._handle_update_config(payload)
    elif parsed.path == "/api/cluster/connect":
      self._handle_cluster_connect(payload)
    elif parsed.path == "/api/jobset/preview":
      self._handle_jobset_preview(payload)
    elif parsed.path == "/api/jobset/delete":
      self._handle_delete()
    elif parsed.path == "/api/signal/start":
      self._handle_start(payload)
    elif parsed.path == "/api/signal/pause":
      self._handle_pause()
    elif parsed.path == "/api/signal/reset":
      self._handle_reset()
    elif parsed.path == "/api/signal/interrupt":
      self._handle_interrupt(payload)
    elif parsed.path == "/api/signal/reconnect":
      self._handle_reconnect(payload)
    else:
      self._send_json({"error": "endpoint not found"}, status=404)

  def _send_json(self, data: Any, status: int = 200) -> None:
    body = json.dumps(data).encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.send_header("Cache-Control", "no-store")
    self.end_headers()
    self.wfile.write(body)

  def _handle_sse(self) -> None:
    """Streams Server-Sent Events (SSE) telemetry updates to the browser."""
    self.send_response(200)
    self.send_header("Content-Type", "text/event-stream")
    self.send_header("Cache-Control", "no-cache")
    self.send_header("Connection", "keep-alive")
    self.end_headers()

    client_queue = queue.Queue(maxsize=1000)
    state.register_subscriber(client_queue)

    try:
      init_evt = {"type": "init", "snapshot": state.get_snapshot()}
      self.wfile.write(f"data: {json.dumps(init_evt)}\n\n".encode("utf-8"))
      self.wfile.flush()

      while not state.stopped:
        try:
          event = client_queue.get(timeout=2.0)
          self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
          self.wfile.flush()
        except queue.Empty:
          self.wfile.write(b": ping\n\n")
          self.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
      pass
    finally:
      state.unregister_subscriber(client_queue)

  def _handle_update_config(self, payload: dict[str, Any]) -> None:
    """Updates bridge configuration and writes the YAML bundle to /tmp."""
    prev_cfg = state.get_config()
    if payload.get("reset_defaults"):
      with state.lock:
        state.config = default_config()
      updated_cfg = state.update_config({})
    else:
      updated_cfg = state.update_config(payload)

    if (
        prev_cfg["cluster_context"] != updated_cfg["cluster_context"]
        or prev_cfg["namespace"] != updated_cfg["namespace"]
        or prev_cfg["jobset_name"] != updated_cfg["jobset_name"]
    ):
      if tailer_instance is not None:
        tailer_instance.stop_current_tail()
        tailer_instance.finished_pods.clear()

    yaml_manifest = payload.get("yaml_manifest")
    ok, msg, bundle = _write_workload_yaml_bundle(
        updated_cfg,
        jobset_yaml_override=yaml_manifest
        if isinstance(yaml_manifest, str)
        else None,
    )
    if not ok:
      self._send_json(
          {
              "status": "error",
              "message": msg,
              "config": updated_cfg,
              "yaml_preview": _build_yaml_preview(updated_cfg),
              "yaml_path": JOBSET_YAML_PATH,
              "kubectl_apply_cmd": _build_kubectl_apply_cmd(updated_cfg),
              "snapshot": state.get_snapshot(),
          },
          400,
      )
      return

    state.broadcast({
        "type": "config_updated",
        "config": updated_cfg,
        "snapshot": state.get_snapshot(),
    })
    self._send_json({
        "status": "ok",
        "message": msg,
        "config": updated_cfg,
        "yaml_preview": _build_yaml_preview(updated_cfg),
        "yaml_bundle": bundle,
        "yaml_path": JOBSET_YAML_PATH,
        "kubectl_apply_cmd": _build_kubectl_apply_cmd(updated_cfg),
        "snapshot": state.get_snapshot(),
    })

  def _handle_cluster_connect(self, payload: dict[str, Any]) -> None:
    """Verifies connectivity to the GKE cluster and inspects TPU and CPU nodes."""
    prev_cfg = state.get_config()
    updated_cfg = state.update_config(payload)
    if (
        prev_cfg["cluster_context"] != updated_cfg["cluster_context"]
        or prev_cfg["namespace"] != updated_cfg["namespace"]
    ):
      if tailer_instance is not None:
        tailer_instance.stop_current_tail()
        tailer_instance.finished_pods.clear()

    ctx = updated_cfg["cluster_context"]
    ns = updated_cfg["namespace"]
    try:
      cmd = _kubectl_cmd(
          "get", "nodes", "-o", "json", context=ctx, include_ns=False
      )
    except ValueError as e:
      self._send_json(
          {
              "status": "error",
              "connected": False,
              "message": str(e),
              "config": updated_cfg,
          },
          400,
      )
      return

    try:
      proc = subprocess.run(
          cmd, capture_output=True, text=True, check=False, timeout=12
      )
    except subprocess.TimeoutExpired:
      self._send_json(
          {
              "status": "error",
              "connected": False,
              "message": f"Timed out connecting to cluster context '{ctx}'.",
              "config": updated_cfg,
          },
          400,
      )
      return

    if proc.returncode != 0:
      err_msg = proc.stderr.strip() or proc.stdout.strip() or "kubectl failed."
      self._send_json(
          {
              "status": "error",
              "connected": False,
              "message": err_msg,
              "config": updated_cfg,
          },
          400,
      )
      return

    try:
      nodes_data = json.loads(proc.stdout)
    except json.JSONDecodeError:
      nodes_data = {}

    items = nodes_data.get("items", [])
    total_nodes = len(items)
    tpu_nodes = 0
    accelerators: set[str] = set()
    topologies: set[str] = set()
    nodepools: set[str] = set()
    for node in items:
      labels = node.get("metadata", {}).get("labels", {})
      np = labels.get("cloud.google.com/gke-nodepool")
      if np:
        nodepools.add(np)
      accel = labels.get("cloud.google.com/gke-tpu-accelerator")
      if accel:
        tpu_nodes += 1
        accelerators.add(accel)
      topo = labels.get("cloud.google.com/gke-tpu-topology")
      if topo:
        topologies.add(topo)

    cpu_nodes = total_nodes - tpu_nodes
    _, _, bundle = _write_workload_yaml_bundle(updated_cfg)
    state.broadcast({
        "type": "config_updated",
        "config": updated_cfg,
        "snapshot": state.get_snapshot(),
    })
    self._send_json({
        "status": "ok",
        "connected": True,
        "cluster_context": ctx,
        "namespace": ns,
        "total_nodes": total_nodes,
        "tpu_nodes": tpu_nodes,
        "cpu_nodes": cpu_nodes,
        "accelerators": sorted(accelerators),
        "topologies": sorted(topologies),
        "nodepools": sorted(nodepools),
        "available_contexts": _get_kubeconfig_contexts(),
        "config": updated_cfg,
        "yaml_preview": _build_yaml_preview(updated_cfg),
        "yaml_bundle": bundle,
        "yaml_path": JOBSET_YAML_PATH,
        "kubectl_apply_cmd": _build_kubectl_apply_cmd(updated_cfg),
        "snapshot": state.get_snapshot(),
        "message": (
            f"Connected to {ctx}: {total_nodes} nodes ({tpu_nodes} TPU,"
            f" {cpu_nodes} CPU)."
        ),
    })

  def _handle_jobset_preview(self, payload: dict[str, Any]) -> None:
    """Generates a YAML preview from form fields without mutating bridge state."""
    preview_cfg = _merge_config_dict(state.get_config(), payload)
    yaml_str = _build_yaml_preview(preview_cfg)
    yaml_override = payload.get("yaml_manifest")
    ok, err, bundle = _build_yaml_bundle(
        preview_cfg,
        jobset_yaml_override=yaml_override
        if isinstance(yaml_override, str)
        else None,
    )
    if not ok:
      self._send_json({"status": "error", "message": err}, status=400)
      return
    self._send_json({
        "status": "ok",
        "yaml": yaml_str,
        "yaml_bundle": bundle,
        "yaml_path": JOBSET_YAML_PATH,
        "kubectl_apply_cmd": _build_kubectl_apply_cmd(preview_cfg),
        "jobset_name": preview_cfg["jobset_name"],
        "chips_per_slice": preview_cfg["chips_per_slice"],
        "vms_per_slice": preview_cfg["vms_per_slice"],
    })

  def _handle_delete(self) -> None:
    """Deletes the active JobSet from GKE and clears bridge state."""
    cfg = state.get_config()
    ctx = cfg["cluster_context"]
    ns = cfg["namespace"]
    js_name = cfg["jobset_name"]
    try:
      safe_js = _validate_k8s_id(js_name, "jobset_name")
    except ValueError as e:
      self._send_json({"status": "error", "message": str(e)}, status=400)
      return
    _logger.info("Deleting JobSet %s (context=%s, ns=%s)...", safe_js, ctx, ns)
    if tailer_instance is not None:
      tailer_instance.stop_current_tail()
      tailer_instance.finished_pods.clear()
    proc = subprocess.run(
        _kubectl_cmd(
            "delete",
            "jobset",
            safe_js,
            "--ignore-not-found=true",
            context=ctx,
            namespace=ns,
            include_ns=True,
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    state.clear_for_delete()
    state.broadcast({"type": "deleted", "snapshot": state.get_snapshot()})
    self._send_json({
        "status": "ok",
        "message": f"JobSet '{safe_js}' deleted and state cleared",
        "output": proc.stdout.strip(),
        "snapshot": state.get_snapshot(),
    })

  def _handle_start(self, payload: dict[str, Any]) -> None:
    """Starts or resumes training on the head pod via port 8090 RPC."""
    del payload
    with state.lock:
      head_pod = state.head_pod_name
      head_phase = state.head_pod_phase
      js_status = state.jobset_status
      cfg = json.loads(json.dumps(state.config))
      ctx = cfg["cluster_context"]
      ns = cfg["namespace"]

    if (
        not head_pod
        or js_status in ("IDLE", "FAILED", "COMPLETED")
        or head_phase in ("Failed", "Succeeded", "Unknown")
    ):
      apply_cmd = _build_kubectl_apply_cmd(cfg)
      self._send_json(
          {
              "status": "error",
              "action": "missing_jobset",
              "message": (
                  f"No active JobSet found on GKE. Run '{apply_cmd}' in your"
                  " terminal first."
              ),
              "kubectl_apply_cmd": apply_cmd,
              "snapshot": state.get_snapshot(),
          },
          400,
      )
      return

    ok, out = _send_workload_control_rpc(
        head_pod, "/start", context=ctx, namespace=ns
    )
    if ok:
      with state.lock:
        state.training_running = True
        if state.jobset_status in ("READY", "PAUSED"):
          state.jobset_status = "RUNNING"
      self._send_json({
          "status": "ok",
          "action": "start",
          "output": out,
          "snapshot": state.get_snapshot(),
      })
    else:
      with state.lock:
        state.auto_start_when_ready = True
      self._send_json({
          "status": "ok",
          "action": "queued_start",
          "message": (
              "Pod is still initializing; Start Training signal queued for when"
              " pre-compilation finishes."
          ),
          "snapshot": state.get_snapshot(),
      })

  def _handle_pause(self) -> None:
    """Pauses training on the head pod via control RPC."""
    with state.lock:
      head_pod = state.head_pod_name
      ctx = state.config["cluster_context"]
      ns = state.config["namespace"]
      state.auto_start_when_ready = False
      state.training_running = False

    if head_pod:
      ok, out = _send_workload_control_rpc(
          head_pod, "/pause", context=ctx, namespace=ns
      )
      self._send_json({
          "status": "ok" if ok else "queued",
          "action": "pause",
          "output": out,
          "snapshot": state.get_snapshot(),
      })
    else:
      self._send_json({
          "status": "ok",
          "action": "pause",
          "snapshot": state.get_snapshot(),
      })

  def _handle_reset(self) -> None:
    """Resets workload step counter and bridge telemetry to step zero."""
    with state.lock:
      head_pod = state.head_pod_name
      ctx = state.config["cluster_context"]
      ns = state.config["namespace"]
      need_pod_reset = state.training_running or state.current_step > 0
    state.reset_telemetry()
    if head_pod and need_pod_reset:
      _send_workload_control_rpc(head_pod, "/reset", context=ctx, namespace=ns)
    state.broadcast({"type": "reset", "snapshot": state.get_snapshot()})
    self._send_json({
        "status": "ok",
        "action": "reset",
        "message": "Demo telemetry and workload reset to Step 0.",
        "snapshot": state.get_snapshot(),
    })

  def _handle_interrupt(self, payload: dict[str, Any]) -> None:
    """Disrupts a slice via Kubernetes node drain or in-workload RPC."""
    mechanism = payload.get("mechanism", "k8s")
    slice_idx = int(payload.get("slice", 1))
    with state.lock:
      ctx = state.config["cluster_context"]
      ns = state.config["namespace"]
      js_name = state.config["jobset_name"]

    if mechanism == "k8s":
      worker_pod = ""
      leader_prefix = f"{js_name}-worker-{slice_idx}-0-"
      for w in state.worker_pods:
        wname = w.get("name", "")
        if (
            w.get("slice") == slice_idx
            and w.get("node")
            and not wname.startswith(leader_prefix)
        ):
          worker_pod = wname
          break
      if not worker_pod:
        for w in state.worker_pods:
          if w.get("slice") == slice_idx and w.get("node"):
            worker_pod = w.get("name", "")
            break

      if not worker_pod:
        cmd = _kubectl_cmd(
            "get",
            "pods",
            "-l",
            (
                f"jobset.sigs.k8s.io/jobset-name={js_name},"
                f"jobset.sigs.k8s.io/job-index={slice_idx}"
            ),
            "-o",
            "jsonpath={.items[*].metadata.name}",
            context=ctx,
            namespace=ns,
            include_ns=True,
        )
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        candidates = proc.stdout.strip().split()
        for cand in candidates:
          if not cand.startswith(leader_prefix):
            worker_pod = cand
            break
        if not worker_pod and candidates:
          worker_pod = candidates[0]

      if not worker_pod:
        self._send_json(
            {
                "status": "error",
                "message": f"Worker pod for slice {slice_idx} not found.",
            },
            404,
        )
        return

      try:
        safe_worker_pod = _validate_k8s_id(worker_pod, "worker_pod")
      except ValueError as e:
        self._send_json({"status": "error", "message": str(e)}, status=400)
        return

      node_cmd = _kubectl_cmd(
          "get",
          "pod",
          safe_worker_pod,
          "-o",
          "jsonpath={.spec.nodeName}",
          context=ctx,
          namespace=ns,
          include_ns=True,
      )
      proc_node = subprocess.run(
          node_cmd, capture_output=True, text=True, check=False
      )
      node_name = proc_node.stdout.strip()
      if not node_name:
        self._send_json(
            {
                "status": "error",
                "message": (
                    f"Could not determine node name for pod {safe_worker_pod}."
                ),
            },
            500,
        )
        return
      try:
        safe_node_name = _validate_k8s_id(node_name, "node_name")
      except ValueError as e:
        self._send_json({"status": "error", "message": str(e)}, status=400)
        return

      _logger.info(
          "Injecting K8s preemption on %s by cordoning & draining node %s...",
          safe_worker_pod,
          safe_node_name,
      )
      with state.lock:
        state.cordoned_nodes[slice_idx] = safe_node_name
        state.jobset_status = "DISRUPTED"
        state.active_slices = [
            i for i in range(state.total_slices) if i != slice_idx
        ]
        state.inactive_slices = [slice_idx]
        state.last_step_wall_time = None
        state.current_event = (
            f"Cordoning & draining node {safe_node_name} (Slice {slice_idx})..."
        )

      state.broadcast({
          "type": "event",
          "event": "disrupted",
          "slice": slice_idx,
          "step": state.current_step,
          "node": safe_node_name,
          "target_pod": safe_worker_pod,
          "message": (
              f"kubectl cordon + drain initiated on {safe_node_name}"
              f" ({safe_worker_pod})"
          ),
          "timestamp": time.time(),
      })

      subprocess.run(
          _kubectl_cmd(
              "cordon", safe_node_name, context=ctx, include_ns=False
          ),
          capture_output=True,
          text=True,
          check=False,
      )
      drain_cmd = _kubectl_cmd(
          "drain",
          safe_node_name,
          "--ignore-daemonsets",
          "--delete-emptydir-data",
          "--force",
          "--grace-period=0",
          context=ctx,
          include_ns=False,
      )
      drain_res = subprocess.run(
          drain_cmd, capture_output=True, text=True, check=False
      )
      _logger.info(
          "Node %s drained with exit code %d: %s",
          safe_node_name,
          drain_res.returncode,
          drain_res.stdout.strip(),
      )

      self._send_json({
          "status": "ok",
          "mechanism": "k8s",
          "action": "cordon_drain",
          "node": safe_node_name,
          "target_pod": safe_worker_pod,
          "slice": slice_idx,
      })

    else:
      head_pod = state.head_pod_name
      if not head_pod:
        self._send_json(
            {"status": "error", "message": "Head pod not found."}, 404
        )
        return

      with state.lock:
        state.jobset_status = "DISRUPTED"
        state.active_slices = [
            i for i in range(state.total_slices) if i != slice_idx
        ]
        state.inactive_slices = [slice_idx]
        state.last_step_wall_time = None
        state.current_event = (
            f"Sending in-workload interrupt signal to Slice {slice_idx}..."
        )

      _logger.info(
          "Sending in-workload disrupt signal for slice %d to %s...",
          slice_idx,
          head_pod,
      )
      ok, out = _send_workload_control_rpc(
          head_pod, f"/disrupt?slice={slice_idx}", context=ctx, namespace=ns
      )
      _logger.info("Workload signal response (%s): %s", ok, out)
      self._send_json({
          "status": "ok" if ok else "error",
          "mechanism": "workload",
          "action": "disrupt",
          "output": out,
          "slice": slice_idx,
      })

  def _handle_reconnect(self, payload: dict[str, Any]) -> None:
    """Reconnects a disrupted slice via Kubernetes uncordon or in-workload RPC."""
    mechanism = payload.get("mechanism", "k8s")
    slice_idx = int(payload.get("slice", 1))
    with state.lock:
      ctx = state.config["cluster_context"]
      ns = state.config["namespace"]

    if mechanism == "k8s":
      with state.lock:
        node_name = state.cordoned_nodes.pop(slice_idx, "")
        state.jobset_status = "SCALING_UP"

      cmd_nodes = _kubectl_cmd(
          "get",
          "nodes",
          "--selector=cloud.google.com/gke-tpu-accelerator",
          "-o",
          "jsonpath={.items[?(@.spec.unschedulable==true)].metadata.name}",
          context=ctx,
          include_ns=False,
      )
      proc_nodes = subprocess.run(
          cmd_nodes, capture_output=True, text=True, check=False
      )
      unsched = proc_nodes.stdout.strip().split()
      if not node_name and unsched:
        node_name = unsched[0]

      if not node_name and not unsched:
        self._send_json(
            {
                "status": "error",
                "message": f"No cordoned node found for slice {slice_idx}.",
            },
            400,
        )
        return

      _logger.info(
          "Reconnecting slice %d by uncordoning node %s...",
          slice_idx,
          node_name,
      )
      _uncordon_all_tpu_nodes(state.cordoned_nodes, context=ctx)

      state.broadcast({
          "type": "event",
          "event": "reconnecting",
          "slice": slice_idx,
          "node": node_name,
          "message": (
              f"kubectl uncordon {node_name}: Pending worker pod scheduling to"
              " rejoin mesh..."
          ),
          "timestamp": time.time(),
      })

      self._send_json({
          "status": "ok",
          "mechanism": "k8s",
          "action": "uncordon",
          "node": node_name,
          "slice": slice_idx,
          "message": (
              f"Node {node_name} uncordoned. Pending worker pod will schedule"
              " and rejoin mesh."
          ),
      })

    else:
      head_pod = state.head_pod_name
      if not head_pod:
        self._send_json(
            {"status": "error", "message": "Head pod not found."}, 404
        )
        return

      with state.lock:
        state.jobset_status = "SCALING_UP"
        state.current_event = (
            f"Sending in-workload reconnect signal to Slice {slice_idx}..."
        )

      _logger.info(
          "Sending in-workload reconnect signal for slice %d to %s...",
          slice_idx,
          head_pod,
      )
      ok, out = _send_workload_control_rpc(
          head_pod, f"/reconnect?slice={slice_idx}", context=ctx, namespace=ns
      )
      _logger.info("Workload signal response (%s): %s", ok, out)
      self._send_json({
          "status": "ok" if ok else "error",
          "mechanism": "workload",
          "action": "reconnect",
          "output": out,
          "slice": slice_idx,
      })


class ThreadingServer(socketserver.ThreadingMixIn, server.HTTPServer):
  daemon_threads = True
  allow_reuse_address = True


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  """Parses CLI flags for starting the GKE elasticity demo server."""
  parser = argparse.ArgumentParser(
      description=(
          "Start the Pathways Elasticity GKE web server and open the demo UI."
      )
  )
  parser.add_argument(
      "--port",
      type=int,
      default=PORT,
      help=f"Port to run the HTTP server on (default: {PORT}).",
  )
  parser.add_argument(
      "--host",
      type=str,
      default="127.0.0.1",
      help="Host interface to bind the HTTP server to (default: 127.0.0.1).",
  )
  parser.add_argument(
      "--no-browser",
      action="store_true",
      help="Do not automatically open the demo webpage in a browser.",
  )
  parser.add_argument(
      "--context",
      type=str,
      default="",
      help="Optional GKE kubeconfig context to select on startup.",
  )
  parser.add_argument(
      "--namespace",
      type=str,
      default="",
      help="Optional Kubernetes namespace to select on startup.",
  )
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
  global tailer_instance
  args = _parse_args(argv)
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
  )
  _logger.info("Initializing GKE Demo Bridge Server...")

  fqdn = (socket.getfqdn() or "").lower()
  short_host = (socket.gethostname() or "").lower()
  if fqdn:
    _ALLOWED_HOSTS.add(fqdn)
  if short_host:
    _ALLOWED_HOSTS.add(short_host)
  if args.host and args.host != "0.0.0.0":
    _ALLOWED_HOSTS.add(args.host.lower())

  startup_updates: dict[str, Any] = {}
  if args.context:
    startup_updates["cluster_context"] = args.context
  if args.namespace:
    startup_updates["namespace"] = args.namespace
  if startup_updates:
    state.update_config(startup_updates)

  ok, msg, _ = _write_workload_yaml_bundle(state.get_config())
  if ok:
    _logger.info(msg)
  else:
    _logger.warning("Could not write initial YAML bundle: %s", msg)

  poller = ClusterPoller(state)
  poller.start()

  tailer_instance = LogTailer(state)
  tailer_instance.start()

  httpd = ThreadingServer((args.host, args.port), DemoRequestHandler)
  actual_port = httpd.server_address[1]
  local_url = f"http://localhost:{actual_port}/"
  remote_url = f"http://{fqdn or 'localhost'}:{actual_port}/"

  _logger.info("=" * 60)
  _logger.info("GKE Demo Bridge Server running on %s", local_url)
  if fqdn and fqdn != "localhost":
    _logger.info("Remote / Cloudtop URL: %s", remote_url)
  _logger.info("=" * 60)

  if not args.no_browser:
    threading.Timer(0.4, lambda: webbrowser.open(local_url)).start()

  try:
    httpd.serve_forever()
  except KeyboardInterrupt:
    _logger.info("Shutting down...")
  finally:
    state.stopped = True
    httpd.server_close()


if __name__ == "__main__":
  main()
