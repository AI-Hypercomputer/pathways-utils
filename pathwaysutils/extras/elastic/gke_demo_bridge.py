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
- CLI helpers to build the workload image, render ConfigMap + JobSet YAML,
  apply a manifest, or delete an active deployment
- Real-time Server-Sent Events (SSE) telemetry stream parsing JAX driver logs
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

from kubernetes import client as k8s_client
import yaml

from pathwaysutils.experimental.gke import jobset as pw_jobset

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
_ORBAX_SNAPSHOTTER_PY_PATH = os.path.normpath(
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
SNAPSHOTTER_PY_PATH = (
    _ORBAX_SNAPSHOTTER_PY_PATH
    if os.path.isfile(_ORBAX_SNAPSHOTTER_PY_PATH)
    else os.path.join(BASE_DIR, "snapshotter.py")
)
JOBSET_YAML_PATH = os.path.join(
    tempfile.gettempdir(), "gke_elastic_workload.yaml"
)
REPO_ROOT = os.path.normpath(os.path.join(PATHWAYSUTILS_DIR, ".."))
DOCKERFILE_PATH = os.path.join(BASE_DIR, "Dockerfile")
CONFIGMAP_NAME = "elastic-workload-code"
PORT = 48259

_SESSION_TOKEN = secrets.token_urlsafe(32)
_ALLOW_ANY_HOST = False
_ALLOWED_HOSTS: set[str] = {
    "localhost",
    "127.0.0.1",
    "::1",
    "[::1]",
}
_SAFE_K8S_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:@/-]{0,252}$")
_SAFE_FILENAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{1,128}$")
_SAFE_CONTROL_PATH_RE = re.compile(
    r"^/(start|pause|reset|status|(disrupt|reconnect)\?slice=\d+|mode\?mode=[a-z_]+)$"
)

DEFAULT_NAMESPACE = "default"
DEFAULT_JOBSET_NAME = "elastic-demo"
DEFAULT_KUEUE_QUEUE = ""
DEFAULT_CPU_NODEPOOL = ""
DEFAULT_CONTAINER_IMAGE = ""
DEFAULT_SERVER_IMAGE = (
    "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:latest"
)
DEFAULT_PROXY_IMAGE = (
    "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:latest"
)
DEFAULT_GCS_SCRATCH = ""
DEFAULT_ORBAX_REF = "main"
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
    "tpuv4": "tpu-v4-podslice",
    "v4": "tpu-v4-podslice",
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
    "tpuv4": "tpuv4",
    "v4": "tpuv4",
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


def _current_kubeconfig_context() -> str:
  """Returns the active kubeconfig context, or an empty string if none is set."""
  proc = subprocess.run(
      ["kubectl", "config", "current-context"],
      capture_output=True,
      text=True,
      check=False,
  )
  if proc.returncode != 0:
    return ""
  return proc.stdout.strip()


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
      "cluster_context": _current_kubeconfig_context(),
      "namespace": DEFAULT_NAMESPACE,
      "jobset_name": DEFAULT_JOBSET_NAME,
      "kueue_queue_name": DEFAULT_KUEUE_QUEUE,
      "num_slices": 2,
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
      "cpu_nodepool": DEFAULT_CPU_NODEPOOL,
      "orbax_ref": DEFAULT_ORBAX_REF,
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
  if _ALLOW_ANY_HOST:
    return True
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
  if _ALLOW_ANY_HOST:
    return bool(origin_host) and origin_host == target_host
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
    return []
  return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


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


def _pod_requests_tpu(pod: dict[str, Any]) -> bool:
  """Returns True if any container in the pod requests google.com/tpu."""
  spec = pod.get("spec", {})
  for container in spec.get("containers", []) + spec.get("initContainers", []):
    limits = container.get("resources", {}).get("limits", {})
    if "google.com/tpu" in limits:
      return True
  return False


def _fetch_tpu_pools(
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
  """Discovers TPU node pools matching the configured slice shape and who uses them.

  Each entry is `{"name", "nodes", "state", "slice", "owner"}` where `state`
  is one of `active` (hosts one of this JobSet's worker slices), `busy`
  (hosts TPU pods from some other workload), `cordoned` (every node is
  unschedulable), or `standby` (idle and schedulable).
  """
  ctx = cfg["cluster_context"]
  ns = cfg["namespace"]
  js_name = cfg["jobset_name"]
  topo = compute_topology_specs(cfg["tpu_type"], cfg["topology"])
  selector = (
      f"cloud.google.com/gke-tpu-accelerator={topo['accelerator']},"
      f"cloud.google.com/gke-tpu-topology={topo['topology']}"
  )
  nodes_proc = subprocess.run(
      _kubectl_cmd(
          "get", "nodes", "-l", selector, "-o", "json",
          context=ctx, include_ns=False,
      ),
      capture_output=True,
      text=True,
      check=False,
  )
  if nodes_proc.returncode != 0:
    return []
  pools: dict[str, dict[str, Any]] = {}
  node_to_pool: dict[str, str] = {}
  for node in json.loads(nodes_proc.stdout).get("items", []):
    meta = node.get("metadata", {})
    pool_name = meta.get("labels", {}).get("cloud.google.com/gke-nodepool", "")
    if not pool_name:
      continue
    pool = pools.setdefault(
        pool_name,
        {
            "name": pool_name,
            "nodes": 0,
            "cordoned_nodes": 0,
            "state": "standby",
            "slice": None,
            "owner": "",
        },
    )
    pool["nodes"] += 1
    if node.get("spec", {}).get("unschedulable"):
      pool["cordoned_nodes"] += 1
    node_to_pool[meta.get("name", "")] = pool_name
  if not pools:
    return []

  pods_proc = subprocess.run(
      _kubectl_cmd(
          "get",
          "pods",
          "--all-namespaces",
          "--field-selector=status.phase!=Succeeded,status.phase!=Failed",
          "-o",
          "json",
          context=ctx,
          include_ns=False,
      ),
      capture_output=True,
      text=True,
      check=False,
  )
  if pods_proc.returncode == 0:
    for pod in json.loads(pods_proc.stdout).get("items", []):
      node_name = pod.get("spec", {}).get("nodeName", "")
      pool_name = node_to_pool.get(node_name, "")
      if not pool_name or not _pod_requests_tpu(pod):
        continue
      meta = pod.get("metadata", {})
      labels = meta.get("labels", {})
      pod_ns = meta.get("namespace", "")
      pod_jobset = labels.get("jobset.sigs.k8s.io/jobset-name", "")
      pool = pools[pool_name]
      if pod_ns == ns and pod_jobset == js_name:
        job_idx = labels.get("jobset.sigs.k8s.io/job-index", "")
        pool["state"] = "active"
        pool["slice"] = int(job_idx) if job_idx.isdigit() else None
        pool["owner"] = f"{ns}/{js_name}"
      elif pool["state"] != "active":
        pool["state"] = "busy"
        pool["owner"] = f"{pod_ns}/{pod_jobset or meta.get('name', '')}"

  for pool in pools.values():
    if pool["state"] == "standby" and pool["cordoned_nodes"] == pool["nodes"]:
      pool["state"] = "cordoned"
  return sorted(pools.values(), key=lambda p: p["name"])


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
      "orbax_ref",
      "workload_preset",
      "custom_command",
  ):
    if key in updates and updates[key] is not None:
      cfg[key] = str(updates[key]).strip()

  for int_key, min_val, max_val in (
      ("num_slices", 1, 16),
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
    self.tpu_pools: list[dict[str, Any]] = []
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
          "standby_slices": sum(
              1 for p in self.tpu_pools if p["state"] == "standby"
          ),
          "tpu_pools": json.loads(json.dumps(self.tpu_pools)),
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
    self._cycle = 0

  def run(self) -> None:
    _logger.info("Cluster poller started.")
    while not self.bridge_state.stopped:
      try:
        self._poll_cluster()
        if self._cycle % 4 == 0:
          self._poll_tpu_pools()
        self._tick_recovery_goodput()
      except (OSError, subprocess.SubprocessError, ValueError, KeyError) as e:
        _logger.debug("Cluster poll error: %s", e)
      self._cycle += 1
      time.sleep(1.5)

  def _poll_tpu_pools(self) -> None:
    """Refreshes TPU node pool occupancy and broadcasts it when it changes."""
    pools = _fetch_tpu_pools(self.bridge_state.get_config())
    with self.bridge_state.lock:
      changed = pools != self.bridge_state.tpu_pools
      self.bridge_state.tpu_pools = pools
    if changed:
      self.bridge_state.broadcast({
          "type": "tpu_pools",
          "tpu_pools": pools,
          "snapshot": self.bridge_state.get_snapshot(),
      })

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
      if meta.get("deletionTimestamp"):
        continue
      pname = meta.get("name", "")
      labels = meta.get("labels", {})
      rep_job = labels.get("jobset.sigs.k8s.io/replicatedjob-name", "")
      job_idx = labels.get("jobset.sigs.k8s.io/job-index", "")
      status = item.get("status", {})
      phase = status.get("phase", "Unknown")

      if rep_job == "pathways-head":
        for cs in status.get("containerStatuses", []):
          if cs.get("name") == "jax-tpu":
            terminated = cs.get("state", {}).get("terminated")
            if terminated is not None:
              phase = (
                  "Succeeded"
                  if terminated.get("exitCode", 0) == 0
                  else "Failed"
              )
        head_name = pname
        head_phase = phase
      elif rep_job == pw_jobset.PATHWAYS_WORKER_JOB_NAME:
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
    self.pod_line_counts: dict[str, int] = {}

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

    already_seen = self.pod_line_counts.get(pod_key, 0)
    line_idx = 0
    for line in iter(stdout.readline, ""):
      clean_line = line.rstrip()
      if clean_line:
        if clean_line.startswith("Error from server"):
          continue
        line_idx += 1
        if line_idx <= already_seen:
          continue
        self.pod_line_counts[pod_key] = line_idx
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

    rc = self.active_process.wait()
    self.active_process = None
    if (rc == 0 and line_idx > 0) or self.bridge_state.head_pod_phase in (
        "Succeeded",
        "Failed",
        "Completed",
    ):
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

    if " slices active. Wanting at least " in line:
      with self.bridge_state.lock:
        self.bridge_state.jobset_status = "RECOVERING"
        self.bridge_state.current_event = (
            "Paused in step_snapshot_pause_resume mode: waiting for all slices"
            " to reconnect..."
        )
      self.bridge_state.broadcast({
          "type": "event",
          "event": "stage1",
          "message": (
              "Paused in step_snapshot_pause_resume mode — waiting for all"
              " slices to reconnect before resuming."
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
    orbax_ref = cfg.get("orbax_ref", "") or DEFAULT_ORBAX_REF
    lines.extend([
        (
            'python3 -c "import grain" >/dev/null 2>&1'
            " || pip install --no-cache-dir grain"
        ),
        (
            'python3 -c "from'
            " orbax.checkpoint.experimental.v1._src.training.pathways import"
            ' snapshotter" >/dev/null 2>&1 || pip install --no-cache-dir'
            " --no-deps"
            f' "git+https://github.com/google/orbax.git@{orbax_ref}#subdirectory=checkpoint"'
        ),
    ])
    if "snapshotter.py" in file_names:
      lines.extend([
          (
              'ORBAX_PW_DIR=$(python3 -c "from'
              " orbax.checkpoint.experimental.v1._src.training.pathways import"
              ' snapshotter as s; import os; print(os.path.dirname(s.__file__))")'
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


GENERATOR_TPU_TYPE_MAP = {
    "tpuv5e": "v5e",
    "v5e": "v5e",
    "tpuv6e": "v6e",
    "v6e": "v6e",
    "tpuv5p": "v5p",
    "v5p": "v5p",
    "tpuv7x": "tpu7x",
    "tpu7x": "tpu7x",
    "v7x": "tpu7x",
    "tpuv4": "v4",
    "v4": "v4",
}


def _build_jobset_manifest(
    cfg: dict[str, Any], has_configmap: bool
) -> dict[str, Any]:
  """Builds the JobSet manifest with `PathwaysJobSet` from the current configuration.

  The generator owns the Pathways resource manager, proxy, and TPU worker
  templates (accelerator and topology selectors, chips per VM, parallelism);
  this function only adds the demo's head container, its ConfigMap volume, and
  the host networking the demo relies on.
  """
  topo_specs = compute_topology_specs(cfg["tpu_type"], cfg["topology"])
  num_slices = int(cfg["num_slices"])
  generator_tpu_type = GENERATOR_TPU_TYPE_MAP[cfg["tpu_type"].strip().lower()]

  head_volumes = [
      k8s_client.V1Volume(
          name="shared-tmp",
          host_path=k8s_client.V1HostPathVolumeSource(
              path="/tmp", type="DirectoryOrCreate"
          ),
      )
  ]
  head_mounts = [k8s_client.V1VolumeMount(name="shared-tmp", mount_path="/tmp")]
  if has_configmap:
    head_volumes.insert(
        0,
        k8s_client.V1Volume(
            name="workload-code",
            config_map=k8s_client.V1ConfigMapVolumeSource(name=CONFIGMAP_NAME),
        ),
    )
    head_mounts.insert(
        0,
        k8s_client.V1VolumeMount(
            name="workload-code", mount_path="/app/workload"
        ),
    )

  head_pod_spec_kwargs: dict[str, Any] = {
      "host_network": True,
      "dns_policy": "ClusterFirstWithHostNet",
      "volumes": head_volumes,
      "node_selector": (
          {pw_jobset.GKE_NODEPOOL_KEY: cfg["cpu_nodepool"]}
          if cfg.get("cpu_nodepool")
          else {}
      ),
  }
  labels = (
      {"kueue.x-k8s.io/queue-name": cfg["kueue_queue_name"]}
      if cfg.get("kueue_queue_name")
      else None
  )

  js = pw_jobset.PathwaysJobSet(
      name=cfg["jobset_name"],
      namespace=cfg["namespace"],
      pathways_dir=cfg["gcs_scratch_location"],
      tpu_type=generator_tpu_type,
      topology=topo_specs["topology"],
      num_slices=num_slices,
      elastic_slices=num_slices,
      labels=labels,
      pathways_rm_and_worker_image=cfg["server_image"],
      pathways_proxy_image=cfg["proxy_image"],
      head_pod_spec_kwargs=head_pod_spec_kwargs,
      worker_pod_spec_kwargs={
          "host_network": True,
          "dns_policy": "ClusterFirstWithHostNet",
      },
      proxy_container_kwargs={
          "args": ["--xla_tpu_use_enhanced_launch_barrier=true"],
          "resources": k8s_client.V1ResourceRequirements(
              requests={"cpu": "2", "memory": "8G"},
              limits={"cpu": "16", "memory": "100G"},
          ),
      },
      rm_container_kwargs={
          "args": ["--xla_tpu_use_enhanced_launch_barrier=true"],
          "resources": k8s_client.V1ResourceRequirements(
              requests={"cpu": "1", "memory": "4G"},
              limits={"cpu": "8", "memory": "32G"},
          ),
      },
      worker_container_kwargs={
          "args": ["--xla_tpu_use_enhanced_launch_barrier=true"],
      },
  )
  js.add_user_workload(
      image=cfg["container_image"],
      command=[
          "bash",
          "-c",
          _build_head_startup_script(cfg, topo_specs["instance_type"]),
      ],
      name="jax-tpu",
      env={
          "XCLOUD_ENVIRONMENT": "GCP",
          "TPU_VMODULE": "real_program_continuator=1",
          "ENABLE_PATHWAYS_PERSISTENCE": "1",
          "ENABLE_PERSISTENCE_API": "1",
          "ENABLE_PJRT_COMPATIBILITY": "true",
      },
      ports=[k8s_client.V1ContainerPort(container_port=8090, protocol="TCP")],
      resources=k8s_client.V1ResourceRequirements(
          requests={"cpu": "2", "memory": "8G"},
          limits={"cpu": "24", "memory": "100G"},
      ),
      security_context=k8s_client.V1SecurityContext(privileged=True),
      volume_mounts=head_mounts,
  )
  return js.to_dict()


class _LiteralDumper(yaml.SafeDumper):
  """YAML dumper that renders multi-line strings as literal blocks."""


def _represent_multiline_str(dumper: yaml.SafeDumper, data: str) -> Any:
  if "\n" in data:
    return dumper.represent_scalar(
        "tag:yaml.org,2002:str", data, style="|"
    )
  return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_LiteralDumper.add_representer(str, _represent_multiline_str)


def _manifest_to_yaml(obj: Any) -> str:
  """Serializes a manifest dict to YAML, keeping key order."""
  return yaml.dump(
      obj, Dumper=_LiteralDumper, sort_keys=False, default_flow_style=False
  ).rstrip("\n")


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
          "name": CONFIGMAP_NAME,
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


def _build_yaml_bundle(
    cfg: dict[str, Any], jobset_yaml_override: str | None = None
) -> tuple[bool, str, str]:
  """Builds the full deployable YAML bundle (ConfigMap + JobSet)."""
  tpu_key = cfg["tpu_type"].strip().lower()
  if tpu_key not in GENERATOR_TPU_TYPE_MAP:
    return (
        False,
        f"Unsupported TPU type {cfg['tpu_type']!r}; use one of"
        f" {sorted(set(GENERATOR_TPU_TYPE_MAP.values()))}.",
        "",
    )
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
  if not jobset_yaml_override:
    if not cfg.get("container_image"):
      return (
          False,
          "No workload container image set. Pass --image=... or"
          " --build-image, or fill in the image field in Step 2.",
          "",
      )
    if not cfg.get("gcs_scratch_location"):
      return (
          False,
          "No GCS scratch location set. Pass --gcs-scratch=gs://... or fill"
          " in the field in Step 2.",
          "",
      )
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


def _run_logged(cmd: list[str], cwd: str | None = None) -> int:
  """Runs a command with output streamed to the terminal and returns its exit code."""
  _logger.info("$ %s", " ".join(cmd))
  proc = subprocess.run(cmd, cwd=cwd, check=False)
  return proc.returncode


def _build_and_push_image(
    image: str, builder: str, project: str, orbax_ref: str = "main"
) -> bool:
  """Builds the demo workload image from the repository root and pushes it.

  The image bundles JAX, Flax, Grain, Orbax (from `--orbax-ref`), and this
  checkout of pathwaysutils so the head pod can run `elastic_workload.py`
  without any runtime pip installs.
  """
  dockerfile_rel = os.path.relpath(DOCKERFILE_PATH, REPO_ROOT)
  build_arg = f"ORBAX_REF={orbax_ref or 'main'}"
  if builder == "docker":
    rc = _run_logged([
        "docker",
        "build",
        "--build-arg",
        build_arg,
        "-f",
        DOCKERFILE_PATH,
        "-t",
        image,
        REPO_ROOT,
    ])
    if rc != 0:
      return False
    return _run_logged(["docker", "push", image]) == 0

  cloudbuild = {
      "steps": [{
          "name": "gcr.io/cloud-builders/docker",
          "args": [
              "build",
              "--build-arg",
              build_arg,
              "-f",
              dockerfile_rel,
              "-t",
              image,
              ".",
          ],
      }],
      "images": [image],
  }
  fd, cfg_path = tempfile.mkstemp(prefix="elastic_demo_cloudbuild_", suffix=".json")
  with os.fdopen(fd, "w", encoding="utf-8") as f:
    json.dump(cloudbuild, f)
  cmd = ["gcloud", "builds", "submit", REPO_ROOT, f"--config={cfg_path}"]
  if project:
    cmd.append(f"--project={project}")
  rc = _run_logged(cmd)
  os.unlink(cfg_path)
  return rc == 0


def _kubectl_apply(cfg: dict[str, Any], yaml_path: str) -> tuple[bool, str]:
  """Applies the ConfigMap + JobSet bundle at yaml_path to the configured cluster."""
  cmd = _kubectl_cmd(
      "apply",
      "-f",
      yaml_path,
      context=cfg.get("cluster_context", ""),
      namespace=cfg.get("namespace", ""),
      include_ns=True,
  )
  _logger.info("$ %s", " ".join(cmd))
  proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
  output = (proc.stdout + proc.stderr).strip()
  return proc.returncode == 0, output


def _delete_workload(cfg: dict[str, Any]) -> tuple[bool, str]:
  """Deletes the JobSet and its workload ConfigMap from the configured cluster."""
  ctx = cfg.get("cluster_context", "")
  ns = cfg.get("namespace", "")
  safe_js = _validate_k8s_id(cfg["jobset_name"], "jobset_name")
  outputs = []
  ok = True
  for kind, name in (("jobset", safe_js), ("configmap", CONFIGMAP_NAME)):
    cmd = _kubectl_cmd(
        "delete",
        kind,
        name,
        "--ignore-not-found=true",
        context=ctx,
        namespace=ns,
        include_ns=True,
    )
    _logger.info("$ %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    ok = ok and proc.returncode == 0
    outputs.append((proc.stdout + proc.stderr).strip())
  return ok, "\n".join(o for o in outputs if o)


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
      self._send_json({
          "config": state.get_config(),
          "snapshot": state.get_snapshot(),
      })
    elif parsed.path == "/api/telemetry/stream":
      self._handle_sse()
    elif parsed.path in ("/", "/demo", "/gke", "/gke_elasticity_demo.html"):
      self._serve_demo_html()
    else:
      super().do_GET()

  def do_POST(self) -> None:  # pylint: disable=invalid-name
    """Routes HTTP POST requests to lifecycle and signal handlers."""
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

    if parsed.path == "/api/jobset/delete":
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
    elif parsed.path == "/api/signal/mode":
      self._handle_mode(payload)
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
    _, output = _delete_workload(cfg)
    state.clear_for_delete()
    state.broadcast({"type": "deleted", "snapshot": state.get_snapshot()})
    self._send_json({
        "status": "ok",
        "message": f"JobSet '{safe_js}' deleted and state cleared",
        "output": output,
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
      self._send_json(
          {
              "status": "error",
              "action": "missing_jobset",
              "message": (
                  f"No active JobSet '{cfg['namespace']}/{cfg['jobset_name']}'"
                  " found on GKE. Deploy with --yaml=... or kubectl apply"
                  " first."
              ),
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
      leader_prefix = f"{js_name}-{pw_jobset.PATHWAYS_WORKER_JOB_NAME}-{slice_idx}-0-"
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

  def _handle_mode(self, payload: dict[str, Any]) -> None:
    """Switches the elasticity recovery mode in the bridge and active head pod."""
    mode = str(payload.get("mode", "")).strip()
    if mode not in (
        "step_snapshot_replica_resize",
        "step_snapshot_pause_resume",
    ):
      self._send_json(
          {"status": "error", "message": f"Invalid elasticity mode: {mode!r}"},
          status=400,
      )
      return

    updated_cfg = state.update_config({"elasticity_mode": mode})
    with state.lock:
      head_pod = state.head_pod_name
      head_phase = state.head_pod_phase
      ctx = state.config["cluster_context"]
      ns = state.config["namespace"]

    ok = False
    out = ""
    if head_pod and head_phase == "Running":
      ok, out = _send_workload_control_rpc(
          head_pod, f"/mode?mode={mode}", context=ctx, namespace=ns
      )
      _logger.info("Set elasticity mode to %s on %s (%s)", mode, head_pod, ok)

    snap = state.get_snapshot()
    state.broadcast({
        "type": "config_updated",
        "config": updated_cfg,
        "snapshot": snap,
    })
    self._send_json({
        "status": "ok",
        "mode": mode,
        "forwarded": ok,
        "output": out,
        "snapshot": snap,
    })


class ThreadingServer(socketserver.ThreadingMixIn, server.HTTPServer):
  daemon_threads = True
  allow_reuse_address = True


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  """Parses CLI flags for starting the GKE elasticity demo server."""
  parser = argparse.ArgumentParser(
      description=(
          "Deploy the Pathways elastic training demo to GKE, start the bridge"
          " web server, and open the demo UI. Without --image, --build-image,"
          " or --yaml, the server only monitors an existing JobSet."
      ),
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  web = parser.add_argument_group("web server")
  web.add_argument(
      "--port", type=int, default=PORT, help="Port to run the HTTP server on."
  )
  web.add_argument(
      "--host",
      type=str,
      default="127.0.0.1",
      help="Host interface to bind the HTTP server to.",
  )
  web.add_argument(
      "--no-browser",
      action="store_true",
      help="Do not automatically open the demo webpage in a browser.",
  )

  cluster = parser.add_argument_group("cluster")
  cluster.add_argument(
      "--context",
      type=str,
      default="",
      help="kubeconfig context (defaults to `kubectl config current-context`).",
  )
  cluster.add_argument(
      "--namespace", type=str, default="", help="Kubernetes namespace."
  )
  cluster.add_argument(
      "--jobset-name", type=str, default="", help="Name of the JobSet."
  )
  cluster.add_argument(
      "--kueue-queue",
      type=str,
      default=None,
      help="Kueue LocalQueue name (omit if Kueue is not installed).",
  )
  cluster.add_argument(
      "--cpu-nodepool",
      type=str,
      default=None,
      help="GKE nodepool for the head pod (omit to let GKE schedule it).",
  )

  workload = parser.add_argument_group("workload")
  workload.add_argument(
      "--image",
      type=str,
      default="",
      help=(
          "Workload container image for the head pod, e.g."
          " REGION-docker.pkg.dev/PROJECT/REPO/elastic-demo:latest."
      ),
  )
  workload.add_argument(
      "--build-image",
      action="store_true",
      help="Build and push --image from this checkout before deploying.",
  )
  workload.add_argument(
      "--builder",
      choices=("gcloud", "docker"),
      default="gcloud",
      help="How to build --image: Cloud Build (gcloud) or local docker.",
  )
  workload.add_argument(
      "--project",
      type=str,
      default="",
      help="GCP project for `gcloud builds submit` (defaults to gcloud config).",
  )
  workload.add_argument(
      "--gcs-scratch",
      type=str,
      default="",
      help="GCS scratch location for Pathways, e.g. gs://BUCKET/elastic-demo/.",
  )
  workload.add_argument(
      "--server-image", type=str, default="", help="Pathways server image."
  )
  workload.add_argument(
      "--proxy-image", type=str, default="", help="Pathways proxy image."
  )
  workload.add_argument(
      "--tpu-type", type=str, default="", help="TPU type, e.g. tpuv5e, v6e."
  )
  workload.add_argument(
      "--topology", type=str, default="", help="Slice topology, e.g. 4x8."
  )
  workload.add_argument(
      "--num-slices", type=int, default=None, help="Number of training slices."
  )
  workload.add_argument(
      "--total-steps", type=int, default=None, help="Total training steps."
  )
  workload.add_argument(
      "--elasticity-mode",
      type=str,
      default="",
      help=(
          "One of step_snapshot_pause_resume, step_snapshot_replica_resize."
      ),
  )
  workload.add_argument(
      "--orbax-ref",
      type=str,
      default="",
      help="Git ref of google/orbax to install in the head pod.",
  )

  deploy = parser.add_argument_group("deploy")
  deploy.add_argument(
      "--render-only",
      type=str,
      default="",
      metavar="PATH",
      help=(
          "Write the generated ConfigMap + JobSet YAML to PATH and exit so you"
          " can review or edit it before deploying with --yaml."
      ),
  )
  deploy.add_argument(
      "--yaml",
      type=str,
      default="",
      help=(
          "Apply this ConfigMap + JobSet YAML instead of generating one."
          " --jobset-name and --namespace must match the manifest."
      ),
  )
  deploy.add_argument(
      "--apply",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Run `kubectl apply` on the generated or provided YAML.",
  )
  deploy.add_argument(
      "--delete",
      action="store_true",
      help="Delete the JobSet and ConfigMap from the cluster and exit.",
  )
  return parser.parse_args(argv)


_ACCEL_TO_TPU_TYPE = {
    "tpu-v5-lite-podslice": "tpuv5e",
    "tpu-v6e-slice": "tpuv6e",
    "tpu-v5p-slice": "tpuv5p",
    "tpu7x": "tpuv7x",
    "tpu-v4-podslice": "tpuv4",
}


def _infer_config_from_yaml(yaml_path: str) -> dict[str, Any]:
  """Extracts JobSet metadata, slice topology, and workload flags from YAML."""
  inferred: dict[str, Any] = {}
  with open(yaml_path, "r", encoding="utf-8") as f:
    docs = list(yaml.safe_load_all(f))
  for doc in docs:
    if not isinstance(doc, dict) or doc.get("kind") != "JobSet":
      continue
    meta = doc.get("metadata") or {}
    if meta.get("name"):
      inferred["jobset_name"] = str(meta["name"])
    if meta.get("namespace"):
      inferred["namespace"] = str(meta["namespace"])
    rjobs = (doc.get("spec") or {}).get("replicatedJobs") or []
    for rjob in rjobs:
      pod_spec = (
          (((rjob.get("template") or {}).get("spec") or {}).get("template") or {})
          .get("spec")
          or {}
      )
      if rjob.get("name") == pw_jobset.PATHWAYS_WORKER_JOB_NAME:
        if rjob.get("replicas"):
          inferred["num_slices"] = int(rjob["replicas"])
        node_sel = pod_spec.get("nodeSelector") or {}
        accel = node_sel.get("cloud.google.com/gke-tpu-accelerator", "")
        if accel in _ACCEL_TO_TPU_TYPE:
          inferred["tpu_type"] = _ACCEL_TO_TPU_TYPE[accel]
        if node_sel.get("cloud.google.com/gke-tpu-topology"):
          inferred["topology"] = str(
              node_sel["cloud.google.com/gke-tpu-topology"]
          )
      elif rjob.get("name") == pw_jobset.PATHWAYS_HEAD_JOB_NAME:
        for container in pod_spec.get("containers") or []:
          tokens = list(container.get("command") or []) + list(
              container.get("args") or []
          )
          joined = " ".join(str(t) for t in tokens)
          m_steps = re.search(r"--total_steps=(\d+)", joined)
          if m_steps:
            inferred["total_steps"] = int(m_steps.group(1))
          m_mode = re.search(r"--elasticity_mode=([a-z0-9_]+)", joined)
          if m_mode:
            inferred["elasticity_mode"] = m_mode.group(1)
  return inferred


def _startup_config_updates(args: argparse.Namespace) -> dict[str, Any]:
  """Maps CLI flags onto bridge configuration keys, skipping unset flags."""
  updates: dict[str, Any] = {}
  flag_values = vars(args)
  for flag, key in (
      ("context", "cluster_context"),
      ("namespace", "namespace"),
      ("jobset_name", "jobset_name"),
      ("image", "container_image"),
      ("gcs_scratch", "gcs_scratch_location"),
      ("server_image", "server_image"),
      ("proxy_image", "proxy_image"),
      ("tpu_type", "tpu_type"),
      ("topology", "topology"),
      ("elasticity_mode", "elasticity_mode"),
      ("orbax_ref", "orbax_ref"),
  ):
    value = flag_values[flag]
    if value:
      updates[key] = value
  for flag, key in (
      ("kueue_queue", "kueue_queue_name"),
      ("cpu_nodepool", "cpu_nodepool"),
      ("num_slices", "num_slices"),
      ("total_steps", "total_steps"),
  ):
    value = flag_values[flag]
    if value is not None:
      updates[key] = value
  return updates


def main(argv: list[str] | None = None) -> None:
  global tailer_instance, _ALLOW_ANY_HOST
  args = _parse_args(argv)
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
      force=True,
  )
  _logger.info("Initializing GKE Demo Bridge Server...")

  fqdn = (socket.getfqdn() or "").lower()
  short_host = (socket.gethostname() or "").lower()
  if fqdn:
    _ALLOWED_HOSTS.add(fqdn)
    _ALLOWED_HOSTS.add(fqdn.split(".")[0])
  if short_host:
    _ALLOWED_HOSTS.add(short_host)
    _ALLOWED_HOSTS.add(short_host.split(".")[0])
  if args.host in ("0.0.0.0", "::", ""):
    _ALLOW_ANY_HOST = True
  elif args.host:
    _ALLOWED_HOSTS.add(args.host.lower())

  yaml_path = JOBSET_YAML_PATH
  if args.yaml:
    yaml_path = os.path.abspath(args.yaml)
    if not os.path.isfile(yaml_path):
      raise SystemExit(f"--yaml file not found: {yaml_path}")
    inferred = _infer_config_from_yaml(yaml_path)
    if inferred:
      state.update_config(inferred)

  startup_updates = _startup_config_updates(args)
  if startup_updates:
    state.update_config(startup_updates)
  cfg = state.get_config()
  if not cfg["cluster_context"]:
    raise SystemExit(
        "No kubeconfig context found. Run `gcloud container clusters"
        " get-credentials ...` or pass --context=..."
    )

  if args.delete:
    ok, output = _delete_workload(cfg)
    if output:
      _logger.info(output)
    raise SystemExit(0 if ok else 1)

  if args.build_image:
    if not args.image:
      raise SystemExit("--build-image requires --image=REGISTRY/REPO/NAME:TAG")
    if not _build_and_push_image(
        args.image, args.builder, args.project, cfg["orbax_ref"]
    ):
      raise SystemExit("Image build failed.")
    _logger.info("Built and pushed %s", args.image)

  if args.render_only:
    ok, err, bundle = _build_yaml_bundle(cfg)
    if not ok:
      raise SystemExit(err)
    if not cfg["container_image"] or not cfg["gcs_scratch_location"]:
      _logger.warning(
          "container image or GCS scratch location is empty; fill in the"
          " `image:` and `--gcs_scratch_location=` fields before deploying."
      )
    out_path = os.path.abspath(args.render_only)
    with open(out_path, "w", encoding="utf-8") as f:
      f.write(bundle)
    _logger.info(
        "Wrote %s. Review or edit it, then deploy with --yaml=%s",
        out_path,
        out_path,
    )
    raise SystemExit(0)

  deploy_requested = bool(args.yaml or args.image or args.build_image)
  if args.yaml:
    _logger.info("Using workload manifest %s", yaml_path)
  elif deploy_requested:
    ok, msg, _ = _write_workload_yaml_bundle(cfg)
    if not ok:
      raise SystemExit(msg)
    _logger.info(msg)

  if deploy_requested and args.apply:
    ok, output = _kubectl_apply(cfg, yaml_path)
    if output:
      _logger.info(output)
    if not ok:
      raise SystemExit("kubectl apply failed.")
    _logger.info(
        "Deployed JobSet %s/%s. Monitoring live telemetry once the head pod"
        " starts.",
        cfg["namespace"],
        cfg["jobset_name"],
    )
  elif deploy_requested:
    _logger.info(
        "Skipping kubectl apply (--no-apply). Run: %s",
        _build_kubectl_apply_cmd(cfg, yaml_path),
    )
  else:
    _logger.info(
        "Monitor-only mode: watching %s/%s on context %s.",
        cfg["namespace"],
        cfg["jobset_name"],
        cfg["cluster_context"],
    )

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
    if args.host in ("0.0.0.0", "::", ""):
      _logger.info("Remote / Cloudtop URL: %s", remote_url)
    else:
      _logger.info("Pass --host=0.0.0.0 to expose on %s", remote_url)
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
