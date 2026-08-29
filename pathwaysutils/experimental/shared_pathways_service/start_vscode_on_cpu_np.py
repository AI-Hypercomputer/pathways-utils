"""Deploys VSCode on a GKE CPU node pool, or connects to an existing running Pathways pod."""

import multiprocessing
import os
import random
import re
import select
import signal
import socket
import string
import threading
import time

from absl import app
from absl import flags
from absl import logging
from kubernetes import client
from kubernetes import config
from kubernetes import stream
from pathwaysutils.experimental.shared_pathways_service import gke_utils

FLAGS = flags.FLAGS

_CLUSTER = flags.DEFINE_string(
    "cluster", None, "The name of the GKE cluster.", required=True
)
_PROJECT = flags.DEFINE_string(
    "project", None, "The GCP project ID.", required=True
)
_REGION = flags.DEFINE_string("region", None, "The GCP region.", required=True)
_NAMESPACE = flags.DEFINE_string(
    "namespace", "default", "Kubernetes namespace."
)
_NAME = flags.DEFINE_string(
    "name", "code-server", "Name of the deployment and service prefix."
)
_IMAGE = flags.DEFINE_string(
    "image", "codercom/code-server:latest", "VS Code image."
)
_PASSWORD = flags.DEFINE_string("password", "mypwd", "Password for VS Code.")
_INSTANCE_TYPE = flags.DEFINE_string(
    "instance_type", "c4-standard-192", "Node instance type selector."
)
_DRY_RUN = flags.DEFINE_boolean(
    "dry_run",
    False,
    "If true, only print the generated YAML without deploying.",
)
_WORKLOAD = flags.DEFINE_string(
    "workload",
    None,
    "Regex pattern to match the running Pod name. If provided, connects to an "
    "existing running pod instead of deploying a fresh one.",
)
_MODE = flags.DEFINE_enum(
    "mode",
    "vscode",
    ["vscode", "jupyter"],
    "IDE mode to launch (vscode or jupyter).",
)
_PORT = flags.DEFINE_integer(
    "port",
    8888,
    "Port to forward for the IDE when connecting to an existing workload.",
)
_BUCKET = flags.DEFINE_string(
    "bucket",
    "",
    "GCS Bucket for syncing state (VS Code only).",
)
_CHECK_ACTIVE_SESSION = flags.DEFINE_boolean(
    "check_active_session",
    False,
    "Check if session exists. If running, skip setup and just tunnel.",
)
_NON_PATHWAYS = flags.DEFINE_boolean(
    "non_pathways",
    False,
    "If true, use workload name directly as search pattern instead of "
    "pathways-head pattern.",
)

_TEMPLATE_FILE = os.path.join(
    os.path.dirname(__file__), "yamls/code-server.yaml"
)


def _load_k8s_config() -> None:
  try:
    config.load_kube_config()
  except Exception:  # pylint: disable=broad-exception-caught
    config.load_incluster_config()


def _find_pod(pattern: str) -> str:
  _load_k8s_config()
  v1 = client.CoreV1Api()
  pods = v1.list_namespaced_pod(_NAMESPACE.value)
  regex = re.compile(pattern)

  for pod in pods.items:
    if regex.search(pod.metadata.name) and pod.status.phase == "Running":
      return pod.metadata.name

  raise RuntimeError(f"No running pod found matching pattern: {pattern}")


def _is_port_active(pod_name: str, port: int, container_name: str) -> bool:
  """Executes a small python snippet inside the pod to check if the port is bound."""
  _load_k8s_config()
  v1 = client.CoreV1Api()
  check_cmd = [
      "python3",
      "-c",
      "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); "
      f"res = s.connect_ex(('127.0.0.1', {port})); "
      "print('OPEN' if res == 0 else 'CLOSED'); s.close()",
  ]
  try:
    resp = stream.stream(
        v1.connect_get_namespaced_pod_exec,
        pod_name,
        _NAMESPACE.value,
        command=check_cmd,
        container=container_name,
        stderr=True,
        stdin=False,
        stdout=True,
        tty=False,
        _preload_content=True,
    )
    return "OPEN" in resp
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.warning(
        "Could not check port status inside pod %s: %s. Assuming closed.",
        pod_name,
        e,
    )
    return False


def _load_script(
    filename: str, port: int, bucket: str, workload: str
) -> list[str]:
  """Reads a bash script from disk and injects variables."""
  try:
    with open(filename, "r") as f:
      script_content = f.read()

    script_content = script_content.replace("{PORT}", str(port))
    script_content = script_content.replace("{BUCKET}", bucket)
    script_content = script_content.replace("{WORKLOAD}", workload)

    return ["/bin/bash", "-c", script_content]
  except FileNotFoundError as e:
    raise ValueError(f"Could not find script file '{filename}'") from e


class PortForwarderServer:
  """Custom port forwarder server using Kubernetes API stream."""

  def __init__(
      self,
      pod_name: str,
      local_port: int,
      remote_port: int,
      namespace: str = "default",
  ):
    self.pod_name = pod_name
    self.local_port = local_port
    self.remote_port = remote_port
    self.namespace = namespace
    self.running = True

  def run(self):
    _load_k8s_config()
    v1 = client.CoreV1Api()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
      server_socket.bind(("127.0.0.1", self.local_port))
      server_socket.listen(5)
      logging.info(
          "[Tunnel] Forwarding 127.0.0.1:%d -> %s:%d",
          self.local_port,
          self.pod_name,
          self.remote_port,
      )
    except OSError as e:
      logging.error(
          "[Tunnel Error] Cannot bind port %d: %s", self.local_port, e
      )
      return

    while self.running:
      try:
        local_conn, _ = server_socket.accept()
        t = threading.Thread(target=self._handle_client, args=(local_conn, v1))
        t.daemon = True
        t.start()
      except KeyboardInterrupt:
        break
      except Exception:  # pylint: disable=broad-exception-caught
        pass

  def _handle_client(self, local_conn, v1):
    k8s_socket = None
    try:
      pf_stream = stream.portforward(
          v1.connect_get_namespaced_pod_portforward,
          self.pod_name,
          self.namespace,
          ports=str(self.remote_port),
      )
      k8s_socket = pf_stream.socket(self.remote_port)
      self._bridge_sockets(local_conn, k8s_socket)
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    finally:
      local_conn.close()
      if k8s_socket:
        k8s_socket.close()

  def _bridge_sockets(self, sock1, sock2):
    sockets = [sock1, sock2]
    buffer_size = 32768
    while True:
      r, _, _ = select.select(sockets, [], [])
      if sock1 in r:
        data = sock1.recv(buffer_size)
        if not data:
          break
        sock2.sendall(data)
      if sock2 in r:
        data = sock2.recv(buffer_size)
        if not data:
          break
        sock1.sendall(data)


def _run_tunnel_process(
    pod_name: str, local_port: int, remote_port: int, namespace: str
) -> None:
  signal.signal(signal.SIGINT, signal.SIG_IGN)
  server = PortForwarderServer(
      pod_name, local_port, remote_port, namespace=namespace
  )
  server.run()


def _prepare_deployment_yaml(service_name: str, remote_port: int) -> str:
  """Prepares the deployment YAML for VS Code."""
  context = {
      "NAME": service_name,
      "NAMESPACE": _NAMESPACE.value,
      "IMAGE": _IMAGE.value,
      "PASSWORD": _PASSWORD.value,
      "INSTANCE_TYPE": _INSTANCE_TYPE.value,
      "SERVICE_NAME": service_name,
      "PORT": str(remote_port),
  }

  logging.info("Loading and substituting template...")
  try:
    with open(_TEMPLATE_FILE, "r") as f:
      template_str = f.read()
  except OSError as err:
    raise ValueError("Could not read template file: " + _TEMPLATE_FILE) from err

  template = string.Template(template_str)
  return template.substitute(context)


def _deploy_vscode(
    service_name: str,
    deployment_yaml: str,
) -> None:
  """Deploys VS Code and sets up port forwarding."""
  gke_utils.deploy_gke_yaml(deployment_yaml, action="create")

  logging.info("Waiting for deployment to be ready...")
  gke_utils.wait_for_deployment(service_name, _NAMESPACE.value)

  logging.info("Waiting for service to get external IP...")
  try:
    ip = gke_utils.wait_for_service_ip(service_name, _NAMESPACE.value)
    logging.info("Service External IP (Internal Load Balancer): %s", ip)
  except RuntimeError as e:
    logging.warning("Could not get service IP: %s. Continuing anyway.", e)


def _start_port_forwarding(
    service_name: str,
    remote_port: int,
) -> None:
  """Starts port forwarding for the given service."""
  logging.info("Starting port forwarding...")
  pf_process = None
  try:
    local_port, pf_process = gke_utils.enable_port_forwarding(
        remote_server=f"svc/{service_name}",
        server_port=remote_port,
        namespace=_NAMESPACE.value,
    )
    logging.info("VS Code is accessible at http://localhost:%d", local_port)
    logging.info("Press Ctrl+C to stop port forwarding.")
    while True:
      time.sleep(1)
  except KeyboardInterrupt:
    logging.info("Stopping port forwarding...")
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Failed to start port forwarding: %s", e)
  finally:
    if pf_process:
      pf_process.terminate()
      pf_process.wait()
      logging.info("Port forwarding stopped.")


def _cleanup_gke_resources(service_name: str, namespace: str) -> None:
  logging.info("Deleting VS Code deployment and service...")
  try:
    gke_utils.delete_gke_resource("deployment", service_name, namespace)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Failed to delete VS Code deployment: %s", e)
  try:
    gke_utils.delete_gke_resource("service", service_name, namespace)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Failed to delete VS Code service: %s", e)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if _WORKLOAD.value:
    logging.info("Fetching cluster credentials...")
    gke_utils.fetch_cluster_credentials(
        cluster_name=_CLUSTER.value,
        project_id=_PROJECT.value,
        location=_REGION.value,
    )

    search_pattern = _WORKLOAD.value
    if _WORKLOAD.value == os.environ.get("USER", "user"):
      search_pattern = f"{_WORKLOAD.value}-pathways-head"
    if _NON_PATHWAYS.value:
      search_pattern = _WORKLOAD.value

    pod_name = _find_pod(search_pattern)
    logging.info("Target Pod: %s", pod_name)

    tunnel_proc = multiprocessing.Process(
        target=_run_tunnel_process,
        args=(pod_name, _PORT.value, _PORT.value, _NAMESPACE.value),
    )
    tunnel_proc.start()
    time.sleep(1)

    skip_setup = False
    if _CHECK_ACTIVE_SESSION.value:
      logging.info(
          "Checking for existing %s session on port %d...",
          _MODE.value,
          _PORT.value,
      )
      if _is_port_active(pod_name, _PORT.value, "jax-tpu"):
        logging.info("Active session detected! Skipping setup script.")
        skip_setup = True
      else:
        logging.info("No active session found. Proceeding with installation.")

    try:
      if skip_setup:
        logging.info(
            "Session ready (Port Forwarding Only). Access at"
            " http://127.0.0.1:%d",
            _PORT.value,
        )
        logging.info("Press Ctrl+C to stop.")
        while True:
          time.sleep(1)
      else:
        script_dir = os.path.join(os.path.dirname(__file__), "scripts")
        if _MODE.value == "jupyter":
          script_file = os.path.join(script_dir, "jupyter_setup.sh")
        else:
          script_file = os.path.join(script_dir, "vscode_setup.sh")

        cmd = _load_script(
            script_file,
            _PORT.value,
            _BUCKET.value,
            _WORKLOAD.value,
        )

        _load_k8s_config()
        v1 = client.CoreV1Api()
        logging.info(
            "Session ready. Access at http://127.0.0.1:%d", _PORT.value
        )
        logging.info("Press Ctrl+C to stop.")
        resp = stream.stream(
            v1.connect_get_namespaced_pod_exec,
            pod_name,
            _NAMESPACE.value,
            command=cmd,
            container="jax-tpu",
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
            _preload_content=False,
        )
        while resp.is_open():
          resp.update(timeout=1)
          if resp.peek_stdout():
            print(resp.read_stdout(), end="")
          if resp.peek_stderr():
            print(resp.read_stderr(), end="")
    except KeyboardInterrupt:
      logging.info("Stopping session...")
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.error("Execution Error: %s", e)
    finally:
      if tunnel_proc.is_alive():
        tunnel_proc.terminate()
        tunnel_proc.join()
        logging.info("Tunnel closed.")

  else:
    service_name = "{}".format(
        _NAME.value
        + f"-{os.environ.get('USER', 'user')}-"
        + "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    )
    logging.info("Service name: %s", service_name)

    remote_port = 8080

    deployment_yaml = _prepare_deployment_yaml(service_name, remote_port)

    if _DRY_RUN.value:
      logging.info(
          "Dry run: Would deploy the following YAML:\n%s", deployment_yaml
      )
      return

    logging.info("Fetching cluster credentials...")
    gke_utils.fetch_cluster_credentials(
        cluster_name=_CLUSTER.value,
        project_id=_PROJECT.value,
        location=_REGION.value,
    )
    try:
      _deploy_vscode(service_name, deployment_yaml)
      _start_port_forwarding(service_name, remote_port)
    finally:
      _cleanup_gke_resources(service_name, _NAMESPACE.value)


if __name__ == "__main__":
  app.run(main)
