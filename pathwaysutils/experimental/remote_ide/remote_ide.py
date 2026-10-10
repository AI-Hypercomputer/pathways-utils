"""Launch Jupyter/VSCode in GKE pods (supporting Pathways and SPS)."""

import argparse
import base64
import logging
import multiprocessing
import os
import re
import select
import signal
import socket
import sys
import threading
import time
from kubernetes import client
from kubernetes import config
from kubernetes import stream
from pathwaysutils.experimental.shared_pathways_service import gke_utils

_logger = logging.getLogger(__name__)
CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
DEFAULT_SPS_YAML = os.path.join(
    CURRENT_PATH,
    "..",
    "shared_pathways_service",
    "yamls",
    "sps-pod.yaml",
)


def get_args():
  """Parses command-line arguments."""
  default_user = os.getenv("USER", "user")
  default_pattern = os.getenv("SESSION_PATTERN", default_user)

  # Parent parser for common flags so they work both globally and after subcommands
  parent_parser = argparse.ArgumentParser(add_help=False)

  parent_parser.add_argument(
      "-m",
      "--mode",
      default=os.getenv("MODE", "vscode"),
      choices=["jupyter", "vscode"],
      help="Mode to launch",
  )
  parent_parser.add_argument(
      "-s",
      "--session",
      default=default_pattern,
      help=(
          "Session name or pod pattern (defaults to $USER).\n"
          "How to get a session:\n"
          "  - Pathways Mode: Deploy your Pathways workload (e.g. using xpk). "
          "remote_ide connects to the existing pod matching '<session>-pathways-head'.\n"
          "  - SPS Mode ('sps'): Run 'remote_ide.py sps -s <session>', which automatically "
          "deploys an SPS pod named '<session>' if one does not already exist."
      ),
  )
  parent_parser.add_argument(
      "-P",
      "--port",
      type=int,
      default=int(os.getenv("PORT", gke_utils.pick_unused_local_port())),
      help="The local port to forward to the VS Code server (code-server) listening inside the remote pod/container.",
  )
  parent_parser.add_argument(
      "-b",
      "--bucket",
      default=os.getenv("GCS_BUCKET", ""),
      help="GCS bucket name for history sync (optional)",
  )
  parent_parser.add_argument(
      "-c",
      "--check-active-session",
      action="store_true",
      help=(
          "Check if session exists. If running, skip setup and just tunnel."
      ),
  )

  parser = argparse.ArgumentParser(
      description="Launch Jupyter/VSCode in GKE pods (supporting Pathways and SPS).",
      parents=[parent_parser],
  )

  # Subparsers for subcommand syntax (e.g. 'remote_ide.py sps ...')
  subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

  # --- SPS Subcommand Parser ---
  sps_parser = subparsers.add_parser(
      "sps",
      help="Launch in Shared Pathways Service (SPS) mode",
      parents=[parent_parser],
  )
  sps_parser.add_argument(
      "-i",
      "--image",
      dest="sps_image",
      required=True,
      help="Container image for SPS pod deployment.",
  )
  sps_parser.add_argument(
      "--instance-type",
      default=None,
      help="Node instance type selector (SPS mode only). Auto-detected if omitted.",
  )

  args = parser.parse_args()
  args.sps = getattr(args, "command", None) == "sps"

  if args.sps and not args.instance_type:
    detected_cpu = gke_utils.get_default_cpu_instance_type()
    if detected_cpu is None:
      _logger.error("Could not automatically detect any CPU nodes in the cluster. Please specify one explicitly with --instance-type.")
      sys.exit(1)
    args.instance_type = detected_cpu

  return args


def is_port_active(pod_name, port, container_name):
  """Executes a small python snippet inside the pod to check if the port is bound.

  Args:
    pod_name: Name of the pod.
    port: Bound port to verify.
    container_name: Target container runtime name.

  Returns:
    True if OPEN, False if CLOSED.
  """
  load_k8s_config()
  v1 = client.CoreV1Api()

  # Python one-liner to check a port (works on almost all images with python3)
  check_cmd = [
      "python3",
      "-c",
      (
          "import socket; s = socket.socket(socket.AF_INET,"
          f" socket.SOCK_STREAM); res = s.connect_ex(('127.0.0.1', {port}));"
          " print('OPEN' if res == 0 else 'CLOSED'); s.close()"
      ),
  ]
  try:
    resp = stream.stream(
        v1.connect_get_namespaced_pod_exec,
        pod_name,
        "default",
        command=check_cmd,
        container=container_name,
        stderr=True,
        stdin=False,
        stdout=True,
        tty=False,
        _preload_content=True,  # Wait for output
    )
    return "OPEN" in resp
  except Exception as e:  # pylint: disable=broad-exception-caught
    _logger.warning("Could not check port status (%s). Assuming closed.", e)
    return False


def load_script(filename, port, bucket, session):
  """Reads a bash script from disk and injects variables."""
  try:
    with open(filename, "r") as f:
      script_content = f.read()

    # Read and encode local ~/.bash_aliases if available
    bash_aliases_base64 = ""
    local_aliases_path = os.path.expanduser("~/.bash_aliases")
    if os.path.exists(local_aliases_path):
      try:
        with open(local_aliases_path, "rb") as fh:
          bash_aliases_base64 = base64.b64encode(fh.read()).decode("utf-8")
      except IOError as e:
        _logger.warning("Could not read local ~/.bash_aliases: %s", e)

    # Replace the placeholders with actual values
    script_content = script_content.replace("{PORT}", str(port))
    script_content = script_content.replace("{BUCKET}", bucket if bucket else "")
    script_content = script_content.replace("{WORKLOAD}", session)
    script_content = script_content.replace(
        "{BASH_ALIASES_BASE64}", bash_aliases_base64
    )

    # Return command formatted for 'bash -c'
    return ["/bin/bash", "-c", script_content]
  except FileNotFoundError:
    _logger.error("Could not find script file '%s'", filename)
    sys.exit(1)


def load_k8s_config():
  """Loads Kubernetes configuration."""
  try:
    config.load_kube_config()
  except Exception:  # pylint: disable=broad-exception-caught
    config.load_incluster_config()


def find_pod(pattern, exit_on_fail=True):
  """Finds a running pod in the default namespace matching the pattern regex."""
  load_k8s_config()
  v1 = client.CoreV1Api()
  pods = v1.list_namespaced_pod("default")
  regex = re.compile(pattern)

  for pod in pods.items:
    if regex.search(pod.metadata.name) and pod.status.phase == "Running":
      return pod.metadata.name

  if exit_on_fail:
    _logger.error("No running pod found matching: %s", pattern)
    sys.exit(1)
  return None


def deploy_yaml(yaml_path, session_name, image, port, instance_type):
  """Reads a YAML file, substitutes placeholders, and deploys it."""
  try:
    with open(yaml_path, "r") as f:
      content = f.read()
  except OSError as e:
    _logger.error("Could not read YAML file '%s': %s", yaml_path, e)
    sys.exit(1)

  # Perform placeholder substitutions
  content = content.replace("{SESSION}", session_name)
  content = content.replace("{IMAGE}", image)
  content = content.replace("{PORT}", str(port))
  content = content.replace("{INSTANCE_TYPE}", instance_type)

  # Deploy using gke_utils
  try:
    gke_utils.deploy_gke_yaml(content, action="apply")
  except Exception as e:  # pylint: disable=broad-exception-caught
    _logger.error("Error deploying YAML file: %s", e)
    sys.exit(1)


class PortForwarderServer:
  """Implements a port forwarder from local to the k8s pod."""

  def __init__(self, pod_name, local_port, remote_port, namespace="default"):
    self.pod_name = pod_name
    self.local_port = local_port
    self.remote_port = remote_port
    self.namespace = namespace
    self.running = True
    self.v1 = None

  def run(self):
    """Starts the port forwarding server loop."""
    load_k8s_config()
    self.v1 = client.CoreV1Api()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
      server_socket.bind(("127.0.0.1", self.local_port))
      server_socket.listen(5)
      _logger.info(
          "[Tunnel] Forwarding 127.0.0.1:%s -> %s",
          self.local_port,
          self.pod_name,
      )
    except OSError as e:
      _logger.error(
          "[Tunnel Error] Cannot bind port %s: %s", self.local_port, e
      )
      return
    while self.running:
      try:
        local_conn, _ = server_socket.accept()
        t = threading.Thread(target=self._handle_client, args=(local_conn,))
        t.daemon = True
        t.start()
      except KeyboardInterrupt:
        break
      except Exception:  # pylint: disable=broad-exception-caught
        pass

  def _handle_client(self, local_conn):
    """Handles an incoming client connection by bridging to kubernetes socket."""
    k8s_socket = None
    v1 = self.v1
    if v1 is None:
      load_k8s_config()
      v1 = client.CoreV1Api()
      self.v1 = v1
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
    """Bridges data transfer between two sockets."""
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


def run_tunnel_process(pod_name, local_port, remote_port):
  """Starts a client-facing port-forwarding server in a subprocess."""
  signal.signal(signal.SIGINT, signal.SIG_IGN)
  server = PortForwarderServer(pod_name, local_port, remote_port)
  server.run()


def ensure_pod_running(
    search_pattern,
    is_sps,
    sps_image,
    port,
    instance_type,
    yaml_path=DEFAULT_SPS_YAML,
):
  """Finds an existing pod or deploys an SPS pod if needed."""
  # We only auto-deploy in SPS mode. Normal mode requires pre-existing deployment
  exit_on_fail = not is_sps
  pod_name = find_pod(search_pattern, exit_on_fail=exit_on_fail)

  if pod_name:
    _logger.info("Found running SPS pod: %s", pod_name)
    return pod_name

  _logger.info(
      "No running SPS pod found. Deploying using SPS template: %s", yaml_path
  )
  deploy_yaml(
      yaml_path,
      search_pattern,
      sps_image,
      port,
      instance_type,
  )

  # Poll until the pod is running
  _logger.info("Waiting for SPS pod to be running...")
  timeout = 90
  start_time = time.time()
  while time.time() - start_time < timeout:
    pod_name = find_pod(search_pattern, exit_on_fail=False)
    if pod_name:
      break
    time.sleep(2)

  if not pod_name:
    _logger.error(
        "Timeout waiting for SPS pod matching '%s' to be running.\n"
        "Please check the pod status and events for troubleshooting:\n"
        "  1. Check pod status:  kubectl get pods -l app=%s\n"
        "  2. Describe pod:      kubectl describe pod -l app=%s\n"
        "  3. View pod logs:     kubectl logs -l app=%s",
        search_pattern,
        search_pattern,
        search_pattern,
        search_pattern,
    )
    sys.exit(1)

  _logger.info("SPS pod is running: %s", pod_name)
  return pod_name


def get_setup_command(mode, port, bucket, session):
  """Loads the setup script command for the requested mode."""
  script_name = "jupyter_setup.sh" if mode == "jupyter" else "vscode_setup.sh"
  _logger.info("Loading '%s'...", script_name)
  setup_script_path = os.path.join(CURRENT_PATH, "scripts", script_name)
  return load_script(setup_script_path, port, bucket, session)


def run_pod_session(pod_name, container_name, cmd, port):
  """Executes the setup command on the remote pod and streams output."""
  load_k8s_config()
  v1 = client.CoreV1Api()
  resp = stream.stream(
      v1.connect_get_namespaced_pod_exec,
      pod_name,
      "default",
      command=cmd,
      container=container_name,
      stderr=True,
      stdin=False,
      stdout=True,
      tty=False,
      _preload_content=False,
  )
  while resp.is_open():
    resp.update(timeout=1)
    if resp.peek_stdout():
      out = resp.read_stdout()
      sys.stdout.write(out)
      sys.stdout.flush()
      if "Starting VS Code Server" in out or "Starting Jupyter" in out:
        _logger.info("========================================")
        _logger.info("🚀 Session is completely ready!")
        _logger.info("👉 Access at http://127.0.0.1:%s", port)
        _logger.info("========================================")
    if resp.peek_stderr():
      sys.stderr.write(resp.read_stderr())
      sys.stderr.flush()


def main():
  args = get_args()

  local_port = args.port
  remote_port = 8888

  # 1. Determine Container Name and Search Pattern
  container_name = "sps-remote-jax" if args.sps else "jax-tpu"
  search_pattern = args.session if args.sps else f"{args.session}-pathways-head"

  # 2. Ensure Pod is Running
  pod_name = ensure_pod_running(
      search_pattern,
      args.sps,
      args.sps_image,
      remote_port,
      args.instance_type,
  )

  # 3. Start Tunnel Subprocess
  tunnel_proc = multiprocessing.Process(
      target=run_tunnel_process, args=(pod_name, local_port, remote_port)
  )
  tunnel_proc.start()
  time.sleep(1)

  # 4. Check Active Session or Execute Remote Setup
  try:
    if args.check_active_session and is_port_active(
        pod_name, remote_port, container_name
    ):
      _logger.info("Active session detected! Skipping setup script.")
      _logger.info(
          "Session ready (Port Forwarding Only). Access at"
          " http://127.0.0.1:%s",
          local_port,
      )
      _logger.info("Press Ctrl+C to stop.")
      while True:
        time.sleep(1)
    else:
      cmd = get_setup_command(args.mode, remote_port, args.bucket, args.session)
      run_pod_session(pod_name, container_name, cmd, local_port)
  except KeyboardInterrupt:
    _logger.info("Stopping session...")
  except Exception as e:  # pylint: disable=broad-exception-caught
    _logger.error("Execution Error: %s", e)
  finally:
    if tunnel_proc.is_alive():
      tunnel_proc.terminate()
      tunnel_proc.join()
      _logger.info("Tunnel closed.")


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s: %(message)s", force=True)
  multiprocessing.freeze_support()
  main()
