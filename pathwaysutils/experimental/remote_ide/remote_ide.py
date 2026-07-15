"""Launch Jupyter/VSCode in GKE pods (supporting pathways, non-pathways, and SPS)."""

import argparse
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

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))


def get_args():
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description="Launch Jupyter/VSCode (File-based Scripts)."
  )
  default_user = os.getenv("USER", "user")
  default_pattern = os.getenv("POD_PATTERN", default_user)

  parser.add_argument(
      "-m",
      "--mode",
      default=os.getenv("MODE", "jupyter"),
      choices=["jupyter", "vscode"],
      help="Mode to launch",
  )
  parser.add_argument(
      "-w",
      "--workload",
      default=default_pattern,
      help="Regex pattern for pod name",
  )
  parser.add_argument(
      "-P",
      "--port",
      type=int,
      default=int(os.getenv("PORT", "8888")),
      help="Port to forward",
  )
  parser.add_argument(
      "-b",
      "--bucket",
      default=os.getenv("GCS_BUCKET", ""),
      help="GCS bucket name for history sync (optional)",
  )
  parser.add_argument(
      "-c",
      "--check-active-session",
      action="store_true",
      help=(
          "Check if session exists. If running, skip setup and just tunnel."
      ),
  )
  parser.add_argument(
      "--non-pathways",
      action="store_true",
      help=(
          "if non-pathways, use workload name directly as search pattern"
          " instead of pathways-head pattern"
      ),
  )
  parser.add_argument(
      "-s",
      "--sps",
      action="store_true",
      help=(
          "if sps, launch in Shared Pathways Service (SPS) mode, using"
          " container name pathways-remote-env"
      ),
  )
  parser.add_argument(
      "-y",
      "--yaml",
      default="",
      help=(
          "Path to YAML file to deploy if no running pod is found (SPS mode"
          " only)."
      ),
  )
  parser.add_argument(
      "-i",
      "--sps-image",
      default="codercom/code-server:latest",
      help="Container image for SPS pod deployment (SPS mode only).",
  )
  parser.add_argument(
      "--instance-type",
      default="c4-standard-192",
      help="Node instance type selector (SPS mode only).",
  )
  return parser.parse_args()


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
    print(f"Warning: Could not check port status ({e}). Assuming closed.")
    return False


def load_script(filename, port, bucket, workload):
  """Reads a bash script from disk and injects variables."""
  try:
    with open(filename, "r") as f:
      script_content = f.read()

    # Replace the placeholders with actual values
    script_content = script_content.replace("{PORT}", str(port))
    script_content = script_content.replace("{BUCKET}", bucket if bucket else "")
    script_content = script_content.replace("{WORKLOAD}", workload)

    # Return command formatted for 'bash -c'
    return ["/bin/bash", "-c", script_content]
  except FileNotFoundError:
    print(f"Error: Could not find script file '{filename}'")
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
    print(f"No running pod found matching: {pattern}")
    sys.exit(1)
  return None


def deploy_yaml(yaml_path, workload, workload_name, image, port, instance_type):
  """Reads a YAML file, substitutes placeholders, and deploys it."""
  try:
    with open(yaml_path, "r") as f:
      content = f.read()
  except OSError as e:
    print(f"Error: Could not read YAML file '{yaml_path}': {e}")
    sys.exit(1)

  # Perform placeholders substitution
  content = content.replace("{WORKLOAD}", workload)
  content = content.replace("${WORKLOAD}", workload)
  content = content.replace("{WORKLOAD_NAME}", workload_name)
  content = content.replace("${WORKLOAD_NAME}", workload_name)
  content = content.replace("{IMAGE}", image)
  content = content.replace("${IMAGE}", image)
  content = content.replace("{PORT}", str(port))
  content = content.replace("${PORT}", str(port))
  content = content.replace("{INSTANCE_TYPE}", instance_type)
  content = content.replace("${INSTANCE_TYPE}", instance_type)

  # Deploy using kubectl
  import subprocess  # pylint: disable=g-import-not-at-top
  try:
    res = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=content,
        text=True,
        capture_output=True,
        check=True,
    )
    print(res.stdout)
  except subprocess.CalledProcessError as e:
    print(f"Error deploying YAML file: {e.stderr}")
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
      sys.stdout.write(
          f"[Tunnel] Forwarding 127.0.0.1:{self.local_port} ->"
          f" {self.pod_name}\n"
      )
      sys.stdout.flush()
    except OSError as e:
      sys.stderr.write(f"[Tunnel Error] Cannot bind port {self.local_port}: {e}\n")
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


def main():
  args = get_args()

  # 1. Determine Container Name and Search Pattern
  container_name = "pathways-remote-env" if args.sps else "jax-tpu"

  search_pattern = args.workload
  if not args.sps:
    if args.non_pathways:
      search_pattern = args.workload
    else:
      search_pattern = f"{args.workload}-pathways-head"

  # 2. Find or Deploy Pod
  if args.sps:
    pod_name = find_pod(search_pattern, exit_on_fail=False)
    if pod_name:
      print(f"Found running SPS pod: {pod_name}")
    else:
      yaml_path = args.yaml
      if not yaml_path:
        yaml_path = os.path.join(
            CURRENT_PATH,
            "..",
            "shared_pathways_service",
            "yamls",
            "sps-pod.yaml",
        )
        print(f"No running SPS pod found. Using default template: {yaml_path}")
      else:
        print(f"No running SPS pod found. Deploying using YAML: {yaml_path}")
      deploy_yaml(
          yaml_path,
          args.workload,
          search_pattern,
          args.sps_image,
          args.port,
          args.instance_type,
      )

      # Poll until the pod is running
      print("Waiting for SPS pod to be running...")
      timeout = 90
      start_time = time.time()
      while time.time() - start_time < timeout:
        pod_name = find_pod(search_pattern, exit_on_fail=False)
        if pod_name:
          break
        time.sleep(2)

      if not pod_name:
        print(
            "Timeout waiting for SPS pod matching"
            f" {search_pattern} to be running."
        )
        sys.exit(1)
      print(f"SPS pod is running: {pod_name}")
  else:
    pod_name = find_pod(search_pattern, exit_on_fail=True)
    print(f"Target Pod: {pod_name}")

  # 3. Start Tunnel
  tunnel_proc = multiprocessing.Process(
      target=run_tunnel_process, args=(pod_name, args.port, args.port)
  )
  tunnel_proc.start()
  time.sleep(1)

  skip_setup = False
  if args.check_active_session:
    print(
        f"Checking for existing {args.mode} session on port {args.port}..."
    )
    if is_port_active(pod_name, args.port, container_name):
      print("Active session detected! Skipping setup script.")
      skip_setup = True
    else:
      print("No active session found. Proceeding with installation.")

  # 4. Execute Logic
  try:
    if skip_setup:
      # If skipping setup, we just need to keep the script alive to keep the tunnel open
      print(
          "Session ready (Port Forwarding Only). Access at"
          f" http://127.0.0.1:{args.port}"
      )
      print("Press Ctrl+C to stop.")
      while True:
        time.sleep(1)
    else:
      # Standard Path: Load and Run Script
      cmd = []
      if args.mode == "jupyter":
        print("Loading 'jupyter_setup.sh'...")
        jupyter_setup_cmd = os.path.join(
            CURRENT_PATH, "scripts", "jupyter_setup.sh"
        )
        cmd = load_script(
            jupyter_setup_cmd, args.port, args.bucket, args.workload
        )
      elif args.mode == "vscode":
        print("Loading 'vscode_setup.sh'...")
        vscode_setup_cmd = os.path.join(
            CURRENT_PATH, "scripts", "vscode_setup.sh"
        )
        cmd = load_script(
            vscode_setup_cmd, args.port, args.bucket, args.workload
        )
      load_k8s_config()
      v1 = client.CoreV1Api()
      print(f"Session ready. Access at http://127.0.0.1:{args.port}")
      print("Press Ctrl+C to stop.")
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
          print(resp.read_stdout(), end="")
        if resp.peek_stderr():
          print(resp.read_stderr(), end="")
  except KeyboardInterrupt:
    print("\nStopping session...")
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"Execution Error: {e}")
  finally:
    if tunnel_proc.is_alive():
      tunnel_proc.terminate()
      tunnel_proc.join()
      print("Tunnel closed.")


if __name__ == "__main__":
  multiprocessing.freeze_support()
  main()
