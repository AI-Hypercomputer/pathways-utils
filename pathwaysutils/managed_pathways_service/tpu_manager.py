# Copyright 2024 Google LLC
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
"""Module for connecting to a Pathways server."""

import contextlib
import logging
import os
import random
import socket
import string
import subprocess

PROXY_PORT = 29000

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def fetch_cluster_credentials(
    cluster_name: str, project_id: str, location: str
) -> None:
  """Fetches credentials for the GKE cluster."""
  # Always ensure we have fresh credentials for kubectl.
  try:
    logger.info("Fetching credentials for '%s'.", cluster_name)
    subprocess.run(
        [
            "gcloud", "container", "clusters", "get-credentials", cluster_name,
            "--zone", location, "--project", project_id
        ],
        check=True, capture_output=True, text=True
    )
  except subprocess.CalledProcessError as e:
    logger.error(
        "Failed to get cluster credentials. gcloud output:\\n%s", e.stderr
    )
    raise


def deploy_pathways_cluster_pods(
    pathways_service: str,
    proxy_name: str,
    expected_instances: dict[str, int],
    gcs_bucket: str,
):
  """Deploys the Pathways proxy pods to the GKE cluster.

  Args:
    pathways_service: The service name and port of the Pathways head.
    proxy_name: The name to use for the deployed proxy.

  Raises:
    subprocess.CalledProcessError: If the kubectl command fails.
  """
  logger.info("Deploying Pathways proxy")
  script_dir = os.path.dirname(__file__)
  yaml_path = os.path.join(script_dir, "pw-proxy.yaml")
  with open(yaml_path, "r") as f:
    yaml_template = f.read()

  pathways_head, pathways_head_port = pathways_service.split(":")

  machine_type, count = list(expected_instances.items())[0]
  instances_str = ",".join([machine_type] * count)

  template = string.Template(yaml_template)
  substituted_yaml = template.substitute(
      PROXY_NAME=proxy_name,
      PATHWAYS_HEAD=pathways_head,
      PATHWAYS_HEAD_PORT=pathways_head_port,
      EXPECTED_INSTANCES=instances_str,
      GCS_BUCKET=gcs_bucket,
  )

  print(f"Proxy name: {proxy_name}")
  try:
    proxy_result = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=substituted_yaml,
        check=True,
        capture_output=True,
        text=True,
    )
    logger.info("Successfully deployed Pathways proxy. %s", proxy_result.stdout)
  except subprocess.CalledProcessError as e:
    logger.error(
        "Failed to deploy Pathways proxy. kubectl output:\\n%s", e.stderr
    )
    raise

  pass


def _find_free_local_port(starting_port: int) -> int:
  """Finds a free local port, starting from the given port."""
  port = starting_port
  while True:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
      try:
        s.bind(("", port))
        logger.info("Port binding successful at port: %d", port)
        return port
      except OSError:
        logger.info("Port %d is in use, trying next port.", port)
        port += 1


class TPUManager:
  """Class for managing TPUs."""

  def __init__(
      self,
      cluster: str,
      project: str,
      region: str,
      bucket: str,
      pathways_service: str,
      expected_instances: dict[str, int],
  ):
    """Initializes the TPU manager."""
    self.cluster = cluster
    self.project = project
    self.region = region
    self.bucket = bucket
    self.pathways_service = pathways_service
    self.expected_instances = expected_instances
    characters = "abcdefghijklmnopqrstuvwxyz0123456789"
    random_string = "".join(random.choice(characters) for _ in range(5))
    self.proxy_name = f"akshu-s4-{random_string}"
    # Save the original JAX environment variables so they can be restored when
    # the context manager exits.
    self.original_jax_platforms = None
    self.original_jax_backend_target = None
    self.port_forward_process = None
    self.proxy_port = None

  def __enter__(self):
    """Enters the context manager, ensuring cluster exists."""
    self.original_jax_platforms = os.environ.get("JAX_PLATFORMS")
    self.original_jax_backend_target = os.environ.get("JAX_BACKEND_TARGET")
    deploy_pathways_cluster_pods(
        self.pathways_service,
        self.proxy_name,
        self.expected_instances,
        self.bucket,
    )
    logger.info("Waiting for proxy pod to be ready...")
    try:
      wait_command = [
          "kubectl",
          "wait",
          "--for=condition=ready",
          "pod",
          "-l",
          f"jobset.sigs.k8s.io/jobset-name={self.proxy_name}",
          "--timeout=30s",
      ]
      subprocess.run(
          wait_command, check=True, capture_output=True, text=True
      )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
      logger.error("Error waiting for proxy pod to become ready: %s", e.stderr)
      try:
        log_command = f"kubectl logs jobset/{self.proxy_name}"
        logs_result = subprocess.run(
            log_command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.error(
            "Logs from jobset/%s:\n%s", self.proxy_name, logs_result.stdout
        )
      except subprocess.CalledProcessError as log_e:
        logger.error(
            "Could not retrieve logs for jobset/%s: %s",
            self.proxy_name,
            log_e.stderr,
        )
      raise RuntimeError("Proxy pod did not become ready.") from e

    get_pod_command = (
        f"kubectl get pods -l jobset.sigs.k8s.io/jobset-name={self.proxy_name} "
        "--no-headers -o custom-columns=':metadata.name'"
    )
    pod_result = subprocess.run(
        get_pod_command,
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    )
    proxy_pod = pod_result.stdout.strip().split("\n")[0]
    logger.info("Proxy pod ready: %s", proxy_pod)

    self.proxy_port = _find_free_local_port(PROXY_PORT)

    # Start port forwarding in the background.
    logger.info(
        "Starting port forwarding from local port %d to %s",
        self.proxy_port,
        proxy_pod,
    )
    self.port_forward_process = subprocess.Popen([
        "kubectl",
        "port-forward",
        proxy_pod,
        f"{self.proxy_port}:{PROXY_PORT}",
    ])

    os.environ["JAX_PLATFORMS"] = "proxy"
    os.environ["JAX_BACKEND_TARGET"] = f"grpc://127.0.0.1:{self.proxy_port}"
    logger.info("TPU manager ready for cluster '%s'.", self.cluster)
    return self

  def get_pathways_service(self):
    """Returns the Pathways service."""
    return self.pathways_service

  def __exit__(self, exc_type, exc_value, traceback):
    """Exits the context manager."""
    if self.port_forward_process:
      self.port_forward_process.terminate()
      self.port_forward_process.wait()
    if self.original_jax_platforms is None:
      if "JAX_PLATFORMS" in os.environ:
        del os.environ["JAX_PLATFORMS"]
    else:
      os.environ["JAX_PLATFORMS"] = self.original_jax_platforms

    if self.original_jax_backend_target is None:
      if "JAX_BACKEND_TARGET" in os.environ:
        del os.environ["JAX_BACKEND_TARGET"]
    else:
      os.environ["JAX_BACKEND_TARGET"] = self.original_jax_backend_target
    logger.info("Deleting Pathways proxy")
    try:
      proxy_result = subprocess.run(
          [
              "kubectl",
              "delete",
              "jobset",
              self.proxy_name,
              "--ignore-not-found",
          ],
          check=True,
          capture_output=True,
          text=True,
      )
      logger.info(
          "Successfully deleted Pathways proxy. %s", proxy_result.stdout
      )
    except subprocess.CalledProcessError as e:
      logger.error(
          "Failed to delete Pathways proxy. kubectl output:\\n%s", e.stderr
      )
      raise
    logger.info("Exiting TPUManager context.")


def validate_instance_list(expected_instances: dict[str, int]):
  """Validates the instance list."""
  if not expected_instances:
    logger.error("No instances found.")
    raise ValueError("No instances found.")
  for inst in expected_instances.keys():
    if not inst.strip():
      logger.error("Instance list contains empty string.")
      raise ValueError("Instance list contains empty string.")
  assert len(expected_instances.keys()) == 1, (
      "Only one machine type is supported at this time."
  )


@contextlib.contextmanager
def connect(
    cluster, project, region, bucket, pathways_service, expected_instances
):
  """Connects to a Pathways server if the cluster exists. If not, creates it."""
  validate_instance_list(expected_instances)
  fetch_cluster_credentials(cluster, project, region)
  with TPUManager(
      cluster, project, region, bucket, pathways_service, expected_instances
  ) as t:
    try:
      yield t
    finally:
      # Release the TPU resources.
      pass


def run():
  pass
