"""Deploys VSCode on a GKE CPU node pool and sets up port forwarding."""

import os
import random
import string
import time

from absl import app
from absl import flags
from absl import logging
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

_TEMPLATE_FILE = os.path.join(
    os.path.dirname(__file__), "yamls/code-server.yaml"
)


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
