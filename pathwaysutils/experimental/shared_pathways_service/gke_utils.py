"""GKE utils for deploying and managing the Pathways proxy."""

import logging
import socket
import subprocess
import urllib.parse

import portpicker

_logger = logging.getLogger(__name__)

# TODO(b/456189271): Evaluate and replace the subprocess calls with Kubernetes
# Python API for kubectl calls.


def fetch_cluster_credentials(
    *, cluster_name: str, project_id: str, location: str
) -> None:
  """Fetches credentials for the GKE cluster."""
  _logger.info("Fetching credentials for '%s'.", cluster_name)
  get_credentials_command = [
      "gcloud",
      "container",
      "clusters",
      "get-credentials",
      cluster_name,
      f"--location={location}",
      f"--project={project_id}",
  ]
  try:
    subprocess.run(
        get_credentials_command,
        check=True,
        capture_output=True,
        text=True,
    )
  except subprocess.CalledProcessError as e:
    _logger.exception(
        r"Failed to get cluster credentials. gcloud output:\n%r", e.stderr
    )
    raise


def job_exists(job_name: str, namespace: str = "default") -> bool:
  """Checks if a Kubernetes Job with the given name exists in the namespace.

  Args:
    job_name: The name of the Job.
    namespace: The Kubernetes namespace to check in. Defaults to "default".

  Returns:
    True if the Job exists, False otherwise.
  """
  command = [
      "kubectl",
      "get",
      "job",
      job_name,
      "-n",
      namespace,
      "-o",
      "name",
  ]

  try:
    _logger.debug(
        "Checking if job '%s' exists in namespace '%s'", job_name, namespace
    )
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    _logger.debug("kubectl get job output: %s", result.stdout.strip())
    # If the command succeeds and returns the name, the job exists.
    return job_name in result.stdout
  except subprocess.CalledProcessError as e:
    if "NotFound" in e.stderr:
      _logger.debug(
          "Job '%s' not found in namespace '%s': %s",
          job_name,
          namespace,
          e.stderr,
      )
      return False
    else:
      _logger.exception(
          "Error checking if job '%s' exists in namespace '%s': %s",
          job_name,
          namespace,
          e.stderr,
      )
      raise  # Re-raise unexpected errors
  except subprocess.TimeoutExpired:
    _logger.error("Timeout checking if job '%s' exists.", job_name)
    raise
  except Exception as e:
    _logger.exception(
        "Unexpected error checking job existence for '%s': %r", job_name, e
    )
    raise


def deploy_gke_yaml(yaml: str) -> None:
  """Deploys the given YAML to the GKE cluster.

  Args:
    yaml: The GKE YAML to deploy.

  Raises:
    subprocess.CalledProcessError: If the kubectl command fails.
  """
  _logger.info("Deploying GKE YAML: %s", yaml)
  kubectl_apply_command = ["kubectl", "apply", "-f", "-"]
  try:
    proxy_result = subprocess.run(
        kubectl_apply_command,
        input=yaml,
        check=True,
        capture_output=True,
        text=True,
    )
  except subprocess.CalledProcessError as e:
    _logger.exception(
        r"Failed to deploy the GKE YAML. kubectl output:\n%r", e.stderr
    )
    raise

  _logger.info(
      "Successfully deployed the GKE YAML. %s", proxy_result.stdout
  )


def get_pod_from_job(job_name: str) -> str:
  """Returns the pod name for the given job.

  Args:
    job_name: The name of the job.

  Returns:
    The name of the pod.

  Raises:
    subprocess.CalledProcessError: If the kubectl command fails.
    RuntimeError: If the pod is missing or the pod name is not in the expected
    format.
  """
  get_pod_command = [
      "kubectl",
      "get",
      "pods",
      "-l",
      f"job-name={job_name}",
      "-o",
      "name",
  ]
  try:
    pod_result = subprocess.run(
        get_pod_command,
        check=True,
        capture_output=True,
        text=True,
    )
  except subprocess.CalledProcessError as e:
    _logger.exception(
        r"Failed to get pod name. kubectl output:\n%r", e.stderr
    )
    raise

  pod_name = pod_result.stdout.strip()
  _logger.info("Pod name: %s", pod_name)

  if (
      not pod_name
      or not pod_name.startswith("pod/")
      or len(pod_name.split("/")) != 2
  ):
    raise RuntimeError(
        "Failed to get pod name. Expected format: pod/<pod_name>. Got:"
        f" {pod_name}"
    )

  # pod_name is in the format of "pod/<pod_name>". We only need the pod name.
  _, pod_name = pod_name.split("/")
  return pod_name


def check_pod_ready(pod_name: str, timeout: int = 30) -> str:
  """Checks if the given pod is ready.

  Args:
    pod_name: The name of the pod.
    timeout: The maximum time in seconds to wait for the pod to be ready.

  Returns:
    The name of the pod.

  Raises:
    RuntimeError: If the pod fails to become ready within the timeout.
  """
  wait_command = [
      "kubectl",
      "wait",
      "--for=condition=Ready",
      f"pod/{pod_name}",
      f"--timeout={timeout}s",
  ]
  try:
    subprocess.run(wait_command, check=True, capture_output=True, text=True)
  except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
    _logger.exception("Pod failed to become ready: %r", e)

    raise RuntimeError(
        f"Pod did not become ready: {e.stderr}."
    ) from e
  except Exception as e:
    _logger.exception("Error setting up the pod: %r", e)
    raise

  _logger.info("Pod is ready: %s.", pod_name)
  return pod_name


def get_log_link(*, cluster: str, project: str, job_name: str) -> str:
  """Returns a link to Cloud Logging for the given cluster and job name."""
  log_filter = (
      'resource.type="k8s_container"\n'
      f'resource.labels.cluster_name="{cluster}"\n'
      'resource.labels.namespace_name="default"\n'
      f'labels.k8s-pod/job-name="{job_name}"'
  )
  encoded_filter = urllib.parse.quote(log_filter, safe="")

  return (
      "https://console.cloud.google.com/logs/query;"
      f"query={encoded_filter};duration=PT1H"
      f"?project={project}"
  )


def wait_for_pod(job_name: str) -> str:
  """Waits for the given job's pod to be ready.

  Args:
    job_name: The name of the job.
  Returns:
    The name of the pod.
  Raises:
    RuntimeError: If the pod is not ready.
  """
  _logger.info("Waiting for pod to be created...")
  pod_name = get_pod_from_job(job_name)

  _logger.info(
      "Pod created: %s. Waiting for it to be ready...", pod_name
  )

  return check_pod_ready(pod_name)


def __test_pod_connection(port: int) -> None:
  """Tests the connection to the pod.

  Args:
    port: The port of the pod to connect to.
  """
  _logger.info("Connecting to localhost:%d", port)
  try:
    with socket.create_connection(("localhost", port), timeout=30):
      _logger.info("Pod is ready.")
  except (socket.timeout, ConnectionRefusedError) as exc:
    raise RuntimeError("Could not connect to the pod.") from exc


def enable_port_forwarding(
    pod_name: str,
    server_port: int,
) -> tuple[int, subprocess.Popen[str]]:
  """Enables port forwarding for the given pod.

  Args:
    pod_name: The name of the pod.
    server_port: The port of the server to forward to.

  Returns:
    A tuple containing the pod port and the port forwarding process.
  Raises:
    RuntimeError: If port forwarding fails to start or the pod connection
      cannot be established.
  """
  try:
    port_available = portpicker.pick_unused_port()
  except Exception as e:
    _logger.exception("Error finding free local port: %r", e)
    raise

  _logger.info("Found free local port: %d", port_available)
  _logger.info(
      "Starting port forwarding from local port %d to %s:%d",
      port_available,
      pod_name,
      server_port,
  )

  port_forward_command = [
      "kubectl",
      "port-forward",
      "--address",
      "localhost",
      pod_name,
      f"{port_available}:{server_port}",
  ]
  try:
    # Start port forwarding in the background.
    port_forward_process = subprocess.Popen(
        port_forward_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
  except Exception as e:
    _logger.exception("Error enabling port forwarding for the pod: %r", e)
    raise

  # Check that the port forwarding is ready.
  if port_forward_process.stdout is None:
    _logger.error("Port-forward process stdout is None. Terminating.")
    port_forward_process.terminate()
    _, stderr = port_forward_process.communicate()
    raise RuntimeError(
        "Failed to start port forwarding: stdout not available.\n"
        f"STDERR: {stderr}"
    )

  ready_line = port_forward_process.stdout.readline()
  if "Forwarding from" in ready_line:
    _logger.info("Port-forward is ready: %s", ready_line.strip())
  else:
    # If the ready line is not found, the process might have exited with an
    # error. We terminate it and raise an error with the stderr.
    _logger.error("Port-forward process exited with error. Terminating.")
    port_forward_process.terminate()
    _, stderr = port_forward_process.communicate()
    raise RuntimeError(
        "Failed to start port forwarding.\n"
        f"STDOUT: {port_forward_process.stdout}\n"
        f"STDERR: {stderr}"
    )

  try:
    __test_pod_connection(port_available)
  except Exception:
    port_forward_process.terminate()
    raise

  return (port_available, port_forward_process)


def delete_gke_job(job_name: str) -> None:
  """Deletes the given job from the GKE cluster.

  Args:
    job_name: The name of the job.
  """
  _logger.info("Deleting job: %s", job_name)
  delete_job_command = [
      "kubectl",
      "delete",
      "job",
      job_name,
      "--ignore-not-found",
  ]
  try:
    result = subprocess.run(
        delete_job_command,
        check=True,
        capture_output=True,
        text=True,
    )
  except subprocess.CalledProcessError as e:
    _logger.exception("Failed to delete job. kubectl output:\\n%r", e.stderr)
    raise
  _logger.info("Successfully deleted job. %s", result.stdout)
