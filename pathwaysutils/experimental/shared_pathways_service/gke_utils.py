"""GKE utils for deploying and managing the Pathways proxy."""

import json
import logging
import re
import socket
import subprocess
import time
import urllib.parse

import portpicker

_logger = logging.getLogger(__name__)

# TODO(b/456189271): Evaluate and replace the subprocess calls with Kubernetes
# Python API for kubectl calls.


def _validate_k8s_name(name: str) -> None:
  """Validates that the name is a valid Kubernetes resource name.

  Args:
    name: The name to validate.

  Raises:
    ValueError: If the name is invalid.
  """
  if not re.match(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$", name):
    raise ValueError(
        f"Invalid Kubernetes resource name: '{name}'. "
        "Must consist of lower case alphanumeric characters or '-', and must "
        "start and end with an alphanumeric character."
    )


def fetch_cluster_credentials(
    *, cluster_name: str, project_id: str, location: str
) -> None:
  """Fetches credentials for the GKE cluster."""
  _validate_k8s_name(cluster_name)
  _logger.info("Fetching credentials for '%s'.", cluster_name)
  get_credentials_command = [
      "gcloud",
      "container",
      "clusters",
      "get-credentials",
      f"--location={location}",
      f"--project={project_id}",
      "--dns-endpoint",
      "--",
      cluster_name,
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


def deploy_gke_yaml(yaml: str, action: str = "apply") -> None:
  """Deploys the given YAML to the GKE cluster.

  Args:
    yaml: The GKE YAML to deploy.
    action: The kubectl action to perform ("apply" or "create"). Create is
      equivalent to "apply" but does not support "replacing" the resource if it
      already exists.

  Raises:
    subprocess.CalledProcessError: If the kubectl command fails.
    ValueError: If action is not "apply" or "create".
  """
  if action not in ("apply", "create"):
    raise ValueError(f"Invalid kubectl action: {action}")
  _logger.info("Deploying GKE YAML with action %s: %s", action, yaml)
  kubectl_command = ["kubectl", action, "-f", "-"]
  try:
    proxy_result = subprocess.run(
        kubectl_command,
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


def delete_gke_resource(
    resource_type: str, name: str, namespace: str = "default"
) -> None:
  """Deletes the given resource from the GKE cluster.

  Args:
    resource_type: The type of resource to delete (e.g. "deployment",
      "service", "job").
    name: The name of the resource.
    namespace: The namespace of the resource.
  """
  _validate_k8s_name(resource_type)
  _validate_k8s_name(name)
  _validate_k8s_name(namespace)
  _logger.info(
      "Deleting %s: %s in namespace: %s", resource_type, name, namespace
  )
  command = [
      "kubectl",
      "delete",
      resource_type,
      "-n",
      namespace,
      "--ignore-not-found",
      "--",
      name,
  ]
  try:
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    _logger.info("Successfully deleted %s. %s", resource_type, result.stdout)
  except subprocess.CalledProcessError as e:
    _logger.exception(
        "Failed to delete %s. kubectl output:\n%r", resource_type, e.stderr
    )
    raise



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
  _validate_k8s_name(job_name)
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
  _validate_k8s_name(pod_name)
  wait_command = [
      "kubectl",
      "wait",
      "--for=condition=Ready",
      f"--timeout={timeout}s",
      "--",
      f"pod/{pod_name}",
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
      f'labels.k8s-pod/job-name:"{job_name}"'
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
  time.sleep(1)
  pod_name = get_pod_from_job(job_name)

  _logger.info(
      "Pod created: %s. Waiting for it to be ready...", pod_name
  )

  return check_pod_ready(pod_name)


def _test_remote_connection(port: int) -> None:
  """Tests the connection to the pod.

  Args:
    port: The port of the pod to connect to.
  """
  _logger.info("Connecting to localhost:%d", port)
  try:
    with socket.create_connection(("localhost", port), timeout=30):
      _logger.info("Connection to localhost:%d is ready.", port)
  except (socket.timeout, ConnectionRefusedError) as exc:
    raise RuntimeError("Could not connect to the pod.") from exc


def enable_port_forwarding(
    remote_server: str,
    server_port: int,
    namespace: str = "default",
) -> tuple[int, subprocess.Popen[str]]:
  """Enables port forwarding for the given pod.

  Args:
    remote_server: The name of the pod or service.
    server_port: The port of the server to forward to.
    namespace: The namespace of the pod.

  Returns:
    A tuple containing the pod port and the port forwarding process.
  Raises:
    RuntimeError: If port forwarding fails to start or the pod connection
      cannot be established.
  """
  try:
    local_port = portpicker.pick_unused_port()
  except Exception as e:
    _logger.exception("Error finding free local port: %r", e)
    raise

  _logger.info("Found free local port: %d", local_port)
  _logger.info(
      "Starting port forwarding from local port %d to %s:%d",
      local_port,
      remote_server,
      server_port,
  )

  if "/" in remote_server:
    parts = remote_server.split("/", 1)
    _validate_k8s_name(parts[0])
    _validate_k8s_name(parts[1])
  else:
    _validate_k8s_name(remote_server)
  _validate_k8s_name(namespace)
  port_forward_command = [
      "kubectl",
      "port-forward",
      "-n",
      namespace,
      "--address",
      "localhost",
      "--",
      f"{remote_server}",
      f"{local_port}:{server_port}",
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
    _test_remote_connection(local_port)
  except Exception:
    port_forward_process.terminate()
    raise

  return (local_port, port_forward_process)


def stream_pod_logs(pod_name: str) -> subprocess.Popen[str]:
  """Streams logs from the given pod.

  Args:
    pod_name: The name of the pod.

  Returns:
    The process for streaming the logs.

  Raises:
    Exception: If the log streaming fails.
  """
  _validate_k8s_name(pod_name)
  command = ["kubectl", "logs", "-f", "--", f"pod/{pod_name}"]
  try:
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line buffered
    )
  except Exception as _:
    _logger.exception("Error streaming logs for pod %s", pod_name)
    raise



def wait_for_deployment(
    name: str, namespace: str = "default", timeout: int = 300
) -> None:
  """Waits for deployment to be ready."""
  _validate_k8s_name(name)
  _validate_k8s_name(namespace)
  _logger.info("Waiting for deployment %s to be ready...", name)
  command = [
      "kubectl",
      "rollout",
      "status",
      f"deployment/{name}",
      "-n",
      namespace,
      f"--timeout={timeout}s",
  ]
  try:
    subprocess.run(command, check=True, capture_output=True, text=True)
  except subprocess.CalledProcessError as e:
    _logger.exception("Deployment failed to become ready: %r", e)
    raise RuntimeError(f"Deployment did not become ready: {e.stderr}") from e
  _logger.info("Deployment %s is ready.", name)


def wait_for_service_ip(
    name: str, namespace: str = "default", timeout: int = 300
) -> str:
  """Waits for service to get an external IP and returns it."""
  _validate_k8s_name(name)
  start_time = time.time()
  while time.time() - start_time < timeout:
    command = [
        "kubectl",
        "get",
        "svc",
        name,
        "-n",
        namespace,
        "-o",
        "jsonpath={.status.loadBalancer.ingress[0].ip}",
    ]
    try:
      result = subprocess.run(
          command, check=True, capture_output=True, text=True
      )
      ip = result.stdout.strip()
      if ip:
        _logger.info("Service IP assigned: %s", ip)
        return ip
    except subprocess.CalledProcessError as e:
      _logger.warning("Failed to get service IP: %r", e)
    time.sleep(2)
  raise RuntimeError(f"Timeout waiting for service IP for {name}")


def pick_unused_local_port() -> int:
  """Picks an unused local port."""
  return portpicker.pick_unused_port()


def is_local_port_free(port: int) -> bool:
  """Checks if a local port is free."""
  return portpicker.is_port_free(port)


def get_worker_sidecar_image(
    pathways_service: str, namespace: str = "default"
) -> str | None:
  """Gets the image of the sidecar container used by the workers."""
  pathways_head_hostname = pathways_service.split(":")[0]
  _validate_k8s_name(namespace)

  # Try to extract the jobset name from the Pathways service hostname.
  jobset_name = None
  if "-pathways-head" in pathways_head_hostname:
    jobset_name = pathways_head_hostname.split("-pathways-head")[0]

  command = ["kubectl", "get", "pods", "-n", namespace, "-o", "json"]
  try:
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
  except subprocess.CalledProcessError as e:
    _logger.exception("Failed to get pods. kubectl output:\n%r", e.stderr)
    return None

  try:
    pods_data = json.loads(result.stdout)
  except json.JSONDecodeError as e:
    _logger.exception("Failed to parse kubectl get pods output: %r", e)
    return None

  items = pods_data.get("items", [])

  # Look for pods belonging to the jobset and having the sidecar
  # container/initContainer.
  if jobset_name:
    for pod in items:
      metadata = pod.get("metadata", {})
      labels = metadata.get("labels", {})
      pod_jobset_name = labels.get("jobset.sigs.k8s.io/jobset-name")
      pod_name = metadata.get("name", "")

      if pod_jobset_name == jobset_name or pod_name.startswith(jobset_name):
        spec = pod.get("spec", {})
        for container in spec.get("initContainers", []) + spec.get(
            "containers", []
        ):
          if container.get("name") == "colocated-python-sidecar":
            image = container.get("image")
            if image:
              return image

  return None


def get_rm_server_image(
    pathways_service: str, namespace: str = "default"
) -> str | None:
  """Gets the server image used by the Pathways Resource Manager Service."""
  pathways_head_hostname = pathways_service.split(":")[0]
  _validate_k8s_name(namespace)
  _validate_k8s_name(pathways_head_hostname)

  # 1. Get the Service to find its selector
  get_service_cmd = [
      "kubectl",
      "get",
      "service",
      pathways_head_hostname,
      "-n",
      namespace,
      "-o",
      "json",
  ]
  try:
    result = subprocess.run(
        get_service_cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    service_data = json.loads(result.stdout)
    selector = service_data.get("spec", {}).get("selector")
    if not selector:
      _logger.warning(
          "No selector found for service %s in namespace %s",
          pathways_head_hostname,
          namespace,
      )
      return None
  except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
    _logger.exception(
        "Failed to get selector for service %s: %r",
        pathways_head_hostname,
        e,
    )
    return None

  # 2. Format the label selector
  label_selector = ",".join(f"{k}={v}" for k, v in selector.items())

  # 3. Get Pods matching the selector
  get_pods_cmd = [
      "kubectl",
      "get",
      "pods",
      "-n",
      namespace,
      "-l",
      label_selector,
      "-o",
      "json",
  ]
  try:
    result = subprocess.run(
        get_pods_cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    pods_data = json.loads(result.stdout)
    items = pods_data.get("items", [])
    if not items:
      _logger.warning(
          "No pods found matching selector %s in namespace %s",
          label_selector,
          namespace,
      )
      return None

    # Get the image from the first container of the first matching pod
    pod_spec = items[0].get("spec", {})
    containers = pod_spec.get("containers", [])
    if containers:
      image = containers[0].get("image")
      if image:
        return image

  except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
    _logger.exception(
        "Failed to get pods or parse image for selector %s: %r",
        label_selector,
        e,
    )
    return None

  return None
