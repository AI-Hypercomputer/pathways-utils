"""Deploys Pathways service to a Kubernetes cluster using a JobSet template."""

from collections.abc import Callable, Sequence
import logging
import math
import os
import string
import subprocess
from typing import Any
from absl import app
from absl import flags
from pathwaysutils.experimental.shared_pathways_service import tpu_specs

_logger = logging.getLogger(__name__)

# Flag definitions
FLAGS = flags.FLAGS
_DEPLOYMENT_NAME = flags.DEFINE_string(
    "deployment_name", None, "Deployment name for the Pathways service"
)
_JAX_VERSION = flags.DEFINE_string(
    "jax_version", "0.10.1", "JAX version (e.g., 0.10.1)"
)
_SERVER_IMAGE = flags.DEFINE_string(
    "server_image", None, "Full path to the server Docker image"
)
_TPU_TYPE = flags.DEFINE_enum(
    "tpu_type", "v6e", ["v5e", "v5p", "v6e", "tpu7x"], "TPU type"
)
_TOPOLOGY = flags.DEFINE_string(
    "topology", "2x2", "TPU topology (e.g., 4x8, 2x2x2)"
)
_NUM_SLICES = flags.DEFINE_integer(
    "num_slices", 2, "Number of TPU slices"
)
_GCS_BUCKET = flags.DEFINE_string(
    "gcs_bucket",
    "gs://pathways-test-bucket",
    "GCS bucket name for RM checkpoint location",
)
_RM_TEMPLATE_FILE = flags.DEFINE_string(
    "rm_template_file",
    os.path.join(
        os.path.dirname(__file__),
        "yamls/shared-rm.yaml",
    ),
    "Path to the Shared RM YAML template file",
)
_KUEUE_TEMPLATE_FILE = flags.DEFINE_string(
    "kueue_template_file",
    os.path.join(
        os.path.dirname(__file__),
        "yamls/kueue-sps.yaml",
    ),
    "Path to the Kueue SPS YAML template file",
)
_NUM_PREDEPLOYED_WORKERS = flags.DEFINE_integer(
    "num_predeployed_workers", 0, "Number of worker JobSets to deploy upfront."
)
_SIDECAR_IMAGE = flags.DEFINE_string(
    "sidecar_image",
    "us-docker.pkg.dev/cloud-tpu-v2-images/pathways-colocated-python/sidecar:20260423-python_3.12-jax_0.10.0",
    "Full path to the sidecar Docker image for workers.",
)
_WORKER_TEMPLATE_FILE = flags.DEFINE_string(
    "worker_template_file",
    os.path.join(
        os.path.dirname(__file__),
        "yamls/tenant-worker.yaml",
    ),
    "Path to the worker JobSet YAML template file",
)
_DRY_RUN = flags.DEFINE_boolean(
    "dry_run",
    False,
    "If true, only print the generated YAML without deploying.",
)
_SIDECAR_SHM_DIR = "/tmp/sidecar_dir"


def _validate_topology(topology):
  """Validates the topology flag format."""
  try:
    dims = topology.split("x")
    if not (2 <= len(dims) <= 3):
      return False
    for dim in dims:
      if not dim.isdigit():
        return False
      if int(dim) <= 0:
        return False
    return True
  except ValueError:
    return False


flags.register_validator(
    "topology",
    _validate_topology,
    message=(
        "--topology must be in the format like 'AxB' or 'AxBxC', where A, B, C"
        " are positive integers."
    ),
)


def load_and_substitute_template(
    template_path: str, context: dict[str, Any]
) -> str:
  """Loads and substitutes the string.Template from the given path."""
  try:
    with open(template_path, "r") as f:
      template_str = f.read()
  except OSError as err:
    raise ValueError(
        f"Could not read template file: {template_path}: {err}"
    ) from err

  _logger.info("Template file: %s", template_path)
  _logger.info("Context: %s", context)
  template = string.Template(template_str)
  return template.substitute(context)


def deploy_yaml_str(yaml_str: str) -> None:
  """Deploys the given YAML string to the GKE cluster using kubectl."""
  try:
    subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=yaml_str,
        check=True,
        text=True,
        capture_output=True,
    )
    _logger.info("Successfully applied YAML.")
  except subprocess.CalledProcessError as e:
    _logger.exception("Error applying YAML: %s", e.stderr)
    raise


def run_deployment(
    deployment_name: str,
    tpu_type: str,
    topology: str,
    num_slices: int,
    gcs_bucket: str,
    server_image: str,
    rm_template_file: str,
    kueue_template_file: str,
    worker_template_file: str,
    num_predeployed_workers: int,
    sidecar_image: str,
    dry_run: bool,
    deploy_func: Callable[[str], None] = deploy_yaml_str,
) -> None:
  """Executes the deployment logic."""
  dims = [int(d) for d in topology.split("x")]
  chips_per_slice = math.prod(dims)

  tpu_params = tpu_specs.get_tpu_params(tpu_type, topology)

  context = {
      "DEPLOYMENT_NAME": deployment_name,
      "SERVER_IMAGE": server_image,
      "GCS_SCRATCH_LOCATION": gcs_bucket,
      "NUM_SLICES": num_slices,
      "TPU_TYPE": tpu_type,
      "NOMINAL_QUOTA": str(chips_per_slice * num_slices),
      **tpu_params,
  }

  rm_config_str = load_and_substitute_template(rm_template_file, context)
  kueue_config_str = load_and_substitute_template(kueue_template_file, context)

  _logger.info("--- Generated RM YAML ---")
  _logger.info("\n%s", rm_config_str)
  _logger.info("--- Generated Kueue YAML ---")
  _logger.info("\n%s", kueue_config_str)

  if not dry_run:
    _logger.info("Deploying RM...")
    deploy_func(rm_config_str)
    _logger.info("Deploying Kueue Configs...")
    deploy_func(kueue_config_str)

    if num_predeployed_workers > 0:
      for i in range(1, num_predeployed_workers + 1):
        worker_context = context.copy()
        worker_context.update({
            "WORKER_NAME": f"{deployment_name}-w{i}",
            "QUEUE_NAME": f"{deployment_name}-shared-tpu-local-q",
            "SIDECAR_IMAGE": sidecar_image,
            "SIDECAR_SHM_DIR": _SIDECAR_SHM_DIR,
            "PATHWAYS_RM_SERVICE_ADDRESS": f"{deployment_name}-rm-svc:29001",
        })
        worker_config_str = load_and_substitute_template(
            worker_template_file, worker_context
        )
        _logger.info("--- Generated Worker %d YAML ---", i)
        _logger.info("\n%s", worker_config_str)
        _logger.info("Deploying Worker %d...", i)
        deploy_func(worker_config_str)
  else:
    _logger.info("Dry run mode, not deploying.")


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  try:
    if (
        flags.FLAGS["jax_version"].present
        and flags.FLAGS["server_image"].present
    ):
      raise ValueError("Cannot provide both --jax_version and --server_image")

    if _SERVER_IMAGE.value:
      server_image = _SERVER_IMAGE.value
    else:
      server_image = f"us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:jax-{_JAX_VERSION.value}"

    if _DEPLOYMENT_NAME.value:
      # Take the first 8 characters to limit the length. GKE Pod name has a max
      # length limit of 63 characters.
      deployment_name = _DEPLOYMENT_NAME.value[:8]
    else:
      deployment_name = f"pw-{_TPU_TYPE.value}"

    run_deployment(
        deployment_name=deployment_name,
        tpu_type=_TPU_TYPE.value,
        topology=_TOPOLOGY.value,
        num_slices=_NUM_SLICES.value,
        gcs_bucket=_GCS_BUCKET.value,
        server_image=server_image,
        rm_template_file=_RM_TEMPLATE_FILE.value,
        kueue_template_file=_KUEUE_TEMPLATE_FILE.value,
        worker_template_file=_WORKER_TEMPLATE_FILE.value,
        num_predeployed_workers=_NUM_PREDEPLOYED_WORKERS.value,
        sidecar_image=_SIDECAR_IMAGE.value,
        dry_run=_DRY_RUN.value,
    )
  except ValueError as e:
    _logger.exception("Error: %s", e)
  except FileNotFoundError:
    _logger.exception(
        "Error: Template file not found at %s", _RM_TEMPLATE_FILE.value
    )


if __name__ == "__main__":
  app.run(main)
