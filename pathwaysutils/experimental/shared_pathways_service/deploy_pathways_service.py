"""Deploys Pathways service to a Kubernetes cluster using a JobSet template."""

from collections.abc import Callable, Sequence
import dataclasses
import logging
import math
import os
import string
from typing import Any
from absl import app
from absl import flags
from kubernetes import client
from kubernetes import config
import yaml

_logger = logging.getLogger(__name__)

# Flag definitions
FLAGS = flags.FLAGS
_JOBSET_NAME = flags.DEFINE_string(
    "jobset_name", "pathways-service", "Name of the JobSet"
)
_JAX_VERSION = flags.DEFINE_string(
    "jax_version", "0.9.0", "JAX version (e.g., 0.9.0)"
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
    "GCS bucket name for scratch space",
)
_TEMPLATE_FILE = flags.DEFINE_string(
    "template_file",
    os.path.join(
        os.path.dirname(__file__), "yamls/pw-service-example.yaml",
    ),
    "Path to the JobSet YAML template file",
)
_DRY_RUN = flags.DEFINE_boolean(
    "dry_run",
    False,
    "If true, only print the generated YAML without deploying.",
)


@dataclasses.dataclass(frozen=True)
class TPUConfig:
  """Holds configuration details for a specific TPU type."""
  machine_type: str
  chips_per_vm: int
  accelerator_label: str
  instance_prefix: str


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


def get_tpu_config(tpu_type: str) -> TPUConfig:
  """Returns a TPUConfig object containing TPU configuration details."""
  tpu_configs = {
      "v5e": TPUConfig(
          machine_type="ct5lp-hightpu-4t",
          chips_per_vm=4,
          accelerator_label="tpu-v5-lite-podslice",
          instance_prefix="tpuv5e",
      ),
      "v5p": TPUConfig(
          machine_type="ct5p-hightpu-4t",
          chips_per_vm=4,
          accelerator_label="tpu-v5p-slice",
          instance_prefix="tpuv5p",
      ),
      "v6e": TPUConfig(
          machine_type="ct6e-standard-4t",
          chips_per_vm=4,
          accelerator_label="tpu-v6e-slice",
          instance_prefix="tpuv6e",
      ),
      "tpu7x": TPUConfig(
          machine_type="tpu7x-standard-4t",
          chips_per_vm=4,
          accelerator_label="tpu-v7-slice",
          instance_prefix="tpu7x",
      ),
  }
  if tpu_type not in tpu_configs:
    raise ValueError(
        f"Unsupported TPU type: {tpu_type}. Supported types are:"
        f" {list(tpu_configs.keys())}"
    )
  return tpu_configs[tpu_type]


def calculate_vms_per_slice(topology: str, chips_per_vm: int) -> int:
  """Calculates the number of VMs per slice based on the topology."""
  try:
    dims = [int(d) for d in topology.split("x")]
    total_chips = math.prod(dims)
    if total_chips % chips_per_vm != 0:
      raise ValueError(
          f"Total chips ({total_chips}) in topology {topology} is not divisible"
          f" by chips_per_vm ({chips_per_vm})"
      )
    return total_chips // chips_per_vm
  except ValueError as e:
    raise ValueError(
        f"Invalid topology format: {topology}. Expected format like 'AxB' or"
        f" 'AxBxC'. {e}"
    ) from e


def load_and_substitute_template(
    template_path: str, context: dict[str, Any]
) -> dict[str, Any]:
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
  _logger.info("Template: %s", template)
  substituted_yaml = template.substitute(context)
  return yaml.safe_load(substituted_yaml)


def deploy_jobset(jobset_yaml: dict[str, Any]) -> None:
  """Deploys the JobSet to the current Kubernetes cluster."""
  try:
    config.load_kube_config()
    api = client.CustomObjectsApi()
    api.create_namespaced_custom_object(
        group="jobset.x-k8s.io",
        version="v1alpha2",
        namespace=jobset_yaml["metadata"]["namespace"],
        body=jobset_yaml,
        plural="jobsets",
    )
    _logger.info(
        "JobSet '%s' created successfully.", jobset_yaml["metadata"]["name"]
    )
  except client.rest.ApiException:
    _logger.exception("Error creating JobSet")
  except config.ConfigException:
    _logger.exception("Error loading Kubernetes configuration")


def run_deployment(
    tpu_type,
    topology,
    num_slices,
    jobset_name,
    gcs_bucket,
    server_image,
    template_file,
    dry_run,
    deploy_func: Callable[[dict[str, Any]], None] = deploy_jobset,
) -> None:
  """Executes the deployment logic."""
  tpu_config = get_tpu_config(tpu_type)
  vms_per_slice = calculate_vms_per_slice(topology, tpu_config.chips_per_vm)

  context = {
      "JOBSET_NAME": jobset_name,
      "SERVER_IMAGE": server_image,
      "GCS_SCRATCH_LOCATION": gcs_bucket,
      "NUM_SLICES": num_slices,
      "INSTANCE_TYPE": f"{tpu_config.instance_prefix}:{topology}",
      "VMS_PER_SLICE": vms_per_slice,
      "CHIPS_PER_VM": tpu_config.chips_per_vm,
      "ACCELERATOR_LABEL": tpu_config.accelerator_label,
      "TOPOLOGY": topology,
  }

  jobset_config = load_and_substitute_template(template_file, context)

  _logger.info("--- Generated JobSet YAML ---")
  _logger.info("\n%s", yaml.dump(jobset_config))

  if not dry_run:
    _logger.info("Deploying JobSet...")
    deploy_func(jobset_config)
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

    run_deployment(
        tpu_type=_TPU_TYPE.value,
        topology=_TOPOLOGY.value,
        num_slices=_NUM_SLICES.value,
        jobset_name=_JOBSET_NAME.value,
        gcs_bucket=_GCS_BUCKET.value,
        server_image=server_image,
        template_file=_TEMPLATE_FILE.value,
        dry_run=_DRY_RUN.value,
    )
  except ValueError as e:
    _logger.exception("Error: %s", e)
  except FileNotFoundError:
    _logger.exception(
        "Error: Template file not found at %s", _TEMPLATE_FILE.value
    )


if __name__ == "__main__":
  app.run(main)
