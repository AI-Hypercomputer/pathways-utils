"""TPU specifications and helper functions for Shared Pathways Service."""

import dataclasses
import logging
import math

_logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class TPUConfig:
  """Holds configuration details for a specific TPU type."""

  machine_type: str
  chips_per_vm: int
  accelerator_label: str
  instance_prefix: str


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
          instance_prefix="tpuv5",
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
          accelerator_label="tpu7x",
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


def parse_tpu_type_string(tpu_type_str: str) -> tuple[str, str]:
  """Parses a tpu_type string (e.g., 'tpuv5e:4x8') into (tpu_type, topology)."""
  if ":" not in tpu_type_str:
    raise ValueError(
        f"Invalid tpu_type string: {tpu_type_str}. Expected format"
        " 'type:topology'"
    )
  prefix, topology = tpu_type_str.split(":", 1)

  # Map prefix back to tpu_type key
  prefix_map = {
      "tpuv5e": "v5e",
      "tpuv5": "v5p",
      "tpuv6e": "v6e",
      "tpu7x": "tpu7x",
  }
  if prefix not in prefix_map:
    raise ValueError(
        f"Unsupported TPU prefix: {prefix}. Supported prefixes are:"
        f" {list(prefix_map.keys())}"
    )

  return prefix_map[prefix], topology


def get_tpu_params(tpu_type: str, topology: str) -> dict[str, str]:
  """Returns a dictionary of TPU parameters for GKE templates."""
  tpu_config = get_tpu_config(tpu_type)
  vms_per_slice = calculate_vms_per_slice(topology, tpu_config.chips_per_vm)
  return {
      "ACCELERATOR_LABEL": tpu_config.accelerator_label,
      "TOPOLOGY": topology,
      "VMS_PER_SLICE": str(vms_per_slice),
      "CHIPS_PER_VM": str(tpu_config.chips_per_vm),
      "INSTANCE_TYPE": f"{tpu_config.instance_prefix}:{topology}",
  }
