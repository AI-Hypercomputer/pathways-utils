"""Validation functions for Shared Pathways Service."""

from collections.abc import Mapping
import logging
import re
from typing import Any

_logger = logging.getLogger(__name__)


def validate_pathways_service(pathways_service: str) -> None:
  """Validates the Pathways service name and port."""
  if not pathways_service:
    raise ValueError("No Pathways service found.")
  try:
    pathways_head, pathways_head_port = pathways_service.split(":")
  except ValueError as e:
    raise ValueError(
        f"pathways_service={pathways_service} is not in the expected format of"
        " `<pathways_head_address>:<port>`"
    ) from e
  if not pathways_head.strip():
    raise ValueError(
        f"pathways_service={pathways_service} contains an empty string for the"
        " service name. Expected `<pathways_head_address>:<port>`"
    )
  if not pathways_head_port.strip():
    raise ValueError(
        f"pathways_service={pathways_service} contains an empty string for the"
        " service port. Expected `<pathways_head_address>:<port>`"
    )
  try:
    int(pathways_head_port)
  except ValueError as e:
    raise ValueError(
        f"pathways_service={pathways_service} contains a non-numeric service"
        " port. Expected `<pathways_head_address>:<port>`"
    ) from e


def _validate_tpu_supported(tpu_instance_with_topology: str) -> None:
  """Checks if the given instance represents a valid single-host TPU.

  Args:
    tpu_instance_with_topology: The TPU instance string, e.g., "tpuv6e:4x8".

  Raises ValueError if the instance is not a valid TPU host.
  """
  # Mapping from Cloud TPU type prefix to max chips per host.
  # Make sure to edit the project README if you update this mapping.
  single_host_max_chips = {
      "tpuv6e": 8,  # Cloud TPU v6e (2x4)
  }

  # Regex to extract topology
  # Examples:
  # ct5lp-hightpu-4t:4x8 -> ct5lp, 4x8
  # ct5p:2x2x1 -> ct5p, 2x2x1
  match = re.match(
      r"^(?P<type>tpuv6e):(?P<topology>\d+(?:x\d+)*)$",
      tpu_instance_with_topology,
  )

  if match:
    tpu_base_type = match.group("type")
    topology_str = match.group("topology")

    if not tpu_base_type:
      raise ValueError(
          f"Unknown TPU type '{type}' from '{tpu_instance_with_topology}'."
      )

    try:
      dims = [int(d) for d in topology_str.split("x")]
      if len(dims) < 2 or len(dims) > 3:
        raise ValueError(
            f"Error: Invalid topology format '{topology_str}', Expected either"
            " 2 or 3 dimensions."
        )
      num_chips = 1
      for dim in dims:
        num_chips *= dim
    except ValueError as exc:
      raise ValueError(
          f"Error: Invalid topology format '{topology_str}' in"
          f" '{tpu_instance_with_topology}'."
      ) from exc

    if num_chips > single_host_max_chips[tpu_base_type]:
      raise ValueError(
          f"Topology '{tpu_instance_with_topology}' exceeds"
          f" {single_host_max_chips[tpu_base_type]}, the maximum supported"
          f" chips for {tpu_base_type}."
      )

    return

  raise ValueError(
      f"Unrecognized instance format: {tpu_instance_with_topology}."
  )


def validate_tpu_instances(expected_tpu_instances: Mapping[Any, Any]) -> None:
  """Validates the instance list."""
  if not expected_tpu_instances:
    raise ValueError("No instances found.")
  for inst in expected_tpu_instances.keys():
    if not inst.strip():
      raise ValueError(
          f"expected_tpu_instances={expected_tpu_instances} contains an "
          "empty string for an instance name."
      )
  if len(expected_tpu_instances.keys()) != 1:
    raise ValueError("Only one machine type is supported at this time.")

  inst = next(iter(expected_tpu_instances.keys()))
  _validate_tpu_supported(inst)
