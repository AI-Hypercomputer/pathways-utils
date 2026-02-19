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
  """Checks if the given instance represents a valid TPU type.

  Args:
    tpu_instance_with_topology: The TPU instance string, e.g., "tpuv6e:4x8".

  Raises ValueError if the instance is not a valid TPU type.
  """
  # Regex to extract TPU type and topology.
  # Examples:
  # tpuv6e:2x4 -> type='tpuv6e', topology='2x4'
  # tpuv5:2x2x1 -> type='tpuv5', topology='2x2x1'
  match = re.match(
      r"^(?:tpuv(?:5e|5|6e)):(?P<topology>\d+(?:x\d+){1,2})$",
      tpu_instance_with_topology,
  )

  if match:
    topology_str = match.group("topology")

    try:
      _ = [int(d) for d in topology_str.split("x")]
    except ValueError as exc:
      raise ValueError(
          f"Error: Invalid topology format '{topology_str}' in"
          f" '{tpu_instance_with_topology}'. Expected all numbers, e.g., 2x4"
          " for 2d topologies or 2x2x2 for 3-d topologies."
      ) from exc

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


def validate_proxy_server_image(proxy_server_image: str) -> None:
  """Validates the proxy server image format."""
  if not proxy_server_image or not proxy_server_image.strip():
    raise ValueError("Proxy server image cannot be empty.")
  if "/" not in proxy_server_image:
    raise ValueError(
        f"Proxy server image '{proxy_server_image}' must contain '/', "
        "separating the registry or namespace from the final image name."
    )
  if ":" not in proxy_server_image and "@" not in proxy_server_image:
    raise ValueError(
        f"Proxy server image '{proxy_server_image}' must contain a tag with ':'"
        " or a digest with '@'."
    )
