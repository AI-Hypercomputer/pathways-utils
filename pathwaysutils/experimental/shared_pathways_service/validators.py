"""Validation functions for Shared Pathways Service."""

from collections.abc import Iterable, Mapping
import logging
import re
import sys
from typing import Any
from absl import flags
import jax

_logger = logging.getLogger(__name__)

_PYTHON_VERSION_REGEX = r"python[-_]?(\d+\.\d+(?:\.\d+)*)"
_JAX_VERSION_REGEX = r"jax[-_]?(\d+\.\d+(?:\.\d+)*)"


def validate_proxy_options(proxy_options: Iterable[str] | None) -> None:
  """Validates that proxy options are in the format 'key:value'."""
  if not proxy_options:
    return
  for item in proxy_options:
    if (
        ":" not in item
        or len(item.split(":")) <= 1
        or not item.split(":", 1)[0]
        or not item.split(":", 1)[1]
    ):
      raise flags.ValidationError(
          f'--proxy_options must be in the format "key:value". Got: {item}'
      )


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
      r"^(?:tpu(?:v5e|v5|v6e|7x)):(?P<topology>\d+(?:x\d+){1,2})$",
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


def validate_xla_flags(xla_flags: Iterable[str] | None) -> None:
  """Validates that all XLA flags start with '--xla_'."""
  if not xla_flags:
    return
  for flag in xla_flags:
    if not flag.startswith("--xla_"):
      raise flags.ValidationError(
          f"XLA flag '{flag}' must start with '--xla_'."
      )


def validate_sidecar_image_versions(sidecar_image: str) -> None:
  """Checks compatibility of sidecar image versions with user environment.

  Compares the Python and JAX versions in the sidecar image tag with the user
  environment's Python and JAX versions.

  Args:
    sidecar_image: The sidecar image string, e.g.,
      "us-docker.pkg.dev/.../sidecar:20260423-python_3.12-jax_0.10.0".

  Raises:
    ValueError: If the sidecar image Python or JAX versions do not match the
      user environment.
  """
  _logger.info(
      "Checking sidecar image version compatibility: %s", sidecar_image
  )

  parts = sidecar_image.rsplit(":", 1)
  if len(parts) < 2:
    _logger.warning(
        "No tag found in sidecar image: %s. Skipping version validation.",
        sidecar_image,
    )
    return
  tag = parts[1]

  sidecar_python_match = re.search(
      _PYTHON_VERSION_REGEX, tag, re.IGNORECASE
  )
  sidecar_jax_match = re.search(
      _JAX_VERSION_REGEX, tag, re.IGNORECASE
  )
  if not sidecar_python_match and not sidecar_jax_match:
    _logger.warning(
        "No Python or JAX versions found in sidecar image tag: %s. Skipping "
        "version validation.",
        tag,
    )
    return

  def clean_version(version_str: str) -> str:
    match = re.match(r"^(\d+(?:\.\d+)*)", version_str)
    return match.group(1) if match else version_str

  def versions_match(sidecar_ver: str, env_ver: str) -> bool:
    sidecar_parts = sidecar_ver.split(".")
    env_parts = env_ver.split(".")
    compare_len = min(len(sidecar_parts), len(env_parts))
    if compare_len == 0:
      return False
    return sidecar_parts[:compare_len] == env_parts[:compare_len]

  if sidecar_python_match:
    sidecar_python = clean_version(sidecar_python_match.group(1))
    env_python = (
        f"{sys.version_info.major}.{sys.version_info.minor}."
        f"{sys.version_info.micro}"
    )
    if not versions_match(sidecar_python, env_python):
      raise ValueError(
          f"Python version mismatch: sidecar image matches Python version "
          f"{sidecar_python}, but the user environment is running Python "
          f"{env_python}. Either rebuild the sidecar image with a matching "
          "Python version or update the user environment to match the sidecar"
          " image."
      )
    _logger.info(
        "Python version match: sidecar image matches Python version %s, and the"
        " user environment is running Python %s.",
        sidecar_python,
        env_python,
    )

  if sidecar_jax_match:
    sidecar_jax = clean_version(sidecar_jax_match.group(1))
    env_jax = clean_version(jax.__version__)
    if not versions_match(sidecar_jax, env_jax):
      raise ValueError(
          f"JAX version mismatch: sidecar image matches JAX version "
          f"{sidecar_jax}, but the user environment is running JAX "
          f"{env_jax}. Either rebuild the sidecar image with a matching "
          "JAX version or update the user environment to match the sidecar "
          "image."
      )
    _logger.info(
        "JAX version match: sidecar image matches JAX version %s, and the user"
        " environment is running JAX %s.",
        sidecar_jax,
        env_jax,
    )

