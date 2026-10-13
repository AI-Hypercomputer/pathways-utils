"""Validation functions for Shared Pathways Service."""

from collections.abc import Iterable, Mapping
import dataclasses
import importlib
import logging
import re
import sys
from typing import Any
from absl import flags

_logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SidecarVersions:
  """Holds Python, JAX, and JAXLib versions for a colocated python sidecar."""

  python_version: str | None = None
  jax_version: str | None = None
  jaxlib_version: str | None = None


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


def validate_xla_flags(xla_flags: Iterable[str] | None) -> None:
  """Validates that all XLA flags start with '--xla_'."""
  if not xla_flags:
    return
  for flag in xla_flags:
    if not flag.startswith("--xla_"):
      raise flags.ValidationError(
          f"XLA flag '{flag}' must start with '--xla_'."
      )


def format_sidecar_versions(
    pathways_service: str,
    sidecar_image: str,
    sidecar_versions: SidecarVersions,
) -> str:
  """Formats the sidecar versions into a human-readable summary string."""
  lines = [
      (
          "Colocated Python sidecar found for Pathways service"
          f" '{pathways_service}':"
      ),
      f"  Sidecar Image: {sidecar_image}",
  ]
  if sidecar_versions.python_version:
    lines.append(f"  Python: {sidecar_versions.python_version}")
  if sidecar_versions.jax_version:
    lines.append(f"  JAX: {sidecar_versions.jax_version}")
  if sidecar_versions.jaxlib_version:
    lines.append(f"  JAXLib: {sidecar_versions.jaxlib_version}")
  return "\n".join(lines)


def validate_sidecar_image_versions(
    sidecar_image: str, sidecar_versions: SidecarVersions
) -> None:
  """Checks compatibility of sidecar package versions with user environment.

  Compares the Python, JAX, and JAXLib versions in the sidecar container with
  the user environment's Python, JAX, and JAXLib versions.

  Args:
    sidecar_image: The sidecar image string.
    sidecar_versions: The SidecarVersions queried from the sidecar container.

  Raises:
    ValueError: If the sidecar Python, JAX, or JAXLib versions do not match the
      user environment.
  """
  _logger.info(
      "Checking sidecar image version compatibility: %s", sidecar_image
  )

  def versions_match(sidecar_ver: str, env_ver: str) -> bool:
    sidecar_parts = sidecar_ver.split(".")
    env_parts = env_ver.split(".")
    compare_len = min(len(sidecar_parts), len(env_parts))
    if compare_len == 0:
      return False
    return sidecar_parts[:compare_len] == env_parts[:compare_len]

  install_hint = ""
  if sidecar_versions.jax_version and sidecar_versions.jaxlib_version:
    install_hint = (
        f" by running: pip install jax=={sidecar_versions.jax_version}"
        f" jaxlib=={sidecar_versions.jaxlib_version}"
    )

  if sidecar_versions.python_version:
    sidecar_python = sidecar_versions.python_version.strip()
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

  for pkg_name, display_name, raw_sidecar_ver in (
      ("jax", "JAX", sidecar_versions.jax_version),
      ("jaxlib", "JAXLib", sidecar_versions.jaxlib_version),
  ):
    if not raw_sidecar_ver:
      continue
    sidecar_ver = raw_sidecar_ver.strip()
    env_ver = None
    try:
      mod = sys.modules.get(pkg_name)
      if mod is None:
        mod = importlib.import_module(pkg_name)
      if hasattr(mod, "__version__"):
        env_ver = mod.__version__.strip()
    except (ImportError, AttributeError):
      env_ver = None

    if env_ver is not None:
      if sidecar_ver != env_ver:
        raise ValueError(
            f"{display_name} version mismatch: sidecar image matches"
            f" {display_name} version {sidecar_ver}, but the user environment"
            f" is running {display_name} {env_ver}. Either rebuild the sidecar"
            f" image with a matching {display_name} version or update the user"
            f" environment to match the sidecar image{install_hint}."
        )
      _logger.info(
          "%s version match: sidecar image matches %s version %s, and the user"
          " environment is running %s %s.",
          display_name,
          display_name,
          sidecar_ver,
          display_name,
          env_ver,
      )
