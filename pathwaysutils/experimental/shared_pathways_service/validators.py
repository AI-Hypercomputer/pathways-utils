"""Validation functions for Shared Pathways Service."""

from collections.abc import Iterable, Mapping
import dataclasses
import importlib
import logging
import re
import sys
from typing import Any
from absl import flags
import jax

_logger = logging.getLogger(__name__)

_PYTHON_VERSION_REGEX = r"python[-_]?(\d+\.\d+(?:\.\d+)*)"
_JAX_VERSION_REGEX = r"jax[-_]?(\d+\.\d+(?:\.\d+)*)"
_JAXLIB_VERSION_REGEX = r"jaxlib[-_]?(\d+\.\d+(?:\.\d+)*)"


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


def _extract_image_tag(sidecar_image: str) -> str | None:
  """Extracts the tag from a container image string, ignoring digests."""
  image_without_digest = sidecar_image.split("@", 1)[0]
  last_slash = image_without_digest.rfind("/")
  if ":" not in image_without_digest[last_slash + 1 :]:
    return None
  return image_without_digest.rsplit(":", 1)[1]


def _clean_version(version_str: str) -> str:
  match = re.match(r"^(\d+(?:\.\d+)*)", version_str)
  return match.group(1) if match else version_str


def extract_sidecar_image_versions(sidecar_image: str) -> SidecarVersions:
  """Extracts Python, JAX, and JAXLib versions from the sidecar image tag.

  Args:
    sidecar_image: The sidecar image string, e.g.,
      "us-docker.pkg.dev/.../sidecar:20260423-python_3.12-jax_0.10.0".

  Returns:
    A SidecarVersions object with the extracted versions, or None for versions
    that could not be determined.
  """
  tag = _extract_image_tag(sidecar_image)
  if not tag:
    return SidecarVersions()

  python_version = None
  py_match = re.search(_PYTHON_VERSION_REGEX, tag, re.IGNORECASE)
  if py_match:
    python_version = _clean_version(py_match.group(1))

  jax_version = None
  jax_match = re.search(_JAX_VERSION_REGEX, tag, re.IGNORECASE)
  if jax_match:
    jax_version = _clean_version(jax_match.group(1))

  jaxlib_version = None
  jaxlib_match = re.search(_JAXLIB_VERSION_REGEX, tag, re.IGNORECASE)
  if jaxlib_match:
    jaxlib_version = _clean_version(jaxlib_match.group(1))
  elif jax_version:
    # JAX and JAXLib release versions correspond to each other by default.
    jaxlib_version = jax_version

  return SidecarVersions(
      python_version=python_version,
      jax_version=jax_version,
      jaxlib_version=jaxlib_version,
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

  if sidecar_versions.jax_version:
    jaxlib = sidecar_versions.jaxlib_version or sidecar_versions.jax_version
    lines.append(
        "To install the matching JAX and JAXLib versions locally, run:"
    )
    lines.append(
        f"  pip install jax=={sidecar_versions.jax_version} jaxlib=={jaxlib}"
    )
  else:
    lines.append(
        "Could not determine JAX version from colocated python sidecar"
        f" image: {sidecar_image}"
    )
  return "\n".join(lines)


def validate_sidecar_image_versions(
    sidecar_image: str, sidecar_versions: SidecarVersions | None = None
) -> None:
  """Checks compatibility of sidecar image versions with user environment.

  Compares the Python, JAX, and JAXLib versions in the sidecar image or
  container with the user environment's Python, JAX, and JAXLib versions.

  Args:
    sidecar_image: The sidecar image string, e.g.,
      "us-docker.pkg.dev/.../sidecar:20260423-python_3.12-jax_0.10.0".
    sidecar_versions: Optional pre-resolved SidecarVersions (e.g., queried from
      the sidecar container or image). If omitted, versions are extracted from
      the sidecar image tag.

  Raises:
    ValueError: If the sidecar image Python, JAX, or JAXLib versions do not
      match the user environment.
  """
  _logger.info(
      "Checking sidecar image version compatibility: %s", sidecar_image
  )

  tag = _extract_image_tag(sidecar_image)
  explicit_versions_provided = sidecar_versions is not None

  if sidecar_versions is None or (
      not sidecar_versions.python_version
      and not sidecar_versions.jax_version
      and not sidecar_versions.jaxlib_version
  ):
    if not tag:
      _logger.warning(
          "No tag found in sidecar image: %s. Skipping version validation.",
          sidecar_image,
      )
      return
    sidecar_versions = extract_sidecar_image_versions(sidecar_image)
    explicit_versions_provided = False

  if (
      not sidecar_versions.python_version
      and not sidecar_versions.jax_version
      and not sidecar_versions.jaxlib_version
  ):
    _logger.warning(
        "No Python or JAX versions found in sidecar image tag: %s. Skipping "
        "version validation.",
        tag,
    )
    return

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
    sidecar_python = _clean_version(sidecar_versions.python_version)
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

  if sidecar_versions.jax_version:
    sidecar_jax = _clean_version(sidecar_versions.jax_version)
    env_jax = _clean_version(jax.__version__)
    if not versions_match(sidecar_jax, env_jax):
      raise ValueError(
          f"JAX version mismatch: sidecar image matches JAX version "
          f"{sidecar_jax}, but the user environment is running JAX "
          f"{env_jax}. Either rebuild the sidecar image with a matching "
          "JAX version or update the user environment to match the sidecar "
          f"image{install_hint}."
      )
    _logger.info(
        "JAX version match: sidecar image matches JAX version %s, and the user"
        " environment is running JAX %s.",
        sidecar_jax,
        env_jax,
    )

  should_check_jaxlib = explicit_versions_provided or bool(
      tag and re.search(_JAXLIB_VERSION_REGEX, tag, re.IGNORECASE)
  )
  if should_check_jaxlib and sidecar_versions.jaxlib_version:
    sidecar_jaxlib = _clean_version(sidecar_versions.jaxlib_version)
    env_jaxlib = None
    try:
      jaxlib_mod = sys.modules.get("jaxlib")
      if jaxlib_mod is None:
        jaxlib_mod = importlib.import_module("jaxlib")
      if hasattr(jaxlib_mod, "__version__"):
        env_jaxlib = _clean_version(jaxlib_mod.__version__)
    except (ImportError, AttributeError):
      env_jaxlib = None

    if env_jaxlib is not None:
      if not versions_match(sidecar_jaxlib, env_jaxlib):
        raise ValueError(
            f"JAXLib version mismatch: sidecar image matches JAXLib version "
            f"{sidecar_jaxlib}, but the user environment is running JAXLib "
            f"{env_jaxlib}. Either rebuild the sidecar image with a matching "
            "JAXLib version or update the user environment to match the"
            f" sidecar image{install_hint}."
        )
      _logger.info(
          "JAXLib version match: sidecar image matches JAXLib version %s, and"
          " the user environment is running JAXLib %s.",
          sidecar_jaxlib,
          env_jaxlib,
      )


