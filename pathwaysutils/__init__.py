# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Package of Pathways-on-Cloud utilities."""

import datetime
import logging
import os
import warnings

import jax
from pathwaysutils import cloud_logging
from pathwaysutils import profiling
from pathwaysutils import proxy_backend
from pathwaysutils.persistence import orbax_handler


_logger = logging.getLogger(__name__)
_initialization_count = 0
# When changing this, also update the CHANGELOG.md.
__version__ = "v0.1.0"


#  This is a brittle implementation since the platforms value is not necessarily
#  which backend is ultimately selected
def _is_pathways_used():
  return jax.config.jax_platforms and "proxy" in jax.config.jax_platforms


def _is_persistence_enabled():
  if "ENABLE_PATHWAYS_PERSISTENCE" in os.environ:
    if os.environ["ENABLE_PATHWAYS_PERSISTENCE"] == "1":
      return True
    if os.environ["ENABLE_PATHWAYS_PERSISTENCE"] == "0":
      return False
    else:
      raise ValueError(
          "ENABLE_PATHWAYS_PERSISTENCE must be set to 1/0 or unset, got: "
          + os.environ["ENABLE_PATHWAYS_PERSISTENCE"]
      )
  return False


def initialize():
  """Initializes pathwaysutils."""
  global _initialization_count
  _initialization_count += 1

  if _initialization_count == 1:
    warnings.warn(
        "pathwaysutils: Legacy initialization. Ensure you also call"
        " pathwaysutils.initialize(). This warning will be removed in a future"
        " release."
    )

  # Ignoring the second call to initialize() is a temporary measure so that this warning is not triggered for customers who are following our instructions and using the new initialize() function only once but have already had the legacy initialization triggered.
  if _initialization_count > 2:
    warnings.warn(
        "pathwaysutils: Already initialized. Ignoring duplicate call."
    )

  if _initialization_count > 1:
    return

  if _is_pathways_used():
    _logger.debug(
        "pathwaysutils: Detected Pathways-on-Cloud backend. Applying changes."
    )
    proxy_backend.register_backend_factory()
    profiling.monkey_patch_jax()
    # TODO: b/365549911 - Remove when OCDBT-compatible
    if _is_persistence_enabled():
      orbax_handler.register_pathways_handlers(datetime.timedelta(hours=1))

    # Turn off JAX compilation cache because Pathways handles its own
    # compilation cache.
    jax.config.update("jax_enable_compilation_cache", False)

    try:
      cloud_logging.setup()
    except Exception as error:  # pylint: disable=broad-except
      _logger.debug(
          "pathwaysutils: Failed to set up cloud logging due to the following"
          " error: %s",
          error,
      )
  else:
    _logger.debug(
        "pathwaysutils: Did not detect Pathways-on-Cloud backend. No changes"
        " applied."
    )

initialize()
