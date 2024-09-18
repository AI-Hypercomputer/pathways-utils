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
import os

from absl import logging
import jax
from pathwaysutils import cloud_logging
from pathwaysutils import profiling
from pathwaysutils import proxy_backend
from pathwaysutils.persistence import pathways_orbax_handler


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


if _is_pathways_used():
  logging.warning(
      "pathwaysutils: Detected Pathways-on-Cloud backend. Applying changes."
  )
  proxy_backend.register_backend_factory()
  profiling.monkey_patch_jax()
  # TODO(b/365549911): Remove when OCDBT-compatible
  if _is_persistence_enabled():
    pathways_orbax_handler.register_pathways_handlers(
        datetime.timedelta(minutes=10)
    )
  try:
    cloud_logging.setup()
  except OSError as e:
    logging.warning("pathwaysutils: Failed to set up cloud logging.")
else:
  logging.warning(
      "pathwaysutils: Did not detect Pathways-on-Cloud backend. No changes"
      " applied."
  )
