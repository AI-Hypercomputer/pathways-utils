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


if _is_pathways_used():
  logging.warning("pathwaysutils: Detected Pathways-on-Cloud backend. Applying changes.")
  proxy_backend.register_backend_factory()
  profiling.monkey_patch_jax()
  # pathways_orbax_handler.register_pathways_handlers(
  #     datetime.timedelta(minutes=10)
  # )
  cloud_logging.setup()
else:
  logging.warning(
      "pathwaysutils: Did not detect Pathways-on-Cloud backend. No changes applied."
  )
