# Copyright 2025 Google LLC
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
"""Pathways JAX abstractions.

This introduces an abstrction layer some JAX APIs that have changed over
`pathwaysutils`'s compatibility window.
"""

from typing import Any

try:
  # jax>=0.7.0
  from jax.extend import backend  # pylint: disable=g-import-not-at-top

  register_backend_cache = backend.register_backend_cache

  del backend
except AttributeError:
  # jax<0.7.0
  from jax._src import util  # pylint: disable=g-import-not-at-top

  def register_backend_cache(cache: Any, name: str, util=util):  # pylint: disable=unused-argument
    return util.cache_clearing_funs.add(cache.cache_clear)

  del util

try:
  # jax>0.7.0
  from jax.extend import backend  # pylint: disable=g-import-not-at-top

  ifrt_proxy = backend.ifrt_proxy
  del backend
except AttributeError:
  # jax<=0.7.0
  from jax.lib import xla_extension  # pylint: disable=g-import-not-at-top

  ifrt_proxy = xla_extension.ifrt_proxy
  del xla_extension
