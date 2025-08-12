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
"""An LRU cache that will be cleared when JAX clears its internal cache."""

import functools
from typing import Any, Callable

from pathwaysutils import jax as pw_jax


def lru_cache(
    maxsize: int = 4096,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
  """An LRU cache that will be cleared when JAX clears its internal cache.

  Args:
    maxsize: The maximum number of entries to keep in the cache. When this limit
      is reached, the least recently used entry will be evicted.

  Returns:
    A function that can be used to decorate a function to cache its results.
  """

  def wrap(f):
    cached = functools.lru_cache(maxsize=maxsize)(f)
    wrapper = functools.wraps(f)(cached)

    wrapper.cache_clear = cached.cache_clear
    wrapper.cache_info = cached.cache_info
    pw_jax.register_backend_cache(wrapper, "Pathways LRU cache")
    return wrapper

  return wrap
