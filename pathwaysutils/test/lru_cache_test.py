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

import jax.extend
from pathwaysutils import lru_cache
from google3.testing.pybase import googletest


class LruCacheTest(googletest.TestCase):

  def test_cache_hits(self):
    x = [100]

    @lru_cache.lru_cache(maxsize=1)
    def f(i):
      x[i] += 1
      return x[i]

    self.assertEqual(f(0), 101)  # Miss
    self.assertEqual(f(0), 101)  # Hit

  def test_cache_hits_and_misses_by_arguments(self):
    x = [100, 200]

    @lru_cache.lru_cache(maxsize=2)
    def f(i):
      x[i] += 1
      return x[i]

    self.assertEqual(f(0), 101)  # Miss
    self.assertEqual(f(0), 101)  # Hit

    self.assertEqual(f(1), 201)  # Miss
    self.assertEqual(f(1), 201)  # Hit

    self.assertEqual(f(0), 101)  # Hit
    self.assertEqual(f(0), 101)  # Hit

  def test_cache_lru_eviction(self):
    x = [100, 200]

    @lru_cache.lru_cache(maxsize=1)
    def f(i):
      x[i] += 1
      return x[i]

    self.assertEqual(f(0), 101)  # Miss
    self.assertEqual(f(0), 101)  # Hit

    self.assertEqual(f(1), 201)  # Miss
    self.assertEqual(f(1), 201)  # Hit

    self.assertEqual(f(0), 102)  # Miss
    self.assertEqual(f(0), 102)  # Hit

  def test_clear_cache_via_jax_clear_backend_cache(self):
    x = [100]

    @lru_cache.lru_cache(maxsize=1)
    def f(i):
      x[i] += 1
      return x[i]

    self.assertEqual(f(0), 101)  # Miss
    self.assertEqual(f(0), 101)  # Hit

    jax.extend.backend.clear_backends()

    self.assertEqual(f(0), 102)  # Miss
    self.assertEqual(f(0), 102)  # Hit


if __name__ == "__main__":
  googletest.main()
