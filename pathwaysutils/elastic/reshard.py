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
"""Resharding API for elastic training."""

from typing import Any, Protocol
import jax

PyTree = Any


class PutArrayFn(Protocol):

  def __call__(
      self,
      arr: jax.Array,
      dst_sharding: jax.sharding.Sharding,
      *,
      donate: bool,
      may_alias: bool | None,
  ) -> jax.Array:
    ...


def reshard(
    x: jax.Array | PyTree,
    sharding: jax.sharding.Sharding | PyTree,
    *,
    donate: bool = False,
    may_alias: bool | None = None,
    put_array: PutArrayFn | None = None,
) -> jax.Array | PyTree:
  """Reshards `x` to the specified `sharding`.

  Args:
      x: A `jax.Array` or a nested `jax.Array` in a Python container (must match
        the structure of `sharding`).
      sharding: A `Sharding` or a nested `Sharding` in a Python container (must
        match the structure of `x`), specifying the target sharding.
      donate: If `True`, donates the input arrays to reduce memory needed for
        resharding. Donated buffers should not be reused.
      may_alias: If `True`, allows aliasing of the input arrays. Default is
        `None`, which means the default behavior of `jax.device_put` will be
        used.
      put_array: A function that takes an array, a sharding, a boolean
        indicating whether to donate the input, and a boolean indicating whether
        to allow aliasing, and returns a copy of the array with the specified
        sharding.

  Returns:
      A copy of `x` with the specified `sharding`.
  """
  if put_array is None:
    put_array = jax.device_put

  return jax.tree.map(
      lambda arr, sharding: put_array(
          arr, sharding, donate=donate, may_alias=may_alias
      ),
      x,
      sharding,
  )
