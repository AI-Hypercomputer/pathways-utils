# Copyright 2026 Google LLC
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
"""Resharding API using the IFRT RemapArray API."""

import collections
from typing import Any, Callable, Mapping, Sequence

import jax
import pathwaysutils.jax


def reshard_generic(
    x: Any,
    sharding: jax.sharding.Sharding | Any,
    *,
    donate: bool,
    may_alias: bool | None,
    jax_array_reshard_fn: Callable[..., Any],
    **kwargs,
) -> Any:
  """Reshards `x` to `sharding`."""
  flat_x, tree_def = jax.tree.flatten(x)
  flat_sharding = jax.api_util.flatten_axes(
      "reshard sharding", tree_def, sharding
  )

  # We must split the arrays into two groups:
  # 1. jax.Array
  # 2. non jax.Array
  # For jax.Array, we will use the provided `jax_array_reshard_fn`.
  # For non jax.Array, we will use jax.device_put to put the array to the
  # destination devices. This is necessary for new-style random keys and
  # possibly other types of arrays.
  #
  # We need to track what index each array is in the original pytree, so we can
  # put them back together in the right order.
  array_info_lambda = lambda: {"arrays": [], "indices": [], "dst_shardings": []}
  jax_arrays = collections.defaultdict(array_info_lambda)
  non_reshardable_arrays = array_info_lambda()
  for index, (arr, dst_sharding) in enumerate(zip(flat_x, flat_sharding)):
    if not isinstance(dst_sharding, jax.sharding.Sharding):
      raise ValueError("`sharding` must contain only `jax.sharding.Sharding`")
    if not isinstance(arr, jax.Array) or (
        hasattr(arr, "dtype")
        and jax.dtypes.issubdtype(arr.dtype, jax.dtypes.prng_key)
    ):
      non_reshardable_arrays["arrays"].append(arr)
      non_reshardable_arrays["indices"].append(index)
      non_reshardable_arrays["dst_shardings"].append(dst_sharding)
    else:
      device_set = frozenset(arr.sharding.device_set)
      jax_arrays[device_set]["arrays"].append(arr)
      jax_arrays[device_set]["indices"].append(index)
      jax_arrays[device_set]["dst_shardings"].append(dst_sharding)

  if non_reshardable_arrays["arrays"]:
    non_reshardable_arrays["arrays"] = jax.device_put(
        non_reshardable_arrays["arrays"],
        non_reshardable_arrays["dst_shardings"],
        donate=donate,
        may_alias=may_alias,
    )

  for array_info in jax_arrays.values():
    array_info["arrays"] = jax_array_reshard_fn(
        array_info, donate=donate, **kwargs
    )

  result = [None] * len(flat_x)
  for arr, idx in zip(
      non_reshardable_arrays["arrays"], non_reshardable_arrays["indices"]
  ):
    result[idx] = arr
  for array_info in jax_arrays.values():
    for arr, idx in zip(array_info["arrays"], array_info["indices"]):
      result[idx] = arr

  return jax.tree.unflatten(tree_def, result)


def _ifrt_jax_array_reshard(
    array_info: Mapping[str, Any], *, donate: bool
) -> Sequence[jax.Array]:
  return pathwaysutils.jax.transfer_to_shardings(
      tuple(arr for arr in array_info["arrays"]),
      tuple(array_info["dst_shardings"]),
      donate,
  )


def reshard(
    x: Any,
    sharding: jax.sharding.Sharding | Any,
    *,
    donate: bool = False,
    may_alias: bool | None = None,
) -> Any:
  """Reshards `x` to `sharding`.

  Args:
    x: An array, scalar, or (nested) standard Python container thereof.
    sharding: A `Sharding` or a (nested) `Sharding` in standard Python container
      (must be a tree prefix of `x`), representing the device(s) and sharding to
      which `x` should be sharded to. The result will be committed to the
      device(s) of the sharding.
    donate: If `True`, donate all input arrays, which may reduce the amount of
      memory needed for resharding. Buffers donated to resharding should not be
      reused.
    may_alias: If `True`, may alias the input array with the output array. May
      reduce the amount of memory needed for resharding. Not used at the moment.

  Returns:
    A copy of `x` whose sharding is `sharding`.
  """
  return reshard_generic(
      x,
      sharding,
      donate=donate,
      may_alias=may_alias,
      jax_array_reshard_fn=_ifrt_jax_array_reshard,
  )
