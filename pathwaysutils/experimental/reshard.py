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
"""Experimental resharding API for elastic device sets."""

import base64
import collections
import json
from typing import Any, Dict, Sequence

import jax
from pathwaysutils import lru_cache
from pathwaysutils import plugin_executable


class ReshardingPlanWrapper:
  """Wrapper around PluginProgram(reshard_request)."""

  _plugin_executable: plugin_executable.PluginExecutable
  _avals: Sequence[jax.core.ShapedArray]
  _out_shardings: Sequence[jax.sharding.Sharding]

  def __init__(
      self,
      avals: Sequence[jax.core.ShapedArray],
      source_shardings: Sequence[jax.sharding.Sharding],
      destination_shardings: Sequence[jax.sharding.Sharding],
      donate: bool,
  ):
    def ifrt_hlo_sharding(
        aval: jax.core.ShapedArray, sharding: jax.sharding.Sharding
    ) -> Dict[str, Any]:
      result = {
          "devices": {
              "device_ids": [
                  device.id for device in sharding._addressable_device_assignment  # pylint: disable=protected-access
              ]
          },
          "xla_hlo_sharding": (
              base64.b64encode(
                  sharding._to_xla_hlo_sharding(aval.ndim)  # pylint: disable=protected-access
                  .to_proto()
                  .SerializeToString()
              ).decode("utf-8")
          ),
      }
      if sharding.memory_kind is not None:
        result["memory_kind"] = sharding.memory_kind
      return result

    request = {
        "reshardRequest": {
            "donateInput": donate,
            "inSharding": [
                ifrt_hlo_sharding(aval, old_sharding)
                for aval, old_sharding in zip(avals, source_shardings)
            ],
            "outSharding": [
                ifrt_hlo_sharding(aval, new_sharding)
                for aval, new_sharding in zip(avals, destination_shardings)
            ],
        }
    }

    self._plugin_executable = plugin_executable.PluginExecutable(
        json.dumps(request)
    )
    self._avals = avals
    self._out_shardings = destination_shardings

  def execute(self, inp_arrays: tuple[jax.Array, ...]) -> Sequence[jax.Array]:
    out_arrays, fut = self._plugin_executable.call(
        inp_arrays, self._out_shardings, self._avals
    )
    fut.result()
    return out_arrays


def _get_resharding_plan(
    avals: tuple[jax.core.ShapedArray, ...],
    old_shardings: tuple[jax.sharding.Sharding, ...],
    new_shardings: tuple[jax.sharding.Sharding, ...],
    donate: bool,
) -> ReshardingPlanWrapper:
  """Returns a resharding plan for the given sharding task."""
  return ReshardingPlanWrapper(avals, old_shardings, new_shardings, donate)


_get_resharding_plan_cached = lru_cache.lru_cache()(_get_resharding_plan)


def reshard(
    x: Any,
    sharding: jax.sharding.Sharding | Any,
    *,
    donate: bool = False,
    may_alias: bool | None = None,  # pylint: disable=unused-argument
    cache_resharding_plans: bool = False,
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
    cache_resharding_plans: If `True`, uses a resharding plan cache to avoid
      recreating plans for the same resharding operation. May improve
      performance for use cases where the same resharding operation is done many
      times. May degrade performance if most reshardings operations are
      different, since the cache will cause Pathways Components to remain loaded
      for each cached plan. `False` by default.

  Returns:
    A copy of `x` whose sharding is `sharding`.
  """
  flat_x, tree_def = jax.tree.flatten(x)
  flat_sharding = jax.api_util.flatten_axes(
      "reshard sharding", tree_def, sharding
  )

  # We must split the arrays into two groups:
  # 1. jax.Array
  # 2. non jax.Array
  # For jax.Array, we will use the ifrt client to get the resharding plan and
  # execute it.
  # These arrays must be further split into groups based on the device set of
  # the sharding, since plugin programs only supports execution on the same
  # device set.
  # For non jax.Array, we will use jax.device_put to put the array to the
  # destination devices.
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
    get_resharding_plan_func = (
        _get_resharding_plan_cached
        if cache_resharding_plans
        else _get_resharding_plan
    )
    array_info["arrays"] = get_resharding_plan_func(
        tuple(arr.aval for arr in array_info["arrays"]),
        tuple(arr.sharding for arr in array_info["arrays"]),
        tuple(array_info["dst_shardings"]),
        donate,
    ).execute(tuple(array_info["arrays"]))

  result = [None] * len(flat_x)
  for arr, idx in zip(
      non_reshardable_arrays["arrays"], non_reshardable_arrays["indices"]
  ):
    result[idx] = arr
  for array_info in jax_arrays.values():
    for arr, idx in zip(array_info["arrays"], array_info["indices"]):
      result[idx] = arr

  return jax.tree.unflatten(tree_def, result)
