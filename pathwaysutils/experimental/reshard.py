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
from collections.abc import Callable, Mapping, Sequence
import json
import logging
from typing import Any

import jax
from pathwaysutils import lru_cache
from pathwaysutils import plugin_executable
from pathwaysutils import reshard as pw_reshard

# Redirects for generic intermediate sharding helpers and exceptions
NoIntermediateShardingError = pw_reshard.NoIntermediateShardingError
NoIntermediateShardingNeededError = pw_reshard.NoIntermediateShardingNeededError
find_intermediate_sharding = pw_reshard.find_intermediate_sharding
reshard_with_intermediate_sharding_generic = (
    pw_reshard.reshard_with_intermediate_sharding_generic
)

_logger = logging.getLogger(__name__)


def _identity(x: Any) -> Any:
  """A helper function that returns its input."""
  return x


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
    ) -> Mapping[str, Any]:
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


def _sidechannel_jax_array_reshard(
    array_info: Mapping[str, Any], *, donate: bool, cache_resharding_plans: bool
) -> Sequence[jax.Array]:
  get_resharding_plan_func = (
      _get_resharding_plan_cached
      if cache_resharding_plans
      else _get_resharding_plan
  )
  return get_resharding_plan_func(
      tuple(arr.aval for arr in array_info["arrays"]),
      tuple(arr.sharding for arr in array_info["arrays"]),
      tuple(array_info["dst_shardings"]),
      donate,
  ).execute(tuple(array_info["arrays"]))


def reshard(
    x: Any,
    sharding: jax.sharding.Sharding | Any,
    *,
    donate: bool = False,
    may_alias: bool | None = None,
    cache_resharding_plans: bool = False,
) -> Any:
  """Reshards `x` to `sharding`.

  This function is an alternative to `pathwaysutils.reshard` (which is the IFRT
  reshard) that uses the sidechannel MPMD resharding API for the final reshard.

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
      recreating plans for the same resharding operation.

  Returns:
    A copy of `x` whose sharding is `sharding`.
  """
  return pw_reshard.reshard_generic(
      x,
      sharding,
      donate=donate,
      may_alias=may_alias,
      jax_array_reshard_fn=_sidechannel_jax_array_reshard,
      cache_resharding_plans=cache_resharding_plans,
  )


def sidechannel_reshard_with_intermediate_sharding(
    x: Any,
    in_sharding: jax.sharding.Sharding,
    out_sharding: jax.sharding.Sharding,
    *,
    donate: bool = False,
    may_alias: bool | None = None,
    cache_resharding_plans: bool = False,
) -> Any:
  """Reshards `x` to `out_sharding`, using an intermediate sharding if possible.

  This function is an alternative to `reshard` that may be faster and sometimes
  essential for certain sharding combinations by using an intermediate sharding
  to avoid expensive all-gathers. If no beneficial intermediate sharding is
  found, it falls back to standard resharding. See `find_intermediate_sharding`
  for more details on when an intermediate sharding is used.

  Uses the sidechannel resharding API for the final reshard.

  Args:
    x: An array, scalar, or (nested) standard Python container thereof.
    in_sharding: The source sharding of `x`.
    out_sharding: The target sharding for `x`.
    donate: If `True`, donate all input arrays, which may reduce the amount of
      memory needed for resharding. Buffers donated to resharding should not be
      reused.
    may_alias: If `True`, may alias the input array with the output array. May
      reduce the amount of memory needed for resharding. Not used at the moment.
    cache_resharding_plans: If `True`, uses a resharding plan cache to avoid
      recreating plans for the same resharding operation.

  Returns:
    A copy of `x` whose sharding is `out_sharding`.
  """
  return reshard_with_intermediate_sharding_generic(
      x,
      in_sharding,
      out_sharding,
      donate=donate,
      may_alias=may_alias,
      jax_array_reshard_fn=_sidechannel_jax_array_reshard,
      cache_resharding_plans=cache_resharding_plans,
  )
