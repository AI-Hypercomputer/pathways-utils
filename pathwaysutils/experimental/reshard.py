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
import logging
import math
import operator
from typing import Any, Callable, Dict, Mapping, Sequence
import warnings

import jax
from pathwaysutils import jax as pw_jax
from pathwaysutils import lru_cache
from pathwaysutils import plugin_executable
from pathwaysutils.experimental import split_by_mesh_axis


_logger = logging.getLogger(__name__)

INTERMEDIATE_SPLIT_SUFFIX = "_intermediate_split"
INTERMEDIATE_REPLICA_SUFFIX = "_intermediate_replica"


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


def _reshard(
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


def _ifrt_jax_array_reshard(
    array_info: Mapping[str, Any], *, donate: bool
) -> Sequence[jax.Array]:
  return pw_jax.transfer_to_shardings(
      tuple(arr for arr in array_info["arrays"]),
      tuple(array_info["dst_shardings"]),
      donate,
  )


def _reshard_with_sidechannel(
    x: Any,
    sharding: jax.sharding.Sharding | Any,
    *,
    donate: bool,
    may_alias: bool | None,
    cache_resharding_plans: bool,
) -> Any:
  """Reshards `x` to `sharding` using sidechannel."""
  return _reshard(
      x,
      sharding,
      donate=donate,
      may_alias=may_alias,
      jax_array_reshard_fn=_sidechannel_jax_array_reshard,
      cache_resharding_plans=cache_resharding_plans,
  )


def _reshard_with_ifrt(
    x: Any,
    sharding: jax.sharding.Sharding | Any,
    *,
    donate: bool,
    may_alias: bool | None,
) -> Any:
  """Reshards `x` to `sharding` using IFRT.

  Note: Resharding plan caching is not applicable to the IFRT implementation
  and is not supported by this function.

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
  return _reshard(
      x,
      sharding,
      donate=donate,
      may_alias=may_alias,
      jax_array_reshard_fn=_ifrt_jax_array_reshard,
  )


def reshard(
    x: Any,
    sharding: jax.sharding.Sharding | Any,
    *,
    donate: bool = False,
    may_alias: bool | None = None,
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
      performance for use cases where the same resharding operation is done
      many times. May degrade performance if most reshardings operations are
      different, since the cache will cause Pathways Components to remain
      loaded for each cached plan. `False` by default. This parameter is only
      used when `pw_jax.ifrt_reshard_available()` is false.

  Returns:
    A copy of `x` whose sharding is `sharding`.
  """
  if pw_jax.ifrt_reshard_available():
    if cache_resharding_plans:
      warnings.warn(
          "`cache_resharding_plans` is only applicable when using the"
          " sidechannel resharding implementation, but IFRT resharding is"
          " available and will be used. The `cache_resharding_plans` argument"
          " will be ignored."
      )
    return _reshard_with_ifrt(
        x,
        sharding,
        donate=donate,
        may_alias=may_alias,
    )
  else:
    return _reshard_with_sidechannel(
        x,
        sharding,
        donate=donate,
        may_alias=may_alias,
        cache_resharding_plans=cache_resharding_plans,
    )


class NoIntermediateShardingError(Exception):
  """Raised when no intermediate sharding is found."""


class NoIntermediateShardingNeededError(NoIntermediateShardingError):
  """Raised when no intermediate sharding is needed for optimization."""


def _get_sharding_spec_dims(sharding: jax.sharding.NamedSharding) -> list[int]:
  """Gets the sharding dimension sizes from a NamedSharding."""
  mesh = sharding.mesh
  dims = []
  for spec in sharding.spec:
    if spec is None:
      dims.append(1)
    elif isinstance(spec, str):
      dims.append(mesh.shape[spec])
    elif isinstance(spec, (list, tuple)):
      dims.append(math.prod([mesh.shape[ax] for ax in spec]))
    else:
      raise ValueError(f"Unsupported partition spec: {spec}")
  return dims


def _check_sharding_divisibility(
    in_sharding: jax.sharding.NamedSharding,
    out_sharding: jax.sharding.NamedSharding,
    src_dims: Sequence[int],
    dst_dims: Sequence[int],
):
  """Checks if source and destination shardings are compatible for optimization."""
  src_largest_dim = max(src_dims) if src_dims else 1
  dst_largest_dim = max(dst_dims) if dst_dims else 1
  src_total_dims = math.prod(src_dims)
  dst_total_dims = math.prod(dst_dims)

  # Not able to handle resharding with undividable shardings.
  if src_largest_dim % dst_largest_dim != 0:
    raise NoIntermediateShardingError(
        "Resharding with undividable shardings is not optimized with"
        " intermediate sharding."
        f" in_sharding={in_sharding}, out_sharding={out_sharding}"
    )
  if src_total_dims <= dst_total_dims:
    raise NoIntermediateShardingError(
        "No intermediate sharding is found because the source sharding is not"
        " larger than the target sharding."
        f" in_sharding={in_sharding}, out_sharding={out_sharding}"
    )
  if src_total_dims % dst_total_dims != 0:
    raise NoIntermediateShardingError(
        "No intermediate sharding is found because the source sharding is not"
        " divisible by the target sharding."
        f" in_sharding={in_sharding}, out_sharding={out_sharding}"
    )


def _get_split_candidates(
    in_sharding: jax.sharding.NamedSharding,
    src_dims: Sequence[int],
    dst_dims: Sequence[int],
    gcd_shards: Sequence[int],
) -> list[tuple[int, str]]:
  """Finds dimensions that are candidates for splitting."""
  split_candidates = []
  for i, spec in enumerate(in_sharding.spec):
    # TODO(b/473889684) - Support splitting an axis that is partitioned over
    # multiple mesh axes.
    if (
        gcd_shards[i] == 1
        and src_dims[i] > dst_dims[i]
        and isinstance(spec, str)
    ):
      split_candidates.append((i, spec))

  if not split_candidates:
    raise NoIntermediateShardingError(
        "No intermediate sharding is found because all of the"
        " gcd(src_dim_shards, dst_dim_shards) are 1s, or no suitable"
        " dimension to split."
    )
  return split_candidates


def _build_intermediate_mesh_and_spec(
    src_mesh: jax.sharding.Mesh,
    in_spec: jax.sharding.PartitionSpec,
    src_dims: Sequence[int],
    dst_dims: Sequence[int],
    split_candidates: list[tuple[int, str]],
) -> tuple[jax.sharding.Mesh, jax.sharding.PartitionSpec, list[str]]:
  """Builds the intermediate Mesh and PartitionSpec."""
  # Build a map of mesh axis to split information: (dim_idx, replicas)
  mesh_axis_to_split_info = {}
  for dim_idx, mesh_axis in split_candidates:
    src_dim = src_dims[dim_idx]
    dst_dim = dst_dims[dim_idx]
    replicas = src_dim // dst_dim
    mesh_axis_to_split_info[mesh_axis] = (dim_idx, replicas)

  # Build the intermediate mesh by expanding axes that need splitting.
  new_replicated_axis_names = []
  new_replicated_mesh_shape = []
  new_axis_names = []
  new_mesh_shape = []
  for axis_name in src_mesh.axis_names:
    axis_size = src_mesh.shape[axis_name]
    if axis_name in mesh_axis_to_split_info:
      dim_idx, replicas = mesh_axis_to_split_info[axis_name]
      dst_dim = dst_dims[dim_idx]
      split_axis_name = axis_name + INTERMEDIATE_SPLIT_SUFFIX
      replica_axis_name = axis_name + INTERMEDIATE_REPLICA_SUFFIX
      new_replicated_axis_names.append(replica_axis_name)
      new_replicated_mesh_shape.append(replicas)
      new_axis_names.append(split_axis_name)
      new_mesh_shape.append(dst_dim)
    else:
      new_axis_names.append(axis_name)
      new_mesh_shape.append(axis_size)

  final_axis_names = new_replicated_axis_names + new_axis_names
  final_mesh_shape = new_replicated_mesh_shape + new_mesh_shape
  intermediate_mesh = jax.sharding.Mesh(
      src_mesh.devices.reshape(final_mesh_shape),
      axis_names=tuple(final_axis_names),
  )

  # Build the intermediate PartitionSpec.
  intermediate_spec_list = list(in_spec)
  for dim_idx, mesh_axis in split_candidates:
    split_axis_name = mesh_axis + INTERMEDIATE_SPLIT_SUFFIX
    intermediate_spec_list[dim_idx] = split_axis_name
  intermediate_spec = jax.sharding.PartitionSpec(*intermediate_spec_list)

  return intermediate_mesh, intermediate_spec, new_replicated_axis_names


def find_intermediate_sharding(
    in_sharding: jax.sharding.Sharding, out_sharding: jax.sharding.Sharding
) -> tuple[jax.sharding.NamedSharding, list[str]]:
  """Finds an intermediate sharding to reshard to before target sharding.

  This function tries to find an intermediate sharding that can be used to
  reshard the in_sharding to the out_sharding. This is useful when resharding
  from an in_sharding to an out_sharding that requires an all-gather, which can
  be expensive.

  For example, consider resharding an array from in_sharding (e.g., [fsdp: 8,
  tp: 1]) to out_sharding (e.g., [fsdp: 1, tp: 4]). In this case, the source
  has a larger sharding factor, 8, than the target's largest sharding factor, 4.
  To avoid an expensive all-gather, we introduce an intermediate sharding, e.g.,
  [fsdp_split: 4, tp: 1, fsdp_replica: 2]). This intermediate sharding allows
  resharding the source array by sharding along the fsdp dimension and
  replicating it on the remaining devices. Then we can reshard any replica of
  the source to the target as normal.

  Args:
    in_sharding: The source sharding.
    out_sharding: The target sharding.

  Returns:
    A tuple containing:
      - An intermediate sharding.
      - A list of axis names that are replicated in the intermediate sharding.

  Raises:
    NoIntermediateShardingError: If no intermediate sharding is found.
    NoIntermediateShardingNeededError: If no intermediate sharding is needed for
      optimization.
  """

  if not isinstance(in_sharding, jax.sharding.NamedSharding) or not isinstance(
      out_sharding, jax.sharding.NamedSharding
  ):
    raise NoIntermediateShardingError(
        "Only NamedSharding is supported for now. Got"
        f" in_sharding={in_sharding} and out_sharding={out_sharding}"
    )

  if not in_sharding.spec and out_sharding.spec:
    in_sharding = jax.sharding.NamedSharding(
        in_sharding.mesh,
        jax.sharding.PartitionSpec(*[None] * len(out_sharding.spec)),
        memory_kind=in_sharding.memory_kind,
    )
  elif not out_sharding.spec and in_sharding.spec:
    out_sharding = jax.sharding.NamedSharding(
        out_sharding.mesh,
        jax.sharding.PartitionSpec(*[None] * len(in_sharding.spec)),
        memory_kind=out_sharding.memory_kind,
    )

  if len(in_sharding.spec) != len(out_sharding.spec):
    raise NoIntermediateShardingError(
        "Source and destination shardings must have the same rank (same"
        f" PartitionSpec length). Got in_sharding.spec={in_sharding.spec} and"
        f" out_sharding.spec={out_sharding.spec}"
    )

  src_dims = _get_sharding_spec_dims(in_sharding)
  dst_dims = _get_sharding_spec_dims(out_sharding)

  _check_sharding_divisibility(in_sharding, out_sharding, src_dims, dst_dims)

  gcd_shards = jax.tree.map(math.gcd, src_dims, dst_dims)

  # If all of the gcd(src_dim_shards, dst_dim_shards) are 1s, an all-gather is
  # needed as the single replica of the source cannot be presented by any
  # sharded form on the target devices.
  if jax.tree.reduce(operator.mul, gcd_shards, 1) != 1:
    raise NoIntermediateShardingNeededError()

  try:
    split_candidates = _get_split_candidates(
        in_sharding, src_dims, dst_dims, gcd_shards
    )
  except NoIntermediateShardingError as e:
    raise NoIntermediateShardingError(
        f"{e} in_sharding={in_sharding}, out_sharding={out_sharding}"
    ) from e

  intermediate_mesh, intermediate_spec, replicated_axes = (
      _build_intermediate_mesh_and_spec(
          in_sharding.mesh,
          in_sharding.spec,
          src_dims,
          dst_dims,
          split_candidates,
      )
  )

  intermediate_sharding = jax.sharding.NamedSharding(
      intermediate_mesh,
      intermediate_spec,
      memory_kind=in_sharding.memory_kind,
  )
  return intermediate_sharding, replicated_axes


def reshard_with_intermediate_sharding(
    x: Any,
    in_sharding: jax.sharding.Sharding,
    out_sharding: jax.sharding.Sharding,
    *,
    donate: bool = False,
    may_alias: bool | None = None,  # pylint: disable=unused-argument
) -> Any:
  """Reshards `x` to `out_sharding`, using an intermediate sharding if possible.

  This function is an alternative to `reshard` that may be faster and sometime
  essential for certain sharding combinations by using an intermediate sharding
  to avoid expensive all-gathers. If no beneficial intermediate sharding is
  found, it falls back to standard resharding. See `find_intermediate_sharding`
  for more details on when an intermediate sharding is used.

  Args:
    x: An array, scalar, or (nested) standard Python container thereof.
    in_sharding: The source sharding of `x`.
    out_sharding: The target sharding for `x`.
    donate: If `True`, donate all input arrays, which may reduce the amount of
      memory needed for resharding. Buffers donated to resharding should not be
      reused.
    may_alias: If `True`, may alias the input array with the output array. May
      reduce the amount of memory needed for resharding. Not used at the moment.

  Returns:
    A copy of `x` whose sharding is `out_sharding`.
  """

  try:
    intermediate_sharding, replicated_axes_names = find_intermediate_sharding(
        in_sharding, out_sharding
    )
  except NoIntermediateShardingError as e:
    _logger.debug("No intermediate sharding needed or found. %s", e)
    x_to_reshard = x
  else:
    x_to_reshard = jax.jit(
        _identity,
        out_shardings=intermediate_sharding,
    )(x)
    for split_axis in replicated_axes_names:
      x_to_reshard, *_ = split_by_mesh_axis.split_by_mesh_axis(
          x_to_reshard,
          split_axis,
          donate=donate,
      )

  return reshard(x_to_reshard, out_sharding, donate=donate, may_alias=may_alias)
