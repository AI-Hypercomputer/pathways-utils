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
"""Pathwaysutils JAX concatenate_by_mesh_axis."""

from typing import Any, Sequence

import jax
import numpy as np

from pathwaysutils import jax as pw_jax


def concatenate_by_mesh_axis(
    array_trees: Sequence[Any],
    mesh_axis: str,
) -> Any:
  """Concatenates meshes by an axis. Returns arrays on the concatenated mesh.

  Note: This API donates the given arrays.

  Args:
    array_trees: Sequence of PyTrees of JAX arrays with `NamedSharding`. All
      PyTrees in the sequence have the same structure. All arrays in each PyTree
      are sharded/replicated on the same mesh.
    mesh_axis: Mesh axis to concatenate.

  Returns:
    A PyTree with the same structure as `array_trees[i]`. It has arrays with
    their shards concatenated to match a concatenated mesh.
  """
  if not array_trees:
    return array_trees

  def _get_named_sharding(array: jax.Array) -> jax.sharding.NamedSharding:
    if not isinstance(array, jax.Array):
      raise ValueError(f"Elements must be jax.Array. Got {type(array)}")
    sharding = array.sharding
    if not isinstance(sharding, jax.sharding.NamedSharding):
      raise ValueError(f"Expected NamedSharding. Got {type(sharding)}")
    return sharding

  flattened_arrays = []
  input_treedefs = None
  for array_tree in array_trees:
    flat, treedef = jax.tree_util.tree_flatten(array_tree)
    flattened_arrays.append(flat)
    if input_treedefs is not None:
      if treedef != input_treedefs:
        raise ValueError(
            "All array trees must have the same treedef. Got"
            f" {treedef} vs. {input_treedefs}"
        )
    else:
      input_treedefs = treedef

  # Convert to have the output array structure in the outer list, and each entry
  # be a list of arrays from each shard for the concatenated output array.
  input_flat_arrays = list(zip(*flattened_arrays))

  # Extract the shared mesh from each PyTree (from an arbitrary array in each).
  meshes_to_concatenate = [
      _get_named_sharding(array).mesh for array in input_flat_arrays[0]
  ]

  # Validate that the meshes are compatible.
  reference_mesh = meshes_to_concatenate[0]
  mesh_axis_idx = reference_mesh.axis_names.index(mesh_axis)
  for mesh in meshes_to_concatenate:
    if mesh.axis_names != reference_mesh.axis_names:
      raise ValueError(
          "Meshes must have the same axis names. Got"
          f" {mesh} vs. {reference_mesh}."
      )
    if (
        mesh.axis_sizes[:mesh_axis_idx]
        != reference_mesh.axis_sizes[:mesh_axis_idx]
        or mesh.axis_sizes[mesh_axis_idx + 1 :]
        != reference_mesh.axis_sizes[mesh_axis_idx + 1 :]
    ):
      raise ValueError(
          "Arrays must have the same mesh axis sizes for all axes except"
          f" {mesh_axis}. Got {mesh} vs. {reference_mesh}."
      )

  # Construct list of the mesh axis section boundaries.
  devices = []
  mesh_axis_sections = []
  cur_mesh_axis_size = 0
  for mesh in meshes_to_concatenate:
    devices.append(mesh.devices)
    cur_mesh_axis_size += mesh.axis_sizes[mesh_axis_idx]
    mesh_axis_sections.append(cur_mesh_axis_size)

  concatenated_mesh = jax.sharding.Mesh(
      np.concatenate(devices, mesh_axis_idx),
      axis_names=reference_mesh.axis_names,
      axis_types=reference_mesh.axis_types,
  )

  def _get_output_sharding(
      arrays: Sequence[jax.Array],
  ) -> jax.sharding.NamedSharding:
    reference_sharding = _get_named_sharding(arrays[0])
    reference_spec = reference_sharding.spec
    return jax.sharding.NamedSharding(concatenated_mesh, reference_spec)

  def _sharded_dim_idx_for_sharding(
      sharding: jax.sharding.NamedSharding,
  ) -> int:
    sharded_dim_idx = -1
    for dim_idx, dim_spec in enumerate(sharding.spec):
      flat_dim_spec, _ = jax.tree_util.tree_flatten(dim_spec)
      if mesh_axis in flat_dim_spec:
        sharded_dim_idx = dim_idx
        break
    return sharded_dim_idx

  out_shardings = [_get_output_sharding(arrays) for arrays in input_flat_arrays]
  sharded_dim_idxs = [
      _sharded_dim_idx_for_sharding(sharding) for sharding in out_shardings
  ]

  flat_output_arrays = pw_jax.concatenate_by_mesh_axis(
      arrays=input_flat_arrays,
      sharded_dim_idxs=sharded_dim_idxs,
      mesh_axis_sizes=concatenated_mesh.axis_sizes,
      mesh_axis_idx=mesh_axis_idx,
      mesh_axis_sections=mesh_axis_sections,
      out_shardings=out_shardings,
      donate=True,
  )

  return jax.tree_util.tree_unflatten(input_treedefs, flat_output_arrays)
