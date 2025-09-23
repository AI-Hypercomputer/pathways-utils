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
"""Experimental split by mesh axis API."""

from typing import Any, Sequence

import jax
from pathwaysutils import jax as pw_jax
from pathwaysutils import lru_cache


@lru_cache.lru_cache(maxsize=16384)
def _cached_named_sharding(
    mesh: jax.sharding.Mesh,
    spec: jax.sharding.PartitionSpec,
    memory_kind: str | None = None,
):
  return jax.sharding.NamedSharding(mesh, spec, memory_kind=memory_kind)


@lru_cache.lru_cache(maxsize=1024)
def _get_per_mesh_shardings(
    meshes: tuple[jax.sharding.Mesh, ...],
    spec: jax.sharding.PartitionSpec,
    memory_kind: str | None = None,
) -> Sequence[jax.sharding.NamedSharding]:
  """Returns per-mesh shardings."""
  return [
      _cached_named_sharding(mesh, spec, memory_kind=memory_kind)
      for mesh in meshes
  ]


def split_by_mesh_axis(
    arrays: Any,
    mesh_axis: str,
    mesh_axis_indices_or_sections: int | Sequence[int] | None = None,
    *,
    donate: bool = False,
) -> Sequence[Any]:
  """Splits arrays by a mesh axis, and returns arrays on each split mesh.

  Args:
    arrays: PyTree of JAX arrays with NamedSharding whose mesh is identical.
    mesh_axis: Mesh axis to split the arrays by.
    mesh_axis_indices_or_sections: If it is an integer, N, the mesh axis will be
      divided into N equal submeshes along `mesh_axis`. If it is a 1-D sequence,
      the entries indicate the boundary on the mesh axis along `mesh_axis`. For
      example, [2, 3] for splitting first mesh axis results in three output
      arrays (per each input array) on `mesh[:2], mesh[2:3], mesh[3:]`,
      respectively. If it is None, it will be the same as `N =
      mesh.axis_size[mesh.axis_names.index(mesh_axis)]`. Note: the sequence must
      be monotonoically increasing and should not contain the start or end
      boundaries.
    donate: Whether to donate input arrays. By default, input arrays are
      aliased.

  Returns:
     A sequence of PyTrees whose structure is the same as `arrays`.
     Each element `i` has arrays with their shards filtered out to match
     mesh corresponding mesh constructed according to
     `mesh_axis_indices_or_sections`. An array's shape remains the same if the
     array is replicated along `mesh_axis`, or is shrunk by a split factor
     computed from `mesh_axis_indices_or_sections` if the array is partitioned
     along `mesh_axis`.
  """
  flat_arrays, treedef = jax.tree.flatten(arrays)

  if not flat_arrays:
    return arrays

  sharding = flat_arrays[0].sharding
  if not isinstance(sharding, jax.sharding.NamedSharding):
    raise ValueError(f"Array must have a NamedSharding. Got {sharding=}")
  mesh = sharding.mesh
  mesh_axis_idx = mesh.axis_names.index(mesh_axis)
  sharded_dim_idxs = []
  for array in flat_arrays:
    sharding = array.sharding
    if not isinstance(sharding, jax.sharding.NamedSharding):
      raise ValueError(f"Array must have a NamedSharding. Got {sharding=}")
    if mesh != sharding.mesh:
      raise ValueError(
          f"Array sharding mesh must match, but got {mesh=}, {sharding.mesh=}"
      )
    if sharding._logical_device_ids is not None:  # pylint: disable=protected-access
      raise ValueError(
          "Array sharding's _logical_device_ids must be None, but got"
          f" {sharding._logical_device_ids=}"  # pylint: disable=protected-access
      )
    sharded_dim = -1
    for dim_idx, dim_spec in enumerate(sharding.spec):
      flat_dim_spec, _ = jax.tree.flatten(dim_spec)
      if mesh_axis in flat_dim_spec:
        sharded_dim = dim_idx
        break
    sharded_dim_idxs.append(sharded_dim)

  # Transform mesh_axis_indices_or_sections into a list of axis boundaries,
  # with the last entry being the size of the mesh_axis.
  if mesh_axis_indices_or_sections is None:
    # If mesh_axis_indices_or_sections is None, the arrays will be divided
    # along the mesh_axis.
    mesh_axis_indices_or_sections = mesh.axis_sizes[mesh_axis_idx]
  if isinstance(mesh_axis_indices_or_sections, int):
    # Expand the mesh_axis_indices_or_sections to a list indicating the
    # boundaries of mesh axis.
    if mesh.axis_sizes[mesh_axis_idx] % mesh_axis_indices_or_sections != 0:
      raise ValueError(
          "The size of the `mesh_axis` must be divisible by"
          " `mesh_axis_indices_or_sections`. Got"
          f" {mesh.axis_sizes[mesh_axis_idx]} and"
          f" {mesh_axis_indices_or_sections=}"
      )
    axis_size = mesh.axis_sizes[mesh_axis_idx] // mesh_axis_indices_or_sections
    mesh_axis_sections = list(
        range(axis_size, mesh.axis_sizes[mesh_axis_idx] + 1, axis_size)
    )
  else:
    mesh_axis_sections = mesh_axis_indices_or_sections
    for i, boundary in enumerate(mesh_axis_sections):
      if boundary <= 0 or boundary >= mesh.axis_sizes[mesh_axis_idx]:
        raise ValueError(
            "Mesh axis sections values must be in range (0,"
            f" axis_size={mesh.axis_sizes[mesh_axis_idx]}) to avoid an empty"
            f" section, but got {mesh_axis_sections=}."
        )
      if i > 0 and mesh_axis_sections[i] <= mesh_axis_sections[i - 1]:
        raise ValueError(
            "Mesh axis sections must be monotonically increasing, but got"
            f" {mesh_axis_sections=}."
        )
    mesh_axis_sections += [mesh.axis_sizes[mesh_axis_idx]]

  submeshes = []
  axis_boundary_start = 0
  slices = [slice(None)] * len(mesh.axis_sizes)
  for axis_boundary_end in mesh_axis_sections:
    slices[mesh_axis_idx] = slice(axis_boundary_start, axis_boundary_end)
    submeshes.append(
        jax.sharding.Mesh(mesh.devices[tuple(slices)], mesh.axis_names)
    )
    axis_boundary_start = axis_boundary_end

  submeshes_tuple = tuple(submeshes)
  submesh_shardings = [
      _get_per_mesh_shardings(
          submeshes_tuple, x.sharding.spec, x.sharding.memory_kind
      )
      for x in flat_arrays
  ]

  flat_split_arrays = pw_jax.jaxlib_pathways._split_by_mesh_axis(  # pylint: disable=protected-access
      arrays=flat_arrays,
      sharded_dim_idxs=sharded_dim_idxs,
      mesh_axis_sizes=mesh.axis_sizes,
      mesh_axis_idx=mesh_axis_idx,
      mesh_axis_sections=mesh_axis_sections,
      submesh_shardings=submesh_shardings,
      donate=donate,
  )

  # Convert the flat arrays to a list of a PyTree per submesh.
  outer_treedef = jax.tree.structure(["*"] * len(flat_split_arrays))
  inner_treedef = jax.tree.structure(["*"] * len(submeshes))
  return [
      jax.tree.unflatten(treedef, flat_submesh_arrays)
      for flat_submesh_arrays in jax.tree.transpose(
          outer_treedef, inner_treedef, flat_split_arrays
      )
  ]
