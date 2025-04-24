# Copyright 2024 Google LLC
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
"""TypeHandlers supporting Pathways backend."""

import collections
from collections.abc import Sequence
import datetime
import functools
import logging
import typing

import jax
from orbax.checkpoint import future
from orbax.checkpoint import type_handlers
from pathwaysutils.persistence import helper


logger = logging.getLogger(__name__)

ParamInfo = type_handlers.ParamInfo
SaveArgs = type_handlers.SaveArgs
RestoreArgs = type_handlers.RestoreArgs
ArrayRestoreArgs = type_handlers.ArrayRestoreArgs


def extract_parent_dir_and_name(
    infos: Sequence[ParamInfo],
) -> tuple[Sequence[str], Sequence[str]]:
  """Extracts names and locations from ParamInfos."""
  parent_dirs = [str(info.parent_dir) for info in infos]
  names = [str(info.name) for info in infos]
  return parent_dirs, names


class CloudPathwaysArrayHandler(type_handlers.ArrayHandler):
  """A TypeHandler for array types when using Pathways."""

  def __init__(
      self,
      read_timeout: datetime.timedelta | None = None,
      use_ocdbt: bool = False,
  ):
    """Constructor.

    Args:
      read_timeout: Duration indicating the timeout for reading arrays
      use_ocdbt: allows using Tensorstore OCDBT driver.
    """
    self._read_timeout = read_timeout

    if use_ocdbt:
      raise ValueError("OCDBT not supported for Pathways.")
    super().__init__()

  async def _background_serialize(
      self,
      values: Sequence[jax.Array],
      locations: Sequence[str],
      names: Sequence[str],
  ) -> None:
    """Uses Pathways Persistence API to serialize a jax array."""
    f = functools.partial(helper.write_one_array, timeout=self._read_timeout)
    futures_results = list(map(f, locations, names, values))
    for future_result in futures_results:
      future_result.result()

  async def serialize(
      self,
      values: Sequence[jax.Array],
      infos: Sequence[ParamInfo],
      args: Sequence[SaveArgs] | None = None,
  ) -> Sequence[future.Future]:
    """Uses Pathways Persistence API to serialize a jax array."""
    type_handlers.check_input_arguments(values, infos, args)

    if any([arg.dtype is not None for arg in args]):
      raise ValueError("Casting during save not supported for Pathways.")

    locations, names = extract_parent_dir_and_name(infos)
    return [
        future.CommitFutureAwaitingContractedSignals(
            self._background_serialize(values, locations, names),
            name="cloud_pathways_array_handler",
        )
    ]

  async def deserialize(
      self,
      infos: Sequence[ParamInfo],
      args: Sequence[RestoreArgs] | None = None,
  ) -> Sequence[jax.Array]:
    """Uses Pathways Persistence API to deserialize a jax array."""
    if args is None:
      raise ValueError("Must provide ArrayRestoreArgs to restore as jax.Array.")
    type_handlers.check_input_arguments(infos, args)

    global_meshes = []
    mesh_axes = []
    global_shapes = []
    dtypes = []
    shardings = []

    should_open_metadata = False
    for arg in args:
      if not isinstance(arg, ArrayRestoreArgs):
        raise ValueError(
            "To restore jax.Array, provide ArrayRestoreArgs; found"
            f" {type(arg).__name__}"
        )
      arg = typing.cast(ArrayRestoreArgs, arg)
      if arg.sharding is None and (arg.mesh is None or arg.mesh_axes is None):
        raise ValueError(
            "Sharding of jax.Array cannot be None. Provide `mesh`"
            " and `mesh_axes` OR `sharding`."
        )
      if arg.sharding is None:
        global_meshes.append(arg.mesh)
        mesh_axes.append(arg.mesh_axes)
        shardings.append(
            jax.sharding.NamedSharding(mesh=arg.mesh, spec=arg.mesh_axes)
        )
      else:
        if not isinstance(arg.sharding, jax.sharding.NamedSharding):
          raise ValueError("Pathways only supports jax.sharding.NamedSharding.")
        sharding = typing.cast(jax.sharding.NamedSharding, arg.sharding)
        global_meshes.append(sharding.mesh)
        mesh_axes.append(sharding.spec)
        shardings.append(sharding)
      if arg.global_shape is None or arg.dtype is None:
        logger.warning(
            "Shape or dtype not provided for restoration. Provide these"
            " properties for improved performance."
        )
        should_open_metadata = True
      global_shapes.append(arg.global_shape)
      dtypes.append(arg.dtype)

    if should_open_metadata:
      metadatas = await self.metadata(infos)
      global_shapes = [
          m.shape if s is None else s for m, s in zip(metadatas, global_shapes)
      ]
      dtypes = [m.dtype if d is None else d for m, d in zip(metadatas, dtypes)]

    # Group inputs by global_mesh so that we can perform batched Array
    # construction for each global_mesh.
    inputs_by_global_mesh = collections.defaultdict(list)
    for i, global_mesh in enumerate(global_meshes):
      inputs_by_global_mesh[global_mesh].append(i)

    results = [None] * len(infos)

    for global_mesh, idxs in inputs_by_global_mesh.items():
      grouped_infos = [infos[idx] for idx in idxs]
      grouped_global_shapes = [global_shapes[idx] for idx in idxs]
      grouped_dtypes = [dtypes[idx] for idx in idxs]
      grouped_shardings = [shardings[idx] for idx in idxs]
      locations, names = extract_parent_dir_and_name(grouped_infos)
      grouped_arrays, read_future = helper.read_arrays(
          locations[0],
          names,
          grouped_dtypes,
          grouped_global_shapes,
          grouped_shardings,
          global_mesh.devices,
          timeout=self._read_timeout,
      )
      # each persistence call is awaited serially.
      read_future.result()
      for idx, arr in zip(idxs, grouped_arrays):
        results[idx] = arr
    return results  # pytype: disable=bad-return-type


def register_pathways_handlers(
    read_timeout: datetime.timedelta | None = None,
):
  """Function that must be called before saving or restoring with Pathways."""
  logger.debug(
      "Registering CloudPathwaysArrayHandler (Pathways Persistence API)."
  )
  type_handlers.register_type_handler(
      jax.Array,
      CloudPathwaysArrayHandler(
          read_timeout=read_timeout,
      ),
      override=True,
  )
