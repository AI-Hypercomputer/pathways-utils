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
"""Pathways JAX abstractions.

This introduces an abstrction layer some JAX APIs that have changed over
`pathwaysutils`'s compatibility window.
"""


import functools


class _FakeJaxFunction:
  """An object that raises an ImportError for __getattr__ and __call__.

  This is used to provide a placeholder for JAX functions that are not
  available in older versions of JAX, raising a helpful error message if they
  are inadvertently used.
  """

  def __init__(self, name, version):
    self.__name__ = name
    self.version = version
    self.error_message = (
        f"Function {self.__name__} does not exist until JAX {self.version}. "
        f"The current version of JAX is {jax.__version__}. "
        "Using this function results in this runtime error."
    )

  def __getattr__(self, name):
    raise ImportError(self.error_message)

  def __call__(self, *args, **kwargs):
    raise ImportError(self.error_message)


try:
  # jax>=0.8.0
  from jaxlib import _pathways  # pylint: disable=g-import-not-at-top

  split_by_mesh_axis = _pathways._split_by_mesh_axis
  del _pathways

except ImportError:
  # jax<0.8.0

  split_by_mesh_axis = _FakeJaxFunction(
      "jax.jaxlib._pathways._split_by_mesh_axis",
      "0.8.0",
  )


try:
  # jax>=0.8.3
  # The import may fail if the JAX version is not new enough.
  from jaxlib import _pathways as jaxlib_pathways  # pylint: disable=g-import-not-at-top

  transfer_to_shardings = jaxlib_pathways._transfer_to_shardings

  del jaxlib_pathways

except ImportError:
  # jax<0.8.3
  transfer_to_shardings = _FakeJaxFunction(
      "jax.jaxlib._pathways._transfer_to_shardings",
      "0.8.3",
  )


@functools.lru_cache(maxsize=1)
def ifrt_reshard_available() -> bool:
  """Checks if transfer_to_shardings is available."""
  try:
    import jax  # pylint: disable=g-import-not-at-top

    transfer_to_shardings(
        [jax.numpy.array([0])],
        [jax.sharding.SingleDeviceSharding(jax.devices()[0])],
    )

  except (ImportError, NameError, jax.errors.JaxRuntimeError):
    return False
  else:
    return True
  finally:
    del jax


del _FakeJaxFunction
del functools
