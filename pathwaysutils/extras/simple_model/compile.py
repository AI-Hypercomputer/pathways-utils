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
"""Mesh partitioning, step compilation, and parameter updates."""

from collections.abc import Sequence
import dataclasses
import functools
import math
from typing import Any

from flax import nnx
import jax
from jax import numpy as jnp
import numpy as np
from pathwaysutils.extras.simple_model import config as config_lib
from pathwaysutils.extras.simple_model import model as model_lib


def compute_mesh_shape(
    num_devices: int, devices_per_slice: int = 0
) -> tuple[int, int]:
  """Computes (dp_size, mp_size) factoring devices across slices and model parallelism."""
  slice_size = devices_per_slice if devices_per_slice > 0 else num_devices
  mp = math.gcd(num_devices, slice_size)
  return (num_devices // mp, mp)


@dataclasses.dataclass(frozen=True)
class CompiledStepArtifacts:
  """Artifacts from precompiling a sharded training step."""
  mesh: jax.sharding.Mesh
  data_sharding: jax.sharding.NamedSharding
  replicated_sharding: jax.sharding.NamedSharding
  param_shardings: Any
  abstract_params: Any
  compiled_step_fn: Any
  global_batch_size: int


def update_step(
    graphdef: nnx.GraphDef,
    vocab_size: int,
    learning_rate: float,
    state: Any,
    tokens: jax.Array,
) -> tuple[jax.Array, Any]:
  """Performs forward, loss, backward, and parameter update."""
  def loss_fn(st, toks):
    logits = nnx.merge(graphdef, st)(toks[:, :-1])
    one_hot = jax.nn.one_hot(toks[:, 1:], vocab_size)
    return jnp.mean(
        -jnp.sum(one_hot * jax.nn.log_softmax(logits, axis=-1), axis=-1)
    )

  loss, grads = jax.value_and_grad(loss_fn)(state, tokens)
  new_state = jax.tree.map(lambda p, g: p - learning_rate * g, state, grads)
  return loss, new_state


def compile_training_step(
    config: config_lib.ModelConfig,
    devices: Sequence[jax.Device],
    devices_per_slice: int = 0,
    learning_rate: float = 0.0005,
    mesh_shape: tuple[int, int] | None = None,
) -> tuple[nnx.GraphDef, CompiledStepArtifacts]:
  """Constructs the model, shardings, and compiles the sharded training step function."""
  dummy_model = model_lib.SimpleModel(config, rngs=nnx.Rngs(0))
  graphdef, state = nnx.split(dummy_model)
  raw_abstract_params = jax.tree.map(
      lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state
  )

  num_devices = len(devices)
  dp_size, mp_size = mesh_shape or compute_mesh_shape(
      num_devices, devices_per_slice
  )
  if dp_size * mp_size != num_devices:
    raise ValueError(
        f"Mesh shape ({dp_size}, {mp_size}) requires {dp_size * mp_size}"
        f" devices, but {num_devices} devices were provided."
    )

  dev_arr = np.array(devices).reshape((dp_size, mp_size))
  mesh = jax.sharding.Mesh(dev_arr, axis_names=("dp", "mp"))
  data_sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec("dp", None)
  )
  replicated_sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec()
  )
  mp_2d = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec(None, "mp")
  )

  def _param_sharding(p):
    return mp_2d if (p.ndim == 2 and mp_size > 1) else replicated_sharding

  abstract_params = jax.tree.map(
      lambda p: jax.ShapeDtypeStruct(
          p.shape, p.dtype, sharding=_param_sharding(p)
      ),
      raw_abstract_params,
  )
  param_shardings = jax.tree.map(lambda x: x.sharding, abstract_params)

  global_batch_size = config.per_device_batch_size * num_devices
  tokens_struct = jax.ShapeDtypeStruct(
      (global_batch_size, config.seq_len), jnp.int32, sharding=data_sharding
  )

  step_fn = functools.partial(
      update_step, graphdef, config.vocab_size, learning_rate
  )
  jitted_step = jax.jit(
      step_fn,
      in_shardings=(param_shardings, data_sharding),
      out_shardings=(replicated_sharding, param_shardings),
  )
  compiled_fn = jitted_step.lower(abstract_params, tokens_struct).compile()

  artifacts = CompiledStepArtifacts(
      mesh=mesh,
      data_sharding=data_sharding,
      replicated_sharding=replicated_sharding,
      param_shardings=param_shardings,
      abstract_params=abstract_params,
      compiled_step_fn=compiled_fn,
      global_batch_size=global_batch_size,
  )
  return graphdef, artifacts
