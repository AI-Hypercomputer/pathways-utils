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
"""Flax NNX Transformer architecture and activation functions."""

from flax import nnx
import jax
from jax import numpy as jnp
from pathwaysutils.extras.simple_model import config as config_lib


def geglu(x: jax.Array) -> jax.Array:
  """Gated GELU activation function."""
  gate, up = jnp.split(x, 2, axis=-1)
  return jax.nn.gelu(gate) * up


class TransformerBlock(nnx.Module):
  """A single Transformer decoder block built with Flax NNX."""

  def __init__(
      self,
      hidden_size: int,
      intermediate_size: int,
      num_heads: int,
      head_dim: int,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_heads, self.head_dim = num_heads, head_dim
    self.rmsnorm_1 = nnx.RMSNorm(num_features=hidden_size, rngs=rngs)
    self.attn_q = nnx.Linear(
        hidden_size, num_heads * head_dim, use_bias=False, rngs=rngs
    )
    self.attn_k = nnx.Linear(
        hidden_size, num_heads * head_dim, use_bias=False, rngs=rngs
    )
    self.attn_v = nnx.Linear(
        hidden_size, num_heads * head_dim, use_bias=False, rngs=rngs
    )
    self.attn_o = nnx.Linear(
        num_heads * head_dim, hidden_size, use_bias=False, rngs=rngs
    )
    self.rmsnorm_2 = nnx.RMSNorm(num_features=hidden_size, rngs=rngs)
    self.mlp_gate_up = nnx.Linear(
        hidden_size, intermediate_size * 2, use_bias=False, rngs=rngs
    )
    self.mlp_down = nnx.Linear(
        intermediate_size, hidden_size, use_bias=False, rngs=rngs
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    h = self.rmsnorm_1(x)
    scores = (
        jnp.einsum("btd,bsd->bts", self.attn_q(h), self.attn_k(h))
        / jnp.sqrt(self.head_dim)
    )
    attn = jnp.einsum(
        "bts,bsd->btd", jax.nn.softmax(scores, axis=-1), self.attn_v(h)
    )
    x = x + self.attn_o(attn)
    return x + self.mlp_down(geglu(self.mlp_gate_up(self.rmsnorm_2(x))))


class SimpleModel(nnx.Module):
  """Flax NNX Transformer language model with tied embeddings."""

  def __init__(self, config: config_lib.ModelConfig, *, rngs: nnx.Rngs):
    self.config = config
    self.embedder = nnx.Embed(
        num_embeddings=config.vocab_size,
        features=config.hidden_size,
        rngs=rngs,
    )
    self.layers = nnx.List([
        TransformerBlock(
            config.hidden_size,
            config.intermediate_size,
            config.num_heads,
            config.head_dim,
            rngs=rngs,
        )
        for _ in range(config.num_layers)
    ])
    self.output_scale = nnx.Param(jnp.array(1.0, dtype=jnp.float32))

  def __call__(self, tokens: jax.Array) -> jax.Array:
    x = self.embedder(tokens)
    for layer in self.layers:
      x = layer(x)
    return jnp.dot(x, self.embedder.embedding[...].T) * self.output_scale[...]
