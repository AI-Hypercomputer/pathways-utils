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
"""Resharding API tests."""

from absl import flags
import jax
import jax.numpy as jnp
import numpy as np
from pathwaysutils.google_internal.elastic import reshard

from google3.learning.pathways.ifrt.proxy.jax.tests import register_jax_grpc_backend_for_testing  # pylint: disable=unused-import
from absl.testing import absltest
from absl.testing import parameterized


P = jax.sharding.PartitionSpec


class ReshardTest(parameterized.TestCase):

  def test_devices_can_be_fetched_from_proxy_backend(self):
    devices = jax.devices("proxy")
    self.assertNotEmpty(devices)

  @parameterized.product(
      source_array=[
          "np_ones",
          "jnp_ones",
          "device_put",
          "execute_with_out_shardings",
          "execute_without_out_shardings",
      ],
      array_type=["1d", "2d_1", "2d_2"],
  )
  def test_two_mesh_transfer(self, source_array: str, array_type: str):
    topology = {
        "mesh1": jax.sharding.Mesh(np.array(jax.devices()[0:4]), ("replica",)),
        "mesh2": jax.sharding.Mesh(np.array(jax.devices()[4:8]), ("replica",)),
    }
    shardings = {
        "y": jax.sharding.NamedSharding(topology["mesh1"], P("replica")),
        "z": jax.sharding.NamedSharding(topology["mesh2"], P("replica")),
    }

    def transfer(x):
      y = reshard.reshard(x, shardings["y"], donate_input=False)
      z = reshard.reshard(x, shardings["z"], donate_input=False)
      return y, z

    if array_type == "1d":
      make_array = lambda: np.ones([32])
    elif array_type == "2d_1":
      make_array = lambda: np.ones([4, 8])
    elif array_type == "2d_2":
      make_array = lambda: np.ones([16, 2])
    else:
      self.fail("Unknown array type")
    self.assertIsNotNone(make_array)

    if source_array == "np_ones":
      x = make_array()
    elif source_array == "jnp_ones":
      x = jnp.asarray(make_array())
    elif source_array == "device_put":
      x = jax.device_put(make_array(), jax.devices()[0])
    elif source_array == "execute_with_out_shardings":
      x = jax.jit(
          make_array,
          out_shardings=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
      )()
    elif source_array == "execute_without_out_shardings":
      x = jax.jit(
          make_array,
          out_shardings=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
      )()
      x = jax.jit(lambda x: x + 1)(x)
    else:
      self.fail("Unknown source array")

    self.assertIsNotNone(x)
    y, z = transfer(x)
    np.testing.assert_array_equal(y, z)

  def test_scalar_replication(self):
    in_mesh = jax.sharding.Mesh(np.array(jax.devices()[0:1]), ("replica",))
    out_mesh = jax.sharding.Mesh(
        np.array(list(reversed(jax.devices()[0:2]))), ("replica",)
    )
    in_sharding = jax.sharding.NamedSharding(in_mesh, P())
    out_sharding = jax.sharding.NamedSharding(out_mesh, P())

    x_copy = jax.device_put(np.array(1), in_sharding)
    x = jax.device_put(np.array(1), in_sharding)
    y = reshard.reshard(x, out_sharding, donate_input=True)

    np.testing.assert_array_equal(x_copy, y)
    for s in y.addressable_shards:
      np.testing.assert_array_equal(x_copy, s.data)

  def test_reshard_multiple_arrays_with_different_shardings(self):
    mesh1 = jax.sharding.Mesh(
        np.array(jax.devices()[0:4]).reshape((2, 2)), ("data", "model")
    )
    mesh2 = jax.sharding.Mesh(
        np.array(jax.devices()[0:8]).reshape((4, 2)), ("data", "model")
    )

    sharding_1_a = jax.sharding.NamedSharding(mesh1, P("data"))
    sharding_1_b = jax.sharding.NamedSharding(mesh1, P("model"))

    sharding_2_a = jax.sharding.NamedSharding(mesh2, P("data"))
    sharding_2_b = jax.sharding.NamedSharding(mesh2, P("model"))

    a = jax.device_put(jnp.arange(32), sharding_1_a)
    b = jax.device_put(jnp.arange(32), sharding_1_b)

    # mesh1 -> mesh2
    result = reshard.reshard([a, [b]], [sharding_2_a, [sharding_2_b]])
    result_a = result[0]
    result_b = result[1][0]

    self.assertEqual(result_a.sharding, sharding_2_a)
    self.assertEqual(result_b.sharding, sharding_2_b)
    np.testing.assert_array_equal(result_a, a)
    np.testing.assert_array_equal(result_b, b)

    # mesh2 -> mesh1
    result = reshard.reshard(
        [result_a, [result_b]], [sharding_1_a, [sharding_1_b]]
    )
    result_a = result[0]
    result_b = result[1][0]

    self.assertEqual(result_a.sharding, sharding_1_a)
    self.assertEqual(result_b.sharding, sharding_1_b)
    np.testing.assert_array_equal(result_a, a)
    np.testing.assert_array_equal(result_b, b)

  @parameterized.product(
      memory_kind=["device", "pinned_host"],
      from_slice=[slice(0, 4), slice(0, 8)],
      to_slice=[slice(0, 4), slice(0, 8)],
  )
  def test_reshard_on_pinned_host(
      self, memory_kind: str, from_slice: slice, to_slice: slice
  ):
    mesh_from = jax.sharding.Mesh(jax.devices()[from_slice], "replica")
    mesh_to = jax.sharding.Mesh(jax.devices()[to_slice], "replica")

    sharding_from = jax.sharding.NamedSharding(
        mesh_from, P(), memory_kind=memory_kind
    )
    sharding_to = jax.sharding.NamedSharding(
        mesh_to, P(), memory_kind=memory_kind
    )

    a = jax.device_put(jnp.arange(32), sharding_from)
    result_a = reshard.reshard(a, sharding_to, donate_input=False)

    self.assertEqual(result_a.sharding.memory_kind, memory_kind)
    self.assertEqual(result_a.sharding, sharding_to)
    np.testing.assert_array_equal(result_a, a)


if __name__ == "__main__":
  # set JAX_ALLOW_UNUSED_TPUS to avoid the error below
  #
  # AssertionError: The host has 4 TPU chips
  # but TPU support is not linked into JAX. You should add a BUILD dependency
  # on //learning/brain/research/jax:tpu_support."
  #
  # This error happens because we are
  # //learning/pathways/data_parallel:tpu_support instead of the more common
  # //learning/brain/research/jax:tpu_support
  flags.FLAGS.jax_allow_unused_tpus = True

  jax.config.parse_flags_with_absl()
  absltest.main()
