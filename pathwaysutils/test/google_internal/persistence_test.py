"""Persistence tests that can only run in google3."""

import datetime
import logging
import tempfile
from typing import Sequence

from absl import flags
import jax
from jax import core
from jax.experimental import mesh_utils
import jax.numpy as jnp
import numpy as np
from pathwaysutils import plugin_executable as pe
from pathwaysutils.persistence import helper

from google3.learning.pathways.ifrt.proxy.jax.tests import register_jax_grpc_backend_for_testing  # pylint: disable=unused-import
from absl.testing import absltest


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
MULTI_ARRAYS_N = 10


def temp_dir_roundtrip_bulk(
    devices: Sequence[jax.Device],
    names: Sequence[str],
    dtype: jnp.dtype,
    input_arrs: Sequence[jax.Array],
    shardings: Sequence[jax.sharding.Sharding],
    timeout: datetime.timedelta,
) -> Sequence[jax.Array]:
  jax_arrays = [
      jax.device_put(input_arr, sharding)
      for input_arr, sharding in zip(input_arrs, shardings)
  ]
  with tempfile.TemporaryDirectory() as tmp_path:
    # write
    bulk_write_request = helper.get_bulk_write_request(
        tmp_path, names, jax_arrays, timeout
    )
    logging.info("bulk_write_request: %s", bulk_write_request)
    bulk_write_executable = pe.PluginExecutable(bulk_write_request)
    _, bulk_write_fut = bulk_write_executable.call(in_arr=jax_arrays)
    bulk_write_fut.result()

    # read
    dtypes = [jax_array.dtype for jax_array in jax_arrays]
    shapes = [jax_array.shape for jax_array in jax_arrays]
    out_shardings = [jax_array.sharding for jax_array in jax_arrays]
    bulk_read_request = helper.get_bulk_read_request(
        tmp_path, names, dtypes, shapes, out_shardings, devices, timeout
    )
    logging.info("bulk_read_request: %s", bulk_read_request)
    bulk_read_executable = pe.PluginExecutable(bulk_read_request)
    out_avals = [core.ShapedArray(shape, dtype) for shape in shapes]
    out_arrays, bulk_read_fut = bulk_read_executable.call(
        out_shardings=out_shardings, out_avals=out_avals
    )
    bulk_read_fut.result()
    return out_arrays


def generate_input_arrays(
    shape: Sequence[int], dtype: jnp.dtype, n_arrays: int
) -> Sequence[jax.Array]:
  return [
      np.arange(i, np.prod(shape) + i, dtype=dtype).reshape(shape)
      for i in range(n_arrays)
  ]


class PersistenceTest(absltest.TestCase):

  def test_devices_can_be_fetched_from_proxy_backend(self):
    devices = jax.devices("proxy")
    self.assertNotEmpty(devices)

  def _persistence_roundtrip_fully_partitioned(
      self,
      devices: Sequence[jax.Device],
      shape: Sequence[int],
      mesh_axes: Sequence[str],
      num_arrays: int,
      timeout: datetime.timedelta = datetime.timedelta(seconds=30),
  ):
    name = "array_name"
    dtype = jnp.int32
    mesh = jax.sharding.Mesh(devices, mesh_axes)
    shardings = [
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(mesh_axes))
    ] * num_arrays
    names = [name + "_" + str(i) for i in range(num_arrays)]
    input_arrs = [
        np.arange(i, np.prod(shape) + i, dtype=dtype).reshape(shape)
        for i in range(num_arrays)
    ]
    out_arrs = temp_dir_roundtrip_bulk(
        devices, names, dtype, input_arrs, shardings, timeout
    )
    logging.info("out_arrs: %s", out_arrs)
    logging.info("input_arrs: %s", input_arrs)

    self.assertTrue(
        all([
            jnp.array_equal(input_arr, out_arr)
            for input_arr, out_arr in zip(input_arrs, out_arrs)
        ])
    )

  def _test_persistence_roundtrip_partially_replicated(
      self,
      devices: Sequence[jax.Device],
      shape: Sequence[int],
      num_arrays: int,
      timeout: datetime.timedelta = datetime.timedelta(seconds=30),
  ):
    dtype = jnp.int32
    array_base_name = "array_name"
    input_arrs = generate_input_arrays(shape, dtype, num_arrays)
    mesh = jax.sharding.Mesh(devices, ("a", "b"))
    # For 1+ dimension array; sharded along 'a', replicated along 'b'
    shardings = [
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("a"))
    ] * num_arrays
    names = [array_base_name + "_" + str(i) for i in range(num_arrays)]
    out_arrs = temp_dir_roundtrip_bulk(
        devices, names, dtype, input_arrs, shardings, timeout
    )
    logging.info("out_arrs: %s", out_arrs)
    logging.info("input_arrs: %s", input_arrs)
    self.assertTrue(
        all([
            jnp.array_equal(input_arr, out_arr)
            for input_arr, out_arr in zip(input_arrs, out_arrs)
        ])
    )

  def test_persistence_roundtrip_single_device_1d_sharding(self):
    """Test that we can persist a JAX array and read it back in a single device."""
    self._persistence_roundtrip_fully_partitioned(
        [jax.devices("proxy")[0]],
        [8, 4],
        ("batch",),
        num_arrays=10,
    )

  def test_persistence_roundtrip_multiple_devices_1d_sharding(self):
    """Test that we can persist a JAX array and read it back in multiple devices."""
    self._persistence_roundtrip_fully_partitioned(
        jax.devices(), [8, 4], ("batch",), num_arrays=1
    )

  def test_persistence_roundtrip_multiple_devices_1d_sharding_multi_arrays(
      self,
  ):
    """Test that we can persist a JAX array and read it back in multiple devices."""
    self._persistence_roundtrip_fully_partitioned(
        jax.devices(), [8, 4], ("batch",), num_arrays=1
    )

  def test_persistence_roundtrip_multiple_devices_reversed_1d_sharding(self):
    """Test that we can persist a JAX array and read it back in multiple devices passed in reverse order from their ids."""
    self._persistence_roundtrip_fully_partitioned(
        [jax.devices()[1], jax.devices()[0]], [8, 4], ("batch",), num_arrays=1
    )

  def test_persistence_roundtrip_multiple_devices_reversed_1d_sharding_multi_arrays(
      self,
  ):
    """Test that we can persist multiple JAX arrays and read them back in multiple devices passed in reverse order from their ids."""
    self._persistence_roundtrip_fully_partitioned(
        [jax.devices()[1], jax.devices()[0]],
        [8, 4],
        ("batch",),
        num_arrays=MULTI_ARRAYS_N,
    )

  def test_persistence_roundtrip_multiple_devices_reversed_2d_sharding(self):
    """Test that we can persist a JAX array and read it back in multiple devices with a 2d mesh."""
    devices = mesh_utils.create_device_mesh((2, 1), jax.devices()[:2])
    self._persistence_roundtrip_fully_partitioned(
        devices, [8, 4], ("x", "y"), num_arrays=1
    )

  def test_persistence_roundtrip_multiple_devices_reversed_2d_sharding_multi_arrays(
      self,
  ):
    """Test that we can persist multiple JAX arrays and read them back in multiple devices with a 2d mesh."""
    devices = mesh_utils.create_device_mesh((2, 1), jax.devices()[:2])
    self._persistence_roundtrip_fully_partitioned(
        devices, [8, 4], ("x", "y"), num_arrays=MULTI_ARRAYS_N
    )

  def test_persistence_roundtrip_partially_replicated(self):
    """Test that we can persist a JAX array and read it back in a partially replicated mesh."""
    devices = mesh_utils.create_device_mesh((2, 2), jax.devices()[:4])
    self._test_persistence_roundtrip_partially_replicated(
        devices, [8, 4], num_arrays=1
    )

  def test_persistence_roundtrip_partially_replicated_multi_arrays(self):
    """Test that we can persist multiple JAX arrays and read them back in a partially replicated mesh."""
    devices = mesh_utils.create_device_mesh((2, 2), jax.devices()[:4])
    self._test_persistence_roundtrip_partially_replicated(
        devices, [8, 4], num_arrays=MULTI_ARRAYS_N
    )


if __name__ == "__main__":
  jax.config.parse_flags_with_absl()
  absltest.main()
