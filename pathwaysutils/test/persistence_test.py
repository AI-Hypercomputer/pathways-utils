"""Tests for the persistence module."""

import datetime

import jax
import numpy as np
from pathwaysutils.persistence import helper

from absl.testing import absltest


class PersistenceTest(absltest.TestCase):
  location = "/path/to/location"
  name = "name"
  dtype = np.dtype(np.int32)
  shape = [8, 4]
  timeout = datetime.timedelta(seconds=3)

  def setUp(self):
    jax.config.update("jax_platforms", "cpu")
    super().setUp()

  def test_get_read_request(self):
    devices = jax.devices()
    # 1d sharding
    mesh = jax.sharding.Mesh(devices, "batch")
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("batch")
    )
    read_request = helper.get_read_request(
        location_path=self.location,
        name=self.name,
        dtype=self.dtype,
        shape=self.shape,
        sharding=sharding,
        devices=devices,
        timeout=self.timeout,
    )
    self.assertNotEmpty(read_request)

  def test_get_bulk_read_request(self):
    devices = jax.devices()
    # 1d sharding
    mesh = jax.sharding.Mesh(devices, "batch")
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("batch")
    )
    bulk_read_request = helper.get_bulk_read_request(
        location_path=self.location,
        names=[self.name, self.name],
        dtypes=[self.dtype, self.dtype],
        shapes=[self.shape, self.shape],
        shardings=[sharding, sharding],
        devices=devices,
        timeout=self.timeout,
    )
    self.assertNotEmpty(bulk_read_request)

  def test_get_write_request(self):
    devices = jax.devices()
    # 1d sharding
    mesh = jax.sharding.Mesh(devices, "batch")
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("batch")
    )
    write_request = helper.get_write_request(
        location_path=self.location,
        name=self.name,
        jax_array=jax.device_put(
            np.arange(np.prod(self.shape), dtype=self.dtype).reshape(
                self.shape
            ),
            sharding,
        ),
        timeout=self.timeout,
    )
    self.assertNotEmpty(write_request)

  def test_get_bulk_write_request(self):
    devices = jax.devices()
    # 1d sharding
    mesh = jax.sharding.Mesh(devices, "batch")
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("batch")
    )
    bulk_write_request = helper.get_bulk_write_request(
        location_path=self.location,
        names=[self.name, self.name],
        jax_arrays=[
            jax.device_put(
                np.arange(np.prod(self.shape), dtype=self.dtype).reshape(
                    self.shape
                ),
                sharding,
            ),
            jax.device_put(
                np.arange(np.prod(self.shape), dtype=self.dtype).reshape(
                    self.shape
                ),
                sharding,
            ),
        ],
        timeout=self.timeout,
    )
    self.assertNotEmpty(bulk_write_request)


if __name__ == "__main__":
  absltest.main()
