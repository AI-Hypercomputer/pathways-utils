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
"""Helper functions for persistence."""

import base64
import datetime
import json
from typing import Sequence, Union

import jax
from jax import core
from jax.lib import xla_client as xc
import numpy as np
from pathwaysutils import plugin_executable


def base64_utf8_stringify(bs: bytes) -> str:
  """Converts bytes to a base64-encoded utf-8 string.

  Args:
    bs: The bytes to convert.

  Returns:
    The base64-encoded utf-8 string.
  """
  return base64.b64encode(bs).decode("utf-8")


def string_to_base64(text: str) -> str:
  """Encodes a string to base64 format.

  Args:
    text: The string to encode.

  Returns:
    The base64-encoded string.
  """
  return base64_utf8_stringify(text.encode("utf-8"))


def get_hlo_sharding_string(
    sharding: jax.sharding.Sharding,
    num_dimensions: int,
) -> str:
  """Serializes the sharding to an hlo-sharding, encodes it to base64 and returns the base-64 as an utf-8 string."""
  return base64_utf8_stringify(
      # pylint:disable=protected-access
      sharding._to_xla_hlo_sharding(num_dimensions)  # pytype: disable=attribute-error
      # pylint:enable=protected-access
      .to_proto().SerializeToString()
  )


def get_shape_string(
    dtype: np.dtype,
    shape: Sequence[int],
) -> str:
  """Serializes the shape, encodes it to base64 and returns the base-64 as an utf-8 string."""
  return base64_utf8_stringify(
      xc.Shape.array_shape(
          xc.PrimitiveType(xc.dtype_to_etype(dtype)),
          shape,
      )
      .with_major_to_minor_layout_if_absent()
      .to_serialized_proto()
  )


def get_write_request(
    location_path: str,
    name: str,
    jax_array: jax.Array,
    timeout: datetime.timedelta,
) -> str:
  """Returns a string representation of the plugin program which writes the given jax_array to the given location."""
  sharding = jax_array.sharding
  assert isinstance(sharding, jax.sharding.Sharding), sharding

  timeout_seconds, timeout_fractional_seconds = divmod(
      timeout.total_seconds(), 1
  )
  timeout_nanoseconds = timeout_fractional_seconds * 1e9
  return json.dumps({
      "persistenceWriteRequest": {
          "b64_location": string_to_base64(location_path),
          "b64_name": string_to_base64(name),
          "b64_hlo_sharding_string": get_hlo_sharding_string(
              jax_array.sharding, len(jax_array.shape)
          ),
          "shape": jax_array.shape,
          "devices": {
              "device_ids": [
                  # pylint:disable=protected-access
                  device.id
                  for device in sharding._device_assignment
                  # pylint:enable=protected-access
              ],
          },
          "timeout": {
              "seconds": int(timeout_seconds),
              "nanos": int(timeout_nanoseconds),
          },
      }
  })


def get_read_request(
    location_path: str,
    name: str,
    dtype: np.dtype,
    shape: Sequence[int],
    sharding: jax.sharding.Sharding,
    devices: Sequence[jax.Device],
    timeout: datetime.timedelta,
) -> str:
  """Returns a string representation of the plugin program which reads the given array from the given location into the provided sharding."""
  if not isinstance(devices, np.ndarray):
    devices = np.array(devices)

  timeout_seconds, timeout_fractional_seconds = divmod(
      timeout.total_seconds(), 1
  )
  timeout_nanoseconds = timeout_fractional_seconds * 1e9
  return json.dumps({
      "persistenceReadRequest": {
          "b64_location": string_to_base64(location_path),
          "b64_shape_proto_string": get_shape_string(dtype, shape),
          "b64_name": string_to_base64(name),
          "b64_hlo_sharding_string": get_hlo_sharding_string(
              sharding, len(shape)
          ),
          "devices": {
              "device_ids": [device.id for device in devices.flatten()]
          },
          "timeout": {
              "seconds": int(timeout_seconds),
              "nanos": int(timeout_nanoseconds),
          },
      }
  })


def write_one_array(
    location: str,
    name: str,
    value: jax.Array,
    timeout: datetime.timedelta,
):
  """Creates the write array plugin program string, compiles it to an executable, calls it and returns an awaitable future."""
  write_request = get_write_request(location, name, value, timeout)
  write_executable = plugin_executable.PluginExecutable(write_request)
  _, write_future = write_executable.call([value])
  return write_future


def read_one_array(
    location: str,
    name: str,
    dtype: np.dtype,
    shape: Sequence[int],
    shardings: jax.sharding.Sharding,
    devices: Union[Sequence[jax.Device], np.ndarray],
    timeout: datetime.timedelta,
):
  """Creates the read array plugin program string, compiles it to an executable, calls it and returns the result."""
  read_request = get_read_request(
      location,
      name,
      dtype,
      shape,
      shardings,
      devices,
      timeout,
  )
  read_executable = plugin_executable.PluginExecutable(read_request)
  out_aval = core.ShapedArray(shape, dtype)
  read_array, read_future = read_executable.call(
      out_shardings=[shardings], out_avals=[out_aval]
  )
  read_future.result()
  return read_array[0]
