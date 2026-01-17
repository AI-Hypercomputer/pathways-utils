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

from collections.abc import Mapping
import json
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from pathwaysutils import jax as pw_jax
from pathwaysutils import plugin_executable
from pathwaysutils.experimental import reshard


class ReshardTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(reshard_kwargs={"donate": True}, expected_donate=True),
      dict(reshard_kwargs={"donate": False}, expected_donate=False),
      dict(reshard_kwargs={}, expected_donate=False),
  )
  def test_ifrt_reshard_donate(
      self, reshard_kwargs: Mapping[str, Any], expected_donate: bool
  ):
    x = jnp.array([1, 2])
    devices = jax.devices()
    sharding = jax.sharding.SingleDeviceSharding(devices[0])

    mock_transfer = self.enter_context(
        mock.patch.object(pw_jax, "transfer_to_shardings", autospec=True)
    )
    self.enter_context(
        mock.patch.object(
            pw_jax, "ifrt_reshard_available", return_value=True, autospec=True
        )
    )

    reshard.reshard(x, sharding, **reshard_kwargs)

    # Signature: transfer_to_shardings(arrays, shardings, donate)
    mock_transfer.assert_called_with(mock.ANY, mock.ANY, expected_donate)

  @parameterized.parameters(
      dict(reshard_kwargs={"donate": True}, expected_donate=True),
      dict(reshard_kwargs={"donate": False}, expected_donate=False),
      dict(reshard_kwargs={}, expected_donate=False),
  )
  def test_sidechannel_reshard_donate(
      self, reshard_kwargs: Mapping[str, Any], expected_donate: bool
  ):
    x = jnp.array([1, 2])
    devices = jax.devices()
    sharding = jax.sharding.SingleDeviceSharding(devices[0])

    self.enter_context(
        mock.patch.object(
            pw_jax, "ifrt_reshard_available", return_value=False, autospec=True
        )
    )
    mock_pe = self.enter_context(
        mock.patch.object(plugin_executable, "PluginExecutable", autospec=True)
    )
    mock_pe.return_value.call.return_value = ([mock.Mock()], mock.Mock())

    reshard.reshard(x, sharding, **reshard_kwargs)

    mock_pe.assert_called()
    (json_request,), _ = mock_pe.call_args
    request = json.loads(json_request)
    self.assertEqual(request["reshardRequest"]["donateInput"], expected_donate)

  @parameterized.parameters(True, False, None)
  def test_ifrt_reshard_cache_resharding_plans(self, cache: bool | None):
    x = jnp.array([1, 2])
    devices = jax.devices()
    sharding = jax.sharding.SingleDeviceSharding(devices[0])

    mock_transfer = self.enter_context(
        mock.patch.object(pw_jax, "transfer_to_shardings")
    )
    self.enter_context(
        mock.patch.object(pw_jax, "ifrt_reshard_available", return_value=True)
    )

    if cache is None:
      reshard.reshard(x, sharding)
    elif cache:
      with self.assertWarnsRegex(
          UserWarning, "cache_resharding_plans` is only applicable"
      ):
        reshard.reshard(x, sharding, cache_resharding_plans=cache)
    else:
      reshard.reshard(x, sharding, cache_resharding_plans=cache)

    mock_transfer.assert_called_once()

  @parameterized.parameters(
      dict(cache=True, expected_cache=True),
      dict(cache=False, expected_cache=False),
      dict(cache=None, expected_cache=False),
  )
  def test_sidechannel_reshard_cache_resharding_plans(
      self, cache, expected_cache
  ):
    x = jnp.array([1, 2])
    devices = jax.devices()
    sharding = jax.sharding.SingleDeviceSharding(devices[0])

    self.enter_context(
        mock.patch.object(pw_jax, "ifrt_reshard_available", return_value=False)
    )
    mock_pe = self.enter_context(
        mock.patch.object(plugin_executable, "PluginExecutable")
    )
    mock_pe.return_value.call.return_value = ([mock.Mock()], mock.Mock())

    mock_get_resharding_plan_cached = self.enter_context(
        mock.patch.object(reshard, "_get_resharding_plan_cached")
    )

    if cache is None:
      reshard.reshard(x, sharding)
    else:
      reshard.reshard(x, sharding, cache_resharding_plans=cache)

    self.assertEqual(mock_pe.call_count, 0 if expected_cache else 1)

    self.assertEqual(
        mock_get_resharding_plan_cached.call_count,
        1 if expected_cache else 0,
    )

if __name__ == "__main__":  absltest.main()
