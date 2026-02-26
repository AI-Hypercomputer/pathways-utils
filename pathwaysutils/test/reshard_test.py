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

from collections.abc import Mapping
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from packaging import version as version_lib
from pathwaysutils import jax as pw_jax
from pathwaysutils import reshard


@absltest.skipIf(
    version_lib.Version(jax.__version__) < version_lib.Version("0.8.3"),
    "Test requires JAX version >= 0.8.3",
)
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
    mock_transfer.return_value = [
        mock.create_autospec(jax.Array, instance=True)
    ]
    reshard.reshard(x, sharding, **reshard_kwargs)

    mock_transfer.assert_called_once_with(
        mock.ANY,
        mock.ANY,
        expected_donate,
    )

  def test_ifrt_reshard_pytree(self):
    x = {"a": jnp.array([1]), "b": [jnp.array([2])]}
    devices = jax.devices()
    sharding = jax.sharding.SingleDeviceSharding(devices[0])
    # Tree prefix sharding
    tree_sharding = {"a": sharding, "b": [sharding]}

    mock_transfer = self.enter_context(
        mock.patch.object(pw_jax, "transfer_to_shardings", autospec=True)
    )
    mock_transfer.return_value = [
        mock.create_autospec(jax.Array, instance=True)
    ]

    reshard.reshard(x, tree_sharding)

    # Since they are on the same device set, they should be grouped together.
    mock_transfer.assert_called_once()
    (
        args,
        _,
    ) = mock_transfer.call_args
    self.assertLen(args[0], 2)
    self.assertLen(args[1], 2)

if __name__ == "__main__":
  absltest.main()
