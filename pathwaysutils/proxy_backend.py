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
"""Register the IFRT Proxy as a backend for JAX."""

import os
import jax
from jax.extend import backend
from jax.extend.backend import ifrt_proxy


def register_backend_factory() -> None:
  """Registers the IFRT Proxy backend factory with JAX."""

  def make_client():
    options = ifrt_proxy.ClientConnectionOptions()
    timeout_secs = os.environ.get("PATHWAYS_PROXY_CONNECTION_TIMEOUT_SECS")
    if timeout_secs:
      options.connection_timeout_in_seconds = int(timeout_secs)
    return ifrt_proxy.get_client(
        jax.config.read("jax_backend_target"),
        options,
    )

  backend.register_backend_factory(
      "proxy",
      make_client,
      priority=-1,
  )
