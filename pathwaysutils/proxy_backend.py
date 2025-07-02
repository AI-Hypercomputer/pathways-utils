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

import jax
from jax.extend import backend

# Attempt to import ifrt_proxy from the new location, fallback to the old one.
try:
    from jax._src.lib import _jax
    ifrt_proxy = _jax.ifrt_proxy
except (ImportError, AttributeError):
    # AttributeError is included in case jax._src.lib exists but _jax or ifrt_proxy does not.
    from jax.lib.xla_extension import ifrt_proxy


def register_backend_factory():
  backend.register_backend_factory(
      "proxy",
      lambda: ifrt_proxy.get_client(
          jax.config.read("jax_backend_target"),
          ifrt_proxy.ClientConnectionOptions(),
      ),
      priority=-1,
  )
