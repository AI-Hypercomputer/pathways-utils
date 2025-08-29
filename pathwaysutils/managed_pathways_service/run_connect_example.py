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
"""Script to run JAX code on TPU with the Managed Pathways service."""

from collections.abc import Sequence
from absl import app
from . import tpu_manager


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  with tpu_manager.connect(
      "pw-scale-test-v5e-32",
      "cloud-tpu-multipod-dev",
      "us-south1",
      "gs://akshu-v5e",
      "pathways-akshu-s4-rw7-pathways-head-0-0.pathways-akshu-s4-rw7:29001",
      {"tpuv5e:4x8": 2},
  ) as tm:
    pass
    # import jax.numpy as jnp
    # import pathwaysutils
    # import pprint

    # pathwaysutils.initialize()

    # orig_matrix = jnp.zeros(5)

    # print("start")
    # result_matrix = orig_matrix + 1
    # print("Original Random Matrix:")
    # pprint.pprint(orig_matrix)
    # print("\nMatrix after adding 1:")
    # pprint.pprint(result_matrix)


if __name__ == "__main__":
  app.run(main)
