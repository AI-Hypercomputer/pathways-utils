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
"""Module for collecting JAX profiles for Pathways on Cloud.

This is a replacement for the `collect_profile` script in JAX that works with
Pathways on Cloud.
"""

import argparse
import logging

from pathwaysutils import profiling

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


_DESCRIPTION = """
To profile running JAX programs, you first need to start the profiler server
in the program of interest. You can do this via
`jax.profiler.start_server(<port>)`. Once the program is running and the
profiler server has started, you can run `collect_profile` to trace the execution
for a provided duration. The trace file will be dumped into a GCS bucket
(determined by `--log_dir`).
"""


def _get_parser():
  """Returns an argument parser for the collect_profile script."""
  parser = argparse.ArgumentParser(description=_DESCRIPTION)
  parser.add_argument(
      "--log_dir",
      required=True,
      help="GCS path to store log files.",
      type=str,
  )
  parser.add_argument("port", help="Port to collect trace", type=int)
  parser.add_argument(
      "duration_ms", help="Duration to collect trace in milliseconds", type=int
  )
  parser.add_argument(
      "--host",
      default="127.0.0.1",
      help=(
          "Host to collect trace. This host IP/DNS address should be accessible"
          " from where this API is being called. Defaults to 127.0.0.1"
      ),
      type=str,
  )

  return parser


def main():
  parser = _get_parser()
  args = parser.parse_args()

  if profiling.collect_profile(
      args.port, args.duration_ms, args.host, args.log_dir
  ):
    _logger.info("Dumped profiling information in: %s", args.log_dir)
  else:
    _logger.error("Failed to collect profiling information.")


if __name__ == "__main__":
  main()
