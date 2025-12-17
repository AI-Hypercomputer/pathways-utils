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
"""Experimental profiling utilites."""

from typing import Any

from pathwaysutils import profiling


def start_trace(
    profile_request: dict[str, Any],
    *,
    create_perfetto_link: bool = False,
    create_perfetto_trace: bool = False,
) -> None:
  """Starts a profiler trace.

  This is primarily for internal use where we can experiment with profile
  requests with additional fields without needing to change
  `pathwaysutils.profiling.start_trace` for each experiment.

  Use `jax.profiler.stop_trace` to end profiling.

  Args:
    profile_request: A dictionary containing the profile request options.
    create_perfetto_link: A boolean which, if true, creates and prints link to
      the Perfetto trace viewer UI (https://ui.perfetto.dev). The program will
      block until the link is opened and Perfetto loads the trace. This feature
      is experimental for Pathways on Cloud and may not be fully supported.
    create_perfetto_trace: A boolean which, if true, additionally dumps a
      ``perfetto_trace.json.gz`` file that is compatible for upload with the
      Perfetto trace viewer UI (https://ui.perfetto.dev). The file will also be
      generated if ``create_perfetto_link`` is true. This could be useful if you
      want to generate a Perfetto-compatible trace without blocking the process.
      This feature is experimental for Pathways on Cloud and may not be fully
      supported.
  """
  log_dir = profile_request["traceLocation"]
  if not str(log_dir).startswith("gs://"):
    raise ValueError(
        "profile_request['traceLocation'] must be a GCS bucket path, got"
        f" {log_dir}"
    )

  profiling._start_pathways_trace_from_profile_request(profile_request)  # pylint: disable=protected-access

  profiling._original_start_trace(  # pylint: disable=protected-access
      log_dir=log_dir,
      create_perfetto_link=create_perfetto_link,
      create_perfetto_trace=create_perfetto_trace,
  )
