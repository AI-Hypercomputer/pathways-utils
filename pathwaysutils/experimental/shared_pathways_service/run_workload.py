r"""Run a TPU workload with Shared Pathways Service.

Run your TPU workload locally using Shared Pathways Service, the service will
deploy a Pathways proxy to run the TPU-specific components of your workload on
the requested TPU slices.

Example:
python3 run_workload.py \
    --cluster my-cluster \
    --project my-project \
    --region=us-central1 \
    --gcs_bucket=my-gcs-bucket \
    --pathways_service=pathways-head:8000 \
    --tpu_type=tpuv6e:4x8 \
    --tpu_count=1 \
    --command "python3 my_workload.py ..."

"""

from collections.abc import Callable, Sequence
import os
import shlex
import subprocess
from typing import Any, ContextManager

from absl import app
from absl import flags
from absl import logging
from pathwaysutils.experimental.shared_pathways_service import isc_pathways

_CLUSTER = flags.DEFINE_string(
    "cluster", None, "The name of the GKE cluster.", required=True
)
_PROJECT = flags.DEFINE_string(
    "project", None, "The GCP project ID.", required=True
)
_REGION = flags.DEFINE_string(
    "region", None, "The GCP region.", required=True
)
_GCS_BUCKET = flags.DEFINE_string(
    "gcs_bucket", None, "The Google Cloud Storage bucket.", required=True
)
_PATHWAYS_SERVICE = flags.DEFINE_string(
    "pathways_service",
    None,
    "The address and port of the Pathways Resource Manager. See"
    " https://github.com/AI-Hypercomputer/pathways-utils/tree/main/pathwaysutils/experimental/shared_pathways_service#4-find-the-pathways-service-address"
    " for instructions on how to get the Pathways service address.",
    required=True,
)
_TPU_TYPE = flags.DEFINE_string(
    "tpu_type", "tpuv6e:2x2", "The TPU machine type and topology."
)
_TPU_COUNT = flags.DEFINE_integer("tpu_count", 1, "The number of TPU slices.")
_PROXY_SERVER_IMAGE = flags.DEFINE_string(
    "proxy_server_image",
    "",
    "The proxy server image to use. If not provided, a default will be used.",
)
_PROXY_OPTIONS = flags.DEFINE_list(
    "proxy_options",
    [],
    "Configuration options for the Pathways proxy. Specify entries in the form"
    ' "key:value". For example: --proxy_options=use_insecure_credentials:true'
    ' or --proxy_options=xla_flags:"--xla_flag1 --xla_flag2"',
)
_COMMAND = flags.DEFINE_string(
    "command", None, "The command to run on TPUs.", required=True
)
_COLLECT_SERVICE_METRICS = flags.DEFINE_bool(
    "collect_service_metrics",
    False,
    "Whether to enable metrics collection for Shared Pathways Service. If"
    " enabled, the service will collect usage metrics such as TPU assignment"
    " time, active user count, capacity in use etc. The metrics will be"
    " stored in Cloud Monitoring.",
)




def run_command(
    *,
    cluster: str,
    project: str,
    region: str,
    gcs_bucket: str,
    pathways_service: str,
    tpu_type: str,
    tpu_count: int,
    command: str,
    proxy_server_image: str | None = None,
    proxy_options: Sequence[str] | None = None,
    collect_service_metrics: bool = False,
    connect_fn: Callable[..., ContextManager[Any]] = isc_pathways.connect,
) -> None:
  """Run the TPU workload within a Shared Pathways connection.

  Args:
    cluster: The name of the GKE cluster.
    project: The GCP project ID.
    region: The GCP region.
    gcs_bucket: The Google Cloud Storage bucket.
    pathways_service: The address and port of the Pathways Resource Manager.
    tpu_type: The TPU machine type and topology.
    tpu_count: The number of TPU slices.
    command: The command to run on TPUs.
    proxy_server_image: The proxy server image to use.
    proxy_options: Configuration options for the Pathways proxy.
    collect_service_metrics: Whether to collect usage metrics for Shared Pathways
      Service. Defaults to False.
    connect_fn: The function to use for establishing the connection context,
      expected to be a callable that returns a context manager.

  Raises:
    subprocess.CalledProcessError: If the workload command fails.
  """
  logging.info("Connecting to Shared Pathways Service...")
  with connect_fn(
      cluster=cluster,
      project=project,
      region=region,
      gcs_bucket=gcs_bucket,
      pathways_service=pathways_service,
      expected_tpu_instances={tpu_type: tpu_count},
      proxy_server_image=(
          proxy_server_image
          if proxy_server_image
          else isc_pathways.DEFAULT_PROXY_IMAGE
      ),
      proxy_options=proxy_options,
      collect_service_metrics=collect_service_metrics,
  ):
    logging.info("Connection established. Running command: %r", command)
    try:
      command_args = shlex.split(command)
      subprocess.run(command_args, check=True, env=os.environ.copy())
    except subprocess.CalledProcessError:
      logging.error(
          "Command failed! Find the underlying error in the logs above, where"
          " the command is invoked."
      )
      raise
    finally:
      logging.info("Command execution finished.")


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  run_command(
      cluster=_CLUSTER.value,
      project=_PROJECT.value,
      region=_REGION.value,
      gcs_bucket=_GCS_BUCKET.value,
      pathways_service=_PATHWAYS_SERVICE.value,
      tpu_type=_TPU_TYPE.value,
      tpu_count=_TPU_COUNT.value,
      command=_COMMAND.value,
      proxy_server_image=_PROXY_SERVER_IMAGE.value,
      proxy_options=_PROXY_OPTIONS.value,
      collect_service_metrics=_COLLECT_SERVICE_METRICS.value,
  )


if __name__ == "__main__":
  app.run(main)
