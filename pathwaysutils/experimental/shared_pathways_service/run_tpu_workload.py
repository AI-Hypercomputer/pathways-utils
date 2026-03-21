r"""Run a TPU workload with Shared Pathways Service.

Run your TPU workload locally using Shared Pathways Service, the service will
deploy a Pathways proxy to run the TPU-specific components of your workload on
the requested TPU slices.

Example:
python3 run_tpu_workload.py \
    --cluster my-cluster \
    --project my-project \
    --region=us-central1 \
    --gcs_bucket=my-gcs-bucket \
    --pathways_service=pathways-head:8000 \
    --tpu_type=tpuv6e:4x8 \
    --tpu_count=1 \
    --command "python3 my_workload.py ..."

"""

import subprocess
from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging
from pathwaysutils.experimental.shared_pathways_service import isc_pathways


FLAGS = flags.FLAGS

flags.DEFINE_string("cluster", None, "The name of the GKE cluster.")
flags.DEFINE_string("project", None, "The GCP project ID.")
flags.DEFINE_string("region", None, "The GCP region.")
flags.DEFINE_string("gcs_bucket", None, "The Google Cloud Storage bucket.")
flags.DEFINE_string(
    "pathways_service",
    None,
    "The address and port of the Pathways Resource Manager.",
)
flags.DEFINE_string(
    "tpu_type", "tpuv6e:2x2", "The TPU machine type and topology."
)
flags.DEFINE_integer("tpu_count", 1, "The number of TPU slices.")
flags.DEFINE_string(
    "proxy_job_name",
    None,
    "The name to use for the GKE job for proxy. If not provided, a random name"
    " will be generated.",
)
flags.DEFINE_string(
    "proxy_server_image",
    None,
    "The proxy server image to use. If not provided, a default will be used.",
)
flags.DEFINE_list(
    "proxy_options",
    None,
    "Configuration options for the Pathways proxy. Specify entries in the form"
    ' "key:value". For example: --proxy_options=use_insecure_credentials:true',
)
flags.DEFINE_string("command", None, "The command to run on TPUs.")


def run_workload(
    *,
    cluster: str,
    project: str,
    region: str,
    gcs_bucket: str,
    pathways_service: str,
    tpu_type: str,
    tpu_count: int,
    command: str,
    proxy_job_name: str | None = None,
    proxy_server_image: str | None = None,
    proxy_options: Sequence[str] | None = None,
    connect_fn=isc_pathways.connect,
) -> None:
  """Runs the TPU workload within a Shared Pathways connection.

  Args:
    cluster: The name of the GKE cluster.
    project: The GCP project ID.
    region: The GCP region.
    gcs_bucket: The Google Cloud Storage bucket.
    pathways_service: The address and port of the Pathways Resource Manager.
    tpu_type: The TPU machine type and topology.
    tpu_count: The number of TPU slices.
    command: The command to run on TPUs.
    proxy_job_name: The name to use for the GKE job for proxy.
    proxy_server_image: The proxy server image to use.
    proxy_options: Configuration options for the Pathways proxy.
    connect_fn: The function to use for establishing the connection context.
  """
  parsed_proxy_options = isc_pathways.ProxyOptions.from_list(proxy_options)

  logging.info("Connecting to Shared Pathways Service...")
  with connect_fn(
      cluster=cluster,
      project=project,
      region=region,
      gcs_bucket=gcs_bucket,
      pathways_service=pathways_service,
      expected_tpu_instances={tpu_type: tpu_count},
      proxy_job_name=proxy_job_name,
      proxy_server_image=proxy_server_image or isc_pathways.DEFAULT_PROXY_IMAGE,
      proxy_options=parsed_proxy_options,
  ):
    logging.info("Connection established. Running command: %s", command)
    try:
      subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
      logging.error("Command failed with error: %s", e)
      raise


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  flags.mark_flags_as_required([
      "cluster",
      "project",
      "region",
      "gcs_bucket",
      "pathways_service",
      "command",
  ])

  run_workload(
      cluster=FLAGS.cluster,
      project=FLAGS.project,
      region=FLAGS.region,
      gcs_bucket=FLAGS.gcs_bucket,
      pathways_service=FLAGS.pathways_service,
      tpu_type=FLAGS.tpu_type,
      tpu_count=FLAGS.tpu_count,
      command=FLAGS.command,
      proxy_job_name=FLAGS.proxy_job_name,
      proxy_server_image=FLAGS.proxy_server_image,
      proxy_options=FLAGS.proxy_options,
  )


if __name__ == "__main__":
  app.run(main)
