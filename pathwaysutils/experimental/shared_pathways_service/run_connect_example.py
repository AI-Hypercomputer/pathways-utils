"""Script to run JAX code on TPU with the Shared Pathways service."""

from collections.abc import Sequence
import pprint

from absl import app
from absl import flags
import jax.numpy as jnp
from pathwaysutils.experimental.shared_pathways_service import isc_pathways


from google3.pyglib.flags.contrib import dict_flag

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
dict_flag.DEFINE_dict(
    "proxy_options",
    None,
    "Configuration options for the Pathways proxy. Specify entries in the form"
    ' "key:value". For example: --proxy_options=use_insecure_credentials:true',
)

flags.mark_flags_as_required([
    "cluster",
    "project",
    "region",
    "gcs_bucket",
    "pathways_service",
])


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  proxy_options = isc_pathways.ProxyOptions.from_dict(FLAGS.proxy_options)

  with isc_pathways.connect(
      cluster=FLAGS.cluster,
      project=FLAGS.project,
      region=FLAGS.region,
      gcs_bucket=FLAGS.gcs_bucket,
      pathways_service=FLAGS.pathways_service,
      expected_tpu_instances={FLAGS.tpu_type: FLAGS.tpu_count},
      proxy_job_name=FLAGS.proxy_job_name,
      proxy_server_image=FLAGS.proxy_server_image
      or isc_pathways.DEFAULT_PROXY_IMAGE,
      proxy_options=proxy_options,
  ):
    orig_matrix = jnp.zeros(5)
    result_matrix = orig_matrix + 1
    print("Original Random Matrix:")
    pprint.pprint(orig_matrix)
    print("\nMatrix after adding 1:")
    pprint.pprint(result_matrix)


if __name__ == "__main__":
  app.run(main)
