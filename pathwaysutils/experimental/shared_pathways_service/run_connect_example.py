"""Script to run JAX code on TPU with the Shared Pathways service."""

from collections.abc import Sequence
import pprint

from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import jax.sharding as jsharding
import pathwaysutils
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
    "Deprecated: The proxy server image to use. If not provided, it will be"
    " auto-detected from the Pathways service.",
)
flags.DEFINE_list(
    "proxy_options",
    None,
    "Configuration options for the Pathways proxy. Specify entries in the form"
    ' "key:value". For example: --proxy_options=use_insecure_credentials:true'
    ' or --proxy_options=xla_flags:"--xla_flag1 --xla_flag2"',
)

flags.DEFINE_bool(
    "collect_service_metrics",
    False,
    "Whether to enable metrics collection for Shared Pathways Service.",
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

  with isc_pathways.connect(
      cluster=FLAGS.cluster,
      project=FLAGS.project,
      region=FLAGS.region,
      gcs_bucket=FLAGS.gcs_bucket,
      pathways_service=FLAGS.pathways_service,
      expected_tpu_instances={FLAGS.tpu_type: FLAGS.tpu_count},
      proxy_job_name=FLAGS.proxy_job_name,
      proxy_server_image=FLAGS.proxy_server_image,
      proxy_options=FLAGS.proxy_options,
      collect_service_metrics=FLAGS.collect_service_metrics,
  ):
    # your-workload
    tpu_devices = jax.devices()
    for device in tpu_devices:
      print("Device: %s, Kind: %s", device, device.device_kind)
      if "tpu" not in device.device_kind.lower():
        print("Error! TPUs not found")
        exit()
      if not pathwaysutils.is_pathways_backend_used():
        print("Error! TPUs not found")
        exit()
    print(
        "All devices are confirmed to be TPUs. TPU devices found:"
        f" {tpu_devices}"
    )
    num_devices = len(tpu_devices)
    mesh = jsharding.Mesh(tpu_devices, axis_names=("data",))
    sharding = jsharding.NamedSharding(mesh, jsharding.PartitionSpec("data"))

    @jax.jit
    def tpu_add_one(x):
      return x + 1

    orig_matrix = jnp.zeros(num_devices)
    orig_matrix_tpu = jax.device_put(orig_matrix, sharding)
    result_matrix_tpu = tpu_add_one(orig_matrix_tpu)
    result_matrix = jax.device_get(result_matrix_tpu)

    print(f"Original Matrix (on all {num_devices} devices):")
    pprint.pprint(orig_matrix)
    print("\nResult Matrix after parallel addition:")
    pprint.pprint(result_matrix)


if __name__ == "__main__":
  app.run(main)
