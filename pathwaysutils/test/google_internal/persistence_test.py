"""Persistence tests that can only run in google3."""

from absl import flags
import jax

from google3.learning.pathways.ifrt.proxy.jax.tests import register_jax_grpc_backend_for_testing  # pylint: disable=unused-import
from absl.testing import absltest


_JAX_BACKEND_TARGET = flags.DEFINE_string(
    "jax_backend_target",
    "ifrt_pathways",
    "Jax backend target to use.",
)

_JAX_PLATFORMS = flags.DEFINE_string(
    "jax_platforms",
    "proxy",
    "Jax platforms to use.",
)

# set JAX_ALLOW_UNUSED_TPUS to avoid the error below
#
# AssertionError: The host has 4 TPU chips
# but TPU support is not linked into JAX. You should add a BUILD dependency
# on //learning/brain/research/jax:tpu_support."
#
# This error happens because we are
# //learning/pathways/data_parallel:tpu_support instead of the more common
# //learning/brain/research/jax:tpu_support
flags.FLAGS.jax_allow_unused_tpus = True


class PersistenceTest(absltest.TestCase):

  def test_devices_can_be_fetched_from_proxy_backend(self):
    devices = jax.devices("proxy")
    self.assertNotEmpty(devices)


if __name__ == "__main__":
  absltest.main()
