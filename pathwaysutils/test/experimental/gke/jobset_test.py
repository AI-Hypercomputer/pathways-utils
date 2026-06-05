from absl.testing import absltest
from absl.testing import parameterized
from kubernetes import client
from pathwaysutils.experimental.gke import jobset


class PathwaysJobSetTest(parameterized.TestCase):

  def test_invalid_tpu_type(self):
    with self.assertRaisesRegex(ValueError, "Unsupported TPU type"):
      jobset.PathwaysJobSet(
          name="test-jobset",
          namespace="default",
          pathways_dir="gs://test-bucket",
          tpu_type="invalid-tpu",
          topology="4x4",
          num_slices=1,
      )

  def test_headless_head_job(self):
    js = jobset.PathwaysJobSet(
        name="test-jobset",
        namespace="default",
        pathways_dir="gs://test-bucket",
        tpu_type="v5e",
        topology="4x8",
        num_slices=2,
        elastic_slices=2,
    )
    config = js.to_dict()

    replicated_jobs = config["spec"]["replicatedJobs"]
    self.assertLen(replicated_jobs, 2)

    head_job = next(j for j in replicated_jobs if j["name"] == "pathways-head")
    self.assertEqual(head_job["replicas"], 1)

    pod_spec = head_job["template"]["spec"]["template"]["spec"]
    self.assertTrue(pod_spec["hostNetwork"])
    self.assertEqual(pod_spec["dnsPolicy"], "ClusterFirstWithHostNet")
    self.assertEqual(pod_spec["restartPolicy"], "Never")

    # In headless mode, RM and Proxy are in containers list
    containers = pod_spec["containers"]
    self.assertLen(containers, 2)
    rm_container = next(c for c in containers if c["name"] == "pathways-rm")
    proxy_container = next(
        c for c in containers if c["name"] == "pathways-proxy"
    )

    self.assertEqual(
        rm_container["image"],
        "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:latest",
    )
    self.assertEqual(
        proxy_container["image"],
        "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:latest",
    )
    self.assertIn("--num_elastic_slices=2", proxy_container["args"])

  def test_non_headless_head_job(self):
    user_pod_template = {
        "metadata": {"annotations": {"example.com/annotation": "value"}},
        "spec": {
            "containers": [{
                "name": "jax-tpu",
                "image": "ubuntu:latest",
                "command": ["sleep", "infinity"],
            }]
        },
    }
    js = jobset.PathwaysJobSet(
        name="test-jobset",
        namespace="default",
        pathways_dir="gs://test-bucket",
        tpu_type="v5e",
        topology="4x8",
        num_slices=2,
        user_pod_template=user_pod_template,
        main_container_name="jax-tpu",
    )
    config = js.to_dict()

    replicated_jobs = config["spec"]["replicatedJobs"]
    head_job = next(j for j in replicated_jobs if j["name"] == "pathways-head")

    pod_spec = head_job["template"]["spec"]["template"]["spec"]
    self.assertTrue(pod_spec["hostNetwork"])
    self.assertEqual(pod_spec["dnsPolicy"], "ClusterFirstWithHostNet")

    # RM and Proxy should be in initContainers
    init_containers = pod_spec["initContainers"]
    self.assertLen(init_containers, 2)
    rm_container = next(
        c for c in init_containers if c["name"] == "pathways-rm"
    )
    proxy_container = next(
        c for c in init_containers if c["name"] == "pathways-proxy"
    )

    self.assertEqual(rm_container["restartPolicy"], "Always")
    self.assertEqual(proxy_container["restartPolicy"], "Always")

    # Main container should have JAX env vars injected
    main_container = next(
        c for c in pod_spec["containers"] if c["name"] == "jax-tpu"
    )
    env_names = [e["name"] for e in main_container["env"]]
    self.assertIn("PATHWAYS_HEAD", env_names)
    self.assertIn("JAX_PLATFORMS", env_names)
    self.assertIn("XCLOUD_ENVIRONMENT", env_names)
    self.assertIn("JAX_BACKEND_TARGET", env_names)

    # Verify annotations are propagated
    self.assertEqual(
        head_job["template"]["metadata"]["annotations"][
            "example.com/annotation"
        ],
        "value",
    )
    self.assertEqual(
        head_job["template"]["spec"]["template"]["metadata"]["annotations"][
            "example.com/annotation"
        ],
        "value",
    )

  def test_monkeypatch_restart_policy(self):
    # Construct V1Container with restart_policy to test monkeypatch.
    c = client.V1Container(
        name="test", restart_policy="Always"
    )  # pytype: disable=wrong-keyword-args
    self.assertEqual(getattr(c, "restart_policy"), "Always")


if __name__ == "__main__":
  absltest.main()
