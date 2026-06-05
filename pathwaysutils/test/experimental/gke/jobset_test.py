from absl.testing import absltest
from absl.testing import parameterized
from pathwaysutils.experimental.gke import jobset


class PathwaysJobSetTest(parameterized.TestCase):

  def test_invalid_tpu_type(self):
    with self.assertRaisesRegex(ValueError, "Unsupported TPU type"):
      jobset.PathwaysJobSet(
          name="test-jobset",
          namespace="default",
          tpu_type="invalid-tpu",
          num_slices=1,
      )

  def test_basic_jobset_structure(self):
    js = jobset.PathwaysJobSet(
        name="test-jobset",
        namespace="default",
        tpu_type="v5e",
        num_slices=2,
        labels={"app": "pathways"},
        annotations={"example.com/annotation": "value"},
    )
    config = js.to_dict()

    self.assertEqual(config["apiVersion"], "jobset.sigs.k8s.io/v1alpha2")
    self.assertEqual(config["kind"], "JobSet")
    self.assertEqual(config["metadata"]["name"], "test-jobset")
    self.assertEqual(config["metadata"]["namespace"], "default")
    self.assertEqual(config["metadata"]["labels"]["app"], "pathways")
    self.assertEqual(
        config["metadata"]["annotations"]["example.com/annotation"], "value"
    )

    self.assertEqual(config["spec"]["failurePolicy"]["maxRestarts"], 0)

    replicated_jobs = config["spec"]["replicatedJobs"]
    self.assertLen(replicated_jobs, 2)

    head_job = replicated_jobs[0]
    self.assertEqual(head_job["name"], "pathways-head")
    self.assertEqual(head_job["replicas"], 1)

    # In K8s API models, V1JobTemplateSpec -> V1JobSpec -> V1PodTemplateSpec
    # -> V1PodSpec. When serialized, they match this structure.
    head_pod_spec = head_job["template"]["spec"]["template"]["spec"]
    self.assertEqual(head_pod_spec["containers"][0]["name"], "placeholder-head")

    worker_job = replicated_jobs[1]
    self.assertEqual(worker_job["name"], "pathways-worker")
    self.assertEqual(worker_job["replicas"], 2)
    worker_pod_spec = worker_job["template"]["spec"]["template"]["spec"]
    self.assertEqual(
        worker_pod_spec["containers"][0]["name"], "placeholder-worker"
    )


if __name__ == "__main__":
  absltest.main()
