import hashlib
import itertools
from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
from kubernetes import client
from pathwaysutils.experimental.gke import jobset


class JobSetManifestHelper:
  """Helper to parse and navigate compiled JobSet manifests without next()."""

  def __init__(self, config: dict[str, Any]):
    self.config = config
    self.jobs = {}
    self.pod_specs = {}
    self.pod_metadatas = {}
    self.job_metadatas = {}
    # map: job_name -> container_name -> container_dict
    self.containers = {}
    # map: job_name -> container_name -> init_container_dict
    self.init_containers = {}
    # map: job_name -> volume_name -> volume_dict
    self.volumes = {}

    for job in config.get("spec", {}).get("replicatedJobs", []):
      job_name = job["name"]
      self.jobs[job_name] = job

      job_template = job.get("template", {})
      self.job_metadatas[job_name] = job_template.get("metadata", {})

      pod_template = job_template.get("spec", {}).get("template", {})
      self.pod_metadatas[job_name] = pod_template.get("metadata", {})

      pod_spec = pod_template.get("spec", {})
      self.pod_specs[job_name] = pod_spec

      job_containers = {}
      for c in pod_spec.get("containers", []):
        job_containers[c["name"]] = c
      self.containers[job_name] = job_containers

      job_init_containers = {}
      for c in pod_spec.get("initContainers", []):
        job_init_containers[c["name"]] = c
      self.init_containers[job_name] = job_init_containers

      job_volumes = {}
      for v in pod_spec.get("volumes", []):
        job_volumes[v["name"]] = v
      self.volumes[job_name] = job_volumes

  def get_all_containers_by_name(
      self, container_name: str
  ) -> list[tuple[str, dict[str, Any]]]:
    """Returns a list of (job_name, container_dict) matching container_name."""
    matches = []
    for job_name in self.jobs:
      if container_name in self.containers[job_name]:
        matches.append((job_name, self.containers[job_name][container_name]))
      if container_name in self.init_containers[job_name]:
        matches.append((
            job_name,
            self.init_containers[job_name][container_name],
        ))
    return matches



class PathwaysJobSetTest(parameterized.TestCase):

  def _create_jobset(
      self,
      name="test-jobset",
      namespace="default",
      pathways_dir="gs://test-bucket",
      tpu_type="v5e",
      topology="4x8",
      num_slices=2,
      **kwargs,
  ) -> jobset.PathwaysJobSet:
    return jobset.PathwaysJobSet(
        name=name,
        namespace=namespace,
        pathways_dir=pathways_dir,
        tpu_type=tpu_type,
        topology=topology,
        num_slices=num_slices,
        **kwargs,
    )

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

  def test_headless_head_job_pod_spec(self):
    js = self._create_jobset(elastic_slices=2)

    config = js.to_dict()
    helper = JobSetManifestHelper(config)

    self.assertIn("pathways-head", helper.jobs)
    self.assertEqual(helper.jobs["pathways-head"]["replicas"], 1)
    pod_spec = helper.pod_specs["pathways-head"]
    self.assertTrue(pod_spec["hostNetwork"])
    self.assertEqual(pod_spec["dnsPolicy"], "ClusterFirstWithHostNet")
    self.assertEqual(pod_spec["restartPolicy"], "Never")

  def test_headless_head_job_containers(self):
    js = self._create_jobset(elastic_slices=2)

    config = js.to_dict()
    helper = JobSetManifestHelper(config)

    pod_spec = helper.pod_specs["pathways-head"]
    self.assertLen(pod_spec["containers"], 2)
    self.assertIn("pathways-rm", helper.containers["pathways-head"])
    self.assertIn("pathways-proxy", helper.containers["pathways-head"])

    rm_container = helper.containers["pathways-head"]["pathways-rm"]
    self.assertEqual(
        rm_container["image"],
        "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:latest",
    )

    proxy_container = helper.containers["pathways-head"]["pathways-proxy"]
    self.assertEqual(
        proxy_container["image"],
        "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:latest",
    )
    self.assertIn("--num_elastic_slices=2", proxy_container["args"])

  def test_non_headless_head_job_init_containers(self):
    user_pod_template = {
        "spec": {
            "containers": [{
                "name": "jax-tpu",
                "image": "ubuntu:latest",
            }]
        }
    }
    js = self._create_jobset(
        user_pod_template=user_pod_template,
        main_container_name="jax-tpu",
    )

    config = js.to_dict()
    helper = JobSetManifestHelper(config)

    pod_spec = helper.pod_specs["pathways-head"]
    self.assertLen(pod_spec["initContainers"], 2)
    self.assertIn("pathways-rm", helper.init_containers["pathways-head"])
    self.assertIn("pathways-proxy", helper.init_containers["pathways-head"])

    rm_container = helper.init_containers["pathways-head"]["pathways-rm"]
    proxy_container = helper.init_containers["pathways-head"]["pathways-proxy"]
    self.assertEqual(rm_container["restartPolicy"], "Always")
    self.assertEqual(proxy_container["restartPolicy"], "Always")

  def test_non_headless_head_job_jax_env(self):
    user_pod_template = {
        "spec": {
            "containers": [{
                "name": "jax-tpu",
                "image": "ubuntu:latest",
            }]
        }
    }
    js = self._create_jobset(
        user_pod_template=user_pod_template,
        main_container_name="jax-tpu",
    )

    config = js.to_dict()
    helper = JobSetManifestHelper(config)

    self.assertIn("jax-tpu", helper.containers["pathways-head"])
    main_container = helper.containers["pathways-head"]["jax-tpu"]
    env_names = [e["name"] for e in main_container["env"]]
    self.assertIn("PATHWAYS_HEAD", env_names)
    self.assertIn("JAX_PLATFORMS", env_names)
    self.assertIn("XCLOUD_ENVIRONMENT", env_names)
    self.assertIn("JAX_BACKEND_TARGET", env_names)

  def test_non_headless_head_job_annotations(self):
    user_pod_template = {
        "metadata": {"annotations": {"example.com/annotation": "value"}},
        "spec": {
            "containers": [{
                "name": "jax-tpu",
                "image": "ubuntu:latest",
            }]
        }
    }
    js = self._create_jobset(
        user_pod_template=user_pod_template,
        main_container_name="jax-tpu",
    )

    config = js.to_dict()
    helper = JobSetManifestHelper(config)

    head_job = helper.jobs["pathways-head"]
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
        name="test",
        restart_policy="Always"  # pyrefly: ignore[unexpected-keyword]
    )  # pytype: disable=wrong-keyword-args
    self.assertEqual(getattr(c, "restart_policy"), "Always")

  def test_worker_job_replicas(self):
    js = self._create_jobset(num_slices=2)

    config = js.to_dict()
    helper = JobSetManifestHelper(config)

    self.assertIn("pathways-worker", helper.jobs)
    self.assertEqual(helper.jobs["pathways-worker"]["replicas"], 2)

  def test_worker_job_completions(self):
    # 4x8 v5e topology has 32 chips. v5e has 4 chips per VM.
    # Total VMs = 32 / 4 = 8 VMs.
    js = self._create_jobset(topology="4x8", max_slice_restarts=3)

    config = js.to_dict()
    helper = JobSetManifestHelper(config)

    job_spec = helper.jobs["pathways-worker"]["template"]["spec"]
    self.assertEqual(job_spec["completions"], 8)
    self.assertEqual(job_spec["parallelism"], 8)
    # backoffLimit = num_vms * max_slice_restarts = 8 * 3 = 24
    self.assertEqual(job_spec["backoffLimit"], 24)

  def test_worker_job_pod_spec(self):
    js = self._create_jobset(termination_grace_period_seconds=60)

    config = js.to_dict()
    helper = JobSetManifestHelper(config)

    pod_spec = helper.pod_specs["pathways-worker"]
    self.assertTrue(pod_spec["hostNetwork"])
    self.assertEqual(pod_spec["dnsPolicy"], "ClusterFirstWithHostNet")
    self.assertEqual(pod_spec["restartPolicy"], "OnFailure")
    self.assertEqual(pod_spec["terminationGracePeriodSeconds"], 60)

  def test_worker_job_scheduling(self):
    js = self._create_jobset(tpu_type="v5e", topology="4x8")

    config = js.to_dict()
    helper = JobSetManifestHelper(config)

    pod_spec = helper.pod_specs["pathways-worker"]
    self.assertEqual(
        pod_spec["nodeSelector"]["cloud.google.com/gke-tpu-accelerator"],
        "tpu-v5-lite-podslice",
    )
    self.assertEqual(
        pod_spec["nodeSelector"]["cloud.google.com/gke-tpu-topology"], "4x8"
    )

  def test_worker_job_resources(self):
    # v5e has 4 chips per VM.
    js = self._create_jobset(tpu_type="v5e", topology="4x8")

    config = js.to_dict()
    helper = JobSetManifestHelper(config)

    self.assertIn("pathways-worker", helper.containers["pathways-worker"])
    container = helper.containers["pathways-worker"]["pathways-worker"]
    self.assertEqual(container["resources"]["limits"]["google.com/tpu"], "4")

  def test_worker_job_small_topology_completions(self):
    # 1x1 v5e topology has 1 chip. v5e has 4 chips per VM.
    # Since total_chips (1) < chips_per_vm (4), num_vms should be 1.
    js = self._create_jobset(tpu_type="v5e", topology="1x1", num_slices=1)

    config = js.to_dict()
    helper = JobSetManifestHelper(config)

    job_spec = helper.jobs["pathways-worker"]["template"]["spec"]
    self.assertEqual(job_spec["completions"], 1)
    self.assertEqual(job_spec["parallelism"], 1)
    # default backoffLimit = num_vms * 4 = 1 * 4 = 4
    self.assertEqual(job_spec["backoffLimit"], 4)

  @parameterized.parameters(True, False)
  def test_add_gcsfuse_read_only(self, read_only):
    pw_jobset = self._create_jobset(topology="2x2", num_slices=1)
    bucket_hash = int(hashlib.md5("my-bucket".encode()).hexdigest(), 16) % (10**8)
    expected_vol_name = f"gcsfuse-{bucket_hash}"

    pw_jobset.add_gcsfuse(
        containers="all",
        mount_path="/gcs/data",
        bucket="my-bucket",
        read_only=read_only,
    )
    helper = JobSetManifestHelper(pw_jobset.to_dict())

    rm_container = helper.containers["pathways-head"]["pathways-rm"]
    mounts = {m["name"]: m for m in rm_container.get("volumeMounts", [])}
    self.assertIn(expected_vol_name, mounts)
    self.assertEqual(mounts[expected_vol_name].get("readOnly", False), read_only)

  @parameterized.named_parameters(
      ("all", "all", ["pathways-rm", "pathways-proxy", "pathways-worker"]),
      ("worker", "pathways-worker", ["pathways-worker"]),
      ("explicit", ["pathways-worker", "pathways-rm"], ["pathways-worker", "pathways-rm"]),
  )
  def test_add_gcsfuse_container_filtering_headless(
      self, containers_param, expected_containers
  ):
    pw_jobset = self._create_jobset(topology="2x2", num_slices=1)
    bucket_hash = int(hashlib.md5("my-bucket".encode()).hexdigest(), 16) % (10**8)
    expected_vol_name = f"gcsfuse-{bucket_hash}"

    pw_jobset.add_gcsfuse(
        containers=containers_param,
        mount_path="/gcs/data",
        bucket="my-bucket",
    )
    helper = JobSetManifestHelper(pw_jobset.to_dict())

    all_possible_containers = ["pathways-rm", "pathways-proxy", "pathways-worker"]
    for c_name in all_possible_containers:
      matches = helper.get_all_containers_by_name(c_name)
      self.assertNotEmpty(matches)
      for job_name, container in matches:
        has_mount = any(m["mountPath"] == "/gcs/data" for m in container.get("volumeMounts", []))
        if c_name in expected_containers:
          self.assertTrue(has_mount, f"Expected {c_name} in {job_name} to have mount")
        else:
          self.assertFalse(has_mount, f"Expected {c_name} in {job_name} NOT to have mount")

  @parameterized.named_parameters(
      ("all", "all", ["pathways-rm", "pathways-proxy", "pathways-worker", "jax-tpu"]),
      ("explicit", ["pathways-worker", "jax-tpu"], ["pathways-worker", "jax-tpu"]),
      ("explicit_rm", ["pathways-worker", "pathways-rm"], ["pathways-worker", "pathways-rm"]),
  )
  def test_add_gcsfuse_container_filtering_non_headless(
      self, containers_param, expected_containers
  ):
    user_pod_template = {
        "spec": {
            "containers": [{
                "name": "jax-tpu",
                "image": "gcr.io/my-project/jax-tpu:latest",
            }]
        }
    }
    pw_jobset = self._create_jobset(
        topology="2x2",
        num_slices=1,
        user_pod_template=user_pod_template,
        main_container_name="jax-tpu",
    )
    bucket_hash = int(hashlib.md5("my-bucket".encode()).hexdigest(), 16) % (10**8)
    expected_vol_name = f"gcsfuse-{bucket_hash}"

    pw_jobset.add_gcsfuse(
        containers=containers_param,
        mount_path="/gcs/data",
        bucket="my-bucket",
    )
    helper = JobSetManifestHelper(pw_jobset.to_dict())

    all_possible_containers = ["pathways-rm", "pathways-proxy", "pathways-worker", "jax-tpu"]
    for c_name in all_possible_containers:
      matches = helper.get_all_containers_by_name(c_name)
      self.assertNotEmpty(matches)
      for job_name, container in matches:
        has_mount = any(m["mountPath"] == "/gcs/data" for m in container.get("volumeMounts", []))
        if c_name in expected_containers:
          self.assertTrue(has_mount, f"Expected {c_name} in {job_name} to have mount")
        else:
          self.assertFalse(has_mount, f"Expected {c_name} in {job_name} NOT to have mount")

  def test_add_gcsfuse_volumes_and_annotations(self):
    pw_jobset = self._create_jobset(topology="2x2", num_slices=1)
    bucket_hash = int(hashlib.md5("my-bucket".encode()).hexdigest(), 16) % (10**8)
    expected_vol_name = f"gcsfuse-{bucket_hash}"

    pw_jobset.add_gcsfuse(
        containers="pathways-worker",
        mount_path="/gcs/data",
        bucket="my-bucket",
    )
    helper = JobSetManifestHelper(pw_jobset.to_dict())

    # Worker job should have volume and annotations
    self.assertIn(expected_vol_name, helper.volumes["pathways-worker"])
    vol = helper.volumes["pathways-worker"][expected_vol_name]
    self.assertEqual(vol["csi"]["driver"], "gcsfuse.csi.storage.gke.io")
    self.assertEqual(vol["csi"]["volumeAttributes"]["bucketName"], "my-bucket")
    self.assertEqual(helper.job_metadatas["pathways-worker"].get("annotations", {}).get("gke-gcsfuse/volumes"), "true")
    self.assertEqual(helper.pod_metadatas["pathways-worker"].get("annotations", {}).get("gke-gcsfuse/volumes"), "true")

    # Head job should NOT have volume or annotations
    self.assertNotIn(expected_vol_name, helper.volumes["pathways-head"])
    self.assertNotEqual(helper.job_metadatas["pathways-head"].get("annotations", {}).get("gke-gcsfuse/volumes"), "true")
    self.assertNotEqual(helper.pod_metadatas["pathways-head"].get("annotations", {}).get("gke-gcsfuse/volumes"), "true")

  def test_add_gcsfuse_handles_none_metadata(self):
    pw_jobset = self._create_jobset(topology="2x2", num_slices=1)
    
    # Force metadata to be None to simulate imported templates or raw specs without metadata
    pw_jobset._head_job_template.metadata = None
    pw_jobset._head_job_template.spec.template.metadata = None
    pw_jobset._worker_job_template.metadata = None
    pw_jobset._worker_job_template.spec.template.metadata = None

    # Should not crash and should correctly add annotations
    pw_jobset.add_gcsfuse(
        containers="all",
        mount_path="/gcs/data",
        bucket="my-bucket",
    )
    helper = JobSetManifestHelper(pw_jobset.to_dict())
    
    self.assertEqual(helper.job_metadatas["pathways-head"].get("annotations", {}).get("gke-gcsfuse/volumes"), "true")
    self.assertEqual(helper.pod_metadatas["pathways-head"].get("annotations", {}).get("gke-gcsfuse/volumes"), "true")
    self.assertEqual(helper.job_metadatas["pathways-worker"].get("annotations", {}).get("gke-gcsfuse/volumes"), "true")
    self.assertEqual(helper.pod_metadatas["pathways-worker"].get("annotations", {}).get("gke-gcsfuse/volumes"), "true")

  def test_add_gcsfuse_preserves_existing_metadata(self):
    pw_jobset = self._create_jobset(topology="2x2", num_slices=1)
    
    # Pre-populate metadata, annotations, and labels
    pw_jobset._head_job_template.metadata = client.V1ObjectMeta(
        labels={"existing-job-label": "value"},
        annotations={"existing-job-anno": "value"}
    )
    pw_jobset._head_job_template.spec.template.metadata = client.V1ObjectMeta(
        labels={"existing-pod-label": "value"},
        annotations={"existing-pod-anno": "value"}
    )

    pw_jobset.add_gcsfuse(
        containers="all",
        mount_path="/gcs/data",
        bucket="my-bucket",
    )
    helper = JobSetManifestHelper(pw_jobset.to_dict())

    # Verify existing annotations and labels are preserved, and new annotation is added
    job_meta = helper.job_metadatas["pathways-head"]
    self.assertEqual(job_meta.get("labels", {}).get("existing-job-label"), "value")
    self.assertEqual(job_meta.get("annotations", {}).get("existing-job-anno"), "value")
    self.assertEqual(job_meta.get("annotations", {}).get("gke-gcsfuse/volumes"), "true")

    pod_meta = helper.pod_metadatas["pathways-head"]
    self.assertEqual(pod_meta.get("labels", {}).get("existing-pod-label"), "value")
    self.assertEqual(pod_meta.get("annotations", {}).get("existing-pod-anno"), "value")
    self.assertEqual(pod_meta.get("annotations", {}).get("gke-gcsfuse/volumes"), "true")


if __name__ == "__main__":
  absltest.main()
