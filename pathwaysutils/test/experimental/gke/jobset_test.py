import hashlib
import itertools
import os
from typing import Any
import unittest
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import yaml

try:
  import kubernetes
except ImportError:
  raise unittest.SkipTest(
      "kubernetes is not installed (requires pathwaysutils[test])"
  )

from kubernetes import client
from kubernetes import config as k8s_config
from pathwaysutils.experimental.gke import jobset


def normalize_k8s_spec(spec: Any) -> Any:
  if isinstance(spec, dict):
    result = {}
    for k, v in spec.items():
      if k == "env" and isinstance(v, list):
        result[k] = sorted(
            [normalize_k8s_spec(x) for x in v], key=lambda x: x.get("name", "")
        )
      elif k == "ports" and isinstance(v, list):
        result[k] = sorted(
            [normalize_k8s_spec(x) for x in v],
            key=lambda x: x.get("containerPort", 0),
        )
      elif k == "volumeMounts" and isinstance(v, list):
        result[k] = sorted(
            [normalize_k8s_spec(x) for x in v],
            key=lambda x: x.get("mountPath", ""),
        )
      else:
        result[k] = normalize_k8s_spec(v)
    return result
  elif isinstance(spec, list):
    return [normalize_k8s_spec(x) for x in spec]
  return spec


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
    self.assertNotIn("hostNetwork", pod_spec)
    self.assertNotIn("dnsPolicy", pod_spec)
    self.assertEqual(pod_spec["restartPolicy"], "Never")

  def test_headless_head_job_containers(self):
    js = self._create_jobset(elastic_slices=2)

    config = js.to_dict()
    helper = JobSetManifestHelper(config)

    pod_spec = helper.pod_specs["pathways-head"]
    self.assertNotIn("initContainers", pod_spec)
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
    self.assertTrue(
        any(
            e.get("name") == "IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS"
            and e.get("value") == "true"
            for e in proxy_container.get("env", [])
        )
    )

  def test_configurable_host_network_and_dns_policy(self):
    js = self._create_jobset(
        head_pod_spec_kwargs={
            "host_network": True,
            "dns_policy": "ClusterFirstWithHostNet",
        },
        worker_pod_spec_kwargs={
            "host_network": True,
            "dns_policy": "ClusterFirstWithHostNet",
        },
    )
    config = js.to_dict()
    helper = JobSetManifestHelper(config)

    head_pod_spec = helper.pod_specs["pathways-head"]
    self.assertTrue(head_pod_spec["hostNetwork"])
    self.assertEqual(head_pod_spec["dnsPolicy"], "ClusterFirstWithHostNet")

    worker_pod_spec = helper.pod_specs["pathways-worker"]
    self.assertTrue(worker_pod_spec["hostNetwork"])
    self.assertEqual(worker_pod_spec["dnsPolicy"], "ClusterFirstWithHostNet")

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
    self.assertNotIn("hostNetwork", pod_spec)
    self.assertNotIn("dnsPolicy", pod_spec)
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
    # default backoffLimit = num_vms * max_slice_restarts = 1 * 1_000_000 = 1_000_000
    self.assertEqual(job_spec["backoffLimit"], 1_000_000)

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
  def test_add_gcsfuse_container_filtering(
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
    self.assertEqual(
        vol["csi"]["volumeAttributes"]["mountOptions"], "implicit-dirs"
    )
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

  def test_add_gcsfuse_preserves_existing_volumes(self):
    pw_jobset = self._create_jobset(topology="2x2", num_slices=1)

    # Pre-populate volumes in head and worker pod specs.
    pw_jobset._head_job_template.spec.template.spec.volumes = [
        client.V1Volume(name="preexisting-head-vol")
    ]
    pw_jobset._worker_job_template.spec.template.spec.volumes = [
        client.V1Volume(name="preexisting-worker-vol")
    ]

    pw_jobset.add_gcsfuse(
        containers="all",
        mount_path="/gcs/data",
        bucket="my-bucket",
    )
    helper = JobSetManifestHelper(pw_jobset.to_dict())

    # Verify existing volumes and newly added GCSFuse volumes coexist.
    self.assertIn("preexisting-head-vol", helper.volumes["pathways-head"])
    self.assertIn("preexisting-worker-vol", helper.volumes["pathways-worker"])

  def test_add_colocated_python_handles_none_volumes(self):
    pw_jobset = self._create_jobset(topology="2x2", num_slices=1)

    # Force volumes to be None
    pw_jobset._worker_job_template.spec.template.spec.volumes = None

    # Should not crash and should correctly add volume
    pw_jobset.add_colocated_python(image="gcr.io/my-project/colocated-python:custom")
    helper = JobSetManifestHelper(pw_jobset.to_dict())

    self.assertIn("shared-memory", helper.volumes["pathways-worker"])

  def test_add_colocated_python_sidecar(self):
    pw_jobset = self._create_jobset(topology="2x2", num_slices=1)

    pw_jobset.add_colocated_python(image="gcr.io/my-project/colocated-python:custom")
    helper = JobSetManifestHelper(pw_jobset.to_dict())

    # The sidecar runs as a native sidecar so it starts before pathways-worker.
    # The restart policy is checked on the container object because older
    # kubernetes clients do not serialize `V1Container.restart_policy`.
    worker_spec = pw_jobset.worker_job_template.spec.template.spec
    self.assertEqual(
        [c.restart_policy for c in worker_spec.init_containers],
        ["Always"],
    )
    self.assertNotIn(
        "colocated-python-sidecar", helper.containers["pathways-worker"]
    )
    self.assertIn(
        "colocated-python-sidecar", helper.init_containers["pathways-worker"]
    )
    sidecar = helper.init_containers["pathways-worker"][
        "colocated-python-sidecar"
    ]
    self.assertEqual(sidecar["image"], "gcr.io/my-project/colocated-python:custom")
    self.assertTrue(
        any(
            m["name"] == "shared-memory" and m["mountPath"] == "/tmp/shared-memory"
            for m in sidecar["volumeMounts"]
        )
    )
    self.assertTrue(
        any(
            e["name"] == "CLOUD_PATHWAYS_SIDECAR_SHM_DIRECTORY"
            and e["value"] == "/tmp/shared-memory"
            for e in sidecar["env"]
        )
    )

  def test_add_colocated_python_preserves_init_containers(self):
    pw_jobset = self._create_jobset(topology="2x2", num_slices=1)

    # Pre-populate init container on worker pod
    worker_spec = pw_jobset._worker_job_template.spec.template.spec
    existing_init = client.V1Container(name="existing-init-container", image="ubuntu:latest")
    worker_spec.init_containers = [existing_init]

    pw_jobset.add_colocated_python(image="gcr.io/my-project/colocated-python:custom")
    helper = JobSetManifestHelper(pw_jobset.to_dict())

    # Verify both exist, and the existing init container still runs first.
    self.assertEqual(
        list(helper.init_containers["pathways-worker"]),
        ["existing-init-container", "colocated-python-sidecar"],
    )

  def test_add_colocated_python_volume_default(self):
    pw_jobset = self._create_jobset(topology="2x2", num_slices=1)

    pw_jobset.add_colocated_python(image="gcr.io/my-project/colocated-python:custom")
    helper = JobSetManifestHelper(pw_jobset.to_dict())

    self.assertIn("shared-memory", helper.volumes["pathways-worker"])
    shm_vol = helper.volumes["pathways-worker"]["shared-memory"]
    self.assertNotIn("sizeLimit", shm_vol["emptyDir"])

  def test_add_colocated_python_worker_mount(self):
    pw_jobset = self._create_jobset(topology="2x2", num_slices=1)

    pw_jobset.add_colocated_python(image="gcr.io/my-project/colocated-python:custom")
    helper = JobSetManifestHelper(pw_jobset.to_dict())

    self.assertIn("pathways-worker", helper.containers["pathways-worker"])
    worker_container = helper.containers["pathways-worker"]["pathways-worker"]
    self.assertTrue(
        any(
            m["name"] == "shared-memory" and m["mountPath"] == "/tmp/shared-memory"
            for m in worker_container["volumeMounts"]
        )
    )
    self.assertTrue(
        any(
            e["name"] == "CLOUD_PATHWAYS_SIDECAR_SHM_DIRECTORY"
            and e["value"] == "/tmp/shared-memory"
            for e in worker_container["env"]
        )
    )

  def test_add_colocated_python_custom_shm(self):
    pw_jobset = self._create_jobset(topology="2x2", num_slices=1)

    pw_jobset.add_colocated_python(
        image="gcr.io/my-project/colocated-python:custom",
        shm_mount_path="/tmp/custom-shm",
        shm_size_limit="50Gi",
    )
    helper = JobSetManifestHelper(pw_jobset.to_dict())

    self.assertIn(
        "colocated-python-sidecar", helper.init_containers["pathways-worker"]
    )
    sidecar = helper.init_containers["pathways-worker"][
        "colocated-python-sidecar"
    ]
    self.assertTrue(
        any(
            m["name"] == "shared-memory" and m["mountPath"] == "/tmp/custom-shm"
            for m in sidecar["volumeMounts"]
        )
    )
    self.assertTrue(
        any(
            e["name"] == "CLOUD_PATHWAYS_SIDECAR_SHM_DIRECTORY"
            and e["value"] == "/tmp/custom-shm"
            for e in sidecar["env"]
        )
    )

    self.assertIn("shared-memory", helper.volumes["pathways-worker"])
    shm_vol = helper.volumes["pathways-worker"]["shared-memory"]
    self.assertEqual(shm_vol["emptyDir"]["sizeLimit"], "50Gi")

    self.assertIn("pathways-worker", helper.containers["pathways-worker"])
    worker_container = helper.containers["pathways-worker"]["pathways-worker"]
    self.assertTrue(
        any(
            m["name"] == "shared-memory" and m["mountPath"] == "/tmp/custom-shm"
            for m in worker_container["volumeMounts"]
        )
    )
    self.assertTrue(
        any(
            e["name"] == "CLOUD_PATHWAYS_SIDECAR_SHM_DIRECTORY"
            and e["value"] == "/tmp/custom-shm"
            for e in worker_container["env"]
        )
    )

  @parameterized.named_parameters(
      ("default_shm", None, "/tmp/shared-memory"),
      ("custom_shm", "/tmp/custom-shm", "/tmp/custom-shm"),
  )
  def test_add_colocated_python_sets_worker_shm_flag(
      self, shm_mount_path, expected_path
  ):
    pw_jobset = self._create_jobset(topology="2x2", num_slices=1)
    kwargs = {"shm_mount_path": shm_mount_path} if shm_mount_path else {}
    pw_jobset.add_colocated_python(
        image="gcr.io/my-project/colocated-python:latest", **kwargs
    )
    helper = JobSetManifestHelper(pw_jobset.to_dict())

    worker_args = helper.containers["pathways-worker"]["pathways-worker"][
        "args"
    ]
    self.assertEqual(
        [a for a in worker_args if "cloud_pathways_sidecar_shm_directory" in a],
        [f"--cloud_pathways_sidecar_shm_directory={expected_path}"],
    )

  def test_add_colocated_python_keeps_existing_worker_shm_flag(self):
    pw_jobset = self._create_jobset(topology="2x2", num_slices=1)
    worker_spec = pw_jobset.worker_job_template.spec.template.spec
    worker_spec.containers[0].args.append(
        "--cloud_pathways_sidecar_shm_directory=/tmp/preset"
    )
    pw_jobset.add_colocated_python(
        image="gcr.io/my-project/colocated-python:latest"
    )
    helper = JobSetManifestHelper(pw_jobset.to_dict())

    worker_args = helper.containers["pathways-worker"]["pathways-worker"][
        "args"
    ]
    self.assertEqual(
        [a for a in worker_args if "cloud_pathways_sidecar_shm_directory" in a],
        ["--cloud_pathways_sidecar_shm_directory=/tmp/preset"],
    )

  def test_add_colocated_python_env(self):
    pw_jobset = self._create_jobset(topology="2x2", num_slices=1)
    pw_jobset.add_colocated_python(
        image="gcr.io/my-project/colocated-python:latest",
        env={"LOGLEVEL": "DEBUG", "GRPC_SERVER_ADDRESS": "0.0.0.0:50052"},
    )
    helper = JobSetManifestHelper(pw_jobset.to_dict())

    sidecar = helper.init_containers["pathways-worker"][
        "colocated-python-sidecar"
    ]
    self.assertEqual(
        sidecar["env"],
        [
            {"name": "GRPC_SERVER_ADDRESS", "value": "0.0.0.0:50052"},
            {
                "name": "CLOUD_PATHWAYS_SIDECAR_SHM_DIRECTORY",
                "value": "/tmp/shared-memory",
            },
            {"name": "LOGLEVEL", "value": "DEBUG"},
        ],
    )

  def test_add_colocated_python_container_kwargs(self):
    pw_jobset = self._create_jobset(topology="2x2", num_slices=1)
    pw_jobset.add_colocated_python(
        image="gcr.io/my-project/colocated-python:latest",
        resources=client.V1ResourceRequirements(
            limits={"cpu": "4", "memory": "16Gi"}
        ),
        volume_mounts=[
            client.V1VolumeMount(name="extra-data", mount_path="/data")
        ],
    )
    helper = JobSetManifestHelper(pw_jobset.to_dict())

    sidecar = helper.init_containers["pathways-worker"][
        "colocated-python-sidecar"
    ]
    self.assertEqual(
        sidecar["resources"], {"limits": {"cpu": "4", "memory": "16Gi"}}
    )
    # User volume mounts are appended to the required default mounts.
    self.assertEqual(
        sidecar["volumeMounts"],
        [
            {"mountPath": "/tmp", "name": "shared-tmp"},
            {"mountPath": "/tmp/shared-memory", "name": "shared-memory"},
            {"mountPath": "/data", "name": "extra-data"},
        ],
    )

  @parameterized.named_parameters(
      ("name", {"name": "my-sidecar"}, "Cannot set 'name'"),
      (
          "unknown_field",
          {"not_a_field": True},
          "Unknown client.V1Container fields for colocated-python-sidecar",
      ),
  )
  def test_add_colocated_python_invalid_container_kwargs(self, kwargs, error):
    pw_jobset = self._create_jobset(topology="2x2", num_slices=1)
    with self.assertRaisesRegex(ValueError, error):
      pw_jobset.add_colocated_python(
          image="gcr.io/my-project/colocated-python:latest", **kwargs
      )

  # Reference similar GKE / JobSet custom object unit test suites:
  # - Google3 GKE JobSet test suite: //depot/google3/cloud/ai/map/catmint/supervisor/client/python/orchestrators/gke_callbacks_test.py
  # - Upstream Kubernetes SIGs JobSet unit tests: https://github.com/kubernetes-sigs/jobset/blob/main/pkg/controllers/jobset_controller_test.go
  @mock.patch("kubernetes.config.load_kube_config")
  @mock.patch("kubernetes.config.load_incluster_config")
  @mock.patch("kubernetes.client.CustomObjectsApi")
  def test_apply_create(
      self, mock_custom_objects_api, mock_load_incluster, mock_load_kube
  ):
    """Tests deploying a JobSet to GKE via Kubernetes CustomObjectsApi.

    Modeled after official GKE JobSet unit test patterns (e.g. gke_callbacks_test.py).
    """
    mock_api = mock_custom_objects_api.return_value
    # Mock GET to return 404 (not exists).
    from kubernetes.client.rest import ApiException

    mock_api.get_namespaced_custom_object.side_effect = ApiException(status=404)

    pw_jobset = jobset.PathwaysJobSet(
        name="test-workload",
        namespace="default",
        pathways_dir="gs://bucket/scratch",
        tpu_type="v5e",
        topology="2x2",
        num_slices=1,
    )
    pw_jobset.apply()

    mock_api.create_namespaced_custom_object.assert_called_once_with(
        group="jobset.x-k8s.io",
        version="v1alpha2",
        namespace="default",
        plural="jobsets",
        body=pw_jobset.to_dict(),
        field_manager="pathwaysutils",
    )

  @mock.patch("kubernetes.config.load_kube_config")
  @mock.patch("kubernetes.config.load_incluster_config")
  @mock.patch("kubernetes.client.CustomObjectsApi")
  def test_apply_exists_recreate(
      self, mock_custom_objects_api, mock_load_incluster, mock_load_kube
  ):
    mock_api = mock_custom_objects_api.return_value
    # Mock GET to return success (exists).
    mock_api.get_namespaced_custom_object.return_value = {}
    # Mock GET after delete to return 404.
    from kubernetes.client.rest import ApiException

    mock_api.get_namespaced_custom_object.side_effect = [
        {},
        ApiException(status=404),
    ]

    pw_jobset = jobset.PathwaysJobSet(
        name="test-workload",
        namespace="default",
        pathways_dir="gs://bucket/scratch",
        tpu_type="v5e",
        topology="2x2",
        num_slices=1,
    )
    pw_jobset.apply(recreate=True)

    mock_api.delete_namespaced_custom_object.assert_called_once_with(
        "jobset.x-k8s.io", "v1alpha2", "default", "jobsets", "test-workload"
    )
    mock_api.create_namespaced_custom_object.assert_called_once_with(
        group="jobset.x-k8s.io",
        version="v1alpha2",
        namespace="default",
        plural="jobsets",
        body=pw_jobset.to_dict(),
        field_manager="pathwaysutils",
    )

  @mock.patch("kubernetes.config.load_kube_config")
  @mock.patch("kubernetes.config.load_incluster_config")
  @mock.patch("kubernetes.client.CustomObjectsApi")
  def test_apply_exists_no_recreate_fails(
      self, mock_custom_objects_api, mock_load_incluster, mock_load_kube
  ):
    mock_api = mock_custom_objects_api.return_value
    # Mock GET to return success (exists).
    mock_api.get_namespaced_custom_object.return_value = {}

    pw_jobset = jobset.PathwaysJobSet(
        name="test-workload",
        namespace="default",
        pathways_dir="gs://bucket/scratch",
        tpu_type="v5e",
        topology="2x2",
        num_slices=1,
    )
    with self.assertRaises(RuntimeError):
      pw_jobset.apply(recreate=False)

  def test_export_import_roundtrip(self):
    pw_jobset = jobset.PathwaysJobSet(
        name="test-workload",
        namespace="default",
        pathways_dir="gs://bucket/scratch",
        tpu_type="v5e",
        topology="2x2",
        num_slices=1,
    )
    pw_jobset.add_colocated_python(image="gcr.io/my-project/colocated-python:custom")
    pw_jobset.add_gcsfuse(
        containers="pathways-worker", mount_path="/tmp/gcs", bucket="my-bucket"
    )

    # Export.
    temp_filepath = os.path.join(self.create_tempdir().full_path, "jobset.yaml")
    pw_jobset.export_yaml(temp_filepath)

    # Import.
    imported_jobset = jobset.PathwaysJobSet.import_yaml(temp_filepath)

    # Verify they are semantically identical.
    self.assertEqual(
        normalize_k8s_spec(pw_jobset.to_dict()),
        normalize_k8s_spec(imported_jobset.to_dict()),
    )

  def test_configurable_images(self):
    pw_jobset = jobset.PathwaysJobSet(
        name="test-workload",
        namespace="default",
        pathways_dir="gs://bucket/scratch",
        tpu_type="v5e",
        topology="2x2",
        num_slices=1,
        pathways_version="v2.0",
        pathways_rm_and_worker_image="gcr.io/custom/server",
        pathways_proxy_image="gcr.io/custom/proxy:tagged",
    )

    self.assertEqual(
        pw_jobset.pathways_rm_and_worker_image, "gcr.io/custom/server"
    )
    self.assertEqual(
        pw_jobset.pathways_proxy_image, "gcr.io/custom/proxy:tagged"
    )

    # Verify head job containers.
    head_containers = {
        c.name: c.image
        for c in pw_jobset.head_job_template.spec.template.spec.containers
    }
    self.assertEqual(head_containers["pathways-rm"], "gcr.io/custom/server:v2.0")
    self.assertEqual(
        head_containers["pathways-proxy"], "gcr.io/custom/proxy:tagged"
    )

    # Verify worker job container.
    worker_image = (
        pw_jobset.worker_job_template.spec.template.spec.containers[0].image
    )
    self.assertEqual(worker_image, "gcr.io/custom/server:v2.0")

    # Verify roundtrip import.
    temp_filepath = os.path.join(
        self.create_tempdir().full_path, "jobset_custom.yaml"
    )
    pw_jobset.export_yaml(temp_filepath)
    imported_jobset = jobset.PathwaysJobSet.import_yaml(temp_filepath)

    self.assertEqual(
        imported_jobset.pathways_rm_and_worker_image,
        "gcr.io/custom/server:v2.0",
    )
    self.assertEqual(
        imported_jobset.pathways_proxy_image, "gcr.io/custom/proxy:tagged"
    )

  def test_import_validation_failures(self):
    temp_dir = self.create_tempdir().full_path

    # 1. Missing kind.
    invalid_config1 = {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "metadata": {"name": "test"},
        "spec": {"replicatedJobs": []},
    }
    path1 = os.path.join(temp_dir, "invalid1.yaml")
    with open(path1, "w") as f:
      yaml.dump(invalid_config1, f)
    with self.assertRaisesRegex(ValueError, "Resource kind is not JobSet"):
      jobset.PathwaysJobSet.import_yaml(path1)

    # 2. Missing head.
    invalid_config2 = {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "metadata": {"name": "test"},
        "spec": {
            "replicatedJobs": [
                {"name": "worker", "replicas": 1, "template": {}}
            ]
        },
    }
    path2 = os.path.join(temp_dir, "invalid2.yaml")
    with open(path2, "w") as f:
      yaml.dump(invalid_config2, f)
    with self.assertRaisesRegex(ValueError, "Missing head replicated job"):
      jobset.PathwaysJobSet.import_yaml(path2)

    # 3. Missing worker.
    invalid_config3 = {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "metadata": {"name": "test"},
        "spec": {
            "replicatedJobs": [
                {"name": "pathways-head", "replicas": 1, "template": {}}
            ]
        },
    }
    path3 = os.path.join(temp_dir, "invalid3.yaml")
    with open(path3, "w") as f:
      yaml.dump(invalid_config3, f)
    with self.assertRaisesRegex(ValueError, "Missing worker replicated job"):
      jobset.PathwaysJobSet.import_yaml(path3)

  def test_labels_and_annotations(self):
    pw_jobset = jobset.PathwaysJobSet(
        name="test-workload",
        namespace="default",
        pathways_dir="gs://bucket/scratch",
        tpu_type="v5e",
        topology="2x2",
        num_slices=1,
        labels={"key1": "val1"},
        annotations={"key2": "val2"},
    )
    config = pw_jobset.to_dict()
    self.assertEqual(config["metadata"]["labels"], {"key1": "val1"})
    self.assertEqual(config["metadata"]["annotations"], {"key2": "val2"})

  def test_direct_mutation(self):
    """Verifies that directly mutating Kubernetes JobTemplateSpec objects is preserved in to_dict()."""
    pw_jobset = jobset.PathwaysJobSet(
        name="test-workload",
        namespace="default",
        pathways_dir="gs://bucket/scratch",
        tpu_type="v5e",
        topology="2x2",
        num_slices=1,
    )
    # Directly modify the Kubernetes template spec objects on the instance.
    head_spec = pw_jobset.head_job_template.spec.template.spec
    head_spec.active_deadline_seconds = 100

    worker_spec = pw_jobset.worker_job_template.spec.template.spec
    worker_spec.active_deadline_seconds = 200

    # Verify that the direct mutations are preserved and serialized in to_dict().
    config = pw_jobset.to_dict()
    replicated_jobs = {
        job["name"]: job for job in config["spec"]["replicatedJobs"]
    }
    self.assertEqual(
        replicated_jobs["pathways-head"]["template"]["spec"]["template"]["spec"][
            "activeDeadlineSeconds"
        ],
        100,
    )
    self.assertEqual(
        replicated_jobs["pathways-worker"]["template"]["spec"]["template"]["spec"][
            "activeDeadlineSeconds"
        ],
        200,
    )

  def test_failure_policy(self):
    pw_jobset = jobset.PathwaysJobSet(
        name="test-workload",
        namespace="default",
        pathways_dir="gs://bucket/scratch",
        tpu_type="v5e",
        topology="2x2",
        num_slices=1,
        max_restarts=5,
    )
    config = pw_jobset.to_dict()
    self.assertEqual(config["spec"]["failurePolicy"]["maxRestarts"], 5)

  def test_shared_pathways_service(self):
    pw_jobset = self._create_jobset(
        name="test-sps",
        shared_pathways_service=True,
    )

    config = pw_jobset.to_dict()
    helper = JobSetManifestHelper(config)

    self.assertNotIn("successPolicy", config["spec"])

    self.assertTrue(config["spec"]["network"]["enableDNSHostnames"])
    self.assertTrue(config["spec"]["network"]["publishNotReadyAddresses"])

    self.assertIn("pathways-head", helper.jobs)
    pod_spec = helper.pod_specs["pathways-head"]

    # Head job should only have pathways-rm container, no pathways-proxy.
    self.assertNotIn("initContainers", pod_spec)
    self.assertIn("pathways-rm", helper.containers["pathways-head"])
    self.assertNotIn("pathways-proxy", helper.containers["pathways-head"])
    self.assertLen(pod_spec["containers"], 1)

  @parameterized.named_parameters(
      ("one_slice", 1, False),
      ("two_slices", 2, False),
      ("two_slices_sps", 2, True),
  )
  def test_rm_instance_count_matches_num_slices(
      self, num_slices, shared_pathways_service
  ):
    pw_jobset = self._create_jobset(
        num_slices=num_slices,
        shared_pathways_service=shared_pathways_service,
    )
    helper = JobSetManifestHelper(pw_jobset.to_dict())
    rm_args = helper.containers["pathways-head"]["pathways-rm"]["args"]
    self.assertIn(f"--instance_count={num_slices}", rm_args)

  def test_add_user_workload(self):
    pw_jobset = self._create_jobset(topology="2x2", num_slices=1)
    pw_jobset.add_user_workload(
        image="us-docker.pkg.dev/my-project/test:v1",
        command="python3 -m test_module",
        env={"CUSTOM_ENV": "123"},
    )

    config = pw_jobset.to_dict()
    helper = JobSetManifestHelper(config)

    # Verify JobSet spec-level success policy targets pathways-head
    self.assertEqual(
        config["spec"]["successPolicy"],
        {
            "operator": "All",
            "targetReplicatedJobs": ["pathways-head"],
        },
    )

    # Verify head pod has user-workload container in containers
    self.assertIn("user-workload", helper.containers["pathways-head"])
    user_c = helper.containers["pathways-head"]["user-workload"]
    self.assertEqual(user_c["image"], "us-docker.pkg.dev/my-project/test:v1")
    self.assertEqual(user_c["command"], ["sh", "-c", "python3 -m test_module"])
    self.assertNotIn("resources", user_c)
    self.assertTrue(
        any(
            e["name"] == "CUSTOM_ENV" and e["value"] == "123"
            for e in user_c["env"]
        )
    )
    self.assertTrue(
        any(
            e["name"] == "JAX_PLATFORMS" and e["value"] == "proxy"
            for e in user_c["env"]
        )
    )
    self.assertTrue(
        any(
            e["name"] == "JAX_BACKEND_TARGET"
            and e["value"] == "grpc://localhost:29000"
            for e in user_c["env"]
        )
    )
    self.assertTrue(
        any(
            e["name"] == "IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS"
            and e["value"] == "true"
            for e in user_c["env"]
        )
    )

    # Verify RM and Proxy are moved to initContainers (native sidecars)
    self.assertIn("pathways-rm", helper.init_containers["pathways-head"])
    self.assertIn("pathways-proxy", helper.init_containers["pathways-head"])
    self.assertNotIn("pathways-rm", helper.containers["pathways-head"])
    self.assertNotIn("pathways-proxy", helper.containers["pathways-head"])
    self.assertIn("user-workload", helper.containers["pathways-head"])

  def test_add_user_workload_roundtrip(self):
    pw_jobset = self._create_jobset(topology="2x2", num_slices=1)
    pw_jobset.add_user_workload(
        image="us-docker.pkg.dev/my-project/test:v1",
        command=["python3", "test.py"],
    )

    temp_filepath = os.path.join(
        self.create_tempdir().full_path, "jobset_user_workload.yaml"
    )
    pw_jobset.export_yaml(temp_filepath)
    imported = jobset.PathwaysJobSet.import_yaml(temp_filepath)

    self.assertEqual(
        normalize_k8s_spec(pw_jobset.to_dict()),
        normalize_k8s_spec(imported.to_dict()),
    )

  def test_add_user_workload_preserves_existing_containers(self):
    pw_jobset = self._create_jobset(topology="2x2", num_slices=1)
    custom_container = client.V1Container(
        name="custom-sidecar", image="custom:v1"
    )
    pw_jobset._head_job_template.spec.template.spec.containers.append(
        custom_container
    )
    custom_resources = client.V1ResourceRequirements(
        limits={"cpu": "8", "memory": "32G"}
    )
    pw_jobset.add_user_workload(
        name="user-workload-1",
        image="us-docker.pkg.dev/my-project/test:v1",
        command="python3 -m test1",
        env={"IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS": "false"},
        resources=custom_resources,
    )
    pw_jobset.add_user_workload(
        name="user-workload-2",
        image="us-docker.pkg.dev/my-project/test:v2",
        command="python3 -m test2",
    )
    config = pw_jobset.to_dict()
    helper = JobSetManifestHelper(config)

    self.assertIn("pathways-rm", helper.init_containers["pathways-head"])
    self.assertIn("pathways-proxy", helper.init_containers["pathways-head"])
    self.assertLen(helper.pod_specs["pathways-head"]["initContainers"], 2)
    self.assertIn("custom-sidecar", helper.containers["pathways-head"])
    self.assertIn("user-workload-1", helper.containers["pathways-head"])
    self.assertIn("user-workload-2", helper.containers["pathways-head"])
    self.assertLen(helper.pod_specs["pathways-head"]["containers"], 3)

    c1 = helper.containers["pathways-head"]["user-workload-1"]
    self.assertEqual(c1["resources"], {"limits": {"cpu": "8", "memory": "32G"}})
    insecure_envs = [
        e["value"]
        for e in c1["env"]
        if e["name"] == "IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS"
    ]
    self.assertEqual(insecure_envs, ["false"])

    # Calling add_user_workload with existing name replaces it rather than appending duplicate
    pw_jobset.add_user_workload(
        name="user-workload-1",
        image="us-docker.pkg.dev/my-project/test:v3",
        command="python3 -m test1_updated",
    )
    config2 = pw_jobset.to_dict()
    helper2 = JobSetManifestHelper(config2)
    self.assertLen(helper2.pod_specs["pathways-head"]["containers"], 3)
    self.assertEqual(
        helper2.containers["pathways-head"]["user-workload-1"]["image"],
        "us-docker.pkg.dev/my-project/test:v3",
    )

  def test_configurable_priority_class_and_head_node_selector(self):
    # Default: no priority class or custom head pod node selector
    default_js = self._create_jobset()
    default_helper = JobSetManifestHelper(default_js.to_dict())
    self.assertNotIn(
        "priorityClassName", default_helper.pod_specs["pathways-head"]
    )
    self.assertEqual(
        default_helper.pod_specs["pathways-head"]["nodeSelector"][
            jobset.GKE_NODEPOOL_KEY
        ],
        "cpu-np",
    )
    self.assertNotIn(
        "priorityClassName", default_helper.pod_specs["pathways-worker"]
    )

    # Configured: priority class and head node selector set via pod_spec_kwargs
    configured_js = self._create_jobset(
        head_pod_spec_kwargs={
            "node_selector": {"zone": "us-central1-a"},
            "priority_class_name": "high",
        },
        worker_pod_spec_kwargs={
            "priority_class_name": "high",
        },
    )
    configured_helper = JobSetManifestHelper(configured_js.to_dict())
    head_spec = configured_helper.pod_specs["pathways-head"]
    worker_spec = configured_helper.pod_specs["pathways-worker"]

    self.assertEqual(head_spec["priorityClassName"], "high")
    self.assertEqual(worker_spec["priorityClassName"], "high")
    self.assertEqual(head_spec["nodeSelector"], {"zone": "us-central1-a"})

    # Overriding GKE_NODEPOOL_KEY explicitly
    custom_np_js = self._create_jobset(
        head_pod_spec_kwargs={
            "node_selector": {jobset.GKE_NODEPOOL_KEY: "custom-cpu-pool"},
        },
    )
    custom_np_helper = JobSetManifestHelper(custom_np_js.to_dict())
    self.assertEqual(
        custom_np_helper.pod_specs["pathways-head"]["nodeSelector"][
            jobset.GKE_NODEPOOL_KEY
        ],
        "custom-cpu-pool",
    )

  def test_container_kwargs(self):
    pw_jobset = self._create_jobset(
        rm_container_kwargs={
            "args": ["--v=1"],
            "env": {"TPU_SKIP_MDS_QUERY": "false", "TEST_TMPDIR": "/tmp"},
            "resources": client.V1ResourceRequirements(
                limits={"cpu": "4", "memory": "16G"}
            ),
        },
        proxy_container_kwargs={
            "args": [
                "--xla_jf_enable_multi_output_fusion=false",
                "--virtual_instances=tpuv5e:2x4",
            ],
            "env": {"XLA_FLAGS": "--xla_disable_hlo_passes=tpu_fusion"},
        },
        worker_container_kwargs={
            "args": ["--logtostderr"],
            "env": {"FLAGS_use_pthread_fiber_wrapper": "true"},
            "volume_mounts": [
                client.V1VolumeMount(name="data", mount_path="/data")
            ],
        },
        worker_pod_spec_kwargs={
            "volumes": [
                client.V1Volume(
                    name="data", empty_dir=client.V1EmptyDirVolumeSource()
                )
            ]
        },
    )
    helper = JobSetManifestHelper(pw_jobset.to_dict())
    rm = helper.containers["pathways-head"]["pathways-rm"]
    proxy = helper.containers["pathways-head"]["pathways-proxy"]
    worker = helper.containers["pathways-worker"]["pathways-worker"]

    # Args are appended to the default args.
    self.assertEqual(
        rm["args"],
        [
            "--server_port=29001",
            "--gcs_scratch_location=gs://test-bucket",
            "--node_type=resource_manager",
            "--instance_count=2",
            "--instance_type=tpuv5e:4x8",
            "--v=1",
        ],
    )
    self.assertEqual(
        proxy["args"][-2:],
        [
            "--xla_jf_enable_multi_output_fusion=false",
            "--virtual_instances=tpuv5e:2x4",
        ],
    )
    self.assertIn("--server_port=29000", proxy["args"])
    self.assertEqual(worker["args"][-1], "--logtostderr")
    self.assertIn("--server_port=29005", worker["args"])

    # Env vars replace defaults with the same name, in place, or are appended.
    rm_env_names = [e["name"] for e in rm["env"]]
    self.assertEqual(
        rm_env_names,
        [
            "REPLICATED_JOB_NAME",
            "JOBSET_NAME",
            "HOST_ADDRESS",
            "TPU_SKIP_MDS_QUERY",
            "TEST_TMPDIR",
        ],
    )
    rm_env = {e["name"]: e.get("value") for e in rm["env"]}
    self.assertEqual(rm_env["TPU_SKIP_MDS_QUERY"], "false")
    self.assertEqual(rm_env["TEST_TMPDIR"], "/tmp")
    self.assertEqual(
        proxy["env"][-1],
        {"name": "XLA_FLAGS", "value": "--xla_disable_hlo_passes=tpu_fusion"},
    )
    self.assertEqual(
        worker["env"][-1],
        {"name": "FLAGS_use_pthread_fiber_wrapper", "value": "true"},
    )

    # Other fields replace the defaults, and volume mounts are appended.
    self.assertEqual(rm["resources"], {"limits": {"cpu": "4", "memory": "16G"}})
    self.assertEqual(
        proxy["resources"], {"limits": {"cpu": "16", "memory": "100G"}}
    )
    self.assertEqual(
        worker["volumeMounts"],
        [
            {"mountPath": "/tmp", "name": "shared-tmp"},
            {"mountPath": "/data", "name": "data"},
        ],
    )
    self.assertEqual(
        [v["name"] for v in helper.pod_specs["pathways-worker"]["volumes"]],
        ["shared-tmp", "data"],
    )

  def test_container_kwargs_env_replaces_field_ref(self):
    pw_jobset = self._create_jobset(
        worker_container_kwargs={"env": {"MEGASCALE_NUM_SLICES": "4"}}
    )
    helper = JobSetManifestHelper(pw_jobset.to_dict())
    worker = helper.containers["pathways-worker"]["pathways-worker"]
    num_slices_env = [
        e for e in worker["env"] if e["name"] == "MEGASCALE_NUM_SLICES"
    ]
    self.assertEqual(
        num_slices_env, [{"name": "MEGASCALE_NUM_SLICES", "value": "4"}]
    )

  def test_container_kwargs_env_value_source(self):
    secret_ref = client.V1EnvVarSource(
        secret_key_ref=client.V1SecretKeySelector(name="hf", key="token")
    )
    pod_name_ref = client.V1EnvVarSource(
        field_ref=client.V1ObjectFieldSelector(field_path="metadata.name")
    )
    pw_jobset = self._create_jobset(
        worker_container_kwargs={"env": {"HF_TOKEN": secret_ref}},
        rm_container_kwargs={"env": {"TPU_SKIP_MDS_QUERY": pod_name_ref}},
    )
    helper = JobSetManifestHelper(pw_jobset.to_dict())
    worker = helper.containers["pathways-worker"]["pathways-worker"]
    rm = helper.containers["pathways-head"]["pathways-rm"]
    # A new variable with a value source is appended.
    self.assertEqual(
        worker["env"][-1],
        {
            "name": "HF_TOKEN",
            "valueFrom": {"secretKeyRef": {"key": "token", "name": "hf"}},
        },
    )
    # A default literal value is replaced in place by a value source.
    self.assertEqual(
        rm["env"][3],
        {
            "name": "TPU_SKIP_MDS_QUERY",
            "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}},
        },
    )

  @parameterized.named_parameters(
      ("int_value", {"CUSTOM_ENV": 123}, "CUSTOM_ENV .* got int"),
      ("none_value", {"CUSTOM_ENV": None}, "CUSTOM_ENV .* got NoneType"),
      (
          "env_var_list",
          [client.V1EnvVar(name="CUSTOM_ENV", value="1")],
          "must be a mapping",
      ),
  )
  def test_container_kwargs_invalid_env(self, env, error):
    with self.assertRaisesRegex(TypeError, error):
      self._create_jobset(worker_container_kwargs={"env": env})

  def test_container_kwargs_warns_when_replacing_default_field(self):
    with self.assertLogs(jobset._logger, level="WARNING") as logs:
      pw_jobset = self._create_jobset(
          worker_container_kwargs={
              "ports": [client.V1ContainerPort(container_port=1234)]
          }
      )
    self.assertIn(
        "Replacing the default ports of the pathways-worker container",
        logs.output[0],
    )
    helper = JobSetManifestHelper(pw_jobset.to_dict())
    worker = helper.containers["pathways-worker"]["pathways-worker"]
    self.assertEqual(worker["ports"], [{"containerPort": 1234}])

  def test_container_kwargs_no_warning_for_unset_field(self):
    with self.assertNoLogs(jobset._logger, level="WARNING"):
      self._create_jobset(
          worker_container_kwargs={
              "security_context": client.V1SecurityContext(privileged=True)
          }
      )

  def test_worker_pod_spec_volumes_keep_shared_tmp(self):
    data_volume = client.V1Volume(
        name="data",
        empty_dir=client.V1EmptyDirVolumeSource(),
    )
    pw_jobset = self._create_jobset(
        worker_pod_spec_kwargs={"volumes": [data_volume]},
        worker_container_kwargs={
            "volume_mounts": [
                client.V1VolumeMount(name="data", mount_path="/data")
            ]
        },
    )
    helper = JobSetManifestHelper(pw_jobset.to_dict())
    worker_spec = helper.pod_specs["pathways-worker"]
    worker = helper.containers["pathways-worker"]["pathways-worker"]
    # The caller's volume is added after the default shared-tmp volume, so
    # every volume mount of the worker container has a backing volume.
    self.assertEqual(
        [v["name"] for v in worker_spec["volumes"]], ["shared-tmp", "data"]
    )
    self.assertEqual(
        [m["name"] for m in worker["volumeMounts"]], ["shared-tmp", "data"]
    )

  def test_pod_spec_volumes_replace_default_with_same_name(self):
    shared_tmp = client.V1Volume(
        name="shared-tmp",
        host_path=client.V1HostPathVolumeSource(
            path="/tmp", type="DirectoryOrCreate"
        ),
    )
    # A volume identical to the default replaces it without a warning.
    with self.assertNoLogs(jobset._logger, level="WARNING"):
      pw_jobset = self._create_jobset(
          worker_pod_spec_kwargs={"volumes": [shared_tmp]}
      )
    helper = JobSetManifestHelper(pw_jobset.to_dict())
    self.assertLen(helper.pod_specs["pathways-worker"]["volumes"], 1)

    # A different volume with the same name replaces it with a warning.
    custom_tmp = client.V1Volume(
        name="shared-tmp", empty_dir=client.V1EmptyDirVolumeSource()
    )
    with self.assertLogs(jobset._logger, level="WARNING") as logs:
      pw_jobset = self._create_jobset(
          worker_pod_spec_kwargs={"volumes": [custom_tmp]}
      )
    self.assertIn(
        "Replacing the default shared-tmp volume of the pathways-worker pod",
        logs.output[0],
    )
    helper = JobSetManifestHelper(pw_jobset.to_dict())
    self.assertEqual(
        helper.pod_specs["pathways-worker"]["volumes"],
        [{"emptyDir": {}, "name": "shared-tmp"}],
    )

  def test_head_pod_spec_volumes(self):
    data_volume = client.V1Volume(
        name="data", empty_dir=client.V1EmptyDirVolumeSource()
    )
    pw_jobset = self._create_jobset(
        head_pod_spec_kwargs={"volumes": [data_volume]}
    )
    helper = JobSetManifestHelper(pw_jobset.to_dict())
    self.assertEqual(
        helper.pod_specs["pathways-head"]["volumes"],
        [{"emptyDir": {}, "name": "data"}],
    )

  def test_container_kwargs_kept_by_add_user_workload(self):
    pw_jobset = self._create_jobset(
        rm_container_kwargs={"args": ["--v=1"]},
        proxy_container_kwargs={"env": {"XLA_FLAGS": "--xla_dump_to=/tmp"}},
    )
    pw_jobset.add_user_workload(
        image="us-docker.pkg.dev/my-project/test:v1",
        command="python3 -m test_module",
    )
    helper = JobSetManifestHelper(pw_jobset.to_dict())
    rm = helper.init_containers["pathways-head"]["pathways-rm"]
    proxy = helper.init_containers["pathways-head"]["pathways-proxy"]
    self.assertEqual(rm["args"][-1], "--v=1")
    self.assertEqual(
        proxy["env"][-1], {"name": "XLA_FLAGS", "value": "--xla_dump_to=/tmp"}
    )

  def test_container_kwargs_export_import_roundtrip(self):
    pw_jobset = self._create_jobset(
        rm_container_kwargs={"args": ["--v=1"]},
        proxy_container_kwargs={"env": {"XLA_FLAGS": "--xla_dump_to=/tmp"}},
        worker_container_kwargs={"args": ["--logtostderr"]},
    )
    temp_filepath = os.path.join(
        self.create_tempdir().full_path, "jobset_container_kwargs.yaml"
    )
    pw_jobset.export_yaml(temp_filepath)
    imported = jobset.PathwaysJobSet.import_yaml(temp_filepath)
    self.assertEqual(
        normalize_k8s_spec(pw_jobset.to_dict()),
        normalize_k8s_spec(imported.to_dict()),
    )

  @parameterized.named_parameters(
      ("rm_name", "rm_container_kwargs", {"name": "rm"}, "Cannot set 'name'"),
      (
          "proxy_image",
          "proxy_container_kwargs",
          {"image": "proxy:v2"},
          "Cannot set 'image'",
      ),
      (
          "worker_unknown_field",
          "worker_container_kwargs",
          {"not_a_field": True},
          "Unknown client.V1Container fields",
      ),
  )
  def test_container_kwargs_invalid_fields(self, arg_name, kwargs, error):
    with self.assertRaisesRegex(ValueError, error):
      self._create_jobset(**{arg_name: kwargs})

  def test_proxy_container_kwargs_with_shared_pathways_service(self):
    with self.assertRaisesRegex(ValueError, "shared_pathways_service"):
      self._create_jobset(
          shared_pathways_service=True,
          proxy_container_kwargs={"args": ["--v=1"]},
      )

  def test_rm_container_kwargs_with_shared_pathways_service(self):
    pw_jobset = self._create_jobset(
        shared_pathways_service=True, rm_container_kwargs={"args": ["--v=1"]}
    )
    helper = JobSetManifestHelper(pw_jobset.to_dict())
    rm = helper.containers["pathways-head"]["pathways-rm"]
    self.assertEqual(rm["args"][-1], "--v=1")

  def test_add_user_workload_env(self):
    pw_jobset = self._create_jobset(topology="2x2", num_slices=1)
    pw_jobset.add_user_workload(
        image="us-docker.pkg.dev/my-project/test:v1",
        command="python3 -m test_module",
        env={"JAX_PLATFORMS": "cpu", "CUSTOM_ENV": "123"},
    )
    helper = JobSetManifestHelper(pw_jobset.to_dict())
    user_c = helper.containers["pathways-head"]["user-workload"]
    # Defaults with the same name are replaced in place; others are appended.
    self.assertEqual(
        user_c["env"],
        [
            {"name": "JAX_PLATFORMS", "value": "cpu"},
            {"name": "JAX_BACKEND_TARGET", "value": "grpc://localhost:29000"},
            {
                "name": "IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS",
                "value": "true",
            },
            {"name": "CUSTOM_ENV", "value": "123"},
        ],
    )

  def test_add_user_workload_unknown_field(self):
    pw_jobset = self._create_jobset(topology="2x2", num_slices=1)
    with self.assertRaisesRegex(
        ValueError, "Unknown client.V1Container fields for user-workload"
    ):
      pw_jobset.add_user_workload(
          image="us-docker.pkg.dev/my-project/test:v1",
          command="python3 -m test_module",
          not_a_field=True,
      )

  def test_kokoro_pretraining_workload_generation(self):
    """Verifies that PathwaysJobSet can generate a JobSet matching Kokoro pretraining test workloads."""
    pw_jobset = jobset.PathwaysJobSet(
        name="maxtext-pretraining-test",
        namespace="default",
        pathways_dir="gs://my-bucket/scratch",
        tpu_type="v5e",
        topology="4x8",
        num_slices=1,
        labels={"kueue.x-k8s.io/queue-name": "multislice-queue"},
        head_pod_spec_kwargs={
            "node_selector": {jobset.GKE_NODEPOOL_KEY: "cpu-np"},
            "priority_class_name": "high",
        },
        worker_pod_spec_kwargs={
            "priority_class_name": "high",
        },
    )
    pw_jobset.add_user_workload(
        image="us-docker.pkg.dev/my-project/maxtext:latest",
        command="python3 MaxText/train.py MaxText/configs/base.yml",
    )

    config = pw_jobset.to_dict()
    helper = JobSetManifestHelper(config)

    # Verify queue label
    self.assertEqual(
        config["metadata"]["labels"]["kueue.x-k8s.io/queue-name"],
        "multislice-queue",
    )

    # Verify success policy
    self.assertEqual(
        config["spec"]["successPolicy"]["targetReplicatedJobs"],
        ["pathways-head"],
    )

    # Verify head pod spec: priorityClassName, nodeSelector
    head_pod_spec = helper.pod_specs["pathways-head"]
    self.assertEqual(head_pod_spec["priorityClassName"], "high")
    self.assertEqual(
        head_pod_spec["nodeSelector"][jobset.GKE_NODEPOOL_KEY],
        "cpu-np",
    )

    # Verify head containers: RM and Proxy are in initContainers, user workload in containers
    self.assertIn("pathways-rm", helper.init_containers["pathways-head"])
    self.assertIn("pathways-proxy", helper.init_containers["pathways-head"])
    self.assertIn("user-workload", helper.containers["pathways-head"])
    self.assertNotIn("pathways-rm", helper.containers["pathways-head"])
    self.assertNotIn("pathways-proxy", helper.containers["pathways-head"])

    user_container = helper.containers["pathways-head"]["user-workload"]
    self.assertEqual(
        user_container["command"],
        ["sh", "-c", "python3 MaxText/train.py MaxText/configs/base.yml"],
    )

    # Verify worker pod spec: priorityClassName
    worker_pod_spec = helper.pod_specs["pathways-worker"]
    self.assertEqual(worker_pod_spec["priorityClassName"], "high")

  def test_with_custom_libtpu_gcsfuse_path(self):
    pw_jobset = self._create_jobset(topology="2x2", num_slices=1)
    returned = pw_jobset.add_gcsfuse(
        containers="all",
        mount_path="/mnt/gcs",
        bucket="my-libtpu-bucket",
        read_only=True,
    ).with_custom_libtpu("/mnt/gcs/nightlies/libtpu.so")
    self.assertIs(returned, pw_jobset)

    helper = JobSetManifestHelper(pw_jobset.to_dict())

    for job_name in ("pathways-head", "pathways-worker"):
      self.assertNotIn("custom-libtpu", helper.volumes[job_name])
      self.assertEqual(
          helper.pod_metadatas[job_name]
          .get("annotations", {})
          .get("gke-gcsfuse/volumes"),
          "true",
      )

    for container_name in ("pathways-rm", "pathways-proxy", "pathways-worker"):
      matches = helper.get_all_containers_by_name(container_name)
      self.assertLen(matches, 1, container_name)
      _, container = matches[0]
      self.assertIn(
          {"name": "TPU_LIBRARY_PATH", "value": "/mnt/gcs/nightlies/libtpu.so"},
          container["env"],
      )
      self.assertTrue(
          any(
              mount["mountPath"] == "/mnt/gcs" and bool(mount.get("readOnly"))
              for mount in container.get("volumeMounts", [])
          )
      )

  def test_with_custom_libtpu_mount_host_dir(self):
    pw_jobset = self._create_jobset(topology="2x2", num_slices=1)
    pw_jobset.with_custom_libtpu(
        "/var/lib/custom_tpu/libtpu.so",
        containers="worker",
        mount_host_dir=True,
    )
    helper = JobSetManifestHelper(pw_jobset.to_dict())

    self.assertNotIn("custom-libtpu", helper.volumes["pathways-head"])
    self.assertEqual(
        helper.volumes["pathways-worker"]["custom-libtpu"],
        {
            "name": "custom-libtpu",
            "hostPath": {
                "path": "/var/lib/custom_tpu",
                "type": "DirectoryOrCreate",
            },
        },
    )

    worker = helper.containers["pathways-worker"]["pathways-worker"]
    self.assertIn(
        {
            "name": "custom-libtpu",
            "mountPath": "/var/lib/custom_tpu",
            "readOnly": True,
        },
        worker["volumeMounts"],
    )
    self.assertIn(
        {"name": "TPU_LIBRARY_PATH", "value": "/var/lib/custom_tpu/libtpu.so"},
        worker["env"],
    )
    for head_container_name in ("pathways-rm", "pathways-proxy"):
      head_container = helper.containers["pathways-head"][head_container_name]
      self.assertNotIn(
          "TPU_LIBRARY_PATH",
          [env_var["name"] for env_var in head_container.get("env", [])],
      )

  @parameterized.named_parameters(
      dict(testcase_name="empty", path=""),
      dict(testcase_name="whitespace", path="   "),
      dict(testcase_name="gcs_uri", path="gs://bucket/libtpu.so"),
      dict(testcase_name="https_uri", path="https://example.com/libtpu.so"),
      dict(testcase_name="relative", path="relative/libtpu.so"),
      dict(testcase_name="root", path="/"),
      dict(testcase_name="directory", path="/mnt/gcs/dir/"),
      dict(
          testcase_name="root_file_with_host_dir",
          path="/libtpu.so",
          mount_host_dir=True,
          expected_error="mount_host_dir",
      ),
  )
  def test_with_custom_libtpu_rejects_invalid_paths(
      self,
      path: str,
      mount_host_dir: bool = False,
      expected_error: str = "libtpu_path",
  ):
    pw_jobset = self._create_jobset(topology="2x2", num_slices=1)
    with self.assertRaisesRegex(ValueError, expected_error):
      pw_jobset.with_custom_libtpu(path, mount_host_dir=mount_host_dir)


if __name__ == "__main__":
  absltest.main()
