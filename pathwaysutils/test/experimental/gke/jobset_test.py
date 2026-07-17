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

    self.assertIn("colocated-python-sidecar", helper.containers["pathways-worker"])
    sidecar = helper.containers["pathways-worker"]["colocated-python-sidecar"]
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

    # Verify both exist
    self.assertIn("existing-init-container", helper.init_containers["pathways-worker"])
    self.assertIn("colocated-python-sidecar", helper.containers["pathways-worker"])

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
            e["name"] == "cloud_pathways_sidecar_shm_directory"
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

    self.assertIn("colocated-python-sidecar", helper.containers["pathways-worker"])
    sidecar = helper.containers["pathways-worker"]["colocated-python-sidecar"]
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
            e["name"] == "cloud_pathways_sidecar_shm_directory"
            and e["value"] == "/tmp/custom-shm"
            for e in worker_container["env"]
        )
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

    self.assertEqual(
        config["spec"]["successPolicy"],
        {
            "operator": "All",
            "targetReplicatedJobs": ["pathways-head"],
        },
    )

    self.assertTrue(config["spec"]["network"]["enableDNSHostnames"])
    self.assertTrue(config["spec"]["network"]["publishNotReadyAddresses"])

    self.assertIn("pathways-head", helper.jobs)
    pod_spec = helper.pod_specs["pathways-head"]

    # Head job should only have pathways-rm container, no pathways-proxy.
    self.assertIn("pathways-rm", helper.containers["pathways-head"])
    self.assertNotIn("pathways-proxy", helper.containers["pathways-head"])
    self.assertLen(pod_spec["containers"], 1)


if __name__ == "__main__":
  absltest.main()
