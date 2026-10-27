# PathwaysJobSet Builder

`PathwaysJobSet` generates and deploys Kubernetes `JobSet` resources (`jobset.x-k8s.io`) for running distributed Pathways workloads on Google Kubernetes Engine (GKE) without requiring custom controllers.

## Installation

```bash
pip install 'pathwaysutils[gke]'
```

## Quickstart

```python
from pathwaysutils.experimental.gke import jobset

pw_jobset = jobset.PathwaysJobSet(
    name="my-workload",
    namespace="default",
    pathways_dir="gs://my-bucket/pathways-scratch",
    tpu_type="v5e",
    topology="2x4",
    num_slices=1,
)

# Export to YAML or deploy directly to the active cluster
pw_jobset.export_yaml("jobset.yaml")
pw_jobset.apply(recreate=True)
```

## Features

### Elasticity & Multislice
Scale across multiple TPU slices with dynamic resilience:

```python
pw_jobset = jobset.PathwaysJobSet(
    name="elastic-v6e",
    namespace="default",
    pathways_dir="gs://my-bucket/scratch",
    tpu_type="v6e",
    topology="4x4",
    num_slices=4,
    elastic_slices=2,  # Configures Pathways Proxy for dynamic slice recovery
)
```

### In-Pod User Workload
Run training or inference directly in the head pod alongside the Resource Manager and Proxy sidecars:

```python
pw_jobset.add_user_workload(
    image="us-docker.pkg.dev/my-project/images/my-trainer:latest",
    command="python3 train.py --batch_size=32",
    env={"LOG_LEVEL": "INFO"},
)
```

### Colocated Python Sidecar
Inject a sidecar container onto worker pods sharing memory with `pathways-worker` for fast tensor transfers:

```python
pw_jobset.add_colocated_python(
    image="us-docker.pkg.dev/my-project/images/worker-sidecar:latest",
    shm_mount_path="/tmp/shared-memory",
    shm_size_limit="16Gi",
)
```

### GCSFuse Cloud Storage Mounts
Mount GCS buckets to head, worker, or all containers:

```python
pw_jobset.add_gcsfuse(
    containers="all",
    mount_path="/data",
    bucket="my-data-bucket",
    read_only=True,
)
```

### Container & Pod Spec Overrides
Customize arguments, environment variables, resources, or volumes on any container or pod:

```python
pw_jobset = jobset.PathwaysJobSet(
    ...,
    rm_container_kwargs={"args": ["--v=1"]},
    proxy_container_kwargs={"env": {"XLA_FLAGS": "--xla_dump_to=/tmp"}},
    worker_container_kwargs={"args": ["--stderrthreshold=0"]},
    worker_pod_spec_kwargs={"priority_class_name": "high-priority"},
)
```

## Serialization & Round-Tripping

Convert between Python objects, YAML files, and dictionaries:

```python
# Save and load JobSet configurations
pw_jobset.export_yaml("jobset.yaml")
loaded_jobset = jobset.PathwaysJobSet.import_yaml("jobset.yaml")

# Inspect raw Kubernetes resource dictionary
config_dict = pw_jobset.to_dict()
```
