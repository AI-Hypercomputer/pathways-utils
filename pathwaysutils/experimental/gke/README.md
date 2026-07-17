# PathwaysJobSet Reference Implementation

`PathwaysJobSet` is a client-side Python library designed to generate and
deploy Kubernetes `JobSet` resources (`kind: JobSet`) for running Pathways
workloads on Google Kubernetes Engine (GKE).

It serves as a direct replacement for the custom `PathwaysJob` CRD (`kind:
PathwaysJob`), allowing you to deploy Pathways workloads using standard,
community-supported `JobSet` resources without needing a custom controller
installed on your GKE cluster.

---

## Purpose & Overview

This directory serves as the official **reference implementation** for
launching Pathways workloads via Kubernetes `JobSet`:

-   **Official Reference Standard:** Other Google teams and external customers
    can base their JobSet specifications on this reference implementation and
    expect their workloads to work reliably with Pathways.
-   **Internal Integration Testing:** Our team uses this reference
    implementation as the foundation for all of our automated integration
    tests.
-   **Minimal Required Specification:** This library defines the baseline,
    minimal required Kubernetes specification to run Pathways on Cloud
    workloads.

### Production Workloads & Cluster Toolkit
While this reference implementation focuses on the minimal required
specification for Pathways, full-featured production deployments often
require additional Google Cloud Platform (GCP) features and infrastructure
integrations.

-   **Bridging the Gap with Cluster Toolkit:** Solutions such as
    [Cluster Toolkit](https://cloud.google.com/cluster-toolkit/docs/overview)
    bridge the gap between this minimal required implementation and
    full-featured, enterprise-grade production workloads.
-   **Guidance for Custom Workloads:** Advanced users and customers who need
    custom JobSets or desire fine-grained control over their workload
    definitions can refer to this `pathwaysutils` JobSet generator to see our
    team's recommended best practices for Pathways JobSets, as well as reference
    implementations for integrating with other GCP services (such as Cloud
    Storage FUSE).

---

## Key Features

-   **Client-Side Generation:** Generates standard `JobSet` YAML manifests
    that can be inspected, version-controlled, or applied manually.
-   **Standard Headless Execution:** Runs the Pathways Resource Manager (RM)
    and Proxy as standalone containers in the head job.
-   **GCSFuse Integration:** Easily mount Cloud Storage buckets to head
    and/or worker pods using GCSFuse.
-   **Colocated Python Support:** Inject a colocated Python sidecar container
    and shared memory volume into worker pods for multi-agent or hybrid
    workloads.
-   **Elasticity Support:** Configure elastic slices for dynamic scaling.

---

## Basic Usage

```python
from pathwaysutils.experimental.gke import jobset

# 1. Initialize the builder
pw_jobset = jobset.PathwaysJobSet(
    name="my-pathways-workload",
    namespace="default",
    pathways_dir="gs://my-bucket/pathways-scratch",
    tpu_type="v5e",
    topology="4x8",
    num_slices=1,
)

# 2. Export to a standard JobSet YAML file
pw_jobset.export_yaml("jobset.yaml")

# 3. Deploy directly to a GKE cluster (requires kubernetes configured)
pw_jobset.apply(
    project_id="my-gcp-project",
    region="us-central1",
    cluster_id="my-gke-cluster",
)
```

---

## Examples

### 1. Single-Slice v5e-32 (Non-elastic)
A standard single-slice workload on TPU v5e-32 (32 chips, `4x8` topology, 8
VMs).

```python
from pathwaysutils.experimental.gke import jobset

js = jobset.PathwaysJobSet(
    name="v5e-32-headless",
    namespace="default",
    pathways_dir="gs://my-bucket/scratch",
    tpu_type="v5e",
    topology="4x8",
    num_slices=1,
)
js.export_yaml("v5e_32_headless.yaml")
```

### 2. Multislice v5p-4x4x4 (2 Slices)
A multislice workload using 2 slices of TPU v5p-4x4x4 (64 chips per slice).

```python
from pathwaysutils.experimental.gke import jobset

js = jobset.PathwaysJobSet(
    name="v5p-multislice",
    namespace="default",
    pathways_dir="gs://my-bucket/scratch",
    tpu_type="v5p",
    topology="4x4x4",
    num_slices=2,  # 2 slices
)
js.export_yaml("v5p_multislice.yaml")
```

### 3. Multislice v6e with Elasticity
A multislice workload on TPU v6e-16 (topology `4x4`, 16 chips, 4 VMs per
slice) with 2 active slices initially, and elasticity configured for up to 4
slices.

```python
from pathwaysutils.experimental.gke import jobset

js = jobset.PathwaysJobSet(
    name="v6e-elastic",
    namespace="default",
    pathways_dir="gs://my-bucket/scratch",
    tpu_type="v6e",
    topology="4x4",
    num_slices=2,       # Initial/Active slices
    elastic_slices=4,   # Enable elasticity (informs proxy of max slices)
)
js.export_yaml("v6e_elastic.yaml")
```

### 4. Advanced Features: GCSFuse and Colocated Python
This example demonstrates chaining advanced features:

-   Mounting a GCS bucket to all containers (head and workers) using GCSFuse.
-   Adding a colocated Python sidecar to the worker pods.

```python
from pathwaysutils.experimental.gke import jobset

js = (
    jobset.PathwaysJobSet(
        name="advanced-workload",
        namespace="default",
        pathways_dir="gs://my-bucket/scratch",
        tpu_type="v5e",
        topology="4x8",
        num_slices=1,
    )
    .add_colocated_python()  # Injects colocated python sidecar to workers
    .add_gcsfuse(
        containers="all",
        mount_path="/data",
        bucket="my-data-bucket",
        read_only=True,
    )
)
js.export_yaml("advanced_workload.yaml")
```

---

## API Reference: `PathwaysJobSet`

### Constructor `__init__`

```python
def __init__(
    self,
    name: str,
    namespace: str,
    pathways_dir: str,
    tpu_type: str,
    topology: str,
    num_slices: int,
    max_restarts: int = 0,
    max_slice_restarts: int = 0,
    termination_grace_period_seconds: int | None = None,
    pathways_version: str = "latest",
    jobset_api_version: str = "v1alpha2",
    elastic_slices: int = 0,
    labels: Mapping[str, str] | None = None,
    annotations: Mapping[str, str] | None = None,
    shared_pathways_service: bool = False,
)
```

-   `tpu_type`: Supported values include `"v5e"`, `"v5p"`, `"v6e"`, `"v4"`.
-   `elastic_slices`: If `> 0`, enables elasticity by passing
    `--num_elastic_slices` to the Pathways Proxy.
-   `shared_pathways_service`: If `True`, configures the head job to run only
    the Resource Manager (`pathways-rm`).

### Methods

-   **`add_colocated_python()`**: Injects a colocated Python container and
    shared memory volume (`/tmp/shared-memory`) into worker pods.
-   **`add_gcsfuse(containers, mount_path, bucket, read_only=False)`**: Mounts
    a GCS bucket using GCSFuse CSI driver. `containers` can be `"head"`,
    `"worker"`, or `"all"`.
-   **`to_dict()`**: Returns the compiled K8s JobSet resource as a Python
    dictionary.
-   **`export_yaml(filepath)`**: Serializes and writes the JobSet to a YAML
    file.
-   **`apply(project_id, region, cluster_id)`**: Deploys the JobSet to the
    specified GKE cluster. Supports delete-and-recreate lifecycle if the JobSet
    already exists.
