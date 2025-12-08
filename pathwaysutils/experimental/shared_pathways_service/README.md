# Shared Pathways Service

The Shared Pathways Service accelerates developer iteration by providing a
persistent, multi-tenant TPU environment. This decouples service creation from
the development loop, allowing JAX clients to connect on-demand from a familiar
local environment (like a laptop or cloud VM) to a long-running Pathways
service that manages scheduling and error handling.

## Requirements

1. You have a GKE cluster with atleast 1 slice of `v6e-4` or `v6e-8`. Note that the Shared Pathways Service supports
single-host Trillium slices only, this support will be extended soon.

2. Start the Shared Pathways Service by using [pw-service-example.yaml](yamls/pw-service-example.yaml).
Make sure to modify the following values to deploy the Pathways pods:

- A unique Jobset name for the cluster's Pathways pods
- GCS bucket path
- TPU type and topology
- Number of slices

3. Verify that the Shared Pathways Service components are started, specifically the Resource Manager (RM) and Worker
pods.

Check that the required pods are running.
```
# Set the environment variables.
$ PROJECT=<your-project>
$ CLUSTER_NAME=<your-cluster>
$ REGION=<cluster-region>  # e.g., us-central2

# Get credentials for your cluster.
$ gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION --project=$PROJECT && kubectl config view && kubectl config set-context --current --namespace=default

# Check the status of RM and Worker pods.
$ kubectl get pods

# Sample expected output
NAME                                       READY   STATUS    RESTARTS   AGE
pathways-cluster-pathways-head-0-0-zzmn2   2/2     Running   0          3m49s
pathways-cluster-worker-0-0-bdzq4          1/1     Running   0          3m36s
pathways-cluster-worker-1-0-km2rf          1/1     Running   0          3m36s
```

You can also verify the pod status by looking at the project logs. Look for the below substring for the respective pod
type.

(Detailed instructions are <a href="https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/troubleshooting-pathways#health_monitoring" target="_blank">here</a>)

```
# Set the environment variables
$ HEAD_POD_NAME=pathways-cluster-pathways-head-0-0-zzmn2
$ WORKER0_POD_NAME=pathways-cluster-worker-0-0-bdzq4
$ WORKER1_POD_NAME=pathways-cluster-worker-1-0-km2rf
```

- RM
```
$ kubectl logs $HEAD_POD_NAME --container pathways-rm
...
I1208 20:10:04.992524       ...] Pathways Server serving on [::]:29001
...
I1208 20:10:23.848070       ...] *** 2/2 Pathways Slices Now Ready
```

- Worker
```
$ kubectl logs $WORKER0_POD_NAME --container pathways-worker
...
I1208 20:10:23.838022       ...] Pathways Server serving on [::]:29005
...
I1208 20:10:25.249167       ...] MegaScale transport initialized.
I1208 20:10:25.249172       ...] MegaScale transport init succeeded.

$ kubectl logs $WORKER1_POD_NAME --container pathways-worker
...
I1208 20:10:23.579361       ...] Pathways Server serving on [::]:29005
I1208 20:10:24.994411       ...] MegaScale transport initialized.
I1208 20:10:24.994416       ...] MegaScale transport init succeeded.
...
```

<a name="find-pw-service"></a>
4. Find the address of the Pathways service.
```
$ kubectl logs $WORKER0_POD_NAME --container pathways-worker | grep "\-\-resource_manager_address"
I1208 20:10:18.148825       ...] argv[2]: '--resource_manager_address=pathways-cluster-pathways-head-0-0.pathways-cluster:29001'
```

## Instructions

1. Clone `pathwaysutils`.

```
git clone https://github.com/AI-Hypercomputer/pathways-utils.git
```

2. Install `portpicker`.

```
pip install portpicker
```

3. In your script,

    - Import `isc_pathways`
    - Add `with isc_pathways.connect(...)` statement. The function takes the below values:
        - Cluster name
        - Project name
        - Region
        - GCS bucket name
        - Pathways Service (See instructions to find the Pathways address [here](#find-pw-service))
    - Write your ML code under this `with` block to run it on the underlying TPUs.

See [run_connect_example.py](run_connect_example.py) for reference. Example code:

```
 from pathwaysutils.experimental.shared_pathways_service import isc_pathways

 with isc_pathways.connect(
     cluster="my-cluster",
     project="my-project",
     region="region",
     gcs_bucket="gs://user-bucket",
     pathways_service="pathways-cluster-pathways-head-0-0.pathways-cluster:29001",
     expected_tpu_instances={"tpuv6e:2x2": 2},
 ) as tm:
   import jax.numpy as jnp
   import pathwaysutils
   import pprint

   pathwaysutils.initialize()
   orig_matrix = jnp.zeros(5)
   ...
```

The connect block will deploy a proxy pod dedicated to your client and connect
your local runtime environment to the proxy pod via port-forwarding.

4. You can start another client that uses the same `pathways_service` (similar to Step#3). If the Shared Pathways
Service finds free TPU(s) that match your request, your workload will start running on the free resources. However,
if all TPUs are occupied, you can expect your script to fail.

## Troubleshooting
Refer to [this guide](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/troubleshooting-pathways)
if your Pathways pods do not come up!
