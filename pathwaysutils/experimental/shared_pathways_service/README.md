# Shared Pathways Service

The Shared Pathways Service accelerates developer iteration by providing a
persistent, multi-tenant TPU environment. This decouples service creation from
the development loop, allowing JAX clients to connect on-demand from a familiar
local environment (like a laptop or cloud VM) to a long-running Pathways
service that manages scheduling and error handling.

## Requirements

### 1. Create a GKE cluster with TPUs

You have a GKE cluster with at least 1 TPU slice (v5e, v5p or v6e).

<a name="pw-service-yaml"></a>

### 2. Deploy the Pathways head pod

Start the Shared Pathways Service by running `deploy_pathways_service.py`.

```shell
python3 -m pathwaysutils.experimental.shared_pathways_service.deploy_pathways_service \
--jobset_name ${SERVICE_JOBSET_NAME} \
--gcs_bucket ${GCS_BUCKET} \
--server_image ${SERVER_IMAGE} \
--tpu_type ${TPU_TYPE} \
--topology ${TOPOLOGY} \
--num_slices ${NUM_SLICES} [--sidecar_image ${SIDECAR_IMAGE}] [--dry_run]
```

Note, `server_image` is the Pathways server image, e.g, `us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:<tag>`.

### 3. Verify that the pods created in [Step#2](#2-deploy-the-pathways-head-pod) are running

Verify that the Shared Pathways Service components are started, specifically the Pathways resource manager (RM) and
Pathways workers.

```shell
# Set the environment variables.
$ PROJECT=<your-project>
$ CLUSTER_NAME=<your-cluster>
$ REGION=<cluster-region>  # e.g., us-central2

# Get credentials for your cluster.
$ gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION --project=$PROJECT && kubectl config view && kubectl config set-context --current --namespace=default
```

#### Option 1: List all pods

```shell
$ kubectl get pods

# Sample expected output (1 Head pod and 1 or more Worker pods)
NAME                                       READY   STATUS    RESTARTS   AGE
pathways-cluster-pathways-head-0-0-zzmn2   2/2     Running   0          3m49s   # HEAD POD
pathways-cluster-worker-0-0-bdzq4          1/1     Running   0          3m36s   # WORKER 0
pathways-cluster-worker-1-0-km2rf          1/1     Running   0          3m36s   # WORKER 1
```

#### Option 2: Check the status of the specific pods that belong to your Pathways Service

```shell
# e.g., pathways-cluster
$ SERVICE_JOBSET_NAME=<your-jobset-name>  # same as you used in [pw-service-example.yaml](#pw-service-yaml)

# e.g., pathways-cluster-pathways-head-0-0-zzmn2
$ HEAD_POD_NAME=$(kubectl get pods --selector=jobset.sigs.k8s.io/jobset-name=${SERVICE_JOBSET_NAME} -o jsonpath='{.items[?(@.status.phase=="Running")].metadata.name}' | sed 's/ /\n/g' | grep head)

# e.g., pathways-cluster-worker-0-0-bdzq4
$ WORKER0_POD_NAME=$(kubectl get pods --selector=jobset.sigs.k8s.io/jobset-name=${SERVICE_JOBSET_NAME} -o jsonpath='{.items[?(@.status.phase=="Running")].metadata.name}' | sed 's/ /\n/g' | grep 'worker-0-0-')
```

#### Option 3: Check project logs

Find the detailed instructions
<a href="https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/troubleshooting-pathways#health_monitoring" target="_blank">here</a>).

<a name="find-pw-service"></a>
### 4. Find the Pathways service address
Find the address of the Pathways service from the logs. We check the worker pod logs in the below command.
```shell
$ kubectl logs $WORKER0_POD_NAME --container pathways-worker | grep "\-\-resource_manager_address"

I1208 20:10:18.148825       ...] argv[2]: '--resource_manager_address=pathways-cluster-pathways-head-0-0.pathways-cluster:29001'
```

## Instructions

### 1. Install `pathwaysutils` and SPS dependencies.

```shell
pip install --upgrade
git+https://github.com/AI-Hypercomputer/pathways-utils[sps]
Note: google-cloud-monitoring is optional. If you don't wish to collect SPS usage metrics, install with pathways-utils[sps_without_monitoring].
```

### 2. Either use `run_workload.py` or move your workload under the `isc_pathways` Context Manager

Make sure to use the proxy image compatible with the server image used while deploying SPS.

```shell
python3 -m pathwaysutils.experimental.shared_pathways_service.run_workload \
--cluster=$CLUSTER_NAME \
--project=$PROJECT \
--region=$REGION \
--gcs_bucket=$GCS_BUCKET \
--pathways_service="$SERVICE_JOBSET_NAME-pathways-head-0-0.$SERVICE_JOBSET_NAME:29001" \
--tpu_type="$TPU_TYPE:$TOPOLOGY" \
--tpu_count=$NUM_SLICES \
--proxy_server_image=$PROXY_IMAGE \
--collect_service_metrics \
--command="python3 run_my_sample_workload.py" [--proxy_options=sidecar:true]
```

Alternatively, in your script,

1.  Import `isc_pathways`
2. Add `with isc_pathways.connect(...)` statement. The function takes the below values:
    - Cluster name
    - Project name
    - Region
    - GCS bucket name
    - Pathways Service (See instructions to find the RM address [here](#4-find-the-pathways-service-address))
    - Expected TPU instances
    - Proxy server image
    - Proxy options
    - Whether to collect SPS usage metrics
<a name="ml-code"></a>
3. Write your ML code under this context manager (the `with` block) to run your JAX code on the underlying TPUs.

See [run_connect_example.py](run_connect_example.py) for reference. Example code:

```shell

python3 pathwaysutils/experimental/shared_pathways_service/run_connect_example.py \
--cluster="my-cluster" \
--project="my-project" \
--region="cluster-region" \
--gcs_bucket="gs://user-bucket" \
--pathways_service="pathways-cluster-pathways-head-0-0.pathways-cluster:29001" \
--tpu_type="tpuv6e:2x2" \
--tpu_count=1 \
--proxy_server_image="proxy_image"
```

The connect block will deploy a proxy pod dedicated to your client and connect
your local runtime environment to the proxy pod via port-forwarding.

4. You can start another client that uses the same `pathways_service` (similar to [Step#3](#ml-code)). If the Shared Pathways
Service finds available TPU(s) that match your request, your workload will start running on these available resources.
However, if all TPUs are occupied, you can expect your script to halt until the TPUs are available again.

## Troubleshooting
- Refer to [this guide](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/troubleshooting-pathways)
if your Pathways pods do not come up!

- Known errors: The cleanup process of the service is not as clean right now. You can safely ignore the
`Segmentation fault` error, if you see any, after your ML job completes.
