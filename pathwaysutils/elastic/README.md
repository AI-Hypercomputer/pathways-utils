# Elastic Training with Pathways

This document demonstrates how to leverage the elasticity primitives within `pathwaysutils.elastic` to create a resilient JAX training loop that can handle hardware failures gracefully. We illustrate this using an example based on the MaxText training loop running on TPUs provisioned by GKE via `PathwaysJob` API.

## Overview

Distributed training jobs, especially long-running ones, are susceptible to various failures, such as machine preemptions and hardware issues. Elasticity allows a training job to adapt to changes in the number of available accelerators without crashing. It typically involves:

1.  **Training State Management**: Regularly snapshotting the training state (model params, optimizer state, data iterator state).
1.  **Failure Detection**: Pathways Resource Manager detects when workers join or leave.
1.  **Failure Propogation**: Pathways runtime propagates the error to JAX client.
1.  **Training Reconfiguration**: Adapting the training computation distribution to the current set of healthy workers.
1.  **Resumption**: Continuing training from the last valid snapshot with the new configuration.

The `pathwaysutils.elastic` primitives provide elastcity building blocks to use within your JAX training loop when using the Pathways' `Proxy` JAX backend.

## Prerequisites

* A [Pathways compatible GKE cluster](https://cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/create-gke-cluster) with TPU and CPU nodepools.
* `kubectl` configured to interact with your cluster.
* Access to a container image containing JAX, your model code (e.g., MaxText), and the `pathwaysutils` package with elasticity features integrated.

## Elastic MaxText Training with Pathways on GKE

This example demonstrates running an elastic MaxText job on 3 x v5e-32 slices using Pathways. See the [PathwaysJob docs](https://cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro#pathwaysjob_api) for more details about the various attributes set in the YAML below. 

### 1. Elastic PathwaysJob Definition (`pathwaysjob-elastic.py`)
Please set the variables marked with `<>` below before executing the script.
```yaml
apiVersion: pathways-job.pathways.domain/v1
kind: PathwaysJob
metadata:
  name: pathways-<USER>
spec:
  maxRestarts: 0
  workers:
  - type: ct5lp-hightpu-4t
    topology: 4x8
    numSlices: 3
    maxSliceRestarts: 2
  pathwaysDir: "gs://<BUCKET>" # Pre-create this bucket.
  controller:
    deploymentMode: default
    elasticSlices: 1
    template:
      spec:
        containers:
        - name: main
          image: <MAXTEXT_IMAGE>
          imagePullPolicy: Always
          command:
          - bash
          - -c
          - >
            python3 -m MaxText.elastic_train MaxText/configs/base.yml
            base_output_directory=gs://<BUCKET>
            per_device_batch_size=4
            enable_checkpointing=false
            remat_policy=full
            global_parameter_scale=8
            steps=50
            max_target_length=2048
            use_iota_embed=true
            reuse_example_batch=1
            dataset_type=synthetic
            attention=flash
            gcs_metrics=True
            enable_pathways_goodput=True
            run_name=pathways-<USER>
```
The MaxText elastic training [script](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/elastic_train.py) invoked by the `main` container above is integrated with `pathwaysutils.elastic` primitives.

### 2. Running the Elastic Training Loop and Simulating hardware failures

The following bash script demonstrates launching the above elastic maxtext job with Pathways, monitoring its progress, simulating a hardware failure by issuing a `kubectl drain` to a randomly selected TPU node, and observing the recovery. Please set the variables marked as `<>` below before executing the script. At the end of the script, we verify elasticity worked as expected.

```bash
#!/bin/bash
WORKING_DIR=</LOCAL/DIRECTORY/PATH>
USER_LABEL_SELECTOR="<USER>"
LOG_DIR="${WORKING_DIR}/logs"
RUN_ID=pathways-${USER_LABEL_SELECTOR}
LOG_FILE="${LOG_DIR}/logs_${RUN_ID}.log"
JOB_DEFINITION_FILE="${WORKING_DIR}/pathwaysjob-elastic.yaml" # Copy the above yaml into this file

mkdir -p ${LOG_DIR}

echo "Running Elastic MaxText with Run ID: ${RUN_ID}"

# 1. Launch the PathwaysJob
kubectl apply -f "$JOB_DEFINITION_FILE"

# 2. Monitor the PathwaysJob
echo "Waiting for pods to start..."
head_pod=""
for i in $(seq 1 10)
do
  head_pod=$(kubectl get pods -o=name --field-selector='status.phase==Running' | grep "$USER_LABEL_SELECTOR" | grep 'head' | head -n 1)
  if [ -n "$head_pod" ]; then
    echo "Found head pod: $head_pod"
    break
  fi
  echo "Head pod not found yet, retrying..."
  sleep 10s
done

if [ -z "$head_pod" ]; then
  echo "Error: Could not find running head pod after multiple attempts. Cleaning up..." 1>&2
  kubectl delete -f "$JOB_DEFINITION_FILE"
  exit 1
fi

echo "Streaming logs from $head_pod to ${LOG_FILE}"
kubectl logs -f "$head_pod" >> "${LOG_FILE}" &
logs_pid=$!
echo "Waiting for job to start making progress..."
sleep 90s

# 3. Simulate Failure: Evict a Worker Pod
echo "Randomly select a worker pod to disrupt..."
read -r node_name pod_name <<<$(kubectl get pods -o wide --field-selector='status.phase==Running' | grep "$USER_LABEL_SELECTOR" | grep worker | shuf | head -n 1 | awk '{print $7, $1}')

if [ -z "$pod_name" ] || [ -z "$node_name" ]; then
  echo "Warning: Could not find a running worker pod to disrupt. Skipping disruption."
else
  echo "Attempting to cordon '$node_name' and kill pod '$pod_name'..."
  kubectl cordon "$node_name"
  kubectl exec -it "$pod_name" -c pathways-worker -- /bin/sh -c "kill -s SIGILL 1"
  echo "Node cordoned. Waiting briefly for training to reconfigure to N-1 slices..."
  sleep 90s

  # 4. Allow Recovery: Uncordon the Node
  echo "Uncordoning node '$node_name' to allow scheduling again."
  kubectl uncordon "$node_name"
fi

# 5. Wait for Training to resume on all slices
sleep 90s

# 6. Terminate the Job and Cleanup
echo "Terminating Run ID ${RUN_ID}"
kubectl delete -f "$JOB_DEFINITION_FILE"
# Ensure log streaming process is killed
kill "$logs_pid" 2>/dev/null 
echo "Completed Run ID ${RUN_ID}."

# 6. Verify by printing steps where training reconfigured from N to N-1 slices and later back to N slices
# Expect output like:
# Step: 5, Old Slice Count: 3, New Slice Count: 2 (3 -> 2 slices)
# Step: 17, Old Slice Count: 2, New Slice Count: 3 (2 -> 3 slices)
awk '
  /step=/ && /elastic_manager\.elastic_down_event_count=/ {
    split($0, fields, " ")
    step = ""
    good_slice_count = ""
    for (i in fields) {
      split(fields[i], kv, "=")
      if (kv[1] == "step") {
        step = kv[2]
      } else if (kv[1] == "elastic_manager.good_slice_count") {
        good_slice_count = kv[2]
      }
    }
    if (prev_good_slice_count != "" && prev_good_slice_count != good_slice_count) {
      print "Step: " step ", Old Slice Count: " prev_good_slice_count ", New Slice Count: " good_slice_count
    }
    prev_step = step
    prev_good_slice_count = good_slice_count
  }
' "${LOG_FILE}"
```
