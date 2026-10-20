# Pathways Elasticity Demos

Evaluating distributed machine learning training on elastic TPU topologies
requires understanding how varying failure scenarios and reinitialization
latencies affect cumulative goodput and step progression. This folder provides
both an interactive browser-based simulation and a live Google Kubernetes Engine
(GKE) demo comparing Pathways elasticity recovery paradigms.

## Browser-Based Recovery Simulation

`model_comparison.html` compares three distinct recovery paradigms side by side
without requiring a live cluster.

* [Hosted GitHub Pages Simulation](https://ai-hypercomputer.github.io/pathways-utils/pathwaysutils/extras/elastic/model_comparison.html)
  provides a live interactive web application with simulation controls, or
  [HTMLPreview Instant Mirror](https://htmlpreview.github.io/?https://github.com/AI-Hypercomputer/pathways-utils/blob/main/pathwaysutils/extras/elastic/model_comparison.html).
* Local browser viewing allows running `model_comparison.html` directly on your workstation by opening the file in Chrome, Firefox, or Safari, requiring zero installation or server dependencies.

### Recovery Paradigms Compared

* Traditional Checkpoint Rollback (McJAX) halts cluster computation upon slice preemption, incurs a reinitialization latency, and rewinds training state to the most recent periodic checkpoint or snapshot, resulting in lost forward progress.
* Snapshot Replica Resize dynamically adjusts the mesh to continue training on surviving healthy slices with a smaller reinitialization latency than a cold start, maintaining forward progress at a degraded throughput factor (depending on your recovery implementation) with no time spent waiting for slices to rejoin.
* Snapshot Pause Resume pauses execution during slice preemption, rewinds to the last snapshot, and resumes with negligible reinitialization once replacement slices become healthy.

### Interactive Parameter Controls

The simulation toolbar allows tuning key workload and infrastructure characteristics in real time:

* Cold Start Initialization Time (seconds) configures the initial cluster startup delay prior to the first training step and is reused as the reinitialization latency for McJAX.
* Scale-Down Initialization Time (seconds) sets the reconfiguration overhead when transitioning into degraded execution.
* Scale-Up Initialization Time (seconds) governs the mesh reconfiguration latency when a replacement slice joins the workload.
* Degraded Scaling Factor (multiplier) sets the relative step throughput while executing on a reduced slice topology.
* Initial Base Loss (L₀) sets the initial loss baseline at step 0 in the exponential decay function L = L₀ * exp(-α * step).
* Loss Decay Rate (α) dictates the per-step exponential decay rate, multiplied by the degraded scaling factor during degraded execution so training loss decays slower per step.
* Checkpoint Interval (steps) dictates the rollback penalty for checkpoint-based training.
* Interruption Duration for Scenario (seconds) configures hardware disruption length during automated scenario runs.

## Live GKE Elasticity Demo

`gke_elasticity_demo.html`, `gke_demo_bridge.py`, and `elastic_workload.py`
drive and monitor a real Pathways training JobSet on a GKE TPU cluster.

* `elastic_workload.py` runs the `pathwaysutils.extras.simple_model` workload
  with `ElasticManager` and `Snapshotter`, exposing an HTTP control server on
  head pod port `8090` for start, pause, reset, and fault-injection signals.
* `gke_demo_bridge.py` binds to `127.0.0.1:48259` by default, streams live JAX
  driver telemetry over Server-Sent Events (SSE), writes a combined ConfigMap
  and JobSet manifest bundle to `/tmp/gke_elastic_workload.yaml`, and triggers
  disruptions via Kubernetes `kubectl cordon` and `drain` or in-workload HTTP
  signaling.
* `gke_elasticity_demo.html` renders live training step, loss, step time,
  cumulative goodput, logical workload slices, physical cluster TPU slices, and
  a real-time JAX driver log stream.

### Running the Live GKE Demo

A single command builds (or reuses) the workload image, generates the
ConfigMap + JobSet manifest, applies it to your cluster, starts the bridge
server, and opens the demo page. Everything below runs from your workstation.

#### Prerequisites

* A GKE cluster with the [JobSet](https://jobset.sigs.k8s.io/) controller
  installed and at least `num_slices + standby_slices` TPU node pools of the
  same type and topology (the defaults need four `tpuv5e` `4x8` slices). Kueue
  is optional; pass `--kueue-queue=NAME` only if your cluster uses it.
* A node pool that can schedule the head pod (`24` CPU, `100G` memory limits).
  Pass `--cpu-nodepool=NAME` to pin it, or leave unset to let GKE schedule it.
* A GCS bucket that the cluster's node service account can read and write,
  used as the Pathways scratch location (`--gcs-scratch=gs://BUCKET/PREFIX/`).
* An Artifact Registry repository you can push to, plus either Cloud Build
  permissions (`roles/cloudbuild.builds.editor`, the default) or a local
  Docker daemon (`--builder=docker`).
* Locally: Python 3.11+, `git`, `gcloud`, `kubectl`, and the
  `gke-gcloud-auth-plugin` component.

#### 1. Clone the branch and install

```bash
git clone -b elastic-demo https://github.com/AI-Hypercomputer/pathways-utils.git
cd pathways-utils
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[elastic-demo]"
```

#### 2. Point kubectl at your cluster

```bash
gcloud container clusters get-credentials CLUSTER --region REGION --project PROJECT
kubectl config current-context   # the demo uses this context unless --context is set
```

#### 3. Build the image, deploy, and open the UI

```bash
python3 -m pathwaysutils.extras.elastic.gke_demo_bridge \
  --build-image \
  --image=REGION-docker.pkg.dev/PROJECT/REPO/elastic-demo:latest \
  --gcs-scratch=gs://BUCKET/elastic-demo/ \
  --tpu-type=tpuv5e --topology=4x8 --num-slices=2 --standby-slices=2
```

This command, in order:

1. Builds `pathwaysutils/extras/elastic/Dockerfile` with Cloud Build and pushes
   it to `--image` (add `--builder=docker` to build locally instead, or drop
   `--build-image` to reuse an image you already pushed).
2. Writes the ConfigMap + JobSet bundle to `/tmp/gke_elastic_workload.yaml`.
   The ConfigMap carries `elastic_workload.py`, the `simple_model` package,
   `elastic/manager.py`, and `test/mock_fault_injection.py` from your checkout
   so edits are picked up without rebuilding the image.
3. Runs `kubectl apply -f /tmp/gke_elastic_workload.yaml` against the current
   context (pass `--no-apply` to only render the YAML and print the command).
4. Starts the bridge on `http://localhost:48259/` and opens it in your browser
   (`--no-browser` to skip, `--host=0.0.0.0 --port=PORT` on a remote VM).

To deploy a manifest you wrote or edited yourself, pass it instead of the
image flags; `--jobset-name` and `--namespace` must match the manifest:

```bash
python3 -m pathwaysutils.extras.elastic.gke_demo_bridge \
  --yaml=my_workload.yaml --jobset-name=my-jobset --namespace=default
```

Running the command with no deployment flags starts the bridge in
monitor-only mode against an already running JobSet.

#### 4. Drive the demo from the browser

* Step 1 shows the selected kubeconfig context and namespace and checks
  cluster connectivity.
* Step 2 shows the generated YAML. Changing topology, slice counts, images, or
  elasticity mode here rewrites `/tmp/gke_elastic_workload.yaml`; re-run the
  command above (or `kubectl apply` the file) to redeploy.
* Step 3 detects the JobSet once the head pod is running, then lets you start
  or pause training, inject real node preemptions (`kubectl cordon` +
  `drain`) or in-workload fault signals, and watch step, loss, goodput, and
  slice topology update live from the head pod logs.

The head pod waits for the Start button (`--wait_for_start_signal=true`), so
nothing trains until you click it.

#### 5. Clean up

```bash
python3 -m pathwaysutils.extras.elastic.gke_demo_bridge --delete
```

This deletes the JobSet and the `elastic-workload-code` ConfigMap from the
configured context and namespace (pass the same `--context`, `--namespace`,
and `--jobset-name` you deployed with). The Delete button in Step 3 does the
same thing and also uncordons any nodes drained during the demo.

#### Useful flags

| Flag | Purpose |
| --- | --- |
| `--context`, `--namespace`, `--jobset-name` | Where the JobSet lives (defaults: current context, `default`, `elastic-demo`). |
| `--image`, `--build-image`, `--builder`, `--project` | Workload image and how to build it. |
| `--server-image`, `--proxy-image` | Pathways server and proxy images (default to the public `cloud-tpu-v2-images/pathways` images). |
| `--gcs-scratch` | Pathways GCS scratch location (required to deploy). |
| `--tpu-type`, `--topology`, `--num-slices`, `--standby-slices` | Slice shape and counts. |
| `--total-steps`, `--elasticity-mode` | Workload length and recovery strategy. |
| `--orbax-ref` | Git ref of `google/orbax` installed in the head pod (default `main`; the demo needs an unreleased snapshot feature). |
| `--kueue-queue`, `--cpu-nodepool` | Optional scheduling hints. |
| `--yaml`, `--apply/--no-apply`, `--delete` | Bring your own manifest, skip applying, or tear down. |
