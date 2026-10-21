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
  provides a live interactive web application with simulation controls.
* Local browser viewing allows running `model_comparison.html` directly on your workstation by opening the file in Chrome, Firefox, or Safari, requiring zero installation or server dependencies.

### Recovery Paradigms Compared

* Traditional Checkpoint Rollback (McJAX) halts training upon interruption, incurs a reinitialization latency, and rewinds training state to the most recent checkpoint, resulting in lost forward progress.
* Snapshot Replica Resize dynamically adjusts the mesh to continue training on surviving healthy slices with a smaller reinitialization latency than a cold start, maintaining forward progress at a degraded throughput factor (depending on your recovery implementation) with no time spent waiting for slices to rejoin.
* Snapshot Pause Resume pauses training during interruptions, rewinds to the last snapshot, and resumes with reinitialization once replacement slices become healthy.

### Interactive Parameter Controls

The simulation toolbar allows tuning key workload and infrastructure characteristics in real time:

* Cold Start Initialization Time (seconds) configures the initial cluster startup delay prior to the first training step and is reused as the reinitialization latency for McJAX.
* Scale-Down Initialization Time (seconds) sets the reconfiguration overhead when transitioning into degraded execution.
* Scale-Up Initialization Time (seconds) governs the mesh reconfiguration latency when a replacement slice joins the workload.
* Pause/Resume Reinitialization Time (seconds) is the latency Snapshot Pause Resume pays to restore its snapshot on the replacement slice; it defaults to less than the scale-up time since no mesh reconfiguration is involved.
* Degraded Scaling Factor (multiplier) sets the relative step throughput while executing on a reduced slice topology.
* Initial Base Loss (L₀) sets the initial loss baseline at step 0 in the exponential decay function L = L₀ * exp(-α * step), which decays toward zero.
* Loss Decay Rate (α, default 0.005) dictates the per-step exponential decay rate, multiplied by the degraded scaling factor during degraded execution so training loss decays slower per step.
* Checkpoint Interval (steps) dictates the rollback penalty for McJAX, which rewinds to the last checkpoint on every interruption.
* Snapshot Interval (steps) dictates the rollback penalty for the snapshot strategies: Snapshot Pause Resume rewinds to the last snapshot on every interruption, and Snapshot Replica Resize rewinds on scale-down only, since a later scale-up re-shards live state without losing progress.
* Interruption Duration for Scenario (seconds) configures hardware disruption length during automated scenario runs.

## Live GKE Elasticity Demo

`gke_elasticity_demo.html`, `gke_demo_bridge.py`, and `elastic_workload.py`
drive and monitor a real Pathways training JobSet on a GKE TPU cluster.

* `elastic_workload.py` runs the `pathwaysutils.extras.simple_model` workload
  with `ElasticManager` and `Snapshotter`, exposing an HTTP control server on
  head pod port `8090` for start, pause, reset, and fault-injection signals.
* `gke_demo_bridge.py` renders the ConfigMap and JobSet manifest bundle from the
  command line, applies or deletes deployments, binds to `127.0.0.1:48259` by
  default, streams live JAX driver telemetry over Server-Sent Events (SSE), and
  triggers disruptions via Kubernetes `kubectl cordon` and `drain` or
  in-workload HTTP signaling.
* `gke_elasticity_demo.html` provides a single-screen monitoring view for live
  training step, loss, cumulative goodput, logical workload slices, physical
  cluster TPU slices, recovery stages, and the JAX driver event stream.

### Running the Live GKE Demo

The flow is: build the workload image once, render a base ConfigMap + JobSet
YAML, adjust the YAML to match how your cluster is set up, deploy it, and open
the demo page. Every command below runs from your workstation.

#### What you need

* A GKE cluster with the [JobSet](https://jobset.sigs.k8s.io/) controller
  installed and TPU node pools for the slices you want to run. You pass the TPU
  type, topology, and slice count as flags and
  `pathwaysutils.experimental.gke.jobset.PathwaysJobSet` generates the worker
  templates for that shape; any additional idle pools of the same shape are
  discovered automatically and shown as spares. Anything else about the
  cluster, such as Kueue, a dedicated CPU node pool for the head pod, custom
  tolerations, or a workload service account, is handled by editing the YAML
  in step 4 below.
* A GCS bucket the cluster's node service account can read and write, used as
  the Pathways scratch location.
* An Artifact Registry repository you can push the workload image to, plus
  either Cloud Build permissions or a local Docker daemon.
* Locally: Python 3.11+, `git`, `gcloud`, `kubectl`, and the
  `gke-gcloud-auth-plugin` component.

#### 1. Clone the branch and install

```bash
git clone -b elastic-demo https://github.com/AI-Hypercomputer/pathways-utils.git
cd pathways-utils
python3.12 -m venv .venv && source .venv/bin/activate   # any Python >= 3.11
pip install --upgrade pip
pip install -e ".[elastic-demo]"
```

Check `python --version` inside the venv before installing. If an older
interpreter such as 3.10 is first on your `PATH`, pip cannot find NumPy or JAX
wheels for it and falls back to a source build that fails.

#### 2. Point kubectl at your cluster

```bash
gcloud container clusters get-credentials CLUSTER --region REGION --project PROJECT
kubectl config current-context   # the demo uses this context unless --context is set
```

#### 3. Build the workload image and render the base YAML

```bash
python3 -m pathwaysutils.extras.elastic.gke_demo_bridge \
  --build-image \
  --image=REGION-docker.pkg.dev/PROJECT/REPO/elastic-demo:latest \
  --render-only elastic_demo.yaml \
  --gcs-scratch=gs://BUCKET/elastic-demo/ \
  --tpu-type=tpuv5e --topology=4x8 --num-slices=2
```

This builds `pathwaysutils/extras/elastic/Dockerfile` with Cloud Build, pushes
it to `--image`, writes `elastic_demo.yaml`, and exits. Add `--builder=docker`
to build with a local Docker daemon instead, or drop `--build-image` on later
runs to only re-render the YAML. The image bundles JAX, Flax, Grain, Orbax,
and this checkout of `pathwaysutils`.

#### 4. Adjust the YAML for your cluster

`elastic_demo.yaml` contains two documents: a `ConfigMap` named
`elastic-workload-code` holding `elastic_workload.py`, the `simple_model`
package, `elastic/manager.py`, and `test/mock_fault_injection.py` from your
checkout, followed by the `JobSet` produced by `PathwaysJobSet`. The JobSet
has one `pathways-head` replicated job (the JAX driver plus the Pathways
resource manager and proxy as init containers) and one `pathways-worker`
replicated job per slice. Open the file and
change whatever your cluster requires. Common edits:

* Kueue: if your cluster admits workloads through Kueue, add the queue label
  under the JobSet `metadata`:
  ```yaml
  metadata:
    name: elastic-demo
    namespace: default
    labels:
      kueue.x-k8s.io/queue-name: YOUR-LOCAL-QUEUE
  ```
  Leave it out if you do not use Kueue; the `kueue.x-k8s.io/podset` labels
  already present on the pod templates are harmless without it.
* Head pod placement: the head pod requests `cpu: "24"` and `memory: 100G`.
  Lower those `resources.limits` to fit your nodes, or pin it to a node pool
  by adding a `nodeSelector` to the `pathways-head` pod `spec`:
  ```yaml
  nodeSelector:
    cloud.google.com/gke-nodepool: YOUR-CPU-NODEPOOL
  ```
* TPU workers: the `pathways-worker` template (node selectors, `google.com/tpu`
  limits, `parallelism`/`completions`) is generated by `PathwaysJobSet` from
  `--tpu-type`, `--topology`, and `--num-slices`, so re-render rather than
  hand-editing those; add `tolerations` to the worker pod `spec` if your TPU
  node pools are tainted.
* Namespace, service account, image pull secrets, or anything else your
  cluster policy requires can be added to the pod `spec`s directly.
* The training flags live in the head container `command`; `--total_steps`,
  `--elasticity_mode`, and `--config_name` can be changed there without
  re-rendering.

Keep the JobSet `metadata.name` and `metadata.namespace` in mind; you pass
them to the bridge in the next step so it can find the pods.

To re-render after changing flags, run the step 3 command again without
`--build-image`; it overwrites the file.

#### 5. Deploy and open the UI

```bash
python3 -m pathwaysutils.extras.elastic.gke_demo_bridge \
  --yaml=elastic_demo.yaml --jobset-name=elastic-demo --namespace=default
```

This runs `kubectl apply -f elastic_demo.yaml` against the current context,
starts the bridge on `http://localhost:48259/`, and opens it in your browser
(`--no-browser` to skip, `--host=0.0.0.0 --port=PORT` when running on a remote
VM). Pass `--no-apply` if you prefer to run `kubectl apply` yourself.

If you later re-run with no deployment flags, the bridge starts in
monitor-only mode and attaches to the JobSet that is already running.

#### 6. Drive the demo from the browser

The monitoring page attaches to the JobSet once the head pod starts. From that
single view you can start, pause, or reset training, inject real node
preemptions (`kubectl cordon` + `drain`) or in-workload fault signals, and watch
step progress, loss, goodput, recovery stages, and slice placement update live
from the head pod logs. The cluster pool section lists matching TPU node pools
from `cloud.google.com/gke-nodepool` and marks each pool as active, spare, busy,
or cordoned.

The head pod waits for the Start Training button (`--wait_for_start_signal=true`),
so nothing trains until you click it.

#### 7. Clean up

```bash
python3 -m pathwaysutils.extras.elastic.gke_demo_bridge --delete \
  --jobset-name=elastic-demo --namespace=default
```

This deletes the JobSet and the `elastic-workload-code` ConfigMap from the
configured context and namespace. The Delete button in the header does the same
thing and also uncordons any nodes drained during the demo.

#### Shortcut for clusters that need no YAML edits

If the base YAML already fits your cluster, one command does steps 3 through 5
together:

```bash
python3 -m pathwaysutils.extras.elastic.gke_demo_bridge \
  --build-image \
  --image=REGION-docker.pkg.dev/PROJECT/REPO/elastic-demo:latest \
  --gcs-scratch=gs://BUCKET/elastic-demo/ \
  --tpu-type=tpuv5e --topology=4x8 --num-slices=2
```

#### Useful flags

| Flag | Purpose |
| --- | --- |
| `--context`, `--namespace`, `--jobset-name` | Where the JobSet lives (defaults: current context, `default`, `elastic-demo`). |
| `--image`, `--build-image`, `--builder`, `--project` | Workload image and how to build it. |
| `--server-image`, `--proxy-image` | Pathways server and proxy images (default to the public `cloud-tpu-v2-images/pathways` images). |
| `--gcs-scratch` | Pathways GCS scratch location (required to render a deployable YAML). |
| `--tpu-type`, `--topology`, `--num-slices` | Slice shape and count in the rendered YAML. Spare node pools are not configured; the bridge discovers every pool with matching accelerator and topology labels and marks each as active (yours), spare, busy (hosting another workload's TPU pods), or cordoned. |
| `--total-steps`, `--elasticity-mode` | Workload length and recovery strategy in the rendered YAML. |
| `--orbax-ref` | Git ref of `google/orbax` installed in the head pod (default `main`; the demo needs an unreleased snapshot feature). |
| `--kueue-queue`, `--cpu-nodepool` | Convenience flags that add the Kueue queue label or head pod `nodeSelector` to the rendered YAML, equivalent to the manual edits above. |
| `--render-only PATH`, `--yaml PATH`, `--apply/--no-apply`, `--delete` | Render the base YAML, deploy an edited one, skip applying, or tear down. |
