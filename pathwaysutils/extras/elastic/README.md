# Pathways Elasticity Demos

Evaluating distributed machine learning training on elastic TPU topologies
requires understanding varying failure scenarios, reinitialization latencies,
and their impact on cumulative goodput and step progression. This folder
provides both an interactive browser-based simulation and a live Google
Kubernetes Engine (GKE) demo comparing Pathways elasticity recovery paradigms.

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
  with `ElasticManager` and `Snapshotter`, pre-compiling executables for all
  slice subsets and exposing an HTTP control server on head pod port `8090` for
  start, pause, reset, live elasticity mode switching, and fault-injection
  signals.
* `gke_demo_bridge.py` renders the ConfigMap and JobSet manifest bundle from the
  command line, applies or deletes deployments, binds to `127.0.0.1:48259` by
  default, streams live JAX driver telemetry over Server-Sent Events (SSE), and
  triggers disruptions via Kubernetes `kubectl cordon` and `drain` or
  in-workload HTTP signaling.
* `gke_elasticity_demo.html` provides a single-screen monitoring view for live
  training step, loss, cumulative goodput, logical workload slices, physical
  cluster TPU slices, live elasticity mode selection, recovery stages, and the
  scrollable JAX driver event stream.

### Prerequisites

* A GKE cluster with the [JobSet](https://jobset.sigs.k8s.io/) controller
  installed and TPU node pools matching the slices you want to run. Passing the
  TPU type, topology, and slice count generates worker templates through
  `pathwaysutils.experimental.gke.jobset.PathwaysJobSet`; any additional idle
  pools of the same shape are discovered automatically as spares. Cluster-specific
  settings such as Kueue queues, CPU node pools, tolerations, or workload
  service accounts are applied in the rendered YAML before deployment.
* A Google Cloud Storage (GCS) bucket readable and writable by the cluster's
  node or Workload Identity service account for Pathways scratch storage and
  checkpoints.
* An Artifact Registry Docker repository you can push the workload image to,
  along with either Cloud Build permissions or a local Docker daemon.
* A workstation or cloud VM with Python 3.11+, `git`, `gcloud`, `kubectl`, and
  the `gke-gcloud-auth-plugin` component installed.

### Running the Live GKE Demo

You can either use the bundled agent skill to let an AI coding assistant inspect
your environment, ask any clarifying cluster questions, customize the manifest,
and start the dashboard in a single pass, or walk through each CLI command
step by step.

#### Option 1: Guided Setup with the Agent Skill

This repository includes a portable Agent Skill at
[`.agents/skills/pathways-elastic-demo/SKILL.md`](../../../.agents/skills/pathways-elastic-demo/SKILL.md)
compatible with Gemini CLI, Claude Code, Codex, Cursor, and other tools supporting
`SKILL.md` discovery.

1. Clone the repository and open your coding agent inside the `pathways-utils`
   directory:

   ```bash
   git clone -b elastic-demo https://github.com/AI-Hypercomputer/pathways-utils.git
   cd pathways-utils
   ```

   If your agent looks for project skills under `.claude/skills/` instead of
   `.agents/skills/`, link the directory first:

   ```bash
   mkdir -p .claude/skills
   ln -s ../../.agents/skills/pathways-elastic-demo .claude/skills/pathways-elastic-demo
   ```

2. Send a single prompt asking your agent to drive setup end to end:

   ```text
   Use the pathways-elastic-demo skill to set up and launch the live Pathways GKE elasticity demo on my cluster. Inspect my environment first and ask me anything unclear about my cluster, Artifact Registry repository, GCS bucket, or YAML configuration before building or deploying.
   ```

The skill checks your local Python 3.11+ interpreter (including `pyenv` or `uv`),
inspects your GKE cluster's CPU nodes, TPU node pools, taints, and Kueue queues,
asks one grouped set of questions for any missing inputs, builds and pushes the
workload image, customizes and dry-run validates `elastic_demo.yaml`, and
starts the live monitoring dashboard on port `48259` (locally, via remote
proxy, or via SSH port forwarding).

#### Option 2: Step-by-Step CLI Setup

Every command below runs from your workstation and takes you from a fresh clone
to a live monitoring session: build the workload image once, render a base
ConfigMap + JobSet YAML, tailor the YAML to your cluster, deploy it, and open
the monitoring page.

##### 1. Clone the branch and install

```bash
git clone -b elastic-demo https://github.com/AI-Hypercomputer/pathways-utils.git
cd pathways-utils
python3.12 -m venv .venv   # any Python >= 3.11 (or `pyenv local 3.12` first)
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[elastic-demo]"
```

Verify `python --version` inside `.venv` prints `3.11` or newer before running
`pip install`. When `python3` resolves to Python `3.10`, `pip` cannot find
prebuilt `numpy` or `jax` wheels and fails attempting a source build.

##### 2. Point kubectl at your cluster

```bash
gcloud container clusters get-credentials CLUSTER --region REGION --project PROJECT
kubectl config current-context   # the demo uses this context unless --context is set
```

##### 3. Build the workload image and render the base YAML

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
runs to only re-render the YAML. By default, the manifest uses the standard
public `us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:latest` and
`proxy_server:latest` images and bundles JAX, Flax, Grain, Orbax, and this
checkout of `pathwaysutils` into `--image`.

##### 4. Adjust the YAML for your cluster

`elastic_demo.yaml` contains two documents: a `ConfigMap` named
`elastic-workload-code` holding `elastic_workload.py`, the `simple_model`
package, `elastic/manager.py`, `test/mock_fault_injection.py`, and
`snapshotter.py` from your checkout, followed by the `JobSet` produced by
`PathwaysJobSet`. The JobSet has
one `pathways-head` replicated job (the JAX driver plus the Pathways resource
manager and proxy as init containers) and one `pathways-worker` replicated job
per slice. Review the file and apply any edits required by your cluster:

* If your cluster admits workloads through Kueue, add the queue label under the
  JobSet `metadata`:
  ```yaml
  metadata:
    name: elastic-demo
    namespace: default
    labels:
      kueue.x-k8s.io/queue-name: YOUR-LOCAL-QUEUE
  ```
  Leave it out when Kueue is not installed; the `kueue.x-k8s.io/podset` labels
  already present on the pod templates are harmless without it.
* For head pod placement and sizing, the `jax-tpu` container sets limits of
  `cpu: "24"` and `memory: 100G`. Lower container `resources.limits` to fit
  smaller nodes, or pin `pathways-head` to a dedicated CPU node pool by adding
  a `nodeSelector` to the head pod `spec`:
  ```yaml
  nodeSelector:
    cloud.google.com/gke-nodepool: YOUR-CPU-NODEPOOL
  ```
* On TPU worker pools, the `pathways-worker` template (node selectors,
  `google.com/tpu` limits, `parallelism`/`completions`) is generated by
  `PathwaysJobSet` from `--tpu-type`, `--topology`, and `--num-slices`, so
  re-render without `--build-image` rather than hand-editing those fields. Add
  `tolerations` to the worker pod `spec` when your TPU node pools carry taints
  such as `google.com/tpu=present:NoSchedule`.
* Add `namespace`, `serviceAccountName`, `imagePullSecrets`, or any other pod
  `spec` settings required by your cluster policy, and validate before
  deploying with `kubectl apply --dry-run=server -f elastic_demo.yaml`.
* Update `--total_steps`, `--elasticity_mode`, `--checkpoint_interval`,
  `--snapshot_interval`, or `--config_name` directly inside the head container
  `command` if needed without re-rendering.

Keep the JobSet `metadata.name` and `metadata.namespace` values matching the
flags you pass to the bridge in the next step so it can locate the pods.

##### 5. Deploy and open the UI

```bash
python3 -m pathwaysutils.extras.elastic.gke_demo_bridge \
  --yaml=elastic_demo.yaml --jobset-name=elastic-demo --namespace=default
```

This runs `kubectl apply -f elastic_demo.yaml` against the target context,
starts the bridge on `http://localhost:48259/`, and opens it in your browser
(`--no-browser` skips opening a browser; forward port `48259` over SSH or pass
`--host=IP_OR_HOSTNAME --port=PORT` when running on a remote VM). Pass
`--no-apply` if you prefer to run `kubectl apply` yourself.

Re-running later with no deployment flags starts the bridge in monitor-only mode
and attaches to the JobSet already running on the cluster.

##### 6. Drive the demo from the browser

The monitoring page attaches to the JobSet once the head pod starts. From that
single view you can start, pause, or reset training, switch the elasticity
recovery mode live (`step_snapshot_replica_resize` vs
`step_snapshot_pause_resume`), inject real node preemptions (`kubectl cordon` +
`drain`) or in-workload fault signals, and watch step progress, loss, goodput,
recovery stages, and slice placement update live from the head pod logs. The
cluster pool section lists matching TPU node pools from
`cloud.google.com/gke-nodepool` and marks each pool as active, spare, busy, or
cordoned.

The head pod waits for the Start Training button (`--wait_for_start_signal=true`),
so nothing steps until you click it.

##### 7. Clean up

```bash
python3 -m pathwaysutils.extras.elastic.gke_demo_bridge --delete \
  --jobset-name=elastic-demo --namespace=default
```

This deletes the JobSet and the `elastic-workload-code` ConfigMap from the
configured context and namespace. Clicking `Delete` in the dashboard header
deletes both resources and also uncordons any nodes drained during the session.

#### Shortcut for clusters that need no YAML edits

When the generated YAML fits your cluster directly without edits, one command
runs steps 3 through 5 together:

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
| `--context`, `--namespace`, `--jobset-name` | Cluster context, namespace, and JobSet name (defaults: current context, `default`, `elastic-demo`). |
| `--image`, `--build-image`, `--builder`, `--project` | Workload image URI and build backend (`gcloud` Cloud Build or `docker`). |
| `--server-image`, `--proxy-image` | Pathways server and proxy images (default to public `cloud-tpu-v2-images/pathways` `:latest` images). |
| `--gcs-scratch` | Pathways GCS scratch location (required to render a deployable YAML). |
| `--tpu-type`, `--topology`, `--num-slices` | Slice shape and active slice count in the rendered YAML. Idle pools with matching accelerator and topology labels are discovered automatically and shown as active, spare, busy, or cordoned. |
| `--total-steps`, `--elasticity-mode` | Workload step count and recovery strategy (`step_snapshot_replica_resize` or `step_snapshot_pause_resume`). |
| `--orbax-ref` | Git ref of `google/orbax` installed in the head image (default `main`). |
| `--kueue-queue`, `--cpu-nodepool` | Convenience flags adding the Kueue queue label or head pod `nodeSelector` to the rendered YAML. |
| `--render-only PATH`, `--yaml PATH`, `--apply/--no-apply`, `--delete` | Render base YAML, deploy an edited manifest, skip applying, or delete resources. |
