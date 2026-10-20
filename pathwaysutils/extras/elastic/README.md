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

1. Authenticate `gcloud` and `kubectl` for your target GKE cluster.
2. Start the bridge server and open the demo webpage in one command:
   ```bash
   python3 -m pathwaysutils.extras.elastic.gke_demo_bridge
   ```
   Optional flags let you pre-select a cluster context and namespace:
   ```bash
   python3 -m pathwaysutils.extras.elastic.gke_demo_bridge \
     --context=gke_my-project_us-south1_my-cluster \
     --namespace=default
   ```
   Pass `--port=48259` to choose a custom port, `--host=0.0.0.0` when serving
   through a remote workstation proxy, or `--no-browser` on a remote host.
3. Follow the three steps on the single-page UI:
   - Step 1 selects your GKE kubeconfig context, namespace, and nodepool, and
     checks cluster connectivity.
   - Step 2 configures TPU topology and slice counts, writes the combined
     ConfigMap and JobSet YAML bundle to `/tmp/gke_elastic_workload.yaml`, and
     gives you the exact `kubectl apply -f /tmp/gke_elastic_workload.yaml`
     command to run in your terminal.
   - Step 3 detects the deployed JobSet automatically, starts or pauses
     training, injects live GKE node preemptions or in-workload fault signals,
     and monitors recovery telemetry.
