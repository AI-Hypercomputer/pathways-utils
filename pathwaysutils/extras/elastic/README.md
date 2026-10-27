# Recovery Mode Simulation & Comparison

Evaluating distributed machine learning training on elastic TPU topologies requires understanding how varying failure scenarios and reinitialization latencies affect cumulative goodput and step progression. This folder provides an interactive browser-based simulation comparing three distinct recovery paradigms: McJAX, Snapshot Replica Resize, and Snapshot Pause Resume.

## Live Interactive Simulation

* [Hosted GitHub Pages Simulation](https://ai-hypercomputer.github.io/pathways-utils/pathwaysutils/extras/elastic/model_comparison.html) is a live interactive web application with simulation controls, or [HTMLPreview Instant Mirror](https://htmlpreview.github.io/?https://github.com/AI-Hypercomputer/pathways-utils/blob/main/pathwaysutils/extras/elastic/model_comparison.html).
* Local browser viewing allows running `model_comparison.html` directly on your workstation by opening the file in Chrome, Firefox, or Safari, requiring zero installation or server dependencies.

## Recovery Paradigms Compared

* Traditional Checkpoint Rollback (McJAX) halts cluster computation upon slice preemption, incurs a reinitialization latency, and rewinds training state to the most recent periodic checkpoint or snapshot, resulting in lost forward progress.
* Snapshot Replica Resize dynamically adjusts the mesh to continue training on surviving healthy slices with a smaller reinitialization latency than a cold start, maintaining forward progress at a degraded throughput factor (depending on your recovery implementation) with no time spent waiting for slices to rejoin.
* Snapshot Pause Resume pauses execution during slice preemption, rewinds to the last snapshot, and resumes with negligible reinitialization once replacement slices become healthy.

## Interactive Parameter Controls

The simulation toolbar allows tuning key workload and infrastructure characteristics in real time:

* Cold Start Initialization Time (seconds) configures the initial cluster startup delay prior to the first training step and is reused as the reinitialization latency for McJAX.
* Scale-Down Initialization Time (seconds) sets the reconfiguration overhead when transitioning into degraded execution.
* Scale-Up Initialization Time (seconds) governs the mesh reconfiguration latency when a replacement slice joins the workload.
* Degraded Scaling Factor (multiplier) sets the relative step throughput while executing on a reduced slice topology.
* Initial Base Loss (L₀) sets the initial loss baseline at step 0 in the exponential decay function L = L₀ * exp(-α * step).
* Loss Decay Rate (α) dictates the per-step exponential decay rate, multiplied by the degraded scaling factor during degraded execution so training loss decays slower per step.
* Checkpoint Interval (steps) dictates the rollback penalty for checkpoint-based training.
* Interruption Duration for Scenario (seconds) configures hardware disruption length during automated scenario runs.
