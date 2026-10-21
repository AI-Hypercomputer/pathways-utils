---
name: pathways-elastic-demo
description: >-
  Guides users end-to-end through setting up, customizing, deploying, and
  running the live Pathways elastic training GKE demo and monitoring dashboard
  from pathways-utils. Use when a user wants to set up or run the Pathways
  elasticity demo on GKE, create a Python 3.11+ environment for pathways-utils,
  connect kubectl to their TPU cluster, build the workload container image,
  render or customize elastic_demo.yaml (Kueue, CPU node pools, tolerations,
  resource limits, elasticity modes), launch gke_demo_bridge, or inject slice
  preemptions and monitor recovery. Don't use for non-GKE local-only unit tests
  unless preparing to deploy the demo, or for unrelated Kubernetes workloads.
---

# Pathways Elastic Training GKE Demo

Every GKE TPU cluster differs in Kueue admission queues, coordinator CPU node pools, node taints, IAM permissions, and available TPU slice capacity. Inspecting existing local settings first and asking targeted upfront questions for anything unclear prevents mid-run failures, allowing Python 3.11+ setup, container building, `elastic_demo.yaml` customization, `JobSet` deployment, and live dashboard monitoring to complete cleanly in a single pass.

## End-to-End Flow

1. Run read-only checks on local Python, `gcloud`, and `kubectl` state, then ask the user upfront for any missing or ambiguous cluster, registry, storage, or scheduling details before creating or modifying resources.
2. Clone `pathways-utils` if not already present, create a verified Python `>= 3.11` virtual environment, and install `.[elastic-demo]`.
3. Fetch GKE cluster credentials, pin `--context` explicitly on every command, and inspect CPU node capacity, TPU node pools, taints, Kueue queues, and active workloads.
4. Build and push the JAX head image and render `elastic_demo.yaml` using the standard public Pathways server and proxy images unless custom images were requested.
5. Apply cluster-specific Kueue labels, CPU `nodeSelector`, container resource limits, TPU `tolerations`, and workload flags to `elastic_demo.yaml`, then validate with `kubectl apply --dry-run=server`.
6. Deploy `elastic_demo.yaml` with `gke_demo_bridge`, open the single-page monitoring dashboard on port `48259` locally or via SSH port forwarding, start training, and run live slice disruptions and scale-up recovery.
7. Delete the `JobSet` and `ConfigMap` cleanly and verify no TPU nodes remain cordoned.

---

## Step 1: Inspect Existing Environment & Ask Upfront Questions

Run quick read-only checks first so you only ask for details the environment does not already provide:

1. Check whether the current working directory already contains `pathwaysutils/extras/elastic/gke_demo_bridge.py` and check `git branch --show-current`.
2. Check available Python interpreters with `python3 --version` and `which -a python3.13 python3.12 python3.11 python3 pyenv uv conda`.
3. Check active Google Cloud and Kubernetes configuration:
   - `gcloud config get-value project`
   - `gcloud container clusters list` (add `--project={project_id}` once known)
   - `gcloud artifacts repositories list --location={region} --project={project_id}` (once project and region are known)
   - `kubectl config current-context` and `kubectl config get-contexts -o name`

### Questions to Ask Before Building or Editing Manifests

Never guess cluster names, registries, GCS buckets, queues, or image overrides. Ask one concise, grouped set of questions covering every item that has not yet been specified or unambiguously discovered:

1. Ask which GCP project ID, GKE cluster name, and region or zone should host the demo. If the requested cluster name matches closely rather than identically—or if all matching TPU pools on that cluster are busy running other pods—list the discovered candidate clusters and free TPU pool counts so the user can choose.
2. Confirm the desired TPU type (`tpuv5e`, `tpuv6e`, `tpuv5p`, `tpuv7x`, or `tpuv4`), slice topology (such as `4x8` or `2x4`), active slice count (`--num-slices`, typically `2` so scale-down to `1` slice and scale-up back to `2` slices can run live), and elasticity recovery mode:
   - `step_snapshot_replica_resize` (recommended default) continues training at reduced batch size on surviving healthy slices during an outage and scales back up when capacity returns.
   - `step_snapshot_pause_resume` pauses training during slice loss, restores the latest in-memory snapshot onto replacement slice capacity, and resumes.
3. Confirm the target Artifact Registry image URI (`{region}-docker.pkg.dev/{project_id}/{artifact_registry_repo}/elastic-demo:latest`), whether to build via Cloud Build (`--build-image`, default), build locally via Docker (`--build-image --builder=docker`), or reuse an existing pushed image, and whether to use the standard public Pathways server and proxy images (recommended default) or custom `--server-image` / `--proxy-image` URIs.
4. Ask which GCS bucket path (`gs://{gcs_bucket}/elastic-demo/`) Pathways should use for scratch files and checkpoints, confirming that the cluster's node or Workload Identity service account has read/write access to that bucket.
5. If cluster inspection shows multiple options or restricted permissions, confirm which Kueue `LocalQueue` should admit the `JobSet`, whether `pathways-head` should pin to a dedicated CPU node pool, and whether any custom Kubernetes `namespace`, `serviceAccountName`, `imagePullSecrets`, or step/checkpoint intervals are required.
6. Confirm whether the user is opening their browser on the same workstation running `gke_demo_bridge` (`http://localhost:48259/`) or connecting from a laptop browser to a remote Linux VM.

---

## Step 2: Set Up Python 3.11+ & Install `pathways-utils`

`pathways-utils` requires Python `>= 3.11`. When `python3` on `PATH` resolves to Python `3.10` (common on older Linux distributions or unpinned `pyenv` shims), `numpy` and `jax` have no prebuilt wheels for 3.10 and `pip install` fails attempting to compile from source.

1. Clone the repository if the user does not already have a local checkout:
   ```bash
   git clone -b elastic-demo https://github.com/AI-Hypercomputer/pathways-utils.git
   cd pathways-utils
   ```
2. Select or configure a Python `>= 3.11` interpreter explicitly before creating `.venv`:
   - When `pyenv` is installed, inspect `pyenv versions`, install Python 3.12 if needed (`pyenv install -s 3.12`), pin it locally (`pyenv local 3.12`), and create `.venv` with `pyenv exec python3 -m venv .venv`.
   - When `python3.12` or `python3.11` is available directly on `PATH`, run `python3.12 -m venv .venv`.
   - When `uv` is installed, run `uv venv --python 3.12 .venv`.
3. Verify the interpreter inside `.venv` before installing packages:
   ```bash
   .venv/bin/python --version   # Must output Python 3.11.x or newer
   .venv/bin/pip install --upgrade pip
   .venv/bin/pip install -e ".[elastic-demo]"
   ```
4. Invoke `.venv/bin/python` directly in every subsequent command so subshells or version-manager shims never fall back to an older system Python.

---

## Step 3: Connect `kubectl` & Inspect Cluster Topology

Fetch credentials for the target cluster and capture the exact `kubeconfig` context name:

```bash
# Pass --region {region} for regional clusters or --zone {zone} for zonal clusters
gcloud container clusters get-credentials {cluster_name} \
  --region {region} \
  --project {project_id}

kubectl config current-context
```

> [!IMPORTANT]
> Other terminal sessions or tools can switch `kubectl config current-context` at any time. Record `{context}` and pass `--context={context}` explicitly on every `kubectl` and `gke_demo_bridge` invocation.

Run these read-only checks against `{context}` so you have exact cluster facts before rendering or modifying `elastic_demo.yaml`:

1. Inspect node pools, TPU accelerator and topology labels, allocatable CPU/memory, cordoned state, and node taints:
   ```bash
   kubectl --context={context} get nodes \
     -o custom-columns='NAME:.metadata.name,POOL:.metadata.labels.cloud\.google\.com/gke-nodepool,ACCEL:.metadata.labels.cloud\.google\.com/gke-tpu-accelerator,TOPO:.metadata.labels.cloud\.google\.com/gke-tpu-topology,CPU:.status.allocatable.cpu,MEM:.status.allocatable.memory,UNSCHED:.spec.unschedulable,TAINTS:.spec.taints'
   ```
   - Identify the CPU node pool (nodes without `ACCEL`) and record its allocatable CPU and memory capacity.
   - Identify all TPU node pools matching the target accelerator and topology, note their exact `TAINTS` (typically `google.com/tpu=present:NoSchedule`), and check whether any TPU nodes still show `UNSCHED=true` from an earlier cordoned run (offer to run `kubectl --context={context} uncordon {node_name}` before deploying).
2. Check active pod placement across node pools:
   ```bash
   kubectl --context={context} get pods -A -o wide
   ```
   - Verify that enough idle TPU node pools exist for `--num-slices` (plus an extra idle slice pool if demonstrating `step_snapshot_pause_resume` replacement scheduling).
3. Verify the `JobSet` CRD and check whether Kueue manages workload admission:
   ```bash
   kubectl --context={context} get crd jobsets.jobset.x-k8s.io
   kubectl --context={context} get localqueues -n {namespace}
   ```
   - If the `JobSet` CRD is missing, pause and help install the [Kubernetes JobSet controller](https://jobset.sigs.k8s.io/docs/installation/) before continuing.
   - If `localqueues` returns queues, note the appropriate multi-slice queue name (or ask the user to confirm which queue to use).

---

## Step 4: Build Workload Image & Render `elastic_demo.yaml`

Verify that `.dockerignore` excludes `.venv` so local virtual environments are never uploaded into the container build context. If the target Artifact Registry repository does not exist yet, ask whether to create it (`gcloud artifacts repositories create {artifact_registry_repo} --repository-format=docker --location={region} --project={project_id}`). When building locally with `--builder=docker`, ensure Docker registry auth is configured via `gcloud auth configure-docker {region}-docker.pkg.dev`.

Build and push the JAX workload image and render the base two-document manifest (`ConfigMap` + `JobSet`):

```bash
.venv/bin/python -m pathwaysutils.extras.elastic.gke_demo_bridge \
  --context={context} \
  --project={project_id} \
  --build-image \
  --image={region}-docker.pkg.dev/{project_id}/{artifact_registry_repo}/elastic-demo:latest \
  --render-only elastic_demo.yaml \
  --gcs-scratch=gs://{gcs_bucket}/elastic-demo/ \
  --tpu-type={tpu_type} \
  --topology={topology} \
  --num-slices={num_slices} \
  --elasticity-mode={elasticity_mode}
```

Keep these details in mind when rendering:
- Leave `--server-image` and `--proxy-image` unset unless the user requested custom images. `gke_demo_bridge` defaults to `us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:latest` and `us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:latest`, which already support elastic training via `--num_elastic_instances`. Do not grep `--helpfull` output inside `proxy_server:latest` to test flag support—the sanitized proxy binary obfuscates flag names in help text even though `--num_elastic_instances` runs natively.
- Pass `--kueue-queue={local_queue}`, `--cpu-nodepool={cpu_nodepool}`, and `--total-steps={steps}` directly during `--render-only` whenever those values are already known.
- Drop `--build-image` on subsequent re-renders once the workload image has been pushed to Artifact Registry.

---

## Step 5: Customize & Validate `elastic_demo.yaml`

`elastic_demo.yaml` contains two resources:
1. A `ConfigMap` (`elastic-workload-code`) bundling `elastic_workload.py`, `simple_model`, `manager.py`, and `mock_fault_injection.py`.
2. A `JobSet` (`elastic-demo`) containing `pathways-head` (`jax-tpu` container on port `8090` plus init sidecar containers `pathways-rm` and `pathways-proxy`) and `pathways-worker` (`replicas: {num_slices}`).

Tailor `elastic_demo.yaml` to match every cluster requirement identified in Step 3:

1. Add the Kueue queue label under `JobSet.metadata.labels` if the cluster admits workloads through Kueue, or omit it if Kueue is not installed:
   ```yaml
   metadata:
     name: elastic-demo
     namespace: default
     labels:
       kueue.x-k8s.io/queue-name: {local_queue}
   ```
2. Pin `pathways-head` to the cluster's CPU pool when dedicated coordinator nodes exist, add any required CPU-pool `tolerations`, and verify container CPU/memory limits fit single-node allocatable resources:
   ```yaml
   nodeSelector:
     cloud.google.com/gke-nodepool: {cpu_nodepool}
   ```
   By default, `jax-tpu` sets `requests: 2 CPU / 8G, limits: 24 CPU / 100G`, `pathways-proxy` sets `requests: 2 CPU / 8G, limits: 16 CPU / 100G`, and `pathways-rm` sets `requests: 1 CPU / 4G, limits: 8 CPU / 32G`. Lower `limits` if the cluster enforces a `LimitRange` or runs smaller CPU VMs.
3. Add matching TPU taint `tolerations` under the `pathways-worker` pod `spec` when TPU nodes are tainted (commonly `google.com/tpu=present:NoSchedule`):
   ```yaml
   tolerations:
   - key: google.com/tpu
     operator: Equal
     value: present
     effect: NoSchedule
   ```
   Never hand-edit `parallelism`, `completions`, `google.com/tpu` counts, or `cloud.google.com/gke-tpu-*` selectors—re-run Step 4 without `--build-image` whenever `--tpu-type`, `--topology`, or `--num-slices` changes.
4. Adjust workload flags in the `jax-tpu` startup command if needed (`--total_steps`, `--elasticity_mode`, `--checkpoint_interval` defaulting to `200`, `--snapshot_interval` defaulting to `2`, or `--checkpoint_dir`), while keeping `--enable_control_server=true`, `--control_port=8090`, and `--wait_for_start_signal=true` intact so the dashboard controls remain active. Add `serviceAccountName` or `imagePullSecrets` to both pod specs if required by cluster policy.
5. Validate the finished manifest against the cluster API server before deploying:
   ```bash
   kubectl --context={context} apply --dry-run=server -f elastic_demo.yaml
   ```
   Fix any schema, webhook, quota, or Kueue errors before proceeding.

---

## Step 6: Deploy & Launch the Monitoring Dashboard

> [!WARNING]
> Kubernetes `JobSet` templates are immutable after creation. If an older `JobSet` with the same name already exists in `{namespace}`, delete it first using Step 7 before applying an edited manifest.

Start `gke_demo_bridge` to apply `elastic_demo.yaml` and serve the single-page monitoring dashboard on port `48259`:

1. When the browser runs on the same local machine:
   ```bash
   .venv/bin/python -m pathwaysutils.extras.elastic.gke_demo_bridge \
     --context={context} \
     --yaml=elastic_demo.yaml \
     --jobset-name={jobset_name} \
     --namespace={namespace}
   ```
   Open `http://localhost:48259/`.
2. When running on a remote VM accessed from a local laptop browser:
   - Start the bridge on `127.0.0.1` without launching a remote browser:
     ```bash
     .venv/bin/python -m pathwaysutils.extras.elastic.gke_demo_bridge \
       --context={context} \
       --yaml=elastic_demo.yaml \
       --jobset-name={jobset_name} \
       --namespace={namespace} \
       --no-browser
     ```
   - Forward port `48259` from the laptop over SSH and open `http://localhost:48259/`:
     ```bash
     ssh -L 48259:127.0.0.1:48259 {user}@{vm_host}
     ```
   - Alternatively, to expose the HTTP server directly on a trusted network interface, pass `--host={vm_ip_or_fqdn} --port=48259 --no-browser` and open `http://{vm_ip_or_fqdn}:48259/` (passing the exact IP or hostname to `--host` adds it to the bridge's `Host`/`Origin` allowlist; `--host=0.0.0.0` allows `localhost` and the machine's short/FQDN hostnames).
3. To re-open the monitoring dashboard later without re-applying `elastic_demo.yaml`, omit `--yaml` and `--image`:
   ```bash
   .venv/bin/python -m pathwaysutils.extras.elastic.gke_demo_bridge \
     --context={context} \
     --jobset-name={jobset_name} \
     --namespace={namespace}
   ```

### Operating the Live Dashboard

Walk the user through what to expect once the monitoring view opens:

- Wait until `pathways-head` and the `pathways-worker` pods start and finish pre-compiling slice executables (`[WORKLOAD READY] Waiting for Start signal...`). Because `--wait_for_start_signal=true` is enabled, training holds at Step 0 until the user clicks `Start Training` in the dashboard.
- Use `Start Training`, `Pause`, and `Reset` to control step progression on `pathways-head` port `8090` while tracking live step count, loss, cumulative goodput, logical workload slices, physical TPU node pools, and JAX driver log events.
- Trigger `Node Cordon + Drain` to inject a real GKE node preemption (`kubectl cordon` + `kubectl drain`) on a worker slice, watching `step_snapshot_replica_resize` reconfigure onto the remaining healthy slices seamlessly or `step_snapshot_pause_resume` wait for spare slice scheduling.
- Trigger `In-Workload RPC` when fast software-level slice loss and reconnect testing is preferred without draining Kubernetes nodes.
- Click `Reconnect Slice` to run `kubectl uncordon` (or `/reconnect`) so the slice rejoins the cluster and scales training back up.

---

## Step 7: Clean Up Resources

When finished—or prior to re-deploying a modified `JobSet`—delete the `JobSet` and `ConfigMap`:

```bash
.venv/bin/python -m pathwaysutils.extras.elastic.gke_demo_bridge \
  --context={context} \
  --delete \
  --jobset-name={jobset_name} \
  --namespace={namespace}
```

Clicking `Delete` inside the browser header deletes the `JobSet` and `ConfigMap` and uncordons any nodes drained during that session. Whenever tearing down via `--delete` on the command line, check that no TPU nodes remain marked `SchedulingDisabled`:

```bash
kubectl --context={context} get nodes \
  --selector=cloud.google.com/gke-tpu-accelerator \
  -o jsonpath='{.items[?(@.spec.unschedulable==true)].metadata.name}'
```

If any nodes cordoned during the demo are listed, run `kubectl --context={context} uncordon {node_name}` so all TPU node pools return to healthy schedulable state.
