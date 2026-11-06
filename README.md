# APIs to get started with using Pathways on Cloud

Pathways is a compute orchestration runtime for single-controller ML workloads
across multiple Cloud TPU slices. A single Python controller orchestrates
computation across hundreds or thousands of TPU chips, unlocking slice-level
training resiliency, fast cross-slice data movement, and interactive
multi-slice development.

This package bundles the Python client integrations, resharding and elasticity
helpers, and GKE deployment builders for running JAX workloads on Pathways on
Cloud.

## Installation

```bash
pip install pathwaysutils
# Optional GKE JobSet generator and kubectl/Kubernetes client dependencies:
pip install 'pathwaysutils[gke]'
# Optional experimental Shared Pathways Service (SPS) dependencies:
pip install 'pathwaysutils[sps]'
```

## Core Modules

- `pathwaysutils.initialize()` registers the Pathways IFRT proxy backend,
  configures the compilation cache and XProf trace hooks, and registers Orbax
  checkpoint handlers.
- `pathwaysutils.reshard.reshard` reshards arrays and PyTrees across device
  meshes directly on Pathways workers via the IFRT `RemapArray` API without
  routing tensors through controller memory.
- `pathwaysutils.elastic.manager.Manager` monitors slice health in the
  background and provides `@elastic_retry` and `ScaleUpSignalError` for
  step-level fault recovery and dynamic slice scaling.
- [`PathwaysJobSet`][gke-readme] (`pathwaysutils.experimental.gke.jobset`)
  builds, customizes, exports, and deploys Kubernetes `JobSet` manifests for
  Pathways workloads on GKE.
- [Shared Pathways Service][sps-readme]
  (`pathwaysutils.experimental.shared_pathways_service`) deploys a persistent
  multi-tenant Pathways service on GKE for interactive JAX sessions.

## Further Reading

- [Get started with Pathways on Google Cloud][cloud-docs]
- [Pathways Research Paper](https://arxiv.org/abs/2203.12533)

[gke-readme]: https://github.com/AI-Hypercomputer/pathways-utils/tree/main/pathwaysutils/experimental/gke
[sps-readme]: https://github.com/AI-Hypercomputer/pathways-utils/tree/main/pathwaysutils/experimental/shared_pathways_service
[cloud-docs]: https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro


