# Shared Pathways Service

Shared pathways service is a multi-tenant Pathways cluster with dedicated TPU
resources. This eliminates the need for complex cloud setup, allowing you to
get started from a familiar local environment (like a laptop or cloud VM) with
minimal overhead: Just wrap your Python entrypoint in a
`with isc_pathways.connect():` block!.

## Requirements

Make sure that your cluster is running the Resource Manager and Worker pods.
If not, you can use [pw-service-example.yaml](yamls/pw-service-example.yaml).
Make sure to modify the following values to deploy these pods:

- A unique Jobset name for the cluster's Pathways pods
- GCS bucket path
- TPU type and topology
- Number of slices

These fields are highlighted in the YAML file with trailing comments for easier
understanding.

## Instructions

1. Clone `pathwaysutils`.

`git clone https://github.com/AI-Hypercomputer/pathways-utils.git`

2. Import `isc_pathways.py` and move your workload under
`with isc_pathways.connect()` statement. Refer to
[run_connect_example.py](run_connect_example.py) for reference. Example code:

```
 from pathwaysutils.experimental.shared_pathways_service import isc_pathways

 with isc_pathways.connect(
     "my-cluster",
     "my-project",
     "region",
     "gs://user-bucket",
     "pathways-cluster-pathways-head-0-0.pathways-cluster:29001",
     {"tpuv6e:2x2": 2},
 ) as tm:
   import jax.numpy as jnp
   import pathwaysutils
   import pprint

   pathwaysutils.initialize()
   orig_matrix = jnp.zeros(5)
   ...
```
