# Shared Pathways Service

The Shared Pathways Service accelerates developer iteration by providing a
persistent, multi-tenant TPU environment. This decouples service creation from
the development loop, allowing JAX clients to connect on-demand from a familiar
local environment (like a laptop or cloud VM) to a long-running Pathways
service that manages scheduling and error handling.

## Requirements

Make sure that your GKE cluster is running the Resource Manager and Worker pods.
You can follow the steps
[here](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/troubleshooting-pathways#health_monitoring)
to confirm the status of these pods. If you haven't started the Pathways pods
yet, you can use [pw-service-example.yaml](yamls/pw-service-example.yaml).
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

2. Import `isc_pathways` and move your workload under
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

The connect block will deploy a proxy pod dedicated to your client and connect
your local runtime environment to the proxy pod via port-forwarding.
