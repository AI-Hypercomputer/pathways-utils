# Changelog

<!--

Changelog follow the https://keepachangelog.com/ standard (at least the headers)

This allow to:

* auto-parsing release notes during the automated releases from github-action:
  https://github.com/marketplace/actions/pypi-github-auto-release
* Have clickable headers in the rendered markdown

To release a new version (e.g. from `1.0.0` -> `2.0.0`):

* Create a new `# [2.0.0] - YYYY-MM-DD` header and add the current
  `[Unreleased]` notes.
* At the end of the file:
  * Define the new link url:
  `[2.0.0]: https://github.com/google-research/my_project/compare/v1.0.0...v2.0.0`
  * Update the `[Unreleased]` url: `v1.0.0...HEAD` -> `v2.0.0...HEAD`

-->

## [Unreleased]

## [0.1.11] - 2026-07-17
## What's Changed
* Update start_server signature in pathwaysutils.profiling to accept requires_backend by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/267
* Extract TPU specifications and helper functions for Shared Pathways Service by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/272
* Add JAX 0.10.2 to test matrix by @wstcliyu in https://github.com/AI-Hypercomputer/pathways-utils/pull/276
* bump py version in pyproject.toml by @sadikneipp in https://github.com/AI-Hypercomputer/pathways-utils/pull/277
* Automated Code Change by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/271
* Add GCSFuse support to PathwaysJobSet. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/252
* Add Colocated Python support to PathwaysJobSet. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/253
* Unify monitor thread lifecycle and implement robust set-based slice tracking in pathwaysutils. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/257
* Add YAML serialization and deployment to GKE for PathwaysJobSet. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/254
* Add Shared Pathways Service support to PathwaysJobSet. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/255
* Replace Shared Pathways Service YAML templates with PathwaysJobSet. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/256

## [0.1.10] - 2026-06-30
## What's Changed
* Add retry_policy and deprecate max_retries in pathwaysutils.elastic.manager.elastic_retry. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/246
* Add support for a static colocated Python sidecar by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/243
* Refactor pathwaysutils resharding APIs to establish IFRT/sidechannel split. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/248
* Add cluster and job name labels to Shared Pathways Service metrics by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/249
* fix serialization of int values matching tensorflow.ProfileOptions.AdvancedConfigValue by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/260
* Add sidecar image version validation to ISC Pathways connection by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/247
* Fix JAX device compatibility in profiling.py for elasticity. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/261
* Add Worker Job configuration to PathwaysJobSet. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/251
* Monkey-patch jax._src.profiler.start_server and stop_server in pathwaysutils by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/266

## [0.1.9] - 2026-06-12
## What's Changed
* Add `--dns-endpoint` to the get-credentials command by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/224
* Add support for passing XLA flags to the Pathways proxy by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/225
* Generalize TPU Slice Health Checks for elasticity by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/218
* Fix the accelerator label for tpu7x by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/226
* Rename "v5p" to "v5" TPU type by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/230
* Fix argument and resource type injection in gke_utils.py by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/228
* Plumb session_id from ProfileOptions to traceSessionName. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/232
* Add JAX 0.10.1 to test matrix by @wstcliyu in https://github.com/AI-Hypercomputer/pathways-utils/pull/234
* Sync session_id format between JAX client and Pathways server in pathwaysutils. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/236
* Clean up ProfileOptions getattr usages in pathwaysutils. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/237
* Bump min JAX version to 0.8.3 by @wstcliyu in https://github.com/AI-Hypercomputer/pathways-utils/pull/240
* Add base PathwaysJobSet builder by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/239
* Add Head Job configuration to PathwaysJobSet by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/241
* Add a script to deploy VS Code on GKE CPU node pool by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/242

## [0.1.8] - 2026-04-24
## What's Changed
* Add background log streaming to detect TPU placement completion by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/204
* Add a script to deploy Pathways service as a JobSet by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/211
* Integrate metrics collection into ISC Pathways by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/214
* add support for max_num_hosts in start_trace. the default now is to trace one host. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/213
* Remove redundant user waiting metric update by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/215
* Rename pw-service-example to pw-service by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/219
* Expose concatenate_by_mesh_axis in pathwaysutils. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/217
* Update concatenate_by_mesh_axis to preserve memory kind by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/220

## [0.1.7] - 2026-04-03
## What's Changed
* Refactor elasticity retry logic into a reusable private method. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/177
* Add a name parameter to the watchdog context manager. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/199
* Use abstract types for return type hints in pathwaysutils. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/198
* Update environment variables for JAX backend by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/203
* Enable Pathways profiling with jax.profiler.ProfileOptions. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/201
* Refactor elastic retry decorators into a single elastic_retry method. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/202
* Fix JaxRuntimeError during profiler stop_trace with profile options by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/205
* Add CLI mode to Shared Pathways Service by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/200

## [0.1.6] - 2026-02-27
* Fix --proxy_options to use flags.DEFINE_list by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/191

## [0.1.5] - 2026-02-27
* Refactor: Simplify Elasticity Manager to focus on slice availability. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/167
* Make Pathways proxy server image user-configurable by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/159
* This change introduces reshard_with_intermediate_sharding which will first look for intermediate shardings, perform all intermediate resharding, and then perform the final reshard into the out sharding. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/145
* Update active_slice_indices after waiting for slices within pause-resume. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/174
* Add cleanup to restore JAX config and environment variables in tests. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/185
* Change Pathways backend target from `localhost` to `127.0.0.1` by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/186
* Move ifrt based reshard out of experimental. Leaving intermediate resharding and sidechannel resharding in experimental. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/176
* Add support for passing environment variables to the Pathways proxy by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/187

## [0.1.4] - 2026-01-26
* Extract lambda to a named function to ensure cache hits. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/127
* Always write array metadata if self._array_metadata_store is not None. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/125
* Add "Shared Pathways Service" for Pathways-on-Cloud by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/128
* Exposes `pathwaysutils.profiling._start_trace_from_profile_request` as `pathwaysutils.experimental.profiling.start_trace`. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/141
* Patch internal JAX profiler functions (enabling `jax.profiler.trace`) and add a test for `jax.profiler.trace`. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/140
* Expose `_split_by_mesh_axis` directly in `pw_jax`. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/147
* Use Pathways `_transfer_to_sharding` for resharding in experimental reshard API. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/148
* Allow specifying a custom proxy job name in _ISCPathways by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/157

## [0.1.3] - 2025-10-08
* Update the github action for PyPI by @lukebaumann in https://github.com/AI-Hypercomputer/pathways-utils/pull/105
* Expose `is_pathways_backend_used` in `pathwaysutils`. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/107
* Treat additional error types as potential slice down issues. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/109
* Adding split_by_mesh_axis to experimental for use by a new experimental reshard. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/112
* Update pathways.experimental.reshard so that PyTrees with arrays that have different device sets can be resharded. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/113
* Update GitHub action unittest matrix by @lukebaumann in https://github.com/AI-Hypercomputer/pathways-utils/pull/111
* Handle new style PRNG keys in `reshard` and `CloudPathwaysArrayHandler`

## [0.1.2] - 2025-08-25
* Updates to Pathways orbax handler. In https://github.com/AI-Hypercomputer/pathways-utils/pull/81 https://github.com/AI-Hypercomputer/pathways-utils/pull/89 https://github.com/AI-Hypercomputer/pathways-utils/pull/90
* Improvements to pathwaysutils/elastic by @copybara-service[bot] inhttps://github.com/AI-Hypercomputer/pathways-utils/pull/72  https://github.com/AI-Hypercomputer/pathways-utils/pull/84 https://github.com/AI-Hypercomputer/pathways-utils/pull/85 https://github.com/AI-Hypercomputer/pathways-utils/pull/86 https://github.com/AI-Hypercomputer/pathways-utils/pull/99 https://github.com/AI-Hypercomputer/pathways-utils/pull/100 https://github.com/AI-Hypercomputer/pathways-utils/pull/102
* Handle JAX API interface compatibility by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/88 https://github.com/AI-Hypercomputer/pathways-utils/pull/93 https://github.com/AI-Hypercomputer/pathways-utils/pull/95
* Adding jax version to the github actions matrix by @lukebaumann in https://github.com/AI-Hypercomputer/pathways-utils/pull/94
* Added LRU cache and tests to pathwaysutils. by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/83
* Moving initialization logic into its own module by @copybara-service[bot] in https://github.com/AI-Hypercomputer/pathways-utils/pull/96

## [0.1.1] - 2025-04-25
* Port the `collect_profile` script from JAX to PathwaysUtils
* Remove support for legacy initialize
* Add collect_profile as a script of pathwaysutils
* Make CloudPathwaysArrayHandler compatible with async directory creation feature in orbax

## [0.1.0] - 2025-04-07
* Bump the JAX requirement to 0.5.1
* Introduce `pathwaysutils.initialize()` to remove relying on side-effects from `import pathwaysutils`. by @copybara-service in https://github.com/AI-Hypercomputer/pathways-utils/pull/47
* Add a test for proxy backend registration by @copybara-service in https://github.com/AI-Hypercomputer/pathways-utils/pull/55
* Adding debugging utilities by @copybara-service in https://github.com/AI-Hypercomputer/pathways-utils/pull/57
* Adding an elastic manager and reshard modules by @copybara-service in https://github.com/AI-Hypercomputer/pathways-utils/pull/58
* Update README for Colocated Python Sidecar. by @copybara-service in https://github.com/AI-Hypercomputer/pathways-utils/pull/60

## [0.0.8] - 2024-02-12
* Disabled JAX's compilation cache
* Updated Orbax handler to use bulk APIs
* Updates to support JAX 0.5.0

## [0.0.7] - 2024-10-23
* Updated `setup.py` to `pyproject.toml`
* Added this changelog
* Added unittests
* Prepared for PyPI release

## [0.0.6] - 2024-10-10
* Decreased logging severity for most logs
* Persistence enabled
* General argument type fixes

[Unreleased]: https://github.com/AI-Hypercomputer/pathways-utils/compare/v0.1.1...HEAD

