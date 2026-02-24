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

