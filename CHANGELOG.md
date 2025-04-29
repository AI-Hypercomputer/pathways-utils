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

