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

[Unreleased]: https://github.com/AI-Hypercomputer/pathways-utils/compare/v0.0.8...HEAD

