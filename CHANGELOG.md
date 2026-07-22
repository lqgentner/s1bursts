# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is pre-1.0 (`0.x`), minor releases may contain breaking changes.

## [Unreleased]

### Added

- `Burst.mode` property deriving the acquisition mode (`IW`/`EW`) from the
  sub-swath.
- `py.typed` marker so downstream type checkers use the inline annotations.
- Packaging metadata: project URLs, keywords, and Development-Status / Topic /
  Typing classifiers.
- GitHub Actions CI (lint, type-check, test matrix on 3.12/3.13, a lowest-bound
  dependency job, and a build + wheel smoke-test), a scheduled/manual CDSE +
  SNAP integration workflow, and a Trusted-Publishing release workflow.

### Changed

- Licensing metadata modernized to PEP 639 (`license = "MIT"` +
  `license-files`), and the sdist no longer ships developer scaffolding
  (`.vscode/`, `CLAUDE.md`, `AGENTS.md`, `uv.lock`, internal pipeline notes).
- `numpy` lower bound raised to `>=2.0` to match the freezecore floor, and
  `requests` given a `>=2.28` floor.

### Fixed

- **EW support**: `rel_orbit_number`, `orbit_direction`, `geometry`, and
  `full_burst_id` now route to the EW burst map for `EW1`–`EW5` bursts instead
  of raising `KeyError` against the IW-only table.
- **S3-only credentials**: `BurstDatabase.download(endpoint="s3")` no longer
  constructs a CDSE OData authentication handler up front, so downloads work
  with only `CDSE_S3_*` credentials configured. OData credentials are requested
  lazily and only when an OData request (primary or fallback) actually runs; a
  missing-credential failure is surfaced as a clear per-burst `BurstDownloadError`.
- Non-integer standard-grid origin now uses true division
  (`grid_spacing / 2`), so a 7.5 m grid yields a 3.75 m pixel-centre origin.
- Downloaded SAFE annotation/calibration/noise/RFI/manifest XML is parsed with a
  hardened parser (no entity resolution, no network access), matching the graph
  parser.
- Relative auxdata cache paths are resolved before symlinking into per-worker
  SNAP userdirs, avoiding broken symlinks.
