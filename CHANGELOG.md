# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is pre-1.0 (`0.x`), minor releases may contain breaking changes.

## [Unreleased]

### Added

- `Burst.mode` property deriving the acquisition mode (`IW`/`EW`) from the
  sub-swath.
- `download_burst_map()` to explicitly fetch the SAR-MPC burst-ID map, verified against a pinned SHA-256.
- `CoherenceConfig.multilook`, a `(n_rg_looks, n_az_looks)` multi-looking step
  after debursting, mirroring `BackscatterConfig.multilook` and defaulting to
  `(5, 1)`. Pass `None` to skip it.
- `cqm_decibel` on `ResolutionWeighter.compose_rtc_lrw()` and
  `compose_coherence_lrw()` writes the QA-CQM map in decibels, consistent with
  Small et al. (2021,
  [doi:10.1109/TGRS.2021.3055562](https://doi.org/10.1109/TGRS.2021.3055562)).

### Changed

- **Breaking:** the root namespace is narrowed to what a pipeline is *built*
  from: `FileManager`, `BurstDatabase`, `BurstProcessor`, `ResolutionWeighter`,
  `SearchQuery`, `SNAPConfig`, the four graph configs, and
  `download_burst_map()`. What a pipeline *produces* (bursts, products, summaries, ...)
  now imports from the module it belongs to. The typical search → download → process →
  composite flow needs none of them, so most code should need no changes.
- **Breaking:** `ProcessingConfig` is renamed to `SNAPConfig` and no longer
  carries `file_manager`; every remaining attribute is a SNAP parameter with a
  default. `BurstProcessor` and `SNAPClient` now take the `FileManager` as their
  first argument and the config as an optional second one, matching
  `ResolutionWeighter`: `BurstProcessor(file_manager)` replaces
  `BurstProcessor(ProcessingConfig(file_manager=fm))`.
- Burst-ID map is now opt-in. Relative orbit number, orbit
  direction, and footprint geometry are propagated from the CDSE search response
  and from SLC filenames onto `Burst`/products, so the search → download →
  process → composite path no longer downloads anything. The ~500 MB SAR-MPC map
  is consulted only for a burst lacking that metadata (e.g. hand-constructed),
  and is no longer fetched automatically — call `download_burst_map()` to opt in.
- `py.typed` marker so downstream type checkers use the inline annotations.
- Packaging metadata: project URLs, keywords, and Development-Status / Topic /
  Typing classifiers.
- GitHub Actions CI (lint, type-check, test matrix on 3.12/3.13, a lowest-bound
  dependency job, and a build + wheel smoke-test), a scheduled/manual CDSE +
  SNAP integration workflow, and a Trusted-Publishing release workflow.
- Licensing metadata modernized to PEP 639 (`license = "MIT"` +
  `license-files`).
- `numpy` lower bound raised to `>=2.0` to match the freezecore floor, and
  `requests` given a `>=2.28` floor.

### Fixed

- **EW support**: `rel_orbit_number`, `orbit_direction`, `geometry`, and
  `full_burst_id` now route to the EW burst map for `EW1`–`EW5` bursts instead
  of raising `KeyError` against the IW-only table.
- **S3-only credentials**: `BurstDatabase.download(endpoint="s3")` no longer
  constructs a CDSE OData authentication handler up front, so downloads work
  with only `CDSE_S3_*` credentials configured. OData credentials are requested
  lazily and only when an OData request (primary or fallback) actually runs.
- Non-integer standard-grid origin now uses true division
  (`grid_spacing / 2`), so a 7.5 m grid yields a 3.75 m pixel-centre origin.
- Downloaded SAFE annotation/calibration/noise/RFI/manifest XML is parsed with a
  hardened parser (no entity resolution, no network access), matching the graph
  parser.
- Relative auxdata cache paths are resolved before symlinking into per-worker
  SNAP userdirs, avoiding broken symlinks.
- `BurstFootprintMixin.rel_orbit_number`, `orbit_direction`, `geometry`, and
  `full_burst_id` document the `FileNotFoundError` they raise when the value was
  not propagated and the burst map has not been downloaded.

## Removed

- **Breaking:** removed `CDSES3Settings.s3_env`, a legacy context manager that
  injected `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` for an external script.
  S3 credentials now flow through the S3 `UPath` and `freezecore.s3.s3_env`.
