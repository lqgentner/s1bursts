# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is pre-1.0 (`0.x`), minor releases may contain breaking changes.

## Unreleased

### Added

- `prepare_dem_for_snap()` converts an orthometric external DEM to WGS 84
  ellipsoidal heights when given a compound source CRS (for example,
  `EPSG:4326+3855` for Copernicus GLO-30), while applying SNAP's half-pixel
  geotransform compensation in the same cached raster pass. PROJ grids can be
  downloaded explicitly with `download_missing_grids=True`.
- `s1bursts.__version__`, read from the installed distribution metadata. The
  version is still derived from git tags at build time by hatch-vcs; it is now
  also reachable at runtime.
- Runtime validation of public inputs: `SearchQuery` rejects unknown
  polarization, mode, sub-swath, orbit-direction and platform identifiers,
  requires positive burst IDs, and bounds the relative orbit number to 1-175;
  `GeocodingConfig.grid_spacing` and `BackscatterConfig.dem_oversampling` must be
  positive; the SNAP resampling options are restricted to the methods SNAP
  accepts (exported as `ResamplingMethod` and `DEMResamplingMethod`); and
  `find_pairs()` rejects a non-positive baseline or negative maximum deviation.
- `CDSE_S3_PROFILE` (`CDSES3Settings.profile`), a named AWS profile resolved from
  `~/.aws/credentials` as an alternative to `CDSE_S3_ACCESS_KEY_ID` /
  `CDSE_S3_SECRET_ACCESS_KEY`. The profile takes precedence when both are configured
  and removes boto's environment-variable credential provider, so ambient env vars
  can never sign CDSE requests. Requires freezebase ≥ 0.3.0.

### Changed

- **Breaking:** `compensate_dem_for_snap()` is renamed to
  `prepare_dem_for_snap()` to reflect that it can now apply both the pixel
  shift and vertical datum conversion of an external DEM for SNAP.
- Requires `freezebase[s3]>=0.4,<0.5`. Burst downloads now close their
  `HTTPDownloader` when done, releasing the connection immediately instead of
  leaving it to the garbage collector.
- **Breaking:** `GeocodingConfig` is now abstract and cannot be instantiated. It
  still carries the Terrain-Correction parameters every per-product config
  inherits, but `to_xml()` is now an abstract method implemented by
  `BackscatterConfig`, `CoherenceConfig`, and `InterferogramConfig`.
  `to_mermaid()` is unchanged and still inherited from the base.
- **Breaking:** `build_coherence_magnitude_graph()` is renamed to
  `build_coherence_graph()`. With the complex variant gone there is only one
  coherence graph, so the qualifier no longer distinguishes anything.

- The first-party dependency `freezecore` was renamed to `freezebase` and is now
  resolved from PyPI instead of a Git source. Imports
  moved from `freezecore.*` to `freezebase.*`; the cache environment variable it
  reads is now `FREEZEBASE_CACHE`.

### Fixed

- **The compensated-DEM cache could return the wrong raster.** The cache is now
  keyed on the source's SHA-256 content digest and a conversion schema version,
  with a verified JSON provenance sidecar per entry, so sources sharing a
  basename can no longer collide.
- **Footprint coordinates in the extracted burst manifest lost precision.** They
  were reformatted through Python's float repr, matching neither the reference
  extractor nor the burst products CDSE serves. The annotation's coordinate text
  is now copied verbatim, preserving every digit.
- `BurstCollection` no longer writes through a view of the DataFrame it is given.
  Slicing a collection triggered a pandas `SettingWithCopyWarning` and could
  modify the caller's frame.
- Reading an SLC measurement TIFF no longer emits a spurious
  `NotGeoreferencedWarning`. SLC products carry GCPs rather than a geotransform,
  so the warning said nothing the caller could act on.
- `SNAPConfig` and `SNAPSettings` now reject resource budgets that cannot
  describe a runnable job (`max_workers`, `gpt_threads_per_job` ≥ 1;
  `max_ram_gb` > 0; `gpt_cache_ratio` in (0, 1]), raising `ConfigurationError`
  with a working budget suggested instead of silently over-allocating heap.

### Removed

- **Breaking:** the complex coherence product and everything that existed only to
  produce it: `CoherenceConfig.write_complex`, `CoherenceConfig.map_looks`,
  `CoherenceComplexProduct`, `FileManager.find_coherence_complex_products()`, and
  the `build_coherence_complex_ifg_graph()` graph builder. It was an experiment
  that did not make the cut. `create_coherence()` / `create_coherences()` now
  always write the single-band `COH` magnitude, and their return types narrow
  from `CoherenceProduct | CoherenceComplexProduct` to `CoherenceProduct`.
- **Breaking:** `s1bursts.dimap` (`BeamDimapDataset`) and `s1bursts.utils`
  (`sliding_window_mean`). Both existed solely to serve the complex coherence
  post-processing step and have no remaining caller.
- **Breaking:** `build_terrain_correction_graph()`, and with it
  `GeocodingConfig.to_xml()`. The standalone Terrain-Correction graph existed
  only as the second stage of the complex coherence pipeline; every remaining
  product runs a single-shot graph with the Terrain-Correction tail built in.

## [0.2.0] - 2026-07-31

### Added

- `compensate_dem_for_snap()`, which makes an external DEM sample at the position its
  georeferencing states. SNAP reads a DEM passed as `externalDEMFile` half a DEM
  pixel towards the lower right. The helper writes a cached copy whose geotransform
  origin is moved onto the centre of the first pixel; elevations pass through untouched.
  Verified against SNAP 13.0.0.
- `SNAP_DEM_CACHE` setting (`SNAPSettings.dem_cache`, default `~/.s1bursts/dem`),
  the directory `compensate_dem_for_snap()` caches converted DEMs in.

### Changed

- **Breaking:** the settings classes no longer read a `.env` file; they read the
  process environment only. To parse an `.env` file, call `dotenv.load_dotenv()`
  or export the variables (`uv run --env-file .env`). `get_settings()` re-reads
  the environment on every call, so a late `load_dotenv()` is picked up.

## [0.1.1] - 2026-07-31

### Fixed

- Fixed SNAP memory leakage: `gpt` now runs with `java.io.tmpdir` pointed inside
  the per-job SNAP userdir. SNAP's jblas extracts 30 MB of native libraries per process
  and deletes them from a shutdown hook that `SIGKILL` skips. A run whose workers
  were OOM-killed therefore leaked ~30 MB each until `/tmp` filled. Cleanup now belongs
  to `snap_worker_userdir`, which removes the tree however `gpt` exited.

## [0.1.0] - 2026-07-27

### Added

- `Burst.mode` property deriving the acquisition mode (`IW`/`EW`) from the
  sub-swath.
- `download_burst_map()` to explicitly fetch the SAR-MPC burst-ID map, verified against a pinned SHA-256.
- `CoherenceConfig.map_looks`, an optional coherent box average applied to the
  *geocoded* complex coherence before the magnitude is taken. Requires
  `write_complex=True`; the product written is a single-band `COH`, so the
  compositing stage is unaffected. Averaging on the map grid gives a uniform
  ground footprint regardless of local incidence, and averaging coherently
  rather than in magnitude is what reduces the small-window estimator bias.
- `cqm_decibel` on `ResolutionWeighter.compose_rtc_lrw()` and
  `compose_coherence_lrw()` writes the QA-CQM map in decibels, consistent with
  Small et al. (2021,
  [doi:10.1109/TGRS.2021.3055562](https://doi.org/10.1109/TGRS.2021.3055562)).

### Changed

- **Breaking:** SNAP's `grSquarePixel` is now pinned off on the `Multilook`
  operator and is not exposed. With it enabled (SNAP's own default, and this
  library's behaviour until now), SNAP derives `nAzLooks` from `nRgLooks` and
  the local geometry and **silently discards the requested azimuth looks**, so a
  `(5, 1)` multilook was applied as `(5, 2)` for Sentinel-1 IW. Non-square pixels
  are a correctly resampled to `grid_spacing` in geocoding (terrain correction).
- **Breaking:** the graph configs now reject unknown fields (`extra="forbid"`)
  instead of ignoring them.
- **Breaking:** the coherence graphs no longer contain a `Multilook` node and
  `CoherenceConfig.multilook` is removed. Multilooking saved no runtime there —
  it can only sit after the coregistration and interferogram stages that
  dominate (162 s against 165 s without it) — and it had no single sensible
  placement: the magnitude graph applied it *after* `Coherence`, averaging
  coherence magnitudes, while the complex graph applied it *before* the
  estimator, enlarging it, so one name meant two different operations. Use
  `window` to set the looks and `map_looks` to average further after terrain
  correction. `BackscatterConfig.multilook` is unaffected: there the step
  precedes terrain flattening and cuts runtime roughly ninefold.
- All four graph configs now document every parameter.
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
