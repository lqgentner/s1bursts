# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is pre-1.0 (`0.x`), minor releases may contain breaking changes.

## Unreleased

### Added

- **GeoTIFF metadata on every product.** Composites, QA layers and per-burst
  products now carry processing and interpretation metadata based on CEOS-ARD
  PFS SAR v1.3.1. Full conformance is not claimed.
- **SNAP 14 support.** Graphs select the SNAP 13 or 14 coregistered-band names
  from the installed version. An unknown version warns and uses SNAP 13 names.
- `CoherenceConfig.correct_coherence_bias` and `remove_local_phase_ramp`, both
  defaulting to `False`. Enabling either requires SNAP 14 or newer.
- **Python 3.14 support**, verified in CI.
- `prepare_dem_for_snap()` can convert compound-CRS orthometric DEMs to WGS 84
  ellipsoidal heights while applying SNAP's pixel-shift compensation. Set
  `download_missing_grids=True` to download required PROJ grids.
- Runtime version access through `s1bursts.__version__`.
- Runtime validation for search fields, SNAP graph settings, resampling methods,
  and burst-pair timing constraints.
- Named AWS profiles through `CDSE_S3_PROFILE` (`CDSES3Settings.profile`). A
  configured profile takes precedence over explicit CDSE S3 keys and ambient
  AWS environment credentials.
- Per-product composite codec profiles through
  `ResolutionWeighter.rasterio_profile`, with an optional `"default"` profile.
  Flat profiles retain their previous behavior.
- **Every composite and QA layer now carries exact band statistics**
  (`STATISTICS_MINIMUM`, `_MAXIMUM`, `_MEAN`, `_STDDEV`, `_VALID_PERCENT`),
  computed over the whole array and written at creation, so a downstream
  catalogue can read them without opening the raster. New
  `product_tags.band_statistics()`.
- `ResolutionWeighter(checksum=True)` records a SHA-256 multihash of every
  written file's exact bytes in the new `ResolutionWeighter.checksums`
  attribute, keyed by output path. The digest is multihash-encoded (`1220`
  followed by the hex) and taken at write time, where the encoded bytes are
  already in hand; computing it later means reading every object back.
  Defaults to `False`, so existing calls are unchanged.

### Fixed

- **SNAP terrain-correction masks are now consistent across SNAP 13 and 14.**
  Graphs use a finite internal no-data marker, preventing interpolation over
  radar-geometry fill pixels.
- **Areas without DEM coverage no longer read as zero.** Per-burst products
  convert all-band zero fill to no-data when `replace_zeros_with_nan` is enabled.
- `SNAPConfig` now warns when `TMPDIR` is set but unusable.
- `compose_*` rejects inputs that do not sit on the output grid.
- Coherence compositing rejects inputs with differing temporal baselines.
- Compositing warns when inputs report different SNAP versions.
- `prepare_dem_for_snap(download_missing_grids=True)` now refreshes the PROJ
  grid cache after downloading and attempts only installable open-license grids.
- The compensated-DEM cache now uses source content and conversion-version keys,
  preventing collisions between sources with the same basename.
- Extracted burst manifests preserve the annotation's coordinate precision.
- `BurstCollection` no longer writes through DataFrame views when sliced.
- Reading GCP-georeferenced SLC TIFFs no longer emits a spurious
  `NotGeoreferencedWarning`.
- `SNAPConfig` and `SNAPSettings` reject resource budgets that cannot run a job
  and suggest a valid configuration.

### Changed

- **Breaking:** `orbit_direction` is renamed to `pass_direction` throughout, and
  its values are now `"Ascending"` and `"Descending"` (instead of `"ASCENDING"`
  and `"DESCENDING"`). This aligns the wording with the
  [CEOS-ARD SAR metadata specification](https://ceos.org/ard/) v1.3.
- **Breaking:** The `QA-NUM` layer of an LRW composite now counts the
  observations that carry a non-zero resolution weight, rather than those
  holding a finite value, matching the CEOS-ARD *Contributing Observations
  Image*. Regenerate coherence QA products to update their counts. RTC results
  are unchanged, and unweighted pass composites still count finite values.
- **Breaking:** `compensate_dem_for_snap()` is renamed to
  `prepare_dem_for_snap()` to reflect that it can now apply both the pixel
  shift and vertical datum conversion of an external DEM for SNAP.
- The area SNAP writes as `simulatedImage` is now called the *local
  contributing area*, following CEOS-ARD terminology.
- Requires `freezebase[s3]>=0.7,<0.8` for product metadata, composite codec
  profiles, `write_cog(checksum=...)`, and context handling of burst downloads.
- **Breaking:** `GeocodingConfig` is now abstract and cannot be instantiated. It
  remains the base for shared Terrain-Correction parameters; concrete configs
  implement `to_xml()`.
- **Breaking:** `build_coherence_magnitude_graph()` is renamed to
  `build_coherence_graph()`.
- The first-party dependency `freezecore` was renamed to `freezebase` and is now
  resolved from PyPI. Imports move from `freezecore.*` to `freezebase.*`, and
  its cache environment variable is now `FREEZEBASE_CACHE`.

### Removed

- **Breaking:** the complex coherence product and everything that existed only to
  produce it: `CoherenceConfig.write_complex`, `CoherenceConfig.map_looks`,
  `CoherenceComplexProduct`, `FileManager.find_coherence_complex_products()`, and
  `build_coherence_complex_ifg_graph()`. Coherence creation now always returns a
  single-band `CoherenceProduct`.
- **Breaking:** `s1bursts.dimap` (`BeamDimapDataset`) and `s1bursts.utils`
  (`sliding_window_mean`), which served only complex coherence processing.
- **Breaking:** `build_terrain_correction_graph()`, and with it
  `GeocodingConfig.to_xml()`. Remaining products use single-shot graphs with
  Terrain-Correction built in.

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
