# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is pre-1.0 (`0.x`), minor releases may contain breaking changes.

## [Unreleased]

### Added

- s1bursts can be cited: each release is archived on Zenodo under the DOI
  [10.5281/zenodo.23918246](https://doi.org/10.5281/zenodo.23918246), and
  `CITATION.cff` holds the citation metadata.

### Changed

- The package summary reads "Sentinel-1 download, processing and compositing
  in Python." instead of "... made easy."

### Fixed

- The warning about SLC bursts and products named without their pass
  direction no longer points to a rename recipe the changelog does not hold.
  It says that such products must be processed again, and that such SLC
  bursts are downloaded again.

## [0.8.0] - 2026-11-03

The first release on PyPI, and the first with a
[documentation site](https://lqgentner.github.io/s1bursts/). Every product now
records the settings it was made with, and s1bursts no longer reuses or
composites products made with other settings.

### Added

- `pip install s1bursts`. The
  [documentation site](https://lqgentner.github.io/s1bursts/) has a user
  guide, three tutorials on real data, and the API reference.
- Every product records every processing setting as a GeoTIFF tag, so a file
  says on its own how it was made. Composites carry the settings of their
  inputs.
- `crs="auto"` geocodes each burst in the UTM zone of its footprint
  (`Burst.utm_zone`), so one call processes bursts across zones. `crs` also
  accepts strings and EPSG codes, such as `"EPSG:32632"` or `32632`.
- `OperationSummary.raise_on_failure()` raises if any item of a batch failed:
  `catalog.download(bursts).raise_on_failure()`.
- Burst collections can be narrowed with `within()`, `starting_within()`,
  `intersecting()` and `filter()` instead of through `.df`.
- `FileManager.open(product, layer)` reads a product from local disk or S3.
- `SNAPConfig(timeout=...)` sets the time limit of one `gpt` run.

### Changed

- **Breaking:** with `overwrite=False`, an existing product is reused only if
  it records the requested settings, and otherwise raises
  `ProductSettingsMismatchError`. Composite inputs must agree on their
  settings and SNAP version, and otherwise raise `InconsistentInputsError`.
  Products written by 0.7 record fewer settings: rebuild them with
  `overwrite=True`.
- **Breaking:** composites default to `aggregation="mean"` instead of
  `"median"`.
- **Breaking:** SLC and burst product file names include the pass direction,
  as in `S1_RTC_066A_030346_IW1_VV_20220105_32N.tif`, so files with the old
  names are no longer found.
- **Breaking:** an external DEM needs a name: `RTCConfig(dem=path,
  dem_name="swissALTI3D 30 m")`.
- **Breaking:** `crs` must be a WGS 84 / UTM zone.
- **Breaking:** the API is reorganized:
  - `BurstDatabase` → `BurstCatalog`, in `s1bursts.catalog`
  - `BackscatterConfig(flatten_terrain=False)` → `GTCConfig()`, and the
    default → `RTCConfig()`
  - `fm.find_rtc_products(...)` and the other `find_*` methods →
    `fm.find(RTCProduct, ...)`
  - `comp, (qa_num, qa_cqm) = compose_rtc_lrw(..., write_qa=True)` →
    `comp = compose_rtc_lrw(..., write_qa=True)`, with the QA layers as
    `fm.get_product_path(comp, Layer.QA_CQM)`
  - `SearchQuery(time_range=("2022-01-01", "2023-01-01"))` →
    `SearchQuery(time_range=TimeRange("2022-01-01", "2022-12-31"))`, and
    `search()` no longer takes `max_items`
  - `find_pairs(12, 6)` →
    `find_pairs(baseline=12, max_delta=pd.Timedelta(hours=6))`
  - `n_workers` → `max_workers`, `gpt_threads_per_job` → `threads_per_job`,
    `gpt_cache_ratio` → `cache_ratio`
  - errors derive from `S1BurstsError`, such as `CatalogError` for a failed
    search and `CompositeError` for a cell that cannot be composited
- **Breaking:** pass composites raise `EmptyCompositeError` for a cell that
  no input reaches, as LRW composites do, instead of writing all-NaN files.
- **Breaking:** requires freezebase 0.11.
- The default `SNAP_MAX_RAM_GB` is 8 GB instead of 4 GB.

### Removed

- **Breaking:** the interferogram product; order
  [HyP3 burst InSAR](https://hyp3-docs.asf.alaska.edu/guides/burst_insar_product_guide)
  products instead. Also decibel output of composites, and
  `CoherenceConfig.square_pixel`.

### Fixed

- **Affects results:** `QA-CQM` of median and quantile composites overstated
  the local resolution, by up to a factor of 4. It is now
  `1 / sum(W * A)` under every aggregation.
- **Affects results:** where bursts overlap, composites take the pixel from
  the same burst on every run.
- `SearchQuery(intersects=...)` rejects an empty geometry or one not in
  WGS 84, which silently found nothing.
- An interrupted download no longer leaves a truncated SLC that later runs
  skip. A burst that both endpoints fail to download reports both errors, and
  rejected credentials stop the download after one login attempt.
- A data error of one burst no longer stops the batch, while a full disk
  does, and Ctrl-C stops a batch promptly.

### Security

- A failed CDSE login no longer keeps the password in the error it records.

## [0.7.0] - 2026-10-26

### Added

- Beta-0 pass composites: `ResolutionWeighter.compose_gtc_by_pass()` averages
  GTC products per pass direction, as `compose_rtc_by_pass()` does for RTC
  (`S1_GTC_COMP_A_…` / `S1_GTC_COMP_D_…`, each with a QA layer).
- Python 3.15 support.

### Changed

- **Breaking:** requires freezebase 0.10.
- **Breaking:** an LRW composite for a cell that no burst covers with valid
  pixels raises `EmptyCompositeError` and writes nothing. Before, it wrote an
  all-NaN composite. To skip such cells, catch `EmptyCompositeError`.
- LRW composites count only bursts with valid pixels in the cell. A burst
  whose footprint merely touches the cell no longer adds to
  `CONTRIBUTING_OBSERVATIONS` or to the first and last acquisition dates.
- **Breaking:** failed downloads, SNAP runs and product writes are still
  recorded per burst, but any other error now stops the run, since it points
  to a bug. Checks for `isinstance(result.error, RuntimeError)` keep working.
- The burst-ID map cache is a third smaller (128 → 85 MB for IW). Existing
  caches stay valid.

### Fixed

- **Affects results:** with SNAP 14.0.2 or newer, `prepare_dem_for_snap()`
  shifted external DEMs by half a pixel (15 m for a 30 m DEM), because SNAP
  now corrects that offset itself. The shift is applied only where SNAP still
  needs it, so SNAP 13 results are unchanged. Prepared DEMs are rebuilt once;
  pass `snap_bin` if SNAP is not found through `SNAP_BIN` or `PATH`.
- When CDSE rejects a login token early, s1bursts logs in again instead of
  failing every later OData download.
- An OData download that returns something other than a file, such as a
  login page, is recorded as that burst's failure instead of stopping the
  batch.

## [0.6.0] - 2026-09-24

### Changed

- **Breaking:** `BurstCollection.find_pairs(12)` now selects 12 d ± 6 h
  (was ± 24 h). The old default mixed 12- and 13-day pairs that
  `ResolutionWeighter` refused to composite.
- **Breaking:** a `max_delta` of 12 hours or more, and a `baseline` that is
  not a whole number of days, now raise `ValueError`. Replace
  `find_pairs(12, max_delta=24)` with e.g. `find_pairs(12, max_delta=6)`.
  Coherence names round pair spacing to whole days; only a tolerance below
  12 hours guarantees the requested baseline.

### Fixed

- `find_pairs()` stopped widening the lag when every spacing exceeded the
  baseline, dropping in-tolerance pairs at later lags. It now stops only
  when every spacing exceeds `baseline + max_delta`.

## [0.5.1] - 2026-09-21

### Changed

- Requires freezebase 0.8 (`freezebase[s3]>=0.8,<0.9`).

### Fixed

- Products and composites written to S3 are uploaded through freezebase's
  s3fs path instead of GDAL's `/vsis3/`, which a gateway could refuse
  without raising an error. A backscatter or coherence product that fails to
  upload is now a failed operation: the summary counts it in `n_failed`, where
  it used to be logged as created.
- A burst `.zip` extracted to a remote `save_path` is uploaded with
  `upload_object`, which checks the stored size.

## [0.5.0] - 2026-09-07

### Added

- `TimeRange`, an inclusive window of calendar days. `SearchQuery` accepts it
  as `time_range` and searches every acquisition on any of its days, so the
  caller no longer translates an inclusive last day into an exclusive stop. A
  pair of instants is still accepted and is now read as `[start, stop)`, which
  differs from before only for an acquisition at exactly the stop instant.
- `time_range` on every `ResolutionWeighter.compose_*()` method: the same
  `TimeRange` the search was made with. The product is named and tagged with
  its days, so all tiles built from one search carry the same dates. The
  observed span moves to the new `FIRST_ACQUISITION_DATE` and
  `LAST_ACQUISITION_DATE` tags, which are always written. Each pass composite
  carries the span of its own direction. An input outside the window is
  refused. Without `time_range` nothing changes.

## [0.4.0] - 2026-09-07

### Added

- Exact band statistics (`STATISTICS_MINIMUM`, `_MAXIMUM`, `_MEAN`, `_STDDEV`,
  `_VALID_PERCENT`) on every composite and QA layer, written at creation. New
  `product_tags.band_statistics()`.
- `ResolutionWeighter(checksum=True)` records a multihash-encoded SHA-256 of
  every written file in `ResolutionWeighter.checksums`, keyed by output path.
- `BurstDatabase.search()` follows the catalog's next-link pages, so
  `max_items` above the 1000 items CDSE serves per request now works.
- Coherence and interferogram products carry a `LAYER_NAME` tag on every band.

### Fixed

- Per-burst coherence and interferogram products tagged `TEMPORAL_BASELINE_DAYS`
  with the truncated timestamp difference, so a pair whose secondary was
  acquired a fraction of a second earlier in the day was tagged 11 days under a
  `S1_COH12` name. The tag now uses the rounded baseline the filename uses.
- Missing CDSE S3 credentials aborted a whole download on the S3 endpoint
  instead of being recorded per burst, so the OData fallback never ran.
- A quantile `aggregation` outside `(0, 1)` was accepted and silently returned
  the minimum or maximum; every `compose_*` call now rejects it.
- `compose_*` rounded the read window to whole pixels, silently shifting an
  input whose pixel corners do not coincide with the target grid's; such an
  input is now refused like one in another CRS or at another pixel size.
- The HyP3 validation campaign no longer fails under Python 3.14 on an
  unclosed file left behind by `hyp3_sdk`'s download helper.

### Changed

- **Breaking:** `median` and quantile aggregation now exclude observations
  carrying zero resolution weight, which were previously interpolated into the
  result. `mean` is unaffected. Regenerate composites built with `median` (the
  default) or a quantile.
- **Breaking:** `QA-CQM` now applies the composite's own aggregation to the
  scattering area instead of always the weighted mean, and ignores
  observations that did not enter the composite. `mean` is unchanged.
  Regenerate `QA-CQM` products built with `median` or a quantile.
- **Breaking:** per-burst products name their bands in snake case, and the
  GeoTIFF band description, the graph's band and the `LAYER_NAME` band tag now
  agree: RTC `gamma0` and `scattering_area` (was `Gamma0`, `simulatedImage`),
  GTC `beta0` (was `Beta0`), coherence `coherence` (was `coh`), interferogram
  `ifg_real`, `ifg_imag`, `intensity_primary`, `intensity_secondary` (was
  `ifg_i`, `ifg_q`, `intensity_1`, `intensity_2`). Code reading bands by
  description must use the new names.
- The area SNAP writes as `simulatedImage` is called the *local scattering
  area* throughout, the CEOS-ARD term; `s1bursts` 0.3.0 and Small et al. use
  *local contributing area*, OPERA uses *area normalization factor*; all terms
  are the same quantity.
- Requires `freezebase[s3]>=0.7,<0.8` for `write_cog(checksum=...)`.

## [0.3.0] - 2026-08-31

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
- Requires `freezebase[s3]>=0.6.1,<0.7` for product metadata, composite codec
  profiles, and context handling of burst downloads.
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
  pixel toward the lower right. The helper writes a cached copy whose geotransform
  origin is moved onto the center of the first pixel; elevations pass through untouched.
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
  library's behavior until now), SNAP derives `nAzLooks` from `nRgLooks` and
  the local geometry and **silently discards the requested azimuth looks**, so a
  `(5, 1)` multilook was applied as `(5, 2)` for Sentinel-1 IW. Non-square pixels
  are correctly resampled to `grid_spacing` in geocoding (terrain correction).
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
  (`grid_spacing / 2`), so a 7.5 m grid yields a 3.75 m pixel-center origin.
- Downloaded SAFE annotation/calibration/noise/RFI/manifest XML is parsed with a
  hardened parser (no entity resolution, no network access), matching the graph
  parser.
- Relative auxdata cache paths are resolved before symlinking into per-worker
  SNAP userdirs, avoiding broken symlinks.
- `BurstFootprintMixin.rel_orbit_number`, `orbit_direction`, `geometry`, and
  `full_burst_id` document the `FileNotFoundError` they raise when the value was
  not propagated and the burst map has not been downloaded.

### Removed

- **Breaking:** removed `CDSES3Settings.s3_env`, a legacy context manager that
  injected `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` for an external script.
  S3 credentials now flow through the S3 `UPath` and `freezecore.s3.s3_env`.
