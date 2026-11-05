# s1bursts Development Setup

s1bursts is a Sentinel-1 SLC burst download and InSAR processing package, built on the CDSE APIs and ESA SNAP GPT. The [user guide](user_guide/) describes the product model and the API from a user's side; the sections below hold the reasoning behind it that users do not need.

Shared geo/IO helpers (MGRS grid, rasterio/S3 raster access, HTTP download tooling) come from the sibling [freezebase](https://github.com/lqgentner/freezebase) package — the only first-party dependency, so that s1bursts can be released on PyPI with a wheel-only dependency set (no GDAL, no scipy).

Treat signatures, return types, exceptions, import paths, and documented behavior as public
API. Preserve compatibility unless a change is explicitly breaking.

## Environment

The project uses uv and supports Python 3.12–3.15. Dependencies are declared in
`pyproject.toml`, and the virtual environment is located at `.venv/`. freezebase is resolved
from PyPI like any other dependency. `.python-version` pins the interpreter of `.venv/`; CI
jobs that test a specific version pass `--python`, which takes precedence over it.

```bash
uv sync
```

The `Programming Language :: Python` classifiers are the source of truth for the supported
versions (PyPI and the README badge are built from them). `tests/test_supported_python_versions.py`
fails if the CI test matrix, `requires-python`, the lowest-bound job, Pyrefly's `python-version`,
the SNAP image's `PYTHON_VERSION` or `.python-version` drift from them, so change them together.

## Required checks

```bash
uv run pytest
uv run ruff format
uv run ruff check --fix
uv run ruff format
uv run pyrefly check
```

After every code change, run tests, Ruff formatting and linting, and Pyrefly. Rerun affected
checks after fixes. Add or update tests for behavior changes; bug fixes should include a
regression test.

```bash
uv run pytest -m integration   # network/SNAP integration tests
uv run pytest -m "integration and snap"  # only the tests that invoke SNAP gpt
uv run pytest -m validation    # scientific cross-validation against OPERA RTC-S1
```

Unit tests are fully offline: network, CDSE credentials, the burst-map lookup tables, and SNAP GPT are mocked. Tests marked `integration` require CDSE credentials and/or a local SNAP install (`SNAP_BIN`) exported into the environment; they are deselected by default via `addopts`, and skip themselves when a prerequisite is missing so the suite degrades gracefully on a partially configured machine.

Set `S1BURSTS_REQUIRE_INTEGRATION=1` to turn those skips into failures. CI sets it, because every prerequisite is provided there and a run that silently skipped everything would look the same as a passing one.

Unexpected warnings fail the suite (`filterwarnings = ["error", ...]` in `pyproject.toml`). Add a narrow `ignore` with its reason and removal condition rather than widening the filter.

### Coverage

PR patch coverage and the project-wide coverage must both be at least 90%. Check before
opening a PR:

```bash
uv run pytest -q --cov=s1bursts --cov-branch --cov-report=term-missing
```

### Dependency lower bounds

Every dependency's lower bound is the oldest release that installs and works on Python 3.12.
When using a newer dependency API, raise its lower bound with a justification and run:

```bash
uv sync --resolution lowest-direct --python 3.12
uv run --no-sync pytest -q
```

Keep `--no-sync`; otherwise `uv run` re-resolves the environment at the latest versions.

### Scientific validation

SNAP is the reference implementation for the processing graphs, so its output cannot be checked against itself. Product goldens answer "did we change?"; `pytest -m validation` answers "are we right?", by comparing against products of the same bursts from independent processors:

| Output | Reference | Producer |
| --- | --- | --- |
| RTC (`gamma0`, `scattering_area`) | [OPERA RTC-S1](https://hyp3-docs.asf.alaska.edu/guides/opera_rtc_product_guide/) | NASA/JPL, ISCE3 |
| Coherence (`coherence`) | [HyP3 burst InSAR](https://hyp3-docs.asf.alaska.edu/guides/burst_insar_product_guide) | ASF, ISCE2 topsApp |

Both are burst-level on the same ESA burst IDs and use the same Copernicus GLO-30 DEM, so the processing chain is the substantial difference.

```bash
uv sync --group validation     # adds asf-search and hyp3-sdk
uv run --env-file .env pytest -m validation
```

Needs CDSE credentials, Earthdata credentials (`EOSDIS_USERNAME` / `EOSDIS_PASSWORD`, which are also the HyP3 login), and SNAP. Each scene downloads bursts and runs GPT, so this takes a few minutes. The committed records are `tests/fixtures/opera_reference.json` and `tests/fixtures/insar_reference.json`; regenerate them with the `regenerate_*_reference.py` scripts beside them and review the diff, since every moved number is a change in what users receive.

HyP3 processes on demand, so its half has to be ordered first — one job per scene, a quarter of an hour each, paid for in HyP3 credits:

```bash
uv run --env-file .env python tests/fixtures/regenerate_insar_reference.py --submit
```

Without `--submit` the script uses jobs already completed under the account and stops if there are none; `pytest -m validation` does the same and skips instead, unless `S1BURSTS_HYP3_SUBMIT=1` says it may order them itself. Products expire about two weeks after processing and then have to be ordered again — which is why `S1BURSTS_REQUIRE_VALIDATION=1` exists, turning any skipped validation test into a failure the same way `S1BURSTS_REQUIRE_INTEGRATION` does, so that a release check cannot pass on a campaign that never ran.

[`tests/fixtures/REFERENCE.md`](tests/fixtures/REFERENCE.md) describes both reference products and the conditions each comparison depends on: for OPERA, that the static layers are compared against the acquisition they were derived from and that gamma-0 is averaged in power; for HyP3, that its coherence is a different estimator (phase-sigma, measured on Goldstein-filtered phase, which saturates near 1 over coherent ground) and that it must be ordered at 5 x 1 looks, the setting whose estimation window matches this package's. Those are why the HyP3 comparison asserts ordering, a bounded signed offset and geolocation rather than equality of values, and why each scene declares whether its reference carries a pattern: the claims are selected from that declaration when tests are collected, so every claim a scene carries is asserted rather than skipped.

SNAP output is bit-reproducible run to run, and does not depend on `-q` or `-Xmx`, so a product golden can be a checksum and a CI container with fewer CPUs than a dev machine cannot move it. It is specific to one SNAP version, though (see [SNAP versions](#snap-versions)). s1ard is built on SNAP, so comparing against it is circular and validates nothing.

### External DEM handling

`prepare_dem_for_snap()` compensates for two reported SNAP bugs: an external DEM is sampled [half a pixel south-east](https://forum.step.esa.int/t/external-dem-is-sampled-half-a-pixel-off/46042) of where its georeferencing places it, and the only geoid SNAP knows is EGM96, which it [applies to the Copernicus DEM](https://forum.step.esa.int/t/copernicus-dem-built-in-path-applies-egm96-while-the-data-is-egm2008/46041) even though those heights are EGM2008. When either is fixed the compensation becomes the error, and nothing else would notice: the unit tests assert what the shift does for a given version, not that the version is still the right one, and every other graph reads the built-in DEM rather than a file.

The half-pixel bug is fixed in the `snap-dem` module from 14.0.2 on ([SNAP-4264](https://github.com/senbox-org/snap-engine/pull/640), cherry-picked to `14.x` on 2 October 2026; the 14.0.1 tag predates it, and `13.x` has no fix). That module is updated separately from SNAP, so `VERSION.txt` still says 14.0.0 after the update. `external_dem_needs_half_pixel_shift()` therefore reads the version the module's jar declares in its manifest (`OpenIDE-Module-Specification-Version`), in the installation and in SNAP's userdir (where updates land when the installation is read-only). It reads the jar rather than NetBeans' `update_tracking` record because the conda package ships none. `prepare_dem_for_snap()` shifts only below `EXTERNAL_DEM_FIXED_IN`. An unreadable version assumes the bug, as the older releases have it. The shift is part of the cache identity.

`TestExternalDEMSampling` therefore measures both against the installed `gpt`. Its probe ([`tests/dem_probe.py`](tests/dem_probe.py), which explains the method) is synthetic and takes seconds: three small DEMs and one target grid, sampled by `AddElevation` in a single graph, no burst and no download. It says nothing about the *built-in* Copernicus path, which would need the tiles themselves; the external-DEM geoid stands in for it, since both call the same `EarthGravitationalModel96`.

### The no-data marker in the graphs

`ReplaceZerosWithNoData` sits between Terrain-Flattening and Terrain-Correction and maps the SLC's zero fill onto a no-data marker, because TC otherwise averages those zeros into its interpolation kernel and biases edge pixels dark ([forum report](https://forum.step.esa.int/t/sentinel-1-slc-terrain-correction-operator-uses-zero-values-in-interpolation-instead-of-treating-them-as-no-data/45336)).

The marker is the finite `NODATA_SENTINEL` (`-2**40`), not `NaN`. TC recognizes a no-data sample only through `v == tileData.noDataValue`, which is false for every `v` when the declared value is `NaN` — so SNAP 13 never reached its no-data path and let the `NaN` poison every output pixel whose kernel touched it, eroding the footprint, while SNAP 14 tests `Double.isNaN(v)` as well and did reach it. That was the whole of the 13-vs-14 mask difference (measured at 52,892 pixels on burst 30346). A finite marker is detected by both, so both fall back to the nearest valid sample.

Two constraints apply to the value. It has to lie outside every band's range: each band's `ReplaceZerosWithNoData` expression keeps only positive values (`x > 0 ? x : marker`), so any negative marker does, and `test_the_no_data_marker_cannot_match_a_valid_sample` asserts that form for every graph — a signed band cannot be added without revisiting the marker. And it has to be exactly representable in float32, because SNAP compares the float32 sample widened to double against the double it parsed from the XML — `-1e12` and `-1e30` are *not*, so they would silently fail the same way `NaN` does. Powers of two written out in full satisfy both. The magnitude, `2**40`, is left over from the removed interferogram, whose signed bands reached `2**31`; it was kept so the golden graphs did not move.

`_normalize_nodata()` in `snap_client.py` converts the marker to `NaN` while the staging GeoTIFF is finalized, so `NaN`-as-no-data stays the product convention and the sentinel is an implementation detail of the graph. It converts the zero fill in the same pass — see below — which is why one flag, `replace_zeros_with_nan`, gates both.

Do not move the node earlier. Terrain-Flattening divides by the simulated area without testing its input against no-data, so a finite marker placed before it is scaled into a different number and stops matching the declaration; SNAP 13's Terrain-Flattening also writes zeros of its own in layover, after any earlier node has run.

### Zeros where the DEM has no coverage

Terrain-Correction skips a tile whose DEM is entirely no-data and fills it with the band's no-data value, but writes that fill into a copy of the tile buffer it never hands back (`Tile.getRawSamples` without the matching `setRawSamples`). The copy is only taken for tiles the image edge clips, so an edge tile over a DEM void reaches the product as exact zeros while interior tiles are correctly no-data. It is not specific to external DEMs — a coastal burst whose edge tile is all ocean does the same under the built-in Copernicus path — and SNAP 13 and 14 behave identically.

`_normalize_nodata()` in `snap_client.py` therefore also converts a pixel that is zero in *every* band into no-data. This is safe because `ReplaceZerosWithNoData` has already removed the radar-geometry zeros, so no valid pixel reaches the writer as zero. For the current products, whose valid pixels are positive in every band, requiring *every* band masks the same pixels as requiring *any*; it is kept as the narrower rule, so that a band which can legitimately be zero never masks the others. That is also why the conversion is gated on `replace_zeros_with_nan` — with the node off, zeros are data. Drop the workaround once SNAP writes its fill back.

### SNAP versions

SNAP 14 added two `Coherence` parameters, exposed as `CoherenceConfig.correct_coherence_bias` and `remove_local_phase_ramp`. GPT rejects a parameter its operator does not know, so the graph carries them only for SNAP 14 and newer (`supports_coherence_options()`), and enabling either under an older or undetermined version raises `ConfigurationError`. Both default off but are still written, because SNAP's own default for `correctCoherenceBias` is on — hence the separate golden `S1_SLCtoCoh-snap14.xml`. Every operator the graphs use exists in both versions.

`SNAPConfig` therefore resolves the version beside the `gpt` path (`snap_version()` reads the installation's `VERSION.txt`) and passes it to `to_xml(snap_version=...)`. An undetermined version logs a warning and builds the pre-14 graph. The keyword is on every config for a uniform call; `BackscatterConfig` does not vary with it. The product tags follow the graph: `COHERENCE_BIAS_CORRECTED` and `LOCAL_PHASE_RAMP_REMOVED` are `false` under SNAP 13, which has neither option, and `N/A` under an undetermined version, whose graph omits them and so gets SNAP's own default. An undetermined version therefore matches any other, in the skip check and among a composite's inputs, and so do these two tags.

`squarePixel` is pinned to `false` and not exposed, like `grSquarePixel` on `Multilook`. Under SNAP 14 (microwave-toolbox commit 988f5d74), `true` makes `CoherenceOp.initialize()` replace `cohWinAz` with `round(cohWinRg × ground-range spacing / azimuth spacing)` (and, below 1, set it to 2 and recompute the range window), so the applied window would not be the requested one and the window tags would be wrong. SNAP 13 declares the parameter but never reads it.

SNAP 14 also renamed the suffixes Back-Geocoding gives a coregistered stack's bands — `mst` became `ref`, `slv1` became `sec1`. No graph names them, because `Coherence` names its own output, but a `BandMaths` expression over the stack would have to pick the naming by version or be rejected with `Undefined symbol`.

`SUPPORTED_SNAP_VERSIONS` in `snap_client.py` is the source of truth for which versions are tested, and `SNAPConfig` warns when it resolves one outside that set. The CI matrix and README badge state the same list, so `tests/test_supported_snap_versions.py` fails if the three drift apart.

SNAP 13 and 14 do not produce identical pixels, so never mix them in one time series; products record `PROCESSING_SOFTWARE_VERSION`, and neither the skip check nor a composite accepts two known versions. On one burst with a shared SLC and DEM cache the median gamma-0 difference is 1.1 % (0.05 dB, unbiased) and p99 is 9.5 %; coherence is 0.004 lower on average. Terrain-Correction is the changed operator, not Terrain-Flattening or calibration: SNAP 14 adds an undocumented bistatic residual correction to its range-Doppler geolocation that cannot be disabled, a sub-pixel azimuth shift (at most 0.05 px) that grows with range. Coherence has a second cause, `correctCoherenceBias`. Already ruled out: the DEM readers (`AddElevation` is bit-identical), the resampling kernels, and the near-zero floor SNAP 13's calibration applies (SNAP-4160), which never reaches the product. SNAP 14 also zero-fills the ragged last tile column of a *tiled-GeoTIFF source product* in `AddElevation`; this package only feeds SNAP SAFE zips, and a ragged tiled external DEM was verified to give bit-identical output, so do not pad the DEM writer.

### SNAP in CI

The scheduled integration workflow runs inside `ghcr.io/lqgentner/s1bursts-snap`, built from [`containers/Dockerfile.snap`](containers/Dockerfile.snap) by `.github/workflows/container.yml`. Installing SNAP per run is slow, and the Actions cache expires after seven days — which the weekly schedule sits right on top of — so the toolchain is baked into a pinned image instead.

More than one SNAP version is supported at a time, so the workflow runs a leg per version: the `legs` job holds the list, one image tag each. The first leg carries the whole suite; the rest run `-m "integration and snap"`, because a second full leg would spend an hour re-downloading the same bursts to prove the CDSE coverage the first leg already proved. `fail-fast` is off, so a break under one version still leaves you the other version's result. Each leg checks the image's `VERSION.txt` against the version its tag claims — the tag is otherwise the only thing asserting which SNAP actually ran.

Tests that invoke `gpt` carry the `snap` marker, applied by `requires_snap` in `tests/test_integration.py` together with the skip, so a new SNAP test cannot end up selected by only one of the two.

To add or bump a SNAP version: run the container workflow with `snap_version` (it defaults to the Dockerfile's `ARG SNAP_VERSION`, the version treated as primary — only that one moves the `latest` tag), then add or bump the image tag in the `legs` job. A dispatch of the integration workflow can name a single `snap_image` to exercise a freshly built image before its tag is pinned there.

That matrix answers whether the package *runs* under each version. It does not answer whether the versions agree numerically — the SNAP assertions check band count, CRS, nodata and finiteness, not values — nor whether the output is still correct, which is what `pytest -m validation` is for. Those bounds are toolchain-agnostic (`tests/test_opera_validation.py` asserts against constants, not against the committed record), so the validation suite can be run under any supported SNAP; the committed records stay the provenance of the SNAP version in their `_toolchain`.

To reproduce a CI run locally:

```bash
docker build -f containers/Dockerfile.snap -t s1bursts-snap:local .
docker run --rm -v "$PWD:/w" -w /w --env-file .env \
    -e SNAP_BIN=/opt/esa-snap/bin \
    -e UV_PROJECT_ENVIRONMENT=/opt/venv \
    -e S1BURSTS_REQUIRE_INTEGRATION=1 \
    s1bursts-snap:local \
    sh -c 'uv sync && uv run pytest -v -m integration'
```

`SNAP_BIN` is overridden because `.env` points at a host install, and `UV_PROJECT_ENVIRONMENT` keeps the container from overwriting the host `.venv` on the bind mount.

### Parity with CDSE's burst products

The burst extractor reproduces the products of CDSE's burst service (the raster is byte-identical on burst 30346). Two fields differ on purpose, so do not "fix" them: GCP rows follow the reference script's corrected behavior, since CDSE's are one whole burst too far (-9006 where ours are -4503) and `tests/safe_conformance.py` reports it as `KNOWN DEVIATION:`; and footprint coordinates are copied as the annotation's raw text, which is what CDSE ships, whereas the reference script's XPath `number()` loses a digit. The wide footprint, blanked manifest checksums and retained out-of-burst grid points are CDSE's behavior too. Before changing what `burst_extractor.py` outputs, run `uv run --env-file .env pytest -m integration -k ReferenceConformance` and diff against CDSE instead of reasoning from the SAFE spec.

### Terminology and band names

The area Terrain-Flattening divides beta-0 by is the **local scattering area** (CEOS-ARD SAR PFS wording; Small et al. say contributing area, OPERA the area normalization factor, SNAP `simulatedImage`), kept only where those are external identifiers. Each band has one snake_case name, used as the graph's target band, the GeoTIFF band description and the `LAYER_NAME` tag: RTC `gamma0` and `scattering_area`, GTC `beta0`, and `coherence`. CEOS-ARD PFS SAR v1.3.1 governs; its Composite Backscatter product covers the LRW output. Do not rename `QA-NUM` to match item 2.21 of that standard, because the token is embedded in output filenames and the `filemanager.py` discovery regexes. The `tests/fixtures/regenerate_*_reference.py` scripts read bands by description and silently fall back to band 1 on a miss, so a band rename must reach them.

### Design decisions

- **Minimal metadata in file names.** A name holds only the identifiers a glob needs; settings are GeoTIFF tags. One table, `_NAME_FORMATS` in `filemanager.py`, builds and parses every name, so `find()` and `get_product_path()` cannot disagree.
- **The scattering area travels with its backscatter**, as band 2 of every RTC GeoTIFF, never as a separate product. It is date-aligned with the acquisition by construction, and a coherence LRW composite takes it from the RTC product of the same burst and date.
- **LRW needs RTC.** Terrain effects must be corrected before weighting, so GTC products get pass composites only. Terrain flattening happens inside the SNAP graph; there is no post-processing step.
- **No Multilook in the coherence graph.** The estimation window is the only control over the looks; a multilook after the estimator would only average coherence magnitudes.
- **Composites are keyed by `mgrs_code`**, since they combine several footprints. Within one relative orbit and date, bursts are joined at the midline of each overlap (see [Burst overlaps](#burst-overlaps)), so seams are never averaged; the orbit-date observations are then aggregated in time.
- **The burst-ID map stays optional.** Relative orbit and pass direction come from the search response or the SLC file name, so the normal workflow never opens the map. Only a hand-built burst, and the footprint of a burst found with `find(Burst)`, need it.

### Metadata tags

`product_tags.py` builds every product's tags; its module docstring maps each tag to the CEOS-ARD metadata parameter it is named after.

- **Vocabulary, not conformance.** Tag names follow the CEOS-ARD SAR specification so a reader can look a term up; tags without a counterpart get plain descriptive names, as OPERA does. `METADATA_VOCABULARY`, identical on every product, is the only tag naming the specification, and it is a pointer, not a claim: no tag key may look like a CEOS-defined field (no `CEOS_ARD_` namespace). No conformance is assessed or claimed; items 2.2 (data mask) and 4.3 (geolocation accuracy) are known unmet Threshold requirements. `PRODUCT_FAMILY = Composite Coherence` is this package's own term, since coherence composites fall between [InSAR] 3.8 and [CB].
- **Provenance.** `DEM` and `PROCESSING_SOFTWARE(_VERSION)` are written per burst and inherited by composites; `PIPELINE_SOFTWARE(_VERSION, _URL)` is stamped by whichever s1bursts writes the file and never inherited, because the definition of `QA-NUM` has changed between releases. `PRODUCT_LEVEL` runs L1 → L2 (burst, as OPERA's RTC-S1) → L3 (composite).
- **The DEM is recorded by name**, the SNAP name or `GeocodingConfig.dem_name`, never by path: a path is machine-specific, so a campaign processed on two machines could not be composited. A config with an external DEM and no or a blank `dem_name` therefore fails validation.
- **Every setting is a tag.** Every field of `RTCConfig`, `GTCConfig` and `CoherenceConfig` is recorded under the tag `_SETTING_TAGS` names for it, so a file read on its own says how it was made and the skip check can compare it; `test_every_field_has_a_tag` fails for a field without one. Only `dem_name`, recorded through `DEM`, and the computed `standard_grid_origin` are exempt. The tags have plain descriptive names and cite no CEOS-ARD item. The settings of an external DEM are written only when `dem` is a path.
- **Every provenance field is always present**, as `N/A` (`product_tags.UNKNOWN`) when it cannot be determined, because absence is ambiguous with a file written before the tag existed. Conditional fields, such as the coherence window on a backscatter composite, are omitted instead.
- **Inherited fields must be unanimous.** `ResolutionWeighter._input_tags` reads every input and refuses inputs that differ on a setting tag, the DEM or a known SNAP version (see [Composite inputs](#composite-inputs)), so a composite carries every setting tag of its inputs, in every layer. Where inputs differ only by an undetermined SNAP version, the version and the tags that depend on it are `N/A`. Any other tag is carried forward only where all inputs agree. `COMPOSITING_PARAMETERS` is built from the arguments of the `compose_*` call, never from a configuration object.
- **Not emitted:** decibel-stored data (a composite still tagged `BACKSCATTER_SCALING` or `CQM_SCALING = decibel` counts as made with other settings), a conversion equation on coherence or `QA-NUM`, a data-mask description of `QA-NUM`, and composite or NRB wording on a per-burst product. Backscatter bands carry the unit `linear power` (the PFS wording), dimensionless bands none; the `BACKSCATTER_CONVENTION` tag stays `Power`, as the metadata specification enumerates it.
- **Backfilling tags.** Burst products are plain tiled GeoTIFFs and can be retagged in place, also on S3: `rasterio_open(path, "r+")` stages through `TMPDIR` and fails without a writable one, `rewrite_tiff()` holds the uncompressed raster in memory. Composites are COGs, which a metadata write would turn into plain GeoTIFFs, so regenerate them instead.

### Burst overlaps

Within one relative orbit and date, `_stitch_by_orbit_date` drops the two outermost pixels of every burst and takes each remaining pixel from the burst whose data edge is farthest from it, which puts the seam on the midline of every overlap. Both were measured on six Swiss tiles: a burst is degraded towards its own data edge in range and azimuth alike, by -2.4 to -8.8 dB in backscatter at the outermost pixel and -0.6 to -1.6 dB at the next (terrain correction resampling against no-data; `ReplaceZerosWithNoData` removes only exact zeros), with a tail of a few tenths of a dB over 10–30 px. Taking the lowest burst ID, as before, put one burst's faded edge into every composite of that orbit.

- **Depth is measured along grid rows and columns**, to the first and last finite pixel of each. Rows and columns are roughly range and azimuth, so this is the map-space analogue of SNAP's TOPSAR-Deburst (one azimuth line in the middle of the overlap) and TOPSAR-Merge (the slant-range midpoint per line), each of which cuts in one radar dimension. For the two parallel edges of an overlap strip, the distance along any fixed axis is proportional to the true distance, so the seam lands on the midline exactly; only the corners where range and azimuth overlaps meet are approximate. Euclidean distance would need a distance transform, which means scipy.
- **The footprint is the product's own finite pixels**, not the ESA burst polygon, which ignores the kilometre-scale terrain displacement of range edges in the Alps. Holes inside it are not edges, because only the first and last finite pixel of a row or column count; their borders are not trimmed either, since backscatter there changes gradually with the geometry towards shadow and layover rather than fading.
- **The edge is looked for 3 km beyond the cell** (`_EDGE_SEARCH_MARGIN_M`), because a cell often cuts an overlap. Depths are capped at the margin plus one pixel, which is also the depth of a run reaching the border of the read. In an overlap no wider than twice the margin, at least one of the bursts has its edge within the margin, so its exact depth ranks against the other's, capped or not; the rule is exact for overlaps up to 6 km. The widest measured on alps-total 2023 is 4.84 km (IW2/IW3, terrain-displaced); azimuth overlaps are about 1.8 km. Where two bursts are both capped, the tie-break decides and a warning is logged. A NaN gap that crosses the border of the read counts as an edge there, which can only matter in the same wide-overlap case. A 10 km cell reads about 2.6 times its area, and each loaded burst holds an int16 depth beside its float32 values.
- **Ties go to the lowest `(burst_id, sub_swath)`**, whatever the input order.
- **The scattering area follows the value's burst.** An RTC composite takes band 2 of the burst band 1 came from; a coherence composite takes it from the RTC product of the same burst and primary date, and refuses a burst some pixel comes from without one.
- **Recorded in `COMPOSITING_PARAMETERS`** (`burst_overlap=midline;burst_edge_trim_px=2`), so the skip check refuses a composite joined another way.

### Composite inputs

Differences that would make a composite wrong are refused: a CRS, pixel size or sub-pixel grid offset other than the cell's (the read window is computed in the cell's coordinates against each source's transform, so it would read the wrong pixels), a mixed DEM, mixed processing settings or SNAP versions (the composite records one of each), and a mixed polarization or coherence baseline (a composite carries one of each in its name) raise `InconsistentInputsError`. It is a `ValueError` but deliberately not a `CompositeError`: such inputs fail every cell they are given to, so a loop over cells must stop at them rather than record one failure per cell. A missing input, including an observation without a matching RTC scattering area, is a `CompositeInputError`, since it affects only the cells that observation reaches.

An undetermined SNAP version (`N/A`) matches any, as in the skip check, while an absent version tag is a difference. A coherence LRW composite compares the RTC products that supply the scattering area of one of its observations among themselves on their settings, which come from another config than the coherence's, and with the coherence inputs on the DEM and the SNAP version; RTC products that weight no observation are not compared. Pass composites compare the inputs of every direction they build with each other, before the first write, so neither direction is written when they differ. The composite of a direction skipped as existing counts as one more input: its measurement file's setting tags, DEM and SNAP version must agree too, under the same rules, so a call cannot add a direction that disagrees with the one already on disk. Its own burst inputs are not read, since they may have been deleted. When every direction exists, nothing is built and nothing is compared.

### Skipping existing products

With `overwrite=False`, `settings_check.py` compares an existing product's tags with the request and raises `ProductSettingsMismatchError` on a difference; it stops the batch rather than being recorded per item, because changed settings affect every product of a call. Per-burst products compare every setting tag (`_BURST_SETTING_KEYS`), with the requested values taken from the builders that write them, so formatting cannot differ, and `PROCESSING_SOFTWARE_VERSION` with the running SNAP version. Composites compare each layer's grid with the target geobox and `COMPOSITING_PARAMETERS`, but nothing inherited, so a skip reads no input.

A missing tag is a difference like any other: a file that does not record a compared setting, such as one written before s1bursts recorded it, raises rather than being skipped unchecked. Another known SNAP version raises like another setting, since SNAP 13 and 14 outputs differ. An undetermined version, recorded or running, is not compared, and neither are `COHERENCE_BIAS_CORRECTED` and `LOCAL_PHASE_RAMP_REMOVED`, which follow it; `SNAPConfig` already warns when it cannot determine the version or resolves an untested one. `PIPELINE_SOFTWARE_VERSION` is not compared. `BurstProcessor` checks all existing products before the first GPT run.

## Releasing

Versions come from git tags via `hatch-vcs`, so there is no version string to edit. Record
changes in [`CHANGELOG.md`](CHANGELOG.md) under `## Unreleased` as you go, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The project is pre-1.0, so minor releases may contain breaking changes, but they must be
labeled. Mark behavior changes with **Breaking:** and show the before/after so downstream
call sites can be updated from the entry alone.

New backward-compatible functionality is a **minor** bump; a pure bug fix with no new public
API is a **patch**.

The release PR moves the `## Unreleased` entries under the new version and date, and
bumps `version` and `date-released` in [`CITATION.cff`](CITATION.cff) to match;
`tests/test_citation.py` fails until both agree. Zenodo archives every published GitHub
Release under the concept DOI in `CITATION.cff` and reads its metadata from that file.

## Code Style Guidelines

### Docstrings

- Use NumPy-style docstrings for all public functions, classes, and methods.
- Avoid `Examples` and `Notes` sections; omit implementation details callers do not need.
- Private helpers, fixtures, and test doubles need only a one-line summary.

Docstrings are the source for the [Great Docs](https://posit-dev.github.io/great-docs/)
API reference, which renders them with `parser: numpy`. That parser passes
docstring bodies through to Quarto verbatim, so RST inline markup is never
interpreted.

A `References` section renders as a numbered list when each entry starts
`.. [N]`, and bare URLs in it are linked automatically. The renderer matches
those entries with a regex that spans a single HTML-free paragraph, so **any
markup inside an entry silently breaks the whole block**.

Because methods render onto their class's page, put the `References` section on
the class only and cite it from the methods as `[1]`, rather than repeating the
bibliography in each one.

### Comments

- Keep comments brief and explain why, not what.
- Describe current behavior; only regression-test comments may reference the former bug, e.g.
  `# Previously raised IndexError for a scalar geometry query.`

### Type checking

- Python 3.12+ annotations: `type X = ...` instead of `TypeVar`, `collections.abc` instead of
  `typing` for collection types, `X | None` instead of `Optional[X]`.
- Pyrefly runs under the `strict` preset on `src/s1bursts` and `tests`. Keep casts and
  `# pyrefly: ignore[code]` suppressions to a minimum, and give each one a comment
  saying why.
- Only annotate what Pyrefly cannot infer on its own (function arguments, return values, empty
  collections).

### Imports

- Place imports at the top of the file in the standard import section.
- Every module starts with `from __future__ import annotations`.
