# s1bursts Development Setup

s1bursts is a Sentinel-1 SLC burst download and InSAR processing package, built on the CDSE APIs and ESA SNAP GPT. For domain context, product hierarchy, and pipeline design decisions, see [`SENTINEL1_LRW_PIPELINE.md`](SENTINEL1_LRW_PIPELINE.md).

Shared geo/IO helpers (MGRS grid, rasterio/S3 raster access, HTTP download tooling) come from the sibling [freezebase](https://github.com/lqgentner/freezebase) package — the only first-party dependency, so that s1bursts can be released on PyPI with a wheel-only dependency set (no GDAL, no scipy).

## Environment

The project uses the uv package manager. The virtual environment is located at `.venv/`. The dependencies are stored in the `pyproject.toml`. freezebase is resolved from PyPI like any other dependency.

To synchronize the environment:

```bash
uv sync
```

To run Python:

```bash
uv run python3 -c "print('helloworld')"
```

## Linting & type checking

```bash
# Auto-format code
uv run ruff format
# Apply automatable fixes
uv run ruff check --fix
# After fixing, run format again
uv run ruff format
# Run mypy
uv run dmypy run src/s1bursts
```

## Testing

```bash
uv run pytest                  # offline unit tests (default)
uv run pytest -m integration   # network/SNAP integration tests
uv run pytest -m "integration and snap"  # only the tests that invoke SNAP gpt
uv run pytest -m validation    # scientific cross-validation against OPERA RTC-S1
```

Unit tests are fully offline: network, CDSE credentials, the burst-map lookup tables, and SNAP GPT are mocked. Tests marked `integration` require CDSE credentials and/or a local SNAP install (`SNAP_BIN`) exported into the environment; they are deselected by default via `addopts`, and skip themselves when a prerequisite is missing so the suite degrades gracefully on a partially configured machine.

Set `S1BURSTS_REQUIRE_INTEGRATION=1` to turn those skips into failures. CI sets it, because every prerequisite is provided there and a run that silently skipped everything would look the same as a passing one.

Unexpected warnings fail the suite (`filterwarnings = ["error", ...]` in `pyproject.toml`). Add a narrow `ignore` with its reason and removal condition rather than widening the filter.

### Scientific validation

SNAP is the reference implementation for the processing graphs, so its output cannot be checked against itself. Product goldens answer "did we change?"; `pytest -m validation` answers "are we right?", by comparing against products of the same bursts from independent processors:

| Output | Reference | Producer |
| --- | --- | --- |
| RTC (`Gamma0`, `simulatedImage`) | [OPERA RTC-S1](https://hyp3-docs.asf.alaska.edu/guides/opera_rtc_product_guide/) | NASA/JPL, ISCE3 |
| Coherence (`coh`) | [HyP3 burst InSAR](https://hyp3-docs.asf.alaska.edu/guides/burst_insar_product_guide) | ASF, ISCE2 topsApp |

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

### External DEM handling

`prepare_dem_for_snap()` compensates for two reported SNAP bugs: an external DEM is sampled [half a pixel south-east](https://forum.step.esa.int/t/external-dem-is-sampled-half-a-pixel-off/46042) of where its georeferencing places it, and the only geoid SNAP knows is EGM96, which it [applies to the Copernicus DEM](https://forum.step.esa.int/t/copernicus-dem-built-in-path-applies-egm96-while-the-data-is-egm2008/46041) even though those heights are EGM2008. When either is fixed the compensation becomes the error, and nothing else would notice: the unit tests assert that the shift is applied, not that SNAP still needs it, and every other graph reads the built-in DEM rather than a file.

`TestExternalDEMSampling` therefore measures both against the installed `gpt`. Its probe ([`tests/dem_probe.py`](tests/dem_probe.py), which explains the method) is synthetic and takes seconds: three small DEMs and one target grid, sampled by `AddElevation` in a single graph, no burst and no download. It says nothing about the *built-in* Copernicus path, which would need the tiles themselves; the external-DEM geoid stands in for it, since both call the same `EarthGravitationalModel96`.

### The no-data marker in the graphs

`ReplaceZerosWithNoData` sits between Terrain-Flattening and Terrain-Correction and maps the SLC's zero fill onto a no-data marker, because TC otherwise averages those zeros into its interpolation kernel and biases edge pixels dark ([forum report](https://forum.step.esa.int/t/sentinel-1-slc-terrain-correction-operator-uses-zero-values-in-interpolation-instead-of-treating-them-as-no-data/45336)).

The marker is the finite `NODATA_SENTINEL` (`-2**40`), not `NaN`. TC recognises a no-data sample only through `v == tileData.noDataValue`, which is false for every `v` when the declared value is `NaN` — so SNAP 13 never reached its no-data path and let the `NaN` poison every output pixel whose kernel touched it, eroding the footprint, while SNAP 14 tests `Double.isNaN(v)` as well and did reach it. That was the whole of the 13-vs-14 mask difference (measured at 52,892 pixels on burst 30346). A finite marker is detected by both, so both fall back to the nearest valid sample.

Two constraints pick the value, `-2**40`. It has to lie outside every band's range: the binding case is the interferogram, whose `i_ifg` and `q_ifg` are signed and, because no operator before them calibrates, are products of raw CInt16 SLC counts bounded by `2 * 32768**2 = 2**31`. And it has to be exactly representable in float32, because SNAP compares the float32 sample widened to double against the double it parsed from the XML — `-1e12` and `-1e30` are *not*, so they would silently fail the same way `NaN` does. Powers of two written out in full satisfy both. Within a complex pair the two bands must also declare the same marker, since TC reads `srcBands[0].getNoDataValue()` and applies it to both.

`_normalize_nodata()` in `snap_client.py` converts the marker to `NaN` while the staging GeoTIFF is finalized, so `NaN`-as-no-data stays the product convention and the sentinel is an implementation detail of the graph. It converts the zero fill in the same pass — see below — which is why one flag, `replace_zeros_with_nan`, gates both.

Do not move the node earlier. Terrain-Flattening divides by the simulated area without testing its input against no-data, so a finite marker placed before it is scaled into a different number and stops matching the declaration; SNAP 13's Terrain-Flattening also writes zeros of its own in layover, after any earlier node has run.

### Zeros where the DEM has no coverage

Terrain-Correction skips a tile whose DEM is entirely no-data and fills it with the band's no-data value, but writes that fill into a copy of the tile buffer it never hands back (`Tile.getRawSamples` without the matching `setRawSamples`). The copy is only taken for tiles the image edge clips, so an edge tile over a DEM void reaches the product as exact zeros while interior tiles are correctly no-data. It is not specific to external DEMs — a coastal burst whose edge tile is all ocean does the same under the built-in Copernicus path — and SNAP 13 and 14 behave identically.

`_normalize_nodata()` in `snap_client.py` therefore also converts a pixel that is zero in *every* band into no-data. Requiring all bands is what keeps the interferogram's signed real and imaginary parts from being masked where one of them legitimately crosses zero; it is safe because `ReplaceZerosWithNoData` has already removed the radar-geometry zeros, so no valid pixel reaches the writer as zero. That is also why the conversion is gated on `replace_zeros_with_nan` — with the node off, zeros are data. Drop the workaround once SNAP writes its fill back.

### SNAP versions

SNAP 14 renamed the suffixes Back-Geocoding gives a coregistered stack's bands — `mst` became `ref`, `slv1` became `sec1`. The interferogram graph's intensity expressions name those bands, so a graph built for the wrong version is rejected with `Undefined symbol` before any pixel is read. Nothing else moved: Coherence and Interferogram name their outputs as before, and every operator the graphs use exists in both versions.

`SNAPConfig` therefore resolves the version beside the `gpt` path (`snap_version()` reads the installation's `VERSION.txt`) and passes it to `to_xml(snap_version=...)`, where `stack_band_suffixes()` picks the naming. An undetermined version logs a warning and uses the pre-14 naming, so existing SNAP 13 installs are unaffected — which is also why the committed golden graphs did not move. The keyword is on every config for a uniform call; `BackscatterConfig` does not vary with it.

SNAP 14 also added two `Coherence` parameters, exposed as `CoherenceConfig.correct_coherence_bias` and `remove_local_phase_ramp`. GPT rejects a parameter its operator does not know, so the graph carries them only for SNAP 14 and newer (`supports_coherence_options()`), and enabling either under an older or undetermined version raises `ConfigurationError`. Both default off but are still written, because SNAP's own default for `correctCoherenceBias` is on — hence the separate golden `S1_SLCtoCoh-snap14.xml`.

`SUPPORTED_SNAP_VERSIONS` in `snap_client.py` is the source of truth for which versions are tested, and `SNAPConfig` warns when it resolves one outside that set. The CI matrix and README badge state the same list, so `tests/test_supported_snap_versions.py` fails if the three drift apart.

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

## Code Style Guidelines

### Docstrings

- Use NumPy-style docstrings for all public functions, classes, and methods
- Do not add `Examples` or `Notes` sections
- Keep descriptions concise; do not add implementation details irrelevant to the end user

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

- Be clear and concise, avoid unnecessary jargon
- Keep comments to a single line wherever one suffices, use more lines only where the reasoning genuinely needs it
- Comment on the "why", not what the code already states
- Comment on the current, not the previous behavior

### Type checking

- Use Python 3.13+ type annotations (`type` instead of `TypeVar`, collection imports from `collections.abc` instead of `typing`, etc.)
- Keep ignores and casts to a minimum
- Only annotate variables that cannot be automatically inferred by mypy (such as function arguments, return values, and empty collections)

### Imports

- Place imports at the top of the file in the standard import section
