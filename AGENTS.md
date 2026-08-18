# s1bursts Development Setup

s1bursts is a Sentinel-1 SLC burst download and InSAR processing package, built on the CDSE APIs and ESA SNAP GPT. For domain context, product hierarchy, and pipeline design decisions, see [`SENTINEL1_LRW_PIPELINE.md`](SENTINEL1_LRW_PIPELINE.md).

Shared geo/IO helpers (MGRS grid, rasterio/S3 raster access, HTTP download tooling) come from the sibling [freezebase](https://github.com/lqgentner/freezebase) package — the only first-party dependency, so that s1bursts can be released on PyPI with a wheel-only dependency set (no GDAL, no scipy).

## Environment

The project uses the uv package manager. The virtual environment is located at `.venv/`. The dependencies are stored in the `pyproject.toml`. freezebase is resolved from PyPI like any other dependency.

To synchronize the environment:

```bash
uv sync --all-extras
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

### SNAP in CI

The scheduled integration workflow runs inside `ghcr.io/lqgentner/s1bursts-snap`, built from [`containers/Dockerfile.snap`](containers/Dockerfile.snap) by `.github/workflows/container.yml`. Installing SNAP per run is slow, and the Actions cache expires after seven days — which the weekly schedule sits right on top of — so the toolchain is baked into a pinned image instead.

To bump SNAP: change `ARG SNAP_VERSION` in the Dockerfile, run the container workflow, then update the pinned tag in `integration.yml`.

To reproduce a CI run locally:

```bash
docker build -f containers/Dockerfile.snap -t s1bursts-snap:local .
docker run --rm -v "$PWD:/w" -w /w --env-file .env \
    -e SNAP_BIN=/opt/esa-snap/bin \
    -e UV_PROJECT_ENVIRONMENT=/opt/venv \
    -e S1BURSTS_REQUIRE_INTEGRATION=1 \
    s1bursts-snap:local \
    sh -c 'uv sync --all-extras && uv run pytest -v -m integration'
```

`SNAP_BIN` is overridden because `.env` points at a host install, and `UV_PROJECT_ENVIRONMENT` keeps the container from overwriting the host `.venv` on the bind mount.

## Code Style Guidelines

### Docstrings

- Use NumPy-style docstrings for all public functions, classes, and methods
- Do not add `Examples` or `Notes` sections
- Keep descriptions concise; do not add implementation details irrelevant to the end user

### Type checking

- Use Python 3.13+ type annotations (`type` instead of `TypeVar`, collection imports from `collections.abc` instead of `typing`, etc.)
- Keep ignores and casts to a minimum
- Only annotate variables that cannot be automatically inferred by mypy (such as function arguments, return values, and empty collections)

### Imports

- Place imports at the top of the file in the standard import section
