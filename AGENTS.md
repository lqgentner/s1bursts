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

SNAP is the reference implementation for the processing graphs, so its output cannot be checked against itself. Product goldens answer "did we change?"; `pytest -m validation` answers "are we right?", by comparing RTC output against [OPERA RTC-S1](https://hyp3-docs.asf.alaska.edu/guides/opera_rtc_product_guide/) — ISCE3-produced, burst-level on the same ESA burst IDs, same Copernicus GLO-30 DEM, so only the processing chain differs.

```bash
uv sync --group validation     # adds earthaccess (Earthdata client)
uv run --env-file .env pytest -m validation
```

Needs CDSE credentials, Earthdata credentials (`EOSDIS_USERNAME` / `EOSDIS_PASSWORD`), and SNAP. Each scene downloads bursts and runs GPT, so this takes a few minutes. The committed record is `tests/fixtures/opera_reference.json`; regenerate it with `tests/fixtures/regenerate_opera_reference.py` and review the diff, since every moved number is a change in what users receive.

[`tests/fixtures/REFERENCE.md`](tests/fixtures/REFERENCE.md) describes the reference product and two conditions the comparison depends on: the static layers must be compared against the acquisition they were derived from, and gamma-0 must be averaged in power before comparing.

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
