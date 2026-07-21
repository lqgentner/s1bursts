# s1bursts Development Setup

s1bursts is a Sentinel-1 SLC burst download and InSAR processing package, built on the CDSE APIs and ESA SNAP GPT. For domain context, product hierarchy, and pipeline design decisions, see [`SENTINEL1_LRW_PIPELINE.md`](SENTINEL1_LRW_PIPELINE.md).

The package was split out of the [deepfreezer](https://github.com/lqgentner/deepfreezer) repository (formerly `deepfreezer.sentinel1`). Shared geo/IO helpers (MGRS grid, rasterio/S3 raster access, HTTP download tooling) come from the sibling [freezecore](https://github.com/lqgentner/freezecore) package — the only first-party dependency, so that s1bursts can be released on PyPI with a wheel-only dependency set (no GDAL, no scipy).

## Environment

The project uses the uv package manager. The virtual environment is located at `.venv/`. The dependencies are stored in the `pyproject.toml`. freezecore is consumed from the sibling checkout via a `[tool.uv.sources]` path entry until it is published on PyPI.

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
```

Unit tests are fully offline: network, CDSE credentials, the burst-map lookup tables, and SNAP GPT are mocked. Tests marked `integration` require CDSE credentials in the environment (or a `.env` file) and/or a local SNAP install (`SNAP_GPT_PATH`); they are deselected by default via `addopts`.

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
