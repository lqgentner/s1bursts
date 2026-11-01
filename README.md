# s1bursts

[![CI](https://github.com/lqgentner/s1bursts/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/lqgentner/s1bursts/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/lqgentner/s1bursts/graph/badge.svg?token=V6PEU2U4DW)](https://codecov.io/gh/lqgentner/s1bursts)
[![PyPI](https://img.shields.io/pypi/v/s1bursts.svg)](https://pypi.org/project/s1bursts/)
[![Python versions](https://img.shields.io/pypi/pyversions/s1bursts.svg)](https://pypi.org/project/s1bursts/)
[![SNAP](https://img.shields.io/badge/SNAP-13%20%7C%2014-blue)](https://step.esa.int/main/download/snap-download/)
[![License](https://img.shields.io/pypi/l/s1bursts.svg)](https://github.com/lqgentner/s1bursts/blob/main/LICENSE)
[![pyrefly](https://img.shields.io/endpoint?url=https://pyrefly.org/badge.json)](https://github.com/facebook/pyrefly)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Sentinel-1 SLC **burst** download, SAR and InSAR processing, built on the
[Copernicus Data Space Ecosystem (CDSE)](https://dataspace.copernicus.eu/) and
[ESA SNAP](https://step.esa.int/main/toolboxes/snap/) GPT.

Working at the burst level — the smallest independently processable unit of a
Sentinel-1 IW/EW acquisition — keeps downloads small (~100 MB instead of ~4 GB
per SLC scene) and makes dense time-series processing of a fixed footprint
practical.

## Features

- **Search** the CDSE burst catalog by burst ID, footprint, time range,
  polarization, orbit, and more (`BurstCatalog`, `SearchQuery`)
- **Download** single bursts via the CDSE OData API or by extracting them
  directly from CDSE S3 object storage, with automatic endpoint fallback
- **Select** part of a search by time, area or field (`within`,
  `intersecting`, `filter`)
- **Pair** bursts into InSAR pairs with a chosen temporal baseline
  (`BurstCollection.find_pairs`)
- **Process** with dynamically generated SNAP GPT graphs:
  - Radiometrically terrain corrected (RTC) gamma-nought backscatter,
    including the local scattering area band for resolution weighting
  - Geometrically terrain corrected (GTC) beta-nought backscatter
  - Coherence and interferograms
- **Composite** bursts into 10 km MGRS grid squares with locally resolution weighted
  (LRW) or per-pass aggregation (`ResolutionWeighter`)
- Local filesystem or S3 storage for all products (`FileManager` + UPath)

## Quick start

```python
import pandas as pd
import pyproj
from s1bursts import BurstCatalog, BurstProcessor, FileManager, SearchQuery, TimeRange

fm = FileManager(base_dir="data")
catalog = BurstCatalog(file_manager=fm)
processor = BurstProcessor(file_manager=fm)

# Search all 2022 acquisitions of one burst footprint (Aletsch glacier).
# A TimeRange is inclusive calendar days; the same object later names the
# composites built from this search.
bursts = catalog.search(
    SearchQuery(
        time_range=TimeRange("2022-01-01", "2022-12-31"),
        burst_id=30346,
        sub_swath="IW1",
        polarization="VV",
    )
)
pairs = bursts.find_pairs(baseline=pd.Timedelta(days=12))

# Download and process. Products are geocoded to a WGS 84 / UTM zone, here
# the one of the burst; utm_zone_of(bursts[0]) would give the same.
catalog.download(bursts, max_workers=3).raise_on_failure()
crs = pyproj.CRS("EPSG:32632")
processor.create_backscatter(bursts, crs=crs)  # RTC; GTC with config=GTCConfig()
processor.create_coherences(pairs, crs=crs)
```

## API layout

`s1bursts` exports what a pipeline is *built* from — `FileManager`, the
`BurstCatalog` / `BurstProcessor` / `ResolutionWeighter` services, and the
configs they take (`SearchQuery`, `TimeRange`, `SNAPConfig`, `RTCConfig`,
`GTCConfig`, `CoherenceConfig`, `InterferogramConfig` and their bases
`BackscatterConfig` and `GeocodingConfig`), plus
`download_burst_map`, `prepare_dem_for_snap` and `utm_zone_of`.

What a pipeline *produces* is returned by those services and imported from the
module it belongs to:

| Module | Contents |
| --- | --- |
| `s1bursts.burst` | `Burst`, `BurstPair`, `BurstCollection`, `BurstPairCollection` |
| `s1bursts.product` | RTC/GTC, coherence, interferogram, and composite product types |
| `s1bursts.summary` | `OperationSummary`, `OkResult`, `ErrorResult` |
| `s1bursts.errors` | exception hierarchy, rooted at `S1BurstsError` |
| `s1bursts.query` | `Polarization`, `Mode`, `SubSwath`, `PassDirection` literal aliases |

## Configuration

Credentials and SNAP settings are read from the process environment. Loading a
`.env` file is left to you — call `dotenv.load_dotenv()` before the first
s1bursts call, or export the variables (e.g. `uv run --env-file .env`):

| Variable | Purpose |
| --- | --- |
| `CDSE_USERNAME`, `CDSE_PASSWORD` | CDSE account, required for OData burst downloads ([register](https://dataspace.copernicus.eu/)) |
| `CDSE_S3_ACCESS_KEY_ID`, `CDSE_S3_SECRET_ACCESS_KEY` | CDSE S3 keys, required for S3 burst extraction ([generate](https://eodata-s3keysmanager.dataspace.copernicus.eu/)) |
| `CDSE_S3_PROFILE` | Alternative to the key pair: a named AWS profile in `~/.aws/credentials` holding the same keys. Takes precedence when set, and keeps the secrets out of the environment |
| `SNAP_BIN` | Path to a SNAP installation's `bin` directory; its `gpt` executable is used for processing. Falls back to `gpt` on `PATH` if unset |
| `SNAP_MAX_WORKERS` | Parallel GPT processes (default: 1) |
| `SNAP_MAX_RAM_GB` | Total RAM budget across GPT processes in GB (default: 8) |
| `SNAP_THREADS_PER_JOB` | Threads per GPT process (default: CPU count divided by `SNAP_MAX_WORKERS`) |
| `SNAP_CACHE_RATIO` | Fraction of each GPT process's heap used as tile cache (default: 0.7) |
| `SNAP_TIMEOUT` | Seconds one GPT process may run before it is stopped and its item recorded as failed; `none` for no limit (default: 3600) |
| `SNAP_AUXDATA_PATH` | Shared DEM/orbit cache (default: `~/.snap/auxdata`) |
| `SNAP_DEM_CACHE` | Where `prepare_dem_for_snap()` caches SNAP-ready external DEMs (default: `~/.s1bursts/dem`) |

Searching the burst catalog requires no credentials.

### External DEM preparation

SNAP expects external DEM heights relative to the WGS 84 ellipsoid and interprets
their geotransform differently from GDAL. Prepare an orthometric DEM once before
passing it as `externalDEMFile`:

```python
from pathlib import Path

from s1bursts import prepare_dem_for_snap

dem = prepare_dem_for_snap(
    Path("Copernicus_GLO30.tif"),
    source_crs="EPSG:4326+3855",  # WGS 84 horizontal + EGM2008 height
)
```

If the required grid is not installed, opt into downloading from PROJ's
open-license grid catalog with `download_missing_grids=True`. Some grids are not
available there and must be installed manually. Omit `source_crs` when the
elevations are already ellipsoidal. Use the returned path only with SNAP and
leave `external_dem_apply_egm=False`.

### Burst-ID map (optional)

Relative orbit number, pass direction, and footprint geometry are propagated
directly from the CDSE search response, and products record their relative
orbit and pass direction in their file names, so the normal search → download
→ process → composite workflow needs no extra data.

These attributes are only looked up in the [SAR-MPC Sentinel-1 burst-ID
map](https://sar-mpc.eu/sar-mpc-products/) for a burst that was **not** obtained
from a search, such as a hand-constructed `Burst` or one found with
`FileManager.find(Burst)`, whose SLC file name records the relative orbit
but not the pass direction. Download it
(~500 MB per acquisition mode) once when you need it:

```python
from s1bursts import download_burst_map

download_burst_map(mode="IW")
```

## Installation

```bash
uv add s1bursts
# or
pip install s1bursts
```

Processing needs a local [ESA SNAP](https://step.esa.int/main/download/snap-download/)
installation (version 13 or 14); searching and downloading do not.

For the domain background (product hierarchy, LRW compositing design), see
[`docs/SENTINEL1_LRW_PIPELINE.md`](https://github.com/lqgentner/s1bursts/blob/main/docs/SENTINEL1_LRW_PIPELINE.md).

## Development

```bash
uv sync                # set up environment
uv run pytest          # offline unit tests
uv run pytest -m integration  # network/SNAP tests (needs credentials + SNAP)
uv run ruff format && uv run ruff check --fix
uv run pyrefly check
```

## License

[MIT](https://github.com/lqgentner/s1bursts/blob/main/LICENSE)
