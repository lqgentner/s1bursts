# s1bursts

Sentinel-1 SLC **burst** download, SAR and InSAR processing, built on the
[Copernicus Data Space Ecosystem (CDSE)](https://dataspace.copernicus.eu/) and
[ESA SNAP](https://step.esa.int/main/toolboxes/snap/) GPT.

Working at the burst level — the smallest independently processable unit of a
Sentinel-1 IW/EW acquisition — keeps downloads small (~100 MB instead of ~4 GB
per SLC scene) and makes dense time-series processing of a fixed footprint
practical.

## Features

- **Search** the CDSE burst catalog by burst ID, footprint, time range,
  polarization, orbit, and more (`BurstDatabase`, `SearchQuery`)
- **Download** single bursts via the CDSE OData API or by extracting them
  directly from CDSE S3 object storage, with automatic endpoint fallback
- **Pair** bursts into InSAR pairs with a chosen temporal baseline
  (`BurstCollection.find_pairs`)
- **Process** with dynamically generated SNAP GPT graphs:
  - Radiometrically terrain corrected (RTC) gamma-nought backscatter,
    including the simulated-area band for resolution weighting
  - Geometrically terrain corrected (GTC) beta-nought backscatter
  - Coherence (magnitude or complex) and interferograms
- **Composite** bursts into 110 km MGRS tiles with locally resolution weighted
  (LRW) or per-orbit-pass aggregation (`ResolutionWeighter`)
- Local filesystem or S3 storage for all products (`FileManager` + UPath)

## Quick start

```python
import pandas as pd
import pyproj
from s1bursts import BurstDatabase, BurstProcessor, FileManager, SearchQuery

fm = FileManager(base_dir="data")
database = BurstDatabase(file_manager=fm)
processor = BurstProcessor(file_manager=fm)

# Search all 2022 acquisitions of one burst footprint (Aletsch glacier)
bursts = database.search(
    SearchQuery(
        time_range=("2022-01-01", "2023-01-01"),
        burst_id=30346,
        sub_swath="IW1",
        polarization="VV",
    )
)
pairs = bursts.find_pairs(baseline=pd.Timedelta(days=12))

# Download and process
database.download(bursts, max_workers=3)
crs = pyproj.CRS("EPSG:32632")
processor.create_backscatter(bursts, crs=crs)
processor.create_coherences(pairs, crs=crs)
```

## API layout

`s1bursts` exports what a pipeline is *built* from — `FileManager`, the
`BurstDatabase` / `BurstProcessor` / `ResolutionWeighter` services, and the
configs they take (`SearchQuery`, `SNAPConfig`, `BackscatterConfig`,
`CoherenceConfig`, `InterferogramConfig`, `GeocodingConfig`), plus
`download_burst_map`.

What a pipeline *produces* is returned by those services and imported from the
module it belongs to:

| Module | Contents |
| --- | --- |
| `s1bursts.burst` | `Burst`, `BurstPair`, `BurstCollection`, `BurstPairCollection` |
| `s1bursts.product` | RTC/GTC, coherence, interferogram, and composite product types |
| `s1bursts.summary` | `OperationSummary`, `OkResult`, `ErrorResult` |
| `s1bursts.errors` | exception hierarchy, rooted at `S1BurstsError` |
| `s1bursts.query` | `Polarization`, `Mode`, `SubSwath`, `OrbitDirection` literal aliases |
| `s1bursts.dimap` | `BeamDimapDataset`, for reading SNAP BEAM-DIMAP bands directly |

## Configuration

Credentials and SNAP settings are read from environment variables (or a
`.env` file in the working directory):

| Variable | Purpose |
| --- | --- |
| `CDSE_USERNAME`, `CDSE_PASSWORD` | CDSE account, required for OData burst downloads ([register](https://dataspace.copernicus.eu/)) |
| `CDSE_S3_ACCESS_KEY_ID`, `CDSE_S3_SECRET_ACCESS_KEY` | CDSE S3 keys, required for S3 burst extraction ([generate](https://eodata-s3keysmanager.dataspace.copernicus.eu/)) |
| `SNAP_BIN` | Path to a SNAP installation's `bin` directory; its `gpt` executable is used for processing. Falls back to `gpt` on `PATH` if unset |
| `SNAP_MAX_WORKERS` | Parallel GPT processes (default: 1) |
| `SNAP_MAX_RAM_GB` | Total RAM budget across GPT processes in GB (default: 4) |
| `SNAP_AUXDATA_PATH` | Shared DEM/orbit cache (default: `~/.snap/auxdata`) |

Searching the burst catalog requires no credentials.

### Burst-ID map (optional)

Relative orbit number, orbit direction, and footprint geometry are propagated
directly from the CDSE search response and from SLC filenames, so the normal
search → download → process → composite workflow needs no extra data.

These attributes are only looked up in the [SAR-MPC Sentinel-1 burst-ID
map](https://sar-mpc.eu/sar-mpc-products/) for a burst that was **not** obtained
from a search or a filename (e.g. a hand-constructed `Burst`). Download it
(~500 MB per aquisition mode) once when you need it:

```pythons
from s1bursts import download_burst_map

download_burst_map(mode="IW")
```

## Installation

Not yet on PyPI; install from GitHub:

```bash
uv add "s1bursts @ git+ssh://git@github.com/lqgentner/s1bursts.git"
```

For the domain background (product hierarchy, LRW compositing design), see
[`SENTINEL1_LRW_PIPELINE.md`](SENTINEL1_LRW_PIPELINE.md).

## Development

```bash
uv sync --all-extras   # set up environment
uv run pytest          # offline unit tests
uv run pytest -m integration  # network/SNAP tests (needs credentials + SNAP)
uv run ruff format && uv run ruff check --fix
uv run dmypy run src/s1bursts
```

## License

MIT
