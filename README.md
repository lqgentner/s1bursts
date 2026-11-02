<div align="center">

<img src="https://raw.githubusercontent.com/lqgentner/s1bursts/main/assets/s1burst-logo.svg" alt="s1bursts" width="360">

*Sentinel-1 download, processing and compositing made easy.*

[![CI](https://github.com/lqgentner/s1bursts/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/lqgentner/s1bursts/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/lqgentner/s1bursts/graph/badge.svg?token=V6PEU2U4DW)](https://codecov.io/gh/lqgentner/s1bursts)
[![PyPI](https://img.shields.io/pypi/v/s1bursts.svg)](https://pypi.org/project/s1bursts/)
[![Python versions](https://img.shields.io/pypi/pyversions/s1bursts.svg)](https://pypi.org/project/s1bursts/)
[![SNAP](https://img.shields.io/badge/SNAP-13%20%7C%2014-blue)](https://step.esa.int/main/download/snap-download/)
[![License](https://img.shields.io/pypi/l/s1bursts.svg)](https://github.com/lqgentner/s1bursts/blob/main/LICENSE)
[![pyrefly](https://img.shields.io/endpoint?url=https://pyrefly.org/badge.json)](https://github.com/facebook/pyrefly)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

s1bursts makes Sentinel-1 synthetic aperture radar (SAR) data easy to use from
Python. It searches and downloads the data from the
[Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) (CDSE),
processes it with [ESA SNAP](https://step.esa.int/main/toolboxes/snap/) into
backscatter and coherence, and combines the results into composites on a
fixed grid.

It works with bursts, the smallest part of a Sentinel-1 scene that can be
processed on its own. A burst is about 100 MB instead of a 4 GB scene, so you
download and process only the area you are interested in. This makes dense
time series of a fixed area practical.

## Features

- **Search** the CDSE catalog by area, time window, polarization, orbit and more
- **Download** single bursts, with automatic fallback between CDSE's burst
  service and its S3 storage
- **Process** to terrain-corrected backscatter and interferometric coherence,
  with SNAP processing graphs built from Python arguments
- **Composite** many acquisitions onto the 10 km MGRS grid with local resolution
  weighting, which favors the acquisitions with the highest spatial
  resolution at each pixel
- **Store** everything on local disk or S3

## Installation

```bash
pip install s1bursts
```

Processing needs [ESA SNAP](https://step.esa.int/main/download/snap-download/)
13 or 14, and downloading a free [CDSE account](https://dataspace.copernicus.eu/).
The [installation guide](https://lqgentner.github.io/s1bursts/user-guide/installation.html)
covers both.

## Quick start

With your CDSE credentials in the environment, download and process two
acquisitions of one burst over the Aletsch glacier, 12 days apart:

```python
from s1bursts import BurstCatalog, BurstProcessor, FileManager, SearchQuery, TimeRange

fm = FileManager("data/quickstart")
catalog = BurstCatalog(fm)

bursts = catalog.search(
    SearchQuery(
        time_range=TimeRange("2022-08-01", "2022-08-13"),
        burst_id=30346,
        sub_swath="IW1",
        polarization="VV",
    )
)
catalog.download(bursts).raise_on_failure()

# "auto" geocodes each burst in its own UTM zone, here EPSG:32632.
processor = BurstProcessor(fm)
rtc = processor.create_backscatter(bursts, crs="auto").raise_on_failure()

pairs = bursts.find_pairs(baseline=12)
coherence = processor.create_coherences(pairs, crs="auto").raise_on_failure()
```

## Documentation

The [documentation](https://lqgentner.github.io/s1bursts/) has a user guide,
tutorials on real data, and the API reference.

## Development

```bash
uv sync                       # set up the environment
uv run pytest                 # offline unit tests
uv run pytest -m integration  # network and SNAP tests (needs credentials and SNAP)
uv run ruff format && uv run ruff check --fix
uv run pyrefly check
```

See the [contributing guide](https://lqgentner.github.io/s1bursts/user-guide/contributing.html)
for the integration tests and how to build the documentation.

## License

[MIT](https://github.com/lqgentner/s1bursts/blob/main/LICENSE)
