# Sentinel-1 Local Resolution Weighting (LRW) Pipeline

> **Maintenance note:** Update this document whenever a major rework of the `s1bursts` package is completed (e.g. new product classes, renamed fields, new pipeline stages, changed filename conventions).

## Overview

This pipeline processes Sentinel-1 SLC bursts to create composites over MGRS grid cells. Two composite types are produced:

- **LRW composites**: Locally resolution weighted (LRW) combination of multiple burst footprints; weights = inverse of simulated area (1/A)
- **Pass composites**: Unweighted temporal mean per orbit-pass direction (ascending / descending)

## Architecture

### Product Hierarchy

```bash
BaseProduct
├── GeocodedBurst (single burst, geocoded)
│   ├── BackscatterProduct (abstract)
│   │   ├── RTCProduct (gamma nought, terrain flattened; includes simulatedImage band)
│   │   └── GTCProduct (beta nought, geometry-corrected only)
│   └── CoregisteredProduct (two co-registered bursts)
│       ├── CoherenceProduct (magnitude)
│       ├── CoherenceComplexProduct (i_coh + q_coh)
│       └── InterferogramProduct (complex interferogram)
└── MGRSTileProduct (MGRS 10 km grid cell composite)
    ├── RTCLRWComposite (LRW amplitude composite)
    ├── CohLRWComposite (LRW coherence composite)
    ├── RTCPassComposite (pass-direction amplitude composite)
    └── CohPassComposite (pass-direction coherence composite)
```

### Key Concepts

- **GTC**: Beta nought backscatter with geometric terrain correction only
- **RTC**: Gamma nought backscatter with terrain flattening (radiometric correction applied by SNAP); the `simulatedImage` band (simulated local illuminated area) is stored as a second band in the same GeoTIFF and used for LRW weighting
- **Footprint**: Unique spatial location defined by `burst_id` + `sub_swath` (stationary over time)
- **Grid cell**: MGRS 10 km grid square; defines height/width, affine transform, and CRS (UTM zone)

## Implemented Components

### ✅ Core Classes

#### 1. **Product Classes** (`product.py`)

- `BackscatterProduct`: Abstract base for single-burst backscatter; exposes `is_terrain_flattened`
  - `RTCProduct`: Terrain flattened; `product_type = "RTC"`; bands: `Gamma0`, `simulatedImage`
  - `GTCProduct`: Geometry-corrected only; `product_type = "GTC"`; band: `Beta0`
- `CoregisteredProduct`: Abstract base for co-registered burst pairs; exposes `baseline_days`
  - `CoherenceProduct`: Coherence magnitude; `product_type = "COH"`
  - `CoherenceComplexProduct`: Complex coherence (i + q); `product_type = "COH_CMPLX"`
  - `InterferogramProduct`: Complex interferogram; `product_type = "IFG"`
- `MGRSTileProduct`: Abstract base for MGRS-tile composites
  - `RTCLRWComposite`: LRW RTC intensity composite; `product_type = "RTC_LRW"`
  - `CohLRWComposite`: LRW coherence composite; `product_type = "COH_LRW"`
  - `RTCPassComposite`: Unweighted pass-direction RTC composite; `product_type = "RTC_COMP_{A/D}"`
  - `CohPassComposite`: Unweighted pass-direction coherence composite; `product_type = "COH_COMP_{A/D}"`

#### 2. **MGRSGeoBox / MGRSGrid** (`mgrs.py`)

- `MGRSGeoBox`: Extends `odc.geo.GeoBox` with a `mgrs_code` attribute. Created via `MGRSGeoBox.from_mgrs(mgrs_code)`.
- `MGRSGrid`: Generates all 10 km MGRS grid squares intersecting a filter geometry, iterates as `MGRSGeoBox` instances.

```python
from deepfreezer.mgrs import MGRSGeoBox, MGRSGrid

cell = MGRSGeoBox.from_mgrs("32TMT64", resolution=10.0)
grid = MGRSGrid(filter_geometry=aoi_geometry)
for cell in grid:
    print(cell.mgrs_code, cell.crs)
```

#### 3. **FileManager** (`filemanager.py`)

Centralized file system operations:

- Directory management (`slc_dir`, `prod_dir`, `comp_dir`)
- Path generation via a single generic engine (`_build_pattern` / `_parse_filename`) driven by per-class entries in `_PARSERS`
- SLC: `get_slc_path(burst)`, `find_slc_bursts`
- Product discovery: `find_rtc_products`, `find_gtc_products`, `find_coherence_products`, `find_coherence_complex_products`, `find_interferogram_products`
- Composite discovery: `find_rtc_lrw_products`, `find_coh_lrw_products`, `find_rtc_pass_composites`, `find_coh_pass_composites`
- Generic helpers: `get_product_path(product)`, `product_exists(product)`

**Key Design:**

- All filename generation flows through one generic engine: each metadata class (including `Burst`) registers a `(prefix, FieldSpec list, suffix)` entry in `_PARSERS`
- SLC paths are deterministic — `get_slc_path(burst).exists()` is the existence check (single HEAD on S3, no glob)
- Date filtering NOT supported in find methods — use list comprehension on results if needed

**Directory Structure:**

```bash
base_dir/
├── s1-slc/   # Downloaded SLC bursts
├── s1-prod/  # Per-burst processed products (RTC, GTC, SIM, COH, IFG)
└── s1-comp/  # MGRS-tile composites (LRW and pass)
```

**Filename Conventions:**

| Product | Pattern |
|---|---|
| SLC burst | `S1_SLC_{rel_orbit:03d}_{burst_id:06d}_{sub_swath}_{polarization}_{date:%Y%m%d}.zip` |
| RTC | `S1_RTC_{rel_orbit:03d}_{burst_id:06d}_{sub_swath}_{polarization}_{date:%Y%m%d}_{utm_zone}.tif` |
| GTC | `S1_GTC_{rel_orbit:03d}_{burst_id:06d}_{sub_swath}_{polarization}_{date:%Y%m%d}_{utm_zone}.tif` |
| Coherence magnitude | `S1_COH{baseline_days:02d}_{rel_orbit:03d}_{burst_id:06d}_{sub_swath}_{polarization}_{date1:%Y%m%d}_{date2:%Y%m%d}_{utm_zone}.tif` |
| Complex coherence | `S1_COH{baseline_days:02d}_CMPLX_{rel_orbit:03d}_{burst_id:06d}_{sub_swath}_{polarization}_{date1:%Y%m%d}_{date2:%Y%m%d}_{utm_zone}.tif` |
| Interferogram | `S1_IFG{baseline_days:02d}_CMPLX_{rel_orbit:03d}_{burst_id:06d}_{sub_swath}_{polarization}_{date1:%Y%m%d}_{date2:%Y%m%d}_{utm_zone}.tif` |
| RTC LRW composite | `S1_RTC_LRW_{mgrs_code}_{polarization}_{start_date:%Y%m%d}_{end_date:%Y%m%d}.tif` |
| Coherence LRW composite | `S1_COH{baseline_days:02d}_LRW_{mgrs_code}_{polarization}_{start_date:%Y%m%d}_{end_date:%Y%m%d}.tif` |
| RTC pass composite | `S1_RTC_COMP_{A/D}_{mgrs_code}_{polarization}_{start_date:%Y%m%d}_{end_date:%Y%m%d}.tif` |
| Coherence pass composite | `S1_COH{baseline_days:02d}_COMP_{A/D}_{mgrs_code}_{polarization}_{start_date:%Y%m%d}_{end_date:%Y%m%d}.tif` |

**SLC filename notes:**

- SLC filenames now follow the same convention as derived products and are produced by the shared `_build_pattern` engine. The outer zip name is set by `FileManager.get_slc_path(burst)` and applied uniformly by both the OData and S3 download paths.
- The **inner** SAFE folder name inside the zip keeps the ESA-conventional `S1{A,B}_SLC_{sensingTime}_{burst_id}_{sub_swath}_{polarization}_{datatake_id}.SAFE` form (preserves platform/datatake provenance and SNAP compatibility).
- Platform letter and datatake ID are intentionally **not** encoded in the outer zip name; they remain inside the SAFE folder.
- `sub_swath` is validated at parse time against `IW[1-3]|EW[1-5]`.
- `Burst.date` is populated from CDSE's `BeginningDateTime` (== annotation `sensingTime`)
**Regex conventions:** The shared `_parse_filename` builds named-group regexes (e.g. `(?P<burst_id>\d{6})`) from each class's `FieldSpec` list.

#### 4. **SNAPClient** (`snap_client.py`)

Low-level interface to SNAP GPT:

- `create_backscatter(burst, crs, *, config: BackscatterConfig)`: Single burst → RTC (default) or GTC, depending on `config.flatten_terrain`. Bands: `Gamma0` + `simulatedImage` (RTC) or `Beta0` (GTC).
- `create_coherence(pair, crs, *, config: CoherenceConfig)`: Burst pair → magnitude (default) or complex coherence depending on `config.write_complex`.
- `create_interferogram(pair, crs, *, config: InterferogramConfig)`: Burst pair → terrain-corrected interferogram.

SNAP graphs are generated dynamically at call time from each per-product config in `snap_graph.py` (`BackscatterConfig`, `CoherenceConfig`, `InterferogramConfig`, plus the shared `GeocodingConfig` base). Each config exposes `to_xml()` and `to_mermaid()` for inspection. (A guided-filter coherence variant existed as an experiment and was removed in the s1bursts split.)

`ProcessingConfig` is now a runtime-only struct: `gpt_path`, `file_manager`, `max_workers`, `max_ram_gb`, `gpt_threads_per_job`, `gpt_cache_ratio`, `auxdata_dir`. It auto-resolves `resolved_max_workers`, `resolved_xmx_mb`, `resolved_threads_per_job`, `resolved_cache_mb`.

#### 5. **BeamDimapDataset** (`dimap.py`)

Wrapper around SNAP's BEAM-DIMAP format for band-level I/O, used in the complex coherence pipeline:

- Parses `.dim` XML to discover band names and data file paths
- `read_band(name)`: Read a named band as float32 ndarray via rasterio
- `write_band(name, data)`: Overwrite a band's `.img` binary in-place
- `cleanup()`: Delete the `.dim` file and its `.data/` directory

**Complex coherence pipeline:**

1. SNAP `S1_SLCtoIfg-noTC.xml` → BEAM-DIMAP with `i_ifg`, `q_ifg`, `intensity1`, `intensity2`
2. `_compute_complex_coherence()` overwrites `i_ifg`/`q_ifg` with `i_coh`/`q_coh` in-place
3. SNAP `S1_TC.xml` geocodes the modified BEAM-DIMAP → GeoTIFF

#### 6. **BurstProcessor** (`processor.py`)

High-level batch processing wrapper around SNAPClient:

- Parallel execution via `ThreadPoolExecutor` (controlled by `ProcessingConfig.max_workers`)
- Progress bars (rich), error handling with continue-on-failure, `OperationSummary` reporting
- Methods:
  - `create_backscatter(bursts, crs, flatten_terrain=True)` → `OperationSummary[RTCProduct] | OperationSummary[GTCProduct]`
  - `create_coherences(pairs, crs, window=(15,3), magnitude=True)` → `OperationSummary[CoherenceProduct] | OperationSummary[CoherenceComplexProduct]`
  - `create_interferograms(pairs, crs)` → `OperationSummary[InterferogramProduct]`

**Result types** (`summary.py`):

- `OkResult[T]` — successful or skipped result; fields: `source`, `result: T`, `skipped: bool`
- `ErrorResult` — failed result; fields: `source`, `error: Exception`
- `type OperationResult[T] = OkResult[T] | ErrorResult` — union alias used in signatures
- `OperationSummary[T]` — aggregated batch summary; `products: list[T]`, `failed_items: list[ErrorResult]`

Use `isinstance(r, OkResult)` / `isinstance(r, ErrorResult)` to discriminate result types.


#### 7. **ResolutionWeighter** (`resolution_weighter.py`)

LRW and per-pass compositing:

- `ResolutionWeighter(file_manager, *, n_workers=8)` — reusable across MGRS cells; `geobox` passed per call
- `compose_rtc_lrw(rtc_products, geobox, *, config, overwrite)` → `RTCLRWComposite`
- `compose_coherence_lrw(coh_products, rtc_products, geobox, *, config, overwrite)` → `CohLRWComposite`
- `compose_rtc_by_pass(rtc_products, geobox, *, config, overwrite)` → `dict[str, RTCPassComposite]`
- `compose_coherence_by_pass(coh_products, geobox, *, config, overwrite)` → `dict[str, CohPassComposite]`

**Config objects:**

- `LRWCompositeConfig(method, quantile, self_weight_power, clip_max_area)` — weighted compositing
- `PassCompositeConfig(method)` — unweighted per-pass compositing

**LRW Formula:**

$$
\text{LRW} = \text{aggregate}_t\!\left(\text{pixel}_t\right), \quad w_t = \frac{1}{A_t} \cdot \text{pixel}_t^{p}
$$

where $p$ is `self_weight_power` (0 = pure resolution weighting; $p > 0$ promotes high values).

## Pipeline Flow

```bash
┌─────────────────────────────────────────────────────────────┐
│ 1. Download Bursts (BurstDatabase, CDSEClient)              │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Process Bursts (BurstProcessor → SNAPClient)             │
│    ├→ RTCProduct or GTCProduct (backscatter)                │
│    ├→ SimulationProduct (one per footprint)                 │
│    ├→ CoherenceProduct / CoherenceComplexProduct            │
│    └→ InterferogramProduct                                  │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Compositing (ResolutionWeighter)                         │
│    ├→ LRW composites                                        │
│    │   ├→ RTCLRWComposite (multi-footprint, RTC only)       │
│    │   └→ CohLRWComposite (multi-footprint)                 │
│    └→ Pass composites (unweighted mean per orbit direction) │
│        ├→ RTCPassComposite                                  │
│        └→ CohPassComposite                                  │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### Environment Variables

```bash
export SNAP_GPT_PATH="/path/to/snap/bin/gpt"
```

### ProcessingConfig

```python
config = ProcessingConfig(
    file_manager=file_manager,
    max_workers=3,        # parallel SNAP jobs (default: auto)
    max_ram_gb=50,        # total RAM budget (default: auto)
    gpt_threads_per_job=5,  # SNAP -q flag (default: auto)
    gpt_cache_ratio=0.7,    # fraction of per-job RAM for SNAP cache (default: 0.7)
)
```

See the benchmarking notes in project memory for recommended settings.

## Design Decisions

1. **Minimal Metadata in Filenames**: Only basic identifiers in filenames for simple glob patterns. Rich metadata stored in xarray attributes.

2. **simulatedImage co-located with RTC**: The simulated local illuminated area (used for LRW weighting) is stored as the second band (`simulatedImage`) of every RTC GeoTIFF rather than as a separate product. This eliminates the need for a separate simulation pass and keeps the area image date-aligned with its corresponding backscatter acquisition by construction.

3. **RTC-Only for LRW**: Only RTC amplitudes can be LRW-combined; terrain effects must be corrected before weighting.

4. **Separate GTC/RTC Product Types**: `GTCProduct` and `RTCProduct` are distinct classes; `is_terrain_flattened` provides a uniform boolean check.

5. **MGRS Code for Composites**: Composite products use `mgrs_code` (MGRS grid cell identifier, consistent with `MGRSGeoBox.mgrs_code`) instead of `burst_id`, as they combine multiple footprints.

6. **Deterministic SLC filenames**: SLC zips are written under `FileManager.get_slc_path(burst)`, derived solely from `(rel_orbit_number, burst_id, sub_swath, polarization, date)`. `rel_orbit_number` is looked up via `create_s1_orbit_number_lut()` from `deepfreezer.datasets.copernicus`, which maps `(burst_id, sub_swath)` → `rel_orbit_number` using the static ESA Sentinel-1 burst ID map. Determinism lets existence checks be a single `.exists()` call instead of a glob.

7. **Cached LUT functions**: `create_s1_orbit_number_lut` and `create_s1_orbit_direction_lut` are decorated with `@lru_cache(maxsize=None)` — the parquet-backed burst map is parsed only once per process.

8. **Terrain flattening in SNAP**: RTC terrain flattening is handled entirely by SNAP's `S1_SLCtoRTC.xml` graph. There is no separate post-processing step.

## References

- Burst database: `deepfreezer/sentinel1/database.py`
- CDSE API client: `deepfreezer/api/cdse.py`
- Burst collections: `deepfreezer/sentinel1/burst.py`
- SNAP graph generation: `deepfreezer/sentinel1/snap_graph.py`

## Example Workflows

### Workflow 1: Process Single Burst

```python
from pathlib import Path
from deepfreezer.sentinel1.filemanager import FileManager
from deepfreezer.sentinel1.snap_client import ProcessingConfig, SNAPClient
from pandas import Timestamp

base_dir = Path("/path/to/data")
file_manager = FileManager(base_dir)
config = ProcessingConfig(file_manager=file_manager)
snap_client = SNAPClient(config)

result = snap_client.create_backscatter(burst, crs=crs, config=BackscatterConfig())
rtc_product = result.result  # RTCProduct
```

### Workflow 2: Process Burst Collection

```python
from deepfreezer.sentinel1 import BackscatterConfig, BurstProcessor

processor = BurstProcessor(config)

# Default config = RTC, Copernicus 30m DEM, multilook (5, 1).
rtc_summary = processor.create_backscatter(bursts, crs=crs)
print(f"Processed {rtc_summary.processed}/{rtc_summary.total} RTC products")

# GTC instead of RTC
processor.create_backscatter(bursts, crs=crs, config=BackscatterConfig(flatten_terrain=False))
```

### Workflow 3: Process Coherence Pairs

```python
from deepfreezer.sentinel1 import CoherenceConfig

pairs = bursts.find_pairs(baseline_days=6)

# Coherence magnitude with ~60m window (defaults: window_rg=15, window_az=3)
coh_summary = processor.create_coherences(pairs, crs=crs)

# Complex coherence
processor.create_coherences(pairs, crs=crs, config=CoherenceConfig(write_complex=True))

# Interferograms
processor.create_interferograms(pairs, crs=crs)
```

### Workflow 4: Query Existing Products

```python
rtc_products = file_manager.find_rtc_products(burst_id=174374, sub_swath="IW1", polarization="VV")
coh_products = file_manager.find_coherence_products(burst_id=174374, sub_swath="IW1", baseline_days=6)

# Date filtering via list comprehension
from pandas import Timestamp
filtered = [p for p in rtc_products if Timestamp("2024-01-01") <= p.date <= Timestamp("2024-12-31")]

# Query composites
lrw = file_manager.find_rtc_lrw_products(mgrs_code="32TMT64", polarization="VV")
pass_asc = file_manager.find_rtc_pass_composites(mgrs_code="32TMT64", orbit_direction="ASCENDING")
```