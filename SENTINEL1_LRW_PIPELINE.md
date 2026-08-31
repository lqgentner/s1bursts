# Sentinel-1 Local Resolution Weighting (LRW) Pipeline

> **Maintenance note:** Update this document whenever a major rework of the `s1bursts` package is completed (e.g. new product classes, renamed fields, new pipeline stages, changed filename conventions).

## Overview

This pipeline processes Sentinel-1 SLC bursts to create composites over MGRS grid cells. Two composite types are produced:

- **LRW composites**: Locally resolution weighted (LRW) combination of multiple burst footprints; weights = inverse of simulated area (1/A)
- **Pass composites**: Unweighted temporal mean per pass direction (ascending / descending)

## Architecture

### Product Hierarchy

```bash
BaseProduct
├── GeocodedBurst (single burst, geocoded)
│   ├── BackscatterProduct (abstract)
│   │   ├── RTCProduct (gamma nought, terrain flattened; includes simulatedImage band)
│   │   └── GTCProduct (beta nought, geometry-corrected only)
│   └── CoregisteredProduct (two co-registered bursts)
│       ├── CoherenceProduct (coherence)
│       └── InterferogramProduct (complex interferogram)
└── MGRSTileProduct (MGRS 10 km grid cell composite)
    ├── RTCLRWComposite (LRW amplitude composite)
    ├── CohLRWComposite (LRW coherence composite)
    ├── RTCPassComposite (pass-direction amplitude composite)
    └── CohPassComposite (pass-direction coherence composite)
```

### Key Concepts

- **GTC**: Beta nought backscatter with geometric terrain correction only
- **RTC**: Gamma nought backscatter with terrain flattening (radiometric correction applied by SNAP); the `simulatedImage` band (the local contributing area) is stored as a second band in the same GeoTIFF and used for LRW weighting
- **Footprint**: Unique spatial location defined by `burst_id` + `sub_swath` (stationary over time)
- **Grid cell**: MGRS 10 km grid square; defines height/width, affine transform, and CRS (UTM zone)
- **Local contributing area** (`A`): the area of all terrain facets within a pixel, expressed in the plane perpendicular to slant range. Terrain flattening divides beta nought by it, so `A = 1` is a flat reference facet and `1/A` is the local resolution that LRW weights by. SNAP calls it `simulatedImage`, CEOS-ARD calls the layer `scatteringArea`, OPERA calls it `rtc_anf_gamma0_to_beta0`; all three are the same quantity

### Borrowed vocabulary

The tags and layer names deliberately follow the [CEOS-ARD Product Family Specification for Synthetic Aperture Radar](https://ceos.org/ard/files/PFS/SAR/v1.3.1/CEOS-ARD_PFS_SAR_v1.3.1.pdf) v1.3.1, so that a reader can look a term up in the accompanying metadata specification rather than learn a private dialect. **Borrowing the vocabulary is not conformance, and none is claimed** — see "What is not claimed" below.

The LRW composites correspond to the specification's **Composite Backscatter (CB)** product, whose Annex 5 restates Small et al. (2022) as the normative algorithm:

| This package | CEOS-ARD SAR v1.3.1 |
| --- | --- |
| `simulatedImage` (RTC band 2) | 2.3 Scattering Area Image |
| LRW composite | Annex 5, eq. A5.1-A5.3 |
| `QA-NUM` | 2.21 Contributing Observations Image |
| `QA-CQM` | 2.22 Composite Quality Map Image |

The `QA-*` filename tokens are this package's own; CEOS-ARD constrains metadata content, not file naming. `cqm_decibel` defaults to `False` because item 2.22 leaves the decibel conversion to the user.

#### GeoTIFF tags

Every product carries dataset tags describing how to read its pixels, built by `product_tags.py` and written on the branch that produced the file. This is deliberate redundancy with any downstream catalogue: a float32 backscatter raster arriving on its own is otherwise silently ambiguous between linear power and decibels, and misreading it produces wrong numbers rather than an error.

Exactly one tag names the specification: **`METADATA_VOCABULARY`**, written identically on every product as `CEOS-ARD PFS SAR v1.3.1; <url>`. It is a pointer, not a claim — it tells a reader holding one raster on its own where to look a term up. It is deliberately *not* in a `CEOS_ARD_` namespace, and there is no `CEOS_ARD_PRODUCT_TYPE`: no tag key may look like a CEOS-defined field, because the vocabulary is borrowed rather than authored by CEOS. Every other tag is a plain descriptive name, following OPERA's convention.

Because the pointer makes no claim, it has nothing to scope: it does not vary by product type, and it is not qualified when a raster is stored in decibels. `BACKSCATTER_SCALING` and `BACKSCATTER_CONVERSION_APPLIED` state that machine-readably instead.

Where a tag maps onto a parameter of the [CEOS-ARD SAR Metadata Specification](https://ceos.org/ard/) v1.3, it is that parameter name upper-snake-cased, so a reader can look it up:

| Tag | Metadata-spec parameter | PFS item |
| --- | --- | --- |
| `BACKSCATTER_MEASUREMENT` | `<BackscatterMeasurement>` | 3.1 |
| `BACKSCATTER_CONVENTION` | `<BackscatterConvention>` | 3.1 |
| `BACKSCATTER_CONVERSION_EQ` | `<BackscatterConversionEq>` | 3.1 (discharges 3.2) |
| `CQM_CONVERSION_EQ` | `<CQMdBScaling>` | 2.22 |
| `CONTRIBUTING_OBSERVATIONS` | `<NbrObservations>` | 2.21 |
| `COMPOSITING_METHOD` | `<CBMethod>` | 1.7.24 |
| `COMPOSITING_METHOD_REFERENCE` | `<CBReference>` | 1.7.24 |
| `PIXEL_COORDINATE_CONVENTION` | `<PixelCoordinateConvention>` | 1.7.10 |
| `RELATIVE_ORBIT_NUMBER` | `<RelativeOrbitNumber>` | 1.6.5 |
| `PASS_DIRECTION` | `<PassDirection>` | 1.6.5 |
| `MEASUREMENT_TYPE` | `<Measurements type=…>` | [InSAR] 3.8 |
| `COHERENCE_AVERAGE` / `COHERENCE_STD` | `<CoherenceAverage>` / `<CoherenceStd>` | [InSAR] 3.8 |
| `PHASE_QUALITY_METHOD` | `<InSARPhaseQualityMethod>` | [InSAR] 1.7.18 |
| `PHASE_QUALITY_REFERENCE` | `<InSARPhaseQualityReference>` | [InSAR] 1.7.18 |
| `COHERENCE_WINDOW_RANGE` / `COHERENCE_WINDOW_AZIMUTH` | `<CoherenceWindowColumn>` / `<CoherenceWindowLine>` | [InSAR] 1.7.18 |
| `COHERENCE_WEIGHTING_SHAPE` | `<CoherenceWeightingShape>` | [InSAR] 1.7.18 |
| `TEMPORAL_BASELINE_DAYS` | `<TemporalBaseline>` | [InSAR] 1.7.16 |

The rest have **no spec counterpart** and are simply descriptive: `METADATA_VOCABULARY`, `PRODUCT_TYPE` (the `s1bursts` identifier, matching `product.product_type`), `PRODUCT_FAMILY` (which reuses the specification's own product names, `Composite Backscatter` / `Composite Coherence`, so a user recognises what they have — a category, not a certification), `MGRS_TILE`, `COMPOSITE_START_DATE` / `COMPOSITE_END_DATE`, `POLARIZATION`, `LAYER_NAME`, `LAYER_DESCRIPTION`, `QA_LAYER`, `COMPOSITING_PARAMETERS`, `TERRAIN_CORRECTION`, `MULTILOOK_RANGE` / `MULTILOOK_AZIMUTH`, `BURST_ID`, `SUB_SWATH`, `ACQUISITION_DATE`, `DATE_PRIMARY` / `DATE_SECONDARY`, `GRID_SPACING`, `FLAT_EARTH_PHASE_REMOVED` / `TOPOGRAPHIC_PHASE_REMOVED`, the provenance tags below, and the decibel tags after them. The metadata spec defines no element for the "specific input parameters used" that item 1.7.24 asks for in prose, so `COMPOSITING_PARAMETERS` is this package's own — hence a descriptive name rather than a CEOS-looking one. It is built from the arguments actually passed to the `compose_*` call, never from a configuration object, so it cannot describe a run that did not happen.

##### Provenance

Following OPERA, which names each component and its version rather than asserting a single "processor":

| Tag | Value | Written by |
| --- | --- | --- |
| `DEM` | the DEM's name, or `N/A` | the per-burst product; inherited by composites |
| `PROCESSING_SOFTWARE` | `ESA SNAP` | the per-burst product; inherited by composites |
| `PROCESSING_SOFTWARE_VERSION` | e.g. `14.0.0`, or `N/A` | ditto |
| `PIPELINE_SOFTWARE` | `s1bursts` | every product, stamped fresh, never inherited |
| `PIPELINE_SOFTWARE_VERSION` | `s1bursts.__version__` | ditto |
| `PIPELINE_SOFTWARE_URL` | the repository | ditto |
| `PRODUCT_LEVEL` | `L2` per burst, `L3` composite | every product |
| `SOURCE_DATA_PRODUCT_LEVEL` | `L1` per burst, `L2` composite | every product |

Three rules make these load-bearing rather than decorative:

- **The DEM is tagged by identity, never by path.** A built-in DEM records its SNAP name; an external one records `GeocodingConfig.dem_name`, falling back to `N/A`. A file path is deliberately never written: it is machine-specific, and `CompositeConfig.to_snap_configs()` resolves an external DEM to its *staged local path*, so two machines processing one campaign would have disagreed and the composite would have recorded `N/A`.
- **Every provenance field is always present, as `N/A` where it could not be determined** (`product_tags.UNKNOWN`). The SNAP version is `N/A` when the installation carries no `VERSION.txt`; a composite's inherited fields are `N/A` when its inputs do not supply one answer. Absence would be ambiguous with a file written before the tag existed, which is exactly what a provenance field must not be. Conditional fields are different and *are* omitted when they do not apply: `COHERENCE_WINDOW_RANGE = N/A` on a backscatter composite would be nonsense, not information.
- **The pipeline version is stamped by whichever `s1bursts` writes the file and is never inherited from an input.** It is what identifies the vocabulary in this table — and the definition of `QA-NUM` has already changed once between releases, so a file without it is ambiguous between the two.
- **A composite built from older burst products still works.** Products written before these tags existed carry none, so the composite records `N/A` for `PROCESSING_SOFTWARE` and `PROCESSING_SOFTWARE_VERSION`, and a coherence composite simply omits the estimation-window fields it cannot inherit. Everything the composite generates itself is unaffected.

  Burst products are plain tiled GeoTIFFs, so `product_tags.burst_metadata()` can be backfilled onto them without reprocessing or touching a pixel — on S3 as well as locally. Two routes, both verified against a real bucket:

  | | mechanism | needs |
  | --- | --- | --- |
  | `rasterio_open(path, "r+")` | GDAL stages the object in a local temp file, patches it, re-uploads | a writable `TMPDIR`; **fails hard without one** |
  | `rewrite_tiff(path, path, tags=..., units=...)` | staged in memory, one `PutObject` | RAM for the uncompressed raster |

  Prefer `r+` where the rasters are large and `TMPDIR` points at disk, since the memory route holds the whole uncompressed image. **Composites are COGs** (`LAYOUT: COG`), and writing metadata into a finished COG costs it that layout, so they must be regenerated rather than patched.
- **Inherited fields must be unanimous.** `ResolutionWeighter._input_tags` reads every input, not just the first, and carries a value forward only where they all agree. A composite spanning two SNAP versions therefore reports `N/A` and logs a warning, instead of silently adopting whichever burst happened to sort first.

Which input differences are tolerated is a deliberate split. **Refused**, because the composite would otherwise be wrong rather than merely inconsistent: a CRS or pixel size other than the target grid's (the read window is computed in the geobox's coordinates against each source's own transform, so it would silently read pixels from somewhere else), a mixed temporal baseline on coherence (a composite carries one `baseline_days`, in its name and its tags), and a mixed polarization. **Tolerated with a warning**: a mixed SNAP version, which degrades consistency without invalidating the arithmetic and which an archive built over several campaigns will legitimately contain. A **mixed DEM is refused**, naming the values found — geocoding against two different elevation models within one composite is a defect, not an inconsistency.

Levels are `L1 → L2 → L3` in the EOSDIS sense; `L2` for a geocoded burst matches OPERA's own RTC-S1, and `L3` for a space-and-time-gridded composite matches ESA's Sentinel-1 monthly mosaic.

##### Band units

`dB` where the values are decibels, `linear power` where they are not, and empty for a genuinely dimensionless band (`simulatedImage`, coherence, `cqm`, `num_obs`). `linear power` is the PFS's own wording — items 2.21, 2.22 and 3.1 each enumerate the expression convention as *[linear amplitude or linear power]* — so the band says what it holds in the specification's words rather than in an invented one.

The `BACKSCATTER_CONVENTION` **tag** stays `Power`, not `linear power`: this is one of the places the two CEOS documents disagree, and the metadata spec's parameter list, which enumerates `<BackscatterConvention>` as `[Amplitude, Power]`, governs the parameter.

Some things are deliberately **not** emitted:

- **No `BACKSCATTER_CONVENTION` on decibel-stored data.** It is enumerated `[Amplitude, Power]` and describes the stored values, so neither fits; the dB branch emits `BACKSCATTER_SCALING = decibel` and `BACKSCATTER_CONVERSION_APPLIED` instead, and drops `BACKSCATTER_CONVERSION_EQ` so nothing invites a second conversion. `CQM_SCALING` / `CQM_CONVERSION_APPLIED` mirror this for `cqm_decibel=True`.
- **No conversion equation on coherence or on `QA-NUM`.** Coherence is a dimensionless magnitude on [0, 1], not a power quantity. The PFS prose for 2.21 lists a dB-scaling expression, but the metadata spec's parameter list for it does not — the bullets are copied from item 3.1, and a per-pixel integer count has no dB scaling.
- **`QA-NUM` is never described as a data mask.** PFS item 2.2 asks for three distinguishable states (valid / invalid / no data) and a zero count collapses two of them into one; item 2.2 is a known-unmet Threshold requirement.
- **No `[CB]` wording on a per-burst product,** which is not a composite, and no NRB or InSAR product-type claim on one either.

#### What is not claimed

The composite coherence product sits in a specification gap: coherence appears only as [InSAR] 3.8 (per-pair) and compositing only as [CB] (backscatter), so it takes its quantity vocabulary from the former and its compositing vocabulary from the latter while belonging to neither. `PRODUCT_FAMILY = Composite Coherence` is this package's own term, which the specification does not define and which therefore cannot be mistaken for an enumerated value.

More broadly, **no conformance assessment has been performed and none is claimed.** Item 2.2 (Data Mask Image) and item 4.3 (geolocation accuracy) are known unmet Threshold requirements, and the per-burst inputs carry no documentary NRB assessment. Demonstrating full compliance is out of scope for this project.

The coherence estimation window exists only in `CoherenceConfig` at graph-build time. It reaches a composite by exactly one route: the per-burst coherence products carry it in their tags, and `ResolutionWeighter` reads them back from its inputs. `PROCESSING_SOFTWARE` and `PROCESSING_SOFTWARE_VERSION` are inherited the same way.

## Implemented Components

### ✅ Core Classes

#### 1. **Product Classes** (`product.py`)

- `BackscatterProduct`: Abstract base for single-burst backscatter; exposes `is_terrain_flattened`
  - `RTCProduct`: Terrain flattened; `product_type = "RTC"`; bands: `Gamma0`, `simulatedImage`
  - `GTCProduct`: Geometry-corrected only; `product_type = "GTC"`; band: `Beta0`
- `CoregisteredProduct`: Abstract base for co-registered burst pairs; exposes `baseline_days`
  - `CoherenceProduct`: Coherence; `product_type = "COH"`
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
- Product discovery: `find_rtc_products`, `find_gtc_products`, `find_coherence_products`, `find_interferogram_products`
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
| Coherence | `S1_COH{baseline_days:02d}_{rel_orbit:03d}_{burst_id:06d}_{sub_swath}_{polarization}_{date1:%Y%m%d}_{date2:%Y%m%d}_{utm_zone}.tif` |
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
- `create_coherence(pair, crs, *, config: CoherenceConfig)`: Burst pair → terrain-corrected coherence.
- `create_interferogram(pair, crs, *, config: InterferogramConfig)`: Burst pair → terrain-corrected interferogram.

SNAP graphs are generated dynamically at call time from each per-product config in `snap_graph.py`: `BackscatterConfig`, `CoherenceConfig`, and `InterferogramConfig`. Each exposes `to_xml()` and `to_mermaid()` for inspection. They share the abstract `GeocodingConfig` base, which carries the Terrain-Correction tail parameters and declares `to_xml()` — it emits no graph of its own and cannot be instantiated. (A guided-filter coherence variant and a complex-coherence variant existed as experiments and have been removed.)

`SNAPConfig` is a runtime-only struct: `bin`, `max_workers`, `max_ram_gb`, `gpt_threads_per_job`, `gpt_cache_ratio`, `auxdata_dir`. It auto-resolves `resolved_max_workers`, `resolved_xmx_mb`, `resolved_threads_per_job`, `resolved_cache_mb`.

#### 5. **BurstProcessor** (`processor.py`)

High-level batch processing wrapper around SNAPClient:

- Parallel execution via `ThreadPoolExecutor` (controlled by `SNAPConfig.max_workers`)
- Progress bars (rich), error handling with continue-on-failure, `OperationSummary` reporting
- Methods:
  - `create_backscatter(bursts, crs, flatten_terrain=True)` → `OperationSummary[RTCProduct] | OperationSummary[GTCProduct]`
  - `create_coherences(pairs, crs, *, config: CoherenceConfig | None = None)` → `OperationSummary[CoherenceProduct]`
  - `create_interferograms(pairs, crs)` → `OperationSummary[InterferogramProduct]`

**Result types** (`summary.py`):

- `OkResult[T]` — successful or skipped result; fields: `source`, `result: T`, `skipped: bool`
- `ErrorResult` — failed result; fields: `source`, `error: Exception`
- `type OperationResult[T] = OkResult[T] | ErrorResult` — union alias used in signatures
- `OperationSummary[T]` — aggregated batch summary; `products: list[T]`, `failed_items: list[ErrorResult]`

Use `isinstance(r, OkResult)` / `isinstance(r, ErrorResult)` to discriminate result types.


#### 6. **ResolutionWeighter** (`resolution_weighter.py`)

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
│    ├→ CoherenceProduct                                      │
│    └→ InterferogramProduct                                  │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Compositing (ResolutionWeighter)                         │
│    ├→ LRW composites                                        │
│    │   ├→ RTCLRWComposite (multi-footprint, RTC only)       │
│    │   └→ CohLRWComposite (multi-footprint)                 │
│    └→ Pass composites (unweighted mean per pass direction) │
│        ├→ RTCPassComposite                                  │
│        └→ CohPassComposite                                  │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### Environment Variables

```bash
export SNAP_BIN="/path/to/snap/bin"
```

### SNAPConfig

```python
config = SNAPConfig(
    max_workers=3,  # parallel SNAP jobs (default: auto)
    max_ram_gb=50,  # total RAM budget (default: auto)
    gpt_threads_per_job=5,  # SNAP -q flag (default: auto)
    gpt_cache_ratio=0.7,  # fraction of per-job RAM for SNAP cache (default: 0.7)
)
```

See the benchmarking notes in project memory for recommended settings.

## Design Decisions

1. **Minimal Metadata in Filenames**: Only basic identifiers in filenames for simple glob patterns. Rich metadata stored in xarray attributes.

2. **simulatedImage co-located with RTC**: The local contributing area (used for LRW weighting) is stored as the second band (`simulatedImage`) of every RTC GeoTIFF rather than as a separate product. This eliminates the need for a separate simulation pass and keeps the area image date-aligned with its corresponding backscatter acquisition by construction.

3. **RTC-Only for LRW**: Only RTC amplitudes can be LRW-combined; terrain effects must be corrected before weighting.

4. **Separate GTC/RTC Product Types**: `GTCProduct` and `RTCProduct` are distinct classes; `is_terrain_flattened` provides a uniform boolean check.

5. **MGRS Code for Composites**: Composite products use `mgrs_code` (MGRS grid cell identifier, consistent with `MGRSGeoBox.mgrs_code`) instead of `burst_id`, as they combine multiple footprints.

6. **Deterministic SLC filenames**: SLC zips are written under `FileManager.get_slc_path(burst)`, derived solely from `(rel_orbit_number, burst_id, sub_swath, polarization, date)`. `rel_orbit_number` is looked up via `create_s1_orbit_number_lut()` from `deepfreezer.datasets.copernicus`, which maps `(burst_id, sub_swath)` → `rel_orbit_number` using the static ESA Sentinel-1 burst ID map. Determinism lets existence checks be a single `.exists()` call instead of a glob.

7. **Cached LUT functions**: `create_s1_orbit_number_lut` and `create_s1_pass_direction_lut` are decorated with `@lru_cache(maxsize=None)` — the parquet-backed burst map is parsed only once per process.

8. **Terrain flattening in SNAP**: RTC terrain flattening is handled entirely by SNAP's `S1_SLCtoRTC.xml` graph. There is no separate post-processing step.

## References

- Burst database: `deepfreezer/sentinel1/database.py`
- CDSE API client: `deepfreezer/api/cdse.py`
- Burst collections: `deepfreezer/sentinel1/burst.py`
- SNAP graph generation: `deepfreezer/sentinel1/snap_graph.py`

Terminology and standards:

- Small, D., Rohner, C., Miranda, N., Ruetschi, M., & Schaepman, M. E. (2022). Wide-Area Analysis-Ready Radar Backscatter Composites. *IEEE TGRS*, 60, 5201814. <https://doi.org/10.1109/TGRS.2021.3055562>
- CEOS-ARD PFS: Synthetic Aperture Radar v1.3.1 (2026-07-07). <https://ceos.org/ard/>

## Example Workflows

### Workflow 1: Process Single Burst

```python
from pathlib import Path
from deepfreezer.sentinel1.filemanager import FileManager
from s1bursts.snap_client import SNAPClient, SNAPConfig
from pandas import Timestamp

base_dir = Path("/path/to/data")
file_manager = FileManager(base_dir)
snap_client = SNAPClient(file_manager)

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
pairs = bursts.find_pairs(baseline_days=6)

# Coherence with ~60m window (defaults: window_rg=15, window_az=3)
coh_summary = processor.create_coherences(pairs, crs=crs)

# Interferograms
processor.create_interferograms(pairs, crs=crs)
```

### Workflow 4: Query Existing Products

```python
rtc_products = file_manager.find_rtc_products(burst_id=174374, sub_swath="IW1", polarization="VV")
coh_products = file_manager.find_coherence_products(
    burst_id=174374, sub_swath="IW1", baseline_days=6
)

# Date filtering via list comprehension
from pandas import Timestamp

filtered = [
    p for p in rtc_products if Timestamp("2024-01-01") <= p.date <= Timestamp("2024-12-31")
]

# Query composites
lrw = file_manager.find_rtc_lrw_products(mgrs_code="32TMT64", polarization="VV")
pass_asc = file_manager.find_rtc_pass_composites(mgrs_code="32TMT64", pass_direction="Ascending")
```