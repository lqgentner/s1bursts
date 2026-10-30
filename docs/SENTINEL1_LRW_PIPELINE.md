# Sentinel-1 Local Resolution Weighting (LRW) Pipeline

> This document is the domain and design reference behind the `s1bursts` API and the basis for the user guide. Keep it in step with the code: product classes, band and tag names, filename conventions, and the compositing formulas.

## Overview

This pipeline processes Sentinel-1 SLC bursts to create composites over MGRS grid cells. Two composite types are produced:

- **LRW composites**: Locally resolution weighted (LRW) combination of multiple burst footprints; weights = inverse of the local scattering area (1/A)
- **Pass composites**: Unweighted temporal mean per pass direction (ascending / descending)

## Architecture

### Product Hierarchy

```bash
BaseProduct
├── GeocodedBurst (single burst, geocoded)
│   ├── BackscatterProduct (abstract)
│   │   ├── RTCProduct (gamma nought, terrain flattened; includes the scattering_area band)
│   │   └── GTCProduct (beta nought, geometry-corrected only)
│   └── CoregisteredProduct (two co-registered bursts)
│       ├── CoherenceProduct (coherence)
│       └── InterferogramProduct (complex interferogram)
└── MGRSTileProduct (MGRS 10 km grid cell composite)
    ├── RTCLRWComposite (LRW intensity composite)
    ├── CohLRWComposite (LRW coherence composite)
    ├── RTCPassComposite (pass-direction gamma nought composite)
    ├── GTCPassComposite (pass-direction beta nought composite)
    ├── CohPassComposite (pass-direction coherence composite)
    ├── RTCLRWQA, CohLRWQA (QA-NUM and QA-CQM layers of an LRW composite)
    └── RTCPassQA, GTCPassQA, CohPassQA (QA-NUM layer of a pass composite)
```

### Key Concepts

- **GTC**: Beta nought backscatter with geometric terrain correction only
- **RTC**: Gamma nought backscatter with terrain flattening (radiometric correction applied by SNAP); the local scattering area is stored as the second band (`scattering_area`) of the same GeoTIFF and used for LRW weighting
- **Footprint**: Unique spatial location defined by `burst_id` + `sub_swath` (stationary over time)
- **Grid cell**: MGRS 10 km grid square; defines height/width, affine transform, and CRS (UTM zone)
- **Local scattering area** (`A`): the area of all terrain facets within a pixel, expressed in the plane perpendicular to slant range. Terrain flattening divides beta nought by it, so `A = 1` is a flat reference facet and `1/A` is the local resolution that LRW weights by. This is the CEOS-ARD term (layer `scatteringArea`); Small et al. (2022) call it the *local contributing area*, OPERA the *area normalization factor* (`rtc_anf_gamma0_to_beta0`), and SNAP's Terrain-Flattening writes it as `simulatedImage`. All four are the same quantity

### Borrowed vocabulary

The tags and layer names deliberately follow the [CEOS-ARD Product Family Specification for Synthetic Aperture Radar](https://ceos.org/ard/files/PFS/SAR/v1.3.1/CEOS-ARD_PFS_SAR_v1.3.1.pdf) v1.3.1, so that a reader can look a term up in the accompanying metadata specification rather than learn a private dialect. **Borrowing the vocabulary is not conformance, and none is claimed** — see "What is not claimed" below.

The LRW composites correspond to the specification's **Composite Backscatter (CB)** product, whose Annex 5 restates Small et al. (2022) as the normative algorithm:

| This package | CEOS-ARD SAR v1.3.1 |
| --- | --- |
| `scattering_area` (RTC band 2) | 2.3 Scattering Area Image |
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

The rest have **no spec counterpart** and are simply descriptive: `METADATA_VOCABULARY`, `PRODUCT_TYPE` (the `s1bursts` identifier, matching `product.product_type`), `PRODUCT_FAMILY` (which reuses the specification's own product names, `Composite Backscatter` / `Composite Coherence`, so a user recognises what they have — a category, not a certification), `MGRS_TILE`, `COMPOSITE_START_DATE` / `COMPOSITE_END_DATE` (the compositing window as inclusive calendar days: the `TimeRange` the caller passed, else the observed span), `FIRST_ACQUISITION_DATE` / `LAST_ACQUISITION_DATE` (the acquisitions that actually contributed to this file), `POLARIZATION`, `LAYER_NAME`, `LAYER_DESCRIPTION`, `QA_LAYER`, `COMPOSITING_PARAMETERS`, `TERRAIN_CORRECTION`, `MULTILOOK_RANGE` / `MULTILOOK_AZIMUTH`, `BURST_ID`, `SUB_SWATH`, `ACQUISITION_DATE`, `DATE_PRIMARY` / `DATE_SECONDARY`, `GRID_SPACING`, `FLAT_EARTH_PHASE_REMOVED` / `TOPOGRAPHIC_PHASE_REMOVED`, the provenance tags below, and the decibel tags after them. The metadata spec defines no element for the "specific input parameters used" that item 1.7.24 asks for in prose, so `COMPOSITING_PARAMETERS` is this package's own — hence a descriptive name rather than a CEOS-looking one. It is built from the arguments actually passed to the `compose_*` call, never from a configuration object, so it cannot describe a run that did not happen.

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

- **The DEM is tagged by identity, never by path.** A built-in DEM records its SNAP name; an external one records `GeocodingConfig.dem_name`, falling back to `N/A`. A file path is deliberately never written: it is machine-specific, so two machines processing one campaign would have disagreed and the composite would have recorded `N/A`.
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

Which input differences are tolerated is a deliberate split. **Refused**, because the composite would otherwise be wrong rather than merely inconsistent: a CRS, pixel size or sub-pixel grid offset other than the target grid's (the read window is computed in the geobox's coordinates against each source's own transform, so it would silently read pixels from somewhere else), a mixed temporal baseline on coherence (a composite carries one `baseline_days`, in its name and its tags), and a mixed polarization. **Tolerated with a warning**: a mixed SNAP version, which degrades consistency without invalidating the arithmetic and which an archive built over several campaigns will legitimately contain. A **mixed DEM is refused**, naming the values found — geocoding against two different elevation models within one composite is a defect, not an inconsistency.

Levels are `L1 → L2 → L3` in the EOSDIS sense; `L2` for a geocoded burst matches OPERA's own RTC-S1, and `L3` for a space-and-time-gridded composite matches ESA's Sentinel-1 monthly mosaic.

##### Band units

`dB` where the values are decibels, `linear power` where they are not, and empty for a genuinely dimensionless band (`scattering_area`, `coherence`, `cqm`, `num_obs`). `linear power` is the PFS's own wording — items 2.21, 2.22 and 3.1 each enumerate the expression convention as *[linear amplitude or linear power]* — so the band says what it holds in the specification's words rather than in an invented one.

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

## Components

### Bursts (`burst.py`, `query.py`, `catalog.py`)

- `Burst`: one acquisition of one footprint, identified by `burst_id`, `sub_swath`, `polarization` and the calendar day of `date`. Relative orbit number, pass direction and footprint geometry are propagated from the CDSE response or an SLC filename; a burst that carries none falls back to the SAR-MPC burst-ID map, an opt-in download (`download_burst_map()`).
- `BurstPair`: primary and secondary `Burst` of one footprint; `baseline_days` is the temporal baseline rounded to whole days.
- `BurstCollection` / `BurstPairCollection`: GeoDataFrame-backed containers with deduplication, slicing, an HTML preview, and `find_pairs(baseline=..., max_delta=pd.Timedelta(hours=6))` (keyword-only; an integer `baseline` is days, and either may be a `pandas.Timedelta`, `datetime.timedelta` or `numpy.timedelta64`, which `max_delta` must be; `baseline` must be whole days and `max_delta` under 12 hours, so every pair is named by the requested baseline). Selections return a new collection of the same type: `BurstCollection.within(time_range)` and `BurstPairCollection.starting_within(time_range)` keep the acquisitions (pairs: the primary) on a day of a `TimeRange`; `intersecting(geometry)` keeps the footprints touching a shapely geometry in WGS 84, an `odc.geo` `Geometry` in any CRS, or a `GeoBox` such as an `MGRSGeoBox` (a projected area is tested in its own CRS, with the footprints projected into it, so polar and near-global areas keep their shape); `filter(**fields)` keeps the entries whose columns equal a value or any of an iterable of values. The columns of `.df` are the burst's public field names, `rel_orbit_number` and `pass_direction` among them.
- `SearchQuery`: validated search parameters; `BurstCatalog.search(query)` turns it into OData filters and returns every match from one catalog request. A query matching more than one page (1000 bursts) raises `TooManyResultsError`; split it and combine the results with `+`.
- `BurstCatalog.download(bursts, *, overwrite=False, max_workers=5, endpoint=None)`: takes a `Burst`, a `BurstPair` or any iterable of them, and downloads each distinct burst once, in parallel via the CDSE OData burst endpoint or by extracting the burst from the parent product in CDSE S3 (`burst_extractor.py`); a failure on one endpoint is retried on the other, and a burst the source does not hold is remembered so it is not requested twice.

### Products (`product.py`)

Products are frozen dataclasses holding identity only; `FileManager` derives every path from them.

- `BackscatterProduct` (abstract, `is_terrain_flattened`): `RTCProduct` (`product_type = "RTC"`, bands `gamma0`, `scattering_area`) and `GTCProduct` (`"GTC"`, band `beta0`).
- `CoregisteredProduct` (abstract, `baseline_days`): `CoherenceProduct` (`"COH"`, band `coherence`) and `InterferogramProduct` (`"IFG"`, bands `ifg_real`, `ifg_imag`, `intensity_primary`, `intensity_secondary`).
- `MGRSTileProduct` (abstract, `crs` from the MGRS code): `RTCLRWComposite` (`"RTC_LRW"`), `CohLRWComposite` (`"COH_LRW"`), `RTCPassComposite` (`"RTC_COMP_{A/D}"`), `GTCPassComposite` (`"GTC_COMP_{A/D}"`), `CohPassComposite` (`"COH_COMP_{A/D}"`), and the QA auxiliaries `RTCLRWQA`, `CohLRWQA` (`qa_metric` is `QaMetric.NUM` or `QaMetric.CQM`), `RTCPassQA`, `GTCPassQA`, `CohPassQA` (always `NUM`).

The band description, the band's name inside the SNAP graph and its `LAYER_NAME` band tag are one and the same name. Composites keep the polarization as their band description.

### Grid cells (`freezebase.mgrs`)

`MGRSGeoBox` extends `odc.geo.GeoBox` with an `mgrs_code`; `MGRSGeoBox.from_mgrs("32TMT64", resolution=10.0)` builds one, and `MGRSGrid(filter_geometry=aoi)` iterates over every 10 km square a WGS 84 geometry intersects. Every `compose_*` call takes its target geobox explicitly, so one `ResolutionWeighter` serves any number of cells.

### FileManager (`filemanager.py`)

Deterministic paths and discovery, local or on S3 through `UPath`:

- `FileManager(base_dir, slc_dir=None, prod_dir=None, comp_dir=None)`; the defaults are `base_dir/s1-slc`, `s1-prod`, `s1-comp`.
- `get_slc_path(burst)`, `get_product_path(product)`, `product_exists(product)`.
- Discovery by glob and regex: `find_slc_bursts`, `find_rtc_products`, `find_gtc_products`, `find_coherence_products`, `find_interferogram_products`, `find_rtc_lrw_products`, `find_coh_lrw_products`, `find_rtc_pass_composites`, `find_gtc_pass_composites`, `find_coh_pass_composites`, `find_rtc_lrw_qa`, `find_coh_lrw_qa`, `find_rtc_pass_qa`, `find_gtc_pass_qa`, `find_coh_pass_qa`. Date filtering is left to a list comprehension over the result.

Every filename is built and parsed by one engine: each product class registers a prefix, an ordered `FieldSpec` list and a suffix in `_PARSERS`, so `get_slc_path(burst).exists()` is the existence check (one HEAD request on S3, no glob).

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
| GTC pass composite | `S1_GTC_COMP_{A/D}_{mgrs_code}_{polarization}_{start_date:%Y%m%d}_{end_date:%Y%m%d}.tif` |
| Coherence pass composite | `S1_COH{baseline_days:02d}_COMP_{A/D}_{mgrs_code}_{polarization}_{start_date:%Y%m%d}_{end_date:%Y%m%d}.tif` |
| LRW QA layers | `S1_RTC_LRW_QA-{NUM,CQM}_…` and `S1_COH{bb}_LRW_QA-{NUM,CQM}_…` with the composite's fields |
| Pass QA layers | `S1_RTC_COMP_{A/D}_QA-NUM_…`, `S1_GTC_COMP_{A/D}_QA-NUM_…` and `S1_COH{bb}_COMP_{A/D}_QA-NUM_…` |

The SAFE folder inside an SLC zip keeps ESA's `S1{A,B}_SLC_{sensingTime}_{burst_id}_{sub_swath}_{polarization}_{datatake_id}.SAFE` name, which preserves the platform and datatake and is what SNAP reads. `Burst.date` is CDSE's `BeginningDateTime`, the annotation's `sensingTime`.

### SNAP processing (`snap_graph.py`, `snap_client.py`, `processor.py`)

Graphs are generated at call time from frozen Pydantic configs: `RTCConfig`, `GTCConfig`, `CoherenceConfig` and `InterferogramConfig`, all deriving from the abstract `GeocodingConfig` that carries the shared Terrain-Correction tail (`grid_spacing`, `dem`, resampling, standard-grid alignment, external-DEM handling). `RTCConfig` (terrain-flattened gamma-0 backscatter) and `GTCConfig` (beta-0 backscatter, geocoded only) share the abstract `BackscatterConfig` (`multilook`, `replace_zeros_with_nan`); only `RTCConfig` has the Terrain-Flattening settings `dem_oversampling` and `additional_overlap`. Each exposes `to_xml(snap_version=...)` and `to_mermaid()`. The band naming of a coregistered stack differs between SNAP 13 and 14, and two Coherence options exist only from SNAP 14, so the client passes the installed version in.

`SNAPConfig` is the runtime side: `bin`, `rasterio_profile`, `max_workers`, `max_ram_gb`, `threads_per_job`, `cache_ratio`, `auxdata_path`, each defaulting to its `SNAP_*` environment variable, resolved into a per-worker Java heap, tile cache and thread count at construction.

- `SNAPClient(file_manager, config=None)`: `create_backscatter(burst, crs, *, config, overwrite=False)` (an `RTCConfig` gives an `RTCProduct`, a `GTCConfig` a `GTCProduct`), `create_coherence(pair, crs, *, config, overwrite=False)`, `create_interferogram(pair, crs, *, config, overwrite=False)`. Each stages the SLC locally if it lives on S3, runs `gpt` in an isolated user directory that shares the auxdata cache, stamps the product's metadata and no-data value, and rewrites the result compressed to its final location.
- `BurstProcessor(file_manager, config=None, *, show_progress=True, continue_on_error=True)`: the batch wrapper. `create_backscatter(bursts, crs, *, config=None, overwrite=False)` (RTC by default, GTC with `config=GTCConfig()`), `create_coherences(pairs, crs, *, config=None, overwrite=False)` and `create_interferograms(pairs, crs, *, config=None, overwrite=False)` run `SNAPConfig.resolved_max_workers` jobs in parallel and return an `OperationSummary`. `crs` must be a WGS 84 / UTM zone, north or south, since product file names record the zone; anything else raises `ValueError` before the first job. `SNAPClient.create_*` and the products' `from_burst()` / `from_burst_pair()` enforce the same. Nothing picks the zone automatically: a composite's inputs take the target cell's (`crs=cell.crs`), and `utm_zone_of(burst)` gives the zone holding a burst's footprint centre for bursts processed on their own.

Results (`summary.py`): `OkResult[T]` (`source`, `result`, `skipped`), `ErrorResult` (`source`, `error`), and `OperationSummary[T]` with `n_total`, `n_processed`, `n_skipped`, `n_failed`, `products`, `failed_items`, `processing_time`, `print_summary()` (failures grouped by error type and first message line, with counts) and `raise_on_failure(ignore=())`, which raises `OperationFailedError` listing the failed items and otherwise returns the summary, so it chains: `catalog.download(bursts).raise_on_failure()`. With `continue_on_error=True`, the default, a batch attempts every item and records the failures, and `raise_on_failure()` then fails once, after every product that could be made is written; the error's `summary` still holds those products, and a rerun skips them and retries only the failures, except bursts a `BurstCatalog` already found missing at the source, which the same catalog reports again without retrying (and records without raising, even with `continue_on_error=False`). `continue_on_error=False` stops at the first error instead. `ignore` takes error types whose failures are tolerated, such as `BurstNotFoundError` for bursts the source no longer serves; they stay in `failed_items` and are counted in the message. A per-item failure is a `BurstError` subclass and is recorded; a `ConfigurationError` (missing executable, unusable environment) propagates and stops the batch.

### Compositing (`resolution_weighter.py`)

`ResolutionWeighter(file_manager, *, rasterio_profile=None, max_workers=8, checksum=False)`:

- `compose_rtc_lrw(rtc_products, geobox, *, aggregation="mean", clip_max_resolution=None, decibel=False, write_qa=False, cqm_decibel=False, overwrite=False)` → `RTCLRWComposite`, or `(composite, (qa_num, qa_cqm))` with `write_qa=True`.
- `compose_coherence_lrw(coh_products, rtc_products, geobox, *, aggregation="mean", clip_max_resolution=None, write_qa=False, cqm_decibel=False, overwrite=False)` → `CohLRWComposite`; the RTC products supply the scattering area of the matching orbit and date.
- `compose_rtc_by_pass(rtc_products, geobox, *, aggregation="mean", decibel=False, write_qa=False, overwrite=False)` → `dict[PassDirection, RTCPassComposite]`, or `(composite, qa_num)` values with `write_qa=True`.
- `compose_gtc_by_pass(gtc_products, geobox, *, aggregation="mean", decibel=False, write_qa=False, overwrite=False)` → `dict[PassDirection, GTCPassComposite]`, the same unweighted composite of beta nought; with no scattering area there is nothing to weight by.
- `compose_coherence_by_pass(coh_products, geobox, *, aggregation="mean", write_qa=False, overwrite=False)` → `dict[PassDirection, CohPassComposite]`.

`aggregation` is `"mean"`, `"median"` or a quantile level in `(0, 1)`. Within one relative orbit and acquisition date the bursts are mosaicked by first valid pixel, so seams are never averaged; the resulting orbit-date observations are then aggregated in time.

**LRW formula** (Small et al. 2022, eq. 3-6; CEOS-ARD SAR v1.3.1, Annex 5):

$$
w_i = \frac{1}{A_i}, \qquad W_i = \frac{w_i}{\sum_j w_j}, \qquad
\text{LRW} = \text{aggregate}_i\!\left(x_i;\, W_i\right), \qquad
\text{CQM} = \frac{1}{\sum_i W_i A_i}
$$

where $A_i$ is the local scattering area of observation $i$ and $x_i$ its intensity or coherence. A pixel whose area is zero or non-finite carries no weight, `clip_max_resolution` caps $w_i$ to tame shadow, and `QA-NUM` counts the observations with non-zero weight. The CQM is this weighted mean of the area under every aggregation. Under `median` or a quantile it describes the weighting rather than the written value: a weighted quantile takes its value from the observations that rank there in the measured value, which changes with speckle from pixel to pixel, so their area would be noise, and the weighted quantile of the area itself ranks by area and drifts to the finest acquisition.

## Pipeline flow

```text
1. Search and download      BurstCatalog.search → BurstCollection → .find_pairs → BurstCatalog.download
2. Process                  BurstProcessor → SNAPClient → RTCProduct | GTCProduct | CoherenceProduct | InterferogramProduct
3. Composite                ResolutionWeighter → RTCLRWComposite | CohLRWComposite (+ QA-NUM, QA-CQM)
                                               → RTCPassComposite | GTCPassComposite | CohPassComposite (+ QA-NUM)
```

## Configuration

Credentials and SNAP settings come from the environment (`CDSE_*`, `CDSE_S3_*`, `SNAP_*`; see the README table). `SNAPConfig` overrides the `SNAP_*` values per run:

```python
config = SNAPConfig(
    max_workers=3,  # parallel gpt processes
    max_ram_gb=50,  # total budget; split into per-process heaps
    threads_per_job=5,  # gpt -q
    cache_ratio=0.7,  # fraction of each heap used as tile cache
)
```

## Design decisions

1. **Minimal metadata in filenames.** Only the identifiers a glob needs; everything else is a GeoTIFF tag.
2. **The scattering area travels with its backscatter.** It is band 2 of every RTC GeoTIFF, never a separate product, so it is date-aligned with the acquisition by construction and a coherence composite can take it from the RTC product of the same orbit and date.
3. **RTC only for LRW.** Terrain effects must be corrected before weighting; GTC products cannot be LRW-composited, only pass-composited.
4. **Separate GTC and RTC classes** with `is_terrain_flattened` as the uniform check.
5. **Composites are keyed by `mgrs_code`**, not by burst, because they combine several footprints.
6. **Deterministic SLC filenames** from `(rel_orbit_number, burst_id, sub_swath, polarization, date)`. The relative orbit number is propagated from the search response or the SLC filename, so the normal workflow never opens the burst-ID map; the map is consulted only for a hand-built burst and must be downloaded first.
7. **Terrain flattening happens in SNAP.** The RTC graph contains Terrain-Flattening; there is no post-processing step.
8. **No Multilook in the coherence graph.** The estimation `window` is the only control over the looks; a multilook after the estimator would only average coherence magnitudes.

## References

- Small, D., Rohner, C., Miranda, N., Ruetschi, M., & Schaepman, M. E. (2022). Wide-Area Analysis-Ready Radar Backscatter Composites. *IEEE TGRS*, 60, 5201814. <https://doi.org/10.1109/TGRS.2021.3055562>
- CEOS-ARD PFS: Synthetic Aperture Radar v1.3.1 (2026-07-07). <https://ceos.org/ard/>

## Example workflows

### Select part of a search

```python
from freezebase.mgrs import MGRSGeoBox

from s1bursts import TimeRange

window = TimeRange("2022-06-01", "2022-08-31")
cell = MGRSGeoBox.from_mgrs("32TMT64", resolution=10.0)

# The bursts of one MGRS cell from an AOI search
cell_bursts = bursts.intersecting(cell)
summer = cell_bursts.within(window).filter(rel_orbit_number=[66, 139])
pairs = cell_bursts.find_pairs(baseline=12).starting_within(window)
```

### Process a single burst

```python
from pathlib import Path

from s1bursts import FileManager, RTCConfig
from s1bursts.snap_client import SNAPClient
from s1bursts.summary import OkResult

file_manager = FileManager(Path("/path/to/data"))
snap_client = SNAPClient(file_manager)

result = snap_client.create_backscatter(burst, crs=crs, config=RTCConfig())
if isinstance(result, OkResult):
    rtc_product = result.result  # RTCProduct
```

### Process a collection

```python
from s1bursts import BurstProcessor, GTCConfig

processor = BurstProcessor(file_manager)

# Default config: RTCConfig(), Copernicus 30 m DEM, multilook (5, 1).
rtc_summary = processor.create_backscatter(bursts, crs=crs)
print(f"Processed {rtc_summary.n_processed}/{rtc_summary.n_total} RTC products")

# GTC instead of RTC: the config type selects the product
gtc_products = processor.create_backscatter(bursts, crs=crs, config=GTCConfig()).products
```

### Coherence and interferograms

```python
pairs = bursts.find_pairs(baseline=12)  # days

# Coherence with the default (25, 5) estimation window
coh_summary = processor.create_coherences(pairs, crs=crs)

processor.create_interferograms(pairs, crs=crs)
```

### Composite an MGRS cell

```python
from freezebase.mgrs import MGRSGeoBox

from s1bursts import ResolutionWeighter

cell = MGRSGeoBox.from_mgrs("32TMT64", resolution=10.0)
weighter = ResolutionWeighter(file_manager)

rtc_products = file_manager.find_rtc_products(polarization="VV")
composite, (qa_num, qa_cqm) = weighter.compose_rtc_lrw(rtc_products, cell, write_qa=True)

coh_products = file_manager.find_coherence_products(polarization="VV", baseline_days=12)
weighter.compose_coherence_lrw(coh_products, rtc_products, cell)
```

### Query existing products

```python
from pandas import Timestamp

rtc_products = file_manager.find_rtc_products(burst_id=174374, sub_swath="IW1", polarization="VV")
coh_products = file_manager.find_coherence_products(
    burst_id=174374, sub_swath="IW1", baseline_days=6
)

# Date filtering via list comprehension
filtered = [
    p for p in rtc_products if Timestamp("2024-01-01") <= p.date <= Timestamp("2024-12-31")
]

# Composites
lrw = file_manager.find_rtc_lrw_products(mgrs_code="32TMT64", polarization="VV")
pass_asc = file_manager.find_rtc_pass_composites(mgrs_code="32TMT64", pass_direction="Ascending")
```
