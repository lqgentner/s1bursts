# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is pre-1.0 (`0.x`), minor releases may contain breaking changes.

## [Unreleased]

### Added

- `FileManager` accepts strings for local directories, and
  `prepare_dem_for_snap()` a string for `dem`.
- The `crs` of every `BurstProcessor.create_*` method, and of the products'
  `from_burst()` and `from_burst_pair()`, now accepts strings and EPSG codes,
  such as `"EPSG:32632"` or `32632`. Previously only objects with a `to_wkt()`
  method worked.
- Every `BurstProcessor.create_*` method accepts any iterable of bursts or
  burst pairs, such as a list, not only a `BurstCollection` or
  `BurstPairCollection`.
- `utm_zone_of(burst)` returns the WGS 84 / UTM zone that holds the centre of
  a burst's or burst pair's footprint, as the `crs` for bursts processed on
  their own. Products meant for a composite still take the cell's zone,
  `crs=cell.crs`.
- `OperationSummary.raise_on_failure()` raises an `OperationFailedError`
  listing the failed items, and otherwise returns the summary, so it chains:

  ```python
  # before
  summary = catalog.download(bursts)
  if summary.n_failed:
      raise RuntimeError(...)
  # after
  catalog.download(bursts).raise_on_failure()
  ```

  A batch run with `continue_on_error=True` (the default) attempts every item
  and still returns normally when every item failed; this is the check to
  call after it. The error carries the summary, so the products that were made
  stay within reach as `err.summary.products`, and the failures as
  `err.failed_items`. Running the batch again skips the existing products and
  retries only the failures; a `BurstCatalog` reports the bursts it already
  found missing at the source again without retrying them. `ignore` tolerates
  failures of the given error types: they do not raise but stay in
  `failed_items`, and the message counts them.

  ```python
  # a burst CDSE no longer serves does not stop the run
  catalog.download(bursts).raise_on_failure(ignore=BurstNotFoundError)
  ```
- `BurstCatalog.download()` accepts any iterable of bursts or burst pairs,
  such as a list or a generator, besides a single `Burst` or `BurstPair` and
  the collections. A burst listed more than once, such as one shared by two
  pairs, is downloaded once.
- Burst collections can be narrowed without touching `.df`. Each selection
  returns a new collection of the same type:
  `BurstCollection.within(time_range)` keeps the bursts acquired on a day of a
  `TimeRange`, `BurstPairCollection.starting_within(time_range)` the pairs
  whose primary burst was, and on both `intersecting(geometry)` keeps the
  footprints that touch a geometry and `filter(**fields)` the entries whose
  columns equal a value or any of an iterable of values. `intersecting` takes
  a shapely geometry in WGS 84, an `odc.geo` `Geometry` in any CRS, or a
  `GeoBox` such as an MGRS cell. A projected area is tested in its own CRS,
  so polar and near-global areas work:

  ```python
  cell_bursts = bursts.intersecting(cell)  # an MGRSGeoBox
  in_window = bursts.within(window).filter(rel_orbit_number=[66, 139])
  pairs = bursts.find_pairs(baseline=12).starting_within(window)
  ```
- `SNAPConfig(timeout=...)` sets how many seconds one `gpt` run may take,
  defaulting to `SNAP_TIMEOUT` or 3600 as before; `None`, or
  `SNAP_TIMEOUT=none`, removes the limit; an infinite or NaN value is
  rejected. It was fixed at one hour.
- `SnapProcessingError` carries `returncode`, the exit code of `gpt`
  (negative for the signal that ended it), and `output_tail`, the last lines
  it printed, so a failure can be inspected without parsing the message.

### Changed

- SNAP failures say more. A `gpt` run ended by a signal names it, as in
  `exit -9 (SIGKILL; usually out of memory: lower SNAPConfig.max_workers or
  max_ram_gb)`. A run that times out reports the last lines `gpt` printed
  instead of the command line. Each failure is logged once, without a
  traceback, instead of twice with one. The message of a failed run
  reads `GPT processing failed with exit 1: …` instead of `GPT processing
  failed (exit 1): …`.
- **Breaking:** `BurstDatabase` is renamed `BurstCatalog`, and its module
  `s1bursts.database` is now `s1bursts.catalog`. It is a client for the CDSE
  burst catalog, and failures of that catalog raise `CatalogError`. The old
  names are gone without an alias.

  ```python
  # before
  from s1bursts import BurstDatabase
  from s1bursts.database import BurstDatabase

  db = BurstDatabase(file_manager=fm)
  # after
  from s1bursts import BurstCatalog
  from s1bursts.catalog import BurstCatalog

  catalog = BurstCatalog(file_manager=fm)
  ```
- **Breaking:** `BurstCollection.find_pairs()` takes `baseline` and
  `max_delta` by keyword only, and `max_delta` must be a duration: a
  `pandas.Timedelta`, `datetime.timedelta` or `numpy.timedelta64`. An integer
  `max_delta` was read as hours, so `find_pairs(12, 6)` read one integer as
  days and the other as hours; a number or string now raises `TypeError`. An
  integer `baseline` is still read as days, and a `baseline` given as a
  `datetime.timedelta` or `numpy.timedelta64` is now taken as is, where it
  raised a pandas `UserWarning` before. A float or string `baseline` raises
  `TypeError`. The default tolerance is unchanged, six hours.

  ```python
  # before
  bursts.find_pairs(12, 6)
  bursts.find_pairs(12, max_delta=3)
  # after
  bursts.find_pairs(baseline=12, max_delta=pd.Timedelta(hours=6))
  bursts.find_pairs(baseline=12, max_delta=pd.Timedelta(hours=3))
  ```
- **Breaking:** worker counts are called `max_workers` everywhere, as in
  `BurstCatalog.download()` and `SNAPConfig`. `ResolutionWeighter` takes
  `max_workers` instead of `n_workers`: replace
  `ResolutionWeighter(fm, n_workers=4)` with
  `ResolutionWeighter(fm, max_workers=4)`.
- **Breaking:** the `SNAPConfig` fields `gpt_threads_per_job` and
  `gpt_cache_ratio` are renamed `threads_per_job` and `cache_ratio`, so each
  field is spelled like the `SNAP_*` environment variable it defaults to. The
  variables `SNAP_THREADS_PER_JOB` and `SNAP_CACHE_RATIO` are unchanged.

  ```python
  # before
  SNAPConfig(gpt_threads_per_job=4, gpt_cache_ratio=0.5)
  # after
  SNAPConfig(threads_per_job=4, cache_ratio=0.5)
  ```
- **Breaking:** errors name the side that has to change: an invalid argument
  raises `ValueError`, an invalid environment raises `ConfigurationError`.
  - An out-of-range `SNAPConfig` argument, such as `max_workers=0`, a
    negative `max_ram_gb` or a `cache_ratio` above 1, raises `ValueError`
    instead of `ConfigurationError`, and is reported before `gpt` is looked
    for. So does a RAM budget too small for the workers when `max_ram_gb` or
    `max_workers` is passed; when both come from `SNAP_MAX_RAM_GB` and
    `SNAP_MAX_WORKERS` (or their defaults), it stays a `ConfigurationError`.
    Replace `except ConfigurationError:` around such a `SNAPConfig(...)` with
    `except ValueError:`.
  - An invalid `SNAP_*` environment variable, such as `SNAP_MAX_WORKERS=0`
    or `SNAP_CACHE_RATIO=1.5`, raises `ConfigurationError` naming the
    variable, from every call that reads the settings: `SNAPConfig()`,
    `prepare_dem_for_snap()`, and, since all settings are read together,
    `CDSEAuth()` and `BurstCatalog.download()`. It used to escape as pydantic's
    `ValidationError`, a `ValueError`: replace `except ValidationError:` or
    `except ValueError:` around those calls with `except ConfigurationError:`.
    `BurstCatalog.download()` raises it rather than recording it per burst as
    it does missing credentials, so the batch stops at once.
- **Breaking:** `BurstCatalog.search()` no longer takes `max_items` and
  returns every match. A query matching more than 1000 bursts, one catalog
  page, raises the new `s1bursts.errors.TooManyResultsError` (previously a
  `RuntimeError` above `max_items`) after a single request. Narrow the
  `TimeRange` or the AOI, or split the query and combine the results with `+`.
  Replace `catalog.search(query, max_items=500)` with `catalog.search(query)`.
- **Breaking:** `CDSEBurstClient.search_bursts()` no longer takes `top` and
  no longer follows next-link pages. It returns one page of up to 1000
  bursts; request `count=True` to tell whether more match. Replace
  `client.search_bursts(filters, top=100)` with
  `client.search_bursts(filters)` and take the first 100 of its `value`.
- **Breaking:** `SearchQuery(time_range=...)` accepts only a `TimeRange`, and
  `query.time_range` is that `TimeRange` rather than a `(start, stop)` tuple of
  Timestamps, so it can be passed on to `compose_*(time_range=...)`. A tuple
  raises `TypeError`. The tuple was a half-open pair of instants and dropped
  every acquisition on its end date, whereas a `TimeRange` keeps all its days.
  Replace `SearchQuery(time_range=("2022-01-01", "2023-01-01"))` with
  `SearchQuery(time_range=TimeRange("2022-01-01", "2022-12-31"))`.
- **Breaking:** every error s1bursts raises about its own domain is now an
  `S1BurstsError` from `s1bursts.errors`, so `except S1BurstsError` catches
  all of them. Invalid arguments still raise `ValueError` and `TypeError`,
  and a path you pass that does not exist still raises `FileNotFoundError`.
  - An unusable catalog response raises `CatalogError`, a `RuntimeError`. This
    covers `BurstCatalog.search()`, `CDSEBurstClient.search_bursts()` and
    `CDSEResponse.to_geodataframe()`. `CDSEResponse.is_complete()` without a
    count raised `AttributeError` before: replace `except AttributeError:`
    with `except CatalogError:`.
  - A missing or undownloadable PROJ grid in `prepare_dem_for_snap()` raises
    `TransformationGridError`, a `ConfigurationError` and a `RuntimeError`.
    `except RuntimeError:` still catches it. An `except ConfigurationError:`
    around `prepare_dem_for_snap()` now catches it too, where before it caught
    only the error for a `snap_bin` without `gpt`.
  - Missing CDSE credentials in `CDSEAuth()` raise `ConfigurationError`
    instead of `ValueError`: replace `except ValueError:` with
    `except ConfigurationError:`.
  - A burst-ID map that has not been downloaded, or cannot be read, raises
    `BurstMapUnavailableError`, a `ConfigurationError`, instead of
    `FileNotFoundError`: replace `except FileNotFoundError:` with
    `except BurstMapUnavailableError:`. It always stops a `BurstProcessor`
    run. Before, a pair whose secondary burst needed the map to stage its SLC
    was recorded as a failed item instead. A corrupt cached map, which raised
    `ValueError` or pyarrow's `ArrowInvalid`, raises it too and names the
    file to delete. A map archive without the mode's database is no longer
    reported as a map that was never downloaded.
  - A burst missing from the burst-ID map raises `BurstNotInMapError`, a
    `LookupError`, instead of `KeyError`: replace `except KeyError:` with
    `except BurstNotInMapError:`.
- **Breaking:** the relative orbit number and pass direction columns of
  `BurstCollection.df` and `BurstPairCollection.df` lose their underscore,
  matching the `Burst` attributes. A data frame passed to either constructor
  uses the new names too.

  ```python
  # before
  bursts.df[bursts.df._rel_orbit_number == 66]
  # after
  bursts.df[bursts.df.rel_orbit_number == 66]  # or bursts.filter(rel_orbit_number=66)
  ```
- **Breaking:** `BackscatterConfig` is split, so the config type selects the
  product. `RTCConfig` gives terrain-flattened gamma-0 backscatter (RTC) and
  holds the Terrain-Flattening settings `dem_oversampling` and
  `additional_overlap`; `GTCConfig` gives beta-0 backscatter that is geocoded
  only (GTC). `BackscatterConfig` remains as their abstract base, and
  `flatten_terrain` is gone. `create_backscatter()` of `BurstProcessor` and
  `SNAPClient` returns exactly `RTCProduct` or `GTCProduct` results, with RTC
  still the default, so the `cast` is no longer needed.

  ```python
  # before
  summary = processor.create_backscatter(
      bursts, crs=crs, config=BackscatterConfig(flatten_terrain=False)
  )
  gtc = cast("list[GTCProduct]", summary.products)
  processor.create_backscatter(bursts, crs=crs, config=BackscatterConfig(dem_oversampling=2.0))
  # after
  gtc = processor.create_backscatter(bursts, crs=crs, config=GTCConfig()).products
  processor.create_backscatter(bursts, crs=crs, config=RTCConfig(dem_oversampling=2.0))
  ```

  A config annotated as `BackscatterConfig` no longer matches an overload of
  `create_backscatter()`; annotate it as `RTCConfig | GTCConfig`.
- **Breaking:** the `crs` of every `BurstProcessor.create_*` and
  `SNAPClient.create_*` method, and of the products' `from_burst()` and
  `from_burst_pair()`, must be a WGS 84 / UTM zone, north or south. Anything
  else raises `ValueError`, in `BurstProcessor` before the first burst is
  processed. A non-UTM CRS used to fail later with
  `Could not extract CRS identifier from: …`, and a UTM zone on another datum,
  such as ETRS89 / UTM zone 32N, was written under the name of the WGS 84 zone
  and read back as that.

  ```python
  # before
  processor.create_coherences(pairs, crs="EPSG:25832")
  # after
  processor.create_coherences(pairs, crs="EPSG:32632")  # or crs=utm_zone_of(pairs[0])
  ```
- **Breaking:** the file names of SLC bursts and of RTC, GTC, coherence and
  interferogram products record the pass direction, as `A` or `D` after the
  relative orbit, and every SLC and product path moves. Composite file names
  are unchanged.

  ```text
  before: S1_SLC_066_030346_IW1_VV_20220105.zip
          S1_RTC_066_030346_IW1_VV_20220105_32N.tif
  after:  S1_SLC_066A_030346_IW1_VV_20220105.zip
          S1_RTC_066A_030346_IW1_VV_20220105_32N.tif
          S1_COH12_066A_030346_IW1_VV_20220105_20220117_32N.tif
  ```

  The products have a public `pass_direction` field, a required keyword
  argument, which `from_burst()` and `from_burst_pair()` take from the burst
  and `FileManager` reads back from the file name. So `compose_*_by_pass`
  never needs the burst-ID map, also for products found on disk, and
  `FileManager.find()` filters these four types and SLC bursts on
  `pass_direction`. The pass direction is part of a product's equality, and
  anything but `"Ascending"` or `"Descending"` raises `ValueError`, in a
  product and in a filter. A burst found with `FileManager.find(Burst)`
  carries the pass direction its file name records, so creating its products
  needs no burst-ID map; its footprint still comes from the map.

  ```python
  # before
  RTCProduct(30346, "IW1", crs, date, "VV")
  # after
  RTCProduct(30346, "IW1", crs, date, "VV", pass_direction="Ascending")
  ```

  Files written by earlier versions are not found under the new names, and
  `BurstCatalog.download()` downloads an SLC burst again unless its file is
  renamed first. `FileManager` warns once per instance and directory if
  `slc_dir` or `prod_dir` still holds some: when `find()` looks for SLC
  bursts or one of those four types, and when `get_slc_path()` resolves an
  SLC path, as every download does. To rename an archive of IW SLC
  bursts and products, look up each burst's pass direction in the burst-ID
  map:

  ```python
  import re

  from s1bursts import FileManager, download_burst_map
  from s1bursts.burstmap import create_s1_pass_direction_lut

  download_burst_map(mode="IW")
  directions = create_s1_pass_direction_lut(mode="IW")
  old_name = re.compile(r"(S1_(?:SLC|RTC|GTC|COH\d{2}|IFG\d{2}_CMPLX)_\d{3})(_(\d{6})_(IW\d)_.*)")
  fm = FileManager("data")
  for path in [*fm.slc_dir.glob("S1_SLC_*.zip"), *fm.prod_dir.glob("S1_*.tif")]:
      if match := old_name.fullmatch(path.name):
          letter = directions[(int(match[3]), match[4])][0]
          path.rename(path.with_name(f"{match[1]}{letter}{match[2]}"))
  ```
- **Breaking:** a composite is one product holding its layers: the
  measurement and its QA layers. Every composite has a keyword-only `layers`
  field, a frozenset of the new `s1bursts.product.Layer` (`MEASUREMENT`,
  `QA_NUM`, `QA_CQM`), defaulting to the measurement alone; pass composites
  have no `QA_CQM`. The QA classes `RTCLRWQA`, `CohLRWQA`, `RTCPassQA`,
  `GTCPassQA` and `CohPassQA` and the `QaMetric` enum are removed.
  - `compose_rtc_lrw()` and `compose_coherence_lrw()` always return the
    composite, and `compose_*_by_pass()` a dict of composites by pass
    direction; with `write_qa=True` the composite holds the QA layers too.
    Unless every requested layer exists already, all of them are written, so
    a composite and its QA layers always come from one run. Before, an
    existing measurement was kept when only its QA layers were missing.
  - `FileManager.get_product_path()` and `product_exists()` take a `layer`,
    the measurement by default, to locate a QA layer's file. A layer's value,
    such as `"QA-NUM"`, works too.
  - The `find_*_qa()` methods are removed. `FileManager.find()` returns one
    composite per identity, holding the layers whose files it found. That
    includes a composite whose measurement file is missing and only QA layers
    are on disk; keep the complete ones with
    `[c for c in found if Layer.MEASUREMENT in c.layers]`.
  - The `pass_direction` of a pass composite, and a `pass_direction`
    filter, must be `"Ascending"` or `"Descending"`; anything else raises
    `ValueError`, where a finder used to return `[]`.
  - `product_tags.qa_metadata()` takes the composite and the layer instead of
    a QA product.

  File names, tags and the keys of a per-layer `rasterio_profile`, such as
  `"RTC_LRW_QA-NUM"`, are unchanged.

  ```python
  # before
  comp, (qa_num, qa_cqm) = weighter.compose_rtc_lrw(rtc, cell, write_qa=True)
  fm.get_product_path(qa_cqm)
  comp, qa_num = weighter.compose_rtc_by_pass(rtc, cell, write_qa=True)["Ascending"]
  fm.find_rtc_lrw_qa(mgrs_code="32TMS03", qa_metric=QaMetric.CQM)
  # after
  comp = weighter.compose_rtc_lrw(rtc, cell, write_qa=True)
  fm.get_product_path(comp, Layer.QA_CQM)
  comp = weighter.compose_rtc_by_pass(rtc, cell, write_qa=True)["Ascending"]
  [c for c in fm.find(RTCLRWComposite, mgrs_code="32TMS03") if Layer.QA_CQM in c.layers]
  ```
- **Breaking:** compositing failures that belong to the cell's data are
  `CompositeError`s, so a loop over cells can tell them from a wrong call.
  Each keeps its builtin base, so `except ValueError:` still catches the first
  two.
  - `EmptyCompositeError` is now a `CompositeError` (still a `ValueError`).
  - The new `InconsistentInputsError`, a `ValueError`, replaces the plain
    `ValueError` for an input on another CRS, pixel size or grid than the
    cell, inputs made with different DEMs, and an observation without a
    matching scattering area.
  - The new `CompositeInputError`, an `OSError`, replaces the
    `FileNotFoundError` for a missing input and rasterio's `RasterioIOError`
    for an unreadable one.

  An invalid argument still raises a plain `ValueError`, which
  `CompositeError` does not catch. That now includes an empty `rtc_products`
  in `compose_coherence_lrw()`, which raised the scattering-area error.

  ```python
  # before
  except EmptyCompositeError: ...
  except (ValueError, FileNotFoundError, RasterioIOError): ...
  # after
  except CompositeError as err: ...
  ```
- **Breaking:** `compose_rtc_by_pass()`, `compose_gtc_by_pass()` and
  `compose_coherence_by_pass()` behave like the LRW methods:
  - Empty input raises `ValueError` instead of returning `{}`.
  - A cell that no input reaches raises `EmptyCompositeError` and writes
    nothing, instead of writing all-NaN composites. A pass direction that no
    input reaches is left out of the returned dict and gets no file.
  - Only orbit-date observations with a valid pixel in the cell count in
    `CONTRIBUTING_OBSERVATIONS` and in `FIRST_ACQUISITION_DATE` and
    `LAST_ACQUISITION_DATE`, which used to count every input of the direction.
  - A direction whose files all exist is skipped before its inputs are read,
    so a resumed run no longer reads every input again for each finished cell.
  - With `overwrite=True`, a direction, or for any `compose_*` method a cell,
    that no input reaches has the files of an earlier composite of the same
    identity deleted, so no stale composite outlives the rerun.

  ```python
  # before: {} for no products, all-NaN files for products that miss the cell
  composites = weighter.compose_rtc_by_pass(products, cell)
  # after
  try:
      composites = weighter.compose_rtc_by_pass(products, cell) if products else {}
  except EmptyCompositeError:
      composites = {}
  ```
- **Breaking:** every `ResolutionWeighter.compose_*` method now defaults to
  `aggregation="mean"` instead of `"median"`. Pass `aggregation="median"` to
  keep the previous composites.
- **Breaking:** `FileManager` finds products with one method,
  `find(product_type, *, time_range=None, **filters)`, instead of a
  `find_*` method per type. The filters, like `time_range`, are keyword-only,
  where the old methods also took them by position, as in
  `find_rtc_lrw_products("32TMS03")` or `find_slc_bursts(66, 30346)`: name
  each one, as in `find(RTCLRWComposite, mgrs_code="32TMS03")` or
  `find(Burst, rel_orbit_number=66, burst_id=30346)`. Each type is filtered on
  the fields its file name records, typed `Polarization`, `SubSwath` and
  `PassDirection` as in `SearchQuery`, and `time_range` takes a `TimeRange`,
  which none of the old methods could filter on. A filter the type does not
  have, or a value no file name holds, such as `polarization="vv"`,
  `sub_swath="IW4"` or a non-UTM `crs`, raises `ValueError` where the old
  methods returned `[]`; a `time_range` that is not a `TimeRange` raises
  `TypeError`. `None` still means no filter.

  | Removed method | Replacement | Filters |
  | --- | --- | --- |
  | `find_slc_bursts(...)` | `find(Burst, ...)` | `rel_orbit_number`, `burst_id`, `sub_swath`, `polarization` |
  | `find_rtc_products(...)` | `find(RTCProduct, ...)` | those of `Burst`, `crs`, `pass_direction` |
  | `find_gtc_products(...)` | `find(GTCProduct, ...)` | as `RTCProduct` |
  | `find_coherence_products(...)` | `find(CoherenceProduct, ...)` | as `RTCProduct`, `baseline_days` |
  | `find_interferogram_products(...)` | `find(InterferogramProduct, ...)` | as `CoherenceProduct` |
  | `find_rtc_lrw_products(...)` | `find(RTCLRWComposite, ...)` | `mgrs_code`, `polarization` |
  | `find_coh_lrw_products(...)` | `find(CohLRWComposite, ...)` | `mgrs_code`, `polarization`, `baseline_days` |
  | `find_rtc_pass_composites(...)` | `find(RTCPassComposite, ...)` | `mgrs_code`, `polarization`, `pass_direction` |
  | `find_gtc_pass_composites(...)` | `find(GTCPassComposite, ...)` | as `RTCPassComposite` |
  | `find_coh_pass_composites(...)` | `find(CohPassComposite, ...)` | as `RTCPassComposite`, `baseline_days` |

  `time_range` keeps SLC bursts, RTC and GTC products acquired on one of its
  days, coherence and interferogram products whose primary acquisition is,
  as `BurstPairCollection.starting_within()` does, and composites whose
  whole period, `start_date` through `end_date`, lies within it.

  ```python
  # before
  rtc = [p for p in fm.find_rtc_products(burst_id=30346, polarization="VV") if p.date in window]
  bursts = fm.find_slc_bursts(burst_id=30346)
  coh = fm.find_coherence_products(baseline_days=12, pass_direction="Ascending")
  lrw = fm.find_coh_lrw_products(mgrs_code="32TMT64", baseline_days=12)
  # after
  from s1bursts.burst import Burst
  from s1bursts.product import CoherenceProduct, CohLRWComposite, RTCProduct

  rtc = fm.find(RTCProduct, burst_id=30346, polarization="VV", time_range=window)
  bursts = fm.find(Burst, burst_id=30346)
  coh = fm.find(CoherenceProduct, baseline_days=12, pass_direction="Ascending")
  lrw = fm.find(CohLRWComposite, mgrs_code="32TMT64", baseline_days=12)
  ```
- **Breaking:** composites are always written in linear units. The
  `decibel` argument of `compose_rtc_lrw()`, `compose_rtc_by_pass()` and
  `compose_gtc_by_pass()` and the `cqm_decibel` argument of
  `compose_rtc_lrw()` and `compose_coherence_lrw()` are removed, as are the
  same parameters of `product_tags.rtc_lrw_metadata()`,
  `rtc_pass_metadata()`, `gtc_pass_metadata()` and `qa_metadata()`. Convert
  after reading instead; the `BACKSCATTER_CONVERSION_EQ` and
  `CQM_CONVERSION_EQ` tags state the equation. A composite written in decibels
  by an earlier version counts as made with other settings, see below.

  ```python
  # before
  comp = weighter.compose_rtc_lrw(rtc, cell, decibel=True, write_qa=True, cqm_decibel=True)
  # after
  comp = weighter.compose_rtc_lrw(rtc, cell, write_qa=True)
  with rasterio.open(fm.get_product_path(comp)) as src:
      power = src.read(1)
  gamma0_db = 10 * np.log10(np.where(power > 0, power, np.nan))  # NaN where not positive, as before
  # likewise 10 * np.log10(cqm) for the QA-CQM layer
  ```
- **Breaking:** an existing product is skipped only if it records the
  requested settings. With `overwrite=False`, `BurstProcessor.create_*`,
  `SNAPClient.create_*` and every `ResolutionWeighter.compose_*` method
  returned an existing file by its name alone, and names carry no settings:
  change `grid_spacing`, `multilook`, the DEM, the coherence `window`, the
  `geobox` grid, `aggregation` or `clip_max_resolution`, rerun, and the old
  products came back as skipped. Such a rerun now raises the new
  `s1bursts.errors.ProductSettingsMismatchError`, naming each differing
  setting with its recorded and requested value; the error carries the file as
  `path` and the settings as `differences`.
  - Per-burst products compare their `GRID_SPACING`, `DEM`, `MULTILOOK_*`,
    `COHERENCE_WINDOW_*` and interferogram `*_PHASE_REMOVED` tags; composites
    their grid (CRS, transform and shape) and `COMPOSITING_PARAMETERS`, in
    every requested layer, and refuse a layer stored in decibels. A composite
    is still skipped before any input is read, so what it inherits from its
    inputs, such as their DEM, is not compared.
  - The error is directly an `S1BurstsError`, not a `CompositeError`, so a
    loop over cells does not pass over it. `BurstProcessor.create_*` check
    every existing product before submitting any work, so it is raised before
    any GPT run, also with `continue_on_error=True`, and the pass composites
    check every direction before building any.
  - An existing file that cannot be read raises its read error, with a note
    naming the file and that `overwrite=True` rebuilds it, instead of being
    skipped.
  - A product whose settings are not recorded is skipped as before: one
    written before s1bursts 0.3.0, and a per-burst product whose DEM is
    external without `dem_name`, on either side. Set
    `GeocodingConfig.dem_name` for an external DEM so that it can be
    compared. A product made by another SNAP version than the running one is
    skipped too, since SNAP 13 and 14 outputs should not be mixed in one time
    series.
  - Such a skip is recorded in the new `OkResult` fields `unchecked`
    (`"untagged"` or `"unknown_dem"`) and `other_snap_version`, counted in
    the new `OperationSummary` fields `n_unchecked` and
    `n_other_snap_version`, which `print_summary()` shows, and logged by
    `SNAPClient.create_*`. `BurstProcessor.create_*` warn once per call for
    each kind, and `ResolutionWeighter.compose_*` once per call for unchecked
    composites.
  - Composites written by 0.7.0 with its default `aggregation="median"`
    record it, so rerunning them with the new default, the mean, raises. Pass
    `aggregation="median"` to keep them, or `overwrite=True` to rebuild them.

  ```python
  # before: the 20 m products came back as skipped
  processor.create_backscatter(bursts, crs=crs, config=RTCConfig(grid_spacing=10.0))
  # after: raises ProductSettingsMismatchError: '...tif' exists but was made with other
  # settings: GRID_SPACING (grid_spacing): recorded '20 m', requested '10 m'. ...
  processor.create_backscatter(bursts, crs=crs, config=RTCConfig(grid_spacing=10.0), overwrite=True)
  # or keep both, each in its own directory
  fm_10m = FileManager("data", prod_dir="data/s1-prod-10m")
  ```
- `pyarrow>=17` is a direct dependency, to recognise a corrupt burst-ID map.
  freezebase already required it, so nothing new is installed.
- The default `SNAP_MAX_RAM_GB` is 8 GB instead of 4 GB to leave more heap for
  SNAP. Set `SNAP_MAX_RAM_GB=4` to keep the old budget.
- `OperationSummary.print_summary()` groups failures by error type and the
  first line of the message, with a count and a few example items each, so an
  outage prints one row such as `2318 × BurstDownloadError: 503 …` instead
  of one per burst.

### Fixed

- **Breaking:** `QA-CQM` of `median` (previously the default) and quantile
  composites overstated the local resolution, on an Alpine tile by a median
  factor of 1.7 and up to 4. It was $1 / \operatorname{wquantile}_q(A;\, W)$ and is now
  $1 / \sum_i W_i A_i$ (Small et al. 2022, eq. 6; CEOS-ARD SAR v1.3.1,
  eq. A5.2) under every aggregation, the same map as the `mean` composite of
  the same inputs. Under `median` or a quantile it describes the resolution
  weighting, not the acquisitions the composite value was taken from.
  Regenerate `QA-CQM` built with `median` or a quantile; `mean` is unaffected.
- **Breaking:** composites are now reproducible in burst overlaps. Where two
  bursts of one acquisition overlap, every `ResolutionWeighter.compose_*`
  method takes the pixel from the burst with the lowest
  `(burst_id, sub_swath)`. Previously it took whichever burst came first in
  the input list, which was non-deterministic for runs with multiple workers.
  Therefore, two runs over the same products could differ in the composite
  and `QA-CQM`. The burst choice is now deterministic. Composites made with
  earlier versions remain valid and need not be regenerated.
- **Breaking:** a burst that neither endpoint of `BurstCatalog.download()`
  could download is recorded with one error naming both failures, with the
  primary endpoint's error as its `__cause__`. Previously only the fallback's
  error was kept, so an OData outage was reported as missing S3 credentials.
  The error is a `BurstNotFoundError` when both endpoints found the burst
  missing and a `BurstDownloadError` otherwise; an endpoint's own error type,
  such as `HTTPError`, is no longer what the summary holds.

  ```python
  # before
  err = summary.failed_items[0].error
  # BurstDownloadError("CDSE S3 credentials not configured ... endpoint='odata' ...")
  # after
  # BurstDownloadError("OData: HTTPError: 503 Server Error ...; S3: CDSE S3 credentials not configured ...")
  err.__cause__  # the OData HTTPError
  ```
- **Breaking:** a burst counts as missing at the source only when both
  endpoints say so. An HTTP 404 from the OData endpoint is now a
  `BurstNotFoundError`, and a `BurstCatalog` caches a burst as unavailable
  only when both endpoints' failures are not-found errors. Previously one
  `BurstNotFoundError` was enough, so an OData 503 together with a product not
  yet on S3 marked the burst missing for the catalog's lifetime, and
  `raise_on_failure(ignore=BurstNotFoundError)` ignored it.
- **Breaking:** credentials that CDSE rejects (HTTP 400, 401 or 403 on the
  token request) raise `ConfigurationError` from `CDSEAuth`, quoting the
  identity server's reason. With OData as the chosen endpoint they stop
  `BurstCatalog.download()` after one login attempt; with OData as the
  fallback each burst records them as a `BurstDownloadError`, so S3 keeps
  working, still after one login attempt per call. Previously the `HTTPError`
  was recorded for every burst, each of which logged in again.

  ```python
  # before
  catalog.download(bursts).failed_items  # one HTTPError 401 per burst
  # after
  catalog.download(bursts)  # raises ConfigurationError: CDSE rejected the username and password
  ```
- Any other failed token request raises an error that keeps neither the
  request nor the response, as cause or context. Previously the `HTTPError`
  recorded for a burst kept the request, whose body held the password (or,
  for a refresh, the refresh token).
- **Breaking:** item-local data errors are recorded per item as `BurstError`
  subclasses instead of aborting the batch: a rasterio error or `OSError`
  while staging the input or writing the product of a `BurstProcessor.create_*`
  item becomes a `SnapProcessingError`, and a malformed annotation,
  calibration, noise, RFI or manifest file of an S3 parent product (an XML
  parse error, or a value that is missing or does not convert) a
  `BurstDownloadError`, each with the original as `__cause__`. A `FileNotFoundError` or `RasterWriteError`
  that `BurstProcessor` already recorded is now wrapped the same way.

  ```python
  # before
  summary.raise_on_failure(ignore=RasterWriteError)
  # after
  summary.raise_on_failure(ignore=SnapProcessingError)
  ```
- **Breaking:** an `OSError` meaning that the disk is full, over quota or
  read-only (`ENOSPC`, `EDQUOT`, `EROFS`) stops `BurstCatalog.download()` and
  the `BurstProcessor.create_*` methods, also when it is the cause of another
  error such as `RasterWriteError`. GPT and GDAL report such a disk only in
  their message, without an errno, so an error whose message, or the message
  of an error it was raised from, says so (`No space left on device`, `Disk
  quota exceeded`, `Read-only file system`) stops the batch too, as does a
  failed GPT run whose output says so. Previously it
  was recorded for every remaining item.
- A batch stops promptly. When Ctrl-C or an error ends
  `BurstCatalog.download()` or a `BurstProcessor.create_*` method, the queued
  items are cancelled and only the running ones finish, instead of the whole
  queue running before the error surfaced. This includes the first failure
  under `continue_on_error=False`, which `download()` previously raised only
  after every queued burst had been attempted. The error carries a note such
  as `Stopped with 12 of 3000 bursts finished; SLCs already downloaded are
  kept, and a rerun without overwrite skips them.`
- `BurstProcessor` summaries list their results in input order whatever
  `max_workers` is, instead of in the order the threads finished.
- Products found with `FileManager.find()` keep the relative orbit
  recorded in their file name, so the LRW composites no longer need the
  burst-ID map for them.
- Comparing or hashing per-burst products no longer looks up the relative orbit,
  so it works without the burst-ID map. The burst ID already identifies a burst,
  so the orbit adds nothing.

## [0.7.0] - 2026-10-26

### Added

- Beta-0 pass composites: `ResolutionWeighter.compose_gtc_by_pass()` averages
  GTC products per pass direction, as `compose_rtc_by_pass()` does for RTC
  (`S1_GTC_COMP_A_…` / `S1_GTC_COMP_D_…`, each with a QA layer).
- Python 3.15 support.

### Changed

- **Breaking:** requires freezebase 0.10.
- **Breaking:** an LRW composite for a cell that no burst covers with valid
  pixels raises `EmptyCompositeError` and writes nothing. Before, it wrote an
  all-NaN composite. To skip such cells, catch `EmptyCompositeError`.
- LRW composites count only bursts with valid pixels in the cell. A burst
  whose footprint merely touches the cell no longer adds to
  `CONTRIBUTING_OBSERVATIONS` or to the first and last acquisition dates.
- **Breaking:** failed downloads, SNAP runs and product writes are still
  recorded per burst, but any other error now stops the run, since it points
  to a bug. Checks for `isinstance(result.error, RuntimeError)` keep working.
- The burst-ID map cache is a third smaller (128 → 85 MB for IW). Existing
  caches stay valid.

### Fixed

- **Affects results:** with SNAP 14.0.2 or newer, `prepare_dem_for_snap()`
  shifted external DEMs by half a pixel (15 m for a 30 m DEM), because SNAP
  now corrects that offset itself. The shift is applied only where SNAP still
  needs it, so SNAP 13 results are unchanged. Prepared DEMs are rebuilt once;
  pass `snap_bin` if SNAP is not found through `SNAP_BIN` or `PATH`.
- When CDSE rejects a login token early, s1bursts logs in again instead of
  failing every later OData download.
- An OData download that returns something other than a file, such as a
  login page, is recorded as that burst's failure instead of stopping the
  batch.

## [0.6.0] - 2026-09-24

### Changed

- **Breaking:** `BurstCollection.find_pairs(12)` now selects 12 d ± 6 h
  (was ± 24 h). The old default mixed 12- and 13-day pairs that
  `ResolutionWeighter` refused to composite.
- **Breaking:** a `max_delta` of 12 hours or more, and a `baseline` that is
  not a whole number of days, now raise `ValueError`. Replace
  `find_pairs(12, max_delta=24)` with e.g. `find_pairs(12, max_delta=6)`.
  Coherence names round pair spacing to whole days; only a tolerance below
  12 hours guarantees the requested baseline.

### Fixed

- `find_pairs()` stopped widening the lag when every spacing exceeded the
  baseline, dropping in-tolerance pairs at later lags. It now stops only
  when every spacing exceeds `baseline + max_delta`.

## [0.5.1] - 2026-09-21

### Changed

- Requires freezebase 0.8 (`freezebase[s3]>=0.8,<0.9`).

### Fixed

- Products and composites written to S3 are uploaded through freezebase's
  s3fs path instead of GDAL's `/vsis3/`, which a gateway could refuse
  without raising an error. A backscatter or coherence product that fails to
  upload is now a failed operation: the summary counts it in `n_failed`, where
  it used to be logged as created.
- A burst `.zip` extracted to a remote `save_path` is uploaded with
  `upload_object`, which checks the stored size.

## [0.5.0] - 2026-09-07

### Added

- `TimeRange`, an inclusive window of calendar days. `SearchQuery` accepts it
  as `time_range` and searches every acquisition on any of its days, so the
  caller no longer translates an inclusive last day into an exclusive stop. A
  pair of instants is still accepted and is now read as `[start, stop)`, which
  differs from before only for an acquisition at exactly the stop instant.
- `time_range` on every `ResolutionWeighter.compose_*()` method: the same
  `TimeRange` the search was made with. The product is named and tagged with
  its days, so all tiles built from one search carry the same dates. The
  observed span moves to the new `FIRST_ACQUISITION_DATE` and
  `LAST_ACQUISITION_DATE` tags, which are always written. Each pass composite
  carries the span of its own direction. An input outside the window is
  refused. Without `time_range` nothing changes.

## [0.4.0] - 2026-09-07

### Added

- Exact band statistics (`STATISTICS_MINIMUM`, `_MAXIMUM`, `_MEAN`, `_STDDEV`,
  `_VALID_PERCENT`) on every composite and QA layer, written at creation. New
  `product_tags.band_statistics()`.
- `ResolutionWeighter(checksum=True)` records a multihash-encoded SHA-256 of
  every written file in `ResolutionWeighter.checksums`, keyed by output path.
- `BurstDatabase.search()` follows the catalog's next-link pages, so
  `max_items` above the 1000 items CDSE serves per request now works.
- Coherence and interferogram products carry a `LAYER_NAME` tag on every band.

### Fixed

- Per-burst coherence and interferogram products tagged `TEMPORAL_BASELINE_DAYS`
  with the truncated timestamp difference, so a pair whose secondary was
  acquired a fraction of a second earlier in the day was tagged 11 days under a
  `S1_COH12` name. The tag now uses the rounded baseline the filename uses.
- Missing CDSE S3 credentials aborted a whole download on the S3 endpoint
  instead of being recorded per burst, so the OData fallback never ran.
- A quantile `aggregation` outside `(0, 1)` was accepted and silently returned
  the minimum or maximum; every `compose_*` call now rejects it.
- `compose_*` rounded the read window to whole pixels, silently shifting an
  input whose pixel corners do not coincide with the target grid's; such an
  input is now refused like one in another CRS or at another pixel size.
- The HyP3 validation campaign no longer fails under Python 3.14 on an
  unclosed file left behind by `hyp3_sdk`'s download helper.

### Changed

- **Breaking:** `median` and quantile aggregation now exclude observations
  carrying zero resolution weight, which were previously interpolated into the
  result. `mean` is unaffected. Regenerate composites built with `median` (the
  default) or a quantile.
- **Breaking:** `QA-CQM` now applies the composite's own aggregation to the
  scattering area instead of always the weighted mean, and ignores
  observations that did not enter the composite. `mean` is unchanged.
  Regenerate `QA-CQM` products built with `median` or a quantile.
- **Breaking:** per-burst products name their bands in snake case, and the
  GeoTIFF band description, the graph's band and the `LAYER_NAME` band tag now
  agree: RTC `gamma0` and `scattering_area` (was `Gamma0`, `simulatedImage`),
  GTC `beta0` (was `Beta0`), coherence `coherence` (was `coh`), interferogram
  `ifg_real`, `ifg_imag`, `intensity_primary`, `intensity_secondary` (was
  `ifg_i`, `ifg_q`, `intensity_1`, `intensity_2`). Code reading bands by
  description must use the new names.
- The area SNAP writes as `simulatedImage` is called the *local scattering
  area* throughout, the CEOS-ARD term; `s1bursts` 0.3.0 and Small et al. use
  *local contributing area*, OPERA uses *area normalization factor*; all terms
  are the same quantity.
- Requires `freezebase[s3]>=0.7,<0.8` for `write_cog(checksum=...)`.

## [0.3.0] - 2026-08-31

### Added

- **GeoTIFF metadata on every product.** Composites, QA layers and per-burst
  products now carry processing and interpretation metadata based on CEOS-ARD
  PFS SAR v1.3.1. Full conformance is not claimed.
- **SNAP 14 support.** Graphs select the SNAP 13 or 14 coregistered-band names
  from the installed version. An unknown version warns and uses SNAP 13 names.
- `CoherenceConfig.correct_coherence_bias` and `remove_local_phase_ramp`, both
  defaulting to `False`. Enabling either requires SNAP 14 or newer.
- **Python 3.14 support**, verified in CI.
- `prepare_dem_for_snap()` can convert compound-CRS orthometric DEMs to WGS 84
  ellipsoidal heights while applying SNAP's pixel-shift compensation. Set
  `download_missing_grids=True` to download required PROJ grids.
- Runtime version access through `s1bursts.__version__`.
- Runtime validation for search fields, SNAP graph settings, resampling methods,
  and burst-pair timing constraints.
- Named AWS profiles through `CDSE_S3_PROFILE` (`CDSES3Settings.profile`). A
  configured profile takes precedence over explicit CDSE S3 keys and ambient
  AWS environment credentials.
- Per-product composite codec profiles through
  `ResolutionWeighter.rasterio_profile`, with an optional `"default"` profile.
  Flat profiles retain their previous behavior.

### Fixed

- **SNAP terrain-correction masks are now consistent across SNAP 13 and 14.**
  Graphs use a finite internal no-data marker, preventing interpolation over
  radar-geometry fill pixels.
- **Areas without DEM coverage no longer read as zero.** Per-burst products
  convert all-band zero fill to no-data when `replace_zeros_with_nan` is enabled.
- `SNAPConfig` now warns when `TMPDIR` is set but unusable.
- `compose_*` rejects inputs that do not sit on the output grid.
- Coherence compositing rejects inputs with differing temporal baselines.
- Compositing warns when inputs report different SNAP versions.
- `prepare_dem_for_snap(download_missing_grids=True)` now refreshes the PROJ
  grid cache after downloading and attempts only installable open-license grids.
- The compensated-DEM cache now uses source content and conversion-version keys,
  preventing collisions between sources with the same basename.
- Extracted burst manifests preserve the annotation's coordinate precision.
- `BurstCollection` no longer writes through DataFrame views when sliced.
- Reading GCP-georeferenced SLC TIFFs no longer emits a spurious
  `NotGeoreferencedWarning`.
- `SNAPConfig` and `SNAPSettings` reject resource budgets that cannot run a job
  and suggest a valid configuration.

### Changed

- **Breaking:** `orbit_direction` is renamed to `pass_direction` throughout, and
  its values are now `"Ascending"` and `"Descending"` (instead of `"ASCENDING"`
  and `"DESCENDING"`). This aligns the wording with the
  [CEOS-ARD SAR metadata specification](https://ceos.org/ard/) v1.3.
- **Breaking:** The `QA-NUM` layer of an LRW composite now counts the
  observations that carry a non-zero resolution weight, rather than those
  holding a finite value, matching the CEOS-ARD *Contributing Observations
  Image*. Regenerate coherence QA products to update their counts. RTC results
  are unchanged, and unweighted pass composites still count finite values.
- **Breaking:** `compensate_dem_for_snap()` is renamed to
  `prepare_dem_for_snap()` to reflect that it can now apply both the pixel
  shift and vertical datum conversion of an external DEM for SNAP.
- The area SNAP writes as `simulatedImage` is now called the *local
  contributing area*, following CEOS-ARD terminology.
- Requires `freezebase[s3]>=0.6.1,<0.7` for product metadata, composite codec
  profiles, and context handling of burst downloads.
- **Breaking:** `GeocodingConfig` is now abstract and cannot be instantiated. It
  remains the base for shared Terrain-Correction parameters; concrete configs
  implement `to_xml()`.
- **Breaking:** `build_coherence_magnitude_graph()` is renamed to
  `build_coherence_graph()`.
- The first-party dependency `freezecore` was renamed to `freezebase` and is now
  resolved from PyPI. Imports move from `freezecore.*` to `freezebase.*`, and
  its cache environment variable is now `FREEZEBASE_CACHE`.

### Removed

- **Breaking:** the complex coherence product and everything that existed only to
  produce it: `CoherenceConfig.write_complex`, `CoherenceConfig.map_looks`,
  `CoherenceComplexProduct`, `FileManager.find_coherence_complex_products()`, and
  `build_coherence_complex_ifg_graph()`. Coherence creation now always returns a
  single-band `CoherenceProduct`.
- **Breaking:** `s1bursts.dimap` (`BeamDimapDataset`) and `s1bursts.utils`
  (`sliding_window_mean`), which served only complex coherence processing.
- **Breaking:** `build_terrain_correction_graph()`, and with it
  `GeocodingConfig.to_xml()`. Remaining products use single-shot graphs with
  Terrain-Correction built in.

## [0.2.0] - 2026-07-31

### Added

- `compensate_dem_for_snap()`, which makes an external DEM sample at the position its
  georeferencing states. SNAP reads a DEM passed as `externalDEMFile` half a DEM
  pixel towards the lower right. The helper writes a cached copy whose geotransform
  origin is moved onto the centre of the first pixel; elevations pass through untouched.
  Verified against SNAP 13.0.0.
- `SNAP_DEM_CACHE` setting (`SNAPSettings.dem_cache`, default `~/.s1bursts/dem`),
  the directory `compensate_dem_for_snap()` caches converted DEMs in.

### Changed

- **Breaking:** the settings classes no longer read a `.env` file; they read the
  process environment only. To parse an `.env` file, call `dotenv.load_dotenv()`
  or export the variables (`uv run --env-file .env`). `get_settings()` re-reads
  the environment on every call, so a late `load_dotenv()` is picked up.

## [0.1.1] - 2026-07-31

### Fixed

- Fixed SNAP memory leakage: `gpt` now runs with `java.io.tmpdir` pointed inside
  the per-job SNAP userdir. SNAP's jblas extracts 30 MB of native libraries per process
  and deletes them from a shutdown hook that `SIGKILL` skips. A run whose workers
  were OOM-killed therefore leaked ~30 MB each until `/tmp` filled. Cleanup now belongs
  to `snap_worker_userdir`, which removes the tree however `gpt` exited.

## [0.1.0] - 2026-07-27

### Added

- `Burst.mode` property deriving the acquisition mode (`IW`/`EW`) from the
  sub-swath.
- `download_burst_map()` to explicitly fetch the SAR-MPC burst-ID map, verified against a pinned SHA-256.
- `CoherenceConfig.map_looks`, an optional coherent box average applied to the
  *geocoded* complex coherence before the magnitude is taken. Requires
  `write_complex=True`; the product written is a single-band `COH`, so the
  compositing stage is unaffected. Averaging on the map grid gives a uniform
  ground footprint regardless of local incidence, and averaging coherently
  rather than in magnitude is what reduces the small-window estimator bias.
- `cqm_decibel` on `ResolutionWeighter.compose_rtc_lrw()` and
  `compose_coherence_lrw()` writes the QA-CQM map in decibels, consistent with
  Small et al. (2021,
  [doi:10.1109/TGRS.2021.3055562](https://doi.org/10.1109/TGRS.2021.3055562)).

### Changed

- **Breaking:** SNAP's `grSquarePixel` is now pinned off on the `Multilook`
  operator and is not exposed. With it enabled (SNAP's own default, and this
  library's behaviour until now), SNAP derives `nAzLooks` from `nRgLooks` and
  the local geometry and **silently discards the requested azimuth looks**, so a
  `(5, 1)` multilook was applied as `(5, 2)` for Sentinel-1 IW. Non-square pixels
  are correctly resampled to `grid_spacing` in geocoding (terrain correction).
- **Breaking:** the graph configs now reject unknown fields (`extra="forbid"`)
  instead of ignoring them.
- **Breaking:** the coherence graphs no longer contain a `Multilook` node and
  `CoherenceConfig.multilook` is removed. Multilooking saved no runtime there —
  it can only sit after the coregistration and interferogram stages that
  dominate (162 s against 165 s without it) — and it had no single sensible
  placement: the magnitude graph applied it *after* `Coherence`, averaging
  coherence magnitudes, while the complex graph applied it *before* the
  estimator, enlarging it, so one name meant two different operations. Use
  `window` to set the looks and `map_looks` to average further after terrain
  correction. `BackscatterConfig.multilook` is unaffected: there the step
  precedes terrain flattening and cuts runtime roughly ninefold.
- All four graph configs now document every parameter.
- **Breaking:** the root namespace is narrowed to what a pipeline is *built*
  from: `FileManager`, `BurstDatabase`, `BurstProcessor`, `ResolutionWeighter`,
  `SearchQuery`, `SNAPConfig`, the four graph configs, and
  `download_burst_map()`. What a pipeline *produces* (bursts, products, summaries, ...)
  now imports from the module it belongs to. The typical search → download → process →
  composite flow needs none of them, so most code should need no changes.
- **Breaking:** `ProcessingConfig` is renamed to `SNAPConfig` and no longer
  carries `file_manager`; every remaining attribute is a SNAP parameter with a
  default. `BurstProcessor` and `SNAPClient` now take the `FileManager` as their
  first argument and the config as an optional second one, matching
  `ResolutionWeighter`: `BurstProcessor(file_manager)` replaces
  `BurstProcessor(ProcessingConfig(file_manager=fm))`.
- Burst-ID map is now opt-in. Relative orbit number, orbit
  direction, and footprint geometry are propagated from the CDSE search response
  and from SLC filenames onto `Burst`/products, so the search → download →
  process → composite path no longer downloads anything. The ~500 MB SAR-MPC map
  is consulted only for a burst lacking that metadata (e.g. hand-constructed),
  and is no longer fetched automatically — call `download_burst_map()` to opt in.
- `py.typed` marker so downstream type checkers use the inline annotations.
- Packaging metadata: project URLs, keywords, and Development-Status / Topic /
  Typing classifiers.
- GitHub Actions CI (lint, type-check, test matrix on 3.12/3.13, a lowest-bound
  dependency job, and a build + wheel smoke-test), a scheduled/manual CDSE +
  SNAP integration workflow, and a Trusted-Publishing release workflow.
- Licensing metadata modernized to PEP 639 (`license = "MIT"` +
  `license-files`).
- `numpy` lower bound raised to `>=2.0` to match the freezecore floor, and
  `requests` given a `>=2.28` floor.

### Fixed

- **EW support**: `rel_orbit_number`, `orbit_direction`, `geometry`, and
  `full_burst_id` now route to the EW burst map for `EW1`–`EW5` bursts instead
  of raising `KeyError` against the IW-only table.
- **S3-only credentials**: `BurstDatabase.download(endpoint="s3")` no longer
  constructs a CDSE OData authentication handler up front, so downloads work
  with only `CDSE_S3_*` credentials configured. OData credentials are requested
  lazily and only when an OData request (primary or fallback) actually runs.
- Non-integer standard-grid origin now uses true division
  (`grid_spacing / 2`), so a 7.5 m grid yields a 3.75 m pixel-centre origin.
- Downloaded SAFE annotation/calibration/noise/RFI/manifest XML is parsed with a
  hardened parser (no entity resolution, no network access), matching the graph
  parser.
- Relative auxdata cache paths are resolved before symlinking into per-worker
  SNAP userdirs, avoiding broken symlinks.
- `BurstFootprintMixin.rel_orbit_number`, `orbit_direction`, `geometry`, and
  `full_burst_id` document the `FileNotFoundError` they raise when the value was
  not propagated and the burst map has not been downloaded.

### Removed

- **Breaking:** removed `CDSES3Settings.s3_env`, a legacy context manager that
  injected `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` for an external script.
  S3 credentials now flow through the S3 `UPath` and `freezecore.s3.s3_env`.
