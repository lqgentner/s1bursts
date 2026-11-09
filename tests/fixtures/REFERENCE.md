# Test fixtures

## OPERA RTC-S1 reference

Product goldens show whether SNAP output changed, but not whether it is correct. The JPL OPERA Level-2 products provide an independent check because they use the same inputs (single Sentinel-1 bursts and Copernicus
GLO-30 DEM) and backscatter convention (gamma-0 in linear power), but process with ISCE3 instead of SNAP.

| | |
| --- | --- |
| Product | OPERA Level-2 RTC-S1 (`OPERA_L2_RTC-S1_V1`) and static layers (`OPERA_L2_RTC-S1-STATIC_V1`) |
| Producer | NASA/JPL, processed with ISCE3 |
| Catalog | <https://doi.org/10.5067/SNWG/OPERA_L2_RTC-S1_V1> |
| Distribution | ASF DAAC, downloaded with `asf_search` (requires Earthdata credentials) |
| Product version | v1.0, recorded per granule in `opera_reference.json` |

Citation:

> NASA/JPL/OPERA. (2023). *OPERA Radiometric Terrain Corrected SAR Backscatter from Sentinel-1 validated product (Version 1)* [Dataset]. NASA Alaska Satellite Facility Distributed Active Archive Center. <https://doi.org/10.5067/SNWG/OPERA_L2_RTC-S1_V1> Date Accessed: 2026-08-18

The processor is open source at <https://github.com/opera-adt/RTC>.

The comparison uses these layers:

- **gamma-0:** OPERA's acquisition layer against this package's `gamma0` band.
- **area normalization factor (ANF):** OPERA's `rtc_anf_gamma0_to_beta0` static layer against this package's `scattering_area` band (SNAP's `simulatedImage`). Both hold the local scattering area, so this isolates terrain flattening and geocoding separately from calibration and speckle.

### Comparison requirements

**Use the acquisition that produced the static layer.** The layer changes slightly with perturbations to the Satellites' orbits, so `Scene.static_date` identifies the matching acquisition.

**Average gamma-0 in power.** Speckle and different pixel sizes make individual pixels unsuitable for comparison. Tests calculate averages over blocks of at least 90 m and convert to decibels afterward.

### Difference near the noise floor

Over the darkest targets, the two products diverge: Comparing the distributions at
their 1st percentile, this package reads between 0.74 dB (Aletsch VV) and 2.44 dB (Iceland VH)
below OPERA's, while the full scenes agree within 0.07 dB. OPERA clips negative
noise-corrected samples to zero before averaging, while SNAP does not. This raises OPERA's result
when the signal is close to the noise level. `opera_reference.json` records the offset for each scene and polarization as `p01_dB_offset`. Gamma-0 values near the noise floor should not be treated as interchangeable between the products.

## ASF HyP3 burst InSAR coherence reference

OPERA has no coherence layer, so coherence is checked against ASF's HyP3 burst InSAR product. HyP3 uses the same bursts and Copernicus GLO-30 DEM, but processes them with ISCE2 topsApp instead of SNAP.

| | |
| --- | --- |
| Product | ASF HyP3 Sentinel-1 burst InSAR (`INSAR_ISCE_BURST`), 5 × 1 looks |
| Producer | ASF, using the `hyp3-isce2` plugin and ISCE2 topsApp |
| Guide | <https://hyp3-docs.asf.alaska.edu/guides/burst_insar_product_guide> |
| Distribution | Ordered with `hyp3_sdk` (requires Earthdata credentials and HyP3 credits) |
| Layer compared | `_corr.tif` against this package's `coherence` band |
| Versions | Recorded per scene in `insar_reference.json` from the product README |

Citation:

> Hogenson, K., Kristenson, H., Kennedy, J., Johnston, A., Rine, J., Logan, T., Zhu, J., Williams, F., Herrmann, J., Smale, J., & Meyer, F. (2020). *Hybrid Pluggable Processing Pipeline (HyP3): A cloud-native infrastructure for generic processing of SAR data* [Computer software]. <https://doi.org/10.5281/zenodo.4646138>

HyP3 products expire and may be recreated with newer software. The reference records each product's plugin and ISCE2 versions. The current measurements use `hyp3_isce2` 3.0.2 and ISCE 2.6.5.

The processor is open source at <https://github.com/ASFHyP3/hyp3-isce2>.

### How the coherence values differ

The pipelines calculate coherence differently:

| | This package | HyP3 |
| --- | --- | --- |
| Calculation | Complex cross-correlation | ISCE2 `phsig.cor` phase consistency |
| Input | Single-look samples before geocoding | A 5 × 1 multilooked, filtered interferogram |
| Window | 25 × 5 samples | 5 × 5 multilooked pixels, equal to 25 × 5 original samples |
| Uses amplitude | Yes | No |

The HyP3 products are ordered at 5 × 1 looks to give both products the same window size. Due to HyP3's phase filtering, it usually reports higher coherence and often reaches 1 over coherent ground.

The tests therefore check that:

- both products rank areas in the same order
- their differences keep the expected sign and stay within fixed bounds
- coherent areas remain coherent in both
- the products line up on the ground

`insar_reference.json` stores quantiles for each scene because the difference changes with coherence and cannot be described by one offset.

### Comparison requirements

**Use the same grid.** At 5 × 1 looks, HyP3 produces a 20 m grid that aligns with this package's grid. `Scene.crs` records the UTM zone, and products in another zone are rejected instead of reprojected.

**Exclude edges.** Edge pixels are removed from both products because filters use nearby pixels and the valid areas differ. HyP3 also uses 0 for missing data, while this package uses NaN.

**Order HyP3 products explicitly.** HyP3 products are created on demand and expire after about two weeks. Use `regenerate_insar_reference.py --submit` or set `S1BURSTS_HYP3_SUBMIT=1`. Submission spends HyP3 credits. Without those options, the tests only use matching completed jobs.

**Record whether a scene has a visible pattern.** `reference_has_pattern` selects the pattern checks: varying scenes are checked for ranking and alignment, nearly uniform scenes for high coherence instead. The value checks apply to every scene. Tests also verify that each scene still matches its declaration.

**Treat a missing alignment peak as a failure.** `peak_on_boundary` means the offset is larger than the three-pixel search. Low `peak_contrast` means the spatial pattern was not reproduced.

**Use co-polarized channels.** HyP3 only supports VV or HH, so the reference scenes use VV.

### Known difference: HyP3 reads higher

Measurements at 5 × 1 looks and 20 m spacing:

| | Aletsch (mixed) | Mojave (arid) |
| --- | --- | --- |
| Median, this package | 0.191 | 0.808 |
| Median, HyP3 | 0.343 | 0.992 |
| Median offset | +0.186 | +0.177 |
| HyP3 10th to 90th percentile | 0.246 to 0.881 | 0.944 to 0.998 |
| This package, same percentiles | 0.065 to 0.590 | 0.584 to 0.905 |
| Rank correlation, 60 m blocks | 0.713 | 0.502 (HyP3 is nearly uniform) |
| Pearson correlation, 60 m blocks | 0.848 | 0.334 (HyP3 is nearly uniform) |
| Ground offset (row, column) | -9.1 m, -12.3 m | Not measurable |

HyP3 is about 0.18 higher at the median in both scenes. Its Mojave values are nearly all at 1, leaving too little variation for a useful ranking or alignment check.

Coherence values and thresholds cannot be transferred directly between the two products. Their spatial ordering can be compared only when the HyP3 reference has enough variation.

## Regenerating a record

**Regenerate every scene after a toolchain change.** Each record states one `_toolchain` block for the scenes in it, so regenerating one scene under a different SNAP, Java or package version would claim the untouched scenes were measured with that toolchain too. Both scripts measure the toolchain before they start and refuse a single-scene run that would move it, naming the entries that changed.
