# Test Fixtures

## Burst extractor reference

`s1bursts.burst_extractor` is a port of `sentinel1_burst_extractor.sh`, the Copernicus Data Space Ecosystem (CDSE) single-burst extraction script.

Because it is published by CDSE, it is the trusted reference for the port's output.

| --- | --- |
| Source | <https://github.com/eu-cdse/utilities/blob/main/sentinel1_burst_extractor.sh> |
| Script version | 1.3 (2025-03-14, "corrected GCPs extraction") |
| Retrieved | 2026-08-14 |
| SHA-256 | `c17c8d1b6191bef2d7e49d85e5941407e588dbc83a2ac8de18ae3c366ef16e4e` |
| Author | CloudFerro (jmusial@cloudferro.com), for ESA |

It is vendored, not executed: running it would need `s5cmd`, `xmlstarlet`, `jq`, `bc`, GDAL ≥ 3.9
and live S3 credentials. `tests/test_esa_reference_parity.py` reads it instead, extracting the
edits it makes so the port can be checked against them, and pins the digest above so that an
upstream revision fails the suite rather than passing unnoticed.

`s1bursts.burst_extractor.REFERENCE_SCRIPT_VERSION` records which version the port tracks.

## OPERA RTC-S1 as the reference for SNAP-derived products

`s1bursts.snap_client` runs the processing graphs built by `s1bursts.snap_graph` through SNAP.
Product goldens can verify that the output has not changed, but not that it is correct. The JPL
OPERA Level-2 products are used to verify correctness.

OPERA RTC-S1 shares this package's inputs and conventions (single Sentinel-1 bursts, Copernicus
GLO-30 DEM, gamma-0 in linear power) but differs in the processing chain, which is ISCE3 rather
than SNAP. That makes it a suitable reference to verify our outputs against.

| | |
| --- | --- |
| Product | OPERA Level-2 RTC-S1 (`OPERA_L2_RTC-S1_V1`) and its static layers (`OPERA_L2_RTC-S1-STATIC_V1`) |
| Producer | NASA/JPL, processed with ISCE3 |
| Catalog | <https://doi.org/10.5067/SNWG/OPERA_L2_RTC-S1_V1> |
| Distribution | ASF DAAC, searched through CMR; requires Earthdata credentials |
| Product version | v1.0 (recorded per granule in `opera_reference.json`) |

Citation:

> NASA/JPL/OPERA. (2023). *OPERA Radiometric Terrain Corrected SAR Backscatter from Sentinel-1 validated product (Version 1)* [Dataset]. NASA Alaska Satellite Facility Distributed Active Archive Center. <https://doi.org/10.5067/SNWG/OPERA_L2_RTC-S1_V1> Date Accessed: 2026-08-18

The processor is open source at <https://github.com/opera-adt/RTC>.

Two layers are compared:

- **gamma-0**, from the per-acquisition granule, against the `Gamma0` band of this package's RTC
  product.
- **the area normalization factor (ANF)**, from the static granule's `rtc_anf_gamma0_to_beta0`
  layer, against the `simulatedImage` band. Since gamma-0 = beta-0 / *A*, the factor that converts
  gamma-0 back to beta-0 is *A*, which is what SNAP writes as `simulatedImage`. The layer depends
  only on geometry and the DEM, so comparing it isolates terrain flattening and geocoding from
  calibration and speckle.

### Conditions the comparison depends on

**The ANF must be compared against the acquisition it was derived from.** Each static granule is
generated from a single Sentinel-1 acquisition, named in its `InputGranules` field, and is then
declared valid for the whole mission. The layers are not, however, time-invariant: a different
position within the orbital tube changes the local incidence angle, so the same burst from another
acquisition differs by about 2% at the median and 9% at the 90th percentile, with about 1% of
pixels changing between illuminated and shadowed. These differences exceed the difference between
processors that the comparison is meant to measure. The burst processed with SNAP must therefore
come from the acquisition recorded in `Scene.static_date`.

**Gamma-0 must be averaged in power before comparison.** Per-pixel differences are dominated by
speckle and by the difference in effective resolution between this package's multilooked product
and OPERA's native 30 m posting. The per-pixel median relative difference is approximately 18%,
while the bias after block averaging is a few hundredths of a decibel and is stable for block
sizes of 90 m and above. Only the block bias supports a statement about radiometry. Averaging is
performed in power, not in decibels.

### Known difference near the noise floor

Over targets of very low backscatter the two products diverge. Comparing the distributions at
their 1st percentile, our product reads between 0.74 dB (Aletsch VV) and 2.44 dB (Iceland VH)
below OPERA's, and the offset grows as the scene darkens; scene-wide the same products agree to
within 0.07 dB. The cause is where each chain removes thermal noise. OPERA clips the
noise-corrected signal at zero, and does so while the data are still single-looked, whereas SNAP
applies no such clip. Clipping discards the negative excursions but keeps the positive ones, so
OPERA's multilooked mean is biased upward wherever the signal approaches the noise level. Note
that even without clipping, the SNAP product over Iceland contains no negative values.

The offset is recorded per scene and polarization as `p01_dB_offset` in `opera_reference.json`.
Levels are compared at a percentile rather than over selected pixels, so no pixels are chosen on
either image. Absolute gamma-0 near the noise floor should not be treated as interchangeable
between the two products.
