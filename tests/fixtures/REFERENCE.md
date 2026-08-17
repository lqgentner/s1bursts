# Vendored reference extractor

`sentinel1_burst_extractor.sh` is the ESA/CDSE single-burst extraction script that
`s1bursts.burst_extractor` is a port of. It is the same program CDSE runs to produce the burst
products served from `bursts.dataspace.copernicus.eu`, which makes it the trusted reference for
the port's output.

| | |
|---|---|
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
