"""Python port of the Sentinel-1 burst extraction script for CDSE S3.

Original script: https://github.com/eu-cdse/utilities/blob/main/sentinel1_burst_extractor.sh

The port targets **byte-level parity** with that reference, which is the same
program CDSE runs to produce the single-burst products it serves from
``bursts.dataspace.copernicus.eu``. Extracting a burst from S3 therefore yields
the same product as downloading it from CDSE, and the live differential test in
``tests/test_integration.py`` asserts exactly that.

Parity is the contract, so the reference's deliberate relaxations of the SAFE
specification are reproduced rather than corrected:

- ``manifest.safe`` byte-stream ``size`` attributes and ``checksum`` values are
  blanked, not recomputed, because the edited files no longer match the parent
  product's values.
- The geolocation grid keeps every point of the parent product with its line
  numbers shifted to the burst origin, so points outside the burst window carry
  negative or out-of-range line numbers. SNAP tolerates this and uses the grid
  for interpolation only.
- The manifest footprint is derived from the whole shifted grid, so an interior
  burst inherits a footprint wider than its own extent.

``tests/safe_conformance.py`` re-derives these rules from the vendored reference
script and pins them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import logging
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING
import warnings

from freezebase.download import retry_request
from freezebase.raster import rasterio_open
from freezebase.s3 import make_s3_upath
from lxml import etree
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window
import requests
from upath import UPath

from s1bursts.config import get_settings
from s1bursts.errors import BurstNotFoundError, ConfigurationError

if TYPE_CHECKING:
    from s1bursts.config import CDSES3Settings

logger = logging.getLogger(__name__)

REFERENCE_SCRIPT_VERSION = "1.3"
"""Version of the ESA/CloudFerro reference extractor this port tracks."""


def _parse_xml(data: bytes) -> etree._Element:
    """Parse untrusted SAFE XML with entity resolution and network access disabled.

    A fresh parser is created per call because ``lxml`` parsers are not safe to
    share across the threads that drive concurrent burst extraction.

    Parameters
    ----------
    data : bytes
        Raw XML document bytes.

    Returns
    -------
    lxml.etree._Element
        Parsed document root.
    """
    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    return etree.fromstring(data, parser=parser)


_CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"

_MANIFEST_NS = {
    "xfdu": "urn:ccsds:schema:xfdu:1",
    "safe": "http://www.esa.int/safe/sentinel-1.0",
    "s1": "http://www.esa.int/safe/sentinel-1.0/sentinel-1",
    "s1sarl1": "http://www.esa.int/safe/sentinel-1.0/sentinel-1/sar/level-1",
    "gml": "http://www.opengis.net/gml",
}


@dataclass(frozen=True)
class _SubswathTiming:
    t0_delta: float  # half-duration of Doppler centroid crossing (seconds)
    tbeam: float  # burst repetition period (seconds)
    t_pre: float  # pre-beam steering interval (seconds)


_SUBSWATH_TIMING: dict[str, _SubswathTiming] = {
    "iw1": _SubswathTiming(0.81531490 / 2.0, 2.758273, 2.299849),
    "iw2": _SubswathTiming(1.06087913 / 2.0, 2.758273, 2.299849),
    "iw3": _SubswathTiming(0.83045822 / 2.0, 2.758273, 2.299849),
    "ew1": _SubswathTiming(0.66551400 / 2.0, 3.038376, 2.299970),
    "ew2": _SubswathTiming(0.54139519 / 2.0, 3.038376, 2.299970),
    "ew3": _SubswathTiming(0.59473403 / 2.0, 3.038376, 2.299970),
    "ew4": _SubswathTiming(0.54745052 / 2.0, 3.038376, 2.299970),
    "ew5": _SubswathTiming(0.60098234 / 2.0, 3.038376, 2.299970),
}

# Sentinel-1 orbital repeat period in seconds (12 days / 175 orbits)
_ORBIT_PERIOD = Decimal(12) * Decimal(24) * Decimal(3600) / Decimal(175)


@dataclass(frozen=True)
class _BurstMeta:
    """Resolved burst metadata used to name and edit output files."""

    burst_number: int
    byte_offset: str
    starting_line: int
    number_of_lines: int
    number_of_samples: int
    azimuth_start: str  # ISO 8601 UTC azimuthTime
    azimuth_end: str  # ISO 8601 UTC azimuthTime + (lines-1) * interval
    sensing_time_compact: str  # e.g. "20220825T171550" for SAFE directory name
    datatake_id: str  # zero-padded 6-digit missionDataTakeId
    platform_number: str  # "A" or "B"
    new_pattern: str  # e.g. "s1a-iw1-slc-vv-030346-20220825T171550-349850"
    new_pattern_short: str  # new_pattern with hyphens removed


@retry_request(logger=logger)
def _lookup_s3_path(product_name: str) -> str:
    """Return the CDSE S3 path for a named SLC parent product."""
    resp = requests.get(
        f"{_CATALOG_URL}/Products",
        params={"$filter": f"Name eq '{product_name}'"},
        timeout=30,
    )
    resp.raise_for_status()
    values = resp.json().get("value", [])
    if not values:
        msg = f"Product '{product_name}' not found in CDSE catalogue."
        raise BurstNotFoundError(msg)
    return str(values[0]["S3Path"])


def _make_s3_upath(s3_path: str, settings: CDSES3Settings) -> UPath:
    """Build an s3fs-backed UPath from an OData S3Path and CDSE S3 settings.

    The OData S3Path starts with a leading slash (e.g. /eodata/Sentinel-1/...),
    which becomes s3://eodata/Sentinel-1/... after prepending ``s3:``.

    Authenticates with the configured profile, falling back to the explicit key
    pair. ``make_s3_upath`` rejects the two together, so only one is ever passed.
    """
    if settings.profile is not None:
        return make_s3_upath(
            f"s3:/{s3_path}",
            profile=settings.profile,
            endpoint_url=settings.endpoint_url,
        )
    if settings.access_key_id is not None and settings.secret_access_key is not None:
        return make_s3_upath(
            f"s3:/{s3_path}",
            key=settings.access_key_id,
            secret=settings.secret_access_key.get_secret_value(),
            endpoint_url=settings.endpoint_url,
        )
    msg = (
        "CDSE S3 credentials not configured (CDSE_S3_PROFILE, or "
        "CDSE_S3_ACCESS_KEY_ID / CDSE_S3_SECRET_ACCESS_KEY)."
    )
    raise ConfigurationError(msg)


def _glob_annotation_xml(product_root: UPath, sub_swath: str, polarization: str) -> UPath:
    """Return the S3 path to the annotation XML for a given subswath and polarization."""
    matches = list(product_root.glob(f"annotation/s1*{sub_swath}-slc-{polarization}*.xml"))
    if len(matches) != 1:
        msg = (
            f"Expected 1 annotation XML for {sub_swath}/{polarization}, "
            f"found {len(matches)}: {matches}"
        )
        raise BurstNotFoundError(msg)
    return matches[0]


def _orbital_burst_number(
    annotation_root: etree._Element,
    manifest_root: etree._Element,
    *,
    relative_burst_id: int,
    sub_swath: str,
) -> int:
    """Compute 1-based burst number via orbital-mechanics timing.

    Used as a fallback for legacy products that pre-date the ``<burstId>``
    annotation field. Uses ``Decimal`` arithmetic to preserve 8-digit
    precision.
    """
    timing = _SUBSWATH_TIMING[sub_swath]

    sensing_times = [
        Decimal(
            str(
                datetime.fromisoformat(burst.findtext("sensingTime"))
                .replace(tzinfo=UTC)
                .timestamp(),
            ),
        )
        for burst in annotation_root.findall("swathTiming/burstList/burst")
    ]

    ant_str = manifest_root.xpath(
        "metadataSection/metadataObject/metadataWrap/xmlData"
        "/safe:orbitReference/safe:extension/s1:orbitProperties/s1:ascendingNodeTime",
        namespaces=_MANIFEST_NS,
    )[0].text
    tanx = Decimal(
        str(datetime.fromisoformat(ant_str).replace(tzinfo=UTC).timestamp()),
    )

    orbit_number = int(
        manifest_root.xpath(
            "metadataSection/metadataObject/metadataWrap/xmlData"
            "/safe:orbitReference/safe:relativeOrbitNumber[@type='start']",
            namespaces=_MANIFEST_NS,
        )[0].text,
    )

    t0_delta = Decimal(str(timing.t0_delta))
    tbeam = Decimal(str(timing.tbeam))
    t_pre = Decimal(str(timing.t_pre))

    for i, t0 in enumerate(sensing_times, 1):
        raw_bid = 1 + ((t0 + t0_delta - tanx) + (orbit_number - 1) * _ORBIT_PERIOD - t_pre) / tbeam
        # Truncate rather than round: the fractional part regularly exceeds
        # 0.5, and rounding would then match the wrong burst.
        if int(raw_bid) == relative_burst_id:
            return i

    msg = f"Relative burst ID {relative_burst_id} not found via orbital fallback."
    raise BurstNotFoundError(msg)


def _resolve_burst(
    annotation_root: etree._Element,
    manifest_root: etree._Element,
    *,
    relative_burst_id: int,
    sub_swath: str,
    annotation_filename: str,
) -> _BurstMeta:
    """Resolve burst position and build all derived naming metadata.

    Tries a direct ``<burstId>`` XPath match first; falls back to
    :func:`_orbital_burst_number` for legacy products without that field.
    """
    bursts = annotation_root.findall("swathTiming/burstList/burst")

    burst_number = 0
    for i, burst_el in enumerate(bursts, 1):
        bid_text = burst_el.findtext("burstId")
        if bid_text is not None and int(bid_text) == relative_burst_id:
            burst_number = i
            break

    if burst_number == 0:
        burst_number = _orbital_burst_number(
            annotation_root,
            manifest_root,
            relative_burst_id=relative_burst_id,
            sub_swath=sub_swath,
        )

    burst_el = bursts[burst_number - 1]
    byte_offset = burst_el.findtext("byteOffset")
    azimuth_start = burst_el.findtext("azimuthTime")
    sensing_time = burst_el.findtext("sensingTime")  # e.g. "2022-08-25T17:15:50.399045"

    lines = int(annotation_root.findtext("swathTiming/linesPerBurst"))
    samples = int(annotation_root.findtext("swathTiming/samplesPerBurst"))
    starting_line = (burst_number - 1) * lines

    azimuth_interval = annotation_root.findtext(
        "imageAnnotation/imageInformation/azimuthTimeInterval",
    )
    azimuth_end = (
        datetime.fromisoformat(azimuth_start)
        + timedelta(seconds=float(azimuth_interval) * (lines - 1))
    ).strftime("%Y-%m-%dT%H:%M:%S.%f")

    datatake_id = f"{int(annotation_root.findtext('adsHeader/missionDataTakeId')):06d}"
    burst_id_padded = f"{relative_burst_id:06d}"
    sensing_compact = sensing_time.split(".")[0].replace("-", "").replace(":", "")
    prefix = annotation_filename[:15]  # "s1a-iw1-slc-vv-"
    new_pattern = f"{prefix}{burst_id_padded}-{sensing_compact}-{datatake_id}"

    platform_number = manifest_root.xpath(
        "metadataSection/metadataObject/metadataWrap/xmlData/safe:platform/safe:number",
        namespaces=_MANIFEST_NS,
    )[0].text

    return _BurstMeta(
        burst_number=burst_number,
        byte_offset=byte_offset,
        starting_line=starting_line,
        number_of_lines=lines,
        number_of_samples=samples,
        azimuth_start=azimuth_start,
        azimuth_end=azimuth_end,
        sensing_time_compact=sensing_compact,
        datatake_id=datatake_id,
        platform_number=platform_number,
        new_pattern=new_pattern,
        new_pattern_short=new_pattern.replace("-", ""),
    )


def _edit_annotation(root: etree._Element, meta: _BurstMeta) -> bytes:
    """Edit the annotation XML tree in-place and return the serialised bytes."""
    burst_list = root.find("swathTiming/burstList")
    for burst in list(burst_list.findall("burst")):
        if burst.findtext("byteOffset") != meta.byte_offset:
            burst_list.remove(burst)
    burst_list.set("count", "1")

    img_info = root.find("imageAnnotation/imageInformation")
    img_info.find("numberOfLines").text = str(meta.number_of_lines)
    img_info.find("productFirstLineUtcTime").text = meta.azimuth_start
    img_info.find("productLastLineUtcTime").text = meta.azimuth_end

    hdr = root.find("adsHeader")
    hdr.find("startTime").text = meta.azimuth_start
    hdr.find("stopTime").text = meta.azimuth_end

    for el in root.findall(
        "geolocationGrid/geolocationGridPointList/geolocationGridPoint/line",
    ):
        el.text = str(int(el.text) - meta.starting_line)

    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", pretty_print=True)


def _edit_calibration(text: str, meta: _BurstMeta) -> bytes:
    """Shift calibrationVector line numbers and update start/stop times."""
    root = _parse_xml(text.encode())
    for el in root.findall("calibrationVectorList/calibrationVector/line"):
        el.text = str(int(el.text) - meta.starting_line)
    root.find("adsHeader/startTime").text = meta.azimuth_start
    root.find("adsHeader/stopTime").text = meta.azimuth_end
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", pretty_print=True)


def _edit_noise(text: str, meta: _BurstMeta) -> bytes:
    """Shift noiseRangeVector line numbers and update start/stop times."""
    root = _parse_xml(text.encode())
    for el in root.findall("noiseRangeVectorList/noiseRangeVector/line"):
        el.text = str(int(el.text) - meta.starting_line)
    root.find("adsHeader/startTime").text = meta.azimuth_start
    root.find("adsHeader/stopTime").text = meta.azimuth_end
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", pretty_print=True)


def _edit_rfi(text: str, meta: _BurstMeta) -> bytes:
    """Update start/stop times in an RFI annotation file."""
    root = _parse_xml(text.encode())
    root.find("adsHeader/startTime").text = meta.azimuth_start
    root.find("adsHeader/stopTime").text = meta.azimuth_end
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", pretty_print=True)


def _build_footprint(root: etree._Element) -> str:
    """Build a GML coordinate string from the annotation geolocation grid.

    Coordinates are ordered: first and last points of the top row
    (``line=0``, sorted by pixel), then last and first of all non-zero-line
    points (document order), closing with the first top point again.
    Format: ``"lon lat,lon lat,..."``.

    The second group spans every remaining grid row rather than just the
    extracted burst, so an interior burst inherits a footprint wider than its
    own extent. This reproduces the reference extractor, and therefore CDSE's
    own burst products, rather than diverging from them.

    Coordinate text is copied verbatim from the annotation. That matches the
    products CDSE serves, which carry the annotation's scientific notation
    (``7.628802608148144e+00``). The vendored reference script instead pipes
    each value through XPath ``number()``, whose libxml2 formatting drops to a
    fixed 15 significant digits and would silently truncate the latitude.
    """
    pts = root.findall("geolocationGrid/geolocationGridPointList/geolocationGridPoint")
    top = sorted(
        (
            (p.findtext("longitude"), p.findtext("latitude"), float(p.findtext("pixel")))
            for p in pts
            if int(p.findtext("line")) == 0
        ),
        key=lambda t: t[2],
    )
    nontop = [
        (p.findtext("longitude"), p.findtext("latitude"))
        for p in pts
        if int(p.findtext("line")) != 0
    ]
    top_first = f"{top[0][0]} {top[0][1]}"
    top_last = f"{top[-1][0]} {top[-1][1]}"
    nontop_last = f"{nontop[-1][0]} {nontop[-1][1]}"
    nontop_first = f"{nontop[0][0]} {nontop[0][1]}"
    return f"{top_first},{top_last},{nontop_last},{nontop_first},{top_first}"


def _edit_manifest(  # noqa: C901
    text: str,
    meta: _BurstMeta,
    *,
    sub_swath: str,
    old_pattern: str,
    footprint: str,
) -> bytes:
    """Produce a burst-specific manifest.safe from the full product manifest.

    Two text-level substitutions rename the annotation-file basename to the
    burst-specific pattern before lxml parsing, so that attribute values
    (``dataObjectID``, ``dataObjectPointer``) are also updated.
    """
    old_pattern_short = old_pattern.replace("-", "")
    text = text.replace(old_pattern, meta.new_pattern)
    text = text.replace(old_pattern_short, meta.new_pattern_short)

    root = _parse_xml(text.encode())

    # Update acquisition period
    for el in root.xpath(
        "metadataSection/metadataObject/metadataWrap/xmlData"
        "/safe:acquisitionPeriod/safe:startTime",
        namespaces=_MANIFEST_NS,
    ):
        el.text = meta.azimuth_start
    for el in root.xpath(
        "metadataSection/metadataObject/metadataWrap/xmlData/safe:acquisitionPeriod/safe:stopTime",
        namespaces=_MANIFEST_NS,
    ):
        el.text = meta.azimuth_end

    # Clear byte stream integrity fields (now stale)
    for el in root.xpath("dataObjectSection/dataObject/byteStream/checksum"):
        el.text = None
    for el in root.xpath("dataObjectSection/dataObject/byteStream"):
        el.set("size", "")

    # Update footprint
    for el in root.xpath(
        "metadataSection/metadataObject/metadataWrap/xmlData"
        "/safe:frameSet/safe:frame/safe:footPrint/gml:coordinates",
        namespaces=_MANIFEST_NS,
    ):
        el.text = footprint

    # Remove thumbnail / overlay dataObjects
    for id_val in ("quicklook", "mapoverlay", "productpreview"):
        for el in root.xpath(f"dataObjectSection/dataObject[@ID='{id_val}']"):
            el.getparent().remove(el)

    # Remove non-target swath entries
    subswath_upper = sub_swath.upper()
    for el in root.xpath(
        "metadataSection/metadataObject/metadataWrap/xmlData/safe:platform"
        "/safe:instrument/safe:extension/s1sarl1:instrumentMode"
        f"/s1sarl1:swath[text()!='{subswath_upper}']",
        namespaces=_MANIFEST_NS,
    ):
        el.getparent().remove(el)

    # Remove dataObjects not referencing this burst
    for el in root.xpath(
        "dataObjectSection/dataObject[not(contains(@ID, $p))]",
        p=meta.new_pattern_short,
    ):
        el.getparent().remove(el)

    # Remove metadataObjects whose pointer doesn't reference this burst
    for ptr in root.xpath(
        "metadataSection/metadataObject/dataObjectPointer[not(contains(@dataObjectID, $p))]",
        p=meta.new_pattern_short,
    ):
        parent = ptr.getparent()
        parent.getparent().remove(parent)

    # Remove informationPackageMap contentUnits not referencing this burst
    for ptr in root.xpath(
        "informationPackageMap/xfdu:contentUnit/xfdu:contentUnit"
        "/dataObjectPointer[not(contains(@dataObjectID, $p))]",
        namespaces=_MANIFEST_NS,
        p=meta.new_pattern_short,
    ):
        parent = ptr.getparent()
        parent.getparent().remove(parent)

    # Remove schema-reference metadata
    for el in root.xpath("metadataSection/metadataObject[@classification='SYNTAX']"):
        el.getparent().remove(el)

    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", pretty_print=True)


def _extract_tiff(
    tiff_s3_path: UPath,
    dst_tiff: Path,
    *,
    geoloc_pts: list[tuple[float, float, float, float, float]],
    meta: _BurstMeta,
) -> None:
    """Extract a single burst from a CDSE S3 TIFF using windowed read and GCPs.

    Parameters
    ----------
    tiff_s3_path : UPath
        S3 path to the full-product measurement TIFF.
    dst_tiff : Path
        Local destination for the extracted burst TIFF.
    geoloc_pts : list of (pixel, line_shifted, lon, lat, height) tuples
        Geolocation grid points; line values are already shifted by
        ``meta.starting_line`` so they are relative to the burst window.
    meta : _BurstMeta
        Burst metadata (for window dimensions).
    """
    gcps = [
        GroundControlPoint(row=line, col=pixel, x=lon, y=lat, z=height)
        for pixel, line, lon, lat, height in geoloc_pts
    ]
    window = Window(
        col_off=0,
        row_off=meta.starting_line,
        width=meta.number_of_samples,
        height=meta.number_of_lines,
    )
    # SLC measurement TIFFs carry no geotransform, so opening one warns. That is
    # expected for this product type and says nothing the caller can act on.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        with rasterio_open(tiff_s3_path) as src:
            profile = src.profile.copy()
            src_tags = src.tags()
            data = src.read(window=window)
    profile.update(width=meta.number_of_samples, height=meta.number_of_lines, crs=None)
    # ESA SLC measurement TIFFs are striped (tiled=False, blockysize=1). The
    # copied profile carries a blockxsize (= full-product width) that is only
    # valid for tiled output, so forwarding it makes GDAL warn "BLOCKXSIZE can
    # only be used with TILED=YES". Drop it to keep the striped layout.
    profile.pop("blockxsize", None)
    # SLC files have no affine CRS — GCPs are the sole georeferencing.
    # Setting GCPs on a file that has any geotransform (even a default one written
    # by rasterio from the source profile) makes GDAL emit a CPLE_AppDefined
    # "geotransform going to be cleared" message. Suppress that message at
    # the logging level.
    _rasterio_logger = logging.getLogger("rasterio")
    _prev_level = _rasterio_logger.level
    _rasterio_logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
            with rasterio.open(dst_tiff, "w", **profile) as dst:
                dst.write(data)
                dst.gcps = (gcps, CRS.from_epsg(4326))
                dst.update_tags(**src_tags)
    finally:
        _rasterio_logger.setLevel(_prev_level)


def _zip_safe_dir(safe_dir: Path) -> Path:
    """Zip a .SAFE directory and delete the source tree."""
    zip_stem = str(safe_dir.with_suffix(""))
    shutil.make_archive(zip_stem, "zip", root_dir=safe_dir.parent, base_dir=safe_dir.name)
    shutil.rmtree(safe_dir)
    return Path(f"{zip_stem}.zip")


def extract_burst_from_s3(  # noqa: PLR0915
    save_path: str | Path | UPath,
    *,
    parent_product_name: str,
    polarization: str,
    sub_swath: str,
    burst_id: int,
) -> Path | UPath:
    """Extract a single Sentinel-1 SLC burst from CDSE S3.

    All intermediate files (edited XMLs, extracted TIFF) are written to a
    local temporary directory. Only the final ``.zip`` is moved or uploaded
    to ``save_path``, which may be a local path or a remote UPath (e.g. S3).

    Parameters
    ----------
    save_path : str, Path, or UPath
        Full destination path (including filename) for the ``.zip``.
    parent_product_name : str
        Name of the parent ``*.SAFE`` product on CDSE.
    polarization : str
        Polarization channel, e.g. ``"VV"``, ``"VH"``, ``"HH"``, ``"HV"``.
    sub_swath : str
        Subswath identifier, e.g. ``"IW1"``, ``"IW2"``, ``"IW3"``.
    burst_id : int
        Relative burst ID within the subswath.

    Returns
    -------
    Path or UPath
        Path to the produced ``.zip`` file.

    Raises
    ------
    ValueError
        If ``sub_swath`` is not a recognised subswath identifier.
    BurstNotFoundError
        If the parent product or the requested burst cannot be found.
    ConfigurationError
        If the CDSE S3 credentials are not configured.
    """
    pol = polarization.lower()
    swath = sub_swath.lower()

    if swath not in _SUBSWATH_TIMING:
        msg = f"Unknown subswath '{swath}'. Must be one of: {sorted(_SUBSWATH_TIMING)}"
        raise ValueError(msg)

    s3_settings = get_settings().cdse.s3
    save_upath = UPath(save_path)

    # --- Catalog lookup ---
    logger.info("Looking up S3 path for '%s'", parent_product_name)
    s3_path = _lookup_s3_path(parent_product_name)
    product_root = _make_s3_upath(s3_path, s3_settings)

    # --- Download annotation and manifest ---
    logger.info("Downloading annotation XML (%s/%s)", swath, pol)
    ann_upath = _glob_annotation_xml(product_root, swath, pol)
    ann_text = ann_upath.read_text()
    manifest_text = (product_root / "manifest.safe").read_text()

    ann_root = _parse_xml(ann_text.encode())
    mfst_root = _parse_xml(manifest_text.encode())

    # --- Resolve burst position and build naming metadata ---
    meta = _resolve_burst(
        ann_root,
        mfst_root,
        relative_burst_id=burst_id,
        sub_swath=swath,
        annotation_filename=ann_upath.name,
    )
    logger.debug(
        "Burst %d → burst_number=%d, starting_line=%d",
        burst_id,
        meta.burst_number,
        meta.starting_line,
    )

    # --- Extract GCPs from the raw (pre-edit) annotation; lines shifted to burst origin ---
    geoloc_pts: list[tuple[float, float, float, float, float]] = [
        (
            float(p.findtext("pixel")),
            float(p.findtext("line")) - meta.starting_line,
            float(p.findtext("longitude")),
            float(p.findtext("latitude")),
            float(p.findtext("height")),
        )
        for p in ann_root.findall(
            "geolocationGrid/geolocationGridPointList/geolocationGridPoint",
        )
    ]

    # --- Build output locally, then move/upload the final zip to save_dir ---
    safe_name = (
        f"S1{meta.platform_number}_SLC_{meta.sensing_time_compact}"
        f"_{burst_id:06d}_{sub_swath.upper()}_{polarization.upper()}_{meta.datatake_id}.SAFE"
    )
    with tempfile.TemporaryDirectory(prefix="s1bursts-") as tmp:
        tmp_dir = Path(tmp)
        safe_dir = tmp_dir / safe_name
        (safe_dir / "annotation" / "calibration").mkdir(parents=True)
        (safe_dir / "measurement").mkdir()

        # --- Annotation (edits ann_root in-place) ---
        logger.info("Editing annotation and auxiliary XML files")
        ann_bytes = _edit_annotation(ann_root, meta)
        (safe_dir / "annotation" / f"{meta.new_pattern}.xml").write_bytes(ann_bytes)

        # --- Calibration ---
        cal_text = (
            product_root / f"annotation/calibration/calibration-{ann_upath.name}"
        ).read_text()
        (
            safe_dir / "annotation" / "calibration" / f"calibration-{meta.new_pattern}.xml"
        ).write_bytes(
            _edit_calibration(cal_text, meta),
        )

        # --- Noise ---
        noise_text = (product_root / f"annotation/calibration/noise-{ann_upath.name}").read_text()
        (safe_dir / "annotation" / "calibration" / f"noise-{meta.new_pattern}.xml").write_bytes(
            _edit_noise(noise_text, meta),
        )

        # --- RFI (optional) ---
        rfi_upath = product_root / f"annotation/rfi/rfi-{ann_upath.name}"
        if rfi_upath.exists():
            (safe_dir / "annotation" / "rfi").mkdir()
            (safe_dir / "annotation" / "rfi" / f"rfi-{meta.new_pattern}.xml").write_bytes(
                _edit_rfi(rfi_upath.read_text(), meta),
            )

        # --- Manifest (must come after annotation edit so footprint uses shifted lines) ---
        old_pattern = ann_upath.name[:-4]  # annotation filename without .xml extension
        footprint = _build_footprint(ann_root)  # ann_root is now the edited (shifted) tree
        (safe_dir / "manifest.safe").write_bytes(
            _edit_manifest(
                manifest_text,
                meta,
                sub_swath=swath,
                old_pattern=old_pattern,
                footprint=footprint,
            ),
        )

        # --- TIFF extraction ---
        tiff_s3_path = product_root / f"measurement/{ann_upath.name[:-4]}.tiff"
        dst_tiff = safe_dir / "measurement" / f"{meta.new_pattern}.tiff"
        logger.info(
            "Extracting burst TIFF (starting_line=%d, lines=%d)",
            meta.starting_line,
            meta.number_of_lines,
        )
        _extract_tiff(tiff_s3_path, dst_tiff, geoloc_pts=geoloc_pts, meta=meta)

        logger.info("Creating zip archive")
        zip_path = _zip_safe_dir(safe_dir)

        # Move local zip to destination (may be local or remote).
        # The outer zip filename is set by the caller via `save_path`; the
        # inner SAFE folder retains the ESA-conventional naming.
        save_upath.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(save_upath, Path):
            shutil.move(zip_path, save_upath)
        else:
            with zip_path.open("rb") as src, save_upath.open("wb") as out:
                shutil.copyfileobj(src, out)

    logger.info("Burst extracted to: %s", save_upath)
    return save_upath
