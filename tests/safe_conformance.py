"""Validate extracted single-burst SAFE products.

The checks describe the structure and metadata observed in CDSE's OData burst
products without reusing the Python extractor's implementation. Agreement is
therefore useful evidence rather than a tautology.

Two entry points matter:

- :func:`check_burst_safe` validates one extracted archive on its own terms and
  returns a list of contract violations (empty means conformant).
- :func:`compare_burst_safes` diffs two archives field by field, which is what
  the live test uses against CDSE's own burst product for the same burst.

:func:`digest` reduces an archive to a stable dictionary suitable for pinning
as a golden record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any
import warnings
import zipfile

from lxml import etree
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning

if TYPE_CHECKING:
    from collections.abc import Iterator

NS = {
    "xfdu": "urn:ccsds:schema:xfdu:1",
    "safe": "http://www.esa.int/safe/sentinel-1.0",
    "s1": "http://www.esa.int/safe/sentinel-1.0/sentinel-1",
    "s1sarl1": "http://www.esa.int/safe/sentinel-1.0/sentinel-1/sar/level-1",
    "gml": "http://www.opengis.net/gml",
}

# CDSE burst-specific files use this pattern:
# <15-char annotation prefix><6-digit burst id>-<compact sensing time>-<datatake>
BURST_PATTERN = re.compile(
    r"^s1[abcd]-(?:iw[123]|ew[12345])-slc-(?:vv|vh|hh|hv)"
    r"-\d{6}-\d{8}T\d{6}-\d{6}$",
)

SAFE_DIR_PATTERN = re.compile(
    r"^S1[ABCD]_SLC_\d{8}T\d{6}_\d{6}_(?:IW[123]|EW[12345])_(?:VV|VH|HH|HV)_\d{6}\.SAFE$",
)


@dataclass
class BurstSafe:
    """An extracted single-burst SAFE, read out of its archive."""

    safe_dir: str
    pattern: str
    members: list[str]
    manifest: etree._Element
    annotation: etree._Element
    calibration: etree._Element
    noise: etree._Element
    rfi: etree._Element | None
    measurement: np.ndarray = field(repr=False)
    gcps: list[tuple[float, float, float, float, float]] = field(repr=False)
    raster_crs: str | None = None
    raster_shape: tuple[int, int] = (0, 0)
    raster_dtype: str = ""

    @property
    def lines_per_burst(self) -> int:
        """Azimuth lines in the extracted burst."""
        return int(self.annotation.findtext("swathTiming/linesPerBurst"))

    @property
    def samples_per_burst(self) -> int:
        """Range samples in the extracted burst."""
        return int(self.annotation.findtext("swathTiming/samplesPerBurst"))

    @property
    def azimuth_start(self) -> str:
        """First-line azimuth time of the extracted burst."""
        return self.annotation.findtext("adsHeader/startTime")

    @property
    def azimuth_stop(self) -> str:
        """Last-line azimuth time of the extracted burst."""
        return self.annotation.findtext("adsHeader/stopTime")

    def geolocation_points(self) -> list[dict[str, float]]:
        """Return the geolocation grid as plain records, in document order."""
        return [
            {
                "line": float(point.findtext("line")),
                "pixel": float(point.findtext("pixel")),
                "longitude": float(point.findtext("longitude")),
                "latitude": float(point.findtext("latitude")),
                "height": float(point.findtext("height")),
            }
            for point in self.annotation.findall(
                "geolocationGrid/geolocationGridPointList/geolocationGridPoint",
            )
        ]


def expected_footprint(annotation: etree._Element) -> str:
    """Recompute the manifest footprint used by CDSE burst products.

    The footprint uses the first and last ``line=0`` points, then the last and
    first of *all* remaining points in document order, closing on the first
    point again.

    Two properties of that rule are worth stating, because both look like
    defects until checked against a real CDSE burst:

    - The second group deliberately spans the whole product's grid, not just the
      extracted burst, so an interior burst inherits a footprint larger than
      itself.
    - Coordinate text is copied verbatim, keeping the annotation's scientific
      notation and precision.
    """
    points = annotation.findall("geolocationGrid/geolocationGridPointList/geolocationGridPoint")
    top = [p for p in points if int(p.findtext("line")) == 0]
    rest = [p for p in points if int(p.findtext("line")) != 0]

    def coord(point: etree._Element) -> str:
        return f"{point.findtext('longitude')} {point.findtext('latitude')}"

    return ",".join(
        [coord(top[0]), coord(top[-1]), coord(rest[-1]), coord(rest[0]), coord(top[0])],
    )


def _parse(data: bytes) -> etree._Element:
    return etree.fromstring(data, parser=etree.XMLParser(resolve_entities=False, no_network=True))


def read_burst_safe(zip_path: Path) -> BurstSafe:
    """Read an extracted burst archive into a :class:`BurstSafe`.

    Parameters
    ----------
    zip_path : Path
        Path to the ``.zip`` produced by the extractor.

    Returns
    -------
    BurstSafe
        Parsed archive contents.

    Raises
    ------
    ValueError
        If the archive does not hold exactly one ``.SAFE`` root or is missing a
        mandatory annotation file.
    """
    with zipfile.ZipFile(zip_path) as archive:
        members = sorted(name for name in archive.namelist() if not name.endswith("/"))
        roots = {name.split("/", 1)[0] for name in members}
        if len(roots) != 1:
            msg = f"Expected exactly one .SAFE root in '{zip_path}', found {sorted(roots)}"
            raise ValueError(msg)
        safe_dir = roots.pop()

        annotations = [
            name
            for name in members
            if name.startswith(f"{safe_dir}/annotation/") and name.endswith(".xml")
        ]
        primary = [
            name for name in annotations if "/calibration/" not in name and "/rfi/" not in name
        ]
        if len(primary) != 1:
            msg = f"Expected exactly one swath annotation, found {primary}"
            raise ValueError(msg)
        pattern = Path(primary[0]).stem

        def read(name: str) -> bytes:
            return archive.read(name)

        manifest = _parse(read(f"{safe_dir}/manifest.safe"))
        annotation = _parse(read(primary[0]))
        calibration = _parse(read(f"{safe_dir}/annotation/calibration/calibration-{pattern}.xml"))
        noise = _parse(read(f"{safe_dir}/annotation/calibration/noise-{pattern}.xml"))
        rfi_name = f"{safe_dir}/annotation/rfi/rfi-{pattern}.xml"
        rfi = _parse(read(rfi_name)) if rfi_name in members else None

        tiff_name = f"{safe_dir}/measurement/{pattern}.tiff"
        tiff_bytes = read(tiff_name)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        with rasterio.MemoryFile(tiff_bytes) as memfile, memfile.open() as src:
            data = src.read(1)
            gcps, crs = src.gcps
            points = [(gcp.col, gcp.row, gcp.x, gcp.y, gcp.z) for gcp in gcps]
            raster_crs = str(crs) if crs else None
            shape = (src.height, src.width)
            dtype = src.dtypes[0]

    return BurstSafe(
        safe_dir=safe_dir,
        pattern=pattern,
        members=members,
        manifest=manifest,
        annotation=annotation,
        calibration=calibration,
        noise=noise,
        rfi=rfi,
        measurement=data,
        gcps=points,
        raster_crs=raster_crs,
        raster_shape=shape,
        raster_dtype=dtype,
    )


def _check_archive_layout(burst: BurstSafe) -> Iterator[str]:
    """Check the archive holds exactly the burst's own files and nothing else."""
    if not SAFE_DIR_PATTERN.match(burst.safe_dir):
        yield f"SAFE directory name is not conventional: '{burst.safe_dir}'"
    if not BURST_PATTERN.match(burst.pattern):
        yield f"Burst file pattern is not conventional: '{burst.pattern}'"

    expected = {
        f"{burst.safe_dir}/manifest.safe",
        f"{burst.safe_dir}/annotation/{burst.pattern}.xml",
        f"{burst.safe_dir}/annotation/calibration/calibration-{burst.pattern}.xml",
        f"{burst.safe_dir}/annotation/calibration/noise-{burst.pattern}.xml",
        f"{burst.safe_dir}/measurement/{burst.pattern}.tiff",
    }
    if burst.rfi is not None:
        expected.add(f"{burst.safe_dir}/annotation/rfi/rfi-{burst.pattern}.xml")

    unexpected = sorted(set(burst.members) - expected)
    if unexpected:
        yield f"Archive holds files outside the burst: {unexpected}"
    missing = sorted(expected - set(burst.members))
    if missing:
        yield f"Archive is missing burst files: {missing}"


def _check_annotation(burst: BurstSafe) -> Iterator[str]:
    """Check the swath annotation describes exactly one burst at line zero."""
    burst_list = burst.annotation.find("swathTiming/burstList")
    bursts = burst_list.findall("burst")
    if len(bursts) != 1:
        yield f"Annotation retains {len(bursts)} bursts, expected 1"
    if burst_list.get("count") != "1":
        yield f"burstList/@count is '{burst_list.get('count')}', expected '1'"

    image_info = burst.annotation.find("imageAnnotation/imageInformation")
    lines = burst.lines_per_burst
    if image_info.findtext("numberOfLines") != str(lines):
        yield (
            f"imageInformation/numberOfLines is "
            f"'{image_info.findtext('numberOfLines')}', expected '{lines}'"
        )

    azimuth_start = bursts[0].findtext("azimuthTime") if bursts else None
    if azimuth_start is not None and burst.azimuth_start != azimuth_start:
        yield f"adsHeader/startTime '{burst.azimuth_start}' != burst azimuthTime '{azimuth_start}'"

    interval = float(image_info.findtext("azimuthTimeInterval"))
    expected_stop = (
        datetime.fromisoformat(burst.azimuth_start) + timedelta(seconds=interval * (lines - 1))
    ).strftime("%Y-%m-%dT%H:%M:%S.%f")
    if burst.azimuth_stop != expected_stop:
        yield f"adsHeader/stopTime '{burst.azimuth_stop}' != expected '{expected_stop}'"

    for tag, value in (
        ("productFirstLineUtcTime", burst.azimuth_start),
        ("productLastLineUtcTime", burst.azimuth_stop),
    ):
        if image_info.findtext(tag) != value:
            yield f"imageInformation/{tag} '{image_info.findtext(tag)}' != '{value}'"

    grid_lines = sorted({int(point["line"]) for point in burst.geolocation_points()})
    if 0 not in grid_lines:
        yield f"Geolocation grid has no row on the burst's first line: {grid_lines}"
    if lines not in grid_lines:
        yield f"Geolocation grid has no row on the burst's last line ({lines}): {grid_lines}"


def _check_auxiliary_annotations(burst: BurstSafe) -> Iterator[str]:
    """Check calibration, noise and RFI headers were retimed to the burst."""
    for name, root, path in (
        ("calibration", burst.calibration, "calibrationVectorList/calibrationVector/line"),
        ("noise", burst.noise, "noiseRangeVectorList/noiseRangeVector/line"),
    ):
        for tag, value in (("startTime", burst.azimuth_start), ("stopTime", burst.azimuth_stop)):
            if root.findtext(f"adsHeader/{tag}") != value:
                yield f"{name} adsHeader/{tag} '{root.findtext(f'adsHeader/{tag}')}' != '{value}'"
        shifted = [int(el.text) for el in root.findall(path)]
        if not any(line <= 0 for line in shifted):
            yield f"{name} vector lines were not shifted to the burst origin: {shifted}"

    if burst.rfi is not None:
        for tag, value in (("startTime", burst.azimuth_start), ("stopTime", burst.azimuth_stop)):
            if burst.rfi.findtext(f"adsHeader/{tag}") != value:
                yield f"rfi adsHeader/{tag} != '{value}'"


def _check_manifest_metadata(burst: BurstSafe, *, sub_swath: str) -> Iterator[str]:
    """Check the manifest's acquisition period, footprint and swath list."""
    manifest = burst.manifest

    for tag, value in (
        ("startTime", burst.azimuth_start),
        ("stopTime", burst.azimuth_stop),
    ):
        found = [
            el.text
            for el in manifest.xpath(
                "metadataSection/metadataObject/metadataWrap/xmlData"
                f"/safe:acquisitionPeriod/safe:{tag}",
                namespaces=NS,
            )
        ]
        if found != [value]:
            yield f"manifest acquisitionPeriod/{tag} is {found}, expected {value}"

    coordinates = [
        el.text
        for el in manifest.xpath(
            "metadataSection/metadataObject/metadataWrap/xmlData"
            "/safe:frameSet/safe:frame/safe:footPrint/gml:coordinates",
            namespaces=NS,
        )
    ]
    footprint = expected_footprint(burst.annotation)
    if coordinates != [footprint]:
        yield f"manifest footprint is {coordinates}, expected '{footprint}'"

    swaths = [
        el.text
        for el in manifest.xpath(
            "metadataSection/metadataObject/metadataWrap/xmlData/safe:platform"
            "/safe:instrument/safe:extension/s1sarl1:instrumentMode/s1sarl1:swath",
            namespaces=NS,
        )
    ]
    if swaths != [sub_swath.upper()]:
        yield f"manifest lists swaths {swaths}, expected ['{sub_swath.upper()}']"


def _check_manifest_pruning(burst: BurstSafe) -> Iterator[str]:
    """Check every object and pointer left in the manifest belongs to this burst."""
    manifest = burst.manifest
    short = burst.pattern.replace("-", "")

    for object_id in manifest.xpath("dataObjectSection/dataObject/@ID"):
        if short not in object_id:
            yield f"manifest retains a foreign dataObject '{object_id}'"
    for removed in ("quicklook", "mapoverlay", "productpreview"):
        if manifest.xpath(f"dataObjectSection/dataObject[@ID='{removed}']"):
            yield f"manifest retains the '{removed}' dataObject"

    if manifest.xpath("metadataSection/metadataObject[@classification='SYNTAX']"):
        yield "manifest retains schema-reference (SYNTAX) metadataObjects"

    for pointer_id in manifest.xpath(
        "metadataSection/metadataObject/dataObjectPointer/@dataObjectID",
    ):
        if short not in pointer_id:
            yield f"manifest retains a foreign metadata pointer '{pointer_id}'"
    for pointer_id in manifest.xpath(
        "informationPackageMap/xfdu:contentUnit/xfdu:contentUnit/dataObjectPointer/@dataObjectID",
        namespaces=NS,
    ):
        if short not in pointer_id:
            yield f"manifest retains a foreign content unit pointer '{pointer_id}'"


def _check_manifest_byte_streams(burst: BurstSafe) -> Iterator[str]:
    """Check the manifest's file references resolve and its integrity fields are blank."""
    manifest = burst.manifest

    for href in manifest.xpath("dataObjectSection/dataObject/byteStream/fileLocation/@href"):
        member = f"{burst.safe_dir}/{str(href).removeprefix('./')}"
        if member not in burst.members:
            yield f"manifest references a missing archive member '{href}'"

    # CDSE burst products blank these fields rather than recomputing them.
    for checksum in manifest.xpath("dataObjectSection/dataObject/byteStream/checksum"):
        if (checksum.text or "") != "":
            yield f"manifest byteStream checksum was not blanked: '{checksum.text}'"
    for stream in manifest.xpath("dataObjectSection/dataObject/byteStream"):
        if stream.get("size") != "":
            yield f"manifest byteStream/@size was not blanked: '{stream.get('size')}'"


def _check_manifest(burst: BurstSafe, *, sub_swath: str) -> Iterator[str]:
    """Check the manifest was pruned and retimed to this burst alone."""
    yield from _check_manifest_metadata(burst, sub_swath=sub_swath)
    yield from _check_manifest_pruning(burst)
    yield from _check_manifest_byte_streams(burst)


def _check_measurement(burst: BurstSafe) -> Iterator[str]:
    """Check the raster window and its GCPs describe the extracted burst."""
    expected_shape = (burst.lines_per_burst, burst.samples_per_burst)
    if burst.raster_shape != expected_shape:
        yield f"measurement is {burst.raster_shape}, expected {expected_shape}"

    if burst.raster_crs is None or "4326" not in burst.raster_crs:
        yield f"measurement GCP CRS is '{burst.raster_crs}', expected EPSG:4326"

    grid = burst.geolocation_points()
    if len(burst.gcps) != len(grid):
        yield f"measurement carries {len(burst.gcps)} GCPs, expected {len(grid)}"
        return

    for gcp, point in zip(burst.gcps, grid, strict=True):
        col, row, lon, lat, height = gcp
        mismatch = (
            col != point["pixel"]
            or row != point["line"]
            or lon != point["longitude"]
            or lat != point["latitude"]
            or height != point["height"]
        )
        if mismatch:
            yield f"GCP {gcp} does not match geolocation point {point}"
            return


def check_burst_safe(zip_path: Path, *, sub_swath: str) -> list[str]:
    """Validate an extracted burst archive against the CDSE product contract.

    Parameters
    ----------
    zip_path : Path
        Archive produced by the extractor.
    sub_swath : str
        Sub-swath the burst was extracted from, e.g. ``"IW1"``.

    Returns
    -------
    list of str
        Contract violations; an empty list means the archive is conformant.
    """
    burst = read_burst_safe(zip_path)
    return [
        *_check_archive_layout(burst),
        *_check_annotation(burst),
        *_check_auxiliary_annotations(burst),
        *_check_manifest(burst, sub_swath=sub_swath),
        *_check_measurement(burst),
    ]


def _canonical_xml(root: etree._Element) -> str:
    """Serialise a tree so that formatting differences do not register as content diffs."""
    copy = etree.fromstring(etree.tostring(root))
    for element in copy.iter():
        if element.text is not None:
            element.text = element.text.strip() or None
        if element.tail is not None:
            element.tail = None
    return etree.tostring(copy, method="c14n2", strip_text=True).decode()


def digest(zip_path: Path) -> dict[str, Any]:
    """Reduce an extracted burst archive to a stable, comparable record.

    Formatting-insensitive: XML is canonicalised and the raster is hashed, so
    the record changes only when the extraction's content changes.

    Parameters
    ----------
    zip_path : Path
        Archive produced by the extractor.

    Returns
    -------
    dict
        Golden record of the archive's structure and content.
    """
    burst = read_burst_safe(zip_path)
    return {
        "safe_dir": burst.safe_dir,
        "pattern": burst.pattern,
        "members": [name.split("/", 1)[1] for name in burst.members],
        "azimuth_start": burst.azimuth_start,
        "azimuth_stop": burst.azimuth_stop,
        "lines_per_burst": burst.lines_per_burst,
        "samples_per_burst": burst.samples_per_burst,
        "raster_shape": list(burst.raster_shape),
        "raster_dtype": burst.raster_dtype,
        "raster_crs": burst.raster_crs,
        "raster_sha256": hashlib.sha256(
            np.ascontiguousarray(burst.measurement).tobytes(),
        ).hexdigest(),
        "gcps": [list(gcp) for gcp in burst.gcps],
        "footprint": expected_footprint(burst.annotation),
        "manifest_sha256": hashlib.sha256(_canonical_xml(burst.manifest).encode()).hexdigest(),
        "annotation_sha256": hashlib.sha256(_canonical_xml(burst.annotation).encode()).hexdigest(),
        "calibration_sha256": hashlib.sha256(
            _canonical_xml(burst.calibration).encode(),
        ).hexdigest(),
        "noise_sha256": hashlib.sha256(_canonical_xml(burst.noise).encode()).hexdigest(),
    }


def compare_burst_safes(actual: Path, reference: Path) -> list[str]:
    """Diff an extracted burst archive against a trusted reference archive.

    Both archives must describe the same burst. File names, XML content, GCPs
    and pixel values are compared; the archives' compression and member order
    are not.

    Parameters
    ----------
    actual : Path
        Archive produced by this package.
    reference : Path
        Archive produced by CDSE for the same burst.

    Returns
    -------
    list of str
        Differences; an empty list means the two archives agree.
    """
    ours = read_burst_safe(actual)
    theirs = read_burst_safe(reference)

    differences: list[str] = []
    if ours.pattern != theirs.pattern:
        differences.append(f"burst pattern '{ours.pattern}' != '{theirs.pattern}'")
    if ours.safe_dir != theirs.safe_dir:
        differences.append(f"SAFE directory '{ours.safe_dir}' != '{theirs.safe_dir}'")

    our_members = {name.split("/", 1)[1] for name in ours.members}
    their_members = {name.split("/", 1)[1] for name in theirs.members}
    if our_members != their_members:
        differences.append(
            f"member set differs: only ours {sorted(our_members - their_members)}, "
            f"only reference {sorted(their_members - our_members)}",
        )

    for name, ours_root, theirs_root in (
        ("annotation", ours.annotation, theirs.annotation),
        ("calibration", ours.calibration, theirs.calibration),
        ("noise", ours.noise, theirs.noise),
        ("manifest", ours.manifest, theirs.manifest),
    ):
        differences.extend(f"{name}: {diff}" for diff in _diff_xml(ours_root, theirs_root))

    if ours.raster_shape != theirs.raster_shape:
        differences.append(f"raster shape {ours.raster_shape} != {theirs.raster_shape}")
    elif not np.array_equal(ours.measurement, theirs.measurement):
        wrong = int(np.count_nonzero(ours.measurement != theirs.measurement))
        differences.append(f"raster differs in {wrong} of {ours.measurement.size} pixels")

    if ours.raster_dtype != theirs.raster_dtype:
        differences.append(f"raster dtype {ours.raster_dtype} != {theirs.raster_dtype}")

    # Compare the CRS separately from the GCP positions: an implementation can
    # produce the right points with no projection and still match position for
    # position.
    if ours.raster_crs != theirs.raster_crs:
        differences.append(f"GCP CRS {ours.raster_crs!r} != {theirs.raster_crs!r}")

    differences.extend(_diff_gcps(ours, theirs))

    return differences


def _diff_gcps(ours: BurstSafe, theirs: BurstSafe) -> Iterator[str]:
    """Diff GCPs, allowing the one deviation CDSE's service is known to have.

    Some products CDSE currently serves subtract the burst's starting line from
    grid rows that were already shifted, leaving every GCP row one whole burst
    offset too far. The S3 extractor emits the correctly shifted rows.

    A uniform row offset that is a whole multiple of the burst length is
    therefore reported as that known deviation rather than as a mismatch. Any
    other difference — in position, coordinates, height, or count — is a real
    disagreement and is reported as one.
    """
    if len(ours.gcps) != len(theirs.gcps):
        yield f"GCP count {len(ours.gcps)} != {len(theirs.gcps)}"
        return

    offsets = set()
    for ours_gcp, theirs_gcp in zip(ours.gcps, theirs.gcps, strict=True):
        our_col, our_row, *our_position = ours_gcp
        their_col, their_row, *their_position = theirs_gcp
        if our_col != their_col or our_position != their_position:
            yield f"GCP {ours_gcp} does not match reference GCP {theirs_gcp}"
            return
        offsets.add(their_row - our_row)

    if offsets == {0}:
        return

    lines = ours.lines_per_burst
    if len(offsets) == 1 and (offset := next(iter(offsets))) % lines == 0:
        yield (
            f"KNOWN DEVIATION: reference GCP rows are offset by {offset:.0f} lines "
            f"({offset / lines:.0f} bursts), the known CDSE double shift"
        )
        return

    yield f"GCP rows differ by inconsistent offsets: {sorted(offsets)}"


def _element_records(root: etree._Element) -> dict[str, list[str]]:
    """Flatten a tree into ``{path: [values]}``, including attributes."""
    records: dict[str, list[str]] = {}
    tree = root.getroottree()
    for element in root.iter():
        if not isinstance(element.tag, str):
            continue
        path = tree.getelementpath(element)
        text = (element.text or "").strip()
        if text or not len(element):
            records.setdefault(path, []).append(text)
        for key, value in sorted(element.attrib.items()):
            records.setdefault(f"{path}/@{key}", []).append(value)
    return records


def _diff_xml(ours: etree._Element, theirs: etree._Element) -> Iterator[str]:
    """Yield per-path differences between two trees."""
    our_records = _element_records(ours)
    their_records = _element_records(theirs)
    for path in sorted(set(our_records) | set(their_records)):
        if our_records.get(path) != their_records.get(path):
            yield f"{path}: {our_records.get(path)} != {their_records.get(path)}"
