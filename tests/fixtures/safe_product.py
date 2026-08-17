"""Build a synthetic Sentinel-1 SLC SAFE product on disk.

The generated tree mirrors the structure the burst extractor relies on: the
XFDU manifest with its namespace layout, information-package map, metadata
pointers and data objects, the swath annotation with its burst list and
geolocation grid, the calibration/noise/RFI annotations, and a striped CInt16
measurement TIFF. Everything is small enough to build in milliseconds, so the
extractor can run end to end offline against it.

Timings are derived from the real orbital constants so that the ``burstId``
lookup and the legacy orbital-mechanics fallback both resolve to the same,
known burst numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
import textwrap
import warnings

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning

# Orbital constants, mirroring s1bursts.burst_extractor._SUBSWATH_TIMING.
_T0_DELTA = {
    "iw1": Decimal("0.81531490") / 2,
    "iw2": Decimal("1.06087913") / 2,
    "iw3": Decimal("0.83045822") / 2,
    "ew1": Decimal("0.66551400") / 2,
    "ew2": Decimal("0.54139519") / 2,
    "ew3": Decimal("0.59473403") / 2,
    "ew4": Decimal("0.54745052") / 2,
    "ew5": Decimal("0.60098234") / 2,
}
_TBEAM = {"iw": Decimal("2.758273"), "ew": Decimal("3.038376")}
_T_PRE = {"iw": Decimal("2.299849"), "ew": Decimal("2.299970")}
_ORBIT_PERIOD = Decimal(12) * Decimal(24) * Decimal(3600) / Decimal(175)

_ANNOTATION_TIME_FMT = "%Y%m%dt%H%M%S"
_XML_TIME_FMT = "%Y-%m-%dT%H:%M:%S.%f"

# Geolocation grid columns, as in a real product: evenly spaced across the swath
# with an explicit final column on the last sample.
_GRID_COLUMNS = 5


@dataclass(frozen=True)
class ProductSpec:
    """Parameters of a synthetic SLC product.

    Defaults describe a small IW1/VV product with four bursts; raster
    dimensions are far below a real product's so the fixture stays cheap.
    """

    sub_swath: str = "IW1"
    polarization: str = "VV"
    n_bursts: int = 4
    lines_per_burst: int = 12
    samples_per_burst: int = 20
    first_burst_id: int = 30343
    with_burst_id: bool = True
    with_rfi: bool = False
    platform: str = "A"
    rel_orbit_number: int = 15
    abs_orbit_number: int = 44762
    datatake_id: int = 349850
    ascending_node_time: str = "2022-08-25T17:03:12.299795"
    azimuth_time_interval: str = "2.055556299999998e-03"
    image_number: str = "001"

    @property
    def mode(self) -> str:
        """Acquisition mode implied by the sub-swath."""
        return self.sub_swath[:2].upper()

    @property
    def swath(self) -> str:
        """Lower-case sub-swath identifier, as used in filenames."""
        return self.sub_swath.lower()

    @property
    def pol(self) -> str:
        """Lower-case polarization, as used in filenames."""
        return self.polarization.lower()

    @property
    def burst_ids(self) -> list[int]:
        """Relative burst IDs of every burst in the product, in order."""
        return [self.first_burst_id + i for i in range(self.n_bursts)]

    @property
    def number_of_lines(self) -> int:
        """Total azimuth lines across all bursts."""
        return self.n_bursts * self.lines_per_burst


@dataclass(frozen=True)
class SyntheticProduct:
    """A synthetic SAFE product written to disk."""

    spec: ProductSpec
    root: Path
    name: str
    annotation_name: str
    sensing_times: list[str]
    azimuth_times: list[str]
    measurement: np.ndarray = field(repr=False)

    def burst_window(self, burst_number: int) -> np.ndarray:
        """Return the measurement rows belonging to a 1-based burst number."""
        start = (burst_number - 1) * self.spec.lines_per_burst
        return self.measurement[start : start + self.spec.lines_per_burst]


def _sensing_time_for(spec: ProductSpec, burst_id: int) -> Decimal:
    """Invert the orbital burst-ID formula to get a sensing time for ``burst_id``.

    The half-burst offset places the result mid-interval, so the truncation the
    extractor applies lands on ``burst_id`` with room to spare for the
    microsecond rounding of the serialised timestamp.
    """
    family = spec.swath[:2]
    t_anx = Decimal(
        str(
            datetime.fromisoformat(spec.ascending_node_time).replace(tzinfo=UTC).timestamp(),
        ),
    )
    return (
        t_anx
        - _T0_DELTA[spec.swath]
        - (spec.rel_orbit_number - 1) * _ORBIT_PERIOD
        + _T_PRE[family]
        + (Decimal(burst_id) + Decimal("0.5") - 1) * _TBEAM[family]
    )


def _format_epoch(seconds: Decimal) -> str:
    """Format a POSIX timestamp as a SAFE-style UTC timestamp."""
    return datetime.fromtimestamp(float(seconds), tz=UTC).strftime(_XML_TIME_FMT)


def _geolocation_points(spec: ProductSpec) -> list[dict[str, object]]:
    """Build the geolocation grid.

    Rows sit on burst boundaries, as in a real product: one row at each burst's
    first line plus a closing row at the end of the last burst.
    """
    points: list[dict[str, object]] = []
    for row in range(spec.n_bursts + 1):
        line = row * spec.lines_per_burst
        for col in range(_GRID_COLUMNS):
            pixel = round(col * (spec.samples_per_burst - 1) / (_GRID_COLUMNS - 1))
            points.append(
                {
                    "line": line,
                    "pixel": pixel,
                    # A smooth, strictly monotonic ramp keeps every point distinct
                    # so that ordering mistakes are visible in the footprint.
                    "longitude": round(7.6 + 0.05 * col + 0.001 * row, 6),
                    "latitude": round(46.3 + 0.2 * row + 0.002 * col, 6),
                    "height": round(400.0 + row + col, 3),
                },
            )
    return points


def _annotation_xml(spec: ProductSpec, product: dict[str, object]) -> str:
    """Render the swath annotation document."""
    bursts = []
    for i, (sensing, azimuth) in enumerate(
        zip(product["sensing_times"], product["azimuth_times"], strict=True),
    ):
        burst_id_el = (
            f"<burstId>{spec.burst_ids[i]}</burstId>\n          " if spec.with_burst_id else ""
        )
        first_valid = " ".join(["0"] * spec.lines_per_burst)
        last_valid = " ".join([str(spec.samples_per_burst - 1)] * spec.lines_per_burst)
        bursts.append(
            f"""        <burst>
          <azimuthTime>{azimuth}</azimuthTime>
          <azimuthAnxTime>{2000.0 + i:.6e}</azimuthAnxTime>
          <sensingTime>{sensing}</sensingTime>
          {burst_id_el}<byteOffset>{100_000 + i * 10_000}</byteOffset>
          <firstValidSample count="{spec.lines_per_burst}">{first_valid}</firstValidSample>
          <lastValidSample count="{spec.lines_per_burst}">{last_valid}</lastValidSample>
        </burst>""",
        )

    grid = [
        f"""      <geolocationGridPoint>
        <azimuthTime>{product["start_time"]}</azimuthTime>
        <slantRangeTime>5.3e-03</slantRangeTime>
        <line>{point["line"]}</line>
        <pixel>{point["pixel"]}</pixel>
        <latitude>{point["latitude"]}</latitude>
        <longitude>{point["longitude"]}</longitude>
        <height>{point["height"]}</height>
        <incidenceAngle>3.4e+01</incidenceAngle>
        <elevationAngle>3.0e+01</elevationAngle>
      </geolocationGridPoint>"""
        for point in _geolocation_points(spec)
    ]

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<product>
  <adsHeader>
    <missionId>S1{spec.platform}</missionId>
    <productType>SLC</productType>
    <polarisation>{spec.polarization}</polarisation>
    <mode>{spec.mode}</mode>
    <swath>{spec.sub_swath}</swath>
    <startTime>{product["start_time"]}</startTime>
    <stopTime>{product["stop_time"]}</stopTime>
    <absoluteOrbitNumber>{spec.abs_orbit_number}</absoluteOrbitNumber>
    <missionDataTakeId>{spec.datatake_id}</missionDataTakeId>
    <imageNumber>{spec.image_number}</imageNumber>
  </adsHeader>
  <imageAnnotation>
    <imageInformation>
      <productFirstLineUtcTime>{product["start_time"]}</productFirstLineUtcTime>
      <productLastLineUtcTime>{product["stop_time"]}</productLastLineUtcTime>
      <azimuthTimeInterval>{spec.azimuth_time_interval}</azimuthTimeInterval>
      <rangePixelSpacing>2.329562e+00</rangePixelSpacing>
      <azimuthPixelSpacing>1.397905e+01</azimuthPixelSpacing>
      <numberOfSamples>{spec.samples_per_burst}</numberOfSamples>
      <numberOfLines>{spec.number_of_lines}</numberOfLines>
    </imageInformation>
  </imageAnnotation>
  <swathTiming>
    <linesPerBurst>{spec.lines_per_burst}</linesPerBurst>
    <samplesPerBurst>{spec.samples_per_burst}</samplesPerBurst>
    <burstList count="{spec.n_bursts}">
{chr(10).join(bursts)}
    </burstList>
  </swathTiming>
  <geolocationGrid>
    <geolocationGridPointList count="{(spec.n_bursts + 1) * _GRID_COLUMNS}">
{chr(10).join(grid)}
    </geolocationGridPointList>
  </geolocationGrid>
</product>
"""


def _calibration_xml(spec: ProductSpec, product: dict[str, object]) -> str:
    """Render the calibration annotation, with one vector per burst boundary."""
    vectors = [
        f"""    <calibrationVector>
      <azimuthTime>{product["start_time"]}</azimuthTime>
      <line>{row * spec.lines_per_burst}</line>
      <pixel count="2">0 {spec.samples_per_burst - 1}</pixel>
      <sigmaNought count="2">1.0 1.0</sigmaNought>
      <betaNought count="2">2.37e+02 2.37e+02</betaNought>
      <gamma count="2">1.0 1.0</gamma>
      <dn count="2">1.0 1.0</dn>
    </calibrationVector>"""
        for row in range(spec.n_bursts + 1)
    ]
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<calibration>
  <adsHeader>
    <missionId>S1{spec.platform}</missionId>
    <polarisation>{spec.polarization}</polarisation>
    <swath>{spec.sub_swath}</swath>
    <startTime>{product["start_time"]}</startTime>
    <stopTime>{product["stop_time"]}</stopTime>
  </adsHeader>
  <calibrationInformation>
    <absoluteCalibrationConstant>1.0</absoluteCalibrationConstant>
  </calibrationInformation>
  <calibrationVectorList count="{spec.n_bursts + 1}">
{chr(10).join(vectors)}
  </calibrationVectorList>
</calibration>
"""


def _noise_xml(spec: ProductSpec, product: dict[str, object]) -> str:
    """Render the noise annotation, with one range vector per burst boundary."""
    vectors = [
        f"""    <noiseRangeVector>
      <azimuthTime>{product["start_time"]}</azimuthTime>
      <line>{row * spec.lines_per_burst}</line>
      <pixel count="2">0 {spec.samples_per_burst - 1}</pixel>
      <noiseRangeLut count="2">1.0e+02 1.1e+02</noiseRangeLut>
    </noiseRangeVector>"""
        for row in range(spec.n_bursts + 1)
    ]
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<noise>
  <adsHeader>
    <missionId>S1{spec.platform}</missionId>
    <polarisation>{spec.polarization}</polarisation>
    <swath>{spec.sub_swath}</swath>
    <startTime>{product["start_time"]}</startTime>
    <stopTime>{product["stop_time"]}</stopTime>
  </adsHeader>
  <noiseRangeVectorList count="{spec.n_bursts + 1}">
{chr(10).join(vectors)}
  </noiseRangeVectorList>
</noise>
"""


def _rfi_xml(spec: ProductSpec, product: dict[str, object]) -> str:
    """Render the RFI annotation."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rfi>
  <adsHeader>
    <missionId>S1{spec.platform}</missionId>
    <polarisation>{spec.polarization}</polarisation>
    <swath>{spec.sub_swath}</swath>
    <startTime>{product["start_time"]}</startTime>
    <stopTime>{product["stop_time"]}</stopTime>
  </adsHeader>
  <rfiMitigationApplied>BLANKING</rfiMitigationApplied>
  <rfiDetectionFromNoiseReportList count="1">
    <rfiDetectionFromNoiseReport>
      <azimuthTime>{product["start_time"]}</azimuthTime>
      <line>{spec.lines_per_burst}</line>
    </rfiDetectionFromNoiseReport>
  </rfiDetectionFromNoiseReportList>
</rfi>
"""


def _manifest_xml(spec: ProductSpec, product: dict[str, object]) -> str:
    """Render the XFDU manifest.

    Namespace usage matches a real product: the root and content units are
    ``xfdu``-prefixed while ``metadataSection`` and ``dataObjectSection`` sit in
    no namespace, which is what the extractor's XPaths assume.
    """
    stem: str = product["annotation_stem"]  # type: ignore[assignment]
    short = stem.replace("-", "")
    all_swaths = (
        ["IW1", "IW2", "IW3"] if spec.mode == "IW" else ["EW1", "EW2", "EW3", "EW4", "EW5"]
    )
    swath_els = "\n".join(
        f"                    <s1sarl1:swath>{sw}</s1sarl1:swath>" for sw in all_swaths
    )

    data_objects = [
        ("product", f"product{short}", "s1Level1ProductSchema", f"./annotation/{stem}.xml"),
        (
            "calibration",
            f"calibration{short}",
            "s1Level1CalibrationSchema",
            f"./annotation/calibration/calibration-{stem}.xml",
        ),
        (
            "noise",
            f"noise{short}",
            "s1Level1NoiseSchema",
            f"./annotation/calibration/noise-{stem}.xml",
        ),
        ("measurement", short, "s1Level1MeasurementSchema", f"./measurement/{stem}.tiff"),
    ]
    if spec.with_rfi:
        data_objects.insert(
            3,
            ("rfi", f"rfi{short}", "s1Level1RfiSchema", f"./annotation/rfi/rfi-{stem}.xml"),
        )

    # A second sub-swath's objects, present in every real multi-swath product.
    other_swath = "IW2" if spec.sub_swath != "IW2" else "IW3"
    other_stem = stem.replace(spec.swath, other_swath.lower())
    other_short = other_stem.replace("-", "")

    object_els = "\n".join(
        f"""    <dataObject ID="{obj_id}" repID="{rep_id}">
      <byteStream mimeType="{"text/xml" if kind != "measurement" else "application/octet-stream"}" size="123456">
        <fileLocation locatorType="URL" href="{href}"/>
        <checksum checksumName="MD5">d41d8cd98f00b204e9800998ecf8427e</checksum>
      </byteStream>
    </dataObject>"""
        for kind, obj_id, rep_id, href in data_objects
    )
    other_object_els = f"""    <dataObject ID="product{other_short}" repID="s1Level1ProductSchema">
      <byteStream mimeType="text/xml" size="123456">
        <fileLocation locatorType="URL" href="./annotation/{other_stem}.xml"/>
        <checksum checksumName="MD5">d41d8cd98f00b204e9800998ecf8427e</checksum>
      </byteStream>
    </dataObject>
    <dataObject ID="{other_short}" repID="s1Level1MeasurementSchema">
      <byteStream mimeType="application/octet-stream" size="123456">
        <fileLocation locatorType="URL" href="./measurement/{other_stem}.tiff"/>
        <checksum checksumName="MD5">d41d8cd98f00b204e9800998ecf8427e</checksum>
      </byteStream>
    </dataObject>"""

    preview_els = "\n".join(
        f"""    <dataObject ID="{obj_id}" repID="s1Level1{obj_id.capitalize()}Schema">
      <byteStream mimeType="{mime}" size="123456">
        <fileLocation locatorType="URL" href="{href}"/>
        <checksum checksumName="MD5">d41d8cd98f00b204e9800998ecf8427e</checksum>
      </byteStream>
    </dataObject>"""
        for obj_id, mime, href in (
            ("quicklook", "image/png", "./preview/quick-look.png"),
            ("mapoverlay", "application/xml", "./preview/map-overlay.kml"),
            ("productpreview", "text/html", "./preview/product-preview.html"),
        )
    )

    pointer_els = "\n".join(
        f"""    <metadataObject ID="{kind}Annotation" classification="POINTER" category="DMD">
      <dataObjectPointer dataObjectID="{obj_id}"/>
    </metadataObject>"""
        for kind, obj_id, _rep, _href in data_objects
    )
    other_pointer_el = f"""    <metadataObject ID="otherSwathAnnotation" classification="POINTER" category="DMD">
      <dataObjectPointer dataObjectID="product{other_short}"/>
    </metadataObject>"""

    content_units = "\n".join(
        f"""      <xfdu:contentUnit unitType="Metadata Unit" repID="{rep_id}" dmdID="{kind}Annotation">
        <dataObjectPointer dataObjectID="{obj_id}"/>
      </xfdu:contentUnit>"""
        for kind, obj_id, rep_id, _href in data_objects
    )
    other_content_unit = f"""      <xfdu:contentUnit unitType="Metadata Unit" repID="s1Level1ProductSchema" dmdID="otherSwathAnnotation">
        <dataObjectPointer dataObjectID="product{other_short}"/>
      </xfdu:contentUnit>"""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<xfdu:XFDU xmlns:xfdu="urn:ccsds:schema:xfdu:1" xmlns:safe="http://www.esa.int/safe/sentinel-1.0" xmlns:s1="http://www.esa.int/safe/sentinel-1.0/sentinel-1" xmlns:s1sarl1="http://www.esa.int/safe/sentinel-1.0/sentinel-1/sar/level-1" xmlns:gml="http://www.opengis.net/gml" version="esa/safe/sentinel-1.0/sentinel-1/sar/level-1/slc/standard/{spec.mode.lower()}dp">
  <informationPackageMap>
    <xfdu:contentUnit unitType="SAFE Archive Information Package" textInfo="Sentinel-1 Level-1 SLC Product" dmdID="acquisitionPeriod platform measurementOrbitReference measurementFrameSet" pdiID="processing">
{content_units}
{other_content_unit}
    </xfdu:contentUnit>
  </informationPackageMap>
  <metadataSection>
    <metadataObject ID="acquisitionPeriod" classification="DESCRIPTION" category="DMD">
      <metadataWrap mimeType="text/xml" vocabularyName="Safe" textInfo="Acquisition Period">
        <xmlData>
          <safe:acquisitionPeriod>
            <safe:startTime>{product["start_time"]}</safe:startTime>
            <safe:stopTime>{product["stop_time"]}</safe:stopTime>
          </safe:acquisitionPeriod>
        </xmlData>
      </metadataWrap>
    </metadataObject>
    <metadataObject ID="platform" classification="DESCRIPTION" category="DMD">
      <metadataWrap mimeType="text/xml" vocabularyName="Safe" textInfo="Platform Description">
        <xmlData>
          <safe:platform>
            <safe:nssdcIdentifier>2014-016A</safe:nssdcIdentifier>
            <safe:familyName>SENTINEL-1</safe:familyName>
            <safe:number>{spec.platform}</safe:number>
            <safe:instrument>
              <safe:familyName abbreviation="SAR">Synthetic Aperture Radar</safe:familyName>
              <safe:extension>
                <s1sarl1:instrumentMode>
                  <s1sarl1:mode>{spec.mode}</s1sarl1:mode>
{swath_els}
                </s1sarl1:instrumentMode>
              </safe:extension>
            </safe:instrument>
          </safe:platform>
        </xmlData>
      </metadataWrap>
    </metadataObject>
    <metadataObject ID="measurementOrbitReference" classification="DESCRIPTION" category="DMD">
      <metadataWrap mimeType="text/xml" vocabularyName="Safe" textInfo="Orbit Reference">
        <xmlData>
          <safe:orbitReference>
            <safe:orbitNumber type="start">{spec.abs_orbit_number}</safe:orbitNumber>
            <safe:relativeOrbitNumber type="start">{spec.rel_orbit_number}</safe:relativeOrbitNumber>
            <safe:extension>
              <s1:orbitProperties>
                <s1:pass>DESCENDING</s1:pass>
                <s1:ascendingNodeTime>{spec.ascending_node_time}</s1:ascendingNodeTime>
              </s1:orbitProperties>
            </safe:extension>
          </safe:orbitReference>
        </xmlData>
      </metadataWrap>
    </metadataObject>
    <metadataObject ID="measurementFrameSet" classification="DESCRIPTION" category="DMD">
      <metadataWrap mimeType="text/xml" vocabularyName="Safe" textInfo="Frame Set">
        <xmlData>
          <safe:frameSet>
            <safe:frame>
              <safe:footPrint srsName="http://www.opengis.net/gml/srs/epsg.xml#4326">
                <gml:coordinates>0.0 0.0,0.0 0.0,0.0 0.0,0.0 0.0,0.0 0.0</gml:coordinates>
              </safe:footPrint>
            </safe:frame>
          </safe:frameSet>
        </xmlData>
      </metadataWrap>
    </metadataObject>
    <metadataObject ID="s1Level1ProductSchema" classification="SYNTAX" category="REP">
      <metadataReference mimeType="text/xml" vocabularyName="SDF" locatorType="URL" href="./support/s1-level-1-product.xsd"/>
    </metadataObject>
    <metadataObject ID="s1Level1MeasurementSchema" classification="SYNTAX" category="REP">
      <metadataReference mimeType="text/xml" vocabularyName="SDF" locatorType="URL" href="./support/s1-level-1-measurement.xsd"/>
    </metadataObject>
{pointer_els}
{other_pointer_el}
  </metadataSection>
  <dataObjectSection>
{object_els}
{other_object_els}
{preview_els}
  </dataObjectSection>
</xfdu:XFDU>
"""


def _write_measurement(path: Path, spec: ProductSpec) -> np.ndarray:
    """Write a striped CInt16 measurement TIFF and return its pixel array.

    A per-line ramp makes every burst window distinguishable, so an extraction
    that reads the wrong rows shows up immediately.
    """
    rows, cols = spec.number_of_lines, spec.samples_per_burst
    real = np.arange(rows * cols, dtype=np.int32).reshape(rows, cols) % 3001
    imag = (real * 7 + 11) % 2999
    data = (real + 1j * imag).astype("complex64")[np.newaxis]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=cols,
            height=rows,
            count=1,
            dtype="complex_int16",
            tiled=False,
            blockysize=1,
        ) as dst:
            dst.write(data)
            dst.update_tags(TIFFTAG_IMAGEDESCRIPTION="Sentinel-1 IW SLC")
    return data[0]


def build_product(
    base_dir: Path,
    spec: ProductSpec | None = None,
) -> SyntheticProduct:
    """Write a synthetic SLC product below ``base_dir`` and describe it.

    Parameters
    ----------
    base_dir : Path
        Directory to create the ``.SAFE`` tree in.
    spec : ProductSpec, optional
        Product parameters; defaults to a four-burst IW1/VV product.

    Returns
    -------
    SyntheticProduct
        The written product, its ``.SAFE`` root, and the values needed to
        assert on the extracted burst.
    """
    spec = spec or ProductSpec()

    sensing_decimals = [_sensing_time_for(spec, bid) for bid in spec.burst_ids]
    sensing_times = [_format_epoch(value) for value in sensing_decimals]
    # A burst's azimuth (first-line) time sits a little before its sensing
    # (mid-burst Doppler-centroid) time, as in a real product.
    azimuth_times = [
        (
            datetime.fromisoformat(sensing)
            - timedelta(seconds=float(spec.azimuth_time_interval)) * (spec.lines_per_burst // 2)
        ).strftime(_XML_TIME_FMT)
        for sensing in sensing_times
    ]

    start_dt = datetime.fromisoformat(azimuth_times[0])
    stop_dt = start_dt + timedelta(
        seconds=float(spec.azimuth_time_interval) * (spec.number_of_lines - 1),
    )
    start_time = start_dt.strftime(_XML_TIME_FMT)
    stop_time = stop_dt.strftime(_XML_TIME_FMT)

    annotation_stem = (
        f"s1{spec.platform.lower()}-{spec.swath}-slc-{spec.pol}"
        f"-{start_dt.strftime(_ANNOTATION_TIME_FMT)}-{stop_dt.strftime(_ANNOTATION_TIME_FMT)}"
        f"-{spec.abs_orbit_number:06d}-{spec.datatake_id:06x}-{spec.image_number}"
    )
    product_name = (
        f"S1{spec.platform}_{spec.mode}_SLC__1SD{spec.pol[1].upper()}"
        f"_{start_dt.strftime('%Y%m%dT%H%M%S')}_{stop_dt.strftime('%Y%m%dT%H%M%S')}"
        f"_{spec.abs_orbit_number:06d}_{spec.datatake_id:06X}_5132.SAFE"
    )

    fields: dict[str, object] = {
        "start_time": start_time,
        "stop_time": stop_time,
        "sensing_times": sensing_times,
        "azimuth_times": azimuth_times,
        "annotation_stem": annotation_stem,
    }

    root = base_dir / product_name
    (root / "annotation" / "calibration").mkdir(parents=True)
    (root / "measurement").mkdir()
    (root / "preview").mkdir()

    (root / "manifest.safe").write_text(_manifest_xml(spec, fields))
    (root / "annotation" / f"{annotation_stem}.xml").write_text(_annotation_xml(spec, fields))
    (root / "annotation" / "calibration" / f"calibration-{annotation_stem}.xml").write_text(
        _calibration_xml(spec, fields),
    )
    (root / "annotation" / "calibration" / f"noise-{annotation_stem}.xml").write_text(
        _noise_xml(spec, fields),
    )
    if spec.with_rfi:
        (root / "annotation" / "rfi").mkdir()
        (root / "annotation" / "rfi" / f"rfi-{annotation_stem}.xml").write_text(
            _rfi_xml(spec, fields),
        )
    (root / "preview" / "quick-look.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    # A second sub-swath, so the manifest pruning has something to remove.
    other_swath = "IW2" if spec.sub_swath != "IW2" else "IW3"
    other_stem = annotation_stem.replace(spec.swath, other_swath.lower())
    (root / "annotation" / f"{other_stem}.xml").write_text(
        textwrap.dedent("""\
            <?xml version="1.0" encoding="UTF-8"?>
            <product><adsHeader><swath>other</swath></adsHeader></product>
        """),
    )

    measurement = _write_measurement(root / "measurement" / f"{annotation_stem}.tiff", spec)

    return SyntheticProduct(
        spec=spec,
        root=root,
        name=product_name,
        annotation_name=f"{annotation_stem}.xml",
        sensing_times=sensing_times,
        azimuth_times=azimuth_times,
        measurement=measurement,
    )
