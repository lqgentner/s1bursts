"""Offline unit tests for s1bursts.burst_extractor helpers.

All tests run without network or S3 access. All fixture data is constructed
inline from known values derived from a reference IW1/VV burst product.
"""

from __future__ import annotations

from textwrap import dedent

from lxml import etree
import pytest

from s1bursts.burst_extractor import (
    _build_footprint,
    _BurstMeta,
    _edit_annotation,
    _edit_calibration,
    _edit_noise,
    _edit_rfi,
    _orbital_burst_number,
    _resolve_burst,
)
from s1bursts.errors import BurstNotFoundError

# ── Known values for the IW1/VV burst (derived from a reference product) ──

_PROD_ANN_NAME = "s1a-iw1-slc-vv-20220825t171550-20220825t171600-044762-055977-001.xml"
_AZIMUTH_START = "2022-08-25T17:15:49.261287"
_AZIMUTH_END = "2022-08-25T17:15:52.344621"
_SENSING_TIME = "2022-08-25T17:15:50.399045"
_BYTE_OFFSET = "406296999"
_BURST_ID_REL = 30346
_LINES_PER_BURST = 1501
_SAMPLES_PER_BURST = 22551
_STARTING_LINE = 4503  # burst_number=4, so 3*1501
_DATATAKE_ID = "349850"
_PLATFORM_NUM = "A"
_AZI_INTERVAL = "2.055556299999998e-03"

# Sensing times for 4 consecutive IW1 bursts (tbeam = 2.758273 s)
_SENSING_TIMES = [
    "2022-08-25T17:15:42.124226",
    "2022-08-25T17:15:44.882499",
    "2022-08-25T17:15:47.640772",
    "2022-08-25T17:15:50.399045",  # our target, position 4, burstId=30346
]
# Corresponding relative burst IDs from the orbital formula
_BURST_IDS = [30343, 30344, 30345, 30346]

# Ascending node from the reference manifest
_ASCENDING_NODE_TIME = "2022-08-25T17:03:12.299795"
_ORBIT_NUMBER = 15


# ── Helper: build synthetic annotation XML ───────────────────────────────────


def _make_annotation_xml(*, with_burst_id: bool = True) -> str:
    """Minimal 4-burst annotation XML for testing."""
    bursts_xml = ""
    byte_offsets = ["100000000", "200000000", "300000000", _BYTE_OFFSET]
    for i, (st, bo) in enumerate(zip(_SENSING_TIMES, byte_offsets, strict=True)):
        bid_el = f"<burstId>{_BURST_IDS[i]}</burstId>" if with_burst_id else ""
        bursts_xml += f"""
      <burst>
        {bid_el}
        <byteOffset>{bo}</byteOffset>
        <azimuthTime>{_AZIMUTH_START}</azimuthTime>
        <sensingTime>{st}</sensingTime>
      </burst>"""

    return dedent(f"""
        <?xml version="1.0" encoding="UTF-8"?>
        <product>
          <adsHeader>
            <missionDataTakeId>{_DATATAKE_ID}</missionDataTakeId>
            <startTime>2022-08-25T17:15:40.000000</startTime>
            <stopTime>2022-08-25T17:15:55.000000</stopTime>
          </adsHeader>
          <imageAnnotation>
            <imageInformation>
              <numberOfLines>6004</numberOfLines>
              <productFirstLineUtcTime>2022-08-25T17:15:40.000000</productFirstLineUtcTime>
              <productLastLineUtcTime>2022-08-25T17:15:55.000000</productLastLineUtcTime>
              <azimuthTimeInterval>{_AZI_INTERVAL}</azimuthTimeInterval>
            </imageInformation>
          </imageAnnotation>
          <swathTiming>
            <linesPerBurst>{_LINES_PER_BURST}</linesPerBurst>
            <samplesPerBurst>{_SAMPLES_PER_BURST}</samplesPerBurst>
            <burstList count="4">{bursts_xml}
            </burstList>
          </swathTiming>
          <geolocationGrid>
            <geolocationGridPointList count="4">
              <geolocationGridPoint>
                <line>0</line><pixel>0</pixel>
                <longitude>7.6</longitude><latitude>46.3</latitude><height>0.0</height>
              </geolocationGridPoint>
              <geolocationGridPoint>
                <line>0</line><pixel>22550</pixel>
                <longitude>8.8</longitude><latitude>46.5</latitude><height>0.0</height>
              </geolocationGridPoint>
              <geolocationGridPoint>
                <line>1501</line><pixel>0</pixel>
                <longitude>7.7</longitude><latitude>47.1</latitude><height>0.0</height>
              </geolocationGridPoint>
              <geolocationGridPoint>
                <line>1501</line><pixel>22550</pixel>
                <longitude>8.9</longitude><latitude>47.3</latitude><height>0.0</height>
              </geolocationGridPoint>
            </geolocationGridPointList>
          </geolocationGrid>
        </product>
    """).strip()


def _make_manifest_xml() -> str:
    """Minimal manifest XML with platform and orbit reference (no namespace prefixes on root)."""
    return dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <XFDU
          xmlns:safe="http://www.esa.int/safe/sentinel-1.0"
          xmlns:s1="http://www.esa.int/safe/sentinel-1.0/sentinel-1">
          <metadataSection>
            <metadataObject>
              <metadataWrap>
                <xmlData>
                  <safe:platform>
                    <safe:number>{_PLATFORM_NUM}</safe:number>
                  </safe:platform>
                </xmlData>
              </metadataWrap>
            </metadataObject>
            <metadataObject>
              <metadataWrap>
                <xmlData>
                  <safe:orbitReference>
                    <safe:relativeOrbitNumber
                      type="start">{_ORBIT_NUMBER}</safe:relativeOrbitNumber>
                    <safe:extension>
                      <s1:orbitProperties>
                        <s1:ascendingNodeTime>{_ASCENDING_NODE_TIME}</s1:ascendingNodeTime>
                      </s1:orbitProperties>
                    </safe:extension>
                  </safe:orbitReference>
                </xmlData>
              </metadataWrap>
            </metadataObject>
          </metadataSection>
        </XFDU>
    """)


# ── Tests: _build_footprint ──────────────────────────────────────────────────


def test_build_footprint_structure() -> None:
    xml = dedent("""\
        <product>
          <geolocationGrid>
            <geolocationGridPointList>
              <geolocationGridPoint>
                <line>0</line><pixel>0</pixel>
                <longitude>10.0</longitude><latitude>50.0</latitude><height>0.0</height>
              </geolocationGridPoint>
              <geolocationGridPoint>
                <line>0</line><pixel>100</pixel>
                <longitude>11.0</longitude><latitude>50.1</latitude><height>0.0</height>
              </geolocationGridPoint>
              <geolocationGridPoint>
                <line>100</line><pixel>0</pixel>
                <longitude>10.1</longitude><latitude>51.0</latitude><height>0.0</height>
              </geolocationGridPoint>
              <geolocationGridPoint>
                <line>100</line><pixel>100</pixel>
                <longitude>11.1</longitude><latitude>51.1</latitude><height>0.0</height>
              </geolocationGridPoint>
            </geolocationGridPointList>
          </geolocationGrid>
        </product>
    """)
    root = etree.fromstring(xml.encode())
    fp = _build_footprint(root)

    # top row sorted by pixel: (10.0, 50.0) → (11.0, 50.1)
    # nontop (all non-zero-line points): first=(10.1,51.0), last=(11.1,51.1)
    # Coordinate text is copied verbatim (see tests/test_esa_reference_parity.py).
    assert fp == "10.0 50.0,11.0 50.1,11.1 51.1,10.1 51.0,10.0 50.0"


# ── Tests: _resolve_burst ────────────────────────────────────────────────────


def test_resolve_burst_via_burst_id() -> None:
    ann_root = etree.fromstring(_make_annotation_xml(with_burst_id=True).encode())
    mfst_root = etree.fromstring(_make_manifest_xml().encode())

    meta = _resolve_burst(
        ann_root,
        mfst_root,
        relative_burst_id=_BURST_ID_REL,
        sub_swath="iw1",
        annotation_filename=_PROD_ANN_NAME,
    )

    assert meta.burst_number == 4
    assert meta.starting_line == _STARTING_LINE
    assert meta.byte_offset == _BYTE_OFFSET
    assert meta.number_of_lines == _LINES_PER_BURST
    assert meta.number_of_samples == _SAMPLES_PER_BURST
    assert meta.azimuth_start == _AZIMUTH_START
    assert meta.azimuth_end == _AZIMUTH_END
    assert meta.sensing_time_compact == "20220825T171550"
    assert meta.datatake_id == _DATATAKE_ID
    assert meta.platform_number == _PLATFORM_NUM
    assert meta.new_pattern == "s1a-iw1-slc-vv-030346-20220825T171550-349850"
    assert meta.new_pattern_short == "s1aiw1slcvv03034620220825T171550349850"


def test_resolve_burst_raises_on_missing_burst() -> None:
    ann_root = etree.fromstring(_make_annotation_xml(with_burst_id=True).encode())
    mfst_root = etree.fromstring(_make_manifest_xml().encode())

    with pytest.raises(BurstNotFoundError, match="not found"):
        _resolve_burst(
            ann_root,
            mfst_root,
            relative_burst_id=99999,
            sub_swath="iw1",
            annotation_filename=_PROD_ANN_NAME,
        )


# ── Tests: _orbital_burst_number ─────────────────────────────────────────────


def test_orbital_burst_number_matches_burst_id() -> None:
    """Orbital fallback returns the same burst position as the burstId lookup."""
    ann_root = etree.fromstring(_make_annotation_xml(with_burst_id=False).encode())
    mfst_root = etree.fromstring(_make_manifest_xml().encode())

    orbit_pos = _orbital_burst_number(
        ann_root,
        mfst_root,
        relative_burst_id=_BURST_ID_REL,
        sub_swath="iw1",
    )
    assert orbit_pos == 4


def test_orbital_burst_number_each_burst() -> None:
    """Every burst in the list resolves to the expected relative ID via orbital math."""
    ann_root = etree.fromstring(_make_annotation_xml(with_burst_id=False).encode())
    mfst_root = etree.fromstring(_make_manifest_xml().encode())

    for expected_pos, expected_id in enumerate(_BURST_IDS, 1):
        pos = _orbital_burst_number(
            ann_root,
            mfst_root,
            relative_burst_id=expected_id,
            sub_swath="iw1",
        )
        assert pos == expected_pos


def test_orbital_burst_number_raises_on_unknown_id() -> None:
    ann_root = etree.fromstring(_make_annotation_xml(with_burst_id=False).encode())
    mfst_root = etree.fromstring(_make_manifest_xml().encode())

    with pytest.raises(BurstNotFoundError, match="not found"):
        _orbital_burst_number(ann_root, mfst_root, relative_burst_id=99999, sub_swath="iw1")


# ── Tests: _edit_annotation ──────────────────────────────────────────────────


def _make_meta() -> _BurstMeta:
    sensing_compact = _SENSING_TIME.split(".", maxsplit=1)[0].replace("-", "").replace(":", "")
    prefix = _PROD_ANN_NAME[:15]
    new_pattern = f"{prefix}{_BURST_ID_REL:06d}-{sensing_compact}-{_DATATAKE_ID}"
    return _BurstMeta(
        burst_number=4,
        byte_offset=_BYTE_OFFSET,
        starting_line=_STARTING_LINE,
        number_of_lines=_LINES_PER_BURST,
        number_of_samples=_SAMPLES_PER_BURST,
        azimuth_start=_AZIMUTH_START,
        azimuth_end=_AZIMUTH_END,
        sensing_time_compact=sensing_compact,
        datatake_id=_DATATAKE_ID,
        platform_number=_PLATFORM_NUM,
        new_pattern=new_pattern,
        new_pattern_short=new_pattern.replace("-", ""),
    )


def test_edit_annotation_keeps_only_target_burst() -> None:
    ann_root = etree.fromstring(_make_annotation_xml().encode())
    meta = _make_meta()
    out = _edit_annotation(ann_root, meta)

    result = etree.fromstring(out)
    bursts = result.findall("swathTiming/burstList/burst")
    assert len(bursts) == 1
    assert bursts[0].findtext("byteOffset") == _BYTE_OFFSET
    assert result.find("swathTiming/burstList").get("count") == "1"


def test_edit_annotation_shifts_geolocation_lines() -> None:
    ann_root = etree.fromstring(_make_annotation_xml().encode())
    meta = _make_meta()
    out = _edit_annotation(ann_root, meta)

    result = etree.fromstring(out)
    lines = [
        int(p.findtext("line"))
        for p in result.findall("geolocationGrid/geolocationGridPointList/geolocationGridPoint")
    ]
    # Original lines were 0 and 1501; after shifting by 4503 → -4503 and -3002
    assert set(lines) == {0 - _STARTING_LINE, 1501 - _STARTING_LINE}


def test_edit_annotation_updates_header_times() -> None:
    ann_root = etree.fromstring(_make_annotation_xml().encode())
    meta = _make_meta()
    out = _edit_annotation(ann_root, meta)

    result = etree.fromstring(out)
    img_info = result.find("imageAnnotation/imageInformation")
    assert result.findtext("adsHeader/startTime") == _AZIMUTH_START
    assert result.findtext("adsHeader/stopTime") == _AZIMUTH_END
    assert img_info.findtext("productFirstLineUtcTime") == _AZIMUTH_START
    assert img_info.findtext("productLastLineUtcTime") == _AZIMUTH_END
    assert img_info.findtext("numberOfLines") == str(_LINES_PER_BURST)


# ── Tests: _edit_calibration ─────────────────────────────────────────────────

_CAL_XML = dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <calibration>
      <adsHeader>
        <startTime>2022-08-25T17:15:40.000000</startTime>
        <stopTime>2022-08-25T17:15:55.000000</stopTime>
      </adsHeader>
      <calibrationVectorList>
        <calibrationVector><line>5000</line></calibrationVector>
        <calibrationVector><line>6004</line></calibrationVector>
      </calibrationVectorList>
    </calibration>
""")


def test_edit_calibration_shifts_lines() -> None:
    meta = _make_meta()
    out = _edit_calibration(_CAL_XML, meta)
    result = etree.fromstring(out)

    lines = [int(el.text) for el in result.findall("calibrationVectorList/calibrationVector/line")]
    assert lines == [5000 - _STARTING_LINE, 6004 - _STARTING_LINE]


def test_edit_calibration_updates_header_times() -> None:
    meta = _make_meta()
    out = _edit_calibration(_CAL_XML, meta)
    result = etree.fromstring(out)

    assert result.findtext("adsHeader/startTime") == _AZIMUTH_START
    assert result.findtext("adsHeader/stopTime") == _AZIMUTH_END


# ── Tests: _edit_noise ───────────────────────────────────────────────────────

_NOISE_XML = dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <noise>
      <adsHeader>
        <startTime>2022-08-25T17:15:40.000000</startTime>
        <stopTime>2022-08-25T17:15:55.000000</stopTime>
      </adsHeader>
      <noiseRangeVectorList>
        <noiseRangeVector><line>4503</line></noiseRangeVector>
        <noiseRangeVector><line>6004</line></noiseRangeVector>
      </noiseRangeVectorList>
    </noise>
""")


def test_edit_noise_shifts_lines() -> None:
    meta = _make_meta()
    out = _edit_noise(_NOISE_XML, meta)
    result = etree.fromstring(out)

    lines = [int(el.text) for el in result.findall("noiseRangeVectorList/noiseRangeVector/line")]
    assert lines == [4503 - _STARTING_LINE, 6004 - _STARTING_LINE]


def test_edit_noise_updates_header_times() -> None:
    meta = _make_meta()
    out = _edit_noise(_NOISE_XML, meta)
    result = etree.fromstring(out)

    assert result.findtext("adsHeader/startTime") == _AZIMUTH_START
    assert result.findtext("adsHeader/stopTime") == _AZIMUTH_END


# ── Tests: _edit_rfi ─────────────────────────────────────────────────────────

_RFI_XML = dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <rfi>
      <adsHeader>
        <startTime>2022-08-25T17:15:40.000000</startTime>
        <stopTime>2022-08-25T17:15:55.000000</stopTime>
      </adsHeader>
      <rfiDetectionReport>
        <line>5000</line>
      </rfiDetectionReport>
    </rfi>
""")


def test_edit_rfi_updates_times_only() -> None:
    meta = _make_meta()
    out = _edit_rfi(_RFI_XML, meta)
    result = etree.fromstring(out)

    assert result.findtext("adsHeader/startTime") == _AZIMUTH_START
    assert result.findtext("adsHeader/stopTime") == _AZIMUTH_END
    # Lines in RFI should NOT be shifted (rfi only updates header times)
    assert result.findtext("rfiDetectionReport/line") == "5000"
