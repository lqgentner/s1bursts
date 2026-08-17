"""Pin the Python port to the ESA reference extractor it was derived from.

``s1bursts.burst_extractor`` is a port of CDSE's ``sentinel1_burst_extractor.sh``,
which is the program CDSE itself runs to produce the burst products it serves.
Parity with it is the port's contract, so these tests read the vendored script
and check the port against what it actually does:

- the vendored copy is the version the port claims to track, and has not been
  swapped for a newer upstream revision;
- every constant the two share (sub-swath timings, orbital period, beam
  parameters) agrees;
- every XML edit the script performs has a counterpart in the port, and the
  port makes no edit the script does not.

They run offline and never execute the script.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re

from lxml import etree
import pytest

from s1bursts.burst_extractor import (
    _ORBIT_PERIOD,
    _SUBSWATH_TIMING,
    REFERENCE_SCRIPT_VERSION,
    _build_footprint,
    _edit_annotation,
    _edit_calibration,
    _edit_manifest,
    _edit_noise,
    _edit_rfi,
)
from tests.fixtures.safe_product import ProductSpec, build_product

SCRIPT_PATH = Path(__file__).parent / "fixtures" / "sentinel1_burst_extractor.sh"

# Pinned in tests/fixtures/REFERENCE.md. An upstream change must be reviewed and
# ported deliberately, not absorbed silently.
SCRIPT_SHA256 = "c17c8d1b6191bef2d7e49d85e5941407e588dbc83a2ac8de18ae3c366ef16e4e"


@pytest.fixture(scope="module")
def script() -> str:
    return SCRIPT_PATH.read_text()


class TestVendoredScript:
    def test_digest_is_pinned(self) -> None:
        digest = hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest()

        assert digest == SCRIPT_SHA256, (
            "The vendored reference extractor changed. Review the upstream diff, port any "
            "behaviour change, then update SCRIPT_SHA256 and tests/fixtures/REFERENCE.md."
        )

    def test_port_tracks_the_vendored_version(self, script: str) -> None:
        declared = re.search(r'^version="([^"]+)"', script, re.MULTILINE)

        assert declared is not None
        assert declared.group(1) == REFERENCE_SCRIPT_VERSION


class TestSharedConstants:
    def test_subswath_half_durations_match(self, script: str) -> None:
        """Each `T0_delta` case in the script must equal the port's table entry."""
        cases = dict(
            re.findall(
                r"^\t(iw[123]|ew[12345])\)\n\t\t\S*T0_delta=\"0\$\(echo 'scale=8;"
                r"([0-9.]+)/2\.0'\|bc\)\"",
                script,
                re.MULTILINE,
            ),
        )

        assert set(cases) == set(_SUBSWATH_TIMING)
        for swath, numerator in cases.items():
            assert _SUBSWATH_TIMING[swath].t0_delta == pytest.approx(float(numerator) / 2.0)

    @pytest.mark.parametrize(
        ("family", "tbeam", "t_pre"),
        [("iw", 2.758273, 2.299849), ("ew", 3.038376, 2.299970)],
    )
    def test_beam_parameters_match(
        self,
        script: str,
        family: str,
        tbeam: float,
        t_pre: float,
    ) -> None:
        assert f"Tbeam={tbeam}" in script
        assert f"T_pre={t_pre}" in script
        for swath, timing in _SUBSWATH_TIMING.items():
            if swath.startswith(family):
                assert timing.tbeam == pytest.approx(tbeam)
                assert timing.t_pre == pytest.approx(t_pre)

    def test_orbital_period_matches(self, script: str) -> None:
        assert 'Od=$(echo "scale=8;12*24*3600/175" | bc)' in script
        assert float(_ORBIT_PERIOD) == pytest.approx(12 * 24 * 3600 / 175)


def _xmlstarlet_edits(script: str) -> set[tuple[str, str]]:
    """Return every ``(operation, xpath)`` the reference script applies.

    Parses the ``xmlstarlet ed`` invocations, normalising away the shell's
    quoting and the ``xfdu:XFDU/`` document-root prefix so the expressions can
    be compared with the port's own, root-relative XPaths.
    """
    edits: set[tuple[str, str]] = set()
    # Several edits share a line, so this cannot anchor on the line start.
    # `xmlstarlet ed` operands are always paths from the document root, so
    # requiring a `/` filters out same-named flags on other commands
    # (`tr -d '\-\:'`, `date -d {}`) without needing to parse the pipeline.
    for operation, expression in re.findall(r"-([ud]) ('[^']*'|\"[^\"]*\")", script):
        xpath = expression.strip("'\"").removeprefix("xfdu:XFDU/")
        if "/" in xpath:
            edits.add((operation, xpath))
    return edits


# Each reference edit, mapped to the port function that implements it. The
# mapping is the review record: adding an edit upstream fails the coverage test
# below until it is ported and listed here.
PORTED_EDITS: dict[tuple[str, str], str] = {
    (
        "d",
        "product/swathTiming/burstList/burst[byteOffset!=$burst_byteoffset]",
    ): "_edit_annotation",
    ("u", "product/geolocationGrid/geolocationGridPointList/geolocationGridPoint/line"): (
        "_edit_annotation"
    ),
    ("u", "product/swathTiming/burstList/@count"): "_edit_annotation",
    ("u", "product/imageAnnotation/imageInformation/@numberOfLines"): "_edit_annotation",
    ("u", "product/imageAnnotation/imageInformation/numberOfLines"): "_edit_annotation",
    ("u", "product/imageAnnotation/imageInformation/productFirstLineUtcTime"): "_edit_annotation",
    ("u", "product/imageAnnotation/imageInformation/productLastLineUtcTime"): "_edit_annotation",
    ("u", "product/adsHeader/startTime"): "_edit_annotation",
    ("u", "product/adsHeader/stopTime"): "_edit_annotation",
    ("u", "calibration/calibrationVectorList/calibrationVector/line"): "_edit_calibration",
    ("u", "calibration/adsHeader/startTime"): "_edit_calibration",
    ("u", "calibration/adsHeader/stopTime"): "_edit_calibration",
    ("u", "noise/noiseRangeVectorList/noiseRangeVector/line"): "_edit_noise",
    ("u", "noise/adsHeader/startTime"): "_edit_noise",
    ("u", "noise/adsHeader/stopTime"): "_edit_noise",
    ("u", "rfi/adsHeader/startTime"): "_edit_rfi",
    ("u", "rfi/adsHeader/stopTime"): "_edit_rfi",
    (
        "u",
        "metadataSection/metadataObject/metadataWrap/xmlData/safe:acquisitionPeriod"
        "/safe:startTime",
    ): "_edit_manifest",
    (
        "u",
        "metadataSection/metadataObject/metadataWrap/xmlData/safe:acquisitionPeriod/safe:stopTime",
    ): "_edit_manifest",
    ("u", "dataObjectSection/dataObject/byteStream/checksum"): "_edit_manifest",
    ("u", "dataObjectSection/dataObject/byteStream/@size"): "_edit_manifest",
    (
        "u",
        "metadataSection/metadataObject/metadataWrap/xmlData/safe:frameSet/safe:frame"
        "/safe:footPrint/gml:coordinates",
    ): "_edit_manifest",
    ("d", 'dataObjectSection/dataObject[@ID = "quicklook"]'): "_edit_manifest",
    ("d", 'dataObjectSection/dataObject[@ID = "mapoverlay"]'): "_edit_manifest",
    ("d", 'dataObjectSection/dataObject[@ID = "productpreview"]'): "_edit_manifest",
    ("d", 'metadataSection/metadataObject[@classification="SYNTAX"]'): "_edit_manifest",
}

# Edits whose shell form embeds an interpolated variable, so the parsed XPath is
# not a stable literal. Matched by a distinctive fragment instead.
PORTED_INTERPOLATED_EDITS = {
    "s1sarl1:swath[not(text()=": "_edit_manifest",
    "dataObjectSection/dataObject[not(contains(@ID,": "_edit_manifest",
    "metadataSection/metadataObject/dataObjectPointer[not(contains(@dataObjectID,": (
        "_edit_manifest"
    ),
    "informationPackageMap/xfdu:contentUnit/xfdu:contentUnit/dataObjectPointer[not(contains(": (
        "_edit_manifest"
    ),
}


class TestEditCoverage:
    def test_every_reference_edit_is_accounted_for(self, script: str) -> None:
        """No edit in the reference may be missing from the port's mapping."""
        edits = _xmlstarlet_edits(script)
        # Guards against the parser silently matching nothing.
        assert len(edits) >= len(PORTED_EDITS)

        unmapped = []
        for edit in edits:
            if edit in PORTED_EDITS:
                continue
            if any(fragment in edit[1] for fragment in PORTED_INTERPOLATED_EDITS):
                continue
            unmapped.append(edit)

        assert unmapped == [], (
            f"The reference extractor performs edits the port does not implement: {unmapped}"
        )

    def test_mapping_has_no_stale_entries(self, script: str) -> None:
        """Every mapped edit must still exist in the vendored script."""
        found = _xmlstarlet_edits(script)
        stale = sorted(edit for edit in PORTED_EDITS if edit not in found)

        assert stale == []

    def test_mapping_names_real_port_functions(self) -> None:
        implementations = {
            "_edit_annotation": _edit_annotation,
            "_edit_calibration": _edit_calibration,
            "_edit_manifest": _edit_manifest,
            "_edit_noise": _edit_noise,
            "_edit_rfi": _edit_rfi,
        }
        named = set(PORTED_EDITS.values()) | set(PORTED_INTERPOLATED_EDITS.values())

        assert named <= set(implementations)


class TestReferenceNamingRules:
    """The port must reproduce the reference's file-naming arithmetic."""

    def test_annotation_basename_is_68_characters(self, tmp_path: Path) -> None:
        """The script slices `${annotation_xml: -68:15}`, which assumes this length."""
        product = build_product(tmp_path)

        assert len(product.annotation_name) == 68

    def test_new_pattern_prefix_is_the_first_15_characters(self, tmp_path: Path) -> None:
        """`${annotation_xml: -68:15}` is the mission/swath/product/polarization prefix."""
        product = build_product(tmp_path)

        assert product.annotation_name[:15] == "s1a-iw1-slc-vv-"

    def test_old_pattern_is_the_basename_without_extension(self, tmp_path: Path) -> None:
        """`${annotation_xml: -68:64}` drops exactly the `.xml` suffix."""
        product = build_product(tmp_path)

        assert product.annotation_name[:64] == product.annotation_name.removesuffix(".xml")


_REAL_GRID_XML = b"""<product><geolocationGrid><geolocationGridPointList>
  <geolocationGridPoint><line>0</line><pixel>0</pixel>
    <longitude>7.628802608148144e+00</longitude>
    <latitude>4.635520751197574e+01</latitude></geolocationGridPoint>
  <geolocationGridPoint><line>0</line><pixel>100</pixel>
    <longitude>8.828522850894695e+00</longitude>
    <latitude>4.651065837375187e+01</latitude></geolocationGridPoint>
  <geolocationGridPoint><line>10</line><pixel>0</pixel>
    <longitude>7.769876371302304e+00</longitude>
    <latitude>4.585936718871039e+01</latitude></geolocationGridPoint>
  <geolocationGridPoint><line>10</line><pixel>100</pixel>
    <longitude>8.525890501901685e+00</longitude>
    <latitude>4.751771297228279e+01</latitude></geolocationGridPoint>
</geolocationGridPointList></geolocationGrid></product>"""


class TestFootprintFormatting:
    """Footprint coordinates: which of three plausible formats the port must emit.

    The reference script pipes each coordinate through XPath ``number()``, but
    the burst products CDSE actually serves keep the annotation's own text. The
    two disagree, and Python's default float formatting matches neither, so the
    choice is pinned here with the values measured from a real product.
    """

    def test_coordinate_text_is_copied_verbatim(self) -> None:
        """Matches the manifest CDSE ships, and keeps every digit of the source."""
        footprint = _build_footprint(etree.fromstring(_REAL_GRID_XML))

        assert footprint == (
            "7.628802608148144e+00 4.635520751197574e+01,"
            "8.828522850894695e+00 4.651065837375187e+01,"
            "8.525890501901685e+00 4.751771297228279e+01,"
            "7.769876371302304e+00 4.585936718871039e+01,"
            "7.628802608148144e+00 4.635520751197574e+01"
        )

    def test_xpath_number_would_lose_digits(self) -> None:
        """Documents why the script's `number()` is not the format to copy.

        libxml2 renders XPath numbers with a fixed 15 significant digits, which
        silently drops the last digit of a two-degree latitude.
        """
        root = etree.fromstring(_REAL_GRID_XML)
        rendered = root.xpath(
            "string(number(geolocationGrid/geolocationGridPointList"
            "/geolocationGridPoint[1]/latitude))",
        )

        assert rendered == "46.3552075119757"
        assert rendered != repr(float("4.635520751197574e+01"))


def test_product_spec_covers_every_reference_subswath() -> None:
    """The reference accepts eight sub-swaths; the fixture must be able to build them all."""
    for swath in _SUBSWATH_TIMING:
        spec = ProductSpec(sub_swath=swath.upper())

        assert spec.mode == swath[:2].upper()
