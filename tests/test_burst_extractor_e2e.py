"""End-to-end burst extraction against a synthetic SAFE product.

These tests run the full :func:`extract_burst_from_s3` orchestration offline.
The S3 layer is replaced by a local directory — the catalogue lookup and the
``UPath`` factory are the only things stubbed — so annotation parsing, burst
resolution, every XML edit, the windowed raster read, GCP assignment, archiving
and the upload branch all execute for real.

Every extracted archive is put through the independent structural checks in
``tests/safe_conformance.py``, and one extraction is pinned against a committed
golden record.
"""

from __future__ import annotations

from decimal import InvalidOperation
import errno
from itertools import count
import json
import os
from pathlib import Path
import re
import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from lxml import etree
import numpy as np
import pytest
import requests
from upath import UPath

from s1bursts.burst import Burst
import s1bursts.burst_extractor as extractor
from s1bursts.burst_extractor import (
    _glob_annotation_xml,
    _lookup_s3_path,
    _make_s3_upath,
    extract_burst_from_s3,
)
from s1bursts.config import CDSES3Settings
from s1bursts.errors import BurstDownloadError, BurstNotFoundError, ConfigurationError
from s1bursts.filemanager import FileManager
from tests.fixtures.safe_product import ProductSpec, SyntheticProduct, build_product
from tests.safe_conformance import check_burst_safe, digest, read_burst_safe

if TYPE_CHECKING:
    from collections.abc import Callable

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "golden_burst.json"


@pytest.fixture
def extract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., tuple[SyntheticProduct, Path | UPath]]:
    """Return a helper that builds a product and extracts one burst from it."""
    builds = count()

    def _extract(
        spec: ProductSpec | None = None,
        *,
        burst_index: int = 0,
        save_path: Path | UPath | None = None,
        corrupt: Callable[[SyntheticProduct], None] | None = None,
    ) -> tuple[SyntheticProduct, Path | UPath]:
        # Variants of one spec can share a product name, so each build gets its
        # own directory.
        source = tmp_path / f"source-{next(builds)}"
        source.mkdir()
        product = build_product(source, spec)
        if corrupt is not None:
            corrupt(product)

        monkeypatch.setattr(extractor, "_lookup_s3_path", lambda _name: str(product.root))
        monkeypatch.setattr(extractor, "_make_s3_upath", lambda path, _settings: UPath(path))

        destination = save_path or tmp_path / "out" / "burst.zip"
        result = extract_burst_from_s3(
            destination,
            parent_product_name=product.name,
            polarization=product.spec.polarization,
            sub_swath=product.spec.sub_swath,
            burst_id=product.spec.burst_ids[burst_index],
        )
        return product, result

    return _extract


class TestExtraction:
    @pytest.mark.parametrize(
        ("position", "burst_index"),
        [("first", 0), ("middle", 1), ("last", 3)],
    )
    def test_extracted_burst_is_conformant(
        self,
        extract: Callable[..., tuple[SyntheticProduct, Path]],
        position: str,
        burst_index: int,
    ) -> None:
        """First, interior and last bursts must all satisfy the product contract."""
        product, zip_path = extract(burst_index=burst_index)

        violations = check_burst_safe(zip_path, sub_swath=product.spec.sub_swath)

        assert violations == [], f"{position} burst: {violations}"

    @pytest.mark.parametrize("burst_index", [0, 1, 2, 3])
    def test_pixels_come_from_the_right_window(
        self,
        extract: Callable[..., tuple[SyntheticProduct, Path]],
        burst_index: int,
    ) -> None:
        """The measurement must hold exactly the parent's rows for that burst."""
        product, zip_path = extract(burst_index=burst_index)

        burst = read_burst_safe(zip_path)

        assert np.array_equal(burst.measurement, product.burst_window(burst_index + 1))

    def test_burst_windows_are_distinguishable(
        self,
        extract: Callable[..., tuple[SyntheticProduct, Path]],
    ) -> None:
        """Guards the previous test: neighbouring bursts must not be identical."""
        product, _ = extract(burst_index=0)

        assert not np.array_equal(product.burst_window(1), product.burst_window(2))

    def test_names_the_safe_directory_from_burst_metadata(
        self,
        extract: Callable[..., tuple[SyntheticProduct, Path]],
    ) -> None:
        product, zip_path = extract(burst_index=2)
        burst = read_burst_safe(zip_path)

        assert burst.safe_dir == (
            f"S1A_SLC_{product.sensing_times[2][:19].replace('-', '').replace(':', '')}"
            f"_{product.spec.burst_ids[2]:06d}_IW1_VV_{product.spec.datatake_id:06d}.SAFE"
        )

    def test_geolocation_lines_are_shifted_to_the_burst(
        self,
        extract: Callable[..., tuple[SyntheticProduct, Path]],
    ) -> None:
        """An interior burst keeps the whole grid, offset so its own row sits at zero."""
        product, zip_path = extract(burst_index=2)
        burst = read_burst_safe(zip_path)

        lines = sorted({int(point["line"]) for point in burst.geolocation_points()})
        step = product.spec.lines_per_burst

        assert lines == [-2 * step, -step, 0, step, 2 * step]

    def test_preserves_measurement_dtype(
        self,
        extract: Callable[..., tuple[SyntheticProduct, Path]],
    ) -> None:
        """SLC pixels are CInt16 and must not be silently promoted."""
        _, zip_path = extract()

        assert read_burst_safe(zip_path).raster_dtype == "complex_int16"


class TestProductVariants:
    def test_legacy_product_without_burst_ids(
        self,
        extract: Callable[..., tuple[SyntheticProduct, Path]],
    ) -> None:
        """Products predating the `burstId` field resolve via the orbital fallback."""
        spec = ProductSpec(with_burst_id=False)
        product, zip_path = extract(spec, burst_index=2)

        assert check_burst_safe(zip_path, sub_swath="IW1") == []
        assert f"{product.spec.burst_ids[2]:06d}" in read_burst_safe(zip_path).pattern

    def test_legacy_and_modern_products_agree(
        self,
        extract: Callable[..., tuple[SyntheticProduct, Path]],
        tmp_path: Path,
    ) -> None:
        """Both resolution paths must select the same burst."""
        _, modern = extract(ProductSpec(), burst_index=1, save_path=tmp_path / "modern.zip")
        _, legacy = extract(
            ProductSpec(with_burst_id=False),
            burst_index=1,
            save_path=tmp_path / "legacy.zip",
        )

        assert np.array_equal(
            read_burst_safe(modern).measurement,
            read_burst_safe(legacy).measurement,
        )

    @pytest.mark.parametrize(
        ("sub_swath", "polarization"),
        [("IW2", "VH"), ("IW3", "VV"), ("EW1", "HH"), ("EW5", "HV")],
    )
    def test_all_modes_and_polarizations(
        self,
        extract: Callable[..., tuple[SyntheticProduct, Path]],
        sub_swath: str,
        polarization: str,
    ) -> None:
        spec = ProductSpec(sub_swath=sub_swath, polarization=polarization)
        _, zip_path = extract(spec, burst_index=1)

        assert check_burst_safe(zip_path, sub_swath=sub_swath) == []

    def test_rfi_annotation_is_carried_over_and_retimed(
        self,
        extract: Callable[..., tuple[SyntheticProduct, Path]],
    ) -> None:
        _, zip_path = extract(ProductSpec(with_rfi=True), burst_index=1)
        burst = read_burst_safe(zip_path)

        assert burst.rfi is not None
        assert burst.rfi.findtext("adsHeader/startTime") == burst.azimuth_start
        assert burst.rfi.findtext("adsHeader/stopTime") == burst.azimuth_stop
        # RFI line numbers are not shifted; the reference only retimes the header.
        assert burst.rfi.findtext("rfiDetectionFromNoiseReportList/*/line") == str(
            ProductSpec().lines_per_burst,
        )

    def test_product_without_rfi_omits_the_directory(
        self,
        extract: Callable[..., tuple[SyntheticProduct, Path]],
    ) -> None:
        _, zip_path = extract(ProductSpec(with_rfi=False))
        burst = read_burst_safe(zip_path)

        assert burst.rfi is None
        assert not any("/rfi/" in name for name in burst.members)

    def test_rejects_unknown_subswath(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown subswath"):
            extract_burst_from_s3(
                tmp_path / "out.zip",
                parent_product_name="anything",
                polarization="VV",
                sub_swath="IW9",
                burst_id=1,
            )


class TestDestination:
    def test_writes_to_a_remote_destination(
        self,
        extract: Callable[..., tuple[SyntheticProduct, Path]],
    ) -> None:
        """A non-local save path takes the streamed-upload branch."""
        remote = UPath("memory://bursts/out/burst.zip")
        _, result = extract(save_path=remote)

        assert result == remote
        assert remote.exists()
        assert remote.stat().st_size > 0

    def test_creates_missing_parent_directories(
        self,
        extract: Callable[..., tuple[SyntheticProduct, Path]],
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "deeply" / "nested" / "burst.zip"
        _, result = extract(save_path=target)

        assert result.exists()

    def test_leaves_no_temporary_directory_behind(
        self,
        extract: Callable[..., tuple[SyntheticProduct, Path]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("TMPDIR", str(scratch))

        extract()

        assert list(scratch.iterdir()) == []

    def test_a_move_cut_short_leaves_nothing_at_the_final_path(
        self,
        extract: Callable[..., tuple[SyntheticProduct, Path]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A copy across file systems that fails partway must not leave a truncated SLC."""
        # Previously the truncated copy stayed at the final path, and skip-if-exists trusted it.
        fm = FileManager(tmp_path / "data")
        target = fm.slc_dir / "S1_SLC_066A_030346_IW1_VV_20220105.zip"
        staged: list[Path] = []

        def _move_cut_short(src: str | Path, dst: str | Path) -> None:
            staged.append(Path(dst))
            Path(dst).write_bytes(Path(src).read_bytes()[:100])
            # While the partial file exists, it must not pass for an SLC.
            assert fm.find(Burst) == []
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(extractor.shutil, "move", _move_cut_short)
        with pytest.raises(OSError, match="No space left on device"):
            extract(save_path=target)

        assert [path.parent for path in staged] == [fm.slc_dir]
        assert list(fm.slc_dir.iterdir()) == []

    def test_the_slc_is_flushed_to_disk_before_it_is_renamed_into_place(
        self,
        extract: Callable[..., tuple[SyntheticProduct, Path]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target = tmp_path / "out" / "burst.zip"
        synced_before_rename: list[bool] = []
        fsync = os.fsync

        def _fsync(fd: int) -> None:
            synced_before_rename.append(not target.exists())
            fsync(fd)

        monkeypatch.setattr(extractor.os, "fsync", _fsync)
        extract(save_path=target)

        assert synced_before_rename == [True]
        assert target.exists()

    def test_removes_what_a_killed_run_left_staged_for_the_same_slc(
        self,
        extract: Callable[..., tuple[SyntheticProduct, Path]],
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "out"
        out.mkdir()
        stale = out / ".burst.zip.0123456789abcdef.tmp"
        in_progress = out / ".burst.zip.fedcba9876543210.tmp"
        other = out / ".other.zip.0123456789abcdef.tmp"
        old = time.time() - extractor._STALE_STAGED_SLC_AGE_S - 60
        for path in (stale, in_progress, other):
            path.write_bytes(b"cut short")
        for path in (stale, other):
            os.utime(path, (old, old))

        extract(save_path=out / "burst.zip")

        # A fresh file may be another run's download, still being written.
        assert sorted(path.name for path in out.iterdir()) == sorted(
            [in_progress.name, other.name, "burst.zip"]
        )


class TestGoldenRecord:
    """A committed digest of one extraction, so silent output drift fails the suite."""

    def test_matches_the_golden_record(
        self,
        extract: Callable[..., tuple[SyntheticProduct, Path]],
    ) -> None:
        _, zip_path = extract(ProductSpec(with_rfi=True), burst_index=2)

        actual = digest(zip_path)
        expected = json.loads(GOLDEN_PATH.read_text())

        assert actual == expected, (
            "Extraction output changed. If the change is intended and matches the reference "
            f"extractor, regenerate {GOLDEN_PATH.name} with tests/fixtures/regenerate_golden.py."
        )


class TestS3Lookup:
    def test_returns_the_catalogued_s3_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        class _Response:
            def raise_for_status(self) -> None: ...

            def json(self) -> dict[str, object]:
                return {"value": [{"S3Path": "/eodata/Sentinel-1/SAR/SLC/2022/08/25/x.SAFE"}]}

        def _get(url: str, params: dict[str, str], timeout: int) -> _Response:
            captured.update(url=url, params=params, timeout=timeout)
            return _Response()

        monkeypatch.setattr(requests, "get", _get)

        assert _lookup_s3_path("S1A_IW_SLC.SAFE") == "/eodata/Sentinel-1/SAR/SLC/2022/08/25/x.SAFE"
        assert captured["params"] == {"$filter": "Name eq 'S1A_IW_SLC.SAFE'"}

    def test_raises_when_the_product_is_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Response:
            def raise_for_status(self) -> None: ...

            def json(self) -> dict[str, object]:
                return {"value": []}

        monkeypatch.setattr(requests, "get", lambda *_args, **_kwargs: _Response())

        with pytest.raises(BurstNotFoundError, match="not found in CDSE catalog"):
            _lookup_s3_path("S1A_MISSING.SAFE")

    @pytest.mark.parametrize(
        ("body", "match"),
        [
            ([], "Unexpected CDSE catalogue response"),
            ({"error": "busy"}, "Unexpected CDSE catalogue response"),
            ({"value": [{"Name": "x.SAFE"}]}, "has no S3Path"),
            ({"value": ["x.SAFE"]}, "has no S3Path"),
        ],
    )
    def test_a_malformed_response_fails_the_burst(
        self,
        monkeypatch: pytest.MonkeyPatch,
        body: object,
        match: str,
    ) -> None:
        # Previously an AttributeError or KeyError, which stopped the whole download.
        class _Response:
            def raise_for_status(self) -> None: ...

            def json(self) -> object:
                return body

        monkeypatch.setattr(requests, "get", lambda *_args, **_kwargs: _Response())

        with pytest.raises(BurstDownloadError, match=match) as info:
            _lookup_s3_path("S1A_IW_SLC.SAFE")
        assert not isinstance(info.value, BurstNotFoundError)


class TestS3Authentication:
    def test_prefers_a_named_profile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def _make(path: str, **kwargs: object) -> UPath:
            captured.update(path=path, **kwargs)
            return UPath("memory://root")

        monkeypatch.setattr(extractor, "make_s3_upath", _make)
        settings = CDSES3Settings(
            profile="cdse",
            access_key_id="key",
            secret_access_key="secret",
        )

        _make_s3_upath("/eodata/x.SAFE", settings)

        assert captured["path"] == "s3://eodata/x.SAFE"
        assert captured["profile"] == "cdse"
        assert "key" not in captured

    def test_falls_back_to_an_explicit_key_pair(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def _make(path: str, **kwargs: object) -> UPath:
            captured.update(path=path, **kwargs)
            return UPath("memory://root")

        monkeypatch.setattr(extractor, "make_s3_upath", _make)
        settings = CDSES3Settings(access_key_id="key", secret_access_key="secret")

        _make_s3_upath("/eodata/x.SAFE", settings)

        assert captured["key"] == "key"
        assert captured["secret"] == "secret"
        assert "profile" not in captured

    def test_reports_missing_credentials(self) -> None:
        with pytest.raises(ConfigurationError, match="CDSE S3 credentials not configured"):
            _make_s3_upath("/eodata/x.SAFE", CDSES3Settings())

    def test_incomplete_key_pair_is_not_accepted(self) -> None:
        settings = CDSES3Settings(access_key_id="key")

        with pytest.raises(ConfigurationError, match="CDSE S3 credentials not configured"):
            _make_s3_upath("/eodata/x.SAFE", settings)


class TestAnnotationLookup:
    def test_finds_the_single_matching_annotation(self, tmp_path: Path) -> None:
        product = build_product(tmp_path)

        found = _glob_annotation_xml(UPath(product.root), "iw1", "vv")

        assert found.name == product.annotation_name

    def test_raises_when_no_annotation_matches(self, tmp_path: Path) -> None:
        product = build_product(tmp_path)

        with pytest.raises(BurstNotFoundError, match="Expected 1 annotation XML"):
            _glob_annotation_xml(UPath(product.root), "iw3", "hh")

    def test_raises_when_several_annotations_match(self, tmp_path: Path) -> None:
        product = build_product(tmp_path)
        duplicate = product.root / "annotation" / "s1a-iw1-slc-vv-duplicate-001.xml"
        duplicate.write_text("<product/>")

        with pytest.raises(BurstNotFoundError, match="found 2"):
            _glob_annotation_xml(UPath(product.root), "iw1", "vv")


def _edit(path: Path, old: str, new: str) -> None:
    """Replace the first match of the regular expression ``old`` in a file."""
    text = path.read_text()
    edited = re.sub(old, new, text, count=1, flags=re.DOTALL)
    assert edited != text, f"{old!r} not found in {path.name}"
    path.write_text(edited)


def _annotation(product: SyntheticProduct) -> Path:
    return product.root / "annotation" / product.annotation_name


def _calibration(product: SyntheticProduct) -> Path:
    return product.root / "annotation" / "calibration" / f"calibration-{product.annotation_name}"


class TestMalformedProduct:
    """A malformed SAFE file fails its burst with a BurstDownloadError, not the batch."""

    @pytest.mark.parametrize(
        ("corrupt", "cause", "what"),
        [
            pytest.param(
                lambda product: _edit(_annotation(product), r"</product>\s*$", ""),
                etree.XMLSyntaxError,
                "annotation or manifest",
                id="truncated-xml",
            ),
            pytest.param(
                lambda product: _edit(
                    _annotation(product),
                    r"<linesPerBurst>\d+</linesPerBurst>",
                    "<linesPerBurst>twelve</linesPerBurst>",
                ),
                ValueError,
                "annotation or manifest",
                id="non-numeric",
            ),
            pytest.param(
                lambda product: _edit(
                    _annotation(product), r"<linesPerBurst>\d+</linesPerBurst>", ""
                ),
                TypeError,
                "annotation or manifest",
                id="missing-element",
            ),
            pytest.param(
                lambda product: _edit(
                    product.root / "manifest.safe", r"<safe:number>\w+</safe:number>", ""
                ),
                IndexError,
                "annotation or manifest",
                id="missing-xpath-match",
            ),
            pytest.param(
                lambda product: _edit(_calibration(product), r"<startTime>[^<]*</startTime>", ""),
                AttributeError,
                "calibration annotation",
                id="missing-edited-element",
            ),
        ],
    )
    def test_a_malformed_file_is_a_download_error(
        self,
        extract: Callable[..., tuple[SyntheticProduct, Path]],
        corrupt: Callable[[SyntheticProduct], None],
        cause: type[Exception],
        what: str,
    ) -> None:
        """Previously raised the parse error itself, which aborted the whole download."""
        with pytest.raises(BurstDownloadError, match=f"^Malformed {what} in 'S1") as excinfo:
            extract(corrupt=corrupt)

        assert isinstance(excinfo.value.__cause__, cause)

    def test_an_invalid_decimal_is_a_download_error(
        self,
        extract: Callable[..., tuple[SyntheticProduct, Path]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The orbital fallback computes in Decimal, whose invalid operations raise."""
        monkeypatch.setattr(extractor, "_resolve_burst", MagicMock(side_effect=InvalidOperation))

        with pytest.raises(BurstDownloadError, match="InvalidOperation") as excinfo:
            extract()

        assert isinstance(excinfo.value.__cause__, InvalidOperation)

    def test_a_missing_burst_is_not_a_parse_error(
        self,
        extract: Callable[..., tuple[SyntheticProduct, Path]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        missing = BurstNotFoundError("not in this product")
        monkeypatch.setattr(extractor, "_resolve_burst", MagicMock(side_effect=missing))

        with pytest.raises(BurstNotFoundError) as excinfo:
            extract()

        assert excinfo.value is missing
