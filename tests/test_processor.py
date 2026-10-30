"""Tests for s1bursts.processor batch processing."""

from __future__ import annotations

from pathlib import Path
import threading
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

from pyproj.exceptions import CRSError
import pytest

from s1bursts.errors import ConfigurationError, SnapProcessingError
from s1bursts.processor import BurstProcessor
from s1bursts.snap_client import SNAPConfig
from s1bursts.snap_graph import CoherenceConfig, GTCConfig, RTCConfig
from s1bursts.summary import ErrorResult, OkResult, OperationSummary

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SNAPConfig:
    """SNAPConfig with a stub gpt executable and isolated env."""
    for var in ("SNAP_BIN", "SNAP_MAX_WORKERS", "SNAP_THREADS_PER_JOB"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    gpt = tmp_path / "gpt"
    gpt.touch()
    gpt.chmod(0o755)
    return SNAPConfig(bin=tmp_path)


@pytest.fixture
def processor(config: SNAPConfig) -> Iterator[BurstProcessor]:
    with patch("s1bursts.processor.SNAPClient") as mock_client_cls:
        proc = BurstProcessor(MagicMock(), config, show_progress=False)
        proc.snap_client = mock_client_cls.return_value
        yield proc


class TestSNAPConfig:
    def test_missing_gpt_path_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SNAP_BIN", raising=False)
        monkeypatch.chdir(tmp_path)

        with (
            patch("s1bursts.snap_client.shutil.which", return_value=None),
            pytest.raises(Exception, match="Could not resolve the SNAP `gpt` executable"),
        ):
            SNAPConfig()

    def test_bin_without_gpt_raises(self, tmp_path: Path) -> None:
        with pytest.raises(Exception, match="No `gpt` executable found"):
            SNAPConfig(bin=tmp_path / "missing-bin")

    def test_non_executable_gpt_path_raises(self, tmp_path: Path) -> None:
        gpt = tmp_path / "gpt"
        gpt.touch()
        gpt.chmod(0o644)

        with pytest.raises(Exception, match="not executable"):
            SNAPConfig(bin=tmp_path)

    def test_resource_resolution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SNAP_MAX_WORKERS", raising=False)
        monkeypatch.chdir(tmp_path)
        gpt = tmp_path / "gpt"
        gpt.touch()
        gpt.chmod(0o755)

        config = SNAPConfig(
            bin=tmp_path,
            max_workers=2,
            max_ram_gb=16,
        )

        assert config.resolved_max_workers == 2
        assert config.resolved_xmx_mb == (16 * 1024 - 3000) // 2
        # Tile cache is a ratio of the per-job heap, never larger than -Xmx
        assert config.resolved_cache_mb == int(config.resolved_xmx_mb * 0.70)
        assert config.resolved_cache_mb < config.resolved_xmx_mb

    def test_ram_budget_too_small_is_reported_not_silently_exceeded(
        self,
        tmp_path: Path,
    ) -> None:
        """A 1 GB budget cannot host SNAP; raising beats handing GPT an oversized heap."""
        gpt = tmp_path / "gpt"
        gpt.touch()
        gpt.chmod(0o755)

        with pytest.raises(ConfigurationError, match="Raise `max_ram_gb`"):
            SNAPConfig(bin=tmp_path, max_ram_gb=1)

    def test_profile_defaults_merged(self, config: SNAPConfig) -> None:
        assert config.rasterio_profile is not None
        assert config.rasterio_profile["dtype"] == "float32"


class TestProcessCollection:
    def test_sequential_collects_results(self, processor: BurstProcessor) -> None:
        items = ["a", "b", "c"]
        process_fn = MagicMock(side_effect=lambda i: OkResult(source=i, result=i.upper()))

        summary = processor._process_collection(items, process_fn, "Test")

        assert isinstance(summary, OperationSummary)
        assert summary.n_total == 3
        assert summary.n_processed == 3
        assert sorted(summary.products) == ["A", "B", "C"]

    def test_sequential_continues_on_error(self, processor: BurstProcessor) -> None:
        error = SnapProcessingError("boom")
        results = [
            OkResult(source="a", result="A"),
            ErrorResult(source="b", error=error),
            OkResult(source="c", result="C"),
        ]
        process_fn = MagicMock(side_effect=results)

        summary = processor._process_collection(["a", "b", "c"], process_fn, "Test")

        assert summary.n_processed == 2
        assert summary.n_failed == 1
        assert summary.failed_items[0].error is error

    def test_sequential_raises_when_continue_on_error_false(
        self,
        processor: BurstProcessor,
    ) -> None:
        processor.continue_on_error = False
        process_fn = MagicMock(
            return_value=ErrorResult(source="a", error=SnapProcessingError("boom")),
        )

        with pytest.raises(SnapProcessingError, match="boom"):
            processor._process_collection(["a"], process_fn, "Test")

    def test_parallel_collects_results(self, processor: BurstProcessor) -> None:
        processor.config.resolved_max_workers = 3
        process_fn = MagicMock(side_effect=lambda i: OkResult(source=i, result=i.upper()))

        summary = processor._process_collection(["a", "b", "c"], process_fn, "Test")

        assert summary.n_processed == 3
        assert sorted(summary.products) == ["A", "B", "C"]

    def test_parallel_results_keep_the_submission_order(self, processor: BurstProcessor) -> None:
        """Results must be listed in input order, not the order the threads finish.

        Previously they were listed as they completed, which composites then
        inherited as their stitch order.
        """
        processor.config.resolved_max_workers = 3
        items = ["a", "b", "c"]
        done = {item: threading.Event() for item in items}

        def process_fn(item: str) -> OkResult[str]:
            # Each item waits for the next one, so they finish in reverse order.
            following = items.index(item) + 1
            if following < len(items):
                assert done[items[following]].wait(timeout=5)
            done[item].set()
            return OkResult(source=item, result=item.upper())

        summary = processor._process_collection(items, process_fn, "Test")

        assert summary.products == ["A", "B", "C"]

    def test_parallel_raises_when_continue_on_error_false(
        self,
        processor: BurstProcessor,
    ) -> None:
        processor.config.resolved_max_workers = 2
        processor.continue_on_error = False
        process_fn = MagicMock(
            return_value=ErrorResult(source="a", error=SnapProcessingError("boom")),
        )

        with pytest.raises(SnapProcessingError, match="boom"):
            processor._process_collection(["a", "b"], process_fn, "Test")


class TestDispatch:
    """The create_* methods must call the matching SNAPClient method per item."""

    def test_create_backscatter(self, processor: BurstProcessor) -> None:
        burst = MagicMock()
        cast("MagicMock", processor.snap_client).create_backscatter.return_value = OkResult(
            source=burst,
            result="rtc",
        )

        summary = processor.create_backscatter([burst], crs="EPSG:32632")

        cast("MagicMock", processor.snap_client).create_backscatter.assert_called_once()
        assert summary.products == ["rtc"]

    def test_create_backscatter_defaults_to_rtc(self, processor: BurstProcessor) -> None:
        client = cast("MagicMock", processor.snap_client)
        client.create_backscatter.return_value = OkResult(source="b", result="rtc")

        processor.create_backscatter([MagicMock()], crs="EPSG:32632")

        assert isinstance(client.create_backscatter.call_args.kwargs["config"], RTCConfig)

    def test_create_backscatter_passes_a_gtc_config_on(self, processor: BurstProcessor) -> None:
        client = cast("MagicMock", processor.snap_client)
        client.create_backscatter.return_value = OkResult(source="b", result="gtc")
        config = GTCConfig(multilook=None)

        summary = processor.create_backscatter(
            [MagicMock()],
            crs="EPSG:32632",
            config=config,
        )

        assert client.create_backscatter.call_args.kwargs["config"] is config
        assert summary.products == ["gtc"]

    def test_create_backscatter_rejects_another_config(self, processor: BurstProcessor) -> None:
        with pytest.raises(TypeError, match="must be an RTCConfig or a GTCConfig"):
            # The wrong type is what is under test.
            # pyrefly: ignore [no-matching-overload]
            processor.create_backscatter([MagicMock()], crs="EPSG:32632", config=CoherenceConfig())

        cast("MagicMock", processor.snap_client).create_backscatter.assert_not_called()

    def test_create_coherences(self, processor: BurstProcessor) -> None:
        pair = MagicMock()
        cast("MagicMock", processor.snap_client).create_coherence.return_value = OkResult(
            source=pair,
            result="coh",
        )

        summary = processor.create_coherences([pair], crs="EPSG:32632")

        cast("MagicMock", processor.snap_client).create_coherence.assert_called_once()
        assert summary.products == ["coh"]

    def test_create_interferograms(self, processor: BurstProcessor) -> None:
        pair = MagicMock()
        cast("MagicMock", processor.snap_client).create_interferogram.return_value = OkResult(
            source=pair,
            result="ifg",
        )

        summary = processor.create_interferograms([pair], crs="EPSG:32632")

        cast("MagicMock", processor.snap_client).create_interferogram.assert_called_once()
        assert summary.products == ["ifg"]

    @pytest.mark.parametrize(
        "method",
        ["create_backscatter", "create_coherences", "create_interferograms"],
    )
    @pytest.mark.parametrize(
        "crs",
        [
            pytest.param("EPSG:3035", id="laea-europe"),
            pytest.param(4326, id="geographic"),
            pytest.param("EPSG:25832", id="etrs89-utm"),
        ],
    )
    def test_crs_that_is_not_a_wgs84_utm_zone_fails_before_processing(
        self,
        processor: BurstProcessor,
        method: str,
        crs: str | int,
    ) -> None:
        with pytest.raises(ValueError, match="must be a WGS 84 / UTM zone"):
            getattr(processor, method)([MagicMock()], crs=crs)

        assert not cast("MagicMock", processor.snap_client).method_calls

    @pytest.mark.parametrize(
        "method",
        ["create_backscatter", "create_coherences", "create_interferograms"],
    )
    def test_crs_is_checked_even_without_items(
        self,
        processor: BurstProcessor,
        method: str,
    ) -> None:
        with pytest.raises(ValueError, match="must be a WGS 84 / UTM zone"):
            getattr(processor, method)([], crs="EPSG:3035")

    @pytest.mark.parametrize(
        "crs",
        ["EPSG:32601", 32760, "+proj=utm +zone=32 +south +datum=WGS84 +units=m"],
    )
    def test_accepts_any_wgs84_utm_zone(self, processor: BurstProcessor, crs: str | int) -> None:
        processor.create_backscatter([], crs=crs)

    def test_invalid_crs_fails_before_processing(self, processor: BurstProcessor) -> None:
        with pytest.raises(CRSError):
            processor.create_backscatter([MagicMock()], crs="EPSG:32632x")

        cast("MagicMock", processor.snap_client).create_backscatter.assert_not_called()
