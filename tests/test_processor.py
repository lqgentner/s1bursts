"""Tests for s1bursts.processor batch processing."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from s1bursts.errors import SnapProcessingError
from s1bursts.processor import BurstProcessor
from s1bursts.snap_client import ProcessingConfig
from s1bursts.summary import ErrorResult, OkResult, OperationSummary

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProcessingConfig:
    """ProcessingConfig with a stub gpt executable and isolated env."""
    for var in ("SNAP_GPT_PATH", "SNAP_MAX_WORKERS", "SNAP_THREADS_PER_JOB"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    gpt = tmp_path / "gpt"
    gpt.touch()
    return ProcessingConfig(file_manager=MagicMock(), gpt_path=gpt)


@pytest.fixture
def processor(config: ProcessingConfig) -> Iterator[BurstProcessor]:
    with patch("s1bursts.processor.SNAPClient") as mock_client_cls:
        proc = BurstProcessor(config, show_progress=False)
        proc.snap_client = mock_client_cls.return_value
        yield proc


class TestProcessingConfig:
    def test_missing_gpt_path_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SNAP_GPT_PATH", raising=False)
        monkeypatch.chdir(tmp_path)

        with pytest.raises(Exception, match="SNAP GPT path is None"):
            ProcessingConfig(file_manager=MagicMock())

    def test_nonexistent_gpt_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(Exception, match="executable not found"):
            ProcessingConfig(file_manager=MagicMock(), gpt_path=tmp_path / "missing-gpt")

    def test_resource_resolution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SNAP_MAX_WORKERS", raising=False)
        monkeypatch.chdir(tmp_path)
        gpt = tmp_path / "gpt"
        gpt.touch()

        config = ProcessingConfig(
            file_manager=MagicMock(),
            gpt_path=gpt,
            max_workers=2,
            max_ram_gb=16,
        )

        assert config.resolved_max_workers == 2
        assert config.resolved_xmx_mb == (16 * 1024 - 3000) // 2
        # Tile cache is a ratio of the per-job heap, never larger than -Xmx
        assert config.resolved_cache_mb == int(config.resolved_xmx_mb * 0.70)
        assert config.resolved_cache_mb < config.resolved_xmx_mb

    def test_xmx_never_negative_for_small_ram(self, tmp_path: Path) -> None:
        gpt = tmp_path / "gpt"
        gpt.touch()

        config = ProcessingConfig(file_manager=MagicMock(), gpt_path=gpt, max_ram_gb=1)

        assert config.resolved_xmx_mb >= 512

    def test_profile_defaults_merged(self, config: ProcessingConfig) -> None:
        assert config.rasterio_profile is not None
        assert config.rasterio_profile["dtype"] == "float32"


class TestProcessCollection:
    def test_sequential_collects_results(self, processor: BurstProcessor) -> None:
        items = ["a", "b", "c"]
        process_fn = MagicMock(side_effect=lambda i: OkResult(source=i, result=i.upper()))

        summary = processor._process_collection(items, process_fn, "Test")  # type: ignore[arg-type]

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

        summary = processor._process_collection(["a", "b", "c"], process_fn, "Test")  # type: ignore[arg-type]

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
            processor._process_collection(["a"], process_fn, "Test")  # type: ignore[arg-type]

    def test_parallel_collects_results(self, processor: BurstProcessor) -> None:
        processor.config.resolved_max_workers = 3
        process_fn = MagicMock(side_effect=lambda i: OkResult(source=i, result=i.upper()))

        summary = processor._process_collection(["a", "b", "c"], process_fn, "Test")  # type: ignore[arg-type]

        assert summary.n_processed == 3
        assert sorted(summary.products) == ["A", "B", "C"]

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
            processor._process_collection(["a", "b"], process_fn, "Test")  # type: ignore[arg-type]


class TestDispatch:
    """The create_* methods must call the matching SNAPClient method per item."""

    def test_create_backscatter(self, processor: BurstProcessor) -> None:
        burst = MagicMock()
        processor.snap_client.create_backscatter.return_value = OkResult(
            source=burst,
            result="rtc",
        )

        summary = processor.create_backscatter([burst], crs=MagicMock())  # type: ignore[arg-type]

        processor.snap_client.create_backscatter.assert_called_once()
        assert summary.products == ["rtc"]

    def test_create_coherences(self, processor: BurstProcessor) -> None:
        pair = MagicMock()
        processor.snap_client.create_coherence.return_value = OkResult(source=pair, result="coh")

        summary = processor.create_coherences([pair], crs=MagicMock())  # type: ignore[arg-type]

        processor.snap_client.create_coherence.assert_called_once()
        assert summary.products == ["coh"]

    def test_create_interferograms(self, processor: BurstProcessor) -> None:
        pair = MagicMock()
        processor.snap_client.create_interferogram.return_value = OkResult(
            source=pair,
            result="ifg",
        )

        summary = processor.create_interferograms([pair], crs=MagicMock())  # type: ignore[arg-type]

        processor.snap_client.create_interferogram.assert_called_once()
        assert summary.products == ["ifg"]
