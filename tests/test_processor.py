"""Tests for s1bursts.processor batch processing."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, patch
import warnings

import numpy as np
from pandas import Timestamp
from pyproj import CRS
import pytest
import rasterio
import rasterio.transform
import shapely
from upath import UPath

from s1bursts import product_tags
from s1bursts.burst import Burst, BurstPair
from s1bursts.errors import (
    BurstMapUnavailableError,
    NoUTMZoneError,
    ProductSettingsMismatchError,
    SnapProcessingError,
)
from s1bursts.filemanager import FileManager
from s1bursts.processor import BurstProcessor
from s1bursts.product import CoherenceProduct, RTCProduct
from s1bursts.snap_client import SNAPConfig
from s1bursts.snap_graph import CoherenceConfig, GTCConfig, RTCConfig
from s1bursts.summary import ErrorResult, OkResult, OperationSummary
from tests.gated_pool import Gate

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
        cast("MagicMock", proc.manager).product_exists.return_value = False
        yield proc


def _burst(date: str, burst_id: int = 30346) -> Burst:
    """Build a burst carrying its orbit and pass direction, as a search returns it."""
    return Burst(
        burst_id=burst_id,
        sub_swath="IW1",
        polarization="VV",
        date=Timestamp(date),
        _rel_orbit_number=66,
        _pass_direction="Ascending",
    )


def _pair() -> BurstPair:
    return BurstPair(_burst("2022-01-05"), _burst("2022-01-17"))


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

        with pytest.raises(ValueError, match="Raise `max_ram_gb`"):
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


class TestStoppingABatch:
    """A batch that stops early cancels its queue and says what it kept."""

    @pytest.fixture
    def gate(self, processor: BurstProcessor, monkeypatch: pytest.MonkeyPatch) -> Gate:
        processor.config.resolved_max_workers = 2
        gate = Gate(n_running=2)
        monkeypatch.setattr("s1bursts.processor.ThreadPoolExecutor", gate.executor())
        return gate

    def test_ctrl_c_cancels_the_queued_items(
        self,
        processor: BurstProcessor,
        gate: Gate,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Previously the queue was drained before the KeyboardInterrupt surfaced."""
        monkeypatch.setattr("s1bursts.processor.as_completed", gate.interrupt)

        def process_fn(item: str) -> OkResult[str]:
            gate.enter(item)
            return OkResult(source=item, result=item)

        with pytest.raises(KeyboardInterrupt) as excinfo:
            processor._process_collection(list("abcdef"), process_fn, "Test")

        assert sorted(cast("list[str]", gate.started)) == ["a", "b"]
        assert excinfo.value.__notes__ == [
            (
                "Stopped with 2 of 6 items finished; products already written are kept, "
                "and a rerun without overwrite skips them."
            )
        ]

    @pytest.mark.parametrize("continue_on_error", [True, False])
    def test_an_error_cancels_the_queued_items(
        self,
        processor: BurstProcessor,
        gate: Gate,
        continue_on_error: bool,
    ) -> None:
        """An unexpected error, or a recorded one without continue_on_error, ends the batch."""
        processor.continue_on_error = continue_on_error

        def process_fn(item: str) -> OkResult[str] | ErrorResult:
            if item == "a":
                if continue_on_error:
                    msg = "bug"
                    raise RuntimeError(msg)
                return ErrorResult(source=item, error=SnapProcessingError("bug"))
            gate.enter(item)
            return OkResult(source=item, result=item)

        with pytest.raises((RuntimeError, SnapProcessingError), match="bug") as excinfo:
            processor._process_collection(list("abcdefgh"), process_fn, "Test")

        # The worker freed by "a" may have taken "c" before the queue was cancelled.
        assert set(cast("list[str]", gate.started)) <= {"b", "c"}
        n_finished = 1 + len(gate.started)
        assert f"Stopped with {n_finished} of 8 items finished" in excinfo.value.__notes__[0]

    def test_ctrl_c_while_submitting_cancels_what_was_queued(
        self,
        processor: BurstProcessor,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        processor.config.resolved_max_workers = 2
        gate = Gate(n_running=2)
        monkeypatch.setattr(
            "s1bursts.processor.ThreadPoolExecutor",
            gate.executor(interrupt_at_submit=4),
        )

        def process_fn(item: str) -> OkResult[str]:
            gate.enter(item)
            return OkResult(source=item, result=item)

        with pytest.raises(KeyboardInterrupt) as excinfo:
            processor._process_collection(list("abcdef"), process_fn, "Test")

        # "c" and "d" were queued and cancelled; "a" or "b" may have been
        # cancelled too if no worker had taken it yet.
        assert set(cast("list[str]", gate.started)) <= {"a", "b"}
        n_finished = len(gate.started)
        assert f"Stopped with {n_finished} of 6 items finished" in excinfo.value.__notes__[0]

    def test_the_sequential_path_adds_the_note_too(self, processor: BurstProcessor) -> None:
        process_fn = MagicMock(
            side_effect=[OkResult(source="a", result="A"), KeyboardInterrupt, None],
        )

        with pytest.raises(KeyboardInterrupt) as excinfo:
            processor._process_collection(["a", "b", "c"], process_fn, "Test")

        assert process_fn.call_count == 2
        assert "Stopped with 1 of 3 items finished" in excinfo.value.__notes__[0]


class TestDispatch:
    """The create_* methods must call the matching SNAPClient method per item."""

    def test_create_backscatter(self, processor: BurstProcessor) -> None:
        burst = _burst("2022-01-05")
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

        processor.create_backscatter([_burst("2022-01-05")], crs="EPSG:32632")

        assert isinstance(client.create_backscatter.call_args.kwargs["config"], RTCConfig)

    def test_create_backscatter_passes_a_gtc_config_on(self, processor: BurstProcessor) -> None:
        client = cast("MagicMock", processor.snap_client)
        client.create_backscatter.return_value = OkResult(source="b", result="gtc")
        config = GTCConfig(multilook=None)

        summary = processor.create_backscatter(
            [_burst("2022-01-05")],
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
        pair = _pair()
        cast("MagicMock", processor.snap_client).create_coherence.return_value = OkResult(
            source=pair,
            result="coh",
        )

        summary = processor.create_coherences([pair], crs="EPSG:32632")

        cast("MagicMock", processor.snap_client).create_coherence.assert_called_once()
        assert summary.products == ["coh"]

    @pytest.mark.parametrize(
        "method",
        ["create_backscatter", "create_coherences"],
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
        ["create_backscatter", "create_coherences"],
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
        # Previously pyproj's CRSError, a RuntimeError naming neither UTM nor "auto".
        with pytest.raises(ValueError, match="must be a WGS 84 / UTM zone"):
            processor.create_backscatter([MagicMock()], crs="EPSG:32632x")

    @pytest.mark.parametrize("crs", ["AUTO", "Auto", " auto"])
    def test_a_misspelled_auto_is_named(self, processor: BurstProcessor, crs: str) -> None:
        with pytest.raises(ValueError, match="matched exactly"):
            processor.create_backscatter([MagicMock()], crs=crs)

        cast("MagicMock", processor.snap_client).create_backscatter.assert_not_called()


def _write_product(
    path: Path | UPath,
    product: RTCProduct | CoherenceProduct,
    config: RTCConfig | CoherenceConfig,
    *,
    snap_version: tuple[int, ...] | None = None,
    tagged: bool = True,
) -> None:
    """Write a one-pixel stand-in of ``product`` as ``config`` would have made it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tags = product_tags.burst_metadata(product, config, snap_version=snap_version).tags
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=1,
        height=1,
        count=1,
        dtype="float32",
        crs=product.crs,
        transform=rasterio.transform.from_origin(400000.0, 5200000.0, 10.0, 10.0),
    ) as dst:
        dst.write(np.ones((1, 1, 1), dtype="float32"))
        if tagged:
            dst.update_tags(**tags)


class TestSkipSettings:
    """Existing products are checked before any GPT run, and their skips summarised."""

    CRS = "EPSG:32632"

    @pytest.fixture
    def real(self, tmp_path: Path, config: SNAPConfig) -> BurstProcessor:
        """Build a processor with a real client and FileManager; tests stub only GPT."""
        return BurstProcessor(FileManager(tmp_path / "data"), config, show_progress=False)

    def _existing(
        self,
        processor: BurstProcessor,
        burst: Burst,
        config: RTCConfig,
        **kwargs: Any,
    ) -> None:
        product = RTCProduct.from_burst(burst, self.CRS)
        _write_product(processor.manager.get_product_path(product), product, config, **kwargs)

    @pytest.mark.parametrize("max_workers", [1, 2])
    def test_a_mismatch_raises_before_any_gpt_run(
        self, real: BurstProcessor, max_workers: int
    ) -> None:
        real.config.resolved_max_workers = max_workers
        missing, existing = _burst("2022-01-05"), _burst("2022-01-17")
        self._existing(real, existing, RTCConfig(grid_spacing=20.0))

        with (
            patch.object(real.snap_client, "_run_gpt") as mock_gpt,
            pytest.raises(ProductSettingsMismatchError, match="GRID_SPACING"),
        ):
            real.create_backscatter([missing, existing], crs=self.CRS, config=RTCConfig())

        # Previously the missing burst was processed before the mismatch surfaced.
        mock_gpt.assert_not_called()

    def test_an_unreadable_product_raises_before_any_gpt_run(self, real: BurstProcessor) -> None:
        missing, existing = _burst("2022-01-05"), _burst("2022-01-17")
        path = real.manager.get_product_path(RTCProduct.from_burst(existing, self.CRS))
        path.write_bytes(b"not a raster")

        with (
            patch.object(real.snap_client, "_run_gpt") as mock_gpt,
            pytest.raises(OSError, match="not recognized"),
        ):
            real.create_backscatter([missing, existing], crs=self.CRS)

        mock_gpt.assert_not_called()

    def test_existing_products_are_not_handed_to_the_client(self, real: BurstProcessor) -> None:
        bursts = [_burst("2022-01-05"), _burst("2022-01-17")]
        for burst in bursts:
            self._existing(real, burst, RTCConfig())

        with patch.object(real.snap_client, "create_backscatter") as create:
            summary = real.create_backscatter(bursts, crs=self.CRS)

        create.assert_not_called()
        assert summary.n_skipped == 2
        assert summary.products == [RTCProduct.from_burst(burst, self.CRS) for burst in bursts]

    # Previously skipped with one warning per call.
    def test_another_snap_version_raises_before_any_gpt_run(self, real: BurstProcessor) -> None:
        real.config.resolved_snap_version = (14, 0, 0)
        missing, existing = _burst("2022-01-05"), _burst("2022-01-17")
        self._existing(real, existing, RTCConfig(), snap_version=(13, 0, 0))

        with (
            patch.object(real.snap_client, "_run_gpt") as mock_gpt,
            pytest.raises(ProductSettingsMismatchError, match="PROCESSING_SOFTWARE_VERSION"),
        ):
            real.create_backscatter([missing, existing], crs=self.CRS)

        mock_gpt.assert_not_called()

    def test_an_unknown_snap_version_is_skipped_without_a_warning(
        self, real: BurstProcessor
    ) -> None:
        real.config.resolved_snap_version = (14, 0, 0)
        bursts = [_burst("2022-01-05"), _burst("2022-01-17")]
        self._existing(real, bursts[0], RTCConfig(), snap_version=None)
        self._existing(real, bursts[1], RTCConfig(), snap_version=(14, 0, 0))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            summary = real.create_backscatter(bursts, crs=self.CRS)

        assert summary.n_skipped == 2

    # Previously skipped without checking its settings, with a warning.
    def test_an_untagged_product_raises_before_any_gpt_run(self, real: BurstProcessor) -> None:
        missing, existing = _burst("2022-01-05"), _burst("2022-01-17")
        self._existing(real, existing, RTCConfig(), tagged=False)

        with (
            patch.object(real.snap_client, "_run_gpt") as mock_gpt,
            pytest.raises(ProductSettingsMismatchError, match="recorded none"),
        ):
            real.create_backscatter([missing, existing], crs=self.CRS)

        mock_gpt.assert_not_called()

    def test_auto_checks_each_existing_product_in_its_own_zone(self, real: BurstProcessor) -> None:
        west = replace(_burst("2022-01-05"), _geometry=shapely.box(8.0, 46.0, 9.0, 47.0))
        east = replace(
            _burst("2022-01-17", burst_id=30347), _geometry=shapely.box(16.0, 46.0, 17.0, 47.0)
        )
        for burst in (west, east):
            product = RTCProduct.from_burst(burst, "auto")
            _write_product(real.manager.get_product_path(product), product, RTCConfig())

        with patch.object(real.snap_client, "create_backscatter") as create:
            summary = real.create_backscatter([west, east], crs="auto")

        create.assert_not_called()
        assert [product.crs for product in summary.products] == [
            CRS.from_epsg(32632),
            CRS.from_epsg(32633),
        ]

    @pytest.mark.parametrize("overwrite", [False, True])
    def test_auto_fails_a_burst_without_a_zone_on_its_own(
        self, real: BurstProcessor, overwrite: bool
    ) -> None:
        polar = replace(_burst("2022-01-05"), _geometry=shapely.box(10.0, 84.5, 11.0, 85.5))
        alpine = replace(_burst("2022-01-17"), _geometry=shapely.box(8.0, 46.0, 9.0, 47.0))

        with (
            patch.object(real.snap_client, "_run_gpt") as mock_gpt,
            patch.object(real.snap_client, "_finalize_product"),
        ):
            summary = real.create_backscatter([polar, alpine], crs="auto", overwrite=overwrite)

        # Previously the polar burst stopped the whole batch.
        assert (summary.n_processed, summary.n_failed) == (1, 1)
        [failure] = summary.failed_items
        assert failure.source is polar
        assert isinstance(failure.error, NoUTMZoneError)
        mock_gpt.assert_called_once()

    def test_auto_without_a_zone_stops_a_batch_that_does_not_continue(
        self, real: BurstProcessor
    ) -> None:
        real.continue_on_error = False
        polar = replace(_burst("2022-01-05"), _geometry=shapely.box(10.0, 84.5, 11.0, 85.5))

        with pytest.raises(NoUTMZoneError, match="No UTM zone covers"):
            real.create_backscatter([polar], crs="auto")

    def test_auto_hands_the_client_auto(self, processor: BurstProcessor) -> None:
        burst = replace(_burst("2022-01-05"), _geometry=shapely.box(8.0, 46.0, 9.0, 47.0))
        client = cast("MagicMock", processor.snap_client)
        client.create_backscatter.return_value = OkResult(source=burst, result="rtc")

        processor.create_backscatter([burst], crs="auto")

        assert client.create_backscatter.call_args.args[1] == "auto"

    def test_auto_without_the_burst_map_fails_before_any_item(
        self, processor: BurstProcessor
    ) -> None:
        with (
            patch(
                "s1bursts.burst.create_s1_geometry_lut",
                side_effect=BurstMapUnavailableError("not downloaded"),
            ),
            pytest.raises(BurstMapUnavailableError),
        ):
            processor.create_coherences([_pair()], crs="auto")

        cast("MagicMock", processor.snap_client).create_coherence.assert_not_called()

    def test_overwrite_neither_checks_nor_skips(self, real: BurstProcessor) -> None:
        burst = _burst("2022-01-05")
        self._existing(real, burst, RTCConfig(grid_spacing=20.0))
        result = OkResult(source=burst, result=RTCProduct.from_burst(burst, self.CRS))

        with (
            patch.object(
                real.manager, "product_exists", wraps=real.manager.product_exists
            ) as exists,
            patch.object(real.snap_client, "create_backscatter", return_value=result) as create,
        ):
            summary = real.create_backscatter([burst], crs=self.CRS, overwrite=True)

        create.assert_called_once()
        exists.assert_not_called()
        assert summary.n_processed == 1

    def test_existing_coherence_products_are_checked_up_front(self, real: BurstProcessor) -> None:
        pair = _pair()
        product = CoherenceProduct.from_burst_pair(pair, self.CRS)
        _write_product(real.manager.get_product_path(product), product, CoherenceConfig())

        with patch.object(real.snap_client, "create_coherence") as create:
            summary = real.create_coherences([pair], crs=self.CRS, config=CoherenceConfig())

        create.assert_not_called()
        assert summary.products == [product]

    def test_a_mismatch_inside_a_worker_still_names_no_rerun(
        self, processor: BurstProcessor
    ) -> None:
        mismatch = ProductSettingsMismatchError("other settings", path=Path("p.tif"))

        with pytest.raises(ProductSettingsMismatchError) as excinfo:
            processor._process_collection(["a", "b"], MagicMock(side_effect=mismatch), "Test")

        # A rerun would meet the same product, so the note does not promise a skip.
        assert excinfo.value.__notes__ == [
            "Stopped with 0 of 2 items finished; products already written are kept."
        ]


class TestUnreadableSlc:
    """A SNAP read failure reaches the summary with the SLC's own location."""

    @pytest.fixture
    def failing_gpt(self, config: SNAPConfig) -> SNAPConfig:
        """Make the stub gpt fail as SNAP 14 does on a truncated source zip."""
        script = (
            "#!/bin/sh\n"
            'for arg in "$@"; do case "$arg" in -Ssource=*) src="${arg#-Ssource=}";; esac; done\n'
            'echo "Error: [NodeId: ReadOp@source] '
            "No product reader found for file '$src'.\" >&2\n"
            "exit 1\n"
        )
        config.resolved_gpt_path.write_text(script)
        return config

    @pytest.mark.parametrize("remote", [False, True], ids=["local", "remote"])
    def test_the_hint_names_where_the_slc_is_kept(
        self,
        tmp_path: Path,
        failing_gpt: SNAPConfig,
        remote: bool,
    ) -> None:
        """A remote SLC is read from a staged copy, which is gone once the error is read."""
        slc_dir = UPath(f"memory://{tmp_path.name}/slc") if remote else tmp_path / "slc"
        manager = FileManager(tmp_path / "data", slc_dir=slc_dir)
        burst = _burst("2022-01-05")
        slc = manager.get_slc_path(burst)
        slc.parent.mkdir(parents=True, exist_ok=True)
        slc.write_bytes(b"PK truncated")
        processor = BurstProcessor(manager, failing_gpt, show_progress=False)

        summary = processor.create_backscatter([burst], crs="EPSG:32632")

        [failure] = summary.failed_items
        assert isinstance(failure, ErrorResult)
        assert str(failure.error).endswith(
            f"\nThe input SLC '{slc}' may be truncated or corrupt: "
            "delete it and download it again."
        )
