"""Live integration tests against CDSE and a local SNAP install.

Deselected by default (see ``addopts`` in pyproject.toml); run with::

    uv run pytest -m integration

Each test skips itself when its prerequisites (network credentials, SNAP)
are not configured, so the suite degrades gracefully on machines without a
full setup. Credentials are read from the environment or a ``.env`` file in
the working directory — on a development machine, symlink or copy the
project ``.env`` here (it is gitignored).
"""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import TYPE_CHECKING

import numpy as np
from pandas import Timestamp
import pyproj
import pytest
import rasterio

from s1bursts import (
    BurstDatabase,
    BurstProcessor,
    FileManager,
    InterferogramConfig,
    SearchQuery,
    SNAPConfig,
)
from s1bursts.config import get_settings
from s1bursts.dem import _get_transformer
from s1bursts.errors import ConfigurationError
from s1bursts.snap_client import _resolve_gpt_path
from tests.safe_conformance import check_burst_safe, compare_burst_safes

if TYPE_CHECKING:
    from s1bursts.burst import Burst, BurstCollection, BurstPairCollection

pytestmark = pytest.mark.integration

# Aletsch glacier burst used across all live tests (~100 MB download)
BURST_ID = 30346
SUB_SWATH = "IW1"
POLARIZATION = "VV"
TIME_RANGE = ("2022-08-20", "2022-08-30")
# Interferometric processing needs a second acquisition of the same burst: the
# 12-day repeat that follows the one above, which is also the pair the HyP3
# coherence validation uses for this burst (see tests/insar_validation.py).
PAIR_TIME_RANGE = ("2022-08-20", "2022-09-10")
PAIR_BASELINE_DAYS = 12

_settings = get_settings()

requires_cdse_credentials = pytest.mark.skipif(
    _settings.cdse.username is None or _settings.cdse.password is None,
    reason="CDSE_USERNAME / CDSE_PASSWORD not configured",
)
requires_cdse_s3 = pytest.mark.skipif(
    _settings.cdse.s3.profile is None
    and (_settings.cdse.s3.access_key_id is None or _settings.cdse.s3.secret_access_key is None),
    reason="CDSE_S3_PROFILE or CDSE_S3_ACCESS_KEY_ID / CDSE_S3_SECRET_ACCESS_KEY not configured",
)


def _gpt_available() -> bool:
    try:
        _resolve_gpt_path(None, _settings.snap)
    except ConfigurationError:
        return False
    return True


requires_snap = pytest.mark.skipif(
    not _gpt_available(),
    reason="SNAP `gpt` executable not resolvable (set SNAP_BIN or add gpt to PATH)",
)


@pytest.fixture(scope="module")
def file_manager(tmp_path_factory: pytest.TempPathFactory) -> FileManager:
    return FileManager(base_dir=tmp_path_factory.mktemp("s1bursts-integration"))


@pytest.fixture(scope="module")
def database(file_manager: FileManager) -> BurstDatabase:
    return BurstDatabase(file_manager=file_manager)


@pytest.fixture(scope="module")
def query() -> SearchQuery:
    return SearchQuery(
        time_range=TIME_RANGE,
        burst_id=BURST_ID,
        sub_swath=SUB_SWATH,
        polarization=POLARIZATION,
    )


class TestCatalogSearch:
    """Catalog search needs network access but no credentials."""

    def test_search_returns_expected_burst(
        self,
        database: BurstDatabase,
        query: SearchQuery,
    ) -> None:
        bursts = database.search(query)

        assert len(bursts) >= 1
        burst = bursts[0]
        assert burst.burst_id == BURST_ID
        assert burst.sub_swath == SUB_SWATH
        assert burst.date >= Timestamp(TIME_RANGE[0], tz="UTC")
        assert burst.cdse_id is not None
        assert burst.parent_product_name is not None

    def test_near_identical_timestamp_products_are_deduplicated(
        self,
        database: BurstDatabase,
    ) -> None:
        """CDSE returns two products for this burst whose timestamps differ by ~6 us.

        They come from two overlapping parent SLC slices and differ in extent. The
        collection must collapse them to a single burst, keeping the more complete
        product (more samples/lines) rather than merely the most recent, and warn.
        """
        query = SearchQuery(
            burst_id=78124,
            sub_swath="IW1",
            polarization="VH",
            time_range=("2018-07-24", "2018-07-27"),
        )

        with pytest.warns(UserWarning, match="most complete"):
            bursts = database.search(query)

        days = [burst.date.normalize() for burst in bursts]
        # No two surviving bursts share a calendar day.
        assert len(days) == len(set(days))
        # The fuller product (23281 vs 22424 samples/burst) is kept, even though it
        # has the slightly earlier sensing time.
        kept = next(b for b in bursts if b.date.normalize() == Timestamp("2018-07-25", tz="UTC"))
        assert kept.cdse_id == "e756a6f6-91a2-4d7e-b818-b2b3c3988cf6"


@pytest.fixture(scope="module")
def downloaded_bursts(database: BurstDatabase, query: SearchQuery) -> BurstCollection:
    """Download one burst via the OData endpoint (skips without credentials)."""
    if _settings.cdse.username is None or _settings.cdse.password is None:
        pytest.skip("CDSE_USERNAME / CDSE_PASSWORD not configured")
    bursts = database.search(query)[0:1]
    summary = database.download(bursts, endpoint="odata")
    assert summary.n_failed == 0, [str(item.error) for item in summary.failed_items]
    return bursts


class TestDownload:
    @requires_cdse_credentials
    def test_odata_download_produces_slc_zip(
        self,
        database: BurstDatabase,
        downloaded_bursts: BurstCollection,
    ) -> None:
        slc_path = database.file_manager.get_slc_path(downloaded_bursts[0])

        assert slc_path.exists()
        assert slc_path.stat().st_size > 10_000_000  # sanity: tens of MB

    @requires_cdse_s3
    def test_s3_extraction_produces_slc_zip(self, s3_extracted_burst: Path) -> None:
        assert s3_extracted_burst.exists()
        assert s3_extracted_burst.stat().st_size > 10_000_000


@pytest.fixture(scope="module")
def s3_extracted_burst(
    database: BurstDatabase,
    query: SearchQuery,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Extract the reference burst from S3 with this package's own extractor."""
    if _settings.cdse.s3.profile is None and (
        _settings.cdse.s3.access_key_id is None or _settings.cdse.s3.secret_access_key is None
    ):
        pytest.skip("CDSE S3 credentials not configured")

    burst: Burst = database.search(query)[0]
    assert burst.parent_product_name is not None
    save_path = tmp_path_factory.mktemp("s3-extract") / "burst-from-s3.zip"

    database.client.download_burst_from_s3(
        save_path=save_path,
        parent_product_name=burst.parent_product_name,
        polarization=burst.polarization,
        sub_swath=burst.sub_swath,
        burst_id=burst.burst_id,
    )
    return save_path


class TestReferenceConformance:
    """Validate the S3 extractor against CDSE's own product for the same burst.

    Products served by CDSE's OData burst endpoint are the behavioral source of
    truth for the S3 fallback wherever the two endpoints overlap.
    """

    @requires_cdse_s3
    def test_extracted_burst_satisfies_the_reference_contract(
        self,
        s3_extracted_burst: Path,
    ) -> None:
        violations = check_burst_safe(s3_extracted_burst, sub_swath=SUB_SWATH)

        assert violations == [], "\n".join(violations)

    @requires_cdse_credentials
    @requires_cdse_s3
    def test_matches_the_burst_cdse_serves_for_the_same_id(
        self,
        s3_extracted_burst: Path,
        downloaded_bursts: BurstCollection,
        database: BurstDatabase,
    ) -> None:
        """The S3 extraction must equal CDSE's own burst, file for file and pixel for pixel.

        The single tolerated difference is the GCP row offset some CDSE products
        carry; it is reported explicitly rather than ignored (see
        ``tests/safe_conformance._diff_gcps``).
        """
        reference = database.file_manager.get_slc_path(downloaded_bursts[0])
        assert Path(reference).exists(), "OData reference burst was not downloaded"

        differences = compare_burst_safes(s3_extracted_burst, Path(reference))
        unexpected = [d for d in differences if not d.startswith("KNOWN DEVIATION:")]

        assert unexpected == [], "\n".join(unexpected)


@pytest.fixture(scope="module")
def downloaded_pair(database: BurstDatabase) -> BurstPairCollection:
    """Download both bursts of one 12-day pair (skips without credentials).

    Shares the module's file manager with :func:`downloaded_bursts`, so the
    primary is only downloaded once when both fixtures are used.
    """
    if _settings.cdse.username is None or _settings.cdse.password is None:
        pytest.skip("CDSE_USERNAME / CDSE_PASSWORD not configured")
    bursts = database.search(
        SearchQuery(
            time_range=PAIR_TIME_RANGE,
            burst_id=BURST_ID,
            sub_swath=SUB_SWATH,
            polarization=POLARIZATION,
        ),
    )
    pairs = bursts.find_pairs(PAIR_BASELINE_DAYS)[0:1]
    assert len(pairs) == 1, (
        f"no {PAIR_BASELINE_DAYS}-day pair for burst {BURST_ID} in {PAIR_TIME_RANGE}"
    )
    summary = database.download(pairs, endpoint="odata")
    assert summary.n_failed == 0, [str(item.error) for item in summary.failed_items]
    return pairs


class TestSnapProcessing:
    @requires_cdse_credentials
    @requires_snap
    def test_gpt_creates_rtc_backscatter(
        self,
        file_manager: FileManager,
        downloaded_bursts: BurstCollection,
    ) -> None:
        """End-to-end GPT smoke test: SLC zip -> RTC GeoTIFF (takes minutes)."""
        # Explicit modest resource budget: the machine-wide SNAP_MAX_RAM_GB /
        # SNAP_MAX_WORKERS may be tuned for a larger production VM, and an
        # oversized -Xmx gets the JVM OOM-killed on smaller dev machines.
        # A single-burst RTC fits in a ~3 GB heap.
        processor = BurstProcessor(
            file_manager,
            SNAPConfig(max_workers=1, max_ram_gb=6),
            show_progress=False,
            continue_on_error=False,
        )

        summary = processor.create_backscatter(
            downloaded_bursts,
            crs=pyproj.CRS("EPSG:32632"),
        )

        assert summary.n_failed == 0, [str(item.error) for item in summary.failed_items]
        product_path = file_manager.get_product_path(summary.products[0])
        assert product_path.exists()
        with rasterio.open(product_path) as src:
            assert src.count == 2  # Gamma0 + simulatedImage
            assert src.crs.to_epsg() == 32632
            # SNAP's GeoTIFF writer drops the NODATA tag; _finalize_product re-stamps it
            assert src.nodata is not None
            assert np.isnan(src.nodata)

    @requires_cdse_s3
    @requires_snap
    def test_gpt_processes_the_exact_s3_extracted_artifact(
        self,
        database: BurstDatabase,
        query: SearchQuery,
        s3_extracted_burst: Path,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """The S3-extracted ZIP itself must be readable by SNAP, not merely well formed.

        The archive produced by :func:`extract_burst_from_s3` is placed where the
        file manager expects the SLC, so GPT consumes that exact byte stream.
        """
        bursts = database.search(query)[0:1]
        file_manager = FileManager(base_dir=tmp_path_factory.mktemp("s3-snap"))
        slc_path = Path(file_manager.get_slc_path(bursts[0]))
        slc_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(s3_extracted_burst, slc_path)

        processor = BurstProcessor(
            file_manager,
            SNAPConfig(max_workers=1, max_ram_gb=6),
            show_progress=False,
            continue_on_error=False,
        )
        summary = processor.create_backscatter(bursts, crs=pyproj.CRS("EPSG:32632"))

        assert summary.n_failed == 0, [str(item.error) for item in summary.failed_items]
        with rasterio.open(file_manager.get_product_path(summary.products[0])) as src:
            assert src.crs.to_epsg() == 32632
            assert src.count == 2
            # A burst that decoded correctly has real signal, not an empty frame.
            assert np.isfinite(src.read(1)).any()

    @requires_cdse_credentials
    @requires_snap
    def test_gpt_creates_interferogram(
        self,
        file_manager: FileManager,
        downloaded_pair: BurstPairCollection,
    ) -> None:
        """End-to-end GPT smoke test for the interferometric graph: two SLC zips -> IFG.

        The interferogram has no independent reference to be validated against
        the way RTC and coherence are, so what is asserted here is that the
        graph runs to completion and that its product is shaped like a
        geocoded complex interferogram.
        """
        # A larger budget than the RTC tests above, and fewer threads: the peak
        # is InterferogramOp's topographic-phase step, which triangulates the
        # DEM per tile, so concurrent tiles multiply it. 6 GB across the default
        # eight threads runs out of Java heap there; 8 GB across two does not.
        processor = BurstProcessor(
            file_manager,
            SNAPConfig(max_workers=1, max_ram_gb=8, gpt_threads_per_job=2),
            show_progress=False,
            continue_on_error=False,
        )
        config = InterferogramConfig()

        summary = processor.create_interferograms(
            downloaded_pair,
            crs=pyproj.CRS("EPSG:32632"),
            config=config,
        )

        assert summary.n_failed == 0, [str(item.error) for item in summary.failed_items]
        product_path = file_manager.get_product_path(summary.products[0])
        assert product_path.exists()

        with rasterio.open(product_path) as src:
            assert src.count == len(config.output_bands)
            assert src.descriptions == tuple(config.output_bands)
            assert src.crs.to_epsg() == 32632
            # SNAP's GeoTIFF writer drops the NODATA tag; _finalize_product re-stamps it
            assert src.nodata is not None
            assert np.isnan(src.nodata)
            # Single-look output on the standard grid: no Multilook step changes
            # the posting, and pixel corners sit on multiples of the spacing.
            spacing = config.grid_spacing
            assert src.res == (spacing, spacing)
            assert src.transform.c % spacing == pytest.approx(0.0)
            assert src.transform.f % spacing == pytest.approx(0.0)

            ifg_i, ifg_q, intensity_1, intensity_2 = src.read()

        valid = np.isfinite(ifg_i)
        # A geocoded burst is a parallelogram in its bounding box, so a large
        # share of the frame is legitimately NaN — but most of it is not.
        assert valid.mean() > 0.1
        # Real and imaginary parts are the two halves of one complex sample and
        # are therefore valid over exactly the same pixels.
        assert np.array_equal(valid, np.isfinite(ifg_q))
        # Interferometric phase, not a magnitude image: both components take
        # either sign.
        for band, name in ((ifg_i, "ifg_i"), (ifg_q, "ifg_q")):
            assert (band[valid] > 0).any(), f"{name} has no positive samples"
            assert (band[valid] < 0).any(), f"{name} has no negative samples"
        # Both intensities are powers of the same coregistered samples, so they
        # are nonnegative and valid over exactly the pixels the interferogram is.
        for band, name in ((intensity_1, "intensity_1"), (intensity_2, "intensity_2")):
            assert np.array_equal(np.isfinite(band), valid), f"{name} has a different extent"
            assert (band[valid] >= 0).all(), f"{name} has negative samples"


class TestVerticalDatumGrids:
    """Downloading a PROJ grid must make it usable without restarting Python.

    PROJ answers grid availability from a cache held by its context, and pyproj
    keeps one context per thread for the life of the process. Only a real PROJ
    install shows that, so the unit suite cannot: against stand-ins, a group
    rebuilt after a download simply reports whatever the stand-in was told to.
    """

    @staticmethod
    def test_downloaded_grid_is_usable_in_the_same_process(tmp_path: Path) -> None:
        # EGM96 reaches EPSG:4979 through a single operation needing a single
        # open-license grid, so nothing else can account for a success here.
        grid = Path(pyproj.datadir.get_user_data_dir(create=True)) / "us_nga_egm96_15.tif"
        stashed = tmp_path / grid.name
        if grid.exists():
            shutil.move(grid, stashed)

        try:
            assert not grid.exists()
            transformer = _get_transformer(
                pyproj.CRS("EPSG:4326+5773"),
                download_missing_grids=True,
            )
            assert grid.is_file()

            # Bernese Alps: EGM96 sits roughly 50 m above the ellipsoid there.
            _, _, ellipsoidal = transformer.transform(7.5, 46.5, 2000.0, errcheck=True)
            assert ellipsoidal == pytest.approx(2050.0, abs=2.0)
        finally:
            if stashed.exists():
                shutil.move(stashed, grid)
