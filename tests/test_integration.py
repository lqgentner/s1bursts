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
from typing import TYPE_CHECKING

from pandas import Timestamp
import pyproj
import pytest
import rasterio

from s1bursts import BurstDatabase, BurstProcessor, FileManager, ProcessingConfig, SearchQuery
from s1bursts.config import get_settings

if TYPE_CHECKING:
    from s1bursts.burst import Burst, BurstCollection

pytestmark = pytest.mark.integration

# Aletsch glacier burst used across all live tests (~100 MB download)
BURST_ID = 30346
SUB_SWATH = "IW1"
POLARIZATION = "VV"
TIME_RANGE = ("2022-08-20", "2022-08-30")

_settings = get_settings()

requires_cdse_credentials = pytest.mark.skipif(
    _settings.cdse.username is None or _settings.cdse.password is None,
    reason="CDSE_USERNAME / CDSE_PASSWORD not configured",
)
requires_cdse_s3 = pytest.mark.skipif(
    _settings.cdse.s3.access_key_id is None or _settings.cdse.s3.secret_access_key is None,
    reason="CDSE_S3_ACCESS_KEY_ID / CDSE_S3_SECRET_ACCESS_KEY not configured",
)
requires_snap = pytest.mark.skipif(
    _settings.snap.gpt_path is None or not Path(_settings.snap.gpt_path).exists(),
    reason="SNAP_GPT_PATH not configured or gpt executable missing",
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
    def test_s3_extraction_produces_slc_zip(
        self,
        database: BurstDatabase,
        query: SearchQuery,
        tmp_path: Path,
    ) -> None:
        burst: Burst = database.search(query)[0]
        assert burst.parent_product_name is not None
        save_path = tmp_path / "burst-from-s3.zip"

        database.client.download_burst_from_s3(
            save_path=save_path,
            parent_product_name=burst.parent_product_name,
            polarization=burst.polarization,
            sub_swath=burst.sub_swath,
            burst_id=burst.burst_id,
        )

        assert save_path.exists()
        assert save_path.stat().st_size > 10_000_000


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
            ProcessingConfig(file_manager=file_manager, max_workers=1, max_ram_gb=6),
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
