"""Regenerate the committed OPERA cross-validation record.

Run this only when the comparison against OPERA deliberately changes -- a SNAP
upgrade, a graph change, a new scene -- and review the resulting diff to
``opera_reference.json`` before committing: every moved number is a change in
what users receive::

    uv sync --group validation
    uv run --env-file .env python tests/fixtures/regenerate_opera_reference.py
    uv run --env-file .env python tests/fixtures/regenerate_opera_reference.py iceland-iw3

Needs CDSE credentials, Earthdata credentials, and a working SNAP install. Each
scene downloads several bursts and runs SNAP GPT, so a full regeneration takes
a couple of minutes.

The record captures the toolchain that produced it, because the numbers are
only meaningful against a stated SNAP, Java, DEM and OPERA product version.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import pyproj

# Run as a script from anywhere: make the repository root importable so that the
# `tests` package resolves the same way it does under pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from s1bursts import (
    BurstDatabase,
    BurstProcessor,
    FileManager,
    SearchQuery,
    SNAPConfig,
)
from s1bursts.snap_graph import BackscatterConfig
from tests.opera_validation import (
    BLOCK,
    GRID_SPACING,
    NOISE_FLOOR_PERCENTILE,
    SCENES,
    Scene,
    agreement,
    percentile_offset_db,
)
from tests.raster_comparison import read_aligned
from tests.validation_toolchain import processing_fingerprint, toolchain, toolchain_conflict

if TYPE_CHECKING:
    from asf_search import ASFSession

    from s1bursts.query import Polarization

RECORD_PATH = Path(__file__).parent / "opera_reference.json"

# ASF distributes both as one dataset, told apart by processing level.
PER_ACQUISITION_LEVEL = "RTC"
STATIC_LEVEL = "RTC-STATIC"


def processing_config() -> BackscatterConfig:
    """Graph configuration every scene is processed with."""
    return BackscatterConfig(grid_spacing=GRID_SPACING)


def scene_fingerprint(scene: Scene) -> str:
    """Cache key for this scene's processed product; see `processing_fingerprint`."""
    return processing_fingerprint(processing_config(), scene.crs)


def fetch_opera(
    scene: Scene,
    destination: Path,
    session: ASFSession,
) -> tuple[dict[str, Path], list[str]]:
    """Download the per-acquisition and STATIC granules for one scene.

    Each search must match exactly one granule. A reprocessing that adds a
    second one is refused rather than resolved silently, because the layers are
    keyed by name and the last match would otherwise win unnoticed.

    Returns
    -------
    tuple of (dict, list of str)
        The downloaded files, mapped from ``"acq:<layer>"`` / ``"static:<layer>"``,
        and the source acquisitions the static granule was derived from.

    Raises
    ------
    RuntimeError
        If a search does not match exactly one granule, or the static granule
        was not derived from the acquisition ``Scene.static_date`` names.
    """
    # Imported here, not at module scope: `asf_search` lives in the optional
    # `validation` dependency group, and tests/test_opera_validation.py imports
    # this module's helpers without it installed.
    import asf_search  # noqa: PLC0415

    destination.mkdir(parents=True, exist_ok=True)
    wanted = {
        PER_ACQUISITION_LEVEL: (
            scene.search_window(scene.gamma0_date),
            (*(f"_{pol}.tif" for pol in scene.polarizations), "_mask.tif"),
        ),
        STATIC_LEVEL: (
            None,
            ("_rtc_anf_gamma0_to_beta0.tif", "_mask.tif"),
        ),
    }

    found: dict[str, Path] = {}
    static_sources: list[str] = []
    for level, (window, suffixes) in wanted.items():
        # The static granule is dated by the mission, not by an acquisition, so
        # only the per-acquisition search takes a time window.
        window_arguments = {} if window is None else {"start": window[0], "end": window[1]}
        granules = asf_search.search(
            dataset=asf_search.DATASET.OPERA_S1,
            processingLevel=level,
            operaBurstID=scene.opera_burst_id,
            maxResults=5,
            **window_arguments,
        )
        if len(granules) != 1:
            names = [granule.properties["fileID"] for granule in granules]
            msg = (
                f"expected exactly one OPERA {level} granule for "
                f"{scene.opera_burst_id}, found {names}; disambiguate the search "
                "before regenerating"
            )
            raise RuntimeError(msg)
        granule = granules[0]

        if level == STATIC_LEVEL:
            # The static layers encode the orbit of the one acquisition they were
            # derived from, so SNAP must process that same acquisition. Verify it
            # here rather than trusting `Scene.static_date` to have stayed correct
            # across an OPERA reprocessing.
            static_sources = list(granule.umm.get("InputGranules", []))
            stamp = scene.static_date.replace("-", "")
            if not any(stamp in source for source in static_sources):
                msg = (
                    f"{scene.name}: static granule was derived from {static_sources}, "
                    f"which does not include the {scene.static_date} acquisition named "
                    f"in Scene.static_date; the ANF comparison would be invalid"
                )
                raise RuntimeError(msg)

        # The granule's own URL is its HDF5 bundle; the GeoTIFF layers this
        # comparison reads hang off it as additional URLs.
        links = [granule.properties["url"], *(granule.properties.get("additionalUrls") or [])]
        urls = [link for link in links if link.endswith(suffixes)]
        missing = [url for url in urls if not (destination / url.rsplit("/", 1)[-1]).exists()]
        if missing:
            asf_search.download_urls(urls=missing, path=str(destination), session=session)

        kind = "static" if level == STATIC_LEVEL else "acq"
        for url in urls:
            name = url.rsplit("/", 1)[-1]
            layer = name.rsplit("v1.0", 1)[-1].lstrip("_")
            found[f"{kind}:{layer}"] = destination / name
    return found, static_sources


def process(scene: Scene, acquisition: str, polarization: Polarization, work: Path) -> Path:
    """Produce this package's RTC for one burst at OPERA's posting.

    The cached product is keyed by :func:`processing_fingerprint`, so a SNAP or
    graph change starts from a clean raster instead of silently reusing the
    previous one.
    """
    fingerprint = scene_fingerprint(scene)
    product = work / f"{scene.name}-{polarization}-{acquisition}-{fingerprint}.tif"
    if product.exists():
        return product

    manager = FileManager(base_dir=work / "s1bursts")
    database = BurstDatabase(file_manager=manager)
    bursts = database.search(
        SearchQuery(
            time_range=scene.search_window(acquisition),
            burst_id=scene.burst_id,
            sub_swath=scene.sub_swath,
            polarization=polarization,
        ),
    )
    if len(bursts) != 1:
        # More than one match means the window caught a neighbouring
        # acquisition, and SNAP would then be compared against OPERA's product
        # for a different date.
        msg = (
            f"expected exactly one burst for {scene.name} {polarization} on "
            f"{acquisition}, found {len(bursts)}; narrow `Scene.search_window` "
            "before regenerating"
        )
        raise RuntimeError(msg)

    summary = database.download(bursts, endpoint="odata")
    if summary.n_failed:
        msg = f"download failed: {[str(item.error) for item in summary.failed_items]}"
        raise RuntimeError(msg)

    processor = BurstProcessor(
        manager,
        SNAPConfig(max_workers=1, max_ram_gb=6),
        show_progress=False,
        continue_on_error=False,
    )
    result = processor.create_backscatter(
        bursts,
        crs=pyproj.CRS(scene.crs),
        config=processing_config(),
        overwrite=True,
    )
    if result.n_failed:
        msg = f"processing failed: {[str(item.error) for item in result.failed_items]}"
        raise RuntimeError(msg)

    # `get_product_path` may hand back a UPath for remote stores; these
    # products are always local, so a plain string round-trip is safe.
    shutil.copyfile(str(manager.get_product_path(result.products[0])), product)
    return product


def compare_scene(scene: Scene, work: Path, session: ASFSession) -> dict:
    """Run both comparisons for one scene and return its record entry."""
    files, static_sources = fetch_opera(scene, work / "opera", session)
    block_key = f"block_{BLOCK}x{BLOCK}_{int(BLOCK * GRID_SPACING)}m"

    entry: dict[str, Any] = {
        "burst_tag": scene.burst_tag,
        "description": scene.description,
        "crs": scene.crs,
        "gamma0_acquisition": scene.gamma0_date,
        "static_source_acquisition": scene.static_date,
        "opera_granules": sorted({path.name for path in files.values()}),
        "opera_static_source_granules": static_sources,
        "processing_fingerprint": scene_fingerprint(scene),
        "gamma0": {},
    }

    acquisition_mask = files["acq:mask.tif"]
    for polarization in scene.polarizations:
        # A scene declares the polarizations it is meant to validate. A missing
        # asset is a broken campaign, not a comparison to quietly skip.
        opera_tif = files[f"acq:{polarization}.tif"]

        ours_path = process(scene, scene.gamma0_date, polarization, work)
        # The mask is read as a companion so that it lands on the same window as
        # the pair it masks, whatever its own bounds are.
        ours, theirs, (mask,), alignment = read_aligned(
            ours_path,
            opera_tif,
            "Gamma0",
            companions=(acquisition_mask,),
        )
        # OPERA's mask: 0 = clear, 1/2/3 = shadow/layover, 255 = nodata.
        clear = (mask == 0) & np.isfinite(ours) & np.isfinite(theirs)

        entry["gamma0"][polarization] = {
            "alignment": alignment,
            block_key: agreement(ours, theirs, clear, k=BLOCK),
            # The dark tail, where the two noise treatments diverge.
            f"p{int(NOISE_FLOOR_PERCENTILE):02d}_dB_offset": percentile_offset_db(
                ours,
                theirs,
                clear,
            ),
        }

    ours_path = process(scene, scene.static_date, "VV", work)
    ours, theirs, (mask,), alignment = read_aligned(
        ours_path,
        files["static:rtc_anf_gamma0_to_beta0.tif"],
        "simulatedImage",
        companions=(files["static:mask.tif"],),
    )
    clear = (mask == 0) & np.isfinite(ours) & np.isfinite(theirs)

    entry["anf"] = {
        "alignment": alignment,
        "layover_shadow_excluded": agreement(ours, theirs, clear, k=1),
    }
    return entry


def login() -> ASFSession:
    """Open an ASF session with the Earthdata credentials in the environment.

    Raises
    ------
    RuntimeError
        If the credentials are not set.
    """
    # Imported here, not at module scope: `asf_search` lives in the optional
    # `validation` dependency group.
    import asf_search  # noqa: PLC0415

    username = os.environ.get("EOSDIS_USERNAME")
    password = os.environ.get("EOSDIS_PASSWORD")
    if not (username and password):
        msg = "set EOSDIS_USERNAME and EOSDIS_PASSWORD (or run with `--env-file .env`)"
        raise RuntimeError(msg)
    return asf_search.ASFSession().auth_with_creds(username, password)


def main() -> int:
    """Regenerate the record, for one named scene or all of them."""
    requested = sys.argv[1] if len(sys.argv) > 1 else None
    scenes = [s for s in SCENES if requested is None or s.name == requested]
    if not scenes:
        print(f"unknown scene {requested!r}; known: {[s.name for s in SCENES]}")
        return 1

    record: dict[str, Any] = {}
    if RECORD_PATH.exists():
        record = json.loads(RECORD_PATH.read_text())

    # Measured before the campaign rather than after it, so that a run which
    # cannot honestly write this provenance stops before spending the downloads.
    versions = toolchain(
        dem="Copernicus 30m Global DEM (SNAP auto-download)",
        grid_spacing_m=GRID_SPACING,
        block_size_px=BLOCK,
    )
    if requested is not None and (
        conflict := toolchain_conflict(record.get("_toolchain"), versions)
    ):
        print(conflict)
        return 1

    # Deliberately a persistent, disk-backed directory rather than a temporary
    # one: a regeneration downloads gigabytes, and re-running it after a failure
    # part-way through should reuse what already landed. S108 is about
    # predictable paths in privileged code; this is a developer script.
    work = Path(
        os.environ.get("S1BURSTS_VALIDATION_WORKDIR", "/var/tmp/s1bursts-validation"),  # noqa: S108
    )
    work.mkdir(parents=True, exist_ok=True)

    session = login()
    for scene in scenes:
        print(f"=== {scene.name} ({scene.burst_tag}) ===", flush=True)
        record[scene.name] = compare_scene(scene, work, session)
        block_key = f"block_{BLOCK}x{BLOCK}_{int(BLOCK * GRID_SPACING)}m"
        for polarization, stats in record[scene.name]["gamma0"].items():
            print(f"  {polarization} bias: {stats[block_key]['median_dB_bias']} dB")
        print(
            f"  ANF ratio: {record[scene.name]['anf']['layover_shadow_excluded']['median_ratio']}",
        )

    record["_toolchain"] = versions
    RECORD_PATH.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {RECORD_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
