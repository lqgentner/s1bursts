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

from contextlib import suppress
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import pyproj

# Run as a script from anywhere: make the repository root importable so that the
# `tests` package resolves the same way it does under pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import s1bursts
from s1bursts import (
    BurstDatabase,
    BurstProcessor,
    FileManager,
    SearchQuery,
    SNAPConfig,
)
from s1bursts.config import get_settings
from s1bursts.errors import ConfigurationError
from s1bursts.snap_client import _resolve_gpt_path
from s1bursts.snap_graph import BackscatterConfig
from tests.opera_validation import (
    BLOCK,
    GRID_SPACING,
    NOISE_FLOOR_PERCENTILE,
    SCENES,
    Scene,
    agreement,
    percentile_offset_db,
    read_aligned,
)

if TYPE_CHECKING:
    from s1bursts.query import Polarization

RECORD_PATH = Path(__file__).parent / "opera_reference.json"

PER_ACQUISITION_COLLECTION = "OPERA_L2_RTC-S1_V1"
STATIC_COLLECTION = "OPERA_L2_RTC-S1-STATIC_V1"


def _snap_home() -> Path | None:
    """Directory the resolved ``gpt`` lives under, if one resolves at all."""
    with suppress(ConfigurationError, OSError):
        return Path(_resolve_gpt_path(None, get_settings().snap)).parent.parent
    return None


def _snap_version() -> str:
    """SNAP's own version string, or ``"unknown"`` if it cannot be read."""
    snap_home = _snap_home()
    if snap_home is not None:
        with suppress(OSError):
            return (snap_home / "VERSION.txt").read_text().strip()
    return "unknown"


def processing_fingerprint(scene: Scene) -> str:
    """Short hash of everything that determines the raster this script produces.

    Products are cached on disk so that a regeneration interrupted part-way
    through gigabytes of downloads can resume. That cache must not survive a
    change to the processing itself, because the reason to regenerate is
    usually a SNAP upgrade or a graph change -- reusing an old raster would
    record the new toolchain beside measurements from the old one.

    The SNAP version and the graph XML are hashed rather than the package
    version, so that unrelated commits do not needlessly discard hours of work.

    Raises
    ------
    RuntimeError
        If the SNAP version cannot be read. Provenance may degrade to
        ``"unknown"``, but a cache key must not: two different SNAP versions
        would then hash alike and the stale products this key exists to discard
        would be reused.
    """
    snap_version = _snap_version()
    if snap_version == "unknown":
        msg = (
            "cannot read the SNAP version, so cached products cannot be keyed "
            "safely; set SNAP_BIN (or run with `--env-file .env`) before regenerating"
        )
        raise RuntimeError(msg)

    config = BackscatterConfig(grid_spacing=GRID_SPACING)
    material = f"{snap_version}\n{scene.crs}\n{config.to_xml()}"
    return hashlib.sha256(material.encode()).hexdigest()[:12]


def _toolchain() -> dict[str, Any]:
    """Record the versions the numbers below are only meaningful against.

    Provenance is best-effort: a missing version string must degrade to
    ``"unknown"`` rather than abort a regeneration that already cost tens of
    minutes of downloads and GPT runs.
    """
    snap_home = _snap_home()
    snap_version = _snap_version()

    # SNAP bundles its own JRE, and that is the one GPT actually runs on.
    java = shutil.which("java")
    if snap_home is not None and (snap_home / "jre" / "bin" / "java").exists():
        java = str(snap_home / "jre" / "bin" / "java")

    java_version = "unknown"
    if java:
        with suppress(OSError, subprocess.SubprocessError, IndexError):
            result = subprocess.run(  # noqa: S603
                [java, "-version"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            java_version = (result.stderr or result.stdout).splitlines()[0].strip()

    return {
        "s1bursts": getattr(s1bursts, "__version__", "unknown"),
        "snap": snap_version,
        "java": java_version,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "dem": "Copernicus 30m Global DEM (SNAP auto-download)",
        "grid_spacing_m": GRID_SPACING,
        "block_size_px": BLOCK,
    }


def fetch_opera(
    scene: Scene,
    destination: Path,
    earthaccess: Any,
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
    """
    destination.mkdir(parents=True, exist_ok=True)
    wanted = {
        PER_ACQUISITION_COLLECTION: (
            f"*{scene.burst_tag}_{scene.gamma0_date.replace('-', '')}*",
            (*(f"_{pol}.tif" for pol in scene.polarizations), "_mask.tif"),
        ),
        STATIC_COLLECTION: (
            f"*{scene.burst_tag}*",
            ("_rtc_anf_gamma0_to_beta0.tif", "_mask.tif"),
        ),
    }

    found: dict[str, Path] = {}
    static_sources: list[str] = []
    for collection, (pattern, suffixes) in wanted.items():
        granules = earthaccess.search_data(
            short_name=collection,
            granule_name=pattern,
            count=5,
        )
        if len(granules) != 1:
            msg = (
                f"expected exactly one {collection} granule matching {pattern}, "
                f"found {len(granules)}; disambiguate the search before regenerating"
            )
            raise RuntimeError(msg)
        granule = granules[0]

        if collection is STATIC_COLLECTION:
            # The static layers encode the orbit of the one acquisition they were
            # derived from, so SNAP must process that same acquisition. Verify it
            # here rather than trusting `Scene.static_date` to have stayed correct
            # across an OPERA reprocessing.
            static_sources = list(granule["umm"].get("InputGranules", []))
            stamp = scene.static_date.replace("-", "")
            if not any(stamp in source for source in static_sources):
                msg = (
                    f"{scene.name}: static granule was derived from {static_sources}, "
                    f"which does not include the {scene.static_date} acquisition named "
                    f"in Scene.static_date; the ANF comparison would be invalid"
                )
                raise RuntimeError(msg)

        urls = [link for link in granule.data_links() if link.endswith(suffixes)]
        missing = [url for url in urls if not (destination / url.rsplit("/", 1)[-1]).exists()]
        if missing:
            earthaccess.download(missing, str(destination))

        kind = "static" if collection is STATIC_COLLECTION else "acq"
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
    fingerprint = processing_fingerprint(scene)
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
    )[0:1]
    if len(bursts) != 1:
        msg = f"expected exactly one burst for {scene.name} {polarization} {acquisition}"
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
        config=BackscatterConfig(grid_spacing=GRID_SPACING),
        overwrite=True,
    )
    if result.n_failed:
        msg = f"processing failed: {[str(item.error) for item in result.failed_items]}"
        raise RuntimeError(msg)

    # `get_product_path` may hand back a UPath for remote stores; these
    # products are always local, so a plain string round-trip is safe.
    shutil.copyfile(str(manager.get_product_path(result.products[0])), product)
    return product


def compare_scene(scene: Scene, work: Path, earthaccess: Any) -> dict:
    """Run both comparisons for one scene and return its record entry."""
    files, static_sources = fetch_opera(scene, work / "opera", earthaccess)
    block_key = f"block_{BLOCK}x{BLOCK}_{int(BLOCK * GRID_SPACING)}m"

    entry: dict[str, Any] = {
        "burst_tag": scene.burst_tag,
        "description": scene.description,
        "crs": scene.crs,
        "gamma0_acquisition": scene.gamma0_date,
        "static_source_acquisition": scene.static_date,
        "opera_granules": sorted({path.name for path in files.values()}),
        "opera_static_source_granules": static_sources,
        "processing_fingerprint": processing_fingerprint(scene),
        "gamma0": {},
    }

    acquisition_mask = files["acq:mask.tif"]
    for polarization in scene.polarizations:
        # A scene declares the polarizations it is meant to validate. A missing
        # asset is a broken campaign, not a comparison to quietly skip.
        opera_tif = files[f"acq:{polarization}.tif"]

        ours_path = process(scene, scene.gamma0_date, polarization, work)
        ours, theirs, alignment = read_aligned(ours_path, opera_tif, "Gamma0")
        _, mask, _ = read_aligned(ours_path, acquisition_mask, "Gamma0")
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
    ours, theirs, alignment = read_aligned(
        ours_path,
        files["static:rtc_anf_gamma0_to_beta0.tif"],
        "simulatedImage",
    )
    _, mask, _ = read_aligned(ours_path, files["static:mask.tif"], "simulatedImage")
    clear = (mask == 0) & np.isfinite(ours) & np.isfinite(theirs)

    entry["anf"] = {
        "alignment": alignment,
        "layover_shadow_excluded": agreement(ours, theirs, clear, k=1),
    }
    return entry


def main() -> int:
    """Regenerate the record, for one named scene or all of them."""
    os.environ.setdefault("EARTHDATA_USERNAME", os.environ.get("EOSDIS_USERNAME", ""))
    os.environ.setdefault("EARTHDATA_PASSWORD", os.environ.get("EOSDIS_PASSWORD", ""))
    # Imported here, not at module scope: `earthaccess` lives in the optional
    # `validation` dependency group, and tests/test_opera_validation.py imports
    # this module's helpers without it installed.
    import earthaccess  # noqa: PLC0415

    earthaccess.login(strategy="environment")

    requested = sys.argv[1] if len(sys.argv) > 1 else None
    scenes = [s for s in SCENES if requested is None or s.name == requested]
    if not scenes:
        print(f"unknown scene {requested!r}; known: {[s.name for s in SCENES]}")
        return 1

    record: dict[str, Any] = {}
    if RECORD_PATH.exists():
        record = json.loads(RECORD_PATH.read_text())

    # Deliberately a persistent, disk-backed directory rather than a temporary
    # one: a regeneration downloads gigabytes, and re-running it after a failure
    # part-way through should reuse what already landed. S108 is about
    # predictable paths in privileged code; this is a developer script.
    work = Path(
        os.environ.get("S1BURSTS_VALIDATION_WORKDIR", "/var/tmp/s1bursts-validation"),  # noqa: S108
    )
    work.mkdir(parents=True, exist_ok=True)

    for scene in scenes:
        print(f"=== {scene.name} ({scene.burst_tag}) ===", flush=True)
        record[scene.name] = compare_scene(scene, work, earthaccess)
        block_key = f"block_{BLOCK}x{BLOCK}_{int(BLOCK * GRID_SPACING)}m"
        for polarization, stats in record[scene.name]["gamma0"].items():
            print(f"  {polarization} bias: {stats[block_key]['median_dB_bias']} dB")
        print(
            f"  ANF ratio: {record[scene.name]['anf']['layover_shadow_excluded']['median_ratio']}",
        )

    record["_toolchain"] = _toolchain()
    RECORD_PATH.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {RECORD_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
