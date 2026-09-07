"""Regenerate the committed InSAR coherence cross-validation record.

Run this only when the comparison against HyP3 deliberately changes -- a SNAP
upgrade, a graph change, a new scene -- and review the resulting diff to
``insar_reference.json`` before committing: every moved number is a change in
what users receive::

    uv sync --group validation
    uv run --env-file .env python tests/fixtures/regenerate_insar_reference.py --submit
    uv run --env-file .env python tests/fixtures/regenerate_insar_reference.py mojave-iw1

Needs CDSE credentials, Earthdata credentials (which are also the HyP3 login),
and a working SNAP install. Each scene downloads two bursts and runs SNAP GPT,
which takes a few minutes.

The reference itself is processed on demand by ASF, so unlike an archived
product it has to be ordered: ``--submit`` allows this script to queue the HyP3
jobs it cannot find, spending the account's credits and then waiting the
quarter of an hour or so each takes. Without it, the script uses jobs already
completed under the account and stops if there are none, so that no run of the
test suite can order processing by itself. HyP3 products expire after a couple
of weeks, at which point the same job has to be ordered again.

The record captures the toolchain that produced it, because the numbers are
only meaningful against a stated SNAP, Java and HyP3 product version.
"""

from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import TYPE_CHECKING, Any, cast
import zipfile

from freezebase.download import HTTPDownloader
import pyproj
import rasterio
import rasterio.crs

# Run as a script from anywhere: make the repository root importable so that the
# `tests` package resolves the same way it does under pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from s1bursts import (
    BurstDatabase,
    BurstProcessor,
    CoherenceConfig,
    FileManager,
    SearchQuery,
    SNAPConfig,
)
from s1bursts.burst import BurstPair, BurstPairCollection
from tests.insar_validation import (
    BLOCK,
    GRID_SPACING,
    LOOKS,
    SCENES,
    Scene,
    agreement,
    alignment_scan,
    quantile_map,
    reference_spread,
    valid_mask,
)
from tests.raster_comparison import read_aligned
from tests.validation_toolchain import processing_fingerprint, toolchain, toolchain_conflict

if TYPE_CHECKING:
    from hyp3_sdk import HyP3, Job

RECORD_PATH = Path(__file__).parent / "insar_reference.json"

JOB_TYPE = "INSAR_ISCE_BURST"

# A burst InSAR job runs in roughly a quarter of an hour, but the queue in front
# of it does not have a length. Two hours is long enough that a busy afternoon
# does not abandon a job that is about to finish.
WATCH_TIMEOUT_S = 7200


class NoReferenceJobError(RuntimeError):
    """No HyP3 product exists for a scene, and this run may not order one."""


def processing_config() -> CoherenceConfig:
    """Graph configuration every scene is processed with.

    The window stays at the package default: what is under validation is the
    coherence users get, not a configuration chosen to flatter the comparison.
    Only the posting is set, to HyP3's, so that the two grids coincide.
    """
    return CoherenceConfig(grid_spacing=GRID_SPACING)


def scene_fingerprint(scene: Scene) -> str:
    """Cache key for this scene's processed product; see `processing_fingerprint`."""
    return processing_fingerprint(processing_config(), scene.crs)


def burst_granule(scene: Scene, acquisition: str) -> str:
    """Look up the ASF burst granule HyP3 processes for one acquisition.

    HyP3 takes granule names, and the name carries a four-digit identifier of
    the parent SLC that cannot be derived from the burst alone, so it has to be
    searched for.

    Raises
    ------
    RuntimeError
        If the search does not match exactly one burst. Two matches mean the
        window caught a neighbouring acquisition, and the wrong pair would then
        be ordered.
    """
    # Imported here, not at module scope: `asf_search` lives in the optional
    # `validation` dependency group, and tests/test_insar_validation.py imports
    # this module's helpers without it installed.
    import asf_search  # noqa: PLC0415

    start, end = scene.search_window(acquisition)
    results = asf_search.search(
        dataset="SLC-BURST",
        fullBurstID=scene.full_burst_id,
        polarization=scene.polarization,
        start=start,
        end=end,
    )
    granules = sorted(product.properties["sceneName"] for product in results)
    if len(granules) != 1:
        msg = (
            f"expected exactly one {scene.full_burst_id} {scene.polarization} burst on "
            f"{acquisition}, found {granules}; narrow `Scene.search_window` before regenerating"
        )
        raise RuntimeError(msg)
    return granules[0]


def _job_matches(job: Job, granules: tuple[str, str]) -> bool:
    """Whether a job named after a scene was in fact run on that scene.

    A job name is not unique and carries no processing options, so the
    parameters are what decide: a job submitted before a scene's dates or looks
    changed keeps the name it was given.
    """
    parameters = job.job_parameters or {}
    return (
        tuple(parameters.get("granules", ())) == granules
        and parameters.get("looks") == LOOKS
        and not parameters.get("apply_water_mask", False)
    )


def queue_job(
    hyp3: HyP3,
    scene: Scene,
    granules: tuple[str, str],
    *,
    allow_submit: bool,
) -> Job:
    """Return the HyP3 job for one scene, ordering one if allowed.

    Does not wait for it: a job that is queued but unfinished comes back as it
    is, so that a campaign can put every scene into HyP3's queue before waiting
    on any of them.

    Raises
    ------
    NoReferenceJobError
        If no usable product exists and ``allow_submit`` is false.
    """
    candidates = [
        job
        for job in hyp3.find_jobs(name=scene.job_name).jobs
        if job.job_type == JOB_TYPE and _job_matches(job, granules)
    ]

    ready = [job for job in candidates if job.succeeded() and not job.expired()]
    waiting = [job for job in candidates if job.pending() or job.running()]
    if ready or waiting:
        return max(ready or waiting, key=lambda job: job.request_time)

    if not allow_submit:
        msg = (
            f"{scene.name}: no completed HyP3 job named {scene.job_name!r} for "
            f"{granules[0]} / {granules[1]}; order one by running "
            "tests/fixtures/regenerate_insar_reference.py --submit, which spends HyP3 credits"
        )
        raise NoReferenceJobError(msg)

    print(f"  submitting {JOB_TYPE} for {granules[0]} / {granules[1]}", flush=True)
    batch = hyp3.submit_insar_isce_burst_job(
        granules[0],
        granules[1],
        name=scene.job_name,
        looks=LOOKS,
    )
    return batch.jobs[0]


def reference_job(
    hyp3: HyP3,
    scene: Scene,
    granules: tuple[str, str],
    *,
    allow_submit: bool,
) -> Job:
    """Wait for the HyP3 product of one scene, ordering it if allowed.

    Raises
    ------
    RuntimeError
        If the job did not succeed.
    """
    job = queue_job(hyp3, scene, granules, allow_submit=allow_submit)
    if job.succeeded():
        return job

    # `watch` is typed to return a Batch or a Job, whichever it was given.
    job = cast("Job", hyp3.watch(job, timeout=WATCH_TIMEOUT_S))
    if not job.succeeded():
        msg = f"{scene.name}: HyP3 job {job.job_id} finished as {job.status_code}"
        raise RuntimeError(msg)
    return job


# The archive's README names the software that produced the product, in a
# sentence rendered across a line break:
#   "... using the hyp3_isce2 plugin version 2.1.0 running ISCE release 2.6.3."
_VERSIONS = re.compile(
    r"using the (?P<plugin>\S+) plugin version\s+(?P<plugin_version>\S+)\s+"
    r"running (?P<processor>\S+) release (?P<processor_version>[^.\s]+(?:\.[^.\s]+)*)\.",
)


def _reference_versions(readme: Path) -> dict[str, str]:
    """Read the processor versions the reference was produced with.

    A HyP3 product is processed on demand, so an expired one is reordered
    against whatever ASF is running by then. The numbers in this record are
    only comparable across regenerations if what produced the reference is
    recorded beside them, and neither the job metadata nor the parameter file
    carries it -- the README does.

    Returns
    -------
    dict of str to str
        The plugin and processor names and versions, or ``{}`` if the README
        no longer spells them the way this expects. Provenance is best-effort:
        a template change must not abort a campaign that has already spent
        credits and downloads.
    """
    match = _VERSIONS.search(readme.read_text(errors="replace"))
    return match.groupdict() if match else {}


def fetch_coherence(job: Job, destination: Path) -> tuple[Path, dict[str, str]]:
    """Download a HyP3 product and extract what the comparison reads from it.

    Only the coherence layer and the README are kept: the archive holds a dozen
    layers this comparison never reads, and the work directory persists between
    runs.

    Returns
    -------
    tuple of (pathlib.Path, dict)
        The coherence GeoTIFF, and the versions of the software that produced
        it.

    Raises
    ------
    RuntimeError
        If the job does not carry exactly one product archive, or the archive
        does not hold the expected coherence layer.
    """
    destination.mkdir(parents=True, exist_ok=True)
    archives = [
        item["filename"] for item in (job.files or []) if item["filename"].endswith(".zip")
    ]
    if len(archives) != 1:
        msg = f"expected exactly one product archive on job {job.job_id}, found {archives}"
        raise RuntimeError(msg)

    product = archives[0].removesuffix(".zip")
    member = f"{product}/{product}_corr.tif"
    coherence = destination / member

    def readme() -> Path | None:
        """Find the extracted README, whatever ASF happens to call it."""
        return next(iter(sorted((destination / product).glob("*README.md.txt"))), None)

    extracted = readme()
    if coherence.exists() and extracted is not None:
        return coherence, _reference_versions(extracted)

    # Fetched with this package's downloader rather than `job.download_files()`,
    # whose progress-bar wrapper leaves the file object to the garbage collector;
    # under Python 3.14 that surfaces as an unraisable-exception warning, which
    # the test suite treats as a failure.
    [archive_entry] = [item for item in job.files or [] if item["filename"] == archives[0]]
    with HTTPDownloader(progress=False) as download:
        downloaded = Path(
            download(url=archive_entry["url"], save_dir=destination, filename=archives[0]),
        )
    with zipfile.ZipFile(downloaded) as archive:
        names = archive.namelist()
        if member not in names:
            msg = f"{downloaded.name} does not hold {member}"
            raise RuntimeError(msg)
        archive.extract(member, destination)
        # Matched by suffix rather than spelled out: the version provenance
        # should not be lost to a rename, and it is not worth failing a
        # campaign over.
        for name in [name for name in names if name.endswith("README.md.txt")][:1]:
            archive.extract(name, destination)
    downloaded.unlink()

    extracted = readme()
    return coherence, _reference_versions(extracted) if extracted is not None else {}


def process(scene: Scene, work: Path) -> Path:
    """Produce this package's coherence for one burst pair at HyP3's posting.

    The cached product is keyed by :func:`processing_fingerprint`, so a SNAP or
    graph change starts from a clean raster instead of silently reusing the
    previous one.
    """
    fingerprint = scene_fingerprint(scene)
    product = (
        work / f"{scene.name}-{scene.reference_date}-{scene.secondary_date}-{fingerprint}.tif"
    )
    if product.exists():
        return product

    manager = FileManager(base_dir=work / "s1bursts")
    database = BurstDatabase(file_manager=manager)

    def search(acquisition: str) -> Any:
        found = database.search(
            SearchQuery(
                time_range=scene.search_window(acquisition),
                burst_id=scene.burst_id,
                sub_swath=scene.sub_swath,
                polarization=scene.polarization,
            ),
        )
        if len(found) != 1:
            # As in `burst_granule`: more than one match means the window caught
            # a neighbouring acquisition, and the wrong pair would be processed.
            msg = (
                f"expected exactly one burst for {scene.name} on {acquisition}, "
                f"found {len(found)}; narrow `Scene.search_window` before regenerating"
            )
            raise RuntimeError(msg)
        return found

    reference, secondary = search(scene.reference_date), search(scene.secondary_date)
    summary = database.download(reference + secondary, endpoint="odata")
    if summary.n_failed:
        msg = f"download failed: {[str(item.error) for item in summary.failed_items]}"
        raise RuntimeError(msg)

    processor = BurstProcessor(
        manager,
        SNAPConfig(max_workers=1, max_ram_gb=6),
        show_progress=False,
        continue_on_error=False,
    )
    result = processor.create_coherences(
        BurstPairCollection.from_pairs([BurstPair(primary=reference[0], secondary=secondary[0])]),
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


def _check_projection(coherence: Path, scene: Scene) -> None:
    """Refuse a HyP3 product that was geocoded into another UTM zone.

    HyP3 picks the zone from the upper-left corner of the burst it processed,
    so a scene near a zone boundary can land somewhere other than the CRS the
    scene declares -- and this package would then be processed into a grid that
    cannot be compared against it.

    Raises
    ------
    RuntimeError
        If the product's CRS is not the scene's.
    """
    with rasterio.open(coherence) as src:
        if src.crs != rasterio.crs.CRS.from_string(scene.crs):
            msg = (
                f"{scene.name}: HyP3 geocoded this burst into {src.crs}, but Scene.crs "
                f"says {scene.crs}; set Scene.crs to what HyP3 used"
            )
            raise RuntimeError(msg)


def compare_scene(
    scene: Scene,
    work: Path,
    hyp3: HyP3,
    *,
    allow_submit: bool = False,
) -> dict[str, Any]:
    """Run the comparison for one scene and return its record entry."""
    granules = (
        burst_granule(scene, scene.reference_date),
        burst_granule(scene, scene.secondary_date),
    )
    job = reference_job(hyp3, scene, granules, allow_submit=allow_submit)
    coherence, versions = fetch_coherence(job, work / "hyp3")
    _check_projection(coherence, scene)

    ours_path = process(scene, work)
    ours, theirs, _, alignment = read_aligned(ours_path, coherence, "coherence")
    valid = valid_mask(ours, theirs)

    baseline = date.fromisoformat(scene.secondary_date) - date.fromisoformat(scene.reference_date)
    return {
        "full_burst_id": scene.full_burst_id,
        "description": scene.description,
        "crs": scene.crs,
        "polarization": scene.polarization,
        "reference_acquisition": scene.reference_date,
        "secondary_acquisition": scene.secondary_date,
        "temporal_baseline_days": baseline.days,
        "asf_granules": list(granules),
        "hyp3_job": {
            "job_id": job.job_id,
            "name": job.name,
            "product": coherence.parent.name,
            "looks": LOOKS,
            "request_time": job.request_time.isoformat(timespec="seconds"),
            "credit_cost": job.credit_cost,
            # What ASF ran to produce this reference. A reordered product may
            # come from newer software, which moves the numbers below without
            # anything in this package having changed.
            **versions,
        },
        "processing_fingerprint": scene_fingerprint(scene),
        "alignment": alignment,
        "compared_px": int(valid.sum()),
        "reference_spread": reference_spread(theirs, valid),
        "per_pixel": agreement(ours, theirs, valid),
        f"block_{BLOCK}x{BLOCK}_{int(BLOCK * GRID_SPACING)}m": agreement(
            ours,
            theirs,
            valid,
            k=BLOCK,
        ),
        "quantiles": quantile_map(ours, theirs, valid),
        "geolocation": alignment_scan(ours, theirs, valid),
    }


def login() -> HyP3:
    """Open a HyP3 session with the Earthdata credentials in the environment.

    Raises
    ------
    RuntimeError
        If the credentials are not set.
    """
    # Imported here, not at module scope: `hyp3_sdk` lives in the optional
    # `validation` dependency group.
    from hyp3_sdk import HyP3  # noqa: PLC0415

    username = os.environ.get("EOSDIS_USERNAME")
    password = os.environ.get("EOSDIS_PASSWORD")
    if not (username and password):
        msg = "set EOSDIS_USERNAME and EOSDIS_PASSWORD (or run with `--env-file .env`)"
        raise RuntimeError(msg)
    return HyP3(username=username, password=password)


def main() -> int:
    """Regenerate the record, for one named scene or all of them."""
    arguments = sys.argv[1:]
    allow_submit = "--submit" in arguments
    requested = next((argument for argument in arguments if not argument.startswith("-")), None)

    scenes = [scene for scene in SCENES if requested is None or scene.name == requested]
    if not scenes:
        print(f"unknown scene {requested!r}; known: {[scene.name for scene in SCENES]}")
        return 1

    record: dict[str, Any] = {}
    if RECORD_PATH.exists():
        record = json.loads(RECORD_PATH.read_text())

    # Measured before the campaign rather than after it, so that a run which
    # cannot honestly write this provenance stops before ordering any HyP3 job.
    versions = toolchain(
        dem="Copernicus 30m Global DEM (SNAP auto-download)",
        grid_spacing_m=GRID_SPACING,
        block_size_px=BLOCK,
        hyp3_job_type=JOB_TYPE,
        hyp3_looks=LOOKS,
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

    hyp3 = login()

    # Put every scene into HyP3's queue before waiting on any of them: the jobs
    # run in parallel there, and ordering them one finished scene at a time
    # would add a full processing round to the campaign for each extra scene.
    for scene in scenes:
        queue_job(
            hyp3,
            scene,
            (
                burst_granule(scene, scene.reference_date),
                burst_granule(scene, scene.secondary_date),
            ),
            allow_submit=allow_submit,
        )

    block_key = f"block_{BLOCK}x{BLOCK}_{int(BLOCK * GRID_SPACING)}m"
    for scene in scenes:
        print(f"=== {scene.name} ({scene.full_burst_id}) ===", flush=True)
        record[scene.name] = compare_scene(scene, work, hyp3, allow_submit=allow_submit)
        entry = record[scene.name]
        blocked = entry[block_key]
        print(f"  reference spread: {entry['reference_spread']}")
        print(f"  rank correlation: {blocked['spearman']}")
        print(f"  median offset:    {blocked['median_diff_hyp3_minus_ours']}")
        print(
            f"  shift (m):        {entry['geolocation']['subpixel_shift_m']} "
            f"(peak contrast {entry['geolocation']['peak_contrast']})",
        )

    record["_toolchain"] = versions
    RECORD_PATH.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {RECORD_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
