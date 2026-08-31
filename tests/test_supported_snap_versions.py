"""The supported SNAP versions are stated in three places; keep them agreeing.

:data:`s1bursts.snap_client.SUPPORTED_SNAP_VERSIONS` is the source of truth; the
CI matrix decides what runs, and the badge is what users read.
"""

from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import unquote

from s1bursts.snap_client import SUPPORTED_SNAP_VERSIONS

REPO_ROOT = Path(__file__).parent.parent
README = REPO_ROOT / "README.md"
INTEGRATION_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "integration.yml"

# The badge names majors only; patch pinning is the workflow's business.
_BADGE_PATTERN = re.compile(r"img\.shields\.io/badge/SNAP-(?P<versions>[^-]+)-")
# Each leg of the matrix in the `legs` job.
_MATRIX_PATTERN = re.compile(r'"snap":\s*"(?P<version>\d+\.\d+\.\d+)"')


def test_readme_badge_names_the_supported_majors() -> None:
    match = _BADGE_PATTERN.search(README.read_text())
    assert match is not None, "no SNAP badge found in README.md"

    badged = {part.strip() for part in unquote(match.group("versions")).split("|")}
    expected = {str(major) for major, _ in SUPPORTED_SNAP_VERSIONS}

    assert badged == expected


def test_ci_matrix_runs_every_supported_version() -> None:
    versions = _MATRIX_PATTERN.findall(INTEGRATION_WORKFLOW.read_text())
    assert versions, "no SNAP versions found in the integration workflow matrix"

    matrix = {tuple(int(part) for part in version.split("."))[:2] for version in versions}

    assert matrix == set(SUPPORTED_SNAP_VERSIONS)
