"""The supported Python versions are stated in several places; keep them agreeing.

The ``Programming Language`` classifiers in ``pyproject.toml`` are the source of
truth, and PyPI and the README badge are built from them. The CI matrix decides
what runs, and the oldest supported version is the floor that ``requires-python``,
the lowest-bound job and Pyrefly are held to. The development interpreter pinned in
``.python-version`` must be one of them.
"""

from __future__ import annotations

from pathlib import Path
import re
import tomllib

REPO_ROOT = Path(__file__).parent.parent
PYPROJECT = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
CI_WORKFLOW = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
SNAP_DOCKERFILE = (REPO_ROOT / "containers" / "Dockerfile.snap").read_text()
PINNED_PYTHON = (REPO_ROOT / ".python-version").read_text()

_CLASSIFIER_PATTERN = re.compile(r"Programming Language :: Python :: (\d+\.\d+)")
_MATRIX_PATTERN = re.compile(r"^\s*python-version:\s*\[(?P<versions>[^\]]*)\]", re.MULTILINE)
_LOWEST_PATTERN = re.compile(r"--resolution lowest-direct --python (\d+\.\d+)")
_DOCKERFILE_PATTERN = re.compile(r"^ARG PYTHON_VERSION=(\d+\.\d+)$", re.MULTILINE)

type Version = tuple[int, int]


def _parse(version: str) -> Version:
    """Parse a ``major.minor`` string."""
    major, minor = version.strip().strip("\"'").split(".")
    return int(major), int(minor)


def _supported() -> list[Version]:
    """Return the classified Python versions, oldest first."""
    classifiers: list[str] = PYPROJECT["project"]["classifiers"]
    versions = sorted(
        _parse(match.group(1))
        for classifier in classifiers
        if (match := _CLASSIFIER_PATTERN.fullmatch(classifier))
    )
    assert versions, "no Python version classifiers found in pyproject.toml"
    return versions


def test_classifiers_are_contiguous() -> None:
    supported = _supported()
    (major, oldest), (_, newest) = supported[0], supported[-1]

    assert supported == [(major, minor) for minor in range(oldest, newest + 1)]


def test_ci_matrix_tests_every_supported_version() -> None:
    match = _MATRIX_PATTERN.search(CI_WORKFLOW)
    assert match is not None, "no python-version matrix found in the CI workflow"

    matrix = {_parse(version) for version in match.group("versions").split(",")}

    assert matrix == set(_supported())


def test_requires_python_is_the_oldest_supported_version() -> None:
    major, minor = _supported()[0]

    assert PYPROJECT["project"]["requires-python"] == f">={major}.{minor}"


def test_lowest_bound_job_runs_the_oldest_supported_version() -> None:
    versions = _LOWEST_PATTERN.findall(CI_WORKFLOW)
    assert versions, "no lowest-direct sync found in the CI workflow"

    assert {_parse(version) for version in versions} == {_supported()[0]}


def test_pyrefly_checks_against_the_oldest_supported_version() -> None:
    assert _parse(PYPROJECT["tool"]["pyrefly"]["python-version"]) == _supported()[0]


def test_snap_container_python_is_supported() -> None:
    match = _DOCKERFILE_PATTERN.search(SNAP_DOCKERFILE)
    assert match is not None, "no PYTHON_VERSION build argument in Dockerfile.snap"

    assert _parse(match.group(1)) in _supported()


def test_pinned_development_python_is_supported() -> None:
    assert _parse(PINNED_PYTHON) in _supported()
