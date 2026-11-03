"""``CITATION.cff`` names the newest release; keep it agreeing with the changelog.

Its ``version`` and ``date-released`` cannot come from the git tag like the package
version does, so they are bumped by hand in each release PR.
"""

from __future__ import annotations

from pathlib import Path
import re

REPO_ROOT = Path(__file__).parent.parent
CITATION = (REPO_ROOT / "CITATION.cff").read_text()
CHANGELOG = (REPO_ROOT / "CHANGELOG.md").read_text()

_RELEASE_PATTERN = re.compile(
    r"^## \[(?P<version>\d+\.\d+\.\d+)\] - (?P<date>\d{4}-\d{2}-\d{2})$", re.MULTILINE
)


def _field(name: str) -> str:
    """Return the value of a top-level scalar field of ``CITATION.cff``."""
    match = re.search(rf"^{name}: (.+)$", CITATION, re.MULTILINE)
    assert match is not None, f"CITATION.cff has no {name!r}"
    return match.group(1).strip()


def test_citation_names_the_newest_release() -> None:
    newest = _RELEASE_PATTERN.search(CHANGELOG)
    assert newest is not None, "no released version found in CHANGELOG.md"

    assert _field("version") == newest.group("version")
    assert _field("date-released") == newest.group("date")


def test_abstract_is_the_readme_introduction() -> None:
    readme = (REPO_ROOT / "README.md").read_text()
    introduction = readme.split("</div>", 1)[1].split("\n## ", 1)[0]
    abstract = re.search(r"^abstract: >-\n((?:  .*\n|\n)+)", CITATION, re.MULTILINE)
    assert abstract is not None, "CITATION.cff has no folded 'abstract'"

    def words(text: str) -> list[str]:
        return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text).split()

    assert words(abstract.group(1)) == words(introduction)


def test_citation_cites_the_concept_doi() -> None:
    doi = _field("doi")

    assert _field("url") == f"https://doi.org/{doi}"
    assert f"https://doi.org/{doi}" in (REPO_ROOT / "README.md").read_text()
