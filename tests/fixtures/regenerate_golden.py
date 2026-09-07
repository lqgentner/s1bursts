"""Regenerate the committed golden record for the offline burst extraction.

Run this only when the extractor's output changes deliberately and the change
has been checked against the CDSE product contract::

    uv run python tests/fixtures/regenerate_golden.py

Review the resulting diff to ``golden_burst.json`` before committing it: every
changed field is a change in what users receive.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

from upath import UPath

import s1bursts.burst_extractor as extractor

# Run as a script from anywhere: make the repository root importable so that the
# `tests` package resolves the same way it does under pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.fixtures.safe_product import ProductSpec, build_product
from tests.safe_conformance import check_burst_safe, digest

GOLDEN_PATH = Path(__file__).parent / "golden_burst.json"

# Must match TestGoldenRecord.test_matches_the_golden_record.
SPEC = ProductSpec(with_rfi=True)
BURST_INDEX = 2


def main() -> int:
    """Write the golden record and report the outcome."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        product = build_product(tmp_dir, SPEC)

        # Stubs replace the S3 lookups.
        # pyrefly: ignore [bad-assignment]
        extractor._lookup_s3_path = lambda _name: str(product.root)
        # Stubs replace the S3 lookups.
        # pyrefly: ignore [bad-assignment]
        extractor._make_s3_upath = lambda path, _settings: UPath(path)

        zip_path = extractor.extract_burst_from_s3(
            tmp_dir / "burst.zip",
            parent_product_name=product.name,
            polarization=SPEC.polarization,
            sub_swath=SPEC.sub_swath,
            burst_id=SPEC.burst_ids[BURST_INDEX],
        )

        violations = check_burst_safe(Path(str(zip_path)), sub_swath=SPEC.sub_swath)
        if violations:
            print("Refusing to pin a non-conformant extraction:")
            for violation in violations:
                print(f"  - {violation}")
            return 1

        GOLDEN_PATH.write_text(
            json.dumps(digest(Path(str(zip_path))), indent=2, sort_keys=True) + "\n"
        )

    print(f"Wrote {GOLDEN_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
