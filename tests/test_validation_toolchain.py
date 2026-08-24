"""Tests for the provenance a cross-validation campaign records."""

from __future__ import annotations

from tests.validation_toolchain import toolchain_conflict

RECORDED = {"s1bursts": "0.5.0", "snap": "12.0.0", "grid_spacing_m": 30}


class TestToolchainConflict:
    """A record states one toolchain for every scene it holds.

    Regenerating a single scene under a changed toolchain would silently
    restate it for the scenes that run did not touch.
    """

    def test_an_unchanged_toolchain_is_no_conflict(self) -> None:
        assert toolchain_conflict(RECORDED, dict(RECORDED)) is None

    def test_a_record_without_a_toolchain_is_no_conflict(self) -> None:
        assert toolchain_conflict(None, dict(RECORDED)) is None
        assert toolchain_conflict({}, dict(RECORDED)) is None

    def test_a_moved_version_names_what_moved(self) -> None:
        message = toolchain_conflict(RECORDED, {**RECORDED, "snap": "13.0.0"})

        assert message is not None
        assert "snap: '12.0.0' -> '13.0.0'" in message
        assert "s1bursts" not in message

    def test_a_campaign_wide_setting_counts_as_a_change(self) -> None:
        """The extras carry the comparison's own parameters, not just versions."""
        message = toolchain_conflict(RECORDED, {**RECORDED, "grid_spacing_m": 20})

        assert message is not None
        assert "grid_spacing_m: 30 -> 20" in message

    def test_an_added_or_dropped_entry_counts_as_a_change(self) -> None:
        assert toolchain_conflict(RECORDED, {**RECORDED, "java": "21"}) is not None
        assert toolchain_conflict({**RECORDED, "java": "21"}, dict(RECORDED)) is not None
