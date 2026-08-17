"""Tests for s1bursts.config settings loading."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError
import pytest

from s1bursts.config import get_settings


class TestGetSettings:
    def test_reads_environment_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CDSE_USERNAME", "user@example.com")
        monkeypatch.setenv("CDSE_PASSWORD", "secret")
        monkeypatch.setenv("SNAP_MAX_WORKERS", "4")
        monkeypatch.setenv("SNAP_MAX_RAM_GB", "16")

        settings = get_settings()

        assert settings.cdse.username == "user@example.com"
        assert settings.cdse.password is not None
        assert settings.cdse.password.get_secret_value() == "secret"
        assert settings.snap.max_workers == 4
        assert settings.snap.max_ram_gb == 16.0

    def test_environment_changes_are_picked_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Settings must be re-read on each call (late load_dotenv support)."""
        monkeypatch.setenv("SNAP_MAX_WORKERS", "1")
        first = get_settings()
        monkeypatch.setenv("SNAP_MAX_WORKERS", "8")
        second = get_settings()

        assert first.snap.max_workers == 1
        assert second.snap.max_workers == 8

    def test_ignores_dotenv_file_in_working_directory(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Loading a ``.env`` belongs to the application, not to this package."""
        monkeypatch.delenv("CDSE_USERNAME", raising=False)
        (tmp_path / ".env").write_text("CDSE_USERNAME=from-dotenv\n")
        monkeypatch.chdir(tmp_path)

        assert get_settings().cdse.username is None

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in ("SNAP_BIN", "SNAP_MAX_WORKERS", "SNAP_MAX_RAM_GB"):
            monkeypatch.delenv(var, raising=False)

        snap = get_settings().snap

        assert snap.bin is None
        assert snap.max_workers == 1
        assert snap.max_ram_gb == 4.0
        assert snap.auxdata_path == Path.home() / ".snap/auxdata"


class TestSnapSettingsValidation:
    """Resource budgets from the environment must be rejected at load time.

    An invalid value here would otherwise reach GPT as a nonsensical JVM flag.
    """

    @pytest.mark.parametrize(
        ("variable", "value"),
        [
            ("SNAP_MAX_WORKERS", "0"),
            ("SNAP_MAX_WORKERS", "-4"),
            ("SNAP_MAX_RAM_GB", "0"),
            ("SNAP_MAX_RAM_GB", "-8"),
            ("SNAP_CACHE_RATIO", "0"),
            ("SNAP_CACHE_RATIO", "-0.5"),
            ("SNAP_CACHE_RATIO", "1.5"),
            ("SNAP_THREADS_PER_JOB", "0"),
        ],
    )
    def test_rejects_out_of_range_values(
        self,
        monkeypatch: pytest.MonkeyPatch,
        variable: str,
        value: str,
    ) -> None:
        monkeypatch.setenv(variable, value)

        with pytest.raises(ValidationError):
            get_settings()

    @pytest.mark.parametrize("ratio", ["0.01", "0.7", "1"])
    def test_accepts_the_full_cache_ratio_range(
        self,
        monkeypatch: pytest.MonkeyPatch,
        ratio: str,
    ) -> None:
        monkeypatch.setenv("SNAP_CACHE_RATIO", ratio)

        assert get_settings().snap.cache_ratio == float(ratio)

    def test_accepts_a_fractional_ram_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SNAP_MAX_RAM_GB", "6.5")

        assert get_settings().snap.max_ram_gb == 6.5
