"""Tests for s1bursts.config settings loading."""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import ValidationError
import pytest

from s1bursts.config import get_settings, warn_if_tmpdir_unusable
from s1bursts.errors import ConfigurationError


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
        assert snap.max_ram_gb == 8.0
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

        with pytest.raises(ConfigurationError, match=variable) as excinfo:
            get_settings()

        # Previously pydantic's ValidationError escaped unwrapped.
        assert isinstance(excinfo.value.__cause__, ValidationError)

    def test_names_every_invalid_variable_without_its_value(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SNAP_MAX_WORKERS", "0")
        monkeypatch.setenv("SNAP_CACHE_RATIO", "1.5")

        with pytest.raises(ConfigurationError) as excinfo:
            get_settings()

        message = str(excinfo.value)
        assert "SNAP_MAX_WORKERS" in message
        assert "SNAP_CACHE_RATIO" in message
        assert "1.5" not in message

    def test_names_a_variable_that_does_not_parse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SNAP_MAX_RAM_GB", "lots")

        with pytest.raises(ConfigurationError, match="SNAP_MAX_RAM_GB: Input should be a valid"):
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

    @pytest.mark.parametrize(
        ("variable", "attribute", "default"),
        [
            ("SNAP_MAX_WORKERS", "max_workers", 1),
            ("SNAP_MAX_RAM_GB", "max_ram_gb", 8.0),
            ("SNAP_CACHE_RATIO", "cache_ratio", 0.70),
        ],
    )
    def test_empty_values_fall_back_to_the_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        variable: str,
        attribute: str,
        default: float,
    ) -> None:
        """An unconfigured GitHub Actions secret interpolates as an empty string.

        These fields are typed, so rejecting one would fail every caller of
        ``get_settings()``, not only the ones that go on to run SNAP.
        """
        monkeypatch.setenv(variable, "")

        assert getattr(get_settings().snap, attribute) == default

    def test_an_empty_bin_is_unset_rather_than_the_working_directory(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``Path("")`` is ``Path(".")``, which would send `gpt` resolution local."""
        monkeypatch.setenv("SNAP_BIN", "")

        assert get_settings().snap.bin is None


class TestTmpdirWarning:
    """A TMPDIR that cannot be written to is a silent fallback, not an error.

    ``tempfile`` skips it and lands on ``/tmp``, which is often a tmpfs — the
    thing an explicit TMPDIR is usually set to avoid.
    """

    def _warnings(self, caplog: pytest.LogCaptureFixture) -> list[str]:
        return [r.getMessage() for r in caplog.records if "TMPDIR" in r.getMessage()]

    def test_warns_when_tmpdir_does_not_exist(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv("TMPDIR", str(tmp_path / "absent"))

        with caplog.at_level(logging.WARNING, logger="s1bursts.config"):
            warn_if_tmpdir_unusable()

        assert self._warnings(caplog)

    def test_silent_when_tmpdir_is_usable(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv("TMPDIR", str(tmp_path))

        with caplog.at_level(logging.WARNING, logger="s1bursts.config"):
            warn_if_tmpdir_unusable()

        assert not self._warnings(caplog)

    def test_silent_when_tmpdir_is_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An unset TMPDIR is the normal case, not a misconfiguration."""
        monkeypatch.delenv("TMPDIR", raising=False)

        with caplog.at_level(logging.WARNING, logger="s1bursts.config"):
            warn_if_tmpdir_unusable()

        assert not self._warnings(caplog)

    def test_does_not_create_the_directory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """TMPDIR belongs to the environment; this package only reports on it."""
        absent = tmp_path / "absent"
        monkeypatch.setenv("TMPDIR", str(absent))

        warn_if_tmpdir_unusable()

        assert not absent.exists()
