"""Tests for s1bursts.config settings loading."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from s1bursts.config import CDSES3Settings, get_settings


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

    def test_reads_dotenv_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv("CDSE_USERNAME", raising=False)
        (tmp_path / ".env").write_text("CDSE_USERNAME=from-dotenv\n")
        monkeypatch.chdir(tmp_path)

        assert get_settings().cdse.username == "from-dotenv"

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        for var in ("SNAP_GPT_PATH", "SNAP_MAX_WORKERS", "SNAP_MAX_RAM_GB"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.chdir(tmp_path)

        snap = get_settings().snap

        assert snap.gpt_path is None
        assert snap.max_workers == 1
        assert snap.max_ram_gb == 4.0
        assert snap.auxdata_path == Path.home() / ".snap/auxdata"


class TestS3Env:
    def test_injects_and_restores_aws_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "previous-key")
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        s3 = CDSES3Settings(access_key_id="cdse-key", secret_access_key="cdse-secret")

        with s3.s3_env():
            assert os.environ["AWS_ACCESS_KEY_ID"] == "cdse-key"
            assert os.environ["AWS_SECRET_ACCESS_KEY"] == "cdse-secret"

        assert os.environ["AWS_ACCESS_KEY_ID"] == "previous-key"
        assert "AWS_SECRET_ACCESS_KEY" not in os.environ

    def test_missing_credentials_raise(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv("CDSE_S3_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("CDSE_S3_SECRET_ACCESS_KEY", raising=False)
        monkeypatch.chdir(tmp_path)
        s3 = CDSES3Settings()

        with pytest.raises(ValueError, match="CDSE S3 credentials not configured"), s3.s3_env():
            pass
