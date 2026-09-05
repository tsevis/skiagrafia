"""test_mps_utils.py  --  Torch device selection (MPS vs CPU fallback)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import mps_utils


class TestGetDevice:
    def test_returns_mps_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            mps_utils.torch.backends.mps, "is_available", lambda: True
        )

        device = mps_utils.get_device()

        assert device.type == "mps"

    def test_falls_back_to_cpu_when_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            mps_utils.torch.backends.mps, "is_available", lambda: False
        )

        device = mps_utils.get_device()

        assert device.type == "cpu"

    def test_cpu_fallback_logs_a_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(
            mps_utils.torch.backends.mps, "is_available", lambda: False
        )

        with caplog.at_level(logging.WARNING, logger="utils.mps_utils"):
            mps_utils.get_device()

        assert any("MPS unavailable" in r.getMessage() for r in caplog.records)

    def test_mps_available_does_not_log_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(
            mps_utils.torch.backends.mps, "is_available", lambda: True
        )

        with caplog.at_level(logging.WARNING, logger="utils.mps_utils"):
            mps_utils.get_device()

        assert not any("MPS unavailable" in r.getMessage() for r in caplog.records)


class TestModuleLevelEnvVar:
    def test_mps_fallback_env_var_is_present(self) -> None:
        import os

        assert "PYTORCH_ENABLE_MPS_FALLBACK" in os.environ
