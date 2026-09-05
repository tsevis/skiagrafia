"""test_cairo_support_extra.py  --  Remaining CairoSVG discovery paths.

Complements tests/test_cairo_support.py (not modified here, per task
constraints) by covering the default-candidate-list branch of
``configure_cairo_library_path`` and the ``load_cairosvg`` import wrapper.
All environment mutations go through monkeypatch so they are undone at
teardown regardless of what the function under test writes directly.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import cairo_support


class TestDefaultCandidateDirs:
    def test_uses_builtin_candidate_list_when_none_supplied(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cairo_support.sys, "platform", "darwin")
        # Registers restoration regardless of what the function writes.
        monkeypatch.delenv("DYLD_FALLBACK_LIBRARY_PATH", raising=False)

        result = cairo_support.configure_cairo_library_path()

        assert isinstance(result, list)
        assert all(isinstance(item, Path) for item in result)

    def test_non_darwin_short_circuits_before_building_candidates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cairo_support.sys, "platform", "linux")

        assert cairo_support.configure_cairo_library_path() == []


class TestLoadCairosvg:
    def test_returns_module_on_successful_import(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            cairo_support, "configure_cairo_library_path", lambda: []
        )
        logger = logging.getLogger("test.cairo_support.success")

        result = cairo_support.load_cairosvg(logger)

        assert result is not None
        assert result.__name__ == "cairosvg"

    def test_returns_none_and_warns_when_import_fails(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(
            cairo_support, "configure_cairo_library_path", lambda: []
        )
        # None in sys.modules makes `import cairosvg` raise ImportError.
        monkeypatch.setitem(sys.modules, "cairosvg", None)
        logger = logging.getLogger("test.cairo_support.failure")

        with caplog.at_level(logging.WARNING, logger=logger.name):
            result = cairo_support.load_cairosvg(logger)

        assert result is None
        assert any("CairoSVG is unavailable" in r.getMessage() for r in caplog.records)

    def test_calls_configure_cairo_library_path_before_importing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []
        monkeypatch.setattr(
            cairo_support,
            "configure_cairo_library_path",
            lambda: calls.append("configured"),
        )
        monkeypatch.setitem(sys.modules, "cairosvg", None)
        logger = logging.getLogger("test.cairo_support.order")

        cairo_support.load_cairosvg(logger)

        assert calls == ["configured"]
