"""Tests for macOS CairoSVG library discovery."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import cairo_support


def test_configure_cairo_library_path_adds_detected_directory(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "libcairo.2.dylib").touch()
    monkeypatch.setattr(cairo_support.sys, "platform", "darwin")
    monkeypatch.setenv("DYLD_FALLBACK_LIBRARY_PATH", "/existing/lib")

    configured = cairo_support.configure_cairo_library_path([tmp_path])

    assert configured == [tmp_path]
    assert os.environ["DYLD_FALLBACK_LIBRARY_PATH"] == (
        f"{tmp_path}:/existing/lib"
    )


def test_configure_cairo_library_path_ignores_missing_libraries(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(cairo_support.sys, "platform", "darwin")
    monkeypatch.delenv("DYLD_FALLBACK_LIBRARY_PATH", raising=False)

    assert cairo_support.configure_cairo_library_path([tmp_path]) == []
    assert "DYLD_FALLBACK_LIBRARY_PATH" not in os.environ
