"""CairoSVG runtime support for macOS installations.

Homebrew keeps ``libcairo`` outside the default lookup path of some Python
environments (notably Conda).  Configure the dynamic-loader fallback path
immediately before importing CairoSVG so all SVG renderers behave consistently.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable


def configure_cairo_library_path(
    candidate_dirs: Iterable[Path] | None = None,
) -> list[Path]:
    """Make installed macOS Cairo libraries discoverable to CairoSVG.

    Returns the usable library directories that were added or already present.
    The optional ``candidate_dirs`` argument keeps the discovery logic testable.
    """
    if sys.platform != "darwin":
        return []

    candidates = list(candidate_dirs) if candidate_dirs is not None else [
        Path("/opt/homebrew/lib"),
        Path("/usr/local/lib"),
        Path(sys.prefix) / "lib",
    ]
    available = [
        directory
        for directory in candidates
        if (directory / "libcairo.2.dylib").exists()
        or (directory / "libcairo.dylib").exists()
    ]
    if not available:
        return []

    existing = [
        Path(path)
        for path in os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "").split(":")
        if path
    ]
    ordered = list(dict.fromkeys([*available, *existing]))
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(map(str, ordered))
    return available


def load_cairosvg(logger: logging.Logger) -> ModuleType | None:
    """Load CairoSVG after configuring the native Cairo lookup path.

    ``None`` lets display-only callers fall back gracefully; PDF export turns
    it into a clear error for the user.
    """
    configure_cairo_library_path()
    try:
        import cairosvg
    except (ImportError, OSError) as exc:
        logger.warning("CairoSVG is unavailable: %s", exc)
        return None
    return cairosvg
