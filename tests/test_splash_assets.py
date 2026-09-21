"""The about window's artwork is present and the right shape.

Deliberately does not import `ui.splash` at module level and never builds a
widget: this suite runs on the developer's own desktop, and a test that
constructs a Toplevel puts a window in front of whatever they are doing.
What can be checked without Tk is that the files exist and are what the
layout assumes -- which is the failure that would otherwise show up as a
blank panel nobody notices.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ASSETS = Path(__file__).resolve().parent.parent / "ui" / "assets"


def test_the_key_art_is_present_and_wide() -> None:
    """Drawn full-bleed across a 660pt window, so it must be wide."""
    art = Image.open(ASSETS / "SplashKeyArt.png")

    assert art.width >= 1600
    assert art.width > art.height * 2


def test_the_studio_mark_is_present_and_square_enough_to_sit_on_a_baseline() -> None:
    mark = Image.open(ASSETS / "TVDLogo.png")

    assert mark.width >= 256
    assert 0.8 < mark.width / mark.height < 1.25


def test_the_application_icon_is_a_full_size_square() -> None:
    icon = Image.open(ASSETS / "AppIcon.png")

    assert icon.size == (1024, 1024)


def test_the_mark_has_a_transparent_margin_rather_than_filling_its_box() -> None:
    """The lockup measures ink, not bounds. A mark that fills its box edge to
    edge would mean the measurement has nothing to correct and the plate would
    sit taller than the type beside it."""
    mark = Image.open(ASSETS / "TVDLogo.png").convert("RGBA")
    alpha = mark.split()[3]

    assert alpha.getextrema()[0] == 0, "expected transparent margin in the artwork"


def test_loading_artwork_never_raises_into_the_startup_path() -> None:
    """`apply_window_icon` runs while the main window is being built, and its
    own docstring calls the icon a cosmetic loss. Pillow raises RuntimeError,
    not TclError, when Tk is not ready for an image -- so the one exception the
    caller guarded was not the one that could reach it.

    Checked with no Tk root at all, which is the strictest form of "not ready"
    and needs no window.
    """
    from ui.splash import _scaled

    assert _scaled(ASSETS / "AppIcon.png", width=64) is None
