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
