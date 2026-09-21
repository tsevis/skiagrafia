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
from types import ModuleType

import numpy
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


def test_the_key_art_panel_is_as_tall_as_the_art_it_draws() -> None:
    """The panel is a fixed height and the art is drawn into it at full width,
    so a panel shorter than the art crops its own bottom edge -- which is
    exactly where the lockup and the studio mark live.
    """
    from ui import splash

    art = Image.open(ASSETS / "SplashKeyArt.png")
    drawn_height = round(art.height * splash.WIDTH / art.width)

    assert drawn_height == splash.KEY_ART_HEIGHT


def test_the_studio_mark_is_already_in_the_key_art() -> None:
    """The mark is composited into the bitmap by `Scripts/make-keyart.py` in
    the native application's repository, because a mark laid out beside the
    type rendered as nothing there. The about window therefore must not draw
    its own: two marks appear, at two sizes.

    This checks the reason that rule exists. If the artwork is ever
    regenerated without the mark, the window loses it silently and this is
    what says so.
    """
    from ui import splash

    mark = _mark_ink_in_the_key_art(Image.open(ASSETS / "SplashKeyArt.png"), splash)

    assert mark > 0.25, f"no studio mark found where the layout says it is ({mark:.3f})"


def _mark_ink_in_the_key_art(art: Image.Image, splash: ModuleType) -> float:
    """Fraction of the mark's box in the artwork covered by the mark's own ink.

    The mark is a solid plate, so comparing coverage is enough and does not
    depend on the photograph behind it.
    """
    panel = art.resize(
        (splash.WIDTH, splash.KEY_ART_HEIGHT), Image.Resampling.LANCZOS
    ).convert("RGB")
    top = splash.KEY_ART_HEIGHT - splash.MARK_BOTTOM - splash.MARK_HEIGHT
    box = panel.crop(
        (
            splash.MARK_INSET,
            top,
            splash.MARK_INSET + splash.MARK_WIDTH,
            top + splash.MARK_HEIGHT,
        )
    )
    plate = _plate_colour()
    pixels = numpy.asarray(box, dtype=numpy.int16)
    close = numpy.all(numpy.abs(pixels - numpy.array(plate)) <= 40, axis=-1)
    return float(close.mean())


def _plate_colour() -> tuple[int, int, int]:
    """The mark's own background colour, read from the artwork rather than
    written down: the file's corner is transparent, not plate."""
    mark = numpy.asarray(Image.open(ASSETS / "TVDLogo.png").convert("RGBA"))
    opaque = mark[mark[..., 3] > 200][:, :3]
    colours, counts = numpy.unique(opaque, axis=0, return_counts=True)
    red, green, blue = colours[counts.argmax()]
    return int(red), int(green), int(blue)
