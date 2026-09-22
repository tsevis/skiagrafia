"""test_layer_transparent_regions.py  --  what a layer stores where nothing is.

A layer TIFF carried the entire source photograph in its RGB channels and
masked it only in alpha. LZW cannot compress a photograph, so every layer
cost a full uncompressed frame however small its object: the APPLE50 book
produced 9,779 layers at 19 GB, median coverage 0.46% of the canvas.

Clearing colour far from the object lets the compression do its work. A
bleed margin of real colour is kept around the edge, because compositing
that expands or blurs a matte reads the colour under the transparency,
and black there shows up as a dark fringe.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from processors.output_writer import COLOUR_BLEED_PIXELS, write_tiff

SIZE = 256
BOX = (100, 100, 150, 150)  # left, top, right, bottom


def _busy_photograph() -> np.ndarray:
    """Noise, so nothing compresses away by being flat."""
    rng = np.random.default_rng(4)
    return rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)


def _alpha_with_one_box() -> np.ndarray:
    alpha = np.zeros((SIZE, SIZE), dtype=np.uint8)
    left, top, right, bottom = BOX
    alpha[top:bottom, left:right] = 255
    return alpha


def _written(tmp_path: Path) -> np.ndarray:
    path = write_tiff(_busy_photograph(), tmp_path / "layer.tiff", _alpha_with_one_box())
    with Image.open(path) as opened:
        return np.array(opened.convert("RGBA"))


def test_the_layer_keeps_its_canvas_and_its_alpha(tmp_path: Path) -> None:
    # Layers are composited by position. Changing the canvas would move
    # every object in the book.
    written = _written(tmp_path)

    assert written.shape == (SIZE, SIZE, 4)
    assert np.array_equal(written[..., 3], _alpha_with_one_box())


def test_colour_under_the_object_is_untouched(tmp_path: Path) -> None:
    source = _busy_photograph()
    written = _written(tmp_path)
    left, top, right, bottom = BOX

    assert np.array_equal(written[top:bottom, left:right, :3],
                          source[top:bottom, left:right])


def test_colour_just_outside_the_edge_survives_for_compositing(tmp_path: Path) -> None:
    # Dark pixels under the transparency become a dark halo the moment a
    # compositor grows or blurs the matte.
    source = _busy_photograph()
    written = _written(tmp_path)
    left, top, right, bottom = BOX
    margin = COLOUR_BLEED_PIXELS - 2

    assert np.array_equal(written[top - margin, left:right, :3],
                          source[top - margin, left:right])


def test_colour_far_from_the_object_is_dropped(tmp_path: Path) -> None:
    written = _written(tmp_path)

    assert written[0, 0, :3].tolist() == [0, 0, 0]
    assert written[-1, -1, :3].tolist() == [0, 0, 0]


def test_a_mostly_transparent_layer_stops_costing_a_whole_photograph(
    tmp_path: Path,
) -> None:
    # The point of the change: a 4% object must not cost a 100% frame.
    kept = write_tiff(_busy_photograph(), tmp_path / "masked.tiff", _alpha_with_one_box())
    whole = write_tiff(_busy_photograph(), tmp_path / "whole.tiff",
                       np.full((SIZE, SIZE), 255, dtype=np.uint8))

    assert kept.stat().st_size < whole.stat().st_size * 0.5
