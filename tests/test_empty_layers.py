"""test_empty_layers.py  --  a layer with nothing in it is not a file.

The APPLE50 book produced 23 TIFFs out of 9,779 whose alpha channel had
not one opaque pixel: a full-canvas image of nothing. They arrive when a
matte survives detection and is then emptied -- by refinement, by the
source's own transparency, or by being clipped to a parent it does not
overlap.

Dropping them silently would be the wrong cure. The run says which layer
produced nothing, because a layer the interrogator asked for and the
exporter skipped is exactly the kind of gap this application promises to
report rather than swallow.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator_fakes import (
    FakeAlphaRefiner,
    FakeDetector,
    FakeInterrogator,
    FakeSegmenter,
    _candidate,
    _make_caps,
    _write_image,
)

from core.orchestrator import Orchestrator, PipelineResult

GHOST_BOX = (0, 0, 24, 24)
BAG_BOX = (32, 32, 60, 60)


class EmptyingRefiner(FakeAlphaRefiner):
    """Returns nothing for the mask in the ghost's corner.

    Refinement emptying a matte that survived detection is one of the ways
    these files arise; clipping a child to a parent it does not overlap is
    another. Either way the exporter is handed nothing.
    """

    def predict(self, image, mask) -> np.ndarray:
        if mask[GHOST_BOX[1], GHOST_BOX[0]] > 0:
            return np.zeros_like(mask)
        return super().predict(image, mask)


def _run(tmp_path: Path) -> PipelineResult:
    # Distinct boxes: identical masks are merged as duplicates, which would
    # leave only one layer and nothing to compare.
    caps = _make_caps(
        FakeInterrogator([_candidate("ghost"), _candidate("bag")]),
        FakeDetector(boxes={"ghost": GHOST_BOX, "bag": BAG_BOX}),
        FakeSegmenter(),
        alpha_refiner=EmptyingRefiner(),
    )
    # Per-layer TIFFs are only written in bitmap mode; that is the mode the
    # book ran in and the one that produced the empty files.
    return Orchestrator(
        caps, output_dir=tmp_path / "out", output_mode="vector+bitmap"
    ).process(_write_image(tmp_path / "in.png"))


def test_no_tiff_is_written_for_a_layer_with_no_opaque_pixels(tmp_path: Path) -> None:
    result = _run(tmp_path)

    assert result.error is None
    assert not any("ghost" in name for name in result.tiff_files), result.tiff_files


def test_the_layers_that_did_produce_something_are_unaffected(tmp_path: Path) -> None:
    result = _run(tmp_path)

    assert any("bag" in name for name in result.tiff_files), result.tiff_files


def test_the_run_says_which_layer_produced_nothing(tmp_path: Path) -> None:
    # Skipping in silence would be the same fault as the empty file: less
    # than was asked for, with nothing said about it.
    result = _run(tmp_path)

    assert any("ghost" in warning for warning in result.warnings), result.warnings
