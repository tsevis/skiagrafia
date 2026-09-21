"""What MLXSAM3 tells the run about work it did not do with MLX.

Two separate faults, both measured on a real corpus earlier: a detector that
quietly hands the image to a different engine, and a warning list that is
never emptied, so image N reports image 1's problems as its own.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.grounded_sam import DetectionResult
from models.mlx_sam3 import MLXSAM3
from models.sam3_grounding import GroundedInstances


class _Fallback:
    """Stands in for GroundingDINO + SAM 2.1."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def detect_instances(
        self, image, label, box_threshold: float = 0.35, text_threshold: float = 0.25
    ) -> list[DetectionResult]:
        self.calls.append(label)
        return [DetectionResult(label=label, bbox=(0, 0, 4, 4), confidence=0.5,
                                source="groundingdino")]

    def clear_cache(self) -> None:
        pass


def _detector(grounded_results: list[DetectionResult]) -> tuple[MLXSAM3, _Fallback]:
    fallback = _Fallback()
    detector = MLXSAM3(Path("/nonexistent"), fallback)  # type: ignore[arg-type]
    # Replace the MLX call itself: these tests are about what happens around
    # it, and the model is not available in CI.
    empty = GroundedInstances(
        boxes=np.zeros((0, 4), dtype=np.float32),
        masks=np.zeros((0, 8, 8), dtype=np.uint8),
        scores=[], presence=0.9, rescued=False,
    )
    detector._ground = lambda image, label, require_presence: empty  # type: ignore[assignment]
    detector._detections = lambda image, label, grounded: grounded_results  # type: ignore[assignment]
    return detector, fallback


def _image() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


def test_a_clean_mlx_run_that_accepts_nothing_says_it_used_another_engine() -> None:
    """The exception path warns; the empty-result path did not.

    MLX runs, recognises nothing, and the caller receives GroundingDINO
    detections instead -- different masks, a different size metric and a
    different part gate, with no record that the engine changed.
    """
    detector, fallback = _detector([])

    results = detector.detect_instances(_image(), "computer")

    assert fallback.calls == ["computer"]
    assert results and results[0].source == "groundingdino"
    assert any("GroundingDINO" in warning for warning in detector.warnings), detector.warnings


def test_warnings_do_not_follow_the_detector_onto_the_next_image() -> None:
    """clear_cache() resets the cached image and state but not the warnings.

    The orchestrator drains this list into every PipelineResult, so in the
    single-image application image two reports image one's problems as well
    as its own.
    """
    detector, _fallback = _detector([])

    detector.detect_instances(_image(), "computer")
    first = list(detector.warnings)
    detector.clear_cache()

    assert first, "expected the first run to warn at all"
    assert detector.warnings == []
