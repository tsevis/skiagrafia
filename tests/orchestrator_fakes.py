"""orchestrator_fakes.py  --  offline stand-ins for the five capabilities.

Lightweight implementations of the core.contracts Protocols (Interrogator,
Detector, Segmenter, AlphaRefiner, Vectorizer) that return canned numpy
masks and boxes. No SAM 2.1, GroundingDINO or VitMatte weights are ever
loaded, and no network or GPU work happens.

Shared by the test_orchestrator_* modules, so the pipeline tests can be
split by what they cover without any of them copying these.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.contracts import AlphaRefiner, CapabilitySet, Vectorizer
from core.interrogation import InterrogationCandidate, InterrogationResult
from core.typography_labels import TypographyObservation
from models.grounded_sam import DetectionResult

IMG_SIZE = 64


class FakeInterrogator:
    def __init__(
        self,
        candidates: list[InterrogationCandidate],
        children_by_parent: dict[str, list[str]] | None = None,
        typography_observation: TypographyObservation | None = None,
    ) -> None:
        self._candidates = candidates
        self._children = children_by_parent or {}
        self._typography_observation = typography_observation
        self.calls: list[tuple] = []
        self.confirmed_selections: list[dict[str, str]] = []

    def set_confirmed_selections(self, selections: dict[str, str]) -> None:
        self.confirmed_selections.append(dict(selections))

    def interrogate(self, image, confirmed_labels=None, knowledge_pack=None):
        self.calls.append((confirmed_labels, knowledge_pack))
        return InterrogationResult(
            candidates=self._candidates,
            children_by_parent=self._children,
        )

    def inspect_individual_glyphs(self, image, candidate):
        return self._typography_observation


class FakeDetector:
    """Returns a canned bbox for labels present in `boxes`, else `default`."""

    def __init__(
        self,
        boxes: dict[str, tuple[int, int, int, int] | None] | None = None,
        default: tuple[int, int, int, int] | None = None,
    ) -> None:
        self._boxes = boxes or {}
        self._default = default
        self.calls: list[str] = []

    def detect_box(self, image, label, box_threshold=0.35, text_threshold=0.25):
        self.calls.append(label)
        if label in self._boxes:
            bbox = self._boxes[label]
            if bbox is None:
                return None
            return DetectionResult(label=label, bbox=bbox, confidence=0.9)
        if self._default is not None:
            return DetectionResult(label=label, bbox=self._default, confidence=0.9)
        return None


class MultiInstanceDetector(FakeDetector):
    """Detector fake that exposes a deterministic ordered proposal set."""

    def __init__(self, instances: list[DetectionResult]) -> None:
        super().__init__()
        self._instances = instances

    def detect_instances(self, image, label, box_threshold=0.35, text_threshold=0.25):
        self.calls.append(label)
        return list(self._instances)


class FakeSegmenter:
    """Produces a full-coverage mask inside the requested bbox by default."""

    def __init__(self, mask_by_label: dict[str, NDArray[np.uint8]] | None = None) -> None:
        self._mask_by_label = mask_by_label or {}
        self.clear_cache_called = False
        self.calls: list[tuple] = []

    def segment(self, image, bbox, label="", prefer_full_box=False):
        self.calls.append((bbox, label, prefer_full_box))
        if label in self._mask_by_label:
            return self._mask_by_label[label]
        h, w = image.shape[:2]
        x0, y0, x1, y1 = bbox
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 255
        return mask

    def clear_cache(self) -> None:
        self.clear_cache_called = True


class FakeAlphaRefiner:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, image, mask):
        self.calls += 1
        return mask.copy()


class FakeVectorizer:
    def __init__(self) -> None:
        self.calls = 0

    def trace(self, mask):
        self.calls += 1
        return (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<path d="M0 0L1 1" fill="#000000"/></svg>'
        )


def _make_caps(
    interrogator: FakeInterrogator,
    detector: FakeDetector,
    segmenter: FakeSegmenter,
    alpha_refiner: AlphaRefiner | None = None,
    vectorizer: Vectorizer | None = None,
) -> CapabilitySet:
    return CapabilitySet(
        interrogator=interrogator,
        detector=detector,
        segmenter=segmenter,
        alpha_refiner=alpha_refiner or FakeAlphaRefiner(),
        vectorizer=vectorizer or FakeVectorizer(),
    )


def _write_image(path: Path, size: int = IMG_SIZE) -> Path:
    img = np.full((size, size, 3), (30, 60, 90), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def _candidate(
    label: str,
    source_model: str = "vlm",
    detector_phrases: list[str] | None = None,
    confidence: float = 0.8,
) -> InterrogationCandidate:
    return InterrogationCandidate(
        canonical_label=label,
        display_label=label,
        detector_phrases=detector_phrases or [label],
        source_model=source_model,
        confidence=confidence,
    )
