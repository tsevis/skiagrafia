"""detection_policy.py  --  turning a named candidate into detections.

Manual boxes, detector calls, per-instance selection, the mask for one
detection, and the candidate a part is detected as. Split out of
orchestrator.py, which had grown past this project's 800-line limit.

A mixin rather than free functions: every method here needs the capability
set and the thresholds the Orchestrator was constructed with, and threading
six of those through each call would say less than naming them once.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

# Runtime imports: _child_candidate CONSTRUCTS an InterrogationCandidate, so
# a TYPE_CHECKING-only import would leave the name undefined when it runs.
# core.interrogation does not import this module, so there is no cycle.
from core.interrogation import InterrogationCandidate
from core.pipeline_geometry import bbox_area, clip_mask_to_bbox
from models.grounded_sam import DetectionResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from core.knowledge import KnowledgePack


class DetectionSizeLike(Protocol):
    """What instance_size() needs from a detection: a box, maybe a mask."""

    bbox: tuple[int, int, int, int]
    mask: NDArray[np.uint8] | None

def instance_size(detections: list[DetectionSizeLike]) -> Callable[[Any], int]:
    """A size metric that is comparable across ALL of these detections.

    Mask pixels are the better measure of how big an object actually is, so
    they are used when every detection carries a mask. The moment one does
    not, the whole list falls back to bounding-box area.

    Mixing the two per element -- which is what this replaced -- compares a
    silhouette against the rectangle drawn around a different object. A mask
    is far smaller than its own box, so a list mixing detector sources (MLX
    SAM 3 populates a mask, the GroundingDINO fallback does not) made
    "largest" pick the unmasked detection whatever its real size.
    """
    if all(detection.mask is not None for detection in detections):
        return lambda detection: int(np.count_nonzero(detection.mask))
    return lambda detection: bbox_area(detection.bbox)


class DetectionPolicyMixin:
    """Detection behaviour for the Orchestrator, which mixes this in.

    The attributes below are PROVIDED BY the Orchestrator: annotations with
    no assignment, so nothing exists at runtime and the MRO is untouched.
    """

    _detector: Any
    _segmenter: Any
    _knowledge_pack: KnowledgePack | None
    _box_threshold: float
    _text_threshold: float
    _quality: str

    def _detect_instances(
        self,
        image: NDArray[np.uint8],
        candidate: InterrogationCandidate,
        manual_lookup: dict[str, list[tuple[int, int, int, int]]] | None,
    ) -> list[tuple[DetectionResult, bool]]:
        manual: list[tuple[DetectionResult, bool]] = []
        while manual_lookup:
            bbox = self._consume_manual_bbox(manual_lookup, candidate)
            if bbox is None:
                break
            h, w = image.shape[:2]
            x0, y0, x1, y1 = bbox
            bbox = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                manual.append((DetectionResult(label=candidate.display_label, bbox=bbox, confidence=1.0, source="manual"), True))
        if manual:
            return manual
        detect_many = cast(
            "Callable[..., list[DetectionResult]] | None",
            getattr(self._detector, "detect_instances", None),
        )
        if candidate.role == "child" and self._quality != "detailed":
            detect_many = cast(
                "Callable[..., list[DetectionResult]] | None",
                getattr(self._detector, "detect_part_instances", detect_many),
            )
        for phrase in candidate.detector_phrases or [candidate.display_label]:
            if callable(detect_many):
                detections = detect_many(image, phrase, self._box_threshold, self._text_threshold)
            else:
                detection = self._detector.detect_box(image, phrase, self._box_threshold, self._text_threshold)
                detections = [detection] if detection else []
            # Never pass invalid coordinates into a predictor.
            h, w = image.shape[:2]
            valid = []
            for detection in detections:
                x0, y0, x1, y1 = detection.bbox
                bbox = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    valid.append(detection.model_copy(update={"bbox": bbox}))
            if valid:
                selection = candidate.selection
                if selection in {"leftmost", "rightmost"}:
                    valid = [sorted(valid, key=lambda d: (d.bbox[0] + d.bbox[2]) / 2)[0 if selection == "leftmost" else -1]]
                elif selection in {"largest", "smallest"}:
                    ordered = sorted(valid, key=instance_size(valid))
                    valid = [ordered[0 if selection == "smallest" else -1]]
                return [(detection, False) for detection in valid]
        return []

    def _detection_mask(
        self,
        image: NDArray[np.uint8],
        detection: DetectionResult,
        label: str,
        manual: bool = False,
    ) -> NDArray[np.uint8]:
        mask = detection.mask
        if mask is None or manual:
            mask = self._segmenter.segment(image, detection.bbox, label, prefer_full_box=manual)
        if mask.shape != image.shape[:2]:
            raise RuntimeError(f"Mask generation failed for '{label}': invalid dimensions.")
        mask = (mask > 127).astype(np.uint8) * 255
        return clip_mask_to_bbox(mask, detection.bbox) if manual else mask

    def _detect_candidate(
        self,
        image: NDArray[np.uint8],
        candidate: InterrogationCandidate,
        manual_lookup: dict[str, list[tuple[int, int, int, int]]] | None = None,
    ) -> DetectionResult | None:
        det, _is_manual = self._detect_candidate_ex(image, candidate, manual_lookup)
        return det

    def _detect_candidate_ex(
        self,
        image: NDArray[np.uint8],
        candidate: InterrogationCandidate,
        manual_lookup: dict[str, list[tuple[int, int, int, int]]] | None = None,
    ) -> tuple[DetectionResult | None, bool]:
        """Like _detect_candidate but also returns whether detection was manual."""
        if manual_lookup:
            manual_bbox = self._consume_manual_bbox(manual_lookup, candidate)
            if manual_bbox is not None:
                return DetectionResult(
                    label=candidate.display_label,
                    bbox=manual_bbox,
                    confidence=1.0,
                ), True
        phrases = candidate.detector_phrases or [candidate.display_label]
        for phrase in phrases:
            detection = self._detector.detect_box(
                image,
                phrase,
                self._box_threshold,
                self._text_threshold,
            )
            if detection is not None:
                return detection, False
        return None, False

    def _build_manual_lookup(
        self,
        manual_detections: list[dict] | None,
    ) -> dict[str, list[tuple[int, int, int, int]]]:
        lookup: dict[str, list[tuple[int, int, int, int]]] = {}
        if not manual_detections:
            return lookup
        for detection in manual_detections:
            bbox = detection.get("bbox")
            label = str(detection.get("label", "")).strip().lower()
            if not label or not bbox:
                continue
            lookup.setdefault(label, []).append(tuple(bbox))
        return lookup

    def _consume_manual_bbox(
        self,
        manual_lookup: dict[str, list[tuple[int, int, int, int]]],
        candidate: InterrogationCandidate,
    ) -> tuple[int, int, int, int] | None:
        possible_labels = [
            candidate.display_label.strip().lower(),
            candidate.canonical_label.strip().lower(),
            *[phrase.strip().lower() for phrase in candidate.detector_phrases],
        ]
        for label in possible_labels:
            queue = manual_lookup.get(label)
            if queue:
                return queue.pop(0)
        return None

    def _child_candidate(
        self,
        parent: InterrogationCandidate,
        child_label: str,
    ) -> InterrogationCandidate:
        knowledge = self._knowledge_pack.find_object(child_label) if self._knowledge_pack else None
        detector_phrases = (
            knowledge.ranked_detector_phrases(4)
            if knowledge is not None
            else [child_label, f"{child_label} detail", f"{parent.display_label} {child_label}"]
        )
        return InterrogationCandidate(
            canonical_label=knowledge.canonical if knowledge else child_label,
            display_label=knowledge.canonical if knowledge else child_label,
            detector_phrases=detector_phrases,
            source_model="child",
            confidence=0.7,
            role="child",
            parent=parent.display_label,
        )