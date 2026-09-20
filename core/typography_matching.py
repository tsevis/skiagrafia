"""typography_matching.py  --  matching semantic glyphs to detector boxes.

Split out of orchestrator.py, which had grown past this project's 800-line
limit. Two of these were already pure functions wearing `self` as a costume:
neither touched any attribute of the Orchestrator.

The local detector and SAM stack stay authoritative for pixel geometry. A VLM
observation only ever SELECTS among detector proposals here; it never creates
or moves one.
"""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from core.interrogation import InterrogationCandidate
from core.pipeline_geometry import (
    bbox_area,
    bbox_iou,
    glyph_layer_label,
    normalized_bbox_to_image,
    rectangle_union_area,
)
from core.typography_labels import GlyphInspection, TypographyObservation
from models.grounded_sam import DetectionResult

# A semantic box must overlap a detector proposal at least this much before
# the two are considered the same glyph.
TYPOGRAPHY_BOX_MATCH_MIN_IOU = 0.30
# A prior glyph box must explain at least this share of a proposal before it
# counts towards the composite test below.
TYPOGRAPHY_COMPOSITE_MIN_CONTRIBUTION = 0.15
# ...and two or more of them must jointly cover this much of the proposal for
# it to be rejected as spanning several glyphs.
TYPOGRAPHY_COMPOSITE_MIN_COVERAGE = 0.80

Detections = list[tuple[DetectionResult, bool]]


class GlyphInspector(Protocol):
    """The optional interrogator capability this module asks for.

    Declared here rather than assumed, because the call site reaches it by
    getattr: an interrogator that cannot read glyphs simply does not have it.
    """

    def inspect_individual_glyphs(
        self, image: NDArray[np.uint8], candidate: InterrogationCandidate
    ) -> Any: ...


def inspect_individual_glyphs(
    interrogator: object,
    image: NDArray[np.uint8],
    candidate: InterrogationCandidate,
) -> GlyphInspection:
    """Ask the interrogator for an optional, local semantic glyph check.

    Reports an unavailable reason rather than a bare absence, so the caller
    can tell the operator that the check did not run.  Anything that is not
    the declared shape counts as unavailable: this is model output reaching a
    Protocol the interrogator only optionally implements, and is not trusted.
    """
    inspect = getattr(interrogator, "inspect_individual_glyphs", None)
    if not callable(inspect):
        return GlyphInspection(
            unavailable_reason="this engine cannot read individual glyphs"
        )
    inspection = inspect(image, candidate)
    if isinstance(inspection, GlyphInspection):
        return inspection
    # Tolerated for an interrogator still returning the bare observation.
    if isinstance(inspection, TypographyObservation):
        return GlyphInspection(observation=inspection)
    return GlyphInspection(
        unavailable_reason="the glyph reading was not of the expected shape"
    )


def match_typography_detections(
    image: NDArray[np.uint8],
    detections: list[tuple[DetectionResult, bool]],
    observation: TypographyObservation,
    category: str,
) -> tuple[list[tuple[DetectionResult, bool]], list[str]] | None:
    """Require a one-to-one match between semantic glyphs and local boxes.

    The local detector/SAM stack remains authoritative for pixel geometry.
    A VLM observation only selects a detector proposal when the two views
    overlap enough.  If even one semantic glyph has no independent local
    match, the method declines to relabel or discard any proposal.
    """
    if any(is_manual for _detection, is_manual in detections):
        return None
    height, width = image.shape[:2]
    matches: list[tuple[float, int, int]] = []
    for element_index, element in enumerate(observation.elements):
        semantic_bbox = normalized_bbox_to_image(element.bbox, width, height)
        for detection_index, (detection, _is_manual) in enumerate(detections):
            score = bbox_iou(semantic_bbox, detection.bbox)
            if score >= TYPOGRAPHY_BOX_MATCH_MIN_IOU:
                matches.append((score, element_index, detection_index))

    assigned_elements: set[int] = set()
    assigned_detections: set[int] = set()
    assignments: dict[int, int] = {}
    for _score, element_index, detection_index in sorted(
        matches,
        key=lambda match: (-match[0], match[1], match[2]),
    ):
        if element_index in assigned_elements or detection_index in assigned_detections:
            continue
        assigned_elements.add(element_index)
        assigned_detections.add(detection_index)
        assignments[element_index] = detection_index

    if len(assignments) != len(observation.elements):
        return None
    ordered_detections = [detections[assignments[index]] for index in range(len(observation.elements))]
    labels = [
        glyph_layer_label(category, observation.elements[index].glyph)
        for index in range(len(observation.elements))
    ]
    return ordered_detections, labels


def reject_cross_glyph_composites(
    detections: list[tuple[DetectionResult, bool]],
) -> tuple[list[tuple[DetectionResult, bool]], int]:
    """Reject a proposal substantially explained by two prior glyph boxes.

    This is the detector-only safety net for when the semantic verifier is
    unavailable.  It targets the characteristic false positive where one
    proposal spans pieces of two neighbouring glyphs; it does not reject
    an ordinary overlapping glyph or any manual box.
    """
    kept: list[tuple[DetectionResult, bool]] = []
    rejected = 0
    for detection, is_manual in detections:
        if is_manual:
            kept.append((detection, is_manual))
            continue
        candidate_area = bbox_area(detection.bbox)
        intersections: list[tuple[int, int, int, int]] = []
        for previous, previous_manual in kept:
            if previous_manual:
                continue
            x0 = max(detection.bbox[0], previous.bbox[0])
            y0 = max(detection.bbox[1], previous.bbox[1])
            x1 = min(detection.bbox[2], previous.bbox[2])
            y1 = min(detection.bbox[3], previous.bbox[3])
            overlap = (x0, y0, x1, y1)
            if candidate_area and bbox_area(overlap) / candidate_area >= TYPOGRAPHY_COMPOSITE_MIN_CONTRIBUTION:
                intersections.append(overlap)
        covered = rectangle_union_area(intersections)
        if (
            len(intersections) >= 2
            and candidate_area
            and covered / candidate_area >= TYPOGRAPHY_COMPOSITE_MIN_COVERAGE
        ):
            rejected += 1
            continue
        kept.append((detection, is_manual))
    return kept, rejected


def resolve_glyph_detections(
    interrogator: object,
    image: NDArray[np.uint8],
    parent: InterrogationCandidate,
    detections: Detections,
    layer_labels: list[str],
) -> tuple[Detections, list[str], list[str]]:
    """Split typography detections into individual glyphs where possible.

    Falls back to rejecting composites that straddle several glyphs.

    Returns (detections, labels, WARNINGS). The warnings are returned rather
    than appended to a PipelineResult: it keeps this module independent of
    the orchestrator's types, and it leaves the caller owning its own result
    instead of having it written to from here.
    """
    inspection = inspect_individual_glyphs(interrogator, image, parent)
    observation = inspection.observation
    matched = (
        match_typography_detections(image, detections, observation, parent.display_label)
        if observation is not None
        else None
    )
    if matched is not None:
        return matched[0], matched[1], []

    detections, rejected = reject_cross_glyph_composites(detections)
    layer_labels = [parent.display_label] * len(detections)
    warnings: list[str] = []
    if rejected:
        warnings.append(
            f"Excluded {rejected} composite typography proposal(s) for "
            f"'{parent.display_label}' because each overlapped multiple glyph instances."
        )
    if observation is not None:
        warnings.append(
            f"Typography reading for '{parent.display_label}' could not be matched "
            "one-to-one with local detections; retained only independently detected glyphs."
        )
    if inspection.unavailable_reason is not None:
        # A reading that arrived and disagreed is reported above; this is the
        # case where the check never ran at all, which used to be silent.
        warnings.append(
            f"Glyphs for '{parent.display_label}' could not be validated individually "
            f"({inspection.unavailable_reason}); kept the detector's own proposals instead."
        )
    return detections, layer_labels, warnings
