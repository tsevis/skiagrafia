"""Filters that decide which detected instances become layers.

Both run per label, after segmentation and before a layer is accepted.
"""
from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from core.pipeline_geometry import mask_containment
from core.pipeline_results import PipelineResult

logger = logging.getLogger(__name__)


class EdgeRefiner(Protocol):
    def refine(self, image: NDArray[np.uint8], mask: NDArray[np.uint8]) -> NDArray[np.uint8]: ...


#: A same-label instance this much inside one at least NESTED_AREA_RATIO times
#: its area is a picture of the object on the object, not another object.
NESTED_CONTAINMENT = 0.90
NESTED_AREA_RATIO = 4.0
#: Context kept around a detection when asking the vision model about it.
VERIFY_CROP_MARGIN = 0.1


def drop_nested_instances(proposals: list, result: PipelineResult) -> list:
    """Drop an instance lying inside a much larger instance of the same label.

    A text grounder answers "phone" for a phone's own Phone icon too. A box
    the person drew is never dropped: they asked for exactly that region.
    """
    areas = [int(np.count_nonzero(mask > 127)) for mask, *_ in proposals]
    kept = []
    for index, (mask, detection, layer_label, is_manual) in enumerate(proposals):
        nested = not is_manual and any(
            other != index
            and proposals[other][2] == layer_label
            and areas[other] >= NESTED_AREA_RATIO * areas[index]
            and mask_containment(mask, proposals[other][0]) >= NESTED_CONTAINMENT
            for other in range(len(proposals))
        )
        if nested:
            result.warnings.append(
                f"Excluded a '{layer_label}' lying inside a larger '{layer_label}'."
            )
            continue
        kept.append((mask, detection, layer_label, is_manual))
    return kept


def verified_proposals(
interrogator: object, image: NDArray[np.uint8], proposals: list, result: PipelineResult
) -> list:
    """Proposals the vision model accepts as the object itself.

    A drawn box is the person's own answer and is never asked about. An
    unreadable or unavailable answer keeps the instance.
    """
    confirms = getattr(interrogator, "confirms_object", None)
    if not callable(confirms):
        result.warnings.append("This engine cannot verify objects; none were checked.")
        return proposals
    height, width = image.shape[:2]
    kept = []
    for proposal in proposals:
        _, detection, layer_label, is_manual = proposal
        if not is_manual:
            x0, y0, x1, y1 = detection.bbox
            margin = int(VERIFY_CROP_MARGIN * max(x1 - x0, y1 - y0))
            crop = image[
                max(0, y0 - margin) : min(height, y1 + margin),
                max(0, x0 - margin) : min(width, x1 + margin),
            ]
            if confirms(crop, layer_label) is False:
                result.warnings.append(
                    f"Excluded a '{layer_label}' the vision model judged to be "
                    "an icon or picture of one."
                )
                continue
        kept.append(proposal)
    return kept


def refined_edges(
    refiner: EdgeRefiner, image: NDArray[np.uint8], proposals: list, result: PipelineResult
) -> list:
    """Proposals with each mask's edge redrawn; a refiner failure keeps SAM's."""
    refined = []
    for mask, detection, layer_label, is_manual in proposals:
        try:
            mask = refiner.refine(image, mask)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            logger.warning("Edge refinement failed for '%s'", layer_label, exc_info=True)
            result.warnings.append(f"Kept the unrefined edge of '{layer_label}': {exc}")
        refined.append((mask, detection, layer_label, is_manual))
    return refined
