"""Run one MLX SAM 3 text prompt and keep both of its scores apart.

``sam3.model.sam3_image_processor.Sam3Processor.set_text_prompt`` folds the
image-level presence probability into every query score before it thresholds,
so a caller downstream of it cannot tell a weak localisation from a confident
one the presence head vetoed. This module makes the same grounding call and
returns the two quantities separately, leaving the vendored source -- which is
shared with other projects -- untouched.

The box scaling and mask interpolation below mirror ``Sam3Processor
._call_grounding``. Keep them in step with the bundled source.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from models.sam3_scoring import SelectionPolicy, select_queries


@dataclass(frozen=True)
class GroundedInstances:
    """Accepted detections for one prompt, best first.

    Attributes
    ----------
    boxes:
        ``(k, 4)`` float array of ``x0, y0, x1, y1`` in source pixels.
    masks:
        ``(k, height, width)`` uint8 array, 0 or 255.
    scores:
        One reported confidence per instance; see ``sam3_scoring.Selection``.
    presence:
        The image-level presence probability, for logging and diagnostics.
    rescued:
        True when the presence gate rejected the image; see ``sam3_scoring``.
    """

    boxes: NDArray[np.float32]
    masks: NDArray[np.uint8]
    scores: list[float]
    presence: float
    rescued: bool


def ground_text_prompt(
    processor: Any,
    state: dict[str, Any],
    label: str,
    width: int,
    height: int,
    policy: SelectionPolicy,
) -> GroundedInstances:
    """Prompt an image already passed to ``processor.set_image``.

    The caller owns ``state`` and is responsible for resetting prompts on it
    between labels, exactly as it would be when calling the processor itself.
    """
    import mlx.core as mx

    model = processor.model
    state["backbone_out"].update(model.backbone.call_text([label]))
    if "geometric_prompt" not in state:
        state["geometric_prompt"] = model._get_dummy_prompt()  # the vendored processor has no public accessor
    outputs = model.call_grounding(
        backbone_out=state["backbone_out"],
        find_input=processor.find_stage,
        geometric_prompt=state["geometric_prompt"],
        find_target=None,
    )

    query_probs = np.asarray(mx.sigmoid(outputs["pred_logits"]))[0].reshape(-1)
    presence = float(np.asarray(mx.sigmoid(outputs["presence_logit_dec"])).reshape(-1)[0])
    selection = select_queries(query_probs, presence, policy)
    empty = GroundedInstances(
        boxes=np.zeros((0, 4), dtype=np.float32),
        masks=np.zeros((0, height, width), dtype=np.uint8),
        scores=[],
        presence=presence,
        rescued=False,
    )
    if not selection.indices:
        return empty

    boxes, masks = _materialise(outputs, selection.indices, width, height)
    return GroundedInstances(
        boxes=boxes,
        masks=masks,
        scores=selection.scores,
        presence=presence,
        rescued=selection.rescued,
    )


def _materialise(
    outputs: dict[str, Any], indices: list[int], width: int, height: int
) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    """Scale the chosen boxes and upsample their masks to the source size."""
    import mlx.core as mx
    from sam3.model import (  # type: ignore[import-not-found]  # vendored; resolved at runtime from the model directory
        box_ops,
    )
    from sam3.model.data_misc import (  # type: ignore[import-not-found]  # vendored; resolved at runtime from the model directory
        interpolate,
    )

    chosen = mx.array(np.asarray(indices, dtype=np.int32))
    boxes = np.asarray(box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"][0])[chosen])
    boxes = boxes * np.asarray([width, height, width, height], dtype=np.float32)
    resized = interpolate(
        outputs["pred_masks"][0][chosen][:, None],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    masks = (np.asarray(mx.sigmoid(resized))[:, 0] > 0.5).astype(np.uint8) * 255
    return boxes.astype(np.float32), masks


def build_policy(confidence: float, localization_confidence: float, require_presence: bool) -> SelectionPolicy:
    """Validate the two thresholds before they reach a detection call."""
    for name, value in (("confidence", confidence), ("localization_confidence", localization_confidence)):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1, got {value!r}")
    return SelectionPolicy(
        confidence=float(confidence),
        localization_confidence=float(localization_confidence),
        require_presence=require_presence,
    )
