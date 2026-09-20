"""pipeline_results.py  --  what a single-image pipeline run produces.

The three types the orchestrator fills in and everything downstream reads:
the loaded source, one output layer, and the run as a whole. They are data,
with no pipeline behaviour attached, and several modules import them without
needing the orchestrator itself.

Separated from orchestrator.py, which had grown past this project's 800-line
limit. core.orchestrator re-exports all three, so existing imports keep
working.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field


def collapse_repeats(messages: list[str]) -> list[str]:
    """Collapse identical messages to one line carrying a count.

    A stage that rejects many candidates emits the same sentence once per
    rejection: a real run produced "Empty or tiny mask for 'button'" ten
    times for a single image, and four copies of the instance-ceiling
    warning for another. Repetition adds nothing a count does not, and it
    buries the warnings that actually differ.

    Order is the order each message FIRST appeared, and a message seen once
    is returned untouched.
    """
    counts: dict[str, int] = {}
    for message in messages:
        counts[message] = counts.get(message, 0) + 1
    return [
        message if count == 1 else f"{message} (x{count})"
        for message, count in counts.items()
    ]


class _SourceImage(NamedTuple):
    """What loading an image produces, carried between pipeline stages.

    A NamedTuple rather than four loose locals threaded through five
    signatures: the pixels, the alpha and the ICC profile belong together,
    and nothing downstream may replace one of them.
    """

    rgb: NDArray[np.uint8]
    alpha: NDArray[np.uint8]
    icc: bytes | None
    detection: NDArray[np.uint8]
    height: int
    width: int


class LayerResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    layer_id: str = ""
    parent_id: str | None = None
    confidence: float | None = None
    source: str = ""
    alpha_path: str | None = None
    mask: NDArray[np.uint8] | None = Field(default=None, exclude=True)
    alpha: NDArray[np.uint8] | None = Field(default=None, exclude=True)
    preview_opacity: float = Field(default=1.0, exclude=True)
    label: str
    role: str
    parent_label: str | None = None
    bbox: tuple[int, int, int, int]
    svg_data: str = ""
    dx: int = 0
    dy: int = 0


class PipelineResult(BaseModel):
    image_path: str
    width: int
    height: int
    layers: list[LayerResult] = []
    svg_path: str | None = None
    tiff_path: str | None = None
    all_objects_tiff_path: str | None = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    tiff_files: list[str] = Field(default_factory=list)
