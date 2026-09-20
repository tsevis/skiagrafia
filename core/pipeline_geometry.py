"""pipeline_geometry.py  --  pure geometry and label helpers for the pipeline.

Masks, bounding boxes and the two label sanitisers the orchestrator needs.
Every function here is pure: it takes arrays and tuples and returns a value,
touching no model, no file and no pipeline state. That is why they are worth
having apart -- they can be read, and reasoned about, without the pipeline.

They were private names inside orchestrator.py, which had grown past this
project's 800-line limit, and one of them was already being imported across
modules by its underscored name.
"""
from __future__ import annotations

import itertools
import re

import numpy as np
from numpy.typing import NDArray

# How far a parent box is grown before asking whether a child falls inside it.
BBOX_EXPAND_RATIO = 0.30
# Max chars of a label used in output filenames; macOS caps a name at 255
# bytes and a VLM can return a very long label.
MAX_LABEL_FILENAME_LEN = 60


def safe_filename_label(label: str) -> str:
    """Truncate and sanitize a label for use in output filenames.

    Moondream can return extremely long child labels (e.g. numbered
    lists of 38 items).  macOS enforces a 255-byte filename limit.
    """
    # Replace path-unsafe characters
    safe = re.sub(r"[^A-Za-z0-9._ -]+", "_", label).strip(". ")
    safe = safe.replace("/", "_").replace("\\", "_").replace(":", "_")
    if not safe:
        safe = "layer"
    if len(safe) > MAX_LABEL_FILENAME_LEN:
        safe = safe[:MAX_LABEL_FILENAME_LEN].rstrip(". ")
    return safe

def mask_iou(a: NDArray[np.uint8], b: NDArray[np.uint8]) -> float:
    a_bool, b_bool = a > 127, b > 127
    inter = np.logical_and(a_bool, b_bool).sum()
    union = np.logical_or(a_bool, b_bool).sum()
    return float(inter / union) if union else 0.0

def mask_containment(a: NDArray[np.uint8], b: NDArray[np.uint8]) -> float:
    """Fraction of the *smaller* mask that is contained in the larger one.

    Returns a value in [0, 1].  A high value means one mask is mostly
    inside the other — strong evidence they represent the same object even
    when IoU is low (because one mask is much larger).
    """
    a_bool, b_bool = a > 127, b > 127
    a_area = int(a_bool.sum())
    b_area = int(b_bool.sum())
    if a_area == 0 or b_area == 0:
        return 0.0
    inter = int(np.logical_and(a_bool, b_bool).sum())
    smaller = min(a_area, b_area)
    return float(inter / smaller)

def bbox_overlaps(
    child_bbox: tuple[int, int, int, int],
    parent_bbox: tuple[int, int, int, int],
    img_h: int,
    img_w: int,
    expand: float = BBOX_EXPAND_RATIO,
) -> bool:
    px0, py0, px1, py1 = parent_bbox
    pw, ph = px1 - px0, py1 - py0
    ex0 = max(0, int(px0 - pw * expand))
    ey0 = max(0, int(py0 - ph * expand))
    ex1 = min(img_w, int(px1 + pw * expand))
    ey1 = min(img_h, int(py1 + ph * expand))
    cx0, cy0, cx1, cy1 = child_bbox
    return cx0 < ex1 and cx1 > ex0 and cy0 < ey1 and cy1 > ey0

def bbox_area(bbox: tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = bbox
    return max(0, x1 - x0) * max(0, y1 - y0)

def bbox_iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    """Intersection-over-union of two bounding boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = bbox_area(a)
    area_b = bbox_area(b)
    union = area_a + area_b - inter
    return float(inter / union) if union else 0.0

def normalized_bbox_to_image(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Convert a 0..1000 VLM box to the current image dimensions."""
    x0, y0, x1, y1 = bbox
    return (
        round(x0 * width / 1000),
        round(y0 * height / 1000),
        round(x1 * width / 1000),
        round(y1 * height / 1000),
    )

def rectangle_union_area(rectangles: list[tuple[int, int, int, int]]) -> int:
    """Return exact union area for a small list of axis-aligned rectangles."""
    rectangles = [rectangle for rectangle in rectangles if bbox_area(rectangle)]
    if not rectangles:
        return 0
    x_edges = sorted({edge for rectangle in rectangles for edge in (rectangle[0], rectangle[2])})
    area = 0
    for left, right in itertools.pairwise(x_edges):
        if right <= left:
            continue
        spans = sorted(
            (rectangle[1], rectangle[3])
            for rectangle in rectangles
            if rectangle[0] < right and rectangle[2] > left
        )
        covered_y = 0
        # ONE optional pair, not two optional ints. The bounds are only ever
        # meaningful together, and as separate names nothing said so: the
        # `start > current_end` branch is reachable only because the other
        # name happens to have been set on the same line. A tuple makes that
        # invariant structural instead of a convention, and each step rebinds
        # rather than mutating.
        current: tuple[int, int] | None = None
        for start, end in spans:
            if current is None:
                current = (start, end)
            elif start > current[1]:
                covered_y += current[1] - current[0]
                current = (start, end)
            else:
                current = (current[0], max(current[1], end))
        if current is not None:
            covered_y += current[1] - current[0]
        area += (right - left) * covered_y
    return area

def glyph_layer_label(category: str, glyph: str) -> str:
    """Make a readable output label without trusting a VLM for filenames."""
    words = set(re.findall(r"[^\W\d_]+|\d+", category.casefold(), flags=re.UNICODE))
    if words & {"letter", "letters"}:
        kind = "letter"
    elif words & {"digit", "digits", "numeral", "numerals", "number", "numbers"}:
        kind = "digit"
    else:
        kind = "glyph"
    return f"{kind} {glyph}"

def crop_to_bbox(
    image: NDArray[np.uint8],
    bbox: tuple[int, int, int, int],
    padding: int = 8,
) -> tuple[NDArray[np.uint8], int, int]:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(w, x1 + padding)
    y1 = min(h, y1 + padding)
    return image[y0:y1, x0:x1], x0, y0

def clip_mask_to_bbox(
    mask: NDArray[np.uint8],
    bbox: tuple[int, int, int, int],
) -> NDArray[np.uint8]:
    """Zero out mask pixels outside the bounding box."""
    x0, y0, x1, y1 = bbox
    h, w = mask.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    clipped = np.zeros_like(mask)
    clipped[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return clipped
