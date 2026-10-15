from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from utils.array_types import as_intp, as_uint8


def boolean_subtract(
    parent_mask: NDArray[np.uint8],
    child_mask: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Subtract child mask from parent, returning the body mask."""
    return as_uint8(cv2.bitwise_and(parent_mask, cv2.bitwise_not(child_mask)))


def boolean_union(
    mask_a: NDArray[np.uint8],
    mask_b: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Combine two masks via bitwise OR."""
    return as_uint8(cv2.bitwise_or(mask_a, mask_b))


def boolean_intersect(
    mask_a: NDArray[np.uint8],
    mask_b: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Intersect two masks via bitwise AND."""
    return as_uint8(cv2.bitwise_and(mask_a, mask_b))


def refine_mask(
    mask: NDArray[np.uint8],
    bilateral_d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
    morph_kernel_size: int = 3,
    min_contour_area: int = 64,
) -> NDArray[np.uint8]:
    """Remove isolated components by pixel area without changing topology.

    Legacy smoothing arguments remain accepted. Binary masks must not be
    blurred or closed implicitly: those operations destroy holes and wires.
    """
    if mask.ndim != 2:
        raise ValueError("Expected a two-dimensional mask")
    binary = (mask > 127).astype(np.uint8)
    if min_contour_area <= 1:
        return binary * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(count, dtype=np.uint8)
    keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= min_contour_area).astype(np.uint8) * 255
    return keep[as_intp(labels)]


#: Enclosed holes smaller than this share of the object's area are filled by
#: ``solidify_mask``. A counter in a letter is far larger relative to its glyph.
SOLID_HOLE_FRACTION = 0.01
#: Diameter of the closing kernel, as a share of the image's shorter side.
SOLID_NOTCH_FRACTION = 0.018
#: Pockets no deeper than this share of the object's shorter side are filled
#: to their chord. The iPhone front's bite was 4.4%; a U or C is far deeper.
SOLID_POCKET_FRACTION = 0.06


def _fill_small_holes(binary: NDArray[np.uint8], max_area: float) -> NDArray[np.uint8]:
    """Fill background regions enclosed by the object and smaller than max_area."""
    padded = np.pad(binary, 1)
    outside = padded.copy()
    cv2.floodFill(outside, None, (0, 0), 1)
    holes = (1 - outside[1:-1, 1:-1]).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=4)
    small = np.zeros(count, dtype=bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] < max_area
    return np.where(small[as_intp(labels)], 1, binary).astype(np.uint8)


def _fill_shallow_pockets(binary: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Fill each outline pocket shallow relative to its object, up to its chord."""
    filled = binary.copy()
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        if len(contour) < 4:
            continue
        defects = cv2.convexityDefects(contour, cv2.convexHull(contour, returnPoints=False))
        if defects is None:
            continue
        max_depth = min(cv2.boundingRect(contour)[2:]) * SOLID_POCKET_FRACTION
        for start, end, _, fixed_depth in defects.reshape(-1, 4):
            if fixed_depth / 256.0 > max_depth:
                continue
            pocket = (
                contour[start : end + 1]
                if start <= end
                else np.concatenate([contour[start:], contour[: end + 1]])
            )
            cv2.fillPoly(filled, [pocket], 1)
    return filled


def solidify_mask(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Close edge notches, fill shallow pockets and small enclosed holes.

    Opt-in only: ``refine_mask`` preserves topology on purpose, because real
    holes and thin wires matter. This is for solid objects whose mask has
    specks and bites where the surface reflected light.
    """
    if mask.ndim != 2:
        raise ValueError("Expected a two-dimensional mask")
    binary = (mask > 127).astype(np.uint8)
    area = int(binary.sum())
    if area == 0:
        return binary * 255
    diameter = max(3, round(min(mask.shape) * SOLID_NOTCH_FRACTION) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    closed = as_uint8(cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel))
    pocketless = _fill_shallow_pockets(closed)
    return _fill_small_holes(pocketless, area * SOLID_HOLE_FRACTION) * 255


def edge_refine(
    mask: NDArray[np.uint8],
    iterations: int = 1,
) -> NDArray[np.uint8]:
    """Apply morphological erosion for edge refinement (per-layer control)."""
    if iterations <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return as_uint8(cv2.erode(mask, kernel, iterations=iterations))


def mask_bbox(mask: NDArray[np.uint8]) -> tuple[int, int, int, int]:
    """Get bounding box (y0, x0, y1, x1) of non-zero pixels."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return (0, 0, mask.shape[0], mask.shape[1])
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return (int(y0), int(x0), int(y1) + 1, int(x1) + 1)


def mask_coverage(mask: NDArray[np.uint8]) -> float:
    """Return percentage of non-zero pixels in mask."""
    return float((mask > 0).sum() / mask.size * 100)
