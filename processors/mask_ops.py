from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def boolean_subtract(
    parent_mask: NDArray[np.uint8],
    child_mask: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Subtract child mask from parent, returning the body mask."""
    return cv2.bitwise_and(parent_mask, cv2.bitwise_not(child_mask))


def boolean_union(
    mask_a: NDArray[np.uint8],
    mask_b: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Combine two masks via bitwise OR."""
    return cv2.bitwise_or(mask_a, mask_b)


def boolean_intersect(
    mask_a: NDArray[np.uint8],
    mask_b: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Intersect two masks via bitwise AND."""
    return cv2.bitwise_and(mask_a, mask_b)


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
    return keep[labels]


def edge_refine(
    mask: NDArray[np.uint8],
    iterations: int = 1,
) -> NDArray[np.uint8]:
    """Apply morphological erosion for edge refinement (per-layer control)."""
    if iterations <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.erode(mask, kernel, iterations=iterations)


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
