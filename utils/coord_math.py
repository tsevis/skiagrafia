from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def remap_mask(
    mask_local: NDArray[np.uint8],
    crop_origin: tuple[int, int],
    canvas_shape: tuple[int, int],
) -> NDArray[np.uint8]:
    """Place a cropped mask back onto a full-canvas-sized mask.

    Args:
        mask_local: Binary mask from the cropped region (H_crop, W_crop).
        crop_origin: (y0, x0) — top-left corner of the crop in full canvas coords.
        canvas_shape: (H_full, W_full) — dimensions of the full canvas.

    Returns:
        Full-canvas binary mask with the local mask placed at crop_origin.
    """
    y0, x0 = crop_origin
    h, w = mask_local.shape[:2]
    full = np.zeros(canvas_shape[:2], dtype=np.uint8)
    # Clip to canvas bounds
    y_end = min(y0 + h, canvas_shape[0])
    x_end = min(x0 + w, canvas_shape[1])
    y_start = max(y0, 0)
    x_start = max(x0, 0)
    local_y = y_start - y0
    local_x = x_start - x0
    full[y_start:y_end, x_start:x_end] = mask_local[
        local_y : local_y + (y_end - y_start),
        local_x : local_x + (x_end - x_start),
    ]
    return full


def tight_bbox(
    mask: NDArray[np.uint8],
    padding: int = 15,
) -> tuple[int, int, int, int]:
    """Compute tight bounding box from a binary mask with optional padding.

    Returns:
        (y0, x0, y1, x1) with padding, clipped to mask bounds.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return (0, 0, mask.shape[0], mask.shape[1])
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    y0 = max(0, int(y0) - padding)
    x0 = max(0, int(x0) - padding)
    y1 = min(mask.shape[0], int(y1) + 1 + padding)
    x1 = min(mask.shape[1], int(x1) + 1 + padding)
    return (y0, x0, y1, x1)


def crop_with_padding(
    image: NDArray[np.uint8],
    bbox: tuple[int, int, int, int],
    padding: int = 15,
) -> tuple[NDArray[np.uint8], tuple[int, int]]:
    """Crop image region with padding. Returns (crop, (y0, x0))."""
    y0, x0, y1, x1 = bbox
    h, w = image.shape[:2]
    y0p = max(0, y0 - padding)
    x0p = max(0, x0 - padding)
    y1p = min(h, y1 + padding)
    x1p = min(w, x1 + padding)
    return image[y0p:y1p, x0p:x1p].copy(), (y0p, x0p)
