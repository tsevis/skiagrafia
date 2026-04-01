from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def bilateral_smooth(
    image: NDArray[np.uint8],
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> NDArray[np.uint8]:
    """Apply bilateral filter for edge-preserving smoothing."""
    return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def kmeans_quantize(
    image: NDArray[np.uint8],
    k: int = 8,
    max_iter: int = 10,
    epsilon: float = 1.0,
) -> NDArray[np.uint8]:
    """Reduce image to k colours using K-Means clustering."""
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, attempts=3, flags=cv2.KMEANS_PP_CENTERS
    )

    centers = centers.astype(np.uint8)
    quantized = centers[labels.flatten()].reshape(h, w, 3)
    return quantized


def apply_mask_to_image(
    image: NDArray[np.uint8],
    mask: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Apply binary mask to image, zeroing masked-out regions."""
    if len(image.shape) == 3:
        mask_3ch = np.stack([mask] * 3, axis=-1)
        return cv2.bitwise_and(image, mask_3ch)
    return cv2.bitwise_and(image, mask)
