"""BiRefNet as an edge refiner for one object's mask.

BiRefNet segments the salient object in an image, without prompts. Run on a
crop around one SAM mask, it draws that object's edge more precisely, but it
may also take in whatever else is salient in the crop. So only the edge moves:
inside a band around SAM's edge the refiner decides, everywhere else SAM does.

Measured on ten EveryMac cut-outs with their alpha as ground truth, this band
merge raised mean Boundary IoU from 0.387 to 0.455 (documents/Mask-Accuracy-
Benchmark.md in SkiagrafiaMac).

BiRefNet's code uses torchvision's deform_conv2d, which has no MPS kernel.
The GPU is used only when PYTORCH_ENABLE_MPS_FALLBACK=1 lets that one op fall
back to the CPU; otherwise the whole model runs on the CPU.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

#: Context around the object, as a share of its longer side.
CROP_MARGIN = 0.10
#: Half-width of the band in which the refiner decides, as a share of the
#: object's shorter side; never under BAND_MIN_PIXELS.
BAND_FRACTION = 0.03
BAND_MIN_PIXELS = 2
#: BiRefNet's training resolution.
INPUT_SIZE = 1024
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def merge_in_band(mask: NDArray[np.uint8], probability: NDArray[np.float32]) -> NDArray[np.uint8]:
    """SAM's mask with its edge band taken from the refiner's probability."""
    binary = (mask > 0).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(binary)
    if w == 0 or h == 0:
        return binary
    half = max(BAND_MIN_PIXELS, round(BAND_FRACTION * min(w, h)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * half + 1, 2 * half + 1))
    band = cv2.dilate(binary, kernel) & (1 - cv2.erode(binary, kernel))
    refined = (probability > 0.5).astype(np.uint8)
    return np.where(band == 1, refined, binary).astype(np.uint8)


class BiRefNetEdgeRefiner:
    """Loads BiRefNet from a local directory on first use and keeps it."""

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = model_dir
        self._model: Any = None
        self._device = "cpu"

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        if not self._model_dir.is_dir():
            raise FileNotFoundError(f"BiRefNet weights not found at {self._model_dir}.")
        import torch
        from transformers import AutoModelForImageSegmentation

        # Local files only: the directory's own modeling code, no download.
        model = AutoModelForImageSegmentation.from_pretrained(
            str(self._model_dir), trust_remote_code=True, local_files_only=True
        )
        fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
        self._device = "mps" if fallback and torch.backends.mps.is_available() else "cpu"
        self._model = model.float().to(self._device).eval()
        return self._model

    def _probability(self, rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
        import torch

        model = self._load()
        height, width = rgb.shape[:2]
        resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
        normalised = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD
        batch = torch.from_numpy(normalised.transpose(2, 0, 1))[None].to(self._device)
        with torch.no_grad():
            logits = model(batch)[-1]
        probability = logits.sigmoid()[0, 0].float().cpu().numpy()
        return cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)

    def refine(self, image: NDArray[np.uint8], mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """A 0/255 mask whose edge BiRefNet has redrawn on the object's crop."""
        binary = (mask > 127).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(binary)
        if w == 0 or h == 0:
            return mask
        margin = int(CROP_MARGIN * max(w, h))
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(image.shape[1], x + w + margin), min(image.shape[0], y + h + margin)
        probability = np.zeros(binary.shape, dtype=np.float32)
        probability[y0:y1, x0:x1] = self._probability(image[y0:y1, x0:x1])
        return merge_in_band(binary, probability) * 255
