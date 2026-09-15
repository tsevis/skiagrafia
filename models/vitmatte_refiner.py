from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from utils.mps_utils import DEVICE

logger = logging.getLogger(__name__)

class VitMatteRefiner:
    """VitMatte ViT-B alpha matting for high-quality mask edges.

    Loaded once and kept resident in memory.

    Parameters
    ----------
    model_dir : Path, optional
        Path to vitmatte-base-composition-1k directory. Falls back to
        the backward-compat model_path() shim when None.
    """

    def __init__(self, model_dir: Path | None = None, quality: str = "balanced") -> None:
        self._model_dir = model_dir
        self._model: object | None = None
        self._processor: object | None = None
        self._max_side = 1536
        self._quality = quality

    def _load(self) -> None:
        """Load VitMatte model weights from local directory (lazy, once)."""
        if self._model is not None:
            return
        try:
            from transformers import VitMatteForImageMatting, VitMatteImageProcessor

            # Resolve from constructor param or backward-compat shim
            if self._model_dir is not None:
                vitmatte_dir = self._model_dir
            else:
                from utils.model_manager import model_path
                vitmatte_dir = model_path("vitmatte-base-composition-1k")

            if not vitmatte_dir.is_dir():
                raise FileNotFoundError(
                    f"VitMatte weights not found at {vitmatte_dir}. "
                    "Download with: huggingface-cli download hustvl/vitmatte-base-composition-1k"
                )

            self._processor = VitMatteImageProcessor.from_pretrained(
                str(vitmatte_dir), local_files_only=True
            )
            self._model = VitMatteForImageMatting.from_pretrained(
                str(vitmatte_dir), local_files_only=True
            )
            self._model.to(DEVICE)
            self._model.eval()
            logger.info("VitMatte loaded on %s from %s", DEVICE, vitmatte_dir)
        except ImportError:
            logger.warning(
                "transformers VitMatte not available — alpha matting disabled."
            )
            raise
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            logger.error("Failed to load VitMatte", exc_info=True)
            raise

    @staticmethod
    def _create_trimap(
        mask: NDArray[np.uint8],
        erosion_size: int = 10,
        dilation_size: int = 20,
    ) -> NDArray[np.uint8]:
        """Create trimap from binary mask: definite fg, definite bg, unknown."""
        kernel_erode = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erosion_size, erosion_size)
        )
        kernel_dilate = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_size, dilation_size)
        )

        fg = cv2.erode(mask, kernel_erode, iterations=1)
        bg_inv = cv2.dilate(mask, kernel_dilate, iterations=1)

        trimap = np.full(mask.shape, 128, dtype=np.uint8)  # unknown
        trimap[fg > 127] = 255  # definite foreground
        trimap[bg_inv < 127] = 0  # definite background
        return trimap

    def _infer(self, image, trimap):
        from PIL import Image

        self._load()
        inputs = self._processor(images=Image.fromarray(image), trimaps=Image.fromarray(trimap), return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.inference_mode():
            alpha = self._model(**inputs).alphas.squeeze().cpu().numpy()
        h, w = image.shape[:2]
        if alpha.shape[0] >= h and alpha.shape[1] >= w:
            alpha = alpha[:h, :w]  # processor pads on right/bottom
        elif alpha.shape != (h, w):
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.nan_to_num(alpha, nan=0.0).clip(0, 1)

    def predict(self, image, mask):
        """Matte the object ROI; detailed mode refines boundary tiles at native resolution.

        Known foreground/background are enforced after inference and resizing.
        Only unknown boundary pixels can change. Empty masks do not load a model.
        """
        if mask.shape != image.shape[:2] or mask.ndim != 2:
            raise ValueError("Mask and image dimensions must match")
        mask = (mask > 127).astype(np.uint8) * 255
        if not mask.any() or np.all(mask):
            return mask
        h, w = mask.shape
        ys, xs = np.nonzero(mask)
        radius = int(np.clip(round(min(xs.max()-xs.min()+1, ys.max()-ys.min()+1) * 0.012), 2, 16))
        trimap = self._create_trimap(mask, 2*radius+1, 2*radius+1)
        pad = max(32, radius * 3)
        x0, x1 = max(0, xs.min()-pad), min(w, xs.max()+pad+1)
        y0, y1 = max(0, ys.min()-pad), min(h, ys.max()+pad+1)
        crop = image[y0:y1, x0:x1]
        tri = trimap[y0:y1, x0:x1]
        ch, cw = tri.shape
        if self._quality == "detailed" and max(ch, cw) > self._max_side:
            # Halo provides context; central regions cover each pixel once, so
            # no averaging can dilute the known opaque/transparent regions.
            alpha = mask[y0:y1, x0:x1].astype(np.float32) / 255
            tile, halo = 768, 96
            for top in range(0, ch, tile):
                for left in range(0, cw, tile):
                    bottom, right = min(ch, top+tile), min(cw, left+tile)
                    if not np.any(tri[top:bottom, left:right] == 128):
                        continue
                    ty, tx = max(0, top-halo), max(0, left-halo)
                    by, rx = min(ch, bottom+halo), min(cw, right+halo)
                    predicted = self._infer(crop[ty:by, tx:rx], tri[ty:by, tx:rx])
                    alpha[top:bottom, left:right] = predicted[top-ty:bottom-ty, left-tx:right-tx]
        else:
            scale = min(1.0, self._max_side / max(ch, cw))
            if scale < 1:
                size = (max(1, round(cw*scale)), max(1, round(ch*scale)))
                predicted = self._infer(cv2.resize(crop, size, interpolation=cv2.INTER_AREA),
                                        cv2.resize(tri, size, interpolation=cv2.INTER_NEAREST))
                alpha = cv2.resize(predicted, (cw, ch), interpolation=cv2.INTER_LINEAR)
            else:
                alpha = self._infer(crop, tri)
        output = np.zeros((h, w), dtype=np.uint8)
        output[y0:y1, x0:x1] = np.rint(alpha * 255).clip(0, 255).astype(np.uint8)
        output[trimap == 0] = 0
        output[trimap == 255] = 255
        return output

    def predict_rgba(
        self,
        image: NDArray[np.uint8],
        mask: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """Predict alpha matte and return 4-channel RGBA image."""
        alpha = self.predict(image, mask)
        rgba = np.dstack([image, alpha])
        return rgba
