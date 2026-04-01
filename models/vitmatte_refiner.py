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

    def __init__(self, model_dir: Path | None = None) -> None:
        self._model_dir = model_dir
        self._model: object | None = None
        self._processor: object | None = None
        self._max_side = 1536

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
        except Exception:
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

    def predict(
        self,
        image: NDArray[np.uint8],
        mask: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """Predict alpha matte from image and coarse binary mask.

        Args:
            image: RGB image (H, W, 3) uint8.
            mask: Binary mask (H, W) uint8, 0 or 255.

        Returns:
            Alpha matte (H, W) uint8, 0-255 with soft edges.
        """
        self._load()
        from PIL import Image

        orig_h, orig_w = image.shape[:2]
        working_image = image
        working_mask = mask
        scale = min(1.0, self._max_side / max(orig_h, orig_w))
        if scale < 1.0:
            target_size = (
                max(1, int(orig_w * scale)),
                max(1, int(orig_h * scale)),
            )
            working_image = cv2.resize(
                image,
                target_size,
                interpolation=cv2.INTER_AREA,
            )
            working_mask = cv2.resize(
                mask,
                target_size,
                interpolation=cv2.INTER_NEAREST,
            )
            logger.info(
                "VitMatte downscaling from %sx%s to %sx%s for memory safety",
                orig_w,
                orig_h,
                target_size[0],
                target_size[1],
            )

        trimap = self._create_trimap(working_mask)

        pil_image = Image.fromarray(working_image)
        pil_trimap = Image.fromarray(trimap)

        inputs = self._processor(
            images=pil_image, trimaps=pil_trimap, return_tensors="pt"
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        alpha = outputs.alphas.squeeze().cpu().numpy()
        alpha_np = (alpha * 255).clip(0, 255).astype(np.uint8)

        # VitMatte processor may pad/resize — crop back to original dimensions.
        work_h, work_w = working_image.shape[:2]
        if alpha_np.shape != (work_h, work_w):
            alpha_np = cv2.resize(
                alpha_np[:work_h, :work_w] if alpha_np.shape[0] >= work_h else alpha_np,
                (work_w, work_h),
                interpolation=cv2.INTER_LINEAR,
            )
        if scale < 1.0:
            alpha_np = cv2.resize(
                alpha_np,
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR,
            )

        logger.info(
            "VitMatte: alpha range [%d, %d], shape %s",
            alpha_np.min(),
            alpha_np.max(),
            alpha_np.shape,
        )
        return alpha_np

    def predict_rgba(
        self,
        image: NDArray[np.uint8],
        mask: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """Predict alpha matte and return 4-channel RGBA image."""
        alpha = self.predict(image, mask)
        rgba = np.dstack([image, alpha])
        return rgba
