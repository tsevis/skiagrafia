from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from utils.model_manager import model_path
from utils.mps_utils import DEVICE

logger = logging.getLogger(__name__)

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


def _patch_onnx_ml_dtypes() -> None:
    """Patch missing ml_dtypes symbols used by ONNX during torchvision import.

    GroundingDINO inference does not need these low-precision ONNX dtypes at runtime,
    but newer ONNX builds may import them unconditionally through torchvision.
    """
    try:
        import ml_dtypes
    except Exception:
        return

    fallbacks = {
        "float4_e2m1fn": getattr(ml_dtypes, "float8_e4m3fn", None),
        "float8_e8m0fnu": getattr(ml_dtypes, "float8_e5m2", None),
    }
    for name, fallback in fallbacks.items():
        if not hasattr(ml_dtypes, name) and fallback is not None:
            setattr(ml_dtypes, name, fallback)
            logger.info("Patched ml_dtypes.%s for ONNX compatibility", name)


_patch_onnx_ml_dtypes()


def _patch_bert_head_mask() -> None:
    """Restore `get_head_mask` on `BertModel` for GroundingDINO compatibility.

    transformers >=5.0 removed `get_head_mask` from `BertModel`, but
    GroundingDINO's `BertModelWarper` reads it as an attribute during init.
    We reinstate the historical implementation.
    """
    try:
        from transformers.models.bert.modeling_bert import BertModel
    except Exception:
        return

    if hasattr(BertModel, "get_head_mask"):
        return

    def get_head_mask(
        self,
        head_mask,
        num_hidden_layers: int,
        is_attention_chunked: bool = False,
    ):
        if head_mask is None:
            return [None] * num_hidden_layers
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        if is_attention_chunked:
            head_mask = head_mask.unsqueeze(-1)
        return head_mask

    BertModel.get_head_mask = get_head_mask  # type: ignore[attr-defined]
    logger.info("Patched BertModel.get_head_mask for GroundingDINO compatibility")


def _patch_get_extended_attention_mask() -> None:
    """Accept legacy `device` positional in `get_extended_attention_mask`.

    transformers >=5.0 dropped the `device` parameter, but GroundingDINO's
    `BertModelWarper.forward` still calls
    `self.get_extended_attention_mask(mask, shape, device)`. We wrap the
    method to detect a `torch.device` in that slot and discard it.
    """
    try:
        from transformers.modeling_utils import ModuleUtilsMixin
    except Exception:
        return

    original = ModuleUtilsMixin.get_extended_attention_mask
    if getattr(original, "_skiagrafia_patched", False):
        return

    def get_extended_attention_mask(
        self,
        attention_mask,
        input_shape,
        device=None,
        dtype=None,
    ):
        if isinstance(device, torch.dtype) and dtype is None:
            dtype, device = device, None
        elif not isinstance(device, torch.device) and device is not None and dtype is None:
            dtype = device
        return original(self, attention_mask, input_shape, dtype=dtype)

    get_extended_attention_mask._skiagrafia_patched = True  # type: ignore[attr-defined]
    ModuleUtilsMixin.get_extended_attention_mask = get_extended_attention_mask  # type: ignore[assignment]
    logger.info(
        "Patched ModuleUtilsMixin.get_extended_attention_mask for GroundingDINO compatibility"
    )


_patch_bert_head_mask()
_patch_get_extended_attention_mask()

# Ambiguous labels that GroundingDINO often fails to detect.
# Maps short/ambiguous label → list of more specific synonyms to try.
_LABEL_SYNONYMS: dict[str, list[str]] = {
    "mouse": ["computer mouse", "mouse pad and mouse"],
    "monitor": ["computer monitor", "display screen"],
    "speaker": ["computer speaker", "loudspeaker"],
    "tower": ["computer tower", "desktop tower", "PC case"],
    "cable": ["power cable", "USB cable"],
}

def _ensure_gsam_on_path(gsam_root: Path) -> None:
    """Ensure Grounded-SAM-2 root is on sys.path for internal imports."""
    if gsam_root.is_dir() and str(gsam_root) not in sys.path:
        sys.path.insert(0, str(gsam_root))


class DetectionResult(BaseModel):
    """Bounding box detection from GroundingDINO."""

    label: str
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1)
    confidence: float


class SegmentationResult(BaseModel):
    """Segmentation mask from SAM 2.1 HQ."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    label: str
    bbox: tuple[int, int, int, int]
    mask_shape: tuple[int, int]


class GroundedSAM:
    """GroundingDINO + SAM 2.1 HQ wrapper for text-guided segmentation.

    Models are loaded once and kept resident in memory for performance.

    Parameters
    ----------
    dino_weights : Path, optional
        Path to groundingdino_swint_ogc.pth. Falls back to model_path() shim.
    sam_weights : Path, optional
        Path to sam2.1_hiera_large.pt. Falls back to model_path() shim.
    gsam_root : Path, optional
        Path to Grounded-SAM-2 source root. Falls back to model_path() shim.
    """

    def __init__(
        self,
        dino_weights: Path | None = None,
        sam_weights: Path | None = None,
        gsam_root: Path | None = None,
    ) -> None:
        self._dino_weights = dino_weights
        self._sam_weights = sam_weights
        self._gsam_root = gsam_root
        self._dino_model: object | None = None
        self._sam_predictor: object | None = None
        self._masks_cache: dict[str, NDArray[np.uint8]] = {}

    def _load_dino(self) -> None:
        """Load GroundingDINO model weights (lazy, once)."""
        if self._dino_model is not None:
            return
        try:
            # Resolve paths from constructor params or backward-compat shim
            gsam_root = self._gsam_root or model_path("groundingdino_swint_ogc.pth").parent.parent
            _ensure_gsam_on_path(gsam_root)

            from grounding_dino.groundingdino.util.inference import load_model

            weights = self._dino_weights or model_path("groundingdino_swint_ogc.pth")
            config_path = (
                gsam_root
                / "grounding_dino"
                / "groundingdino"
                / "config"
                / "GroundingDINO_SwinT_OGC.py"
            )
            self._dino_model = load_model(
                str(config_path), str(weights), device=str(DEVICE)
            )
            logger.info("GroundingDINO loaded on %s", DEVICE)
        except Exception:
            logger.error("Failed to load GroundingDINO", exc_info=True)
            raise

    def _load_sam(self) -> None:
        """Load SAM 2.1 predictor (lazy, once)."""
        if self._sam_predictor is not None:
            return
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            weights = self._sam_weights or model_path("sam2.1_hiera_large.pt")
            config = "configs/sam2.1/sam2.1_hiera_l.yaml"
            sam = build_sam2(
                config_file=config,
                ckpt_path=str(weights),
                device=str(DEVICE),
            )
            self._sam_predictor = SAM2ImagePredictor(sam)
            logger.info("SAM 2.1 loaded on %s", DEVICE)
        except Exception:
            logger.error("Failed to load SAM 2.1", exc_info=True)
            raise

    def detect_box(
        self,
        image: NDArray[np.uint8],
        label: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        skip_synonyms: bool = False,
    ) -> DetectionResult | None:
        """Run GroundingDINO to get bounding box for a text label.

        Parameters
        ----------
        skip_synonyms : bool
            If True, skip synonym retry loop (faster for preview use).
        """
        self._load_dino()

        from grounding_dino.groundingdino.util.inference import predict
        import grounding_dino.groundingdino.datasets.transforms as T

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        from PIL import Image

        pil_image = Image.fromarray(image)
        image_transformed, _ = transform(pil_image, None)

        # GroundingDINO expects period-terminated captions for proper grounding
        caption = label if label.endswith(".") else f"{label}."

        boxes, logits, phrases = predict(
            model=self._dino_model,
            image=image_transformed,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=str(DEVICE),
        )

        if len(boxes) == 0 and not skip_synonyms:
            # Retry with synonym labels for ambiguous terms
            synonyms = _LABEL_SYNONYMS.get(label.lower(), [])
            for synonym in synonyms:
                syn_caption = f"{synonym}."
                logger.info("Retrying detection with synonym '%s' for '%s'", synonym, label)
                boxes, logits, phrases = predict(
                    model=self._dino_model,
                    image=image_transformed,
                    caption=syn_caption,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    device=str(DEVICE),
                )
                if len(boxes) > 0:
                    break

        if len(boxes) == 0:
            logger.warning("No detection for label '%s'", label)
            return None

        # Take highest confidence detection
        best_idx = logits.argmax().item()
        box = boxes[best_idx]
        h, w = image.shape[:2]

        # Convert from normalized [cx, cy, w, h] to absolute [x0, y0, x1, y1]
        cx, cy, bw, bh = box.tolist()
        x0 = int((cx - bw / 2) * w)
        y0 = int((cy - bh / 2) * h)
        x1 = int((cx + bw / 2) * w)
        y1 = int((cy + bh / 2) * h)

        return DetectionResult(
            label=label,
            bbox=(x0, y0, x1, y1),
            confidence=float(logits[best_idx]),
        )

    def segment(
        self,
        image: NDArray[np.uint8],
        bbox: tuple[int, int, int, int],
        label: str = "",
        prefer_full_box: bool = False,
    ) -> NDArray[np.uint8]:
        """Run SAM 2.1 HQ segmentation within a bounding box.

        Parameters
        ----------
        prefer_full_box : bool
            When True (used for manual bounding boxes), request multiple
            mask candidates from SAM and pick the one with the highest
            coverage inside the bbox.  This prevents SAM from segmenting
            only a sub-object (e.g. the screen content instead of the
            whole monitor).

        Returns binary mask (0/255) at image resolution.
        """
        self._load_sam()

        self._sam_predictor.set_image(image)

        box_array = np.array(bbox, dtype=np.float32)

        if prefer_full_box:
            masks, scores, _ = self._sam_predictor.predict(
                box=box_array,
                multimask_output=True,
            )
            # Pick the mask with the highest coverage inside the bbox
            best_idx = self._best_mask_for_bbox(masks, bbox)
            mask = (masks[best_idx] > 0).astype(np.uint8) * 255
        else:
            masks, scores, _ = self._sam_predictor.predict(
                box=box_array,
                multimask_output=False,
            )
            mask = (masks[0] > 0).astype(np.uint8) * 255

        logger.info(
            "SAM segment '%s': mask %s, coverage %.1f%%, prefer_full_box=%s",
            label,
            mask.shape,
            (mask > 0).sum() / mask.size * 100,
            prefer_full_box,
        )

        cache_key = f"{label}_{bbox}"
        self._masks_cache[cache_key] = mask
        return mask

    @staticmethod
    def _best_mask_for_bbox(
        masks: NDArray,
        bbox: tuple[int, int, int, int],
    ) -> int:
        """Return index of the mask with the highest fill ratio inside bbox."""
        x0, y0, x1, y1 = bbox
        box_area = max(1, (x1 - x0) * (y1 - y0))
        best_idx = 0
        best_fill = -1.0
        for i in range(masks.shape[0]):
            roi = masks[i, y0:y1, x0:x1]
            fill = float((roi > 0).sum()) / box_area
            if fill > best_fill:
                best_fill = fill
                best_idx = i
        return best_idx

    def detect_and_segment(
        self,
        image: NDArray[np.uint8],
        label: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> tuple[DetectionResult, NDArray[np.uint8]] | None:
        """Convenience: detect bbox then segment in one call."""
        detection = self.detect_box(
            image, label, box_threshold, text_threshold
        )
        if detection is None:
            return None
        mask = self.segment(image, detection.bbox, label)
        return detection, mask

    def clear_cache(self) -> None:
        """Clear cached masks (call between images)."""
        self._masks_cache.clear()
