from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from models.vendored_contracts import SamPredictorLike
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
        import ml_dtypes  # type: ignore[import-not-found]
    except ImportError:
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
    except (ImportError, AttributeError):
        return

    if hasattr(BertModel, "get_head_mask"):
        _patch_bert_invert_attention_mask(BertModel)
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
    _patch_bert_invert_attention_mask(BertModel)
    logger.info("Patched BertModel.get_head_mask for GroundingDINO compatibility")


def _patch_bert_invert_attention_mask(bert_model_type) -> None:
    """Restore the historical BERT encoder-mask helper removed in v5."""
    if hasattr(bert_model_type, "invert_attention_mask"):
        return

    def invert_attention_mask(self, encoder_attention_mask):
        if encoder_attention_mask.dim() == 3:
            extended = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            extended = encoder_attention_mask[:, None, None, :]
        else:
            raise ValueError("encoder_attention_mask must have two or three dimensions")
        target_dtype = getattr(self, "dtype", torch.float32)
        return (1.0 - extended.to(dtype=target_dtype)) * torch.finfo(target_dtype).min

    bert_model_type.invert_attention_mask = invert_attention_mask  # type: ignore[attr-defined]
    logger.info("Patched BertModel.invert_attention_mask for GroundingDINO compatibility")


def _patch_get_extended_attention_mask() -> None:
    """Accept legacy `device` positional in `get_extended_attention_mask`.

    transformers >=5.0 dropped the `device` parameter, but GroundingDINO's
    `BertModelWarper.forward` still calls
    `self.get_extended_attention_mask(mask, shape, device)`. We wrap the
    method to detect a `torch.device` in that slot and discard it.
    """
    try:
        from transformers.modeling_utils import ModuleUtilsMixin
    except (ImportError, AttributeError):
        return

    original = getattr(ModuleUtilsMixin, "get_extended_attention_mask", None)
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
        if original is not None:
            return original(self, attention_mask, input_shape, dtype=dtype)

        # transformers 5 removed this helper.  GroundingDINO still calls the
        # historical BERT API, so reproduce the non-causal BERT mask shape
        # rather than pinning an EOL transformers release with advisories.
        if attention_mask.dim() == 3:
            extended = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended = attention_mask[:, None, None, :]
        else:
            raise ValueError("attention_mask must have two or three dimensions")
        target_dtype = dtype or getattr(self, "dtype", torch.float32)
        extended = extended.to(dtype=target_dtype)
        return (1.0 - extended) * torch.finfo(target_dtype).min

    get_extended_attention_mask._skiagrafia_patched = True  # type: ignore[attr-defined]
    ModuleUtilsMixin.get_extended_attention_mask = get_extended_attention_mask  # type: ignore[attr-defined,assignment]
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

    model_config = ConfigDict(arbitrary_types_allowed=True)
    mask: NDArray[np.uint8] | None = None
    source: str = "groundingdino"
    label: str
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1)
    confidence: float


class SegmentationResult(BaseModel):
    """Segmentation mask from SAM 2.1."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    label: str
    bbox: tuple[int, int, int, int]
    mask_shape: tuple[int, int]


class GroundedSAM:
    """GroundingDINO + SAM 2.1 wrapper for text-guided segmentation.

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
        self._sam_predictor: SamPredictorLike | None = None
        self._masks_cache: dict[str, NDArray[np.uint8]] = {}
        self._encoded_image = None

    def _load_dino(self) -> None:
        """Load GroundingDINO model weights (lazy, once)."""
        if self._dino_model is not None:
            return
        try:
            # Resolve paths from constructor params or backward-compat shim
            gsam_root = self._gsam_root or model_path("groundingdino_swint_ogc.pth").parent.parent
            _ensure_gsam_on_path(gsam_root)

            from grounding_dino.groundingdino.util.inference import (  # type: ignore[import-not-found]  # vendored; resolved at runtime from the model directory
                load_model,
            )

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
        except (ImportError, FileNotFoundError, OSError, RuntimeError, ValueError, AttributeError):
            logger.exception("Failed to load GroundingDINO")
            raise

    def _load_sam(self) -> SamPredictorLike:
        """Load SAM 2.1 predictor (lazy, once) and return it.

        Returning the predictor rather than None is what lets callers hold a
        value that cannot be None. Every failure path below raises, so
        reaching the end means it is loaded.
        """
        if self._sam_predictor is not None:
            return self._sam_predictor
        try:
            _ensure_gsam_on_path(self._gsam_root or model_path("groundingdino_swint_ogc.pth").parent.parent)
            from sam2.build_sam import (  # type: ignore[import-not-found]  # vendored; resolved at runtime from the model directory
                build_sam2,
            )
            from sam2.sam2_image_predictor import (  # type: ignore[import-not-found]  # vendored; resolved at runtime from the model directory
                SAM2ImagePredictor,
            )

            weights = self._sam_weights or model_path("sam2.1_hiera_large.pt")
            config = "configs/sam2.1/sam2.1_hiera_l.yaml"
            sam = build_sam2(
                config_file=config,
                ckpt_path=str(weights),
                device=str(DEVICE),
            )
            predictor: SamPredictorLike = SAM2ImagePredictor(sam)
            self._sam_predictor = predictor
            logger.info("SAM 2.1 loaded on %s", DEVICE)
            return predictor
        except (ImportError, FileNotFoundError, OSError, RuntimeError, ValueError, AttributeError):
            logger.exception("Failed to load SAM 2.1")
            raise

    def detect_instances(
        self,
        image: NDArray[np.uint8],
        label: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        skip_synonyms: bool = False,
    ) -> list[DetectionResult]:
        """Run GroundingDINO to get bounding box for a text label.

        Parameters
        ----------
        skip_synonyms : bool
            If True, skip synonym retry loop (faster for preview use).
        """
        self._load_dino()

        import grounding_dino.groundingdino.datasets.transforms as T  # type: ignore[import-not-found]
        from grounding_dino.groundingdino.util.inference import (  # type: ignore[import-not-found]  # vendored; resolved at runtime from the model directory
            predict,
        )

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

        h, w = image.shape[:2]
        detections = []
        for idx in logits.argsort(descending=True).tolist():
            cx, cy, bw, bh = boxes[idx].tolist()
            bbox = (
                max(0, int((cx - bw / 2) * w)), max(0, int((cy - bh / 2) * h)),
                min(w, int(np.ceil((cx + bw / 2) * w))), min(h, int(np.ceil((cy + bh / 2) * h))),
            )
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            # Duplicate proposals are common; suppress near-identical boxes only.
            if any(self._box_iou(bbox, d.bbox) > 0.85 for d in detections):
                continue
            detections.append(DetectionResult(label=label, bbox=bbox, confidence=float(logits[idx])))
        return detections

    @staticmethod
    def _box_iou(a, b):
        inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / union if union else 0.0

    def detect_box(self, image, label, box_threshold=0.35, text_threshold=0.25, skip_synonyms=False):
        results = self.detect_instances(image, label, box_threshold, text_threshold, skip_synonyms)
        return max(results, key=lambda d: d.confidence) if results else None

    def segment(
        self,
        image: NDArray[np.uint8],
        bbox: tuple[int, int, int, int],
        label: str = "",
        prefer_full_box: bool = False,
    ) -> NDArray[np.uint8]:
        """Run SAM 2.1 segmentation within a bounding box.

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
        predictor = self._load_sam()

        if self._encoded_image is not image:
            predictor.set_image(image)
            self._encoded_image = image
            self._masks_cache.clear()
        cache_key = f"{label}_{bbox}_{prefer_full_box}"
        if cache_key in self._masks_cache:
            return self._masks_cache[cache_key].copy()

        box_array = np.array(bbox, dtype=np.float32)

        if prefer_full_box:
            masks, scores, _ = predictor.predict(
                box=box_array,
                multimask_output=True,
            )
            # Pick the mask with the highest coverage inside the bbox
            best_idx = self._best_mask_for_bbox(masks, bbox)
            mask = (masks[best_idx] > 0).astype(np.uint8) * 255
        else:
            masks, scores, _ = predictor.predict(
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
        self._encoded_image = None
        if self._sam_predictor is not None:
            self._sam_predictor.reset_predictor()
