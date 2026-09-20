"""Local MLX SAM 3 text-to-instance masks, with SAM 2.1 box fallback."""
from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from models.grounded_sam import DetectionResult, GroundedSAM
from models.vendored_contracts import Sam3ImageModelLike

logger = logging.getLogger(__name__)
_MLX_LOCK = threading.RLock()


class MLXSAM3:
    def __init__(self, source_dir: Path, fallback: GroundedSAM, confidence: float = 0.2):
        self.source_dir = source_dir
        self.checkpoint = source_dir / "sam3-mod-weights/model.safetensors"
        self.fallback = fallback
        self.confidence = confidence
        self._processor: Sam3ImageModelLike | None = None
        self._image = None
        self._state = None
        self._failed = False
        self.warnings: list[str] = []

    def _load(self) -> Sam3ImageModelLike:
        """Load the MLX SAM 3 processor once and return it.

        Returns the processor rather than None so a caller holds a value
        that cannot be None; every path below either returns it or raises.
        """
        if self._processor is not None:
            return self._processor
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"MLX SAM 3 checkpoint missing: {self.checkpoint}")
        existing = sys.modules.get("sam3")
        # A namespace package has __file__ set to None, so this cannot assume
        # a path is there to compare; an entry without one is not the
        # vendored source tree and must not be accepted as if it were.
        existing_file = getattr(existing, "__file__", None) if existing else None
        if existing is not None and (
            existing_file is None
            or not Path(existing_file).resolve().is_relative_to(self.source_dir.resolve())
        ):
            raise RuntimeError("A different sam3 package is already loaded in this process")
        if str(self.source_dir) not in sys.path:
            sys.path.insert(0, str(self.source_dir))
        import mlx.core as mx
        from sam3 import (  # type: ignore[import-not-found]  # vendored; resolved at runtime from the model directory
            build_sam3_image_model,
        )
        from sam3.model.sam3_image_processor import (  # type: ignore[import-not-found]  # vendored; resolved at runtime from the model directory
            Sam3Processor,
        )

        model = build_sam3_image_model(checkpoint_path=str(self.checkpoint))
        mx.eval(model.parameters())
        processor: Sam3ImageModelLike = Sam3Processor(
            model, confidence_threshold=self.confidence
        )
        self._processor = processor
        return processor

    def detect_instances(self, image, label, box_threshold=0.35, text_threshold=0.25, allow_fallback=True):
        with _MLX_LOCK:
            if not self._failed:
                try:
                    processor = self._load()
                    import mlx.core as mx

                    # Hold the actual array, never just id(array), to avoid id reuse.
                    # Pipeline input arrays are immutable for each processing pass.
                    if self._image is not image:
                        self._state = processor.set_image(Image.fromarray(image))
                        mx.eval(self._state)
                        self._image = image
                    processor.reset_all_prompts(self._state)
                    state = processor.set_text_prompt(label, self._state)
                    mx.eval(state)
                    masks = np.asarray(state["masks"])
                    boxes = np.asarray(state["boxes"])
                    scores = np.asarray(state["scores"])
                    h, w = image.shape[:2]
                    results = []
                    for mask, box, score in zip(masks, boxes, scores, strict=True):
                        x0, y0, x1, y1 = box
                        bbox = (max(0, int(x0)), max(0, int(y0)), min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1))))
                        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                            continue
                        results.append(DetectionResult(
                            label=label, bbox=bbox, confidence=float(score),
                            mask=(mask.reshape(h, w) > 0).astype(np.uint8) * 255,
                            source="mlx-sam3",
                        ))
                    if results:
                        # Spatial order keeps instance naming stable across prompt runs.
                        return sorted(results, key=lambda d: (d.bbox[0], d.bbox[1]))
                except (
                    AttributeError,
                    ImportError,
                    IndexError,
                    KeyError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as exc:
                    self._failed = True
                    warning = f"MLX SAM 3 unavailable; using GroundingDINO + SAM 2.1: {exc}"
                    self.warnings.append(warning)
                    logger.warning(warning, exc_info=True)
            if allow_fallback or self._failed:
                return self.fallback.detect_instances(image, label, box_threshold, text_threshold)
            return []

    def detect_part_instances(self, image, label, box_threshold=0.35, text_threshold=0.25):
        return self.detect_instances(image, label, box_threshold, text_threshold, allow_fallback=False)

    def detect_box(self, image, label, box_threshold=0.35, text_threshold=0.25):
        results = self.detect_instances(image, label, box_threshold, text_threshold)
        return max(results, key=lambda d: d.confidence) if results else None

    def segment(self, image, bbox, label="", prefer_full_box=False):
        return self.fallback.segment(image, bbox, label, prefer_full_box)

    def clear_cache(self):
        with _MLX_LOCK:
            self._image = self._state = None
            self.fallback.clear_cache()
