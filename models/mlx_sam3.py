"""Local MLX SAM 3 text-to-instance masks, with SAM 2.1 box fallback."""
from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from models.grounded_sam import DetectionResult, GroundedSAM
from models.sam3_grounding import GroundedInstances, build_policy, ground_text_prompt
from models.vendored_contracts import Sam3ImageModelLike

logger = logging.getLogger(__name__)
_MLX_LOCK = threading.RLock()


class MLXSAM3:
    def __init__(
        self,
        source_dir: Path,
        fallback: GroundedSAM,
        confidence: float = 0.2,
        localization_confidence: float = 0.65,
    ) -> None:
        self.source_dir = source_dir
        self.checkpoint = source_dir / "sam3-mod-weights/model.safetensors"
        self.fallback = fallback
        self.confidence = confidence
        self.localization_confidence = localization_confidence
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

    def _ground(
        self, image: NDArray[np.uint8], label: str, require_presence: bool
    ) -> GroundedInstances:
        """Prompt the loaded model, reusing the cached image embedding."""
        processor = self._load()
        import mlx.core as mx

        # Hold the actual array, never just id(array), to avoid id reuse.
        # Pipeline input arrays are immutable for each processing pass.
        if self._image is not image or self._state is None:
            self._state = processor.set_image(Image.fromarray(image))
            mx.eval(self._state)
            self._image = image
        state: dict[str, Any] = self._state
        processor.reset_all_prompts(state)
        h, w = image.shape[:2]
        # Scores are read out here rather than through set_text_prompt, which
        # would already have multiplied the presence veto into them.
        return ground_text_prompt(
            processor,
            state,
            label,
            w,
            h,
            build_policy(self.confidence, self.localization_confidence, require_presence),
        )

    def _detections(
        self, image: NDArray[np.uint8], label: str, grounded: GroundedInstances
    ) -> list[DetectionResult]:
        h, w = image.shape[:2]
        results = []
        for mask, box, score in zip(grounded.masks, grounded.boxes, grounded.scores, strict=True):
            x0, y0, x1, y1 = box
            bbox = (max(0, int(x0)), max(0, int(y0)), min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1))))
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1] or not mask.any():
                continue
            results.append(DetectionResult(
                label=label, bbox=bbox, confidence=float(score),
                mask=mask, source="mlx-sam3",
            ))
        # Spatial order keeps instance naming stable across prompt runs.
        return sorted(results, key=lambda d: (d.bbox[0], d.bbox[1]))

    def detect_instances(
        self,
        image: NDArray[np.uint8],
        label: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        allow_fallback: bool = True,
        require_presence: bool = False,
    ) -> list[DetectionResult]:
        """Return every instance of `label` that MLX SAM 3 accepts.

        Parameters
        ----------
        require_presence:
            Keep SAM 3's presence veto, which discards an image the model does
            not recognise the label in even when it localises something
            confidently. Off by default because a label reaching this method
            has already been confirmed upstream; see models/sam3_scoring.py.
        """
        with _MLX_LOCK:
            if not self._failed:
                try:
                    grounded = self._ground(image, label, require_presence)
                    if grounded.rescued:
                        logger.info(
                            "MLX SAM 3 presence score %.4f rejected '%s'; kept the "
                            "best localisation instead", grounded.presence, label,
                        )
                    results = self._detections(image, label, grounded)
                    if results:
                        return results
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

    def detect_part_instances(
        self,
        image: NDArray[np.uint8],
        label: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> list[DetectionResult]:
        # Parts are the model's own suggestions, so recognition is still an
        # open question and SAM 3's presence veto stays in force.
        return self.detect_instances(
            image, label, box_threshold, text_threshold,
            allow_fallback=False, require_presence=True,
        )

    def detect_box(
        self,
        image: NDArray[np.uint8],
        label: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> DetectionResult | None:
        results = self.detect_instances(image, label, box_threshold, text_threshold)
        return max(results, key=lambda d: d.confidence) if results else None

    def segment(
        self,
        image: NDArray[np.uint8],
        bbox: tuple[int, int, int, int],
        label: str = "",
        prefer_full_box: bool = False,
    ) -> NDArray[np.uint8]:
        return self.fallback.segment(image, bbox, label, prefer_full_box)

    def clear_cache(self) -> None:
        with _MLX_LOCK:
            self._image = self._state = None
            self.fallback.clear_cache()
