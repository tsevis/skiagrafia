"""orchestrator.py  --  v5.0  (Contracts & DI Refactor)
Single-image pipeline manager for Skiagrafia.

Steps 0-9  = Structural branch.
Steps 10-12 = Stylization branch (only when stylizer is provided).

The Orchestrator no longer imports or instantiates concrete model clients.
It receives a CapabilitySet via constructor injection.

All inference runs locally. VLM clients communicate with local services.
"""
from __future__ import annotations

import itertools
import logging
import re
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import NamedTuple, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from core.contracts import CapabilitySet
from core.interrogation import (
    InterrogationCandidate,
    TypographyObservation,
    is_individual_glyph_label,
)
from core.knowledge import KnowledgePack
from core.layer_editing import all_objects_alpha, body_alpha
from models.grounded_sam import DetectionResult
from processors.mask_ops import refine_mask
from processors.output_writer import write_svg, write_tiff
from processors.source_image import detection_image, load_source_image
from processors.vectorizer import assemble_svg
from utils.array_types import as_uint8
from utils.coord_math import tight_bbox
from utils.security import SecurityError, safe_child_path

logger = logging.getLogger(__name__)

MIN_CHILD_COVERAGE_PCT = 0.5
# Parts are model suggestions, so use a stricter SAM 3 acceptance threshold.
MIN_SAM3_PART_SCORE = 0.65
MAX_CHILD_PARENT_IOU = 0.85
MAX_CHILD_CHILD_IOU = 0.80
MIN_PARENT_COVERAGE_PCT = 0.5
MIN_CONFIRMED_COVERAGE_PCT = 0.05
PARENT_IOU_MERGE = 0.50
PARENT_CONTAINMENT_MERGE = 0.75
BBOX_IOU_MERGE = 0.60
BBOX_EXPAND_RATIO = 0.30
MAX_LABEL_FILENAME_LEN = 60  # max chars of a label used in output filenames
TYPOGRAPHY_BOX_MATCH_MIN_IOU = 0.30
TYPOGRAPHY_COMPOSITE_MIN_CONTRIBUTION = 0.15
TYPOGRAPHY_COMPOSITE_MIN_COVERAGE = 0.80


def _safe_filename_label(label: str) -> str:
    """Truncate and sanitize a label for use in output filenames.

    Moondream can return extremely long child labels (e.g. numbered
    lists of 38 items).  macOS enforces a 255-byte filename limit.
    """
    # Replace path-unsafe characters
    safe = re.sub(r"[^A-Za-z0-9._ -]+", "_", label).strip(". ")
    safe = safe.replace("/", "_").replace("\\", "_").replace(":", "_")
    if not safe:
        safe = "layer"
    if len(safe) > MAX_LABEL_FILENAME_LEN:
        safe = safe[:MAX_LABEL_FILENAME_LEN].rstrip(". ")
    return safe

STRUCTURAL_STEPS = [
    "Loading image",
    "Object recognition",
    "Instance detection",
    "Object masks",
    "Component masks",
    "Coordinate remapping",
    "VitMatte alpha refinement",
    "Mask refinement",
    "VTracer vectorization",
    "Structural SVG assembly & export",
]
PIPELINE_STEPS = STRUCTURAL_STEPS


class _SourceImage(NamedTuple):
    """What loading an image produces, carried between pipeline stages.

    A NamedTuple rather than four loose locals threaded through five
    signatures: the pixels, the alpha and the ICC profile belong together,
    and nothing downstream may replace one of them.
    """

    rgb: NDArray[np.uint8]
    alpha: NDArray[np.uint8]
    icc: bytes | None
    detection: NDArray[np.uint8]
    height: int
    width: int


class LayerResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    layer_id: str = ""
    parent_id: str | None = None
    confidence: float | None = None
    source: str = ""
    alpha_path: str | None = None
    mask: NDArray[np.uint8] | None = Field(default=None, exclude=True)
    alpha: NDArray[np.uint8] | None = Field(default=None, exclude=True)
    preview_opacity: float = Field(default=1.0, exclude=True)
    label: str
    role: str
    parent_label: str | None = None
    bbox: tuple[int, int, int, int]
    svg_data: str = ""
    dx: int = 0
    dy: int = 0


class PipelineResult(BaseModel):
    image_path: str
    width: int
    height: int
    layers: list[LayerResult] = []
    svg_path: str | None = None
    tiff_path: str | None = None
    all_objects_tiff_path: str | None = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    tiff_files: list[str] = Field(default_factory=list)


def _mask_iou(a: NDArray[np.uint8], b: NDArray[np.uint8]) -> float:
    a_bool, b_bool = a > 127, b > 127
    inter = np.logical_and(a_bool, b_bool).sum()
    union = np.logical_or(a_bool, b_bool).sum()
    return float(inter / union) if union else 0.0


def _mask_containment(a: NDArray[np.uint8], b: NDArray[np.uint8]) -> float:
    """Fraction of the *smaller* mask that is contained in the larger one.

    Returns a value in [0, 1].  A high value means one mask is mostly
    inside the other — strong evidence they represent the same object even
    when IoU is low (because one mask is much larger).
    """
    a_bool, b_bool = a > 127, b > 127
    a_area = int(a_bool.sum())
    b_area = int(b_bool.sum())
    if a_area == 0 or b_area == 0:
        return 0.0
    inter = int(np.logical_and(a_bool, b_bool).sum())
    smaller = min(a_area, b_area)
    return float(inter / smaller)


def _bbox_overlaps(
    child_bbox: tuple[int, int, int, int],
    parent_bbox: tuple[int, int, int, int],
    img_h: int,
    img_w: int,
    expand: float = BBOX_EXPAND_RATIO,
) -> bool:
    px0, py0, px1, py1 = parent_bbox
    pw, ph = px1 - px0, py1 - py0
    ex0 = max(0, int(px0 - pw * expand))
    ey0 = max(0, int(py0 - ph * expand))
    ex1 = min(img_w, int(px1 + pw * expand))
    ey1 = min(img_h, int(py1 + ph * expand))
    cx0, cy0, cx1, cy1 = child_bbox
    return cx0 < ex1 and cx1 > ex0 and cy0 < ey1 and cy1 > ey0


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = bbox
    return max(0, x1 - x0) * max(0, y1 - y0)


def _bbox_iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    """Intersection-over-union of two bounding boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = _bbox_area(a)
    area_b = _bbox_area(b)
    union = area_a + area_b - inter
    return float(inter / union) if union else 0.0


def _normalized_bbox_to_image(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Convert a 0..1000 VLM box to the current image dimensions."""
    x0, y0, x1, y1 = bbox
    return (
        round(x0 * width / 1000),
        round(y0 * height / 1000),
        round(x1 * width / 1000),
        round(y1 * height / 1000),
    )


def _rectangle_union_area(rectangles: list[tuple[int, int, int, int]]) -> int:
    """Return exact union area for a small list of axis-aligned rectangles."""
    rectangles = [rectangle for rectangle in rectangles if _bbox_area(rectangle)]
    if not rectangles:
        return 0
    x_edges = sorted({edge for rectangle in rectangles for edge in (rectangle[0], rectangle[2])})
    area = 0
    for left, right in itertools.pairwise(x_edges):
        if right <= left:
            continue
        spans = sorted(
            (rectangle[1], rectangle[3])
            for rectangle in rectangles
            if rectangle[0] < right and rectangle[2] > left
        )
        covered_y = 0
        # ONE optional pair, not two optional ints. The bounds are only ever
        # meaningful together, and as separate names nothing said so: the
        # `start > current_end` branch is reachable only because the other
        # name happens to have been set on the same line. A tuple makes that
        # invariant structural instead of a convention, and each step rebinds
        # rather than mutating.
        current: tuple[int, int] | None = None
        for start, end in spans:
            if current is None:
                current = (start, end)
            elif start > current[1]:
                covered_y += current[1] - current[0]
                current = (start, end)
            else:
                current = (current[0], max(current[1], end))
        if current is not None:
            covered_y += current[1] - current[0]
        area += (right - left) * covered_y
    return area


def _glyph_layer_label(category: str, glyph: str) -> str:
    """Make a readable output label without trusting a VLM for filenames."""
    words = set(re.findall(r"[^\W\d_]+|\d+", category.casefold(), flags=re.UNICODE))
    if words & {"letter", "letters"}:
        kind = "letter"
    elif words & {"digit", "digits", "numeral", "numerals", "number", "numbers"}:
        kind = "digit"
    else:
        kind = "glyph"
    return f"{kind} {glyph}"


def _crop_to_bbox(
    image: NDArray[np.uint8],
    bbox: tuple[int, int, int, int],
    padding: int = 8,
) -> tuple[NDArray[np.uint8], int, int]:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(w, x1 + padding)
    y1 = min(h, y1 + padding)
    return image[y0:y1, x0:x1], x0, y0


def _clip_mask_to_bbox(
    mask: NDArray[np.uint8],
    bbox: tuple[int, int, int, int],
) -> NDArray[np.uint8]:
    """Zero out mask pixels outside the bounding box."""
    x0, y0, x1, y1 = bbox
    h, w = mask.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    clipped = np.zeros_like(mask)
    clipped[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return clipped


class Orchestrator:
    """Single-image pipeline -- 10 structural steps.

    v5.0: Receives a CapabilitySet via constructor injection.
    The Orchestrator never imports or instantiates concrete model clients.

    Pipeline-level parameters that remain on the Orchestrator:
    - box_threshold / text_threshold: passed to detector on every call
    - bilateral_d: used in mask refinement (pure OpenCV)
    - output_mode / output_dir: output concerns
    """

    def __init__(
        self,
        capabilities: CapabilitySet,
        output_dir: Path | None = None,
        output_mode: str = "vector+bitmap",
        bilateral_d: int = 9,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        progress_callback: Callable[[int, str], None] | None = None,
        knowledge_pack: KnowledgePack | None = None,
        quality: str = "balanced",
    ) -> None:
        # Capability injection
        self._interrogator = capabilities.interrogator
        self._detector = capabilities.detector
        self._segmenter = capabilities.segmenter
        self._alpha_refiner = capabilities.alpha_refiner
        self._vectorizer = capabilities.vectorizer

        # Pipeline parameters
        self._output_dir = output_dir or Path.home() / "Desktop" / "skiagrafia_out"
        self._output_mode = output_mode
        self._bilateral_d = bilateral_d
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._progress = progress_callback or (lambda step, msg: None)
        self._knowledge_pack = knowledge_pack
        self._quality = quality

    def _report(self, step: int, msg: str | None = None) -> None:
        text = msg or PIPELINE_STEPS[step]
        self._progress(step, text)
        logger.info("Step %d/%d: %s", step + 1, len(PIPELINE_STEPS), text)

    def set_confirmed_selections(self, selections: dict[str, str]) -> None:
        """Apply one image's approved instance policy before ``process()``."""
        set_selections = getattr(self._interrogator, "set_confirmed_selections", None)
        if callable(set_selections):
            set_selections(selections)

    # ─────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────

    def process(
        self,
        image_path: str | Path,
        confirmed_labels: list[str] | None = None,
        manual_detections: list[dict] | None = None,
    ) -> PipelineResult:
        image_path = Path(image_path)
        result = PipelineResult(image_path=str(image_path), width=0, height=0)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = self._run_structural_branch(
                image_path,
                result,
                confirmed_labels,
                manual_detections,
            )
        except (OSError, ValueError, RuntimeError, SecurityError) as exc:
            result.error = str(exc)
            logger.exception("Pipeline failed for %s: %s", image_path, exc)

        finally:
            self._segmenter.clear_cache()
        return result

    # ─────────────────────────────────────────────────────────────────────
    # Structural branch  (steps 0–9)  — identical logic to v1
    # ─────────────────────────────────────────────────────────────────────

    def _run_structural_branch(
        self,
        image_path: Path,
        result: PipelineResult,
        confirmed_labels: list[str] | None,
        manual_detections: list[dict] | None,
    ) -> PipelineResult:
        """The structural pipeline, as the sequence of stages it already was.

        Each stage below used to be a block inside one 220-line function,
        separated only by a comment and sharing every local. They are the
        same blocks in the same order, doing the same work: what changed is
        that each one now states what it needs and what it produces, so a
        stage can be read -- and a suspect stage stepped through -- without
        holding the other four in your head.
        """
        source = self._load_source(image_path, result)
        parents, children_by_parent = self._interrogate(source.detection, confirmed_labels)
        manual_lookup = self._build_manual_lookup(manual_detections)

        masks, accepted = self._detect_parent_layers(
            source, parents, children_by_parent, manual_lookup, result
        )
        self._attach_part_layers(source, accepted, children_by_parent, masks, result)

        self._report(5, "Object and part coordinates resolved")
        all_alpha = self._write_layer_alphas(source, image_path, accepted, masks, result)
        self._write_all_objects_alpha(source, image_path, all_alpha, result)
        self._write_vector_output(source, image_path, masks, result)

        result.warnings.extend(getattr(self._detector, "warnings", []))
        return result

    # ── Stage 0: load ───────────────────────────────────────────────────

    def _load_source(self, image_path: Path, result: PipelineResult) -> _SourceImage:
        """Read the image and record its dimensions on the result."""
        self._report(0)
        try:
            source_rgb, source_alpha, source_icc = load_source_image(image_path)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Invalid image input: {exc}") from exc
        detection = detection_image(source_rgb, source_alpha)
        height, width = detection.shape[:2]
        result.width, result.height = width, height
        return _SourceImage(
            rgb=source_rgb,
            alpha=source_alpha,
            icc=source_icc,
            detection=detection,
            height=height,
            width=width,
        )

    # ── Stage 2-3: parent objects ───────────────────────────────────────

    def _detect_parent_layers(
        self,
        source: _SourceImage,
        parents: list,
        children_by_parent: dict[str, list[str]],
        manual_lookup: dict,
        result: PipelineResult,
    ) -> tuple[dict[str, NDArray[np.uint8]], list]:
        """Detect and segment every parent object across the whole image.

        Detects all parents before any cropping, so one full-image encoding
        is reused. Returns the layer masks and the accepted (layer, parent)
        pairs; `children_by_parent` is updated in place when a duplicate
        detection folds its parts into the layer already accepted.
        """
        image = source.detection
        masks: dict[str, NDArray[np.uint8]] = {}
        accepted = []
        for parent in parents:
            self._report(2, f"Finding instances: {parent.display_label}")
            detections = self._detect_instances(image, parent, manual_lookup)
            layer_labels = [parent.display_label] * len(detections)
            if is_individual_glyph_label(parent.display_label) and detections:
                detections, layer_labels = self._resolve_glyph_detections(
                    image, parent, detections, layer_labels, result
                )
            if not detections:
                result.warnings.append(f"Could not locate '{parent.display_label}'. Draw a box to select it manually.")
                continue
            for (detection, is_manual), layer_label in zip(detections, layer_labels, strict=True):
                if len(accepted) >= 64:
                    result.warnings.append("Stopped at 64 object instances; narrow the prompt or process a crop.")
                    break
                self._report(3, f"Object mask: {parent.display_label}")
                mask = self._detection_mask(image, detection, layer_label, is_manual)
                mask[source.alpha == 0] = 0
                # Keep small legitimate objects; reject only empty/tiny noise.
                if np.count_nonzero(mask) < 8:
                    result.warnings.append(f"Empty or tiny mask for '{parent.display_label}'.")
                    continue
                # Containment and overlapping boxes alone do not imply duplication.
                duplicate = next((entry for entry in accepted if _mask_iou(mask, masks[entry[0].layer_id]) > 0.9), None)
                if duplicate:
                    existing_label = duplicate[1].display_label
                    extra = children_by_parent.get(parent.display_label, [])
                    children_by_parent[existing_label] = list(dict.fromkeys(children_by_parent.get(existing_label, []) + extra))
                    continue
                layer_id = self._layer_id(len(accepted) + 1, layer_label)
                layer = LayerResult(
                    layer_id=layer_id, label=layer_label, role="parent", bbox=detection.bbox,
                    confidence=detection.confidence, source="manual" if is_manual else detection.source,
                )
                masks[layer_id] = refine_mask(mask, min_contour_area=0)
                accepted.append((layer, parent))
        return masks, accepted

    def _resolve_glyph_detections(
        self,
        image: NDArray[np.uint8],
        parent: InterrogationCandidate,
        detections: list,
        layer_labels: list[str],
        result: PipelineResult,
    ) -> tuple[list, list[str]]:
        """Split typography detections into individual glyphs where possible.

        Falls back to rejecting composites that straddle several glyphs, and
        records a warning for each path so the result explains what it did.
        """
        observation = self._inspect_individual_glyphs(image, parent)
        matched = (
            self._match_typography_detections(image, detections, observation, parent.display_label)
            if observation is not None
            else None
        )
        if matched is not None:
            return matched
        detections, rejected = self._reject_cross_glyph_composites(detections)
        layer_labels = [parent.display_label] * len(detections)
        if rejected:
            result.warnings.append(
                f"Excluded {rejected} composite typography proposal(s) for "
                f"'{parent.display_label}' because each overlapped multiple glyph instances."
            )
        if observation is not None:
            result.warnings.append(
                f"Typography reading for '{parent.display_label}' could not be matched "
                "one-to-one with local detections; retained only independently detected glyphs."
            )
        return detections, layer_labels

    # ── Stage 4: parts, inside each parent's crop ───────────────────────

    def _attach_part_layers(
        self,
        source: _SourceImage,
        accepted: list,
        children_by_parent: dict[str, list[str]],
        masks: dict[str, NDArray[np.uint8]],
        result: PipelineResult,
    ) -> None:
        """Name, detect and segment each part inside its parent's crop.

        Appends the parent layer and then its child layers to `result`, and
        adds each child's mask to `masks`, in the order the original single
        function produced them.
        """
        image = source.detection
        # getattr on an optional capability yields `object`, so the list it
        # returns was not iterable as far as a checker could tell. The lookup
        # stays dynamic -- only its shape is now declared.
        query_parts = cast(
            "Callable[..., list[str]] | None",
            getattr(self._interrogator, "discover_parts", None),
        )
        part_limit = getattr(self._interrogator, "part_query_limit", 0)
        for parent_index, (layer, parent) in enumerate(accepted):
            result.layers.append(layer)
            parent_mask = masks[layer.layer_id]
            y0, x0, y1, x1 = tight_bbox(parent_mask, padding=0)
            crop, dx, dy = _crop_to_bbox(image, (x0, y0, x1, y1), padding=8)
            parts = children_by_parent.get(parent.display_label, [])
            if not parts and callable(query_parts) and parent_index < part_limit:
                self._report(4, f"Inspecting visible parts: {parent.display_label}")
                parts = query_parts(crop, parent, self._knowledge_pack)
            child_masks = []
            for part in parts:
                candidate = self._child_candidate(parent, part)
                for detection, _ in self._detect_instances(crop, candidate, None):
                    if detection.source == "mlx-sam3" and detection.confidence < MIN_SAM3_PART_SCORE:
                        logger.info("Excluded tentative part %s (SAM 3 score %.3f)", part, detection.confidence)
                        continue
                    mask = self._contained_child_mask(
                        source, crop, (dx, dy), detection, part, parent_mask, child_masks
                    )
                    if mask is None:
                        continue
                    child_masks.append(mask)
                    child_id = f"{layer.layer_id}-part-{len(child_masks):03d}"
                    bx0, by0, bx1, by1 = detection.bbox
                    masks[child_id] = refine_mask(mask, min_contour_area=0)
                    result.layers.append(LayerResult(
                        layer_id=child_id, label=part, role="child", parent_label=layer.label,
                        parent_id=layer.layer_id, bbox=(bx0 + dx, by0 + dy, bx1 + dx, by1 + dy),
                        confidence=detection.confidence, source=detection.source,
                    ))

    # ── Stage 6: mattes ─────────────────────────────────────────────────

    def _contained_child_mask(
        self,
        source: _SourceImage,
        crop: NDArray[np.uint8],
        offset: tuple[int, int],
        detection: DetectionResult,
        part: str,
        parent_mask: NDArray[np.uint8],
        child_masks: list[NDArray[np.uint8]],
    ) -> NDArray[np.uint8] | None:
        """A part's mask in full-image coordinates, or None if it is rejected.

        Four independent reasons to reject, previously four bare `continue`
        statements inside a doubly-nested loop, where the condition that
        fired was not visible from the loop header:

        - the mask is empty or tiny;
        - less than 90% of it falls inside its parent;
        - it covers so much of the parent that it IS the parent again;
        - it duplicates a sibling already accepted.
        """
        dx, dy = offset
        local = self._detection_mask(crop, detection, part)
        mask = np.zeros((source.height, source.width), dtype=np.uint8)
        mask[dy:dy + crop.shape[0], dx:dx + crop.shape[1]] = local
        area = np.count_nonzero(mask)
        if area < 8:
            return None
        intersection = as_uint8(cv2.bitwise_and(mask, parent_mask))
        if np.count_nonzero(intersection) / area < 0.90:
            return None
        mask = intersection
        if _mask_iou(mask, parent_mask) > MAX_CHILD_PARENT_IOU:
            return None
        if any(_mask_iou(mask, other) > MAX_CHILD_CHILD_IOU for other in child_masks):
            return None
        return mask

    def _write_layer_alphas(
        self,
        source: _SourceImage,
        image_path: Path,
        accepted: list,
        masks: dict[str, NDArray[np.uint8]],
        result: PipelineResult,
    ) -> NDArray[np.uint8]:
        """Produce the all-objects alpha, exporting per-layer TIFFs if asked.

        In bitmap mode every layer gets its own matte and TIFF, and a body
        TIFF is written for each parent that has children. Otherwise only the
        all-objects sidecar is produced. Returns the all-objects alpha.
        """
        image = source.detection
        if "bitmap" not in self._output_mode:
            return self._all_objects_only_alpha(source, masks, result)

        self._report(6)
        alphas = {}
        for layer in result.layers:
            mask = masks[layer.layer_id]
            try:
                alpha = mask.copy() if self._quality == "fast" else self._alpha_refiner.predict(image, mask)
            except (OSError, RuntimeError, ValueError) as exc:
                raise RuntimeError(f"Alpha refinement failed for '{layer.label}'.") from exc
            if alpha.shape != image.shape[:2]:
                raise RuntimeError(f"Alpha refinement failed for '{layer.label}': invalid dimensions.")
            alpha = np.minimum(alpha, source.alpha)
            if layer.parent_id:
                alpha = np.minimum(alpha, alphas[layer.parent_id])
            alphas[layer.layer_id] = alpha
            layer.alpha = alpha
            path = self._output_path(f"{self._image_token(image_path)}_{layer.layer_id}.tiff")
            try:
                write_tiff(source.rgb, path, alpha, icc_profile=source.icc)
            except (OSError, RuntimeError, ValueError, SecurityError) as exc:
                raise RuntimeError(f"TIFF export failed for '{layer.label}'.") from exc
            layer.alpha_path = str(path)
            result.tiff_files.append(str(path))
        # Bodies use subtraction of the actual child mattes, not a second
        # independent matting pass that could grow over the child again.
        for layer, _ in accepted:
            children = [c for c in result.layers if c.parent_id == layer.layer_id]
            if children:
                body = body_alpha(alphas[layer.layer_id], [alphas[c.layer_id] for c in children])
                path = self._output_path(f"{self._image_token(image_path)}_{layer.layer_id}-body.tiff")
                try:
                    write_tiff(source.rgb, path, body, icc_profile=source.icc)
                except (OSError, RuntimeError, ValueError, SecurityError) as exc:
                    raise RuntimeError(f"TIFF body export failed for '{layer.label}'.") from exc
                result.tiff_files.append(str(path))
        return all_objects_alpha(list(alphas.values()), (source.height, source.width))

    def _all_objects_only_alpha(
        self,
        source: _SourceImage,
        masks: dict[str, NDArray[np.uint8]],
        result: PipelineResult,
    ) -> NDArray[np.uint8]:
        """The all-objects alpha for a run that exports no per-layer TIFFs.

        The sidecar is intentionally present even for a vector-only run: it
        gives every processed image one complete foreground asset, while
        per-layer TIFFs remain opt-in.
        """
        image = source.detection
        self._report(6, "Creating all-objects alpha sidecar")
        all_mask = all_objects_alpha(list(masks.values()), (source.height, source.width))
        if result.layers and self._quality != "fast":
            try:
                all_alpha = self._alpha_refiner.predict(image, all_mask)
            except (OSError, RuntimeError, ValueError) as exc:
                raise RuntimeError("All-objects alpha refinement failed.") from exc
            if all_alpha.shape != image.shape[:2]:
                raise RuntimeError("All-objects alpha refinement failed: invalid dimensions.")
        else:
            all_alpha = all_mask
        return np.minimum(all_alpha, source.alpha)

    def _write_all_objects_alpha(
        self,
        source: _SourceImage,
        image_path: Path,
        all_alpha: NDArray[np.uint8],
        result: PipelineResult,
    ) -> None:
        """Always export one complete foreground asset.

        With no accepted layers this is an intentionally transparent TIFF: a
        truthful, predictable result rather than a full-frame background
        fallback.
        """
        all_objects_path = self._output_path(
            f"{self._image_token(image_path)}_all-objects.tiff"
        )
        try:
            write_tiff(source.rgb, all_objects_path, all_alpha, icc_profile=source.icc)
        except (OSError, RuntimeError, ValueError, SecurityError) as exc:
            raise RuntimeError("All-objects TIFF export failed.") from exc
        result.all_objects_tiff_path = str(all_objects_path)
        result.tiff_files.append(str(all_objects_path))
        result.tiff_path = str(self._output_dir)

    # ── Stage 7-9: vector output ────────────────────────────────────────

    def _write_vector_output(
        self,
        source: _SourceImage,
        image_path: Path,
        masks: dict[str, NDArray[np.uint8]],
        result: PipelineResult,
    ) -> None:
        """Trace every layer mask and assemble the combined SVG."""
        self._report(7, "Preserving holes and fine mask details")
        self._report(8)
        svg_layers = []
        for layer in result.layers:
            layer.mask = masks[layer.layer_id]
            try:
                layer.svg_data = self._vectorizer.trace(layer.mask)
            except (OSError, RuntimeError, ValueError) as exc:
                raise RuntimeError(f"SVG tracing failed for '{layer.label}'.") from exc
            svg_layers.append({"id": layer.layer_id, "label": layer.label, "parent_id": layer.parent_id or "",
                               "svg_data": layer.svg_data, "dx": 0, "dy": 0})
        self._report(9)
        if svg_layers:
            path = self._output_path(f"{self._image_token(image_path)}.svg")
            try:
                write_svg(assemble_svg(source.width, source.height, svg_layers), path)
            except (OSError, RuntimeError, ValueError, SecurityError) as exc:
                raise RuntimeError("SVG export failed integrity validation.") from exc
            result.svg_path = str(path)
        else:
            result.warnings.append("No usable object masks were found. Review the prompt or draw a box.")

    def _output_path(self, filename: str) -> Path:
        """Create a flat output path under the configured, canonical root."""
        return safe_child_path(self._output_dir, filename)

    @staticmethod
    def _image_token(image_path: Path) -> str:
        """A readable, collision-resistant filename stem for one source image."""
        stem = _safe_filename_label(image_path.stem)[:80]
        digest = sha256(str(image_path.resolve(strict=False)).encode("utf-8")).hexdigest()[:10]
        return f"{stem}-{digest}"

    @staticmethod
    def _layer_id(index: int, label: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:40] or "object"
        return f"object-{index:03d}-{slug}"

    def _detect_instances(self, image, candidate, manual_lookup):
        manual = []
        while manual_lookup:
            bbox = self._consume_manual_bbox(manual_lookup, candidate)
            if bbox is None:
                break
            h, w = image.shape[:2]
            x0, y0, x1, y1 = bbox
            bbox = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                manual.append((DetectionResult(label=candidate.display_label, bbox=bbox, confidence=1.0, source="manual"), True))
        if manual:
            return manual
        detect_many = cast(
            "Callable[..., list[DetectionResult]] | None",
            getattr(self._detector, "detect_instances", None),
        )
        if candidate.role == "child" and self._quality != "detailed":
            detect_many = cast(
                "Callable[..., list[DetectionResult]] | None",
                getattr(self._detector, "detect_part_instances", detect_many),
            )
        for phrase in candidate.detector_phrases or [candidate.display_label]:
            if callable(detect_many):
                detections = detect_many(image, phrase, self._box_threshold, self._text_threshold)
            else:
                detection = self._detector.detect_box(image, phrase, self._box_threshold, self._text_threshold)
                detections = [detection] if detection else []
            # Never pass invalid coordinates into a predictor.
            h, w = image.shape[:2]
            valid = []
            for detection in detections:
                x0, y0, x1, y1 = detection.bbox
                bbox = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    valid.append(detection.model_copy(update={"bbox": bbox}))
            if valid:
                selection = candidate.selection
                if selection in {"leftmost", "rightmost"}:
                    valid = [sorted(valid, key=lambda d: (d.bbox[0] + d.bbox[2]) / 2)[0 if selection == "leftmost" else -1]]
                elif selection in {"largest", "smallest"}:
                    valid = [sorted(valid, key=lambda d: np.count_nonzero(d.mask) if d.mask is not None else _bbox_area(d.bbox))[0 if selection == "smallest" else -1]]
                return [(detection, False) for detection in valid]
        return []

    def _inspect_individual_glyphs(
        self,
        image: NDArray[np.uint8],
        candidate: InterrogationCandidate,
    ) -> TypographyObservation | None:
        """Ask the interrogator for an optional, local semantic glyph check."""
        inspect = getattr(self._interrogator, "inspect_individual_glyphs", None)
        if not callable(inspect):
            return None
        observation = inspect(image, candidate)
        return observation if isinstance(observation, TypographyObservation) else None

    def _match_typography_detections(
        self,
        image: NDArray[np.uint8],
        detections: list[tuple[DetectionResult, bool]],
        observation: TypographyObservation,
        category: str,
    ) -> tuple[list[tuple[DetectionResult, bool]], list[str]] | None:
        """Require a one-to-one match between semantic glyphs and local boxes.

        The local detector/SAM stack remains authoritative for pixel geometry.
        A VLM observation only selects a detector proposal when the two views
        overlap enough.  If even one semantic glyph has no independent local
        match, the method declines to relabel or discard any proposal.
        """
        if any(is_manual for _detection, is_manual in detections):
            return None
        height, width = image.shape[:2]
        matches: list[tuple[float, int, int]] = []
        for element_index, element in enumerate(observation.elements):
            semantic_bbox = _normalized_bbox_to_image(element.bbox, width, height)
            for detection_index, (detection, _is_manual) in enumerate(detections):
                score = _bbox_iou(semantic_bbox, detection.bbox)
                if score >= TYPOGRAPHY_BOX_MATCH_MIN_IOU:
                    matches.append((score, element_index, detection_index))

        assigned_elements: set[int] = set()
        assigned_detections: set[int] = set()
        assignments: dict[int, int] = {}
        for _score, element_index, detection_index in sorted(
            matches,
            key=lambda match: (-match[0], match[1], match[2]),
        ):
            if element_index in assigned_elements or detection_index in assigned_detections:
                continue
            assigned_elements.add(element_index)
            assigned_detections.add(detection_index)
            assignments[element_index] = detection_index

        if len(assignments) != len(observation.elements):
            return None
        ordered_detections = [detections[assignments[index]] for index in range(len(observation.elements))]
        labels = [
            _glyph_layer_label(category, observation.elements[index].glyph)
            for index in range(len(observation.elements))
        ]
        return ordered_detections, labels

    def _reject_cross_glyph_composites(
        self,
        detections: list[tuple[DetectionResult, bool]],
    ) -> tuple[list[tuple[DetectionResult, bool]], int]:
        """Reject a proposal substantially explained by two prior glyph boxes.

        This is the detector-only safety net for when the semantic verifier is
        unavailable.  It targets the characteristic false positive where one
        proposal spans pieces of two neighbouring glyphs; it does not reject
        an ordinary overlapping glyph or any manual box.
        """
        kept: list[tuple[DetectionResult, bool]] = []
        rejected = 0
        for detection, is_manual in detections:
            if is_manual:
                kept.append((detection, is_manual))
                continue
            candidate_area = _bbox_area(detection.bbox)
            intersections: list[tuple[int, int, int, int]] = []
            for previous, previous_manual in kept:
                if previous_manual:
                    continue
                x0 = max(detection.bbox[0], previous.bbox[0])
                y0 = max(detection.bbox[1], previous.bbox[1])
                x1 = min(detection.bbox[2], previous.bbox[2])
                y1 = min(detection.bbox[3], previous.bbox[3])
                overlap = (x0, y0, x1, y1)
                if candidate_area and _bbox_area(overlap) / candidate_area >= TYPOGRAPHY_COMPOSITE_MIN_CONTRIBUTION:
                    intersections.append(overlap)
            covered = _rectangle_union_area(intersections)
            if (
                len(intersections) >= 2
                and candidate_area
                and covered / candidate_area >= TYPOGRAPHY_COMPOSITE_MIN_COVERAGE
            ):
                rejected += 1
                continue
            kept.append((detection, is_manual))
        return kept, rejected

    def _detection_mask(self, image, detection, label, manual=False):
        mask = detection.mask
        if mask is None or manual:
            mask = self._segmenter.segment(image, detection.bbox, label, prefer_full_box=manual)
        if mask.shape != image.shape[:2]:
            raise RuntimeError(f"Mask generation failed for '{label}': invalid dimensions.")
        mask = (mask > 127).astype(np.uint8) * 255
        return _clip_mask_to_bbox(mask, detection.bbox) if manual else mask

    # ─────────────────────────────────────────────────────────────────────
    # Interrogation helper
    # ─────────────────────────────────────────────────────────────────────

    def _interrogate(
        self,
        image: NDArray[np.uint8],
        confirmed_labels: list[str] | None,
    ) -> tuple[list[InterrogationCandidate], dict[str, list[str]]]:
        interrogation = self._interrogator.interrogate(
            image,
            confirmed_labels=confirmed_labels,
            knowledge_pack=self._knowledge_pack,
        )
        logger.info(
            "Interrogation stage=%s summary=%s candidates=%s",
            interrogation.escalation_stage,
            interrogation.confidence_summary,
            [candidate.display_label for candidate in interrogation.candidates],
        )
        return interrogation.candidates, interrogation.children_by_parent

    def _detect_candidate(
        self,
        image: NDArray[np.uint8],
        candidate: InterrogationCandidate,
        manual_lookup: dict[str, list[tuple[int, int, int, int]]] | None = None,
    ):
        det, _is_manual = self._detect_candidate_ex(image, candidate, manual_lookup)
        return det

    def _detect_candidate_ex(
        self,
        image: NDArray[np.uint8],
        candidate: InterrogationCandidate,
        manual_lookup: dict[str, list[tuple[int, int, int, int]]] | None = None,
    ) -> tuple[DetectionResult | None, bool]:
        """Like _detect_candidate but also returns whether detection was manual."""
        if manual_lookup:
            manual_bbox = self._consume_manual_bbox(manual_lookup, candidate)
            if manual_bbox is not None:
                return DetectionResult(
                    label=candidate.display_label,
                    bbox=manual_bbox,
                    confidence=1.0,
                ), True
        phrases = candidate.detector_phrases or [candidate.display_label]
        for phrase in phrases:
            detection = self._detector.detect_box(
                image,
                phrase,
                self._box_threshold,
                self._text_threshold,
            )
            if detection is not None:
                return detection, False
        return None, False

    def _build_manual_lookup(
        self,
        manual_detections: list[dict] | None,
    ) -> dict[str, list[tuple[int, int, int, int]]]:
        lookup: dict[str, list[tuple[int, int, int, int]]] = {}
        if not manual_detections:
            return lookup
        for detection in manual_detections:
            bbox = detection.get("bbox")
            label = str(detection.get("label", "")).strip().lower()
            if not label or not bbox:
                continue
            lookup.setdefault(label, []).append(tuple(bbox))
        return lookup

    def _consume_manual_bbox(
        self,
        manual_lookup: dict[str, list[tuple[int, int, int, int]]],
        candidate: InterrogationCandidate,
    ) -> tuple[int, int, int, int] | None:
        possible_labels = [
            candidate.display_label.strip().lower(),
            candidate.canonical_label.strip().lower(),
            *[phrase.strip().lower() for phrase in candidate.detector_phrases],
        ]
        for label in possible_labels:
            queue = manual_lookup.get(label)
            if queue:
                return queue.pop(0)
        return None

    def _child_candidate(
        self,
        parent: InterrogationCandidate,
        child_label: str,
    ) -> InterrogationCandidate:
        knowledge = self._knowledge_pack.find_object(child_label) if self._knowledge_pack else None
        detector_phrases = (
            knowledge.ranked_detector_phrases(4)
            if knowledge is not None
            else [child_label, f"{child_label} detail", f"{parent.display_label} {child_label}"]
        )
        return InterrogationCandidate(
            canonical_label=knowledge.canonical if knowledge else child_label,
            display_label=knowledge.canonical if knowledge else child_label,
            detector_phrases=detector_phrases,
            source_model="child",
            confidence=0.7,
            role="child",
            parent=parent.display_label,
        )
