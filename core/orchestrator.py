"""orchestrator.py  --  v5.0  (Contracts & DI Refactor)
Single-image pipeline manager for Skiagrafia.

Steps 0-9  = Structural branch.
Steps 10-12 = Stylization branch (only when stylizer is provided).

The Orchestrator no longer imports or instantiates concrete model clients.
It receives a CapabilitySet via constructor injection.

All inference is local-only; no network calls.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from core.contracts import CapabilitySet
from core.interrogation import InterrogationCandidate
from core.knowledge import KnowledgePack
from models.grounded_sam import DetectionResult
from processors.mask_ops import boolean_subtract, mask_coverage, refine_mask
from processors.vectorizer import assemble_svg
from processors.output_writer import write_svg, write_tiff
from utils.coord_math import tight_bbox

logger = logging.getLogger(__name__)

MIN_CHILD_COVERAGE_PCT = 0.5
MAX_CHILD_PARENT_IOU = 0.85
MAX_CHILD_CHILD_IOU = 0.80
MIN_PARENT_COVERAGE_PCT = 0.5
MIN_CONFIRMED_COVERAGE_PCT = 0.05
PARENT_IOU_MERGE = 0.50
PARENT_CONTAINMENT_MERGE = 0.75
BBOX_IOU_MERGE = 0.60
BBOX_EXPAND_RATIO = 0.30
MAX_LABEL_FILENAME_LEN = 60  # max chars of a label used in output filenames


def _safe_filename_label(label: str) -> str:
    """Truncate and sanitize a label for use in output filenames.

    Moondream can return extremely long child labels (e.g. numbered
    lists of 38 items).  macOS enforces a 255-byte filename limit.
    """
    # Replace path-unsafe characters
    safe = label.replace("/", "_").replace("\\", "_").replace(":", "_")
    if len(safe) > MAX_LABEL_FILENAME_LEN:
        safe = safe[:MAX_LABEL_FILENAME_LEN].rstrip(". ")
    return safe

STRUCTURAL_STEPS = [
    "Loading image",
    "Moondream interrogation",
    "GroundingDINO detection",
    "SAM 2.1 — parent mask",
    "SAM 2.1 — child masks",
    "Coordinate remapping",
    "VitMatte alpha refinement",
    "Mask refinement",
    "VTracer vectorization",
    "Structural SVG assembly & export",
]
PIPELINE_STEPS = STRUCTURAL_STEPS


class LayerResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

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
    error: str | None = None


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

    def _report(self, step: int, msg: str | None = None) -> None:
        text = msg or PIPELINE_STEPS[step]
        self._progress(step, text)
        logger.info("Step %d/%d: %s", step + 1, len(PIPELINE_STEPS), text)

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
        except Exception as exc:
            result.error = str(exc)
            logger.error("Pipeline failed for %s: %s", image_path, exc, exc_info=True)

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

        # STEP 0 — Load
        self._report(0)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        result.width, result.height = w, h
        manual_lookup = self._build_manual_lookup(manual_detections)

        # STEP 1 — Moondream
        self._report(1)
        parents, children_by_parent = self._interrogate(image, confirmed_labels)

        masks: dict[str, NDArray[np.uint8]] = {}
        bboxes: dict[str, tuple[int, int, int, int]] = {}
        svg_layers: list[dict] = []
        accepted_parents: list[str] = []
        fallback_bbox_parents: set[str] = set()  # parents that used full-image bbox fallback

        for parent in parents:
            # STEP 2 — GroundingDINO
            self._report(2, f"GroundingDINO: {parent.display_label}")
            det_result, is_manual = self._detect_candidate_ex(image, parent, manual_lookup)
            used_fallback_bbox = False
            if det_result is None:
                if parent.source_model == "confirmed":
                    # User-confirmed label: fall back to full image bbox
                    logger.info(
                        "No detection for confirmed label '%s', using full image bbox",
                        parent.display_label,
                    )
                    det_result = DetectionResult(
                        label=parent.display_label,
                        bbox=(0, 0, w, h),
                        confidence=0.5,
                    )
                    used_fallback_bbox = True
                else:
                    logger.warning("No detection for '%s', skipping", parent.display_label)
                    continue

            # STEP 3 — SAM parent
            self._report(3, f"SAM 2.1 — {parent.display_label}")
            parent_mask = self._segmenter.segment(
                image, det_result.bbox, parent.display_label,
                prefer_full_box=is_manual,
            )
            if is_manual:
                parent_mask = _clip_mask_to_bbox(parent_mask, det_result.bbox)
            coverage = mask_coverage(parent_mask)
            min_coverage = (
                MIN_CONFIRMED_COVERAGE_PCT
                if parent.source_model == "confirmed"
                else MIN_PARENT_COVERAGE_PCT
            )
            if coverage < min_coverage:
                logger.warning(
                    "Mask for '%s' too small (%.1f%% < %.1f%%), skipping",
                    parent.display_label, coverage, min_coverage,
                )
                continue

            # IoU dedup — merge duplicate detections of the same physical
            # object.  Three complementary checks, any one triggers merge:
            #   1. Mask pixel IoU > 0.50
            #   2. Smaller mask ≥75% contained inside the larger mask
            #   3. Detection bounding-box IoU > 0.60
            # Check (2) catches the common "urn"/"vase" vs "guitar" case
            # where SAM segments the same area but the IoU is low because
            # one mask is a strict subset of the other.
            skip = False
            # Convert tight_bbox (y0,x0,y1,x1) to (x0,y0,x1,y1) for comparison
            det_bbox_xyxy = det_result.bbox  # already (x0,y0,x1,y1)
            for ex_lbl in list(accepted_parents):
                m_iou = _mask_iou(parent_mask, masks[ex_lbl])
                m_contain = _mask_containment(parent_mask, masks[ex_lbl])
                # Convert stored tight_bbox (y0,x0,y1,x1) → (x0,y0,x1,y1)
                ey0, ex0, ey1, ex1 = bboxes[ex_lbl]
                ex_bbox_xyxy = (ex0, ey0, ex1, ey1)
                b_iou = _bbox_iou(det_bbox_xyxy, ex_bbox_xyxy)
                logger.info(
                    "Dedup check '%s' vs '%s': mask_iou=%.3f, containment=%.3f, "
                    "bbox_iou=%.3f | det_bbox=%s, ex_bbox=%s",
                    parent.display_label, ex_lbl, m_iou, m_contain, b_iou,
                    det_bbox_xyxy, ex_bbox_xyxy,
                )
                is_dup = (
                    m_iou > PARENT_IOU_MERGE
                    or m_contain > PARENT_CONTAINMENT_MERGE
                    or b_iou > BBOX_IOU_MERGE
                )
                if not is_dup:
                    continue
                logger.info(
                    "Dedup '%s' vs '%s': mask_iou=%.2f, containment=%.2f, bbox_iou=%.2f",
                    parent.display_label, ex_lbl, m_iou, m_contain, b_iou,
                )
                ex_is_fallback = ex_lbl in fallback_bbox_parents
                if ex_is_fallback and not used_fallback_bbox:
                    # Current has a real detection — evict the fallback entry
                    logger.info(
                        "Replacing fallback-bbox '%s' with properly-detected '%s'",
                        ex_lbl, parent.display_label,
                    )
                    accepted_parents.remove(ex_lbl)
                    fallback_bbox_parents.discard(ex_lbl)
                    del masks[ex_lbl]
                    del bboxes[ex_lbl]
                    result.layers = [l for l in result.layers if l.label != ex_lbl]
                    # Merge children from evicted parent
                    extra = children_by_parent.get(ex_lbl, [])
                    existing_c = set(children_by_parent.get(parent.display_label, []))
                    children_by_parent.setdefault(parent.display_label, []).extend(
                        [c for c in extra if c not in existing_c]
                    )
                else:
                    # Existing wins (both proper, or current is fallback)
                    logger.info(
                        "Keeping '%s', dropping duplicate '%s'",
                        ex_lbl, parent.display_label,
                    )
                    extra = children_by_parent.get(parent.display_label, [])
                    existing_c = set(children_by_parent.get(ex_lbl, []))
                    children_by_parent.setdefault(ex_lbl, []).extend(
                        [c for c in extra if c not in existing_c]
                    )
                    skip = True
                break
            if skip:
                continue

            if used_fallback_bbox:
                fallback_bbox_parents.add(parent.display_label)

            masks[parent.display_label] = parent_mask
            bboxes[parent.display_label] = tight_bbox(parent_mask)
            accepted_parents.append(parent.display_label)
            result.layers.append(
                LayerResult(label=parent.display_label, role="parent", bbox=det_result.bbox)
            )

            # STEP 4 — Children
            children = children_by_parent.get(parent.display_label, [])
            if children:
                self._report(4, f"SAM 2.1 — {len(children)} children of '{parent.display_label}'")
                body_mask = parent_mask.copy()
                accepted_child_masks: list[tuple[str, NDArray[np.uint8]]] = []
                for child_label in children:
                    child_det = self._detect_candidate(
                        image,
                        self._child_candidate(parent, child_label),
                        None,
                    )
                    if child_det is None:
                        continue
                    if not _bbox_overlaps(child_det.bbox, det_result.bbox, h, w):
                        continue
                    child_mask = self._segmenter.segment(image, child_det.bbox, child_label)
                    if mask_coverage(child_mask) < MIN_CHILD_COVERAGE_PCT:
                        continue

                    # Reject child if its mask is too similar to the parent
                    # (GroundingDINO couldn't isolate the sub-part)
                    parent_iou = _mask_iou(child_mask, parent_mask)
                    if parent_iou > MAX_CHILD_PARENT_IOU:
                        logger.info(
                            "Child '%s' mask too similar to parent '%s' "
                            "(iou=%.2f > %.2f), skipping",
                            child_label, parent.display_label,
                            parent_iou, MAX_CHILD_PARENT_IOU,
                        )
                        continue

                    # Reject child if its mask duplicates an already-accepted child
                    child_dup = False
                    for ex_child_label, ex_child_mask in accepted_child_masks:
                        if _mask_iou(child_mask, ex_child_mask) > MAX_CHILD_CHILD_IOU:
                            logger.info(
                                "Child '%s' duplicates '%s' (iou > %.2f), skipping",
                                child_label, ex_child_label, MAX_CHILD_CHILD_IOU,
                            )
                            child_dup = True
                            break
                    if child_dup:
                        continue

                    accepted_child_masks.append((child_label, child_mask))
                    masks[child_label] = child_mask
                    body_mask = boolean_subtract(body_mask, child_mask)
                    result.layers.append(
                        LayerResult(label=child_label, role="child", parent_label=parent.display_label, bbox=child_det.bbox)
                    )
                masks[f"{parent.display_label}_body"] = body_mask

        self._report(5)  # coordinate remapping (logged)

        # STEP 6 — VitMatte
        if "bitmap" in self._output_mode:
            self._report(6)
            for label, mask in masks.items():
                alpha = self._alpha_refiner.predict(image, mask)
                tiff_path = self._output_dir / f"{image_path.stem}_{_safe_filename_label(label)}.tiff"
                write_tiff(image, tiff_path, alpha)
                if result.tiff_path is None:
                    result.tiff_path = str(tiff_path.parent)
        else:
            self._report(6, "VitMatte skipped (vector-only mode)")

        # STEP 7 — Refinement
        self._report(7)
        refined: dict[str, NDArray[np.uint8]] = {
            lbl: refine_mask(msk, bilateral_d=self._bilateral_d)
            for lbl, msk in masks.items()
        }

        # STEP 8 — VTracer
        self._report(8)
        for layer in result.layers:
            if layer.label not in refined:
                continue
            svg_data = self._vectorizer.trace(refined[layer.label])
            layer.svg_data = svg_data
            svg_layers.append({"id": _safe_filename_label(layer.label).replace(" ", "_"), "svg_data": svg_data, "dx": layer.dx, "dy": layer.dy})

        # STEP 9 — Structural SVG export
        self._report(9)
        if svg_layers:
            full_svg = assemble_svg(w, h, svg_layers)
            svg_path = self._output_dir / f"{image_path.stem}.svg"
            write_svg(full_svg, svg_path)
            result.svg_path = str(svg_path)

        self._segmenter.clear_cache()
        logger.info("Structural branch complete: %s", image_path.name)
        return result

    # ─────────────────────────────────────────────────────────────────────
    # Moondream helper
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
