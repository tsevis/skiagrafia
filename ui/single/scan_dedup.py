"""scan_dedup.py  --  Overlap-based deduplication of scan detections.

A scan can return two labels for the same physical object (e.g. "guitar"
and "urn" over the same region). These helpers collapse such pairs down to
the higher-confidence detection. Pure geometry -- no Tk involvement.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

SCAN_BBOX_IOU_THRESHOLD = 0.50
SCAN_CONTAINMENT_THRESHOLD = 0.70


def bbox_iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    """IoU of two (x0, y0, x1, y1) bounding boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union else 0.0


def bbox_containment(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    """Fraction of the smaller bbox's area that overlaps the larger one."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    smaller = min(area_a, area_b)
    return float(inter / smaller) if smaller else 0.0


def dedup_scan_detections(detections: list[dict]) -> list[dict]:
    """Remove duplicate scan detections whose bboxes overlap heavily.

    When two detections cover the same physical object (e.g. 'guitar'
    and 'urn'), keep only the one with the higher confidence.
    """
    if len(detections) <= 1:
        return detections

    # Sort by confidence descending so higher-confidence detections win
    ranked = sorted(detections, key=lambda d: d.get("confidence", 0), reverse=True)
    kept: list[dict] = []
    for det in ranked:
        bbox = det.get("bbox")
        if bbox is None:
            kept.append(det)
            continue
        is_dup = False
        for existing in kept:
            ex_bbox = existing.get("bbox")
            if ex_bbox is None:
                continue
            iou = bbox_iou(bbox, ex_bbox)
            containment = bbox_containment(bbox, ex_bbox)
            if iou > SCAN_BBOX_IOU_THRESHOLD or containment > SCAN_CONTAINMENT_THRESHOLD:
                logger.info(
                    "Scan dedup: dropping '%s' (iou=%.2f, contain=%.2f with '%s')",
                    det.get("label"), iou, containment, existing.get("label"),
                )
                is_dup = True
                break
        if not is_dup:
            kept.append(det)
    return kept
