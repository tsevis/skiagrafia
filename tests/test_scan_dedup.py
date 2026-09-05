"""test_scan_dedup.py  --  Overlap-based dedup of scan detections.

Pure geometry extracted from ui/single/left_panel.py; no Tk involved, so
these run in the default (non-gui) suite.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui.single.scan_dedup import (
    SCAN_BBOX_IOU_THRESHOLD,
    SCAN_CONTAINMENT_THRESHOLD,
    bbox_containment,
    bbox_iou,
    dedup_scan_detections,
)


class TestBboxIou:
    def test_identical_boxes_score_one(self) -> None:
        assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_disjoint_boxes_score_zero(self) -> None:
        assert bbox_iou((0, 0, 5, 5), (10, 10, 20, 20)) == 0.0

    def test_touching_edges_score_zero(self) -> None:
        assert bbox_iou((0, 0, 5, 5), (5, 0, 10, 5)) == 0.0

    def test_half_overlap(self) -> None:
        # intersection 50, union 150
        assert bbox_iou((0, 0, 10, 10), (5, 0, 15, 10)) == 50 / 150

    def test_zero_area_boxes_do_not_divide_by_zero(self) -> None:
        assert bbox_iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0

    def test_is_symmetric(self) -> None:
        a, b = (0, 0, 10, 10), (3, 3, 12, 12)
        assert bbox_iou(a, b) == bbox_iou(b, a)


class TestBboxContainment:
    def test_fully_nested_box_is_fully_contained(self) -> None:
        assert bbox_containment((0, 0, 10, 10), (2, 2, 4, 4)) == 1.0

    def test_disjoint_boxes_score_zero(self) -> None:
        assert bbox_containment((0, 0, 5, 5), (10, 10, 20, 20)) == 0.0

    def test_measured_against_the_smaller_box(self) -> None:
        # Half of the small box lies inside the large one.
        assert bbox_containment((0, 0, 10, 10), (5, 0, 15, 10)) == 0.5

    def test_zero_area_box_does_not_divide_by_zero(self) -> None:
        assert bbox_containment((0, 0, 10, 10), (5, 5, 5, 5)) == 0.0


class TestDedupScanDetections:
    @staticmethod
    def _det(label: str, bbox: tuple[int, int, int, int], conf: float) -> dict:
        return {"label": label, "bbox": bbox, "confidence": conf}

    def test_empty_and_single_pass_through(self) -> None:
        assert dedup_scan_detections([]) == []
        one = [self._det("guitar", (0, 0, 10, 10), 0.9)]
        assert dedup_scan_detections(one) == one

    def test_overlapping_pair_keeps_higher_confidence(self) -> None:
        dets = [
            self._det("urn", (0, 0, 10, 10), 0.4),
            self._det("guitar", (0, 0, 10, 10), 0.9),
        ]
        result = dedup_scan_detections(dets)
        assert [d["label"] for d in result] == ["guitar"]

    def test_distinct_objects_are_both_kept(self) -> None:
        dets = [
            self._det("guitar", (0, 0, 10, 10), 0.9),
            self._det("amp", (100, 100, 120, 120), 0.8),
        ]
        assert len(dedup_scan_detections(dets)) == 2

    def test_nested_box_is_dropped_by_containment(self) -> None:
        # IoU is low, but the small box sits entirely inside the big one.
        dets = [
            self._det("body", (0, 0, 100, 100), 0.9),
            self._det("speck", (10, 10, 20, 20), 0.5),
        ]
        result = dedup_scan_detections(dets)
        assert [d["label"] for d in result] == ["body"]

    def test_detection_without_bbox_is_always_kept(self) -> None:
        dets = [
            self._det("guitar", (0, 0, 10, 10), 0.9),
            {"label": "no-box", "confidence": 0.1},
        ]
        result = dedup_scan_detections(dets)
        assert {d["label"] for d in result} == {"guitar", "no-box"}

    def test_missing_confidence_defaults_and_does_not_raise(self) -> None:
        dets = [
            {"label": "a", "bbox": (0, 0, 10, 10)},
            self._det("b", (0, 0, 10, 10), 0.9),
        ]
        result = dedup_scan_detections(dets)
        assert [d["label"] for d in result] == ["b"]

    def test_output_is_ranked_by_confidence(self) -> None:
        dets = [
            self._det("low", (0, 0, 10, 10), 0.2),
            self._det("high", (200, 200, 210, 210), 0.9),
            self._det("mid", (100, 100, 110, 110), 0.5),
        ]
        result = dedup_scan_detections(dets)
        assert [d["label"] for d in result] == ["high", "mid", "low"]

    def test_input_list_is_not_mutated(self) -> None:
        dets = [
            self._det("urn", (0, 0, 10, 10), 0.4),
            self._det("guitar", (0, 0, 10, 10), 0.9),
        ]
        before = list(dets)
        dedup_scan_detections(dets)
        assert dets == before

    def test_thresholds_are_in_a_sane_range(self) -> None:
        assert 0.0 < SCAN_BBOX_IOU_THRESHOLD < 1.0
        assert 0.0 < SCAN_CONTAINMENT_THRESHOLD < 1.0
