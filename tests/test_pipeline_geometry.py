"""test_pipeline_geometry.py  --  the pure geometry and label helpers.

Masks, bounding boxes and filename-safe labels. Nothing here builds an
Orchestrator: these functions take arrays and tuples and return a value,
which is the whole reason they live in core/pipeline_geometry.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pipeline_geometry import (
    MAX_LABEL_FILENAME_LEN,
    bbox_area,
    bbox_iou,
    bbox_overlaps,
    clip_mask_to_bbox,
    crop_to_bbox,
    mask_containment,
    mask_iou,
    safe_filename_label,
)

IMG_SIZE = 64


class TestSafeFilenameLabel:
    def test_short_label_passes_through_unchanged(self) -> None:
        assert safe_filename_label("chalice") == "chalice"

    def test_replaces_path_separators(self) -> None:
        assert safe_filename_label("a/b\\c:d") == "a_b_c_d"

    def test_truncates_to_max_length(self) -> None:
        label = "x" * (MAX_LABEL_FILENAME_LEN + 50)
        safe = safe_filename_label(label)
        assert len(safe) <= MAX_LABEL_FILENAME_LEN

    def test_truncation_strips_trailing_dot_and_space(self) -> None:
        # Construct a label whose MAX_LABEL_FILENAME_LEN-th char lands on
        # a run of dots/spaces so rstrip has visible effect.
        label = ("a" * (MAX_LABEL_FILENAME_LEN - 3)) + "   ." + "y" * 20
        safe = safe_filename_label(label)
        assert not safe.endswith(" ")
        assert not safe.endswith(".")

    def test_historical_bug_long_moondream_list_label(self) -> None:
        # Regression test: Moondream once returned a ~300-char numbered-list
        # label that produced an OSError: File name too long on macOS.
        long_label = ", ".join(f"item {i}" for i in range(1, 40))
        assert len(long_label) > 200
        safe = safe_filename_label(long_label)
        assert len(safe) <= MAX_LABEL_FILENAME_LEN
        # Must still be usable as an actual filename component.
        assert "/" not in safe and "\\" not in safe

    def test_unsafe_characters_do_not_raise(self) -> None:
        weird = "object*?<>|\"'"
        # Should not raise; only "/", "\\", ":" are explicitly sanitized.
        result = safe_filename_label(weird)
        assert isinstance(result, str)


# ── Geometry helpers ─────────────────────────────────────────────────────


class TestGeometryHelpers:
    def test_mask_iou_identical_masks_is_one(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:6, 2:6] = 255
        assert mask_iou(mask, mask) == pytest.approx(1.0)

    def test_mask_iou_disjoint_masks_is_zero(self) -> None:
        a = np.zeros((10, 10), dtype=np.uint8)
        a[0:3, 0:3] = 255
        b = np.zeros((10, 10), dtype=np.uint8)
        b[7:10, 7:10] = 255
        assert mask_iou(a, b) == 0.0

    def test_mask_iou_empty_masks_returns_zero(self) -> None:
        a = np.zeros((10, 10), dtype=np.uint8)
        b = np.zeros((10, 10), dtype=np.uint8)
        assert mask_iou(a, b) == 0.0

    def test_mask_containment_full_subset(self) -> None:
        big = np.zeros((10, 10), dtype=np.uint8)
        big[0:10, 0:10] = 255
        small = np.zeros((10, 10), dtype=np.uint8)
        small[2:4, 2:4] = 255
        assert mask_containment(small, big) == pytest.approx(1.0)

    def test_mask_containment_zero_area_returns_zero(self) -> None:
        a = np.zeros((10, 10), dtype=np.uint8)
        b = np.zeros((10, 10), dtype=np.uint8)
        b[0:3, 0:3] = 255
        assert mask_containment(a, b) == 0.0

    def test_bbox_overlaps_true_for_touching_expanded_boxes(self) -> None:
        parent = (10, 10, 20, 20)
        child = (18, 18, 25, 25)
        assert bbox_overlaps(child, parent, img_h=64, img_w=64) is True

    def test_bbox_overlaps_false_when_far_apart(self) -> None:
        parent = (0, 0, 5, 5)
        child = (50, 50, 60, 60)
        assert bbox_overlaps(child, parent, img_h=64, img_w=64) is False

    def testbbox_area(self) -> None:
        assert bbox_area((0, 0, 10, 5)) == 50

    def test_bbox_area_degenerate_returns_zero(self) -> None:
        assert bbox_area((10, 10, 5, 5)) == 0

    def test_bbox_iou_full_overlap_is_one(self) -> None:
        box = (0, 0, 10, 10)
        assert bbox_iou(box, box) == pytest.approx(1.0)

    def test_bbox_iou_no_overlap_is_zero(self) -> None:
        a = (0, 0, 5, 5)
        b = (50, 50, 60, 60)
        assert bbox_iou(a, b) == 0.0

    def test_bbox_iou_partial_overlap(self) -> None:
        a = (0, 0, 10, 10)
        b = (5, 5, 15, 15)
        # intersection = 5x5=25, union = 100+100-25=175
        assert bbox_iou(a, b) == pytest.approx(25 / 175)

    def test_crop_to_bbox_applies_padding(self) -> None:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        crop, x0, y0 = crop_to_bbox(image, (10, 10, 20, 20), padding=5)
        assert x0 == 5 and y0 == 5
        assert crop.shape[:2] == (20, 20)

    def test_crop_to_bbox_clips_to_image_bounds(self) -> None:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        crop, x0, y0 = crop_to_bbox(image, (0, 0, 10, 10), padding=20)
        assert x0 == 0 and y0 == 0
        assert crop.shape[:2] == (30, 30)

    def test_clip_mask_to_bbox_zeroes_outside_region(self) -> None:
        mask = np.full((20, 20), 255, dtype=np.uint8)
        clipped = clip_mask_to_bbox(mask, (5, 5, 10, 10))
        assert clipped[0, 0] == 0
        assert clipped[7, 7] == 255
        assert clipped.sum() == 255 * 5 * 5
