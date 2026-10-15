"""test_mask_ops.py -- boolean mask algebra and mask utility functions.

All tests operate on small synthetic numpy arrays (no ML models, no I/O).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from processors.mask_ops import (
    boolean_intersect,
    boolean_subtract,
    boolean_union,
    edge_refine,
    mask_bbox,
    mask_coverage,
    refine_mask,
    solidify_mask,
)

# ── Boolean mask algebra ─────────────────────────────────────────────────────


class TestBooleanSubtract:
    def test_removes_child_region_from_parent(self) -> None:
        parent = np.zeros((10, 10), dtype=np.uint8)
        parent[2:8, 2:8] = 255
        child = np.zeros((10, 10), dtype=np.uint8)
        child[4:6, 4:6] = 255

        result = boolean_subtract(parent, child)

        assert result[5, 5] == 0
        assert result[2, 2] == 255
        assert result.dtype == np.uint8

    def test_child_outside_parent_has_no_effect(self) -> None:
        parent = np.zeros((10, 10), dtype=np.uint8)
        parent[0:3, 0:3] = 255
        child = np.zeros((10, 10), dtype=np.uint8)
        child[7:9, 7:9] = 255

        result = boolean_subtract(parent, child)

        assert np.array_equal(result, parent)

    def test_empty_masks_produce_empty_result(self) -> None:
        empty = np.zeros((5, 5), dtype=np.uint8)
        result = boolean_subtract(empty, empty)
        assert not result.any()


class TestBooleanUnion:
    def test_combines_disjoint_regions(self) -> None:
        a = np.zeros((6, 6), dtype=np.uint8)
        a[0:2, 0:2] = 255
        b = np.zeros((6, 6), dtype=np.uint8)
        b[4:6, 4:6] = 255

        result = boolean_union(a, b)

        assert result[0, 0] == 255
        assert result[5, 5] == 255
        assert result[3, 3] == 0

    def test_overlapping_regions_stay_set(self) -> None:
        a = np.zeros((4, 4), dtype=np.uint8)
        a[0:3, 0:3] = 255
        b = np.zeros((4, 4), dtype=np.uint8)
        b[2:4, 2:4] = 255

        result = boolean_union(a, b)

        assert result[2, 2] == 255
        assert result[3, 3] == 255


class TestBooleanIntersect:
    def test_keeps_only_overlap(self) -> None:
        a = np.zeros((6, 6), dtype=np.uint8)
        a[0:4, 0:4] = 255
        b = np.zeros((6, 6), dtype=np.uint8)
        b[2:6, 2:6] = 255

        result = boolean_intersect(a, b)

        assert result[3, 3] == 255
        assert result[0, 0] == 0
        assert result[5, 5] == 0

    def test_disjoint_masks_intersect_to_empty(self) -> None:
        a = np.zeros((6, 6), dtype=np.uint8)
        a[0:2, 0:2] = 255
        b = np.zeros((6, 6), dtype=np.uint8)
        b[4:6, 4:6] = 255

        result = boolean_intersect(a, b)

        assert not result.any()


# ── refine_mask ──────────────────────────────────────────────────────────────


class TestRefineMask:
    def test_removes_speckles_below_min_area(self) -> None:
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[10:30, 10:30] = 255  # big blob, area 400
        mask[2:5, 2:5] = 255  # speckle, area 9 < default min 64

        result = refine_mask(mask, min_contour_area=64)

        assert result[3, 3] == 0
        assert result[20, 20] == 255

    def test_keeps_contour_at_or_above_min_area(self) -> None:
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[5:15, 5:15] = 255  # area 100

        result = refine_mask(mask, min_contour_area=64)

        assert result[10, 10] == 255

    def test_empty_mask_stays_empty(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        result = refine_mask(mask)
        assert not result.any()

    def test_single_pixel_mask_is_removed_as_speckle(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[10, 10] = 255

        result = refine_mask(mask, min_contour_area=1)

        # A single pixel has near-zero contour area, so it is dropped even
        # with a minimal threshold.
        assert mask_coverage(result) <= mask_coverage(mask)

    def test_mask_touching_border_does_not_crash(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[0:20, 0:5] = 255  # touches top/left/bottom edges

        result = refine_mask(mask, min_contour_area=64)

        assert result.shape == mask.shape


# ── edge_refine ──────────────────────────────────────────────────────────────


class TestEdgeRefine:
    def test_zero_iterations_returns_mask_unchanged(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:7, 3:7] = 255

        result = edge_refine(mask, iterations=0)

        assert result is mask

    def test_negative_iterations_returns_mask_unchanged(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:7, 3:7] = 255

        result = edge_refine(mask, iterations=-1)

        assert result is mask

    def test_positive_iterations_shrinks_region(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:7, 3:7] = 255

        result = edge_refine(mask, iterations=1)

        assert mask_coverage(result) < mask_coverage(mask)

    def test_more_iterations_shrink_further(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:18, 2:18] = 255

        once = edge_refine(mask, iterations=1)
        twice = edge_refine(mask, iterations=3)

        assert mask_coverage(twice) <= mask_coverage(once)

    def test_empty_mask_stays_empty(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        result = edge_refine(mask, iterations=2)
        assert not result.any()


# ── mask_bbox ────────────────────────────────────────────────────────────────


class TestMaskBbox:
    def test_empty_mask_returns_full_frame(self) -> None:
        mask = np.zeros((8, 12), dtype=np.uint8)
        assert mask_bbox(mask) == (0, 0, 8, 12)

    def test_single_pixel_mask(self) -> None:
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 3] = 255
        assert mask_bbox(mask) == (2, 3, 3, 4)

    def test_mask_touching_all_borders(self) -> None:
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[0, 0] = 255
        mask[4, 4] = 255
        assert mask_bbox(mask) == (0, 0, 5, 5)

    def test_rectangular_region(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 3:9] = 255
        assert mask_bbox(mask) == (2, 3, 5, 9)


# ── mask_coverage ────────────────────────────────────────────────────────────


class TestMaskCoverage:
    def test_empty_mask_is_zero_percent(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        assert mask_coverage(mask) == pytest.approx(0.0)

    def test_full_mask_is_hundred_percent(self) -> None:
        mask = np.full((10, 10), 255, dtype=np.uint8)
        assert mask_coverage(mask) == pytest.approx(100.0)

    def test_half_mask_is_fifty_percent(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[:5, :] = 255
        assert mask_coverage(mask) == pytest.approx(50.0)

    def test_single_pixel_mask_coverage(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0, 0] = 255
        assert mask_coverage(mask) == pytest.approx(1.0)


# ── Solid masks (opt-in) ─────────────────────────────────────────────────────


def _phone_like(size: int = 200) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[20:180, 40:160] = 255
    return mask


class TestSolidifyMask:
    def test_fills_a_reflection_speck_inside_the_object(self) -> None:
        mask = _phone_like()
        mask[60:66, 80:86] = 0  # 36 px against 19,200: a reflection, not a hole

        assert solidify_mask(mask)[63, 83] == 255

    def test_keeps_a_hole_too_large_to_be_a_speck(self) -> None:
        # A letter's counter is a real hole: here 40x40 = 8% of the object.
        mask = _phone_like()
        mask[80:120, 80:120] = 0

        assert solidify_mask(mask)[100, 100] == 0

    def test_closes_most_of_a_narrow_bite_out_of_the_edge(self) -> None:
        # Open to the background, so no hole fill reaches it. Closing leaves a
        # sliver at the bite's mouth; at this size about one pixel deep.
        mask = np.zeros((700, 700), dtype=np.uint8)
        mask[50:650, 150:550] = 255
        mask[300:308, 150:160] = 0

        still_open = int((solidify_mask(mask)[300:308, 150:160] == 0).sum())

        assert still_open <= 16  # of 80

    def test_fills_a_deep_bite_that_closing_cannot_reach(self) -> None:
        # The iPhone front: a bite 13.9 px deep on an object 319 px wide.
        mask = np.zeros((700, 700), dtype=np.uint8)
        mask[50:650, 150:470] = 255
        mask[300:312, 150:164] = 0

        assert int((solidify_mask(mask)[300:312, 150:164] == 0).sum()) == 0

    def test_keeps_a_genuinely_concave_shape(self) -> None:
        # A U: its opening is 40% of the object's width deep.
        mask = np.zeros((400, 400), dtype=np.uint8)
        mask[50:350, 50:350] = 255
        mask[50:170, 150:250] = 0

        assert solidify_mask(mask)[100, 200] == 0
