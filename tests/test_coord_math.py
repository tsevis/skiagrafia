"""test_coord_math.py  --  Crop/canvas coordinate geometry helpers.

Pure numpy logic, no I/O or GUI involved.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.coord_math import crop_with_padding, remap_mask, tight_bbox


# ── remap_mask ───────────────────────────────────────────────────────────────


class TestRemapMask:
    def test_places_local_mask_inside_canvas(self) -> None:
        local = np.ones((3, 3), dtype=np.uint8)
        full = remap_mask(local, crop_origin=(2, 4), canvas_shape=(10, 10))

        assert full.shape == (10, 10)
        assert full[2:5, 4:7].sum() == 9
        assert full.sum() == 9

    def test_clips_when_crop_extends_past_bottom_right(self) -> None:
        local = np.ones((5, 5), dtype=np.uint8)
        full = remap_mask(local, crop_origin=(8, 8), canvas_shape=(10, 10))

        assert full.shape == (10, 10)
        # Only the (2, 2) bottom-right corner of the canvas is reachable.
        assert full[8:10, 8:10].sum() == 4
        assert full.sum() == 4

    def test_negative_origin_clips_top_left(self) -> None:
        local = np.ones((4, 4), dtype=np.uint8)
        full = remap_mask(local, crop_origin=(-2, -3), canvas_shape=(10, 10))

        # Only rows [0,2) and cols [0,1) of the local mask land on canvas.
        assert full.shape == (10, 10)
        assert full[0:2, 0:1].sum() == 2
        assert full.sum() == 2

    def test_origin_entirely_outside_canvas_yields_all_zero(self) -> None:
        local = np.ones((3, 3), dtype=np.uint8)
        full = remap_mask(local, crop_origin=(20, 20), canvas_shape=(10, 10))

        assert full.shape == (10, 10)
        assert full.sum() == 0

    def test_negative_origin_beyond_local_size_yields_all_zero(self) -> None:
        # Symmetric with test_origin_entirely_outside_canvas_yields_all_zero:
        # a crop lying entirely off the top/left contributes nothing rather
        # than raising on the clipped slice arithmetic.
        local = np.ones((3, 3), dtype=np.uint8)
        full = remap_mask(local, crop_origin=(-10, -10), canvas_shape=(10, 10))
        assert full.shape == (10, 10)
        assert not full.any()

    def test_partially_negative_origin_keeps_overlapping_region(self) -> None:
        local = np.ones((4, 4), dtype=np.uint8)
        full = remap_mask(local, crop_origin=(-2, -2), canvas_shape=(10, 10))
        # Only the bottom-right 2x2 of the local mask lands on the canvas.
        assert full[:2, :2].all()
        assert full.sum() == 4

    def test_off_canvas_on_one_axis_only_yields_all_zero(self) -> None:
        local = np.ones((3, 3), dtype=np.uint8)
        full = remap_mask(local, crop_origin=(-5, 2), canvas_shape=(10, 10))
        assert not full.any()

    def test_local_mask_larger_than_canvas_is_clipped_on_all_sides(self) -> None:
        local = np.ones((20, 20), dtype=np.uint8)
        full = remap_mask(local, crop_origin=(0, 0), canvas_shape=(5, 5))

        assert full.shape == (5, 5)
        assert full.sum() == 25

    def test_preserves_nonuniform_local_pattern(self) -> None:
        local = np.zeros((3, 3), dtype=np.uint8)
        local[0, 0] = 1
        local[2, 2] = 1
        full = remap_mask(local, crop_origin=(1, 1), canvas_shape=(6, 6))

        assert full[1, 1] == 1
        assert full[3, 3] == 1
        assert full.sum() == 2


# ── tight_bbox ───────────────────────────────────────────────────────────────


class TestTightBbox:
    def test_all_zero_mask_returns_full_extent(self) -> None:
        mask = np.zeros((12, 20), dtype=np.uint8)

        assert tight_bbox(mask, padding=15) == (0, 0, 12, 20)

    def test_single_pixel_mask_padded_and_clipped(self) -> None:
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[10, 10] = 1

        y0, x0, y1, x1 = tight_bbox(mask, padding=5)

        assert (y0, x0, y1, x1) == (5, 5, 16, 16)

    def test_padding_larger_than_bounds_clips_to_zero_and_shape(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5, 5] = 1

        y0, x0, y1, x1 = tight_bbox(mask, padding=1000)

        assert (y0, x0, y1, x1) == (0, 0, 10, 10)

    def test_zero_padding_returns_exact_extent(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 3:6] = 1

        y0, x0, y1, x1 = tight_bbox(mask, padding=0)

        assert (y0, x0, y1, x1) == (2, 3, 5, 6)

    def test_default_padding_is_fifteen(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 1

        y0, x0, y1, x1 = tight_bbox(mask)

        assert (y0, x0, y1, x1) == (35, 35, 66, 66)

    def test_mask_touching_top_left_corner_clips_padding_at_zero(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0, 0] = 1

        y0, x0, y1, x1 = tight_bbox(mask, padding=5)

        assert y0 == 0
        assert x0 == 0
        assert y1 == 6
        assert x1 == 6


# ── crop_with_padding ────────────────────────────────────────────────────────


class TestCropWithPadding:
    def test_crop_within_bounds_applies_padding_symmetrically(self) -> None:
        image = np.arange(400).reshape(20, 20).astype(np.uint8)
        crop, origin = crop_with_padding(image, bbox=(5, 5, 10, 10), padding=2)

        assert origin == (3, 3)
        assert crop.shape == (9, 9)
        np.testing.assert_array_equal(crop, image[3:12, 3:12])

    def test_padding_clipped_at_top_left_edge(self) -> None:
        image = np.arange(100).reshape(10, 10).astype(np.uint8)
        crop, origin = crop_with_padding(image, bbox=(1, 1, 4, 4), padding=15)

        assert origin == (0, 0)
        assert crop.shape == (10, 10)
        np.testing.assert_array_equal(crop, image)

    def test_padding_clipped_at_bottom_right_edge(self) -> None:
        image = np.arange(100).reshape(10, 10).astype(np.uint8)
        crop, origin = crop_with_padding(image, bbox=(6, 6, 9, 9), padding=15)

        assert origin == (0, 0)
        assert crop.shape == (10, 10)
        np.testing.assert_array_equal(crop, image)

    def test_zero_padding_returns_exact_bbox_region(self) -> None:
        image = np.arange(400).reshape(20, 20).astype(np.uint8)
        crop, origin = crop_with_padding(image, bbox=(3, 4, 8, 9), padding=0)

        assert origin == (3, 4)
        assert crop.shape == (5, 5)
        np.testing.assert_array_equal(crop, image[3:8, 4:9])

    def test_returned_crop_is_a_copy_not_a_view(self) -> None:
        image = np.zeros((10, 10), dtype=np.uint8)
        crop, _ = crop_with_padding(image, bbox=(2, 2, 5, 5), padding=0)
        crop[0, 0] = 255

        assert image[2, 2] == 0

    def test_bbox_with_color_channels_preserved(self) -> None:
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        image[4, 4] = (10, 20, 30)
        crop, origin = crop_with_padding(image, bbox=(4, 4, 5, 5), padding=1)

        assert origin == (3, 3)
        assert crop.shape == (3, 3, 3)
        np.testing.assert_array_equal(crop[1, 1], (10, 20, 30))
