"""test_image_filter.py -- bilateral smoothing, k-means quantization, masking.

All tests operate on small synthetic numpy arrays (no ML models, no I/O).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from processors.image_filter import (
    apply_mask_to_image,
    bilateral_smooth,
    kmeans_quantize,
)


# ── bilateral_smooth ─────────────────────────────────────────────────────────


class TestBilateralSmooth:
    def test_preserves_shape_and_dtype(self) -> None:
        image = np.random.default_rng(0).integers(
            0, 256, size=(16, 16, 3), dtype=np.uint8
        )
        result = bilateral_smooth(image)
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_flat_image_stays_flat(self) -> None:
        image = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = bilateral_smooth(image)
        assert np.array_equal(result, image)

    def test_custom_parameters_do_not_crash(self) -> None:
        image = np.random.default_rng(1).integers(
            0, 256, size=(8, 8, 3), dtype=np.uint8
        )
        result = bilateral_smooth(image, d=5, sigma_color=30.0, sigma_space=30.0)
        assert result.shape == image.shape


# ── kmeans_quantize ──────────────────────────────────────────────────────────


class TestKmeansQuantize:
    def test_result_has_at_most_k_unique_colours(self) -> None:
        image = np.random.default_rng(2).integers(
            0, 256, size=(20, 20, 3), dtype=np.uint8
        )
        result = kmeans_quantize(image, k=4)
        colours = {tuple(px) for px in result.reshape(-1, 3)}
        assert len(colours) <= 4

    def test_preserves_shape_and_dtype(self) -> None:
        image = np.random.default_rng(3).integers(
            0, 256, size=(12, 14, 3), dtype=np.uint8
        )
        result = kmeans_quantize(image, k=3)
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_k_larger_than_distinct_colours(self) -> None:
        # Only two distinct colours present, but k requests eight clusters.
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        image[:8] = [10, 20, 30]
        image[8:] = [200, 210, 220]

        result = kmeans_quantize(image, k=8)

        colours = {tuple(px) for px in result.reshape(-1, 3)}
        assert colours <= {(10, 20, 30), (200, 210, 220)}

    def test_single_colour_image_with_k_one(self) -> None:
        image = np.full((10, 10, 3), 77, dtype=np.uint8)
        result = kmeans_quantize(image, k=1)
        assert np.all(result == 77)


# ── apply_mask_to_image ──────────────────────────────────────────────────────


class TestApplyMaskToImage:
    def test_zeroes_out_masked_region_on_colour_image(self) -> None:
        image = np.full((10, 10, 3), 200, dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 2:5] = 255

        result = apply_mask_to_image(image, mask)

        assert np.all(result[2:5, 2:5] == 200)
        assert np.all(result[0, 0] == 0)

    def test_grayscale_image_masking(self) -> None:
        image = np.full((8, 8), 150, dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[0:4, 0:4] = 255

        result = apply_mask_to_image(image, mask)

        assert result.shape == (8, 8)
        assert result[1, 1] == 150
        assert result[6, 6] == 0

    def test_empty_mask_zeroes_everything(self) -> None:
        image = np.full((6, 6, 3), 100, dtype=np.uint8)
        mask = np.zeros((6, 6), dtype=np.uint8)

        result = apply_mask_to_image(image, mask)

        assert not result.any()

    def test_full_mask_keeps_image_unchanged(self) -> None:
        image = np.random.default_rng(4).integers(
            0, 256, size=(6, 6, 3), dtype=np.uint8
        )
        mask = np.full((6, 6), 255, dtype=np.uint8)

        result = apply_mask_to_image(image, mask)

        assert np.array_equal(result, image)
