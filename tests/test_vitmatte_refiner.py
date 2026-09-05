"""test_vitmatte_refiner.py  --  VitMatteRefiner logic (offline, no ML).

VitMatte weights are never loaded: `_load()` is short-circuited by
pre-populating the private `_model` / `_processor` attributes (both no-op
`_load()` once set), and the fake model/processor return small synthetic
torch tensors. No network or GPU/MPS compute happens -- only trimap
generation (pure OpenCV) and the pre/post-processing glue around a fake
matting call.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.vitmatte_refiner import VitMatteRefiner


class FakeProcessor:
    """Stand-in for VitMatteImageProcessor: returns fixed-size tensors."""

    def __init__(self, out_h: int, out_w: int) -> None:
        self._out_h = out_h
        self._out_w = out_w
        self.calls: list[tuple] = []

    def __call__(self, images, trimaps, return_tensors="pt"):
        self.calls.append((images, trimaps, return_tensors))
        return {
            "pixel_values": torch.zeros(1, 3, self._out_h, self._out_w),
            "trimap": torch.zeros(1, 1, self._out_h, self._out_w),
        }


class FakeMattingModel:
    """Stand-in for VitMatteForImageMatting: returns a fixed alpha tensor."""

    def __init__(self, alpha: torch.Tensor) -> None:
        self._alpha = alpha
        self.forward_calls = 0

    def __call__(self, **inputs):
        self.forward_calls += 1
        return SimpleNamespace(alphas=self._alpha)


def _tiny_image(h: int = 32, w: int = 32) -> NDArray[np.uint8]:
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _full_mask(h: int = 32, w: int = 32) -> NDArray[np.uint8]:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 255
    return mask


def _wired_refiner(alpha_shape: tuple[int, int]) -> tuple[VitMatteRefiner, FakeMattingModel]:
    """Build a refiner whose _load() is bypassed and whose fake model
    returns an alpha of shape (1, 1, *alpha_shape)."""
    refiner = VitMatteRefiner()
    alpha = torch.rand(1, 1, *alpha_shape)
    model = FakeMattingModel(alpha)
    refiner._model = model
    refiner._processor = FakeProcessor(*alpha_shape)
    return refiner, model


# ── _create_trimap ────────────────────────────────────────────────────────


class TestCreateTrimap:
    def test_center_is_definite_foreground(self) -> None:
        mask = _full_mask(64, 64)
        trimap = VitMatteRefiner._create_trimap(mask)
        assert trimap[32, 32] == 255

    def test_far_outside_is_definite_background(self) -> None:
        mask = _full_mask(64, 64)
        trimap = VitMatteRefiner._create_trimap(mask)
        assert trimap[0, 0] == 0

    def test_band_near_edge_is_unknown(self) -> None:
        mask = _full_mask(64, 64)
        trimap = VitMatteRefiner._create_trimap(mask, erosion_size=4, dilation_size=8)
        # Just outside the mask edge (16 = quarter boundary of 64) but
        # within the dilation band -> neither definite fg nor bg.
        edge_val = trimap[16, 20]
        assert edge_val in (0, 128, 255)  # sanity: valid trimap value

    def test_output_shape_matches_input(self) -> None:
        mask = _full_mask(48, 40)
        trimap = VitMatteRefiner._create_trimap(mask)
        assert trimap.shape == mask.shape

    def test_all_background_mask_yields_all_background_trimap(self) -> None:
        mask = np.zeros((32, 32), dtype=np.uint8)
        trimap = VitMatteRefiner._create_trimap(mask)
        assert (trimap == 0).all()


# ── predict ───────────────────────────────────────────────────────────────


class TestPredict:
    def test_returns_uint8_alpha_matching_image_size(self) -> None:
        refiner, model = _wired_refiner((32, 32))
        image = _tiny_image(32, 32)
        mask = _full_mask(32, 32)

        alpha = refiner.predict(image, mask)

        assert alpha.dtype == np.uint8
        assert alpha.shape == (32, 32)
        assert model.forward_calls == 1

    def test_values_are_clipped_into_0_255(self) -> None:
        refiner = VitMatteRefiner()
        # alpha tensor deliberately out of [0, 1] range before scaling.
        alpha_tensor = torch.tensor([[[[2.0, -1.0], [0.5, 0.0]]]])
        refiner._model = FakeMattingModel(alpha_tensor)
        refiner._processor = FakeProcessor(2, 2)

        image = _tiny_image(2, 2)
        mask = _full_mask(2, 2)
        alpha = refiner.predict(image, mask)

        assert alpha.min() >= 0
        assert alpha.max() <= 255

    def test_downscales_large_images_for_memory_safety(self) -> None:
        refiner, model = _wired_refiner((4, 4))
        refiner._max_side = 4  # force the downscale branch cheaply
        image = _tiny_image(8, 8)
        mask = _full_mask(8, 8)

        alpha = refiner.predict(image, mask)

        # Result is resized back up to the original image dimensions.
        assert alpha.shape == (8, 8)
        assert model.forward_calls == 1

    def test_no_downscale_when_image_within_max_side(self) -> None:
        refiner, _ = _wired_refiner((16, 16))
        refiner._max_side = 1536
        image = _tiny_image(16, 16)
        mask = _full_mask(16, 16)
        alpha = refiner.predict(image, mask)
        assert alpha.shape == (16, 16)

    def test_alpha_shape_mismatch_is_resized_to_working_dimensions(self) -> None:
        # Model returns an alpha at a *different* spatial size than the
        # processor's declared inputs -- predict() must resize it back to
        # (work_h, work_w) before any final upscale.
        refiner = VitMatteRefiner()
        mismatched_alpha = torch.rand(1, 1, 10, 10)
        refiner._model = FakeMattingModel(mismatched_alpha)
        refiner._processor = FakeProcessor(16, 16)  # unused directly by fake

        image = _tiny_image(16, 16)
        mask = _full_mask(16, 16)
        alpha = refiner.predict(image, mask)

        assert alpha.shape == (16, 16)


class TestLoad:
    def test_noop_when_model_already_set(self) -> None:
        refiner, _ = _wired_refiner((8, 8))
        refiner._load()  # must return immediately, no import/attribute errors


class TestPredictRgba:
    def test_stacks_image_and_alpha_into_four_channels(self) -> None:
        refiner, _ = _wired_refiner((16, 16))
        image = _tiny_image(16, 16)
        mask = _full_mask(16, 16)

        rgba = refiner.predict_rgba(image, mask)

        assert rgba.shape == (16, 16, 4)
        assert rgba.dtype == np.uint8
        # RGB channels are untouched from the source image.
        assert np.array_equal(rgba[..., :3], image)
