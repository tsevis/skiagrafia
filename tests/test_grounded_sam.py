"""test_grounded_sam.py  --  GroundedSAM detect/segment orchestration logic.

All tests are offline: SAM 2.1 and GroundingDINO weights are never loaded.
`_load_dino` / `_load_sam` are short-circuited by pre-populating the private
`_dino_model` / `_sam_predictor` attributes (both no-op when already set),
and the `grounding_dino` package -- which is not a pip dependency and is
only ever vendored alongside the real model weights -- is stubbed into
`sys.modules` so `detect_box` can run its pure bbox-conversion logic with a
fake `predict()` function. No network or GPU work happens anywhere here.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.grounded_sam import (
    DetectionResult,
    GroundedSAM,
    _ensure_gsam_on_path,
    _patch_bert_head_mask,
    _patch_get_extended_attention_mask,
    _patch_onnx_ml_dtypes,
)


IMG_SIZE = 64


def _tiny_image(size: int = IMG_SIZE) -> NDArray[np.uint8]:
    return np.zeros((size, size, 3), dtype=np.uint8)


class FakeSamPredictor:
    """Stand-in for sam2.sam2_image_predictor.SAM2ImagePredictor."""

    def __init__(self, masks: NDArray | None = None) -> None:
        self.set_image_calls: list[NDArray] = []
        self.predict_calls: list[dict] = []
        self._masks = masks

    def set_image(self, image: NDArray) -> None:
        self.set_image_calls.append(image)

    def predict(self, box, multimask_output: bool):
        self.predict_calls.append({"box": box, "multimask_output": multimask_output})
        if self._masks is not None:
            masks = self._masks
        elif multimask_output:
            masks = np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=bool)
            masks[0, 0:10, 0:10] = True   # small fill inside any bbox
            masks[1, 0:40, 0:40] = True   # best fill for a bbox like (0,0,40,40)
            masks[2, :, :] = True         # full-image fill
        else:
            masks = np.zeros((1, IMG_SIZE, IMG_SIZE), dtype=bool)
            masks[0, 5:20, 5:20] = True
        scores = np.ones(masks.shape[0], dtype=np.float32)
        return masks, scores, None


def _install_fake_grounding_dino(monkeypatch: pytest.MonkeyPatch, predict_fn) -> None:
    """Insert a minimal fake `grounding_dino` package tree into sys.modules."""

    class FakeCompose:
        def __init__(self, transforms) -> None:
            self.transforms = transforms

        def __call__(self, image, target):
            return image, target

    class _NoOpTransform:
        def __init__(self, *args, **kwargs) -> None:
            pass

    inference_mod = types.ModuleType("grounding_dino.groundingdino.util.inference")
    inference_mod.predict = predict_fn

    transforms_mod = types.ModuleType("grounding_dino.groundingdino.datasets.transforms")
    transforms_mod.Compose = FakeCompose
    transforms_mod.RandomResize = _NoOpTransform
    transforms_mod.ToTensor = _NoOpTransform
    transforms_mod.Normalize = _NoOpTransform

    util_mod = types.ModuleType("grounding_dino.groundingdino.util")
    util_mod.inference = inference_mod
    datasets_mod = types.ModuleType("grounding_dino.groundingdino.datasets")
    datasets_mod.transforms = transforms_mod
    groundingdino_mod = types.ModuleType("grounding_dino.groundingdino")
    groundingdino_mod.util = util_mod
    groundingdino_mod.datasets = datasets_mod
    grounding_dino_mod = types.ModuleType("grounding_dino")
    grounding_dino_mod.groundingdino = groundingdino_mod

    for name, mod in {
        "grounding_dino": grounding_dino_mod,
        "grounding_dino.groundingdino": groundingdino_mod,
        "grounding_dino.groundingdino.util": util_mod,
        "grounding_dino.groundingdino.util.inference": inference_mod,
        "grounding_dino.groundingdino.datasets": datasets_mod,
        "grounding_dino.groundingdino.datasets.transforms": transforms_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)


def _make_gsam_with_dino_stubbed() -> GroundedSAM:
    gsam = GroundedSAM()
    gsam._dino_model = object()  # short-circuits _load_dino()
    return gsam


# ── Compatibility-shim patch functions (pure logic, no model weights) ────
#
# These wrap third-party library internals (transformers' BertModel /
# ModuleUtilsMixin) for GroundingDINO compatibility. They are exercised
# against fake stand-in classes injected into sys.modules so the *real*
# transformers classes are never mutated -- this only tests the module's
# own patching logic, never touches ML weights, and monkeypatch restores
# sys.modules automatically after each test.


class TestPatchOnnxMlDtypes:
    def test_noop_when_ml_dtypes_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "ml_dtypes", None)
        _patch_onnx_ml_dtypes()  # must not raise

    def test_sets_missing_fallback_attrs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_ml_dtypes = types.SimpleNamespace(
            float8_e4m3fn="e4m3fn", float8_e5m2="e5m2"
        )
        monkeypatch.setitem(sys.modules, "ml_dtypes", fake_ml_dtypes)
        _patch_onnx_ml_dtypes()
        assert fake_ml_dtypes.float4_e2m1fn == "e4m3fn"
        assert fake_ml_dtypes.float8_e8m0fnu == "e5m2"

    def test_does_not_overwrite_existing_attrs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_ml_dtypes = types.SimpleNamespace(
            float4_e2m1fn="already-there",
            float8_e4m3fn="e4m3fn",
            float8_e8m0fnu="already-there-2",
            float8_e5m2="e5m2",
        )
        monkeypatch.setitem(sys.modules, "ml_dtypes", fake_ml_dtypes)
        _patch_onnx_ml_dtypes()
        assert fake_ml_dtypes.float4_e2m1fn == "already-there"
        assert fake_ml_dtypes.float8_e8m0fnu == "already-there-2"


class TestPatchBertHeadMask:
    def _fake_bert_module(self):
        class FakeBertModel:
            pass

        module = types.ModuleType("transformers.models.bert.modeling_bert")
        module.BertModel = FakeBertModel
        return module, FakeBertModel

    def test_noop_when_transformers_bert_unimportable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(
            sys.modules, "transformers.models.bert.modeling_bert", None
        )
        _patch_bert_head_mask()  # must not raise

    def test_skips_when_attribute_already_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module, FakeBertModel = self._fake_bert_module()
        sentinel = object()
        FakeBertModel.get_head_mask = sentinel
        monkeypatch.setitem(
            sys.modules, "transformers.models.bert.modeling_bert", module
        )
        _patch_bert_head_mask()
        assert FakeBertModel.get_head_mask is sentinel

    def test_installs_and_runs_patched_get_head_mask(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module, FakeBertModel = self._fake_bert_module()
        monkeypatch.setitem(
            sys.modules, "transformers.models.bert.modeling_bert", module
        )
        _patch_bert_head_mask()
        assert "get_head_mask" in FakeBertModel.__dict__

        instance = FakeBertModel()
        assert FakeBertModel.get_head_mask(instance, None, 3) == [None, None, None]

        head_mask_1d = torch.ones(2)
        expanded = FakeBertModel.get_head_mask(instance, head_mask_1d, 4)
        assert expanded.shape[0] == 4

        head_mask_2d = torch.ones(4, 2)
        chunked = FakeBertModel.get_head_mask(
            instance, head_mask_2d, 4, is_attention_chunked=True
        )
        assert chunked.dim() == head_mask_2d.dim() + 4  # 3 unsqueezes + chunk unsqueeze


class TestPatchGetExtendedAttentionMask:
    def _fake_module_utils_module(self):
        calls: list[tuple] = []

        class FakeModuleUtilsMixin:
            def get_extended_attention_mask(
                self, attention_mask, input_shape, dtype=None
            ):
                calls.append((attention_mask, input_shape, dtype))
                return "extended"

        module = types.ModuleType("transformers.modeling_utils")
        module.ModuleUtilsMixin = FakeModuleUtilsMixin
        return module, FakeModuleUtilsMixin, calls

    def test_noop_when_transformers_modeling_utils_unimportable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "transformers.modeling_utils", None)
        _patch_get_extended_attention_mask()  # must not raise

    def test_wraps_and_discards_legacy_device_positional(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module, FakeModuleUtilsMixin, calls = self._fake_module_utils_module()
        monkeypatch.setitem(sys.modules, "transformers.modeling_utils", module)
        _patch_get_extended_attention_mask()

        patched = FakeModuleUtilsMixin.get_extended_attention_mask
        assert getattr(patched, "_skiagrafia_patched", False) is True
        instance = FakeModuleUtilsMixin()

        # dtype passed directly in the legacy "device" slot.
        patched(instance, "mask", (1, 2), torch.float32)
        assert calls[-1][2] == torch.float32

        # a real torch.device is discarded -> dtype stays None.
        patched(instance, "mask", (1, 2), torch.device("cpu"))
        assert calls[-1][2] is None

        # anything else legacy-positional is treated as dtype.
        patched(instance, "mask", (1, 2), "not-a-device")
        assert calls[-1][2] == "not-a-device"

    def test_second_patch_call_is_idempotent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module, FakeModuleUtilsMixin, _ = self._fake_module_utils_module()
        monkeypatch.setitem(sys.modules, "transformers.modeling_utils", module)
        _patch_get_extended_attention_mask()
        first_patched = FakeModuleUtilsMixin.get_extended_attention_mask
        _patch_get_extended_attention_mask()
        assert FakeModuleUtilsMixin.get_extended_attention_mask is first_patched


class TestLoadDinoFailurePath:
    def test_raises_when_grounding_dino_package_unavailable(self, tmp_path: Path) -> None:
        # dino_weights/gsam_root are explicitly supplied so model_path() (which
        # would resolve against the user's real, possibly-populated model
        # directory) is never consulted -- only the missing `grounding_dino`
        # import is exercised.
        gsam = GroundedSAM(dino_weights=tmp_path / "fake.pth", gsam_root=tmp_path)
        with pytest.raises(Exception):
            gsam._load_dino()
        assert gsam._dino_model is None


# ── _ensure_gsam_on_path ──────────────────────────────────────────────────


class TestEnsureGsamOnPath:
    def test_adds_existing_directory_to_sys_path(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(sys, "path", list(sys.path))
        assert str(tmp_path) not in sys.path
        _ensure_gsam_on_path(tmp_path)
        assert str(tmp_path) in sys.path

    def test_missing_directory_is_not_added(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(sys, "path", list(sys.path))
        missing = tmp_path / "does_not_exist"
        _ensure_gsam_on_path(missing)
        assert str(missing) not in sys.path

    def test_does_not_duplicate_existing_entry(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(sys, "path", list(sys.path) + [str(tmp_path)])
        before = list(sys.path)
        _ensure_gsam_on_path(tmp_path)
        assert sys.path == before


# ── detect_box ─────────────────────────────────────────────────────────────


class TestDetectBox:
    def test_returns_none_when_no_boxes_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_predict(**kwargs):
            return torch.empty(0, 4), torch.empty(0), []

        _install_fake_grounding_dino(monkeypatch, fake_predict)
        gsam = _make_gsam_with_dino_stubbed()

        result = gsam.detect_box(_tiny_image(), "widget", skip_synonyms=True)
        assert result is None

    def test_converts_best_box_to_absolute_pixel_coords(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_predict(**kwargs):
            boxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]])  # centered, half image
            logits = torch.tensor([0.9])
            return boxes, logits, ["widget"]

        _install_fake_grounding_dino(monkeypatch, fake_predict)
        gsam = _make_gsam_with_dino_stubbed()

        result = gsam.detect_box(_tiny_image(64), "widget", skip_synonyms=True)
        assert isinstance(result, DetectionResult)
        assert result.bbox == (16, 16, 48, 48)
        assert result.confidence == pytest.approx(0.9)

    def test_picks_highest_confidence_box(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_predict(**kwargs):
            boxes = torch.tensor(
                [[0.2, 0.2, 0.2, 0.2], [0.5, 0.5, 0.4, 0.4]]
            )
            logits = torch.tensor([0.3, 0.95])
            return boxes, logits, ["a", "b"]

        _install_fake_grounding_dino(monkeypatch, fake_predict)
        gsam = _make_gsam_with_dino_stubbed()

        result = gsam.detect_box(_tiny_image(64), "widget", skip_synonyms=True)
        # Second box (index 1, cx=cy=0.5, w=h=0.4 of a 64px image) wins.
        assert result.bbox == (int(0.3 * 64), int(0.3 * 64), int(0.7 * 64), int(0.7 * 64))

    def test_appends_period_to_caption_when_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captions: list[str] = []

        def fake_predict(**kwargs):
            captions.append(kwargs["caption"])
            return torch.empty(0, 4), torch.empty(0), []

        _install_fake_grounding_dino(monkeypatch, fake_predict)
        gsam = _make_gsam_with_dino_stubbed()

        gsam.detect_box(_tiny_image(), "widget", skip_synonyms=True)
        gsam.detect_box(_tiny_image(), "widget.", skip_synonyms=True)
        assert captions == ["widget.", "widget."]

    def test_retries_with_synonyms_when_ambiguous_label_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []

        def fake_predict(**kwargs):
            calls.append(kwargs["caption"])
            if kwargs["caption"] == "computer mouse.":
                return (
                    torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
                    torch.tensor([0.8]),
                    ["computer mouse"],
                )
            return torch.empty(0, 4), torch.empty(0), []

        _install_fake_grounding_dino(monkeypatch, fake_predict)
        gsam = _make_gsam_with_dino_stubbed()

        result = gsam.detect_box(_tiny_image(), "mouse")
        assert result is not None
        assert calls[0] == "mouse."
        assert "computer mouse." in calls

    def test_no_detection_after_exhausting_synonyms_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_predict(**kwargs):
            return torch.empty(0, 4), torch.empty(0), []

        _install_fake_grounding_dino(monkeypatch, fake_predict)
        gsam = _make_gsam_with_dino_stubbed()

        result = gsam.detect_box(_tiny_image(), "mouse")
        assert result is None

    def test_unknown_label_has_no_synonyms_single_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []

        def fake_predict(**kwargs):
            calls.append(kwargs["caption"])
            return torch.empty(0, 4), torch.empty(0), []

        _install_fake_grounding_dino(monkeypatch, fake_predict)
        gsam = _make_gsam_with_dino_stubbed()

        gsam.detect_box(_tiny_image(), "chalice")
        assert calls == ["chalice."]


# ── segment ────────────────────────────────────────────────────────────────


class TestSegment:
    def test_single_mask_path_returns_binary_mask(self) -> None:
        gsam = GroundedSAM()
        gsam._sam_predictor = FakeSamPredictor()
        mask = gsam.segment(_tiny_image(), (5, 5, 20, 20), label="chalice")
        assert mask.shape == (IMG_SIZE, IMG_SIZE)
        assert set(np.unique(mask)).issubset({0, 255})
        assert (mask > 0).any()

    def test_prefer_full_box_picks_best_fill_mask(self) -> None:
        gsam = GroundedSAM()
        gsam._sam_predictor = FakeSamPredictor()
        bbox = (0, 0, 40, 40)
        mask = gsam.segment(_tiny_image(), bbox, label="monitor", prefer_full_box=True)
        # Mask index 1 (fill 0:40,0:40) has the best fill ratio for this bbox.
        assert mask[35, 35] == 255
        assert mask[50, 50] == 0

    def test_caches_result_by_label_and_bbox(self) -> None:
        gsam = GroundedSAM()
        gsam._sam_predictor = FakeSamPredictor()
        bbox = (1, 1, 10, 10)
        mask = gsam.segment(_tiny_image(), bbox, label="cross")
        cache_key = f"cross_{bbox}"
        assert cache_key in gsam._masks_cache
        assert np.array_equal(gsam._masks_cache[cache_key], mask)

    def test_clear_cache_empties_masks_cache(self) -> None:
        gsam = GroundedSAM()
        gsam._sam_predictor = FakeSamPredictor()
        gsam.segment(_tiny_image(), (0, 0, 5, 5), label="x")
        assert gsam._masks_cache
        gsam.clear_cache()
        assert gsam._masks_cache == {}


class TestBestMaskForBbox:
    def test_selects_index_with_highest_fill_ratio(self) -> None:
        masks = np.zeros((3, 20, 20), dtype=bool)
        masks[0, 0:2, 0:2] = True     # low fill
        masks[1, 0:10, 0:10] = True   # full fill of bbox (0,0,10,10)
        masks[2, 0:5, 0:5] = True     # partial fill
        idx = GroundedSAM._best_mask_for_bbox(masks, (0, 0, 10, 10))
        assert idx == 1


class TestDetectAndSegment:
    def test_returns_none_when_detection_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_predict(**kwargs):
            return torch.empty(0, 4), torch.empty(0), []

        _install_fake_grounding_dino(monkeypatch, fake_predict)
        gsam = _make_gsam_with_dino_stubbed()

        result = gsam.detect_and_segment(_tiny_image(), "widget")
        assert result is None

    def test_returns_detection_and_mask_on_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_predict(**kwargs):
            return (
                torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
                torch.tensor([0.9]),
                ["widget"],
            )

        _install_fake_grounding_dino(monkeypatch, fake_predict)
        gsam = _make_gsam_with_dino_stubbed()
        gsam._sam_predictor = FakeSamPredictor()

        result = gsam.detect_and_segment(_tiny_image(64), "widget")
        assert result is not None
        detection, mask = result
        assert isinstance(detection, DetectionResult)
        assert mask.shape == (64, 64)
