"""Regression tests for output identity, localization, transparency and topology."""
import io
import logging
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image
from test_grounded_sam import FakeSamPredictor
from test_orchestrator import (
    FakeDetector,
    FakeInterrogator,
    FakeSegmenter,
    _candidate,
    _make_caps,
    _write_image,
)
from test_vitmatte_refiner import _wired_refiner

from core.interrogation import GuidedInterrogator, InterrogationSettings
from core.orchestrator import Orchestrator
from models.grounded_sam import DetectionResult, GroundedSAM
from models.local_vlm import LOCAL_FALLBACK, LOCAL_PRIMARY, resolve_local_model
from models.vlm_client import LlamaCppVLMClient, OllamaVLMClient
from processors.mask_ops import refine_mask
from processors.vectorizer import VTracerVectorizer, assemble_svg


def _not_none[T](value: T | None) -> T:
    """Narrow an Optional this pipeline guarantees is set at this point."""
    assert value is not None
    return value


class Instances(FakeDetector):
    def detect_instances(self, image, label, *args) -> list[DetectionResult]:
        if label == "bag":
            return [DetectionResult(label=label, bbox=box, confidence=.9)
                    for box in [(2, 2, 28, 60), (34, 2, 62, 60)]]
        if label == "handle":
            return [DetectionResult(label=label, bbox=(12, 12, 18, 20), confidence=.9)]
        return []


def test_repeated_objects_and_parts_have_independent_exports(tmp_path) -> None:
    result = Orchestrator(_make_caps(FakeInterrogator([_candidate("bag")], {"bag": ["handle"]}), Instances(), FakeSegmenter()),
                          output_dir=tmp_path / "out").process(_write_image(tmp_path / "in.png"))
    assert result.error is None
    parents = [x for x in result.layers if x.role == "parent"]
    children = [x for x in result.layers if x.role == "child"]
    assert len(parents) == len(children) == 2
    assert len({x.layer_id for x in result.layers}) == 4
    assert {x.parent_id for x in children} == {x.layer_id for x in parents}
    assert len(result.tiff_files) == len(set(result.tiff_files)) == 7
    assert result.all_objects_tiff_path is not None
    with Image.open(result.all_objects_tiff_path) as rgba:
        alpha = np.asarray(rgba)[..., 3]
    expected = np.maximum.reduce([_not_none(layer.alpha) for layer in result.layers])
    assert np.array_equal(alpha, expected)
    assert not np.array_equal(_not_none(children[0].mask), _not_none(children[1].mask))
    root = ET.parse(_not_none(result.svg_path))
    ids = [node.attrib['id'] for node in root.iter() if 'id' in node.attrib]
    assert len(ids) == len(set(ids))


def test_nested_objects_are_not_deleted_as_duplicates(tmp_path) -> None:
    detector = FakeDetector(boxes={"tray": (0, 0, 64, 64), "cup": (15, 15, 30, 30)})
    result = Orchestrator(_make_caps(FakeInterrogator([_candidate("tray"), _candidate("cup")]), detector, FakeSegmenter()),
                          output_dir=tmp_path).process(_write_image(tmp_path / "image.png"))
    assert [x.label for x in result.layers] == ["tray", "cup"]


def test_child_overlapping_neighbor_is_rejected(tmp_path) -> None:
    class LeakySegmenter(FakeSegmenter):
        def segment(self, image, bbox, label="", prefer_full_box=False) -> NDArray[np.uint8]:
            if label == "handle":
                # Crop includes 8 px of neighboring background. Most of this
                # supposed part is outside the actual parent's silhouette.
                mask = np.zeros(image.shape[:2], np.uint8)
                mask[:10, :10] = 255
                return mask
            return super().segment(image, bbox, label, prefer_full_box)
    result = Orchestrator(_make_caps(FakeInterrogator([_candidate("bag")], {"bag": ["handle"]}),
                    FakeDetector(boxes={"bag": (20, 20, 50, 50), "handle": (0, 0, 10, 10)}), LeakySegmenter()),
                    output_dir=tmp_path).process(_write_image(tmp_path / "image.png"))
    assert [x.role for x in result.layers] == ["parent"]


def test_spatial_selector_returns_one_requested_instance() -> None:
    orch = Orchestrator(_make_caps(FakeInterrogator([]), Instances(), FakeSegmenter()))
    candidate = _candidate("bag")
    candidate.selection = "rightmost"
    found = orch._detect_instances(np.zeros((64, 64, 3), np.uint8), candidate, None)
    assert len(found) == 1 and found[0][0].bbox[0] == 34


def test_prompt_json_preserves_attributes_and_selector() -> None:
    settings = InterrogationSettings("", LOCAL_PRIMARY, [], LOCAL_FALLBACK, user_prompt="Only the leftmost red bag", discover_parts=False)
    interrogator = GuidedInterrogator(settings)
    client = Mock()
    client.query_vision.return_value = '{"objects":[{"label":"red bag","selection":"leftmost"}]}'
    interrogator._clients[LOCAL_PRIMARY] = client
    result = interrogator.interrogate(np.zeros((32, 32, 3), np.uint8))
    assert [(c.display_label, c.selection) for c in result.candidates] == [("red bag", "leftmost")]
    assert client.query_vision.call_count == 1
    assert not client.get_children.called


def test_encoding_reused_for_multiple_boxes_but_reset_between_images() -> None:
    sam = GroundedSAM()
    predictor = FakeSamPredictor()
    sam._sam_predictor = predictor
    image = np.zeros((64, 64, 3), np.uint8)
    sam.segment(image, (1, 1, 20, 20))
    sam.segment(image, (25, 25, 40, 40))
    assert len(predictor.set_image_calls) == 1
    sam.clear_cache()
    sam.segment(image.copy(), (1, 1, 20, 20))
    assert len(predictor.set_image_calls) == 2


def test_cleanup_preserves_hole_and_one_pixel_wire() -> None:
    mask = np.zeros((64, 64), np.uint8)
    mask[8:50, 8:50] = 255
    mask[20:30, 20:30] = 0
    mask[40, 50:62] = 255
    assert np.array_equal(refine_mask(mask, min_contour_area=0), mask)


def test_matte_known_regions_stay_opaque_or_transparent() -> None:
    refiner, _ = _wired_refiner((64, 64))
    mask = np.zeros((64, 64), np.uint8)
    mask[10:54, 10:54] = 255
    mask[25:40, 25:40] = 0
    alpha = refiner.predict(np.zeros((64, 64, 3), np.uint8), mask)
    assert alpha[15, 15] == 255
    assert alpha[0, 0] == alpha[32, 32] == 0
    assert np.any((alpha > 0) & (alpha < 255))


def test_vector_roundtrip_keeps_holes_and_thin_structure() -> None:
    from utils.cairo_support import load_cairosvg
    mask = np.zeros((64, 64), np.uint8)
    mask[8:50, 8:50] = 255
    mask[20:30, 20:30] = 0
    mask[40:42, 50:62] = 255
    traced = VTracerVectorizer(preserve_detail=True).trace(mask)
    svg = assemble_svg(64, 64, [{"id": "ring", "svg_data": traced}])
    png = _not_none(load_cairosvg(logging.getLogger(__name__))).svg2png(bytestring=svg.encode())
    rendered = np.array(Image.open(io.BytesIO(png)).convert("RGBA"))[..., 3] > 127
    truth = mask > 0
    assert (rendered & truth).sum() / (rendered | truth).sum() > .98
    assert not rendered[25, 25] and rendered[40, 58]


def test_duplicate_svg_ids_are_uniquified() -> None:
    svg = assemble_svg(4, 4, [{"id": "handle", "svg_data": "<path/>"}] * 3)
    ids = [el.attrib['id'] for el in ET.fromstring(svg).iter() if 'id' in el.attrib]
    assert len(ids) == len(set(ids)) == 3


def test_gemma_ollama_disables_thinking() -> None:
    client = OllamaVLMClient(model="gemma4:e4b")
    client._client = Mock()
    client._client.chat.return_value = SimpleNamespace(message=SimpleNamespace(content="bag"))
    assert client.query_text("objects?") == "bag"
    assert client._client.chat.call_args.kwargs['think'] is False


def test_truncated_response_is_rejected() -> None:
    import pytest
    client = LlamaCppVLMClient()
    client._request_json = Mock(return_value={"choices":[{"finish_reason":"length", "message":{"content":"bag, car, bag"}}]})
    with pytest.raises(ValueError, match="token budget"):
        client.query_text("objects?")


def test_managed_model_lookup_rejects_missing_files(tmp_path) -> None:
    import pytest
    with pytest.raises(FileNotFoundError, match="No files were downloaded"):
        resolve_local_model(LOCAL_PRIMARY, cache=tmp_path)


def test_body_and_parts_recompose_without_transparent_seams() -> None:
    from core.layer_editing import body_alpha
    parent = np.array([[255, 200, 120, 0]], np.uint8)
    child = np.array([[128, 180, 100, 0]], np.uint8)
    body = body_alpha(parent, [child]).astype(float) / 255
    c = child.astype(float) / 255
    recomposed = np.rint((c + body * (1 - c)) * 255)
    assert np.max(np.abs(recomposed - parent)) <= 1


def test_layer_edit_reclips_children_and_updates_export(tmp_path) -> None:
    from core.layer_editing import replace_layer_mask
    caps = _make_caps(FakeInterrogator([_candidate("bag")], {"bag": ["handle"]}), Instances(), FakeSegmenter())
    result = Orchestrator(caps, output_dir=tmp_path / "out").process(_write_image(tmp_path / "in.png"))
    parent = result.layers[0]
    replacement = _not_none(parent.mask).copy()
    replacement[:25] = 0
    replace_layer_mask(result, parent.layer_id, replacement, np.zeros((64, 64, 3), np.uint8), caps)
    child = next(c for c in result.layers if c.parent_id == parent.layer_id)
    assert not _not_none(child.mask).any()
    with Image.open(_not_none(child.alpha_path)) as rgba:
        assert not np.asarray(rgba)[..., 3].any()
    with Image.open(_not_none(result.all_objects_tiff_path)) as rgba:
        composite_alpha = np.asarray(rgba)[..., 3]
    expected = np.maximum.reduce([_not_none(layer.alpha) for layer in result.layers])
    assert np.array_equal(composite_alpha, expected)


def test_existing_transparency_is_preserved_in_export(tmp_path) -> None:
    rgb = np.full((64, 64, 4), 120, np.uint8)
    rgb[..., 3] = 0
    rgb[12:50, 12:50, 3] = 160
    path = tmp_path / "transparent.png"
    Image.fromarray(rgb).save(path)
    caps = _make_caps(FakeInterrogator([_candidate("bag")]), FakeDetector(default=(0, 0, 64, 64)), FakeSegmenter())
    result = Orchestrator(caps, output_dir=tmp_path / "out").process(path)
    assert result.error is None
    with Image.open(result.tiff_files[0]) as rgba:
        pixels = np.asarray(rgba)
        assert pixels[20, 20, 3] == 160
        assert pixels[0, 0, 3] == 0
        # Colour is kept under the object and for a bleed margin around it,
        # and cleared beyond, so the compression has something to work with.
        # See processors.output_writer.COLOUR_BLEED_PIXELS.
        assert np.array_equal(pixels[12:50, 12:50, :3], rgb[12:50, 12:50, :3])
        assert pixels[0, 0, :3].tolist() == [0, 0, 0]


def test_local_backend_routes_primary_and_fallback_independently() -> None:
    from core.factory import build_interrogation_settings
    settings = build_interrogation_settings({"vlm_backend":"local"})
    assert settings.primary_vlm == LOCAL_PRIMARY
    assert settings.fallback_vlms == [LOCAL_FALLBACK]
    assert settings.reasoner_model == LOCAL_FALLBACK


def test_detailed_matte_only_runs_tiles_containing_boundary() -> None:
    refiner, model = _wired_refiner((960, 960))
    refiner._quality = "detailed"
    refiner._max_side = 1536
    image = np.zeros((1800, 1800, 3), np.uint8)
    mask = np.zeros((1800, 1800), np.uint8)
    mask[20:1780, 20:1780] = 255
    alpha = refiner.predict(image, mask)
    assert alpha.shape == mask.shape
    assert alpha[900, 900] == 255 and alpha[0, 0] == 0
    assert 1 < model.forward_calls < 9  # skip the wholly opaque center tile


def test_uncertain_sam3_parts_are_not_promoted_to_layers(tmp_path) -> None:
    class UncertainParts(Instances):
        def detect_instances(self, image, label, *args) -> list[DetectionResult]:
            found = super().detect_instances(image, label, *args)
            if label == "handle":
                for detection in found:
                    detection.source = "mlx-sam3"
                    detection.confidence = .54
            return found
    result = Orchestrator(_make_caps(FakeInterrogator([_candidate("bag")], {"bag": ["handle"]}),
                          UncertainParts(), FakeSegmenter()), output_dir=tmp_path / "out").process(_write_image(tmp_path / "in.png"))
    assert len(result.layers) == 2
    assert all(layer.role == "parent" for layer in result.layers)


def test_mlx_adapter_reuses_encoding_and_returns_all_masks(tmp_path, monkeypatch) -> None:
    import sys
    import types

    from models import mlx_sam3 as mlx_sam3_module
    from models.mlx_sam3 import MLXSAM3
    from models.sam3_grounding import GroundedInstances
    mlx = types.ModuleType("mlx")
    core = types.ModuleType("mlx.core")
    # types.ModuleType has no declared `eval`/`core` attributes; monkeypatch
    # both type-checks the assignment and restores it after the test.
    monkeypatch.setattr(core, "eval", lambda *_: None, raising=False)
    monkeypatch.setattr(mlx, "core", core, raising=False)
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    processor = Mock()
    processor.set_image.return_value = {}
    # The vendored grounding call needs the MLX runtime, so stand in for it at
    # the seam and keep this test about encoding reuse and fallback routing.
    grounded = GroundedInstances(
        boxes=np.array([[0, 0, 6, 6], [8, 8, 14, 14]], dtype=np.float32),
        masks=np.full((2, 16, 16), 255, dtype=np.uint8),
        scores=[.8, .9], presence=.9, rescued=False,
    )
    holder = {"value": grounded}
    monkeypatch.setattr(
        mlx_sam3_module, "ground_text_prompt", lambda *_args, **_kwargs: holder["value"]
    )
    # MLXSAM3.fallback only ever calls detect_instances/segment/clear_cache on
    # this, all of which Instances (via FakeDetector) provides; cast because
    # MLXSAM3 pins the parameter to the concrete GroundedSAM class.
    adapter = MLXSAM3(tmp_path, cast(GroundedSAM, Instances()))
    adapter._processor = processor
    image = np.zeros((16, 16, 3), np.uint8)
    assert len(adapter.detect_instances(image, "bag")) == 2
    adapter.detect_instances(image, "handle")
    assert processor.set_image.call_count == 1
    adapter.detect_instances(image.copy(), "bag")
    assert processor.set_image.call_count == 2
    holder["value"] = GroundedInstances(
        boxes=np.zeros((0, 4), dtype=np.float32),
        masks=np.zeros((0, 16, 16), dtype=np.uint8),
        scores=[], presence=.001, rescued=False,
    )
    assert adapter.detect_part_instances(image, "bag") == []
    assert len(adapter.detect_instances(image, "bag")) == 2  # parent fallback remains available


def test_exif_orientation_is_applied_before_masks(tmp_path) -> None:
    from processors.source_image import load_source_image
    image = Image.new("RGB", (30, 20), "white")
    exif = image.getexif()
    exif[274] = 6
    path = tmp_path / "rotated.jpg"
    image.save(path, exif=exif)
    rgb, alpha, _ = load_source_image(path)
    assert rgb.shape == (30, 20, 3) and alpha.shape == (30, 20)


def test_gui_alpha_view_uses_actual_matte(single_view, tmp_path) -> None:
    path = _write_image(tmp_path / "image.png")
    caps = _make_caps(FakeInterrogator([_candidate("bag")]), FakeDetector(default=(10, 10, 50, 50)), FakeSegmenter())
    result = Orchestrator(caps, output_dir=tmp_path / "out").process(path)
    single_view.left_panel._load_image(str(path))
    single_view.on_processing_complete(result)
    canvas = single_view.canvas_panel
    canvas._view_mode.set("alpha")
    canvas.refresh_overlays()
    single_view.root.update()
    assert canvas._photo_image is not None
    single_view.right_panel._select_layer(0)
    single_view.right_panel._opacity_var.set(50)
    assert result.layers[0].preview_opacity == .5


def test_batch_interrogation_overrides_match_processing_settings() -> None:
    from core.factory import build_interrogation_settings
    settings = build_interrogation_settings({"vlm_backend": "local"}, overrides={
        "interrogation_profile": "deep", "enable_tiled_fallback": False,
        "preferred_vlm": "gemma4:e4b",
    })
    assert settings.profile == "deep" and settings.enable_tiling is False
    assert settings.primary_vlm == LOCAL_FALLBACK


def test_layer_edit_names_the_layer_that_was_never_segmented(tmp_path) -> None:
    """LayerResult.mask is Optional, so a parent without one is expressible.

    The reclip then handed None to numpy, which reports an operand type
    error naming neither the layer nor the field it was missing. Nothing
    else in the call chain says which layer was never segmented.
    """
    from core.layer_editing import replace_layer_mask
    from core.pipeline_results import LayerResult, PipelineResult

    caps = _make_caps(FakeInterrogator([_candidate("bag")], {"bag": ["handle"]}),
                      Instances(), FakeSegmenter())
    result = PipelineResult(image_path=str(tmp_path / "in.png"), width=8, height=8)
    result.layers = [
        LayerResult(layer_id="L01", label="parent", role="parent", bbox=(0, 0, 8, 8)),
        LayerResult(layer_id="L01-part-001", label="child", role="child",
                    parent_id="L01", bbox=(0, 0, 4, 4)),
    ]

    with pytest.raises(ValueError, match="L01"):
        replace_layer_mask(
            result,
            "L01-part-001",
            np.zeros((8, 8), dtype=np.uint8),
            np.zeros((8, 8, 3), dtype=np.uint8),
            caps,
        )
