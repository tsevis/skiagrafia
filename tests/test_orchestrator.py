"""test_orchestrator.py  --  Orchestrator pipeline logic (offline, no ML).

All tests are offline: the Orchestrator is driven with lightweight fake
implementations of the core.contracts Protocols (Interrogator, Detector,
Segmenter, AlphaRefiner, Vectorizer) that return canned numpy masks/boxes.
No SAM 2.1, GroundingDINO, or VitMatte weights are ever loaded, and no
network or GPU work happens. Images are tiny synthetic 64x64 uint8 arrays
written to pytest's tmp_path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.contracts import CapabilitySet
from core.interrogation import InterrogationCandidate, InterrogationResult
from core.knowledge import KnowledgePack, KnowledgeDomain, ObjectKnowledge
from core.orchestrator import (
    MAX_LABEL_FILENAME_LEN,
    Orchestrator,
    _bbox_area,
    _bbox_iou,
    _bbox_overlaps,
    _clip_mask_to_bbox,
    _crop_to_bbox,
    _mask_containment,
    _mask_iou,
    _safe_filename_label,
)
from models.grounded_sam import DetectionResult


IMG_SIZE = 64


# ── Fake Protocol implementations ───────────────────────────────────────────


class FakeInterrogator:
    def __init__(
        self,
        candidates: list[InterrogationCandidate],
        children_by_parent: dict[str, list[str]] | None = None,
    ) -> None:
        self._candidates = candidates
        self._children = children_by_parent or {}
        self.calls: list[tuple] = []

    def interrogate(self, image, confirmed_labels=None, knowledge_pack=None):
        self.calls.append((confirmed_labels, knowledge_pack))
        return InterrogationResult(
            candidates=self._candidates,
            children_by_parent=self._children,
        )


class FakeDetector:
    """Returns a canned bbox for labels present in `boxes`, else `default`."""

    def __init__(
        self,
        boxes: dict[str, tuple[int, int, int, int] | None] | None = None,
        default: tuple[int, int, int, int] | None = None,
    ) -> None:
        self._boxes = boxes or {}
        self._default = default
        self.calls: list[str] = []

    def detect_box(self, image, label, box_threshold=0.35, text_threshold=0.25):
        self.calls.append(label)
        if label in self._boxes:
            bbox = self._boxes[label]
            if bbox is None:
                return None
            return DetectionResult(label=label, bbox=bbox, confidence=0.9)
        if self._default is not None:
            return DetectionResult(label=label, bbox=self._default, confidence=0.9)
        return None


class FakeSegmenter:
    """Produces a full-coverage mask inside the requested bbox by default."""

    def __init__(self, mask_by_label: dict[str, NDArray[np.uint8]] | None = None) -> None:
        self._mask_by_label = mask_by_label or {}
        self.clear_cache_called = False
        self.calls: list[tuple] = []

    def segment(self, image, bbox, label="", prefer_full_box=False):
        self.calls.append((bbox, label, prefer_full_box))
        if label in self._mask_by_label:
            return self._mask_by_label[label]
        h, w = image.shape[:2]
        x0, y0, x1, y1 = bbox
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 255
        return mask

    def clear_cache(self) -> None:
        self.clear_cache_called = True


class FakeAlphaRefiner:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, image, mask):
        self.calls += 1
        return mask.copy()


class FakeVectorizer:
    def __init__(self) -> None:
        self.calls = 0

    def trace(self, mask):
        self.calls += 1
        return (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<path d="M0 0L1 1" fill="#000000"/></svg>'
        )


def _make_caps(
    interrogator: FakeInterrogator,
    detector: FakeDetector,
    segmenter: FakeSegmenter,
    alpha_refiner: FakeAlphaRefiner | None = None,
    vectorizer: FakeVectorizer | None = None,
) -> CapabilitySet:
    return CapabilitySet(
        interrogator=interrogator,
        detector=detector,
        segmenter=segmenter,
        alpha_refiner=alpha_refiner or FakeAlphaRefiner(),
        vectorizer=vectorizer or FakeVectorizer(),
    )


def _write_image(path: Path, size: int = IMG_SIZE) -> Path:
    img = np.full((size, size, 3), (30, 60, 90), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def _candidate(
    label: str,
    source_model: str = "vlm",
    detector_phrases: list[str] | None = None,
    confidence: float = 0.8,
) -> InterrogationCandidate:
    return InterrogationCandidate(
        canonical_label=label,
        display_label=label,
        detector_phrases=detector_phrases or [label],
        source_model=source_model,
        confidence=confidence,
    )


# ── _safe_filename_label ─────────────────────────────────────────────────


class TestSafeFilenameLabel:
    def test_short_label_passes_through_unchanged(self) -> None:
        assert _safe_filename_label("chalice") == "chalice"

    def test_replaces_path_separators(self) -> None:
        assert _safe_filename_label("a/b\\c:d") == "a_b_c_d"

    def test_truncates_to_max_length(self) -> None:
        label = "x" * (MAX_LABEL_FILENAME_LEN + 50)
        safe = _safe_filename_label(label)
        assert len(safe) <= MAX_LABEL_FILENAME_LEN

    def test_truncation_strips_trailing_dot_and_space(self) -> None:
        # Construct a label whose MAX_LABEL_FILENAME_LEN-th char lands on
        # a run of dots/spaces so rstrip has visible effect.
        label = ("a" * (MAX_LABEL_FILENAME_LEN - 3)) + "   ." + "y" * 20
        safe = _safe_filename_label(label)
        assert not safe.endswith(" ")
        assert not safe.endswith(".")

    def test_historical_bug_long_moondream_list_label(self) -> None:
        # Regression test: Moondream once returned a ~300-char numbered-list
        # label that produced an OSError: File name too long on macOS.
        long_label = ", ".join(f"item {i}" for i in range(1, 40))
        assert len(long_label) > 200
        safe = _safe_filename_label(long_label)
        assert len(safe) <= MAX_LABEL_FILENAME_LEN
        # Must still be usable as an actual filename component.
        assert "/" not in safe and "\\" not in safe

    def test_unsafe_characters_do_not_raise(self) -> None:
        weird = "object*?<>|\"'"
        # Should not raise; only "/", "\\", ":" are explicitly sanitized.
        result = _safe_filename_label(weird)
        assert isinstance(result, str)


# ── Geometry helpers ─────────────────────────────────────────────────────


class TestGeometryHelpers:
    def test_mask_iou_identical_masks_is_one(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:6, 2:6] = 255
        assert _mask_iou(mask, mask) == pytest.approx(1.0)

    def test_mask_iou_disjoint_masks_is_zero(self) -> None:
        a = np.zeros((10, 10), dtype=np.uint8)
        a[0:3, 0:3] = 255
        b = np.zeros((10, 10), dtype=np.uint8)
        b[7:10, 7:10] = 255
        assert _mask_iou(a, b) == 0.0

    def test_mask_iou_empty_masks_returns_zero(self) -> None:
        a = np.zeros((10, 10), dtype=np.uint8)
        b = np.zeros((10, 10), dtype=np.uint8)
        assert _mask_iou(a, b) == 0.0

    def test_mask_containment_full_subset(self) -> None:
        big = np.zeros((10, 10), dtype=np.uint8)
        big[0:10, 0:10] = 255
        small = np.zeros((10, 10), dtype=np.uint8)
        small[2:4, 2:4] = 255
        assert _mask_containment(small, big) == pytest.approx(1.0)

    def test_mask_containment_zero_area_returns_zero(self) -> None:
        a = np.zeros((10, 10), dtype=np.uint8)
        b = np.zeros((10, 10), dtype=np.uint8)
        b[0:3, 0:3] = 255
        assert _mask_containment(a, b) == 0.0

    def test_bbox_overlaps_true_for_touching_expanded_boxes(self) -> None:
        parent = (10, 10, 20, 20)
        child = (18, 18, 25, 25)
        assert _bbox_overlaps(child, parent, img_h=64, img_w=64) is True

    def test_bbox_overlaps_false_when_far_apart(self) -> None:
        parent = (0, 0, 5, 5)
        child = (50, 50, 60, 60)
        assert _bbox_overlaps(child, parent, img_h=64, img_w=64) is False

    def test_bbox_area(self) -> None:
        assert _bbox_area((0, 0, 10, 5)) == 50

    def test_bbox_area_degenerate_returns_zero(self) -> None:
        assert _bbox_area((10, 10, 5, 5)) == 0

    def test_bbox_iou_full_overlap_is_one(self) -> None:
        box = (0, 0, 10, 10)
        assert _bbox_iou(box, box) == pytest.approx(1.0)

    def test_bbox_iou_no_overlap_is_zero(self) -> None:
        a = (0, 0, 5, 5)
        b = (50, 50, 60, 60)
        assert _bbox_iou(a, b) == 0.0

    def test_bbox_iou_partial_overlap(self) -> None:
        a = (0, 0, 10, 10)
        b = (5, 5, 15, 15)
        # intersection = 5x5=25, union = 100+100-25=175
        assert _bbox_iou(a, b) == pytest.approx(25 / 175)

    def test_crop_to_bbox_applies_padding(self) -> None:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        crop, x0, y0 = _crop_to_bbox(image, (10, 10, 20, 20), padding=5)
        assert x0 == 5 and y0 == 5
        assert crop.shape[:2] == (20, 20)

    def test_crop_to_bbox_clips_to_image_bounds(self) -> None:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        crop, x0, y0 = _crop_to_bbox(image, (0, 0, 10, 10), padding=20)
        assert x0 == 0 and y0 == 0
        assert crop.shape[:2] == (30, 30)

    def test_clip_mask_to_bbox_zeroes_outside_region(self) -> None:
        mask = np.full((20, 20), 255, dtype=np.uint8)
        clipped = _clip_mask_to_bbox(mask, (5, 5, 10, 10))
        assert clipped[0, 0] == 0
        assert clipped[7, 7] == 255
        assert clipped.sum() == 255 * 5 * 5


# ── Manual detection lookup ──────────────────────────────────────────────


class TestManualLookup:
    def _orch(self) -> Orchestrator:
        caps = _make_caps(
            FakeInterrogator([]), FakeDetector(), FakeSegmenter()
        )
        return Orchestrator(capabilities=caps)

    def test_build_manual_lookup_empty_input(self) -> None:
        orch = self._orch()
        assert orch._build_manual_lookup(None) == {}
        assert orch._build_manual_lookup([]) == {}

    def test_build_manual_lookup_skips_missing_bbox_or_label(self) -> None:
        orch = self._orch()
        lookup = orch._build_manual_lookup(
            [
                {"label": "cross", "bbox": None},
                {"label": "", "bbox": [1, 2, 3, 4]},
                {"bbox": [1, 2, 3, 4]},
                {"label": "chalice", "bbox": [1, 2, 3, 4]},
            ]
        )
        assert list(lookup.keys()) == ["chalice"]

    def test_build_manual_lookup_queues_same_label_in_order(self) -> None:
        orch = self._orch()
        lookup = orch._build_manual_lookup(
            [
                {"label": "Cross", "bbox": [0, 0, 1, 1]},
                {"label": "cross", "bbox": [2, 2, 3, 3]},
            ]
        )
        assert lookup["cross"] == [(0, 0, 1, 1), (2, 2, 3, 3)]

    def test_consume_manual_bbox_matches_display_label(self) -> None:
        orch = self._orch()
        lookup = {"cross": [(1, 2, 3, 4)]}
        candidate = _candidate("cross")
        bbox = orch._consume_manual_bbox(lookup, candidate)
        assert bbox == (1, 2, 3, 4)
        assert lookup["cross"] == []

    def test_consume_manual_bbox_matches_detector_phrase(self) -> None:
        orch = self._orch()
        lookup = {"metal cross": [(5, 6, 7, 8)]}
        candidate = _candidate("cross", detector_phrases=["metal cross"])
        assert orch._consume_manual_bbox(lookup, candidate) == (5, 6, 7, 8)

    def test_consume_manual_bbox_returns_none_when_absent(self) -> None:
        orch = self._orch()
        candidate = _candidate("cross")
        assert orch._consume_manual_bbox({}, candidate) is None

    def test_consume_manual_bbox_returns_none_when_queue_exhausted(self) -> None:
        orch = self._orch()
        lookup = {"cross": []}
        candidate = _candidate("cross")
        assert orch._consume_manual_bbox(lookup, candidate) is None


# ── _child_candidate ──────────────────────────────────────────────────────


class TestChildCandidate:
    def test_without_knowledge_pack_uses_generic_phrases(self) -> None:
        caps = _make_caps(FakeInterrogator([]), FakeDetector(), FakeSegmenter())
        orch = Orchestrator(capabilities=caps)
        parent = _candidate("chalice")
        child = orch._child_candidate(parent, "stem")
        assert child.canonical_label == "stem"
        assert child.role == "child"
        assert child.parent == "chalice"
        assert child.detector_phrases == ["stem", "stem detail", "chalice stem"]

    def test_with_knowledge_pack_uses_ranked_phrases(self) -> None:
        knowledge_pack = KnowledgePack(
            domain=KnowledgeDomain(name="test"),
            objects=[
                ObjectKnowledge(
                    canonical="stem",
                    aliases=["base stem"],
                    detector_phrases=["ornate stem"],
                )
            ],
        )
        caps = _make_caps(FakeInterrogator([]), FakeDetector(), FakeSegmenter())
        orch = Orchestrator(capabilities=caps, knowledge_pack=knowledge_pack)
        parent = _candidate("chalice")
        child = orch._child_candidate(parent, "stem")
        assert child.canonical_label == "stem"
        assert "ornate stem" in child.detector_phrases


# ── _detect_candidate / _detect_candidate_ex ─────────────────────────────


class TestDetectCandidate:
    def test_manual_lookup_hit_returns_manual_true(self) -> None:
        caps = _make_caps(FakeInterrogator([]), FakeDetector(), FakeSegmenter())
        orch = Orchestrator(capabilities=caps)
        candidate = _candidate("cross")
        manual_lookup = {"cross": [(1, 2, 3, 4)]}
        det, is_manual = orch._detect_candidate_ex(
            np.zeros((10, 10, 3), dtype=np.uint8), candidate, manual_lookup
        )
        assert is_manual is True
        assert det.bbox == (1, 2, 3, 4)
        assert det.confidence == 1.0

    def test_tries_phrases_in_order_until_hit(self) -> None:
        detector = FakeDetector(boxes={"first": None, "second": (1, 1, 2, 2)})
        caps = _make_caps(FakeInterrogator([]), detector, FakeSegmenter())
        orch = Orchestrator(capabilities=caps)
        candidate = _candidate("thing", detector_phrases=["first", "second", "third"])
        det, is_manual = orch._detect_candidate_ex(
            np.zeros((10, 10, 3), dtype=np.uint8), candidate, None
        )
        assert is_manual is False
        assert det.bbox == (1, 1, 2, 2)
        # "third" should never have been queried once "second" hit.
        assert detector.calls == ["first", "second"]

    def test_returns_none_when_no_phrase_matches(self) -> None:
        detector = FakeDetector(boxes={"only": None})
        caps = _make_caps(FakeInterrogator([]), detector, FakeSegmenter())
        orch = Orchestrator(capabilities=caps)
        candidate = _candidate("thing", detector_phrases=["only"])
        det = orch._detect_candidate(
            np.zeros((10, 10, 3), dtype=np.uint8), candidate, None
        )
        assert det is None

    def test_falls_back_to_display_label_when_no_phrases(self) -> None:
        detector = FakeDetector(default=(0, 0, 4, 4))
        caps = _make_caps(FakeInterrogator([]), detector, FakeSegmenter())
        orch = Orchestrator(capabilities=caps)
        candidate = InterrogationCandidate(
            canonical_label="thing", display_label="thing", detector_phrases=[]
        )
        det = orch._detect_candidate(
            np.zeros((10, 10, 3), dtype=np.uint8), candidate, None
        )
        assert det is not None
        assert detector.calls == ["thing"]


# ── Orchestrator.process() end-to-end (structural branch) ───────────────


class TestProcessBasicFlow:
    def test_single_parent_no_children_produces_svg_and_tiff(self, tmp_path: Path) -> None:
        image_path = _write_image(tmp_path / "img.png")
        out_dir = tmp_path / "out"
        candidates = [_candidate("chalice")]
        detector = FakeDetector(default=(5, 5, 40, 40))
        segmenter = FakeSegmenter()
        caps = _make_caps(FakeInterrogator(candidates), detector, segmenter)
        orch = Orchestrator(capabilities=caps, output_dir=out_dir)

        result = orch.process(image_path)

        assert result.error is None
        assert result.width == IMG_SIZE and result.height == IMG_SIZE
        assert len(result.layers) == 1
        assert result.layers[0].label == "chalice"
        assert result.layers[0].role == "parent"
        assert result.svg_path is not None
        assert Path(result.svg_path).exists()
        assert result.tiff_path is not None
        assert segmenter.clear_cache_called is True

    def test_image_load_failure_sets_result_error(self, tmp_path: Path) -> None:
        caps = _make_caps(FakeInterrogator([]), FakeDetector(), FakeSegmenter())
        orch = Orchestrator(capabilities=caps, output_dir=tmp_path / "out")
        result = orch.process(tmp_path / "does_not_exist.png")
        assert result.error is not None
        assert "Cannot read image" in result.error

    def test_no_candidates_yields_empty_layers_and_no_svg(self, tmp_path: Path) -> None:
        image_path = _write_image(tmp_path / "img.png")
        caps = _make_caps(FakeInterrogator([]), FakeDetector(), FakeSegmenter())
        orch = Orchestrator(capabilities=caps, output_dir=tmp_path / "out")
        result = orch.process(image_path)
        assert result.error is None
        assert result.layers == []
        assert result.svg_path is None

    def test_output_dir_is_created(self, tmp_path: Path) -> None:
        image_path = _write_image(tmp_path / "img.png")
        out_dir = tmp_path / "nested" / "out"
        caps = _make_caps(FakeInterrogator([]), FakeDetector(), FakeSegmenter())
        orch = Orchestrator(capabilities=caps, output_dir=out_dir)
        orch.process(image_path)
        assert out_dir.is_dir()

    def test_vector_only_mode_skips_vitmatte_tiff(self, tmp_path: Path) -> None:
        image_path = _write_image(tmp_path / "img.png")
        candidates = [_candidate("chalice")]
        detector = FakeDetector(default=(5, 5, 40, 40))
        alpha_refiner = FakeAlphaRefiner()
        caps = _make_caps(
            FakeInterrogator(candidates), detector, FakeSegmenter(),
            alpha_refiner=alpha_refiner,
        )
        orch = Orchestrator(
            capabilities=caps, output_dir=tmp_path / "out", output_mode="vector",
        )
        result = orch.process(image_path)
        assert result.error is None
        assert result.tiff_path is None
        assert alpha_refiner.calls == 0

    def test_progress_callback_receives_step_updates(self, tmp_path: Path) -> None:
        image_path = _write_image(tmp_path / "img.png")
        events: list[tuple[int, str]] = []
        caps = _make_caps(FakeInterrogator([]), FakeDetector(), FakeSegmenter())
        orch = Orchestrator(
            capabilities=caps,
            output_dir=tmp_path / "out",
            progress_callback=lambda step, msg: events.append((step, msg)),
        )
        orch.process(image_path)
        assert events  # at least the load step fired
        assert events[0][0] == 0


class TestProcessConfirmedFallback:
    def test_confirmed_label_falls_back_to_full_image_bbox(self, tmp_path: Path) -> None:
        image_path = _write_image(tmp_path / "img.png")
        candidates = [_candidate("mystery object", source_model="confirmed")]
        detector = FakeDetector(boxes={"mystery object": None})
        caps = _make_caps(FakeInterrogator(candidates), detector, FakeSegmenter())
        orch = Orchestrator(capabilities=caps, output_dir=tmp_path / "out")

        result = orch.process(image_path)

        assert result.error is None
        assert len(result.layers) == 1
        assert result.layers[0].bbox == (0, 0, IMG_SIZE, IMG_SIZE)

    def test_non_confirmed_label_without_detection_is_skipped(self, tmp_path: Path) -> None:
        image_path = _write_image(tmp_path / "img.png")
        candidates = [_candidate("ghost", source_model="vlm")]
        detector = FakeDetector(boxes={"ghost": None})
        caps = _make_caps(FakeInterrogator(candidates), detector, FakeSegmenter())
        orch = Orchestrator(capabilities=caps, output_dir=tmp_path / "out")

        result = orch.process(image_path)

        assert result.error is None
        assert result.layers == []

    def test_low_coverage_mask_is_skipped(self, tmp_path: Path) -> None:
        image_path = _write_image(tmp_path / "img.png")
        candidates = [_candidate("tiny")]
        detector = FakeDetector(default=(0, 0, 40, 40))
        # Mask with near-zero coverage — below MIN_PARENT_COVERAGE_PCT.
        tiny_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        segmenter = FakeSegmenter(mask_by_label={"tiny": tiny_mask})
        caps = _make_caps(FakeInterrogator(candidates), detector, segmenter)
        orch = Orchestrator(capabilities=caps, output_dir=tmp_path / "out")

        result = orch.process(image_path)

        assert result.error is None
        assert result.layers == []


class TestProcessManualDetections:
    def test_manual_detection_clips_mask_to_bbox(self, tmp_path: Path) -> None:
        image_path = _write_image(tmp_path / "img.png")
        candidates = [_candidate("cross")]
        detector = FakeDetector()  # would return None; manual lookup wins first
        caps = _make_caps(FakeInterrogator(candidates), detector, FakeSegmenter())
        orch = Orchestrator(capabilities=caps, output_dir=tmp_path / "out")

        manual_bbox = [10, 10, 20, 20]
        result = orch.process(
            image_path, manual_detections=[{"label": "cross", "bbox": manual_bbox}]
        )

        assert result.error is None
        assert len(result.layers) == 1
        assert result.layers[0].bbox == tuple(manual_bbox)
        # detector.detect_box must never have been called for a manual hit.
        assert detector.calls == []


class TestProcessParentDedup:
    def test_fallback_parent_replaced_by_real_detection(self, tmp_path: Path) -> None:
        image_path = _write_image(tmp_path / "img.png")
        # Same physical object reported under two labels: one confirmed
        # (fallback full-image bbox) and one properly detected with an
        # identical mask -> the fallback entry should be evicted.
        candidates = [
            _candidate("thing a", source_model="confirmed", confidence=1.0),
            _candidate("thing b", source_model="vlm", confidence=0.8),
        ]
        detector = FakeDetector(boxes={"thing a": None, "thing b": (0, 0, IMG_SIZE, IMG_SIZE)})
        full_mask = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)
        segmenter = FakeSegmenter(
            mask_by_label={"thing a": full_mask, "thing b": full_mask}
        )
        caps = _make_caps(FakeInterrogator(candidates), detector, segmenter)
        orch = Orchestrator(capabilities=caps, output_dir=tmp_path / "out")

        result = orch.process(image_path)

        assert result.error is None
        labels = [layer.label for layer in result.layers]
        assert labels == ["thing b"]

    def test_duplicate_real_detections_keep_first_and_merge_children(
        self, tmp_path: Path
    ) -> None:
        image_path = _write_image(tmp_path / "img.png")
        candidates = [
            _candidate("thing a", confidence=0.9),
            _candidate("thing b", confidence=0.8),
        ]
        detector = FakeDetector(
            boxes={"thing a": (0, 0, IMG_SIZE, IMG_SIZE), "thing b": (0, 0, IMG_SIZE, IMG_SIZE)}
        )
        full_mask = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)
        segmenter = FakeSegmenter(
            mask_by_label={"thing a": full_mask, "thing b": full_mask}
        )
        caps = _make_caps(FakeInterrogator(candidates), detector, segmenter)
        orch = Orchestrator(capabilities=caps, output_dir=tmp_path / "out")

        result = orch.process(image_path)

        assert result.error is None
        labels = [layer.label for layer in result.layers]
        assert labels == ["thing a"]


class TestProcessChildren:
    def test_child_accepted_and_subtracted_from_body_mask(self, tmp_path: Path) -> None:
        image_path = _write_image(tmp_path / "img.png")
        candidates = [_candidate("chalice")]
        children = {"chalice": ["stem"]}
        detector = FakeDetector(
            boxes={"chalice": (0, 0, IMG_SIZE, IMG_SIZE), "stem": (10, 10, 20, 20)}
        )
        parent_mask = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)
        child_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        child_mask[10:20, 10:20] = 255
        segmenter = FakeSegmenter(
            mask_by_label={"chalice": parent_mask, "stem": child_mask}
        )
        caps = _make_caps(FakeInterrogator(candidates, children), detector, segmenter)
        orch = Orchestrator(capabilities=caps, output_dir=tmp_path / "out")

        result = orch.process(image_path)

        assert result.error is None
        roles = {layer.label: layer.role for layer in result.layers}
        assert roles == {"chalice": "parent", "stem": "child"}
        child_layer = next(
            layer for layer in result.layers if layer.label == "stem"
        )
        assert child_layer.parent_label == "chalice"

    def test_child_outside_parent_bbox_is_skipped(self, tmp_path: Path) -> None:
        image_path = _write_image(tmp_path / "img.png")
        candidates = [_candidate("chalice")]
        children = {"chalice": ["far_away"]}
        detector = FakeDetector(
            boxes={"chalice": (0, 0, 20, 20), "far_away": (60, 60, 64, 64)}
        )
        parent_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        parent_mask[0:20, 0:20] = 255
        segmenter = FakeSegmenter(mask_by_label={"chalice": parent_mask})
        caps = _make_caps(FakeInterrogator(candidates, children), detector, segmenter)
        orch = Orchestrator(capabilities=caps, output_dir=tmp_path / "out")

        result = orch.process(image_path)

        assert result.error is None
        assert [layer.label for layer in result.layers] == ["chalice"]

    def test_child_too_similar_to_parent_is_skipped(self, tmp_path: Path) -> None:
        image_path = _write_image(tmp_path / "img.png")
        candidates = [_candidate("chalice")]
        children = {"chalice": ["clone"]}
        detector = FakeDetector(
            boxes={"chalice": (0, 0, IMG_SIZE, IMG_SIZE), "clone": (0, 0, IMG_SIZE, IMG_SIZE)}
        )
        full_mask = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)
        # Child mask identical to parent -> IoU 1.0 > MAX_CHILD_PARENT_IOU
        segmenter = FakeSegmenter(
            mask_by_label={"chalice": full_mask, "clone": full_mask}
        )
        caps = _make_caps(FakeInterrogator(candidates, children), detector, segmenter)
        orch = Orchestrator(capabilities=caps, output_dir=tmp_path / "out")

        result = orch.process(image_path)

        assert result.error is None
        assert [layer.label for layer in result.layers] == ["chalice"]

    def test_duplicate_children_second_one_skipped(self, tmp_path: Path) -> None:
        image_path = _write_image(tmp_path / "img.png")
        candidates = [_candidate("chalice")]
        children = {"chalice": ["stem", "stem_alias"]}
        detector = FakeDetector(
            boxes={
                "chalice": (0, 0, IMG_SIZE, IMG_SIZE),
                "stem": (10, 10, 20, 20),
                "stem_alias": (10, 10, 20, 20),
            }
        )
        parent_mask = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)
        child_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        child_mask[10:20, 10:20] = 255
        segmenter = FakeSegmenter(
            mask_by_label={
                "chalice": parent_mask,
                "stem": child_mask,
                "stem_alias": child_mask,
            }
        )
        caps = _make_caps(FakeInterrogator(candidates, children), detector, segmenter)
        orch = Orchestrator(capabilities=caps, output_dir=tmp_path / "out")

        result = orch.process(image_path)

        assert result.error is None
        child_labels = [layer.label for layer in result.layers if layer.role == "child"]
        assert child_labels == ["stem"]

    def test_child_low_coverage_is_skipped(self, tmp_path: Path) -> None:
        image_path = _write_image(tmp_path / "img.png")
        candidates = [_candidate("chalice")]
        children = {"chalice": ["speck"]}
        detector = FakeDetector(
            boxes={"chalice": (0, 0, IMG_SIZE, IMG_SIZE), "speck": (10, 10, 20, 20)}
        )
        parent_mask = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)
        empty_child_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        segmenter = FakeSegmenter(
            mask_by_label={"chalice": parent_mask, "speck": empty_child_mask}
        )
        caps = _make_caps(FakeInterrogator(candidates, children), detector, segmenter)
        orch = Orchestrator(capabilities=caps, output_dir=tmp_path / "out")

        result = orch.process(image_path)

        assert result.error is None
        assert [layer.label for layer in result.layers] == ["chalice"]
