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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator_fakes import (
    IMG_SIZE,
    FakeDetector,
    FakeInterrogator,
    FakeSegmenter,
    MultiInstanceDetector,
    _candidate,
    _make_caps,
    _write_image,
)

from core.interrogation import InterrogationCandidate
from core.knowledge import KnowledgeDomain, KnowledgePack, ObjectKnowledge
from core.orchestrator import Orchestrator
from models.grounded_sam import DetectionResult

# ── Fake Protocol implementations ───────────────────────────────────────────




# ── _safe_filename_label ─────────────────────────────────────────────────




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
        assert det is not None
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
        assert det is not None
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




class TestProcessConfirmedFallback:
    def test_unlocated_confirmed_label_reports_warning_without_inventing_mask(self, tmp_path: Path) -> None:
        image_path = _write_image(tmp_path / "img.png")
        candidates = [_candidate("mystery object", source_model="confirmed")]
        detector = FakeDetector(boxes={"mystery object": None})
        caps = _make_caps(FakeInterrogator(candidates), detector, FakeSegmenter())
        orch = Orchestrator(capabilities=caps, output_dir=tmp_path / "out")

        result = orch.process(image_path)

        assert result.error is None
        assert result.layers == []
        assert "Could not locate" in result.warnings[0]
        assert result.svg_path is None

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

    def test_a_label_folded_into_a_duplicate_is_reported(self, tmp_path: Path) -> None:
        image_path = _write_image(tmp_path / "img.png")
        candidates = [_candidate("name tag"), _candidate("medal ribbon")]
        box = (0, 0, IMG_SIZE, IMG_SIZE)
        detector = FakeDetector(boxes={"name tag": box, "medal ribbon": box})
        full_mask = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)
        segmenter = FakeSegmenter(
            mask_by_label={"name tag": full_mask, "medal ribbon": full_mask}
        )
        caps = _make_caps(FakeInterrogator(candidates), detector, segmenter)
        orch = Orchestrator(capabilities=caps, output_dir=tmp_path / "out")

        result = orch.process(image_path)

        folded = [w for w in result.warnings if "'medal ribbon'" in w]
        assert len(folded) == 1
        assert "'name tag'" in folded[0]


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


class TestInstanceCeiling:
    def test_reaching_the_ceiling_warns_once_and_stops_detecting(
        self, tmp_path: Path
    ) -> None:
        """The ceiling `break` only left the INNER loop.

        The outer loop over parents kept going, so every remaining parent
        still cost a full detection round-trip and appended the same warning
        again. A real 231-image run produced the identical sentence up to
        five times for one image.
        """
        from core.orchestrator import MAX_OBJECT_INSTANCES

        # A grid inside the 64x64 test image: boxes outside it are clipped
        # away before they can count towards the ceiling.
        step = 6
        cells = [
            (x, y)
            for y in range(0, IMG_SIZE, step)
            for x in range(0, IMG_SIZE, step)
        ][: MAX_OBJECT_INSTANCES + 5]
        instances = [
            DetectionResult(label="widget", bbox=(x, y, x + 4, y + 4), confidence=0.9)
            for x, y in cells
        ]
        detector = MultiInstanceDetector(instances)
        parents = [_candidate(f"widget{index}") for index in range(6)]
        caps = _make_caps(FakeInterrogator(parents), detector, FakeSegmenter())

        result = Orchestrator(caps, output_dir=tmp_path / "out").process(
            _write_image(tmp_path / "in.png")
        )

        ceiling = [w for w in result.warnings if "object instances" in w]
        assert ceiling == [
            f"Stopped at {MAX_OBJECT_INSTANCES} object instances; "
            "narrow the prompt or process a crop."
        ]
        # One parent already overflows the ceiling, so no further parent
        # should have cost a detection round-trip.
        assert len(detector.calls) == 1, (
            "detection ran for parents that could never be accepted"
        )


class TestInstanceSizeComparison:
    def test_largest_compares_one_metric_across_all_detections(self) -> None:
        """The sort key switched units per element.

        It read mask pixels when a detection carried a mask and bounding-box
        area when it did not. A mask is far smaller than the box around it,
        so in a list mixing detector sources -- MLX SAM 3 populates a mask,
        the GroundingDINO fallback does not -- "largest" reliably picked the
        UNMASKED detection, whatever its real size.
        """
        from core.detection_policy import instance_size

        # A sparse mask inside a large box is what makes the two keys
        # disagree: the old key scored this 100 (mask pixels) against the
        # other's 3600 (box area) and called the smaller object larger.
        sparse = np.zeros((100, 100), dtype=np.uint8)
        sparse[:10, :10] = 255
        big_box = DetectionResult(
            label="a", bbox=(0, 0, 100, 100), confidence=0.9, mask=sparse,
        )
        unmasked = DetectionResult(label="b", bbox=(0, 0, 60, 60), confidence=0.9)

        size = instance_size([big_box, unmasked])

        assert size(big_box) > size(unmasked), (
            "with one detection unmasked, every size must be a box area"
        )

    def test_mask_pixels_are_used_when_every_detection_has_one(self) -> None:
        """A mask is the better measure, so it is used when all of them have it."""
        from core.detection_policy import instance_size

        wide_box_thin_object = DetectionResult(
            label="a", bbox=(0, 0, 100, 100), confidence=0.9,
            mask=np.pad(np.ones((2, 2), dtype=np.uint8) * 255, ((0, 98), (0, 98))),
        )
        small_box_solid = DetectionResult(
            label="b", bbox=(0, 0, 10, 10), confidence=0.9,
            mask=(np.ones((10, 10), dtype=np.uint8) * 255),
        )

        size = instance_size([wide_box_thin_object, small_box_solid])

        assert size(small_box_solid) > size(wide_box_thin_object)


class _RecordingRefiner:
    """Keeps the masks the pipeline hands to alpha refinement."""

    def __init__(self) -> None:
        self.masks: list[np.ndarray] = []

    def predict(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.masks.append(mask.copy())
        return mask.copy()


class TestSolidMasks:
    def _run(self, tmp_path: Path, **orchestrator_kwargs) -> np.ndarray:
        image_path = _write_image(tmp_path / "img.png")
        speckled = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        speckled[8:56, 8:56] = 255
        speckled[30, 30] = 0  # a reflection speck inside a solid object
        refiner = _RecordingRefiner()
        caps = _make_caps(
            FakeInterrogator([_candidate("phone")]),
            FakeDetector(default=(8, 8, 56, 56)),
            FakeSegmenter(mask_by_label={"phone": speckled}),
            alpha_refiner=refiner,
        )
        orch = Orchestrator(capabilities=caps, output_dir=tmp_path / "out", **orchestrator_kwargs)

        assert orch.process(image_path).error is None
        return refiner.masks[0]

    def test_solid_masks_fill_the_speck(self, tmp_path: Path) -> None:
        assert self._run(tmp_path, solid_masks=True)[30, 30] == 255

    def test_topology_is_kept_unless_asked(self, tmp_path: Path) -> None:
        assert self._run(tmp_path)[30, 30] == 0


class TestNestedSameLabelInstances:
    """The iPhone photo's fourth "phone" was the Phone app icon on the screen."""

    def _layers(self, tmp_path: Path, boxes: list[tuple[int, int, int, int]]) -> list:
        instances = [DetectionResult(label="phone", bbox=box, confidence=0.9) for box in boxes]
        caps = _make_caps(
            FakeInterrogator([_candidate("phone")]),
            MultiInstanceDetector(instances),
            FakeSegmenter(),
        )
        result = Orchestrator(caps, output_dir=tmp_path / "out").process(
            _write_image(tmp_path / "in.png")
        )
        assert result.error is None
        return [layer for layer in result.layers if layer.role == "parent"]

    def test_a_small_instance_inside_a_larger_one_is_dropped(self, tmp_path: Path) -> None:
        layers = self._layers(tmp_path, [(4, 4, 60, 60), (20, 20, 28, 28)])

        assert [layer.bbox for layer in layers] == [(4, 4, 60, 60)]

    def test_separate_instances_of_different_sizes_are_kept(self, tmp_path: Path) -> None:
        layers = self._layers(tmp_path, [(0, 0, 40, 40), (48, 48, 60, 60)])

        assert len(layers) == 2


class _JudgingInterrogator(FakeInterrogator):
    """Rejects any crop narrower than `min_width` as an icon of the object."""

    def __init__(self, candidates, min_width: int) -> None:
        super().__init__(candidates)
        self._min_width = min_width

    def confirms_object(self, image: np.ndarray, label: str) -> bool | None:
        return image.shape[1] >= self._min_width


class TestVerifyObjects:
    def _parents(self, tmp_path: Path, **orchestrator_kwargs) -> list:
        instances = [
            DetectionResult(label="phone", bbox=(0, 0, 30, 60), confidence=0.9),
            DetectionResult(label="phone", bbox=(40, 4, 48, 12), confidence=0.9),
        ]
        caps = _make_caps(
            _JudgingInterrogator([_candidate("phone")], min_width=20),
            MultiInstanceDetector(instances),
            FakeSegmenter(),
        )
        result = Orchestrator(caps, output_dir=tmp_path / "out", **orchestrator_kwargs).process(
            _write_image(tmp_path / "in.png")
        )
        assert result.error is None
        return [layer for layer in result.layers if layer.role == "parent"]

    def test_an_instance_the_vision_model_rejects_is_dropped(self, tmp_path: Path) -> None:
        layers = self._parents(tmp_path, verify_objects=True)

        assert [layer.bbox for layer in layers] == [(0, 0, 30, 60)]


class _MarkingEdgeRefiner:
    """Clears one pixel, so a test can tell the refined mask from SAM's."""

    def refine(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        marked = mask.copy()
        marked[20, 20] = 0
        return marked


class TestEdgeRefiner:
    def test_layers_use_the_refined_edge(self, tmp_path: Path) -> None:
        refiner = _RecordingRefiner()
        caps = _make_caps(
            FakeInterrogator([_candidate("phone")]),
            FakeDetector(default=(8, 8, 56, 56)),
            FakeSegmenter(),
            alpha_refiner=refiner,
        )
        orch = Orchestrator(
            capabilities=caps, output_dir=tmp_path / "out", edge_refiner=_MarkingEdgeRefiner()
        )

        assert orch.process(_write_image(tmp_path / "in.png")).error is None
        assert refiner.masks[0][20, 20] == 0

    def test_a_refiner_that_cannot_load_keeps_sams_edge(self, tmp_path: Path) -> None:
        # A missing einops/kornia failed every image instead of one step.
        class _Unloadable:
            def refine(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
                raise ImportError("This modeling file requires einops, kornia")

        caps = _make_caps(
            FakeInterrogator([_candidate("phone")]),
            FakeDetector(default=(8, 8, 56, 56)),
            FakeSegmenter(),
        )
        orch = Orchestrator(capabilities=caps, output_dir=tmp_path / "out", edge_refiner=_Unloadable())

        result = orch.process(_write_image(tmp_path / "in.png"))

        assert result.error is None
        assert [layer.label for layer in result.layers] == ["phone"]
