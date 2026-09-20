"""test_orchestrator_glyphs.py  --  the typography fork of the pipeline.

Individual-glyph labels take a different path: the interrogator may read the
glyphs semantically, and the orchestrator will only relabel a detector
proposal when the two views agree one-to-one. When they do not, composites
spanning several glyphs are rejected and the run says so.

Offline throughout -- see orchestrator_fakes for the stand-ins.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator_fakes import (
    IMG_SIZE,
    FakeAlphaRefiner,
    FakeDetector,
    FakeInterrogator,
    FakeSegmenter,
    MultiInstanceDetector,
    _candidate,
    _make_caps,
    _write_image,
)

from core.orchestrator import Orchestrator
from core.typography_labels import TypographyElement, TypographyObservation
from models.grounded_sam import DetectionResult


class TestProcessIndividualGlyphs:
    @staticmethod
    def _glyph_detections() -> list[DetectionResult]:
        # Four real glyph proposals plus the failure observed in PETE: one
        # high-area proposal composed of portions of the middle two glyphs.
        return [
            DetectionResult(label="letters", bbox=(1, 0, 15, 64), confidence=0.95),
            DetectionResult(label="letters", bbox=(17, 0, 30, 64), confidence=0.94),
            DetectionResult(label="letters", bbox=(32, 0, 46, 64), confidence=0.93),
            DetectionResult(label="letters", bbox=(48, 0, 62, 64), confidence=0.92),
            DetectionResult(label="letters", bbox=(22, 8, 40, 64), confidence=0.70),
        ]

    @staticmethod
    def _observation() -> TypographyObservation:
        return TypographyObservation(
            elements=(
                TypographyElement("P", (0, 0, 230, 1000)),
                TypographyElement("E", (250, 0, 470, 1000)),
                TypographyElement("T", (500, 0, 720, 1000)),
                TypographyElement("E", (750, 0, 970, 1000)),
            )
        )

    def test_semantic_reading_selects_and_names_only_matching_glyphs(self, tmp_path: Path) -> None:
        image_path = _write_image(tmp_path / "glyphs.png")
        interrogator = FakeInterrogator(
            [_candidate("letters")],
            typography_observation=self._observation(),
        )
        caps = _make_caps(
            interrogator,
            MultiInstanceDetector(self._glyph_detections()),
            FakeSegmenter(),
        )
        result = Orchestrator(capabilities=caps, output_dir=tmp_path / "out", quality="fast").process(image_path)

        assert result.error is None
        assert [layer.label for layer in result.layers] == [
            "letter P", "letter E", "letter T", "letter E",
        ]
        assert [layer.bbox for layer in result.layers] == [
            (1, 0, 15, 64), (17, 0, 30, 64), (32, 0, 46, 64), (48, 0, 62, 64),
        ]
        assert not any("Typography reading" in warning for warning in result.warnings)

    def test_detector_only_fallback_rejects_composite_cross_glyph_proposal(self, tmp_path: Path) -> None:
        image_path = _write_image(tmp_path / "glyphs.png")
        caps = _make_caps(
            FakeInterrogator([_candidate("letters")]),
            MultiInstanceDetector(self._glyph_detections()),
            FakeSegmenter(),
        )
        result = Orchestrator(capabilities=caps, output_dir=tmp_path / "out", quality="fast").process(image_path)

        assert result.error is None
        assert len(result.layers) == 4
        assert all(layer.label == "letters" for layer in result.layers)
        assert any("Excluded 1 composite typography proposal" in warning for warning in result.warnings)

    def test_confirmed_selections_are_applied_before_confirmed_labels(
        self, tmp_path: Path
    ) -> None:
        image_path = _write_image(tmp_path / "img.png")
        interrogator = FakeInterrogator([_candidate("chalice")])
        caps = _make_caps(
            interrogator,
            FakeDetector(default=(5, 5, 40, 40)),
            FakeSegmenter(),
        )

        orchestrator = Orchestrator(capabilities=caps, output_dir=tmp_path / "out")
        orchestrator.set_confirmed_selections({"chalice": "largest"})
        orchestrator.process(
            image_path,
            confirmed_labels=["chalice"],
        )

        assert interrogator.confirmed_selections == [{"chalice": "largest"}]
        assert interrogator.calls[0][0] == ["chalice"]

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
        assert result.all_objects_tiff_path is not None
        with Image.open(result.all_objects_tiff_path) as rgba:
            pixels = np.asarray(rgba)
        assert pixels.shape == (IMG_SIZE, IMG_SIZE, 4)
        assert not pixels[..., 3].any()

    def test_output_dir_is_created(self, tmp_path: Path) -> None:
        image_path = _write_image(tmp_path / "img.png")
        out_dir = tmp_path / "nested" / "out"
        caps = _make_caps(FakeInterrogator([]), FakeDetector(), FakeSegmenter())
        orch = Orchestrator(capabilities=caps, output_dir=out_dir)
        orch.process(image_path)
        assert out_dir.is_dir()

    def test_invalid_mask_has_a_clear_pipeline_error(self, tmp_path: Path) -> None:
        image_path = _write_image(tmp_path / "img.png")
        bad_mask = np.zeros((4, 4), dtype=np.uint8)
        caps = _make_caps(
            FakeInterrogator([_candidate("chalice")]),
            FakeDetector(default=(5, 5, 40, 40)),
            FakeSegmenter(mask_by_label={"chalice": bad_mask}),
        )
        result = Orchestrator(capabilities=caps, output_dir=tmp_path / "out").process(image_path)
        assert result.error is not None
        assert "Mask generation failed" in result.error

    def test_failed_alpha_refinement_has_a_clear_pipeline_error(self, tmp_path: Path) -> None:
        class FailingAlpha:
            def predict(self, image, mask):
                raise RuntimeError("backend unavailable")

        image_path = _write_image(tmp_path / "img.png")
        caps = _make_caps(
            FakeInterrogator([_candidate("chalice")]),
            FakeDetector(default=(5, 5, 40, 40)),
            FakeSegmenter(),
            alpha_refiner=FailingAlpha(),
        )
        result = Orchestrator(capabilities=caps, output_dir=tmp_path / "out").process(image_path)
        assert result.error is not None
        assert "Alpha refinement failed" in result.error

    def test_unsafe_vector_output_has_a_clear_export_error(self, tmp_path: Path) -> None:
        class UnsafeVectorizer:
            def trace(self, mask):
                return '<script>alert(1)</script>'

        image_path = _write_image(tmp_path / "img.png")
        caps = _make_caps(
            FakeInterrogator([_candidate("chalice")]),
            FakeDetector(default=(5, 5, 40, 40)),
            FakeSegmenter(),
            vectorizer=UnsafeVectorizer(),
        )
        result = Orchestrator(capabilities=caps, output_dir=tmp_path / "out").process(image_path)
        assert result.error == "SVG export failed integrity validation."

    def test_vector_only_mode_keeps_the_all_objects_alpha_sidecar(self, tmp_path: Path) -> None:
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
        assert result.tiff_path is not None
        assert result.all_objects_tiff_path is not None
        assert result.tiff_files == [result.all_objects_tiff_path]
        assert alpha_refiner.calls == 1

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
