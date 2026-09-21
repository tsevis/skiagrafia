"""Caps, gates and budgets that stopped work without saying so.

Each of these is correct behaviour reported as nothing at all. The parent
path already warns when it drops a mask ("Empty or tiny mask for 'X'"); the
child path, the part-discovery budget and the label cap did not.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.interrogation import MAX_PARENTS, parse_label_candidates
from core.pipeline_results import PipelineResult, collapse_repeats


def test_a_label_list_longer_than_the_cap_says_it_was_truncated() -> None:
    """A model naming twelve object types loses four, silently."""
    raw = ", ".join(f"object{index}" for index in range(MAX_PARENTS + 4))

    labels, dropped = parse_label_candidates(raw, report_dropped=True)

    assert len(labels) == MAX_PARENTS
    assert dropped == 4


def test_the_cap_reports_nothing_when_it_did_not_bite() -> None:
    labels, dropped = parse_label_candidates("keyboard, mouse", report_dropped=True)

    assert len(labels) == 2
    assert dropped == 0


def test_the_existing_single_value_form_is_unchanged() -> None:
    """Every current caller passes no flag and expects a plain list."""
    assert parse_label_candidates("keyboard, mouse") == ["keyboard", "mouse"]


def test_repeated_rejections_collapse_into_one_countable_line() -> None:
    result = PipelineResult(image_path="/in/a.png", width=10, height=10)
    for _ in range(4):
        result.warnings.append("Excluded a part of 'screen' that fell outside it")

    assert collapse_repeats(result.warnings) == [
        "Excluded a part of 'screen' that fell outside it (x4)"
    ]


class _RecordingDetector:
    """Records which detection entry point a part label was routed to."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def detect_instances(
        self, image, label, box_threshold: float = 0.35, text_threshold: float = 0.25
    ) -> list:
        self.calls.append("detect_instances")
        return []

    def detect_part_instances(
        self, image, label, box_threshold: float = 0.35, text_threshold: float = 0.25
    ) -> list:
        self.calls.append("detect_part_instances")
        return []

    def clear_cache(self) -> None:
        pass


def test_a_part_keeps_its_presence_gate_at_every_quality_profile() -> None:
    """`detailed` routed model-suggested parts past the strict entry point.

    detect_part_instances is what sets requirePresence. Bypassing it lets the
    localisation rescue accept a part the model does not recognise -- on the
    measured corpus, "printer" on g4cube.png at 0.901.
    """
    import numpy as np

    from core.detection_policy import DetectionPolicyMixin
    from core.interrogation_types import InterrogationCandidate

    for quality in ("fast", "balanced", "detailed"):
        detector = _RecordingDetector()
        policy = DetectionPolicyMixin()
        policy._detector = detector          # type: ignore[attr-defined]
        policy._quality = quality            # type: ignore[attr-defined]
        policy._box_threshold = 0.35         # type: ignore[attr-defined]
        policy._text_threshold = 0.25        # type: ignore[attr-defined]
        child = InterrogationCandidate(
            display_label="printer", canonical_label="printer", confidence=0.9, role="child"
        )

        policy._detect_instances(np.zeros((8, 8, 3), dtype=np.uint8), child, None)

        assert detector.calls == ["detect_part_instances"], (quality, detector.calls)
