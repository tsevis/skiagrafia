"""test_closed_vocabulary.py  --  a Domain Guide that fixes its own terms.

The APPLE50 book ran with no guide and produced 626 distinct labels over
1,957 uses, 65% of them used exactly once: `screen`, `computer screen`
and `screens` became three different things, as did `logo` and `apple
logo`. A guide already maps a known alias to its canonical term. What it
could not do was refuse a term it has never heard of, so anything the
model invented still became a layer of its own.

Offline, like the other interrogation tests: the VLM clients are fakes.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_interrogation import FakeVLMClient, _make_interrogator, _tiny_image

from core.interrogation import GuidedInterrogator
from core.interrogation_types import InterrogationResult
from core.knowledge import KnowledgeDomain, KnowledgePack, ObjectKnowledge

SCREENS = ObjectKnowledge(canonical="screen", aliases=["computer screen", "screens"])
LOGOS = ObjectKnowledge(canonical="logo", aliases=["apple logo"])


def _pack(*, closed: bool) -> KnowledgePack:
    return KnowledgePack(
        domain=KnowledgeDomain(name="apple history", closed_vocabulary=closed),
        objects=[SCREENS, LOGOS],
    )


def _proposing(*labels: str) -> GuidedInterrogator:
    interrogator = _make_interrogator()
    interrogator._clients["moondream"] = FakeVLMClient(
        vision_responses={"default": ", ".join(labels)}
    )
    return interrogator


def test_an_alias_still_becomes_its_canonical_term() -> None:
    result = _proposing("computer screen").interrogate(
        _tiny_image(), knowledge_pack=_pack(closed=True)
    )

    assert [c.canonical_label for c in result.candidates] == ["screen"]


def test_a_closed_guide_turns_away_a_term_it_does_not_know() -> None:
    result = _proposing("screens", "bicycle").interrogate(
        _tiny_image(), knowledge_pack=_pack(closed=True)
    )

    assert [c.canonical_label for c in result.candidates] == ["screen"]


def test_a_closed_guide_says_which_terms_it_turned_away() -> None:
    # Dropping in silence would be the same fault as the empty layer file:
    # less than was asked for, with nothing said about it.
    result = _proposing("bicycle", "trombone").interrogate(
        _tiny_image(), knowledge_pack=_pack(closed=True)
    )

    assert result.labels_outside_vocabulary == ["bicycle", "trombone"]


def test_an_open_guide_still_takes_whatever_the_model_proposes() -> None:
    # The default. An existing guide must behave exactly as it did.
    result = _proposing("bicycle").interrogate(
        _tiny_image(), knowledge_pack=_pack(closed=False)
    )

    assert [c.canonical_label for c in result.candidates] == ["bicycle"]
    assert result.labels_outside_vocabulary == []


def test_a_run_with_no_guide_at_all_is_unchanged() -> None:
    result = _proposing("bicycle").interrogate(_tiny_image())

    assert [c.canonical_label for c in result.candidates] == ["bicycle"]


def test_a_term_the_person_confirmed_is_kept_even_outside_the_vocabulary() -> None:
    # A closed vocabulary constrains what the model may invent. A person
    # who names a label has asked for it by name.
    result = _proposing().interrogate(
        _tiny_image(), confirmed_labels=["bicycle"], knowledge_pack=_pack(closed=True)
    )

    assert [c.canonical_label for c in result.candidates] == ["bicycle"]


def test_the_run_reports_the_terms_its_guide_turned_away(tmp_path: Path) -> None:
    # The interrogator recording them is not enough: the person reads the
    # run's warnings, not the interrogation result.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from orchestrator_fakes import (
        FakeDetector,
        FakeInterrogator,
        FakeSegmenter,
        _candidate,
        _make_caps,
        _write_image,
    )

    from core.orchestrator import Orchestrator

    class TurningAwayInterrogator(FakeInterrogator):
        def interrogate(self, image, confirmed_labels=None, knowledge_pack=None) -> InterrogationResult:
            result = super().interrogate(image, confirmed_labels, knowledge_pack)
            return InterrogationResult(
                candidates=result.candidates,
                children_by_parent=result.children_by_parent,
                labels_outside_vocabulary=["bicycle", "trombone"],
            )

    caps = _make_caps(
        TurningAwayInterrogator([_candidate("screen")]),
        FakeDetector(default=(0, 0, 32, 32)),
        FakeSegmenter(),
    )
    result = Orchestrator(caps, output_dir=tmp_path / "out").process(
        _write_image(tmp_path / "in.png")
    )

    assert result.error is None
    assert any("bicycle" in w and "trombone" in w for w in result.warnings), result.warnings
