"""Telling "nothing is there" apart from "I could not look".

Every vision stage returns [] when the local VLM cannot be reached, and the
escalation ladder then tries the same unreachable server through each of its
stages. The run reaches the orchestrator with no parents and the operator is
told to review their prompt -- for a server that was never up.

On a long batch this is the expensive one: if the server dies at image 40,
images 40-300 each produce an empty result and that same sentence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.interrogation import GuidedInterrogator, InterrogationSettings
from models.vlm_client import BaseVLMClient


class _UnreachableClient(BaseVLMClient):
    def __init__(self) -> None:
        super().__init__(host="http://127.0.0.1:1", model="offline-model")

    def query_vision(self, image, prompt, **kwargs) -> str:
        raise ConnectionError("connection refused")

    def query_text(self, prompt: str, *, num_predict: int = 256) -> str:
        raise ConnectionError("connection refused")

    def get_children(self, image, parent_label: str) -> list[str]:
        raise ConnectionError("connection refused")

    def health_check(self) -> bool:
        return False


def _offline_interrogator() -> GuidedInterrogator:
    settings = InterrogationSettings(
        host="http://127.0.0.1:1",
        primary_vlm="primary",
        fallback_vlms=["fallback"],
        reasoner_model="reasoner",
    )
    interrogator = GuidedInterrogator(settings)
    for name in ("primary", "fallback", "reasoner"):
        interrogator._clients[name] = _UnreachableClient()
    return interrogator


def test_an_unreachable_model_is_reported_rather_than_read_as_an_empty_image() -> None:
    result = _offline_interrogator().interrogate(np.zeros((8, 8, 3), dtype=np.uint8))

    assert result.candidates == []
    assert result.vision_unavailable is True
    assert "offline-model" in result.confidence_summary or "reach" in result.confidence_summary.lower()


def test_a_reachable_model_that_sees_nothing_is_not_reported_as_unavailable() -> None:
    """An empty image is a legitimate answer, and must stay distinguishable."""

    class _Silent(_UnreachableClient):
        def query_vision(self, image, prompt, **kwargs) -> str:
            return ""

        def query_text(self, prompt: str, *, num_predict: int = 256) -> str:
            return ""

    settings = InterrogationSettings(
        host="http://127.0.0.1:1", primary_vlm="primary",
        fallback_vlms=[], reasoner_model="reasoner",
    )
    interrogator = GuidedInterrogator(settings)
    for name in ("primary", "reasoner"):
        interrogator._clients[name] = _Silent()

    result = interrogator.interrogate(np.zeros((8, 8, 3), dtype=np.uint8))

    assert result.candidates == []
    assert result.vision_unavailable is False


def test_a_run_with_no_reachable_model_does_not_blame_the_operators_prompt(tmp_path) -> None:
    """The sentence the operator actually reads.

    With no parents the orchestrator ends with "No usable object masks were
    found. Review the prompt or draw a box." -- advice about a prompt that was
    never sent anywhere. On a long batch this is every image after the server
    dies.
    """
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from orchestrator_fakes import (
        FakeAlphaRefiner,
        FakeDetector,
        FakeSegmenter,
        _make_caps,
        _write_image,
    )

    from core.interrogation_types import InterrogationResult
    from core.orchestrator import Orchestrator

    class _OfflineInterrogator:
        def interrogate(
            self, image, confirmed_labels=None, knowledge_pack=None
        ) -> InterrogationResult:
            return InterrogationResult(
                candidates=[],
                vision_unavailable=True,
                confidence_summary="could not reach any vision model: primary (ConnectionError)",
            )

        def set_confirmed_selections(self, selections: dict[str, str]) -> None:
            pass

    image_path = _write_image(tmp_path / "img.png")
    caps = _make_caps(
        _OfflineInterrogator(),  # type: ignore[arg-type]  # a stand-in, not a FakeInterrogator
        FakeDetector(),
        FakeSegmenter(),
        alpha_refiner=FakeAlphaRefiner(),
    )
    result = Orchestrator(capabilities=caps, output_dir=tmp_path / "out").process(image_path)

    joined = " | ".join(result.warnings)
    assert "reach" in joined.lower(), joined
    assert "Review the prompt" not in joined, joined
