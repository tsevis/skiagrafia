"""test_interrogation.py  --  GuidedInterrogator orchestration logic.

All tests are offline: VLM clients are lightweight fakes injected directly
into GuidedInterrogator._clients, so create_vlm_client / network transports
are never invoked. No model weights are loaded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.interrogation import (
    GuidedInterrogator,
    InterrogationCandidate,
    InterrogationSettings,
    parse_label_candidates,
    rank_detector_phrases,
)
from core.knowledge import KnowledgeDomain, KnowledgePack, ObjectKnowledge


# ── Fakes ────────────────────────────────────────────────────────────────


class FakeVLMClient:
    """Vision/text/children client whose responses are keyed by substring
    of the prompt (vision/text) or exact parent label (children)."""

    def __init__(
        self,
        vision_responses: dict[str, str] | None = None,
        text_response: str = "",
        children_response: list[str] | None = None,
        children_error: bool = False,
    ) -> None:
        self.vision_responses = vision_responses or {}
        self.text_response = text_response
        self.children_response = children_response or []
        self.children_error = children_error
        self.vision_calls: list[str] = []
        self.text_calls: list[str] = []
        self.children_calls: list[str] = []

    def query_vision(self, image, prompt: str) -> str:
        self.vision_calls.append(prompt)
        for key, resp in self.vision_responses.items():
            if key in prompt:
                return resp
        return self.vision_responses.get("default", "")

    def query_text(self, prompt: str, *, num_predict: int = 256) -> str:
        self.text_calls.append(prompt)
        return self.text_response

    def get_children(self, image, parent_label: str) -> list[str]:
        self.children_calls.append(parent_label)
        if self.children_error:
            raise RuntimeError("children lookup failed")
        return self.children_response


def _settings(**overrides) -> InterrogationSettings:
    base = dict(
        host="http://localhost:11434",
        primary_vlm="moondream",
        fallback_vlms=["minicpm-v"],
        reasoner_model="qwen3.5",
    )
    base.update(overrides)
    return InterrogationSettings(**base)


def _make_interrogator(**overrides) -> GuidedInterrogator:
    """Build a GuidedInterrogator with every configured model (primary,
    fallbacks, reasoner) pre-stubbed with a harmless no-op fake client.

    This guarantees interrogate() can NEVER fall through to
    create_vlm_client() -- and therefore never attempts a real network
    connection -- even when a test's escalation path reaches a stage whose
    client it didn't explicitly register.
    """
    settings = _settings(**overrides)
    interrogator = GuidedInterrogator(settings)
    for model in {settings.primary_vlm, settings.reasoner_model, *settings.fallback_vlms}:
        interrogator._clients[model] = FakeVLMClient()
    return interrogator


def _tiny_image(size: int = 32) -> np.ndarray:
    return np.zeros((size, size, 3), dtype=np.uint8)


def _candidate(label: str, confidence: float = 0.8, phrases=None) -> InterrogationCandidate:
    return InterrogationCandidate(
        canonical_label=label,
        display_label=label,
        detector_phrases=phrases or [label],
        confidence=confidence,
    )


# Unique substrings per prompt_style (see GuidedInterrogator._build_prompt).
COMPOSITION_KEY = "Describe the whole foreground composition first"
GUIDED_KEY = "Reply only as a comma-separated object list"
PRIMARY_KEY = "Reply only as a comma-separated list of object names"


# ── interrogate(): confirmed-labels shortcut ─────────────────────────────


class TestConfirmedLabelsPath:
    def test_confirmed_labels_skip_vision_entirely(self) -> None:
        interrogator = _make_interrogator()
        result = interrogator.interrogate(
            _tiny_image(), confirmed_labels=["chalice", "cross"]
        )
        assert result.escalation_stage == "confirmed"
        assert [c.display_label for c in result.candidates] == ["chalice", "cross"]
        assert all(c.source_model == "confirmed" for c in result.candidates)
        assert all(c.confidence == 1.0 for c in result.candidates)

    def test_confirmed_labels_use_knowledge_pack_parts_as_children(self) -> None:
        pack = KnowledgePack(
            domain=KnowledgeDomain(name="test"),
            objects=[ObjectKnowledge(canonical="chalice", parts=["stem", "base"])],
        )
        interrogator = _make_interrogator()
        result = interrogator.interrogate(
            _tiny_image(), confirmed_labels=["chalice"], knowledge_pack=pack
        )
        assert result.children_by_parent.get("chalice") == ["stem", "base"]


# ── interrogate(): escalation chain ──────────────────────────────────────


class TestEscalationChain:
    def test_high_confidence_composition_short_circuits(self) -> None:
        interrogator = _make_interrogator()
        client = FakeVLMClient(vision_responses={COMPOSITION_KEY: "cross, chalice"})
        interrogator._clients["moondream"] = client
        result = interrogator.interrogate(_tiny_image())

        assert result.escalation_stage == "composition"
        assert result.confidence_summary == "high-confidence results"
        assert {c.display_label for c in result.candidates} == {"cross", "chalice"}
        # Only the composition prompt should have been sent.
        assert len(client.vision_calls) == 1

    def test_low_confidence_composition_escalates_to_primary(self) -> None:
        interrogator = _make_interrogator()
        client = FakeVLMClient(
            vision_responses={
                COMPOSITION_KEY: "object",  # vague -> escalate
                PRIMARY_KEY: "keyboard, mouse",
            }
        )
        interrogator._clients["moondream"] = client
        result = interrogator.interrogate(_tiny_image())

        assert result.escalation_stage in {"primary", "guided", "tiled"}
        labels = {c.display_label for c in result.candidates}
        assert "keyboard" in labels or "mouse" in labels

    def test_composition_first_false_starts_at_primary(self) -> None:
        interrogator = _make_interrogator(composition_first=False, enable_tiling=False)
        client = FakeVLMClient(vision_responses={PRIMARY_KEY: "keyboard, mouse"})
        interrogator._clients["moondream"] = client
        result = interrogator.interrogate(_tiny_image())

        assert not any(COMPOSITION_KEY in call for call in client.vision_calls)
        assert {c.display_label for c in result.candidates} == {"keyboard", "mouse"}

    def test_fallback_vlm_used_when_primary_still_low_confidence(self) -> None:
        interrogator = _make_interrogator(composition_first=False, enable_tiling=False)
        primary_client = FakeVLMClient(vision_responses={"default": "object"})
        fallback_client = FakeVLMClient(vision_responses={"default": "keyboard, mouse"})
        interrogator._clients["moondream"] = primary_client
        interrogator._clients["minicpm-v"] = fallback_client
        result = interrogator.interrogate(_tiny_image())

        assert fallback_client.vision_calls  # fallback model was queried
        assert result.escalation_stage == "fallback:minicpm-v"

    def test_moondream_only_fallback_mode_never_queries_fallback_vlms(self) -> None:
        interrogator = _make_interrogator(
            composition_first=False,
            enable_tiling=False,
            fallback_mode="moondream_only",
        )
        primary_client = FakeVLMClient(vision_responses={"default": "object"})
        fallback_client = FakeVLMClient(vision_responses={"default": "keyboard, mouse"})
        interrogator._clients["moondream"] = primary_client
        interrogator._clients["minicpm-v"] = fallback_client
        interrogator.interrogate(_tiny_image())

        assert fallback_client.vision_calls == []

    def test_tiling_stage_queries_four_quadrants(self) -> None:
        interrogator = _make_interrogator(
            composition_first=False,
            enable_tiling=True,
            fallback_mode="moondream_only",
            profile="balanced",
        )
        client = FakeVLMClient(vision_responses={"default": "object"})
        interrogator._clients["moondream"] = client
        interrogator.interrogate(_tiny_image(size=40))

        # primary + guided (still low confidence) + 4 tiles = 6 vision calls
        assert len(client.vision_calls) >= 4

    def test_tiling_skipped_for_fast_profile(self) -> None:
        interrogator = _make_interrogator(
            composition_first=False,
            enable_tiling=True,
            fallback_mode="moondream_only",
            profile="fast",
        )
        client = FakeVLMClient(vision_responses={"default": "object"})
        interrogator._clients["moondream"] = client
        interrogator.interrogate(_tiny_image())

        assert interrogator._allows_tiling() is False

    def test_always_enrich_forces_primary_pass_and_knowledge_seed(self) -> None:
        pack = KnowledgePack(
            domain=KnowledgeDomain(name="test"),
            objects=[ObjectKnowledge(canonical="cross", aliases=["metal cross"])],
        )
        interrogator = _make_interrogator(
            fallback_mode="always_enrich",
            composition_first=True,
            enable_tiling=False,
        )
        client = FakeVLMClient(
            vision_responses={COMPOSITION_KEY: "object", PRIMARY_KEY: "object"}
        )
        interrogator._clients["moondream"] = client
        result = interrogator.interrogate(_tiny_image(), knowledge_pack=pack)

        assert any(c.canonical_label == "cross" for c in result.candidates)

    def test_vision_stage_exception_is_swallowed(self) -> None:
        interrogator = _make_interrogator(
            enable_tiling=False, fallback_mode="moondream_only"
        )

        class RaisingClient:
            def query_vision(self, image, prompt):
                raise RuntimeError("network down")

        interrogator._clients["moondream"] = RaisingClient()
        result = interrogator.interrogate(_tiny_image())
        assert result.candidates == []


# ── children_by_parent ────────────────────────────────────────────────────


class TestChildrenMap:
    def test_knowledge_pack_parts_used_without_querying_client(self) -> None:
        pack = KnowledgePack(
            domain=KnowledgeDomain(name="test"),
            objects=[ObjectKnowledge(canonical="chalice", parts=["stem", "cup"])],
        )
        interrogator = GuidedInterrogator(_settings())
        client = FakeVLMClient()
        interrogator._clients["moondream"] = client
        children = interrogator._children_map(
            [_candidate("chalice", confidence=0.9)], pack, _tiny_image()
        )
        assert children == {"chalice": ["stem", "cup"]}
        assert client.children_calls == []

    def test_queries_client_when_no_knowledge_and_high_confidence(self) -> None:
        interrogator = GuidedInterrogator(_settings())
        client = FakeVLMClient(children_response=["stem", "cup", "coffee"])
        interrogator._clients["moondream"] = client
        children = interrogator._children_map(
            [_candidate("chalice", confidence=0.9)], None, _tiny_image()
        )
        assert children["chalice"] == ["stem", "cup"]  # "coffee" filtered out
        assert client.children_calls == ["chalice"]

    def test_low_confidence_candidate_is_not_queried(self) -> None:
        interrogator = GuidedInterrogator(_settings())
        client = FakeVLMClient(children_response=["stem"])
        interrogator._clients["moondream"] = client
        children = interrogator._children_map(
            [_candidate("chalice", confidence=0.3)], None, _tiny_image()
        )
        assert children == {}
        assert client.children_calls == []

    def test_vague_display_label_is_not_queried(self) -> None:
        interrogator = GuidedInterrogator(_settings())
        client = FakeVLMClient(children_response=["stem"])
        interrogator._clients["moondream"] = client
        children = interrogator._children_map(
            [_candidate("object", confidence=0.9)], None, _tiny_image()
        )
        assert children == {}

    def test_client_exception_yields_no_children(self) -> None:
        interrogator = GuidedInterrogator(_settings())
        client = FakeVLMClient(children_error=True)
        interrogator._clients["moondream"] = client
        children = interrogator._children_map(
            [_candidate("chalice", confidence=0.9)], None, _tiny_image()
        )
        assert children == {}

    def test_limits_number_of_parents_queried_by_profile(self) -> None:
        interrogator = GuidedInterrogator(_settings(profile="fast"))
        client = FakeVLMClient(children_response=["part"])
        interrogator._clients["moondream"] = client
        candidates = [
            _candidate("a", confidence=0.9),
            _candidate("b", confidence=0.85),
            _candidate("c", confidence=0.8),
        ]
        interrogator._children_map(candidates, None, _tiny_image())
        assert len(client.children_calls) == 1  # fast profile caps at 1


# ── Reasoner ranking ──────────────────────────────────────────────────────


class TestReasonerRanking:
    def test_reasoner_reorders_and_updates_phrases(self) -> None:
        interrogator = GuidedInterrogator(_settings(profile="deep"))
        candidates = [
            _candidate("cross", confidence=0.5),
            _candidate("chalice", confidence=0.9),
        ]
        client = FakeVLMClient(
            text_response=(
                '{"candidates": ['
                '{"canonical_label": "cross", "detector_phrases": ["ornate cross"]}, '
                '{"canonical_label": "chalice"}'
                "]}"
            )
        )
        interrogator._clients["qwen3.5"] = client
        ranked = interrogator._reason_and_rank(candidates, None, {}, stage="guided")
        assert [c.canonical_label for c in ranked] == ["cross", "chalice"]
        cross = next(c for c in ranked if c.canonical_label == "cross")
        assert cross.detector_phrases == ["ornate cross"]

    def test_reasoner_invalid_json_falls_back_to_heuristic_order(self) -> None:
        interrogator = GuidedInterrogator(_settings(profile="deep"))
        candidates = [
            _candidate("cross", confidence=0.5),
            _candidate("chalice", confidence=0.9),
        ]
        client = FakeVLMClient(text_response="not json at all")
        interrogator._clients["qwen3.5"] = client
        ranked = interrogator._reason_and_rank(candidates, None, {}, stage="guided")
        assert [c.canonical_label for c in ranked] == ["chalice", "cross"]

    def test_reasoner_exception_falls_back_to_heuristic_order(self) -> None:
        interrogator = GuidedInterrogator(_settings(profile="deep"))
        candidates = [
            _candidate("cross", confidence=0.5),
            _candidate("chalice", confidence=0.9),
        ]

        class RaisingClient:
            def query_text(self, prompt, *, num_predict=256):
                raise RuntimeError("down")

        interrogator._clients["qwen3.5"] = RaisingClient()
        ranked = interrogator._reason_and_rank(candidates, None, {}, stage="guided")
        assert [c.canonical_label for c in ranked] == ["chalice", "cross"]

    def test_reasoner_skipped_for_fast_profile(self) -> None:
        interrogator = GuidedInterrogator(_settings(profile="fast"))
        assert (
            interrogator._should_run_reasoner(
                [_candidate("cross", confidence=0.9)], None, stage="primary"
            )
            is False
        )

    def test_reason_and_rank_empty_candidates_returns_empty(self) -> None:
        interrogator = GuidedInterrogator(_settings())
        assert interrogator._reason_and_rank([], None, {}, stage="primary") == []

    def test_parse_reasoner_response_ignores_unknown_labels(self) -> None:
        interrogator = GuidedInterrogator(_settings())
        existing = [_candidate("cross")]
        ranked = interrogator._parse_reasoner_response(
            '{"candidates": [{"canonical_label": "unknown_thing"}]}', existing
        )
        assert ranked == []


# ── _merge_candidates ─────────────────────────────────────────────────────


class TestMergeCandidates:
    def test_higher_confidence_candidate_replaces_existing(self) -> None:
        interrogator = GuidedInterrogator(_settings())
        left = [_candidate("cross", confidence=0.5, phrases=["cross"])]
        right = [_candidate("cross", confidence=0.9, phrases=["metal cross"])]
        merged = interrogator._merge_candidates(left, right)
        assert len(merged) == 1
        assert merged[0].confidence == 0.9
        assert merged[0].detector_phrases == ["metal cross"]

    def test_lower_confidence_candidate_only_merges_phrases(self) -> None:
        interrogator = GuidedInterrogator(_settings())
        left = [_candidate("cross", confidence=0.9, phrases=["cross"])]
        right = [_candidate("cross", confidence=0.3, phrases=["ornate cross"])]
        merged = interrogator._merge_candidates(left, right)
        assert len(merged) == 1
        assert merged[0].confidence == 0.9
        assert "ornate cross" in merged[0].detector_phrases

    def test_new_label_is_appended(self) -> None:
        interrogator = GuidedInterrogator(_settings())
        left = [_candidate("cross")]
        right = [_candidate("chalice")]
        merged = interrogator._merge_candidates(left, right)
        assert {c.canonical_label for c in merged} == {"cross", "chalice"}


# ── _prepare_image / _iter_tiles ─────────────────────────────────────────


class TestImagePreparation:
    def test_small_image_untouched(self) -> None:
        interrogator = GuidedInterrogator(_settings())
        image = _tiny_image(size=100)
        prepared = interrogator._prepare_image(image)
        assert prepared.shape == image.shape

    def test_large_image_is_downscaled(self) -> None:
        interrogator = GuidedInterrogator(_settings())
        image = np.zeros((2000, 1000, 3), dtype=np.uint8)
        prepared = interrogator._prepare_image(image)
        assert max(prepared.shape[:2]) == 1280

    def test_iter_tiles_splits_into_four_quadrants(self) -> None:
        interrogator = GuidedInterrogator(_settings())
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        tiles = interrogator._iter_tiles(image)
        assert len(tiles) == 4
        assert tiles[0].shape == (20, 30, 3)


# ── generic terms / phrase ranking ────────────────────────────────────────


class TestGenericTermsFromLabel:
    def test_known_alias_expands_terms(self) -> None:
        interrogator = GuidedInterrogator(_settings())
        terms = interrogator._generic_terms_from_label("iPhone")
        assert "smartphone" in terms

    def test_unknown_label_gets_object_suffix(self) -> None:
        interrogator = GuidedInterrogator(_settings())
        terms = interrogator._generic_terms_from_label("gizmo")
        assert any("object" in t for t in terms)

    def test_underscored_label_is_normalized(self) -> None:
        interrogator = GuidedInterrogator(_settings())
        terms = interrogator._generic_terms_from_label("power_cable")
        assert "power cable" in terms


class TestCandidatesFromConfirmedLabels:
    def test_uses_knowledge_when_available(self) -> None:
        pack = KnowledgePack(
            domain=KnowledgeDomain(name="test"),
            objects=[
                ObjectKnowledge(canonical="cross", detector_phrases=["ornate cross"])
            ],
        )
        interrogator = GuidedInterrogator(_settings())
        candidates = interrogator._candidates_from_confirmed_labels(["cross"], pack)
        assert candidates[0].detector_phrases[0] == "ornate cross"

    def test_falls_back_to_generic_terms_without_knowledge(self) -> None:
        interrogator = GuidedInterrogator(_settings())
        candidates = interrogator._candidates_from_confirmed_labels(["gizmo"], None)
        assert candidates[0].canonical_label == "gizmo"
        assert candidates[0].detector_phrases


# ── parse_label_candidates / rank_detector_phrases edge cases ────────────


class TestParseLabelCandidatesEdgeCases:
    def test_empty_string_returns_empty_list(self) -> None:
        assert parse_label_candidates("") == []

    def test_rejects_pure_numeric_chunks(self) -> None:
        assert parse_label_candidates("42, cross") == ["cross"]

    def test_strips_leading_articles(self) -> None:
        assert parse_label_candidates("a cross, an urn, the chalice") == [
            "cross", "urn", "chalice",
        ]

    def test_deduplicates_case_insensitively(self) -> None:
        assert parse_label_candidates("Cross, cross, CROSS") == ["Cross"]

    def test_respects_limit(self) -> None:
        text = ", ".join(f"item{i}" for i in range(20))
        assert len(parse_label_candidates(text, limit=3)) == 3

    def test_rejects_single_character_chunks(self) -> None:
        assert parse_label_candidates("x, cross") == ["cross"]


class TestRankDetectorPhrases:
    def test_deduplicates_and_orders_generic_first(self) -> None:
        phrases = rank_detector_phrases(
            canonical="cross",
            aliases=["cross"],
            generic_terms=["metal cross"],
            description="an ornate cross",
        )
        assert phrases[0] == "metal cross"
        assert phrases.count("cross") == 1


class TestFilterChildPartsProfiles:
    def test_fast_profile_caps_at_three(self) -> None:
        interrogator = GuidedInterrogator(_settings(profile="fast"))
        parts = interrogator._filter_child_parts(
            ["a", "b", "c", "d", "e"], "parent"
        )
        assert len(parts) == 3

    def test_deep_profile_caps_at_eight(self) -> None:
        interrogator = GuidedInterrogator(_settings(profile="deep"))
        parts = interrogator._filter_child_parts(
            [f"part{i}" for i in range(10)], "parent"
        )
        assert len(parts) == 8

    def test_rejects_part_equal_to_parent_word_set(self) -> None:
        interrogator = GuidedInterrogator(_settings())
        parts = interrogator._filter_child_parts(["cross handle", "handle cross"], "cross handle")
        assert parts == []
