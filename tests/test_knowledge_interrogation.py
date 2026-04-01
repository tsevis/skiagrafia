from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from core.interrogation import (
    GuidedInterrogator,
    InterrogationCandidate,
    InterrogationSettings,
    parse_label_candidates,
    rank_detector_phrases,
)
from core.knowledge import KnowledgePack, load_knowledge_pack


class KnowledgePackTests(unittest.TestCase):
    def test_load_valid_knowledge_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            guide = root / "skiagrafia_guide.toml"
            guide.write_text(
                textwrap.dedent(
                    """
                    [domain]
                    name = "Greek Orthodox liturgical artifacts"
                    description = "Rare ornate church objects"

                    [batch_defaults]
                    preferred_vlm = "minicpm-v"
                    enable_tiling = true

                    [[objects]]
                    canonical = "ripidion"
                    aliases = ["liturgical fan"]
                    generic_terms = ["ornate metal fan"]
                    description = "round embossed ceremonial fan with handle"
                    parts = ["handle"]
                    detector_phrases = ["ceremonial fan", "ornate metal fan"]
                    """
                ),
                encoding="utf-8",
            )
            notes = root / "skiagrafia_guide.md"
            notes.write_text("# Notes", encoding="utf-8")

            pack = KnowledgePack.load(guide)

            self.assertEqual(pack.domain.name, "Greek Orthodox liturgical artifacts")
            self.assertEqual(pack.objects[0].canonical, "ripidion")
            self.assertEqual(pack.batch_defaults.preferred_vlm, "minicpm-v")
            self.assertEqual(pack.notes_markdown, "# Notes")

    def test_load_knowledge_pack_missing_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertIsNone(load_knowledge_pack(tmpdir))


class InterrogationHelpersTests(unittest.TestCase):
    def test_parse_candidates_from_list_and_prose(self) -> None:
        self.assertEqual(
            parse_label_candidates("cross, chalice, spoon"),
            ["cross", "chalice", "spoon"],
        )
        self.assertEqual(
            parse_label_candidates("The image shows ornate cross, chalice; spoon."),
            ["ornate cross", "chalice", "spoon"],
        )

    def test_rank_detector_phrases_prefers_generic_aliases(self) -> None:
        phrases = rank_detector_phrases(
            canonical="ripidion",
            aliases=["liturgical fan"],
            generic_terms=["ornate metal fan", "ceremonial fan"],
            description="round embossed ceremonial fan with handle",
            limit=4,
        )
        self.assertEqual(
            phrases[:3],
            ["ornate metal fan", "ceremonial fan", "liturgical fan"],
        )

    def test_composition_prompt_is_explicitly_whole_object_first(self) -> None:
        interrogator = GuidedInterrogator(
            InterrogationSettings(
                host="http://localhost:11434",
                primary_vlm="moondream",
                fallback_vlms=["minicpm-v"],
                reasoner_model="qwen3.5",
            )
        )
        prompt = interrogator._build_prompt(None, "composition")
        self.assertIn("whole foreground composition", prompt)
        self.assertIn("main whole objects", prompt)

    def test_reasoner_is_skipped_for_fast_high_confidence_results(self) -> None:
        interrogator = GuidedInterrogator(
            InterrogationSettings(
                host="http://localhost:11434",
                primary_vlm="moondream",
                fallback_vlms=["minicpm-v"],
                reasoner_model="qwen3.5",
            )
        )
        should_run = interrogator._should_run_reasoner(
            [
                InterrogationCandidate(
                    canonical_label="cross",
                    display_label="cross",
                    detector_phrases=["cross", "metal cross"],
                    confidence=0.85,
                )
            ],
            knowledge_pack=None,
            stage="primary",
        )
        self.assertFalse(should_run)

    def test_child_parts_filter_rejects_parent_echo_and_drinks(self) -> None:
        interrogator = GuidedInterrogator(
            InterrogationSettings(
                host="http://localhost:11434",
                primary_vlm="moondream",
                fallback_vlms=["minicpm-v"],
                reasoner_model="qwen3.5",
                profile="balanced",
            )
        )
        parts = interrogator._filter_child_parts(
            ["chalice", "cup", "stem", "coffee", "base", "hot chocolate"],
            "chalice",
        )
        self.assertEqual(parts, ["cup", "stem", "base"])


if __name__ == "__main__":
    unittest.main()
