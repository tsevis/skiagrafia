"""test_knowledge.py  --  KnowledgePack construction, TOML round-trip, I/O.

Extends the coverage already provided by test_knowledge_interrogation.py
(KnowledgePack.load happy path, load_knowledge_pack missing-folder case,
label parsing / ranking helpers used by the interrogator) without repeating
it. This file focuses on: ObjectKnowledge helpers, KnowledgePack.find_object,
error handling in KnowledgePack.load, to_toml/save round-trips,
build_knowledge_pack, default_guide_markdown, and load_knowledge_pack's
success and error branches.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.knowledge import (
    BatchGuideDefaults,
    KnowledgeDomain,
    KnowledgePack,
    ObjectKnowledge,
    build_knowledge_pack,
    default_guide_markdown,
    load_knowledge_pack,
)


class TestObjectKnowledgeAllTerms:
    def test_all_terms_collects_canonical_aliases_generic_and_detector_phrases(
        self,
    ) -> None:
        obj = ObjectKnowledge(
            canonical="cross",
            aliases=["crucifix"],
            generic_terms=["ornate cross"],
            detector_phrases=["metal cross"],
        )
        assert obj.all_terms() == ["cross", "crucifix", "ornate cross", "metal cross"]

    def test_all_terms_dedupes_and_strips_blank_values(self) -> None:
        obj = ObjectKnowledge(
            canonical="cross",
            aliases=["cross", "  ", "crucifix"],
            generic_terms=[""],
        )
        assert obj.all_terms() == ["cross", "crucifix"]


class TestObjectKnowledgeRankedDetectorPhrases:
    def test_prefers_detector_phrases_then_generic_then_aliases_then_canonical(
        self,
    ) -> None:
        obj = ObjectKnowledge(
            canonical="ripidion",
            aliases=["liturgical fan"],
            generic_terms=["ornate metal fan"],
            detector_phrases=["ceremonial fan"],
        )
        assert obj.ranked_detector_phrases(limit=4) == [
            "ceremonial fan",
            "ornate metal fan",
            "liturgical fan",
            "ripidion",
        ]

    def test_appends_description_when_room_remains_under_limit(self) -> None:
        obj = ObjectKnowledge(
            canonical="ripidion",
            description="round embossed ceremonial fan with handle",
        )
        phrases = obj.ranked_detector_phrases(limit=2)
        assert phrases == [
            "ripidion",
            "round embossed ceremonial fan with handle",
        ]

    def test_respects_limit_even_with_description(self) -> None:
        obj = ObjectKnowledge(
            canonical="ripidion",
            aliases=["liturgical fan"],
            generic_terms=["ornate metal fan"],
            detector_phrases=["ceremonial fan"],
            description="a long description",
        )
        phrases = obj.ranked_detector_phrases(limit=1)
        assert phrases == ["ceremonial fan"]


class TestKnowledgePackName:
    def test_name_uses_domain_name_when_present(self) -> None:
        pack = KnowledgePack(
            path="/tmp/whatever/skiagrafia_guide.toml",
            domain=KnowledgeDomain(name="Byzantine Iconography"),
        )
        assert pack.name == "Byzantine Iconography"

    def test_name_falls_back_to_path_stem_when_domain_name_blank(self) -> None:
        pack = KnowledgePack(path="/tmp/my_collection/skiagrafia_guide.toml")
        assert pack.name == "skiagrafia_guide"


class TestKnowledgePackFindObject:
    def _pack(self) -> KnowledgePack:
        return KnowledgePack(
            objects=[
                ObjectKnowledge(
                    canonical="chalice",
                    aliases=["cup"],
                    generic_terms=["drinking vessel"],
                    detector_phrases=["ornate cup"],
                ),
            ]
        )

    def test_matches_canonical_case_insensitively(self) -> None:
        pack = self._pack()
        assert pack.find_object("Chalice") is not None
        assert pack.find_object("Chalice").canonical == "chalice"

    def test_matches_alias(self) -> None:
        pack = self._pack()
        assert pack.find_object("cup") is not None

    def test_matches_generic_term_or_detector_phrase(self) -> None:
        pack = self._pack()
        assert pack.find_object("drinking vessel") is not None
        assert pack.find_object("ornate cup") is not None

    def test_returns_none_when_nothing_matches(self) -> None:
        pack = self._pack()
        assert pack.find_object("spoon") is None


class TestKnowledgePackLoadErrors:
    def test_load_raises_value_error_on_invalid_object_schema(
        self, tmp_path: Path
    ) -> None:
        guide = tmp_path / "skiagrafia_guide.toml"
        guide.write_text(
            textwrap.dedent(
                """
                [domain]
                name = "Test domain"

                [[objects]]
                aliases = ["missing canonical field"]
                """
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="Invalid knowledge pack"):
            KnowledgePack.load(guide)


class TestKnowledgePackToTomlAndSave:
    def _full_pack(self) -> KnowledgePack:
        return KnowledgePack(
            domain=KnowledgeDomain(
                name="Liturgical Objects", description="Church items"
            ),
            batch_defaults=BatchGuideDefaults(
                preferred_vlm="minicpm-v",
                fallback_vlms=["moondream"],
                enable_tiling=True,
                max_aliases_per_object=3,
            ),
            objects=[
                ObjectKnowledge(
                    canonical="chalice",
                    aliases=["cup"],
                    generic_terms=["drinking vessel"],
                    description='ornate "silver" cup',
                    parts=["stem", "base"],
                    detector_phrases=["ornate cup"],
                ),
            ],
        )

    def test_to_toml_round_trips_through_load(self, tmp_path: Path) -> None:
        pack = self._full_pack()
        toml_text = pack.to_toml()
        guide = tmp_path / "skiagrafia_guide.toml"
        guide.write_text(toml_text, encoding="utf-8")

        reloaded = KnowledgePack.load(guide)

        assert reloaded.domain.name == "Liturgical Objects"
        assert reloaded.domain.description == "Church items"
        assert reloaded.batch_defaults.preferred_vlm == "minicpm-v"
        assert reloaded.batch_defaults.fallback_vlms == ["moondream"]
        assert reloaded.batch_defaults.enable_tiling is True
        assert reloaded.batch_defaults.max_aliases_per_object == 3
        assert len(reloaded.objects) == 1
        assert reloaded.objects[0].canonical == "chalice"
        assert reloaded.objects[0].description == 'ornate "silver" cup'
        assert reloaded.objects[0].parts == ["stem", "base"]

    def test_to_toml_omits_batch_defaults_section_when_all_unset(self) -> None:
        pack = KnowledgePack(domain=KnowledgeDomain(name="Bare"))
        toml_text = pack.to_toml()
        assert "[batch_defaults]" not in toml_text

    def test_save_writes_toml_and_markdown_notes(self, tmp_path: Path) -> None:
        pack = self._full_pack()
        pack.notes_markdown = "# Notes\n\nSome guidance."
        guide = tmp_path / "skiagrafia_guide.toml"

        pack.save(guide)

        assert guide.exists()
        assert guide.with_suffix(".md").read_text(encoding="utf-8") == (
            "# Notes\n\nSome guidance."
        )

    def test_save_without_notes_markdown_skips_md_file(self, tmp_path: Path) -> None:
        pack = self._full_pack()
        guide = tmp_path / "skiagrafia_guide.toml"

        pack.save(guide)

        assert guide.exists()
        assert not guide.with_suffix(".md").exists()


class TestBuildKnowledgePackHelper:
    def test_builds_pack_with_objects_and_batch_defaults(self) -> None:
        pack = build_knowledge_pack(
            "/tmp/x/skiagrafia_guide.toml",
            domain_name="Test Domain",
            domain_description="desc",
            object_specs=[{"canonical": "cross", "aliases": ["crucifix"]}],
            preferred_vlm="moondream",
            fallback_vlms=["minicpm-v"],
            enable_tiling=False,
            max_aliases_per_object=2,
            notes_markdown="# notes",
        )

        assert pack.domain.name == "Test Domain"
        assert pack.objects[0].canonical == "cross"
        assert pack.objects[0].aliases == ["crucifix"]
        assert pack.batch_defaults.preferred_vlm == "moondream"
        assert pack.batch_defaults.enable_tiling is False
        assert pack.notes_markdown == "# notes"

    def test_builds_pack_with_no_objects_by_default(self) -> None:
        pack = build_knowledge_pack("/tmp/x/skiagrafia_guide.toml", domain_name="Empty")
        assert pack.objects == []
        assert pack.batch_defaults.fallback_vlms == []


class TestDefaultGuideMarkdown:
    def test_uses_domain_name_as_heading(self) -> None:
        text = default_guide_markdown("My Domain")
        assert text.startswith("# My Domain\n")

    def test_falls_back_to_generic_heading_when_blank(self) -> None:
        text = default_guide_markdown("")
        assert text.startswith("# Skiagrafia Domain Guide\n")


class TestLoadKnowledgePackFunction:
    def test_returns_pack_when_guide_exists_and_is_valid(self, tmp_path: Path) -> None:
        guide = tmp_path / "skiagrafia_guide.toml"
        guide.write_text(
            textwrap.dedent(
                """
                [domain]
                name = "Valid Domain"
                """
            ),
            encoding="utf-8",
        )

        pack = load_knowledge_pack(tmp_path)

        assert pack is not None
        assert pack.domain.name == "Valid Domain"

    def test_returns_none_when_folder_has_no_guide(self, tmp_path: Path) -> None:
        assert load_knowledge_pack(tmp_path) is None

    def test_returns_none_and_logs_when_guide_is_malformed(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        guide = tmp_path / "skiagrafia_guide.toml"
        guide.write_text("this is not [valid toml", encoding="utf-8")

        with caplog.at_level("ERROR"):
            pack = load_knowledge_pack(tmp_path)

        assert pack is None
        assert any("Failed to load knowledge pack" in msg for msg in caplog.messages)
