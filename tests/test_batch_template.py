"""test_batch_template.py  --  BatchTemplate save/load/list-all behaviour.

CRITICAL: BatchTemplate.save() and .list_all() read/write
~/.config/skiagrafia/templates by default. Every test here monkeypatches
Path.home() to a pytest tmp_path so the user's real templates directory is
never touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.batch_template import BatchTemplate


def _make_template(name: str = "My Template", **overrides: object) -> BatchTemplate:
    fields: dict[str, object] = dict(
        name=name,
        source_image="/images/source.png",
        confirmed_labels=["cross", "chalice"],
        confirmed_children={"chalice": ["cup", "stem"]},
        output_mode="vector+bitmap",
        recursion_depth=2,
        corner_threshold=60,
        speckle=8,
        smoothing=4,
        length_threshold=4.0,
        vtracer_quality="balanced",
    )
    fields.update(overrides)
    return BatchTemplate(**fields)


@pytest.fixture()
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect Path.home() to an isolated tmp_path for this test only."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


class TestSave:
    def test_save_writes_json_under_config_templates_dir(
        self, fake_home: Path
    ) -> None:
        template = _make_template(name="My Template")

        path = template.save()

        expected_dir = fake_home / ".config" / "skiagrafia" / "templates"
        assert path.parent == expected_dir
        assert path.name == "my_template.json"
        assert path.exists()

    def test_save_populates_created_at(self, fake_home: Path) -> None:
        template = _make_template()
        assert template.created_at == ""

        template.save()

        assert template.created_at != ""

    def test_save_round_trips_via_load(self, fake_home: Path) -> None:
        template = _make_template(name="Round Trip", preferred_vlm="minicpm-v")

        path = template.save()
        loaded = BatchTemplate.load(path)

        assert loaded.name == "Round Trip"
        assert loaded.confirmed_labels == ["cross", "chalice"]
        assert loaded.preferred_vlm == "minicpm-v"
        assert loaded.created_at == template.created_at


class TestNameSlugging:
    @pytest.mark.parametrize(
        ("name", "expected_filename"),
        [
            ("simple", "simple.json"),
            ("Spaced Out Name", "spaced_out_name.json"),
            ("UPPERCASE", "uppercase.json"),
            ("Mixed Case Name", "mixed_case_name.json"),
            # Runs of unsafe characters collapse to one "_", and leading or
            # trailing separators are trimmed, so the stem stays filename-safe.
            ("trailing space ", "trailing_space.json"),
            ("multi   spaces", "multi_spaces.json"),
            ("punct'n! (ok)", "punct_n_ok.json"),
        ],
    )
    def test_slug_matches_expected_filename(
        self, fake_home: Path, name: str, expected_filename: str
    ) -> None:
        template = _make_template(name=name)

        path = template.save()

        assert path.name == expected_filename

    def test_name_with_path_separator_is_flattened(self, fake_home: Path) -> None:
        """A '/' in the name must not escape the templates directory."""
        template = _make_template(name="Foo/Bar")

        path = template.save()

        assert path.parent == fake_home / ".config" / "skiagrafia" / "templates"
        assert "/" not in path.name
        assert path.exists()

    def test_name_cannot_traverse_out_of_templates_dir(
        self, fake_home: Path
    ) -> None:
        templates = fake_home / ".config" / "skiagrafia" / "templates"
        template = _make_template(name="../../escaped")

        path = template.save()

        assert path.parent == templates
        assert path.resolve().is_relative_to(templates.resolve())

    def test_overlong_name_is_truncated_to_a_writable_filename(
        self, fake_home: Path
    ) -> None:
        template = _make_template(name="x" * 400)

        path = template.save()

        assert path.exists()
        assert len(path.name.encode()) < 255


class TestLoad:
    def test_load_malformed_json_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "broken.json"
        bad_file.write_text("{not valid json,,,")

        with pytest.raises(Exception):
            BatchTemplate.load(bad_file)

    def test_load_valid_json_missing_required_field_raises(
        self, tmp_path: Path
    ) -> None:
        # "source_image" is required and absent here.
        incomplete = tmp_path / "incomplete.json"
        incomplete.write_text('{"name": "x", "confirmed_labels": []}')

        with pytest.raises(Exception):
            BatchTemplate.load(incomplete)


class TestListAll:
    def test_list_all_returns_empty_when_dir_missing(self, fake_home: Path) -> None:
        assert BatchTemplate.list_all() == []

    def test_list_all_returns_saved_templates_newest_first(
        self, fake_home: Path
    ) -> None:
        first = _make_template(name="First")
        first_path = first.save()

        second = _make_template(name="Second")
        second_path = second.save()

        # Ensure a detectable mtime ordering regardless of filesystem
        # timestamp resolution.
        import os
        import time

        now = time.time()
        os.utime(first_path, (now - 10, now - 10))
        os.utime(second_path, (now, now))

        templates = BatchTemplate.list_all()

        assert [t.name for t in templates] == ["Second", "First"]

    def test_list_all_skips_corrupt_template_file(self, fake_home: Path) -> None:
        """One unreadable template must not break the whole listing."""
        good = _make_template(name="good one")
        good.save()
        templates = fake_home / ".config" / "skiagrafia" / "templates"
        (templates / "corrupt.json").write_text("{ not valid json")

        result = BatchTemplate.list_all()

        assert [t.name for t in result] == ["good one"]
