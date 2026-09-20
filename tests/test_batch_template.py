"""test_batch_template.py  --  BatchTemplate save/load/list-all behaviour.

CRITICAL: BatchTemplate.save() and .list_all() read/write
~/.config/skiagrafia/templates by default. Every test here monkeypatches
Path.home() to a pytest tmp_path so the user's real templates directory is
never touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.batch_template import BatchTemplate


def _make_template(name: str = "My Template", **overrides: Any) -> BatchTemplate:
    fields: dict[str, Any] = {
        "name": name,
        "source_image": "/images/source.png",
        "confirmed_labels": ["cross", "chalice"],
        "confirmed_children": {"chalice": ["cup", "stem"]},
        "output_mode": "vector+bitmap",
        "corner_threshold": 60,
        "speckle": 8,
        "length_threshold": 4.0,
        "vtracer_quality": "balanced",
    }
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
        """The stamp lands in the FILE. It never belonged on the caller's
        object, so this reads it back from where it is actually needed."""
        template = _make_template()
        assert template.created_at == ""

        path = template.save()

        assert BatchTemplate.load(path).created_at != ""

    def test_save_round_trips_via_load(self, fake_home: Path) -> None:
        template = _make_template(
            name="Round Trip",
            preferred_vlm="minicpm-v",
            selection_request="Select computers only.\nExclude captions.",
            guide_path="/guides/apple.toml",
            guide_name="Apple — The First 50 Years",
        )

        path = template.save()
        loaded = BatchTemplate.load(path)

        assert loaded.name == "Round Trip"
        assert loaded.confirmed_labels == ["cross", "chalice"]
        assert loaded.preferred_vlm == "minicpm-v"
        assert loaded.selection_request == "Select computers only.\nExclude captions."
        assert loaded.guide_path == "/guides/apple.toml"
        assert loaded.guide_name == "Apple — The First 50 Years"
        # Stamped on write, so it is not equal to the unsaved original's
        # empty value -- it has to be a real timestamp.
        assert loaded.created_at != ""


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

    def test_save_refuses_symlinked_template_target(self, fake_home: Path) -> None:
        templates = fake_home / ".config" / "skiagrafia" / "templates"
        templates.mkdir(parents=True)
        external = fake_home / "outside.json"
        (templates / "symlinked.json").symlink_to(external)

        with pytest.raises(ValueError, match="symlink"):
            _make_template(name="symlinked").save()

        assert not external.exists()


class TestLoad:
    def test_load_malformed_json_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "broken.json"
        bad_file.write_text("{not valid json,,,")

        with pytest.raises(ValidationError):
            BatchTemplate.load(bad_file)

    def test_load_valid_json_missing_required_field_raises(
        self, tmp_path: Path
    ) -> None:
        # "source_image" is required and absent here.
        incomplete = tmp_path / "incomplete.json"
        incomplete.write_text('{"name": "x", "confirmed_labels": []}')

        with pytest.raises(ValidationError):
            BatchTemplate.load(incomplete)

    def test_legacy_template_without_selection_request_loads_with_empty_default(
        self, tmp_path: Path
    ) -> None:
        legacy = _make_template().model_dump()
        legacy.pop("selection_request")
        legacy.pop("guide_name")
        path = tmp_path / "legacy.json"
        path.write_text(__import__("json").dumps(legacy), encoding="utf-8")

        loaded = BatchTemplate.load(path)

        assert loaded.selection_request == ""
        assert loaded.guide_name is None


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

    def test_list_all_with_paths_preserves_template_source_paths(
        self, fake_home: Path
    ) -> None:
        template = _make_template(name="with path")
        path = template.save()

        listed = BatchTemplate.list_all_with_paths()

        # Compared against what was written rather than the in-memory
        # original: save() stamps the copy it writes and leaves this one be.
        assert listed == [(path, BatchTemplate.load(path))]
        assert listed[0][1].created_at != ""


class TestSaveDoesNotMutate:
    def test_save_leaves_the_receiver_untouched(self, tmp_path, monkeypatch) -> None:
        """save() stamped created_at onto the caller's own object.

        The template a caller holds is theirs; writing it to disk is not a
        reason to edit it under them. Saving twice silently changed a field
        of an object someone else may still be reading.
        """
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        template = _make_template(name="unmutated")

        path = template.save()

        assert template.created_at == ""
        written = BatchTemplate.load(path)
        assert written.created_at != ""


def test_a_template_saved_before_the_inert_fields_were_removed_still_loads(tmp_path):
    """Templates already on disk carry recursion_depth and smoothing, which the
    pipeline never read. Removing the fields must not make a person's saved
    configurations unreadable."""
    saved = {
        "name": "Older template",
        "created_at": "2026-01-01T00:00:00+00:00",
        "source_image": "/tmp/product.png",
        "confirmed_labels": ["product"],
        "confirmed_children": {},
        "output_mode": "vector+bitmap",
        "recursion_depth": 3,
        "corner_threshold": 60,
        "speckle": 8,
        "smoothing": 4,
        "length_threshold": 4.0,
        "vtracer_quality": "balanced",
    }
    path = tmp_path / "older.json"
    path.write_text(json.dumps(saved), encoding="utf-8")

    template = BatchTemplate.model_validate_json(path.read_text(encoding="utf-8"))

    assert template.name == "Older template"
    assert template.confirmed_labels == ["product"]
    assert template.vtracer_quality == "balanced"
    assert not hasattr(template, "recursion_depth")
    assert not hasattr(template, "smoothing")
