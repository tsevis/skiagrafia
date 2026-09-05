"""test_gui_preferences.py  --  Windowed tests for Preferences and the guide editor.

Marked `gui` automatically (via tk_root) and excluded from a plain pytest
run. Run deliberately with: pytest -m gui

ISOLATION IS THE POINT HERE. Unlike the other GUI suites, this window
*writes* to the user's real configuration: _save calls save_preferences,
which writes ~/.config/skiagrafia/preferences.json, and the Templates tab
reads ~/.config/skiagrafia/templates. Every test therefore redirects both
    utils.preferences._CONFIG_DIR  (resolved at call time by _prefs_path)
    Path.home                      (used directly by BatchTemplate)
at a tmp_path. A test that forgot one would quietly overwrite real settings,
so isolation itself is asserted in TestIsolation below.

Never called from here, and why:
  _browse_output_dir / _browse_models_dir / _open_guide / _save_guide_as
      block on a native modal file chooser
  _reveal_templates
      mkdirs the templates directory and shells out to `open` (Finder)
  _download_missing
      downloads multi-GB model weights
  _test_backend
      makes a live call to the configured VLM server
  _save_guide with _current_path unset
      falls through to _save_guide_as, i.e. the modal chooser
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import utils.preferences as prefs_mod
from utils.preferences import DEFAULT_PREFERENCES


class _StubApp:
    """Stand-in for MainWindow: PreferencesWindow only needs these three."""

    def __init__(self, root, prefs: dict) -> None:
        self.root = root
        self.prefs = prefs
        self.applied: list[dict] = []

    def apply_preferences(self, prefs: dict) -> None:
        self.applied.append(dict(prefs))


@pytest.fixture
def sandbox_home(tmp_path, monkeypatch):
    """Redirect every route to the real config directory into tmp_path."""
    config_dir = tmp_path / ".config" / "skiagrafia"
    config_dir.mkdir(parents=True)
    monkeypatch.setattr(prefs_mod, "_CONFIG_DIR", config_dir)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def prefs_app(tk_root, tmp_path, sandbox_home):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    prefs = dict(DEFAULT_PREFERENCES)
    prefs["models_directory"] = str(models_dir)
    prefs["output_directory"] = str(tmp_path / "out")
    return _StubApp(tk_root, prefs)


@pytest.fixture
def prefs_window(prefs_app, tk_root):
    from ui.preferences.preferences_window import PreferencesWindow

    window = PreferencesWindow(prefs_app)
    tk_root.update()
    yield window
    try:
        window._win.destroy()
    except Exception:
        pass  # _save already destroyed it


# ── Isolation ───────────────────────────────────────────────────────────────


class TestIsolation:
    """If these fail, the other tests may be writing real user settings."""

    def test_config_dir_is_redirected(self, sandbox_home) -> None:
        assert prefs_mod._prefs_path().is_relative_to(sandbox_home)

    def test_home_is_redirected(self, sandbox_home) -> None:
        assert Path.home() == sandbox_home

    def test_saving_writes_inside_the_sandbox_only(
        self, prefs_window, sandbox_home
    ) -> None:
        prefs_window._save()
        written = prefs_mod._prefs_path()
        assert written.is_file()
        assert written.is_relative_to(sandbox_home)


# ── Construction ────────────────────────────────────────────────────────────


class TestPreferencesWindowConstruction:
    def test_window_opens(self, prefs_window) -> None:
        assert prefs_window._win.winfo_exists()

    def test_it_is_titled(self, prefs_window) -> None:
        assert prefs_window._win.title() == "Preferences"

    def test_all_six_tabs_are_present(self, prefs_window) -> None:
        assert len(prefs_window._notebook.tabs()) == 6

    def test_tab_labels(self, prefs_window) -> None:
        labels = [
            prefs_window._notebook.tab(t, "text").strip()
            for t in prefs_window._notebook.tabs()
        ]
        assert labels == [
            "General",
            "Models",
            "Pipeline",
            "Appearance",
            "Templates",
            "Domain Guides",
        ]

    def test_it_copies_rather_than_aliases_app_prefs(
        self, prefs_window, prefs_app
    ) -> None:
        prefs_window._prefs["output_directory"] = "/mutated"
        assert prefs_app.prefs["output_directory"] != "/mutated"

    def test_the_guide_editor_tab_is_built(self, prefs_window) -> None:
        assert prefs_window._guide_editor is not None


# ── Saving ──────────────────────────────────────────────────────────────────


class TestSave:
    def test_save_persists_an_edited_value(
        self, prefs_window, tk_root
    ) -> None:
        prefs_window._ollama_model_var.set("qwen2.5vl:7b")
        prefs_window._save()
        tk_root.update()
        saved = json.loads(prefs_mod._prefs_path().read_text())
        assert saved["ollama_model"] == "qwen2.5vl:7b"

    def test_save_notifies_the_app(self, prefs_window, prefs_app) -> None:
        prefs_window._save()
        assert len(prefs_app.applied) == 1

    def test_save_closes_the_window(self, prefs_window, tk_root) -> None:
        prefs_window._save()
        tk_root.update()
        assert not prefs_window._win.winfo_exists()

    def test_cancel_neither_writes_nor_notifies(
        self, prefs_window, prefs_app, tk_root
    ) -> None:
        prefs_window._win.destroy()
        tk_root.update()
        assert prefs_app.applied == []
        assert not prefs_mod._prefs_path().exists()

    def test_saved_payload_covers_every_tab(self, prefs_window) -> None:
        prefs_window._save()
        saved = json.loads(prefs_mod._prefs_path().read_text())
        assert {
            "output_directory",      # General
            "models_directory",      # Models
            "sam_box_threshold",     # Pipeline
            "theme",                 # Appearance
        } <= set(saved)

    def test_boolean_round_trips_as_json_bool(self, prefs_window) -> None:
        prefs_window._scan_boxes_var.set(False)
        prefs_window._save()
        saved = json.loads(prefs_mod._prefs_path().read_text())
        assert saved["scan_preview_show_boxes"] is False

    def test_numeric_round_trips(self, prefs_window) -> None:
        prefs_window._sam_box_var.set(0.42)
        prefs_window._save()
        saved = json.loads(prefs_mod._prefs_path().read_text())
        assert saved["sam_box_threshold"] == pytest.approx(0.42)


# ── Models tab ──────────────────────────────────────────────────────────────


class TestModelsTab:
    def test_default_models_dir_is_stored_as_empty(self, prefs_window) -> None:
        # An explicit path equal to the default is normalised away, so the
        # preference keeps tracking the default if it ever moves.
        from utils.preferences import DEFAULT_MODELS_DIR

        prefs_window._models_dir_var.set(str(DEFAULT_MODELS_DIR))
        prefs_window._save()
        saved = json.loads(prefs_mod._prefs_path().read_text())
        assert saved["models_directory"] == ""

    def test_custom_models_dir_is_kept(self, prefs_window, tmp_path) -> None:
        custom = tmp_path / "elsewhere"
        custom.mkdir()
        prefs_window._models_dir_var.set(str(custom))
        prefs_window._save()
        saved = json.loads(prefs_mod._prefs_path().read_text())
        assert saved["models_directory"] == str(custom)

    def test_scan_lists_the_registry(self, prefs_window, tk_root) -> None:
        # The sandboxed models dir is empty, so every known model is missing.
        prefs_window._scan_models()
        tk_root.update()
        rows = prefs_window._models_tree.get_children()
        assert rows, "model registry produced no rows"

    def test_missing_models_are_reported_missing(
        self, prefs_window, tk_root
    ) -> None:
        prefs_window._scan_models()
        tk_root.update()
        statuses = {
            prefs_window._models_tree.item(row, "values")[2]
            for row in prefs_window._models_tree.get_children()
        }
        assert statuses == {"Missing"}

    def test_stray_weight_files_are_listed_as_extra(
        self, prefs_window, tk_root, tmp_path
    ) -> None:
        (tmp_path / "models" / "stray.pth").write_bytes(b"x" * 2048)
        prefs_window._scan_models()
        tk_root.update()
        rows = [
            prefs_window._models_tree.item(row, "values")
            for row in prefs_window._models_tree.get_children()
        ]
        assert any(r[0] == "stray.pth" and r[2] == "Extra" for r in rows)

    def test_reset_restores_the_default_path(self, prefs_window, tk_root) -> None:
        from utils.preferences import DEFAULT_MODELS_DIR

        prefs_window._models_dir_var.set("/somewhere/else")
        prefs_window._reset_models_dir()
        tk_root.update()
        assert prefs_window._models_dir_var.get() == str(DEFAULT_MODELS_DIR)


# ── Templates tab ───────────────────────────────────────────────────────────


class TestTemplatesTab:
    @staticmethod
    def _save_template(name: str) -> None:
        from core.batch_template import BatchTemplate

        BatchTemplate(
            name=name,
            source_image="/tmp/a.png",
            confirmed_labels=["chalice"],
            confirmed_children={"chalice": ["stem"]},
            output_mode="vector+bitmap",
            recursion_depth=2,
            corner_threshold=60,
            speckle=8,
            smoothing=5,
            length_threshold=4.0,
            vtracer_quality="balanced",
        ).save()

    def test_empty_templates_dir_lists_nothing(self, prefs_window) -> None:
        assert prefs_window._templates_tree.get_children() == ()

    def test_saved_template_is_listed(self, prefs_window, tk_root) -> None:
        self._save_template("Liturgical")
        prefs_window._load_templates()
        tk_root.update()
        names = [
            prefs_window._templates_tree.item(r, "values")[0]
            for r in prefs_window._templates_tree.get_children()
        ]
        assert names == ["Liturgical"]

    def test_listing_shows_the_label_count(self, prefs_window, tk_root) -> None:
        self._save_template("Liturgical")
        prefs_window._load_templates()
        tk_root.update()
        row = prefs_window._templates_tree.item(
            prefs_window._templates_tree.get_children()[0], "values"
        )
        assert int(row[1]) == 1

    def test_reloading_does_not_duplicate_rows(
        self, prefs_window, tk_root
    ) -> None:
        self._save_template("Liturgical")
        prefs_window._load_templates()
        prefs_window._load_templates()
        tk_root.update()
        assert len(prefs_window._templates_tree.get_children()) == 1

    def test_corrupt_template_does_not_break_the_listing(
        self, prefs_window, tk_root, sandbox_home
    ) -> None:
        self._save_template("Good One")
        templates = sandbox_home / ".config" / "skiagrafia" / "templates"
        (templates / "broken.json").write_text("{ not json")
        prefs_window._load_templates()
        tk_root.update()
        names = [
            prefs_window._templates_tree.item(r, "values")[0]
            for r in prefs_window._templates_tree.get_children()
        ]
        assert names == ["Good One"]


# ── Appearance tab ──────────────────────────────────────────────────────────


class TestAppearanceTab:
    def test_swatch_follows_the_canvas_background(
        self, prefs_window, tk_root
    ) -> None:
        prefs_window._canvas_bg_var.set("#FF0000")
        prefs_window._update_swatch()
        tk_root.update()
        assert str(prefs_window._canvas_swatch.cget("bg")).lower() == "#ff0000"

    def test_swatch_survives_an_invalid_colour(
        self, prefs_window, tk_root
    ) -> None:
        prefs_window._canvas_bg_var.set("#FF0000")
        prefs_window._update_swatch()
        prefs_window._canvas_bg_var.set("not-a-colour")
        prefs_window._update_swatch()  # TclError is swallowed by design
        tk_root.update()
        assert str(prefs_window._canvas_swatch.cget("bg")).lower() == "#ff0000"

    def test_theme_choice_is_saved(self, prefs_window) -> None:
        prefs_window._theme_var.set("dark")
        prefs_window._save()
        saved = json.loads(prefs_mod._prefs_path().read_text())
        assert saved["theme"] == "dark"

    def test_opacity_is_saved_as_a_percentage(self, prefs_window) -> None:
        # Opacity is an IntVar holding 0-100; canvas_drawing divides by 100.
        prefs_window._mask_opacity_var.set(55)
        prefs_window._save()
        saved = json.loads(prefs_mod._prefs_path().read_text())
        assert saved["mask_overlay_opacity"] == 55

    def test_opacity_defaults_are_percentages_not_fractions(self) -> None:
        # A fraction here would render every overlay effectively invisible.
        for key in (
            "mask_overlay_opacity",
            "scan_preview_box_opacity",
            "scan_preview_heatmap_opacity",
        ):
            assert 1 <= DEFAULT_PREFERENCES[key] <= 100


# ── Domain guide editor ─────────────────────────────────────────────────────


class TestGuideEditor:
    def test_it_renders_toml_on_construction(self, prefs_window) -> None:
        editor = prefs_window._guide_editor
        assert editor._toml_text.get("1.0", "end").strip()

    def test_domain_name_reaches_the_toml_preview(
        self, prefs_window, tk_root
    ) -> None:
        editor = prefs_window._guide_editor
        editor._dom_name_var.set("Greek Orthodox liturgical artifacts")
        tk_root.update()
        assert "Greek Orthodox" in editor._toml_text.get("1.0", "end")

    def test_adding_an_object_extends_the_model(
        self, prefs_window, tk_root
    ) -> None:
        editor = prefs_window._guide_editor
        before = len(editor._objects)
        editor._add_object()
        tk_root.update()
        assert len(editor._objects) == before + 1

    def test_removing_an_object_shrinks_the_model(
        self, prefs_window, tk_root
    ) -> None:
        editor = prefs_window._guide_editor
        editor._add_object()
        tk_root.update()
        editor._remove_object(0)
        tk_root.update()
        assert editor._objects == []

    def test_new_guide_clears_the_form(self, prefs_window, tk_root) -> None:
        editor = prefs_window._guide_editor
        editor._dom_name_var.set("Something")
        editor._add_object()
        tk_root.update()
        editor._new_guide()
        tk_root.update()
        assert editor._dom_name_var.get() == ""
        assert editor._objects == []
        assert editor._current_path is None

    def test_save_writes_a_loadable_guide(
        self, prefs_window, tk_root, tmp_path
    ) -> None:
        # _current_path must be set: _save_guide falls through to a modal
        # file chooser when it is None.
        from core.knowledge import load_knowledge_pack

        folder = tmp_path / "guided"
        folder.mkdir()
        editor = prefs_window._guide_editor
        editor._dom_name_var.set("Liturgical objects")
        editor._current_path = folder / "skiagrafia_guide.toml"
        tk_root.update()

        editor._save_guide()

        assert editor._current_path.is_file()
        assert load_knowledge_pack(folder) is not None

    def test_saved_guide_round_trips_the_domain_name(
        self, prefs_window, tk_root, tmp_path
    ) -> None:
        from core.knowledge import load_knowledge_pack

        folder = tmp_path / "guided2"
        folder.mkdir()
        editor = prefs_window._guide_editor
        editor._dom_name_var.set("Liturgical objects")
        editor._current_path = folder / "skiagrafia_guide.toml"
        tk_root.update()

        editor._save_guide()

        assert load_knowledge_pack(folder).name == "Liturgical objects"

    def test_path_label_reflects_the_saved_location(
        self, prefs_window, tk_root, tmp_path
    ) -> None:
        editor = prefs_window._guide_editor
        target = tmp_path / "guide.toml"
        editor._current_path = target
        editor._update_path_label()
        tk_root.update()
        assert str(target) in editor._path_label.cget("text")

    def test_path_label_says_unsaved_before_a_path_exists(
        self, prefs_window, tk_root
    ) -> None:
        editor = prefs_window._guide_editor
        editor._new_guide()
        tk_root.update()
        assert editor._path_label.cget("text") == "Unsaved"
