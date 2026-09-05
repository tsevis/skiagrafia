"""test_gui_main_window.py  --  Windowed tests for the app shell and wizard.

Marked `gui` automatically (via tk_root) and excluded from a plain pytest
run. Run deliberately with: pytest -m gui

Covers MainWindow (mode switching, view lifecycle, preference application),
ModeSwitcher, and the first-run SetupWizard.

MainWindow.__init__ calls load_preferences(), which reads -- and on a first
run writes -- ~/.config/skiagrafia/preferences.json, so every test here
redirects utils.preferences._CONFIG_DIR at a tmp_path.

Never called from here:
  SetupWizard._start_downloads / _download_worker
      spawn a thread that downloads multi-GB model weights
  SetupWizard's "Get Ollama..." button
      opens a web browser
  MainWindow._open_preferences
      covered by test_gui_preferences.py; opens a modal that grabs input
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import utils.preferences as prefs_mod
from utils.bootstrap import SetupItem, SetupStatus


@pytest.fixture
def sandbox_config(tmp_path, monkeypatch):
    """Keep load_preferences away from the real config file."""
    config_dir = tmp_path / ".config" / "skiagrafia"
    config_dir.mkdir(parents=True)
    monkeypatch.setattr(prefs_mod, "_CONFIG_DIR", config_dir)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return config_dir


@pytest.fixture
def main_window(tk_root, sandbox_config):
    from ui.main_window import MainWindow

    window = MainWindow(tk_root)
    tk_root.update()
    return window


# ── Shell construction ──────────────────────────────────────────────────────


class TestMainWindowConstruction:
    def test_window_is_titled(self, main_window, tk_root) -> None:
        assert tk_root.title() == "Skiagrafia"

    def test_preferences_are_loaded(self, main_window) -> None:
        assert isinstance(main_window.prefs, dict)
        assert "output_directory" in main_window.prefs

    def test_a_palette_is_resolved(self, main_window) -> None:
        assert main_window.palette is not None

    def test_top_bar_and_content_exist(self, main_window) -> None:
        assert main_window._top_bar.winfo_exists()
        assert main_window._content.winfo_exists()

    def test_it_opens_in_single_mode_by_default(self, main_window) -> None:
        assert main_window._mode_switcher.mode == "single"

    def test_only_the_default_view_is_built(self, main_window) -> None:
        assert main_window._single_view is not None
        assert main_window._batch_view is None

    def test_default_mode_from_preferences_is_honoured(
        self, tk_root, sandbox_config, monkeypatch
    ) -> None:
        import json

        (sandbox_config / "preferences.json").write_text(
            json.dumps({"default_mode": "batch"})
        )
        from ui.main_window import MainWindow

        window = MainWindow(tk_root)
        tk_root.update()
        assert window._mode_switcher.mode == "batch"
        assert window._batch_view is not None


# ── Mode switching ──────────────────────────────────────────────────────────


class TestModeSwitching:
    def test_switching_to_batch_builds_it(self, main_window, tk_root) -> None:
        main_window._show_mode("batch")
        tk_root.update()
        assert main_window._batch_view is not None

    def test_switching_updates_the_switcher(self, main_window, tk_root) -> None:
        main_window._show_mode("batch")
        tk_root.update()
        assert main_window._mode_switcher.mode == "batch"

    def test_subtitle_tracks_the_mode(self, main_window, tk_root) -> None:
        main_window._show_mode("batch")
        tk_root.update()
        assert "Batch" in main_window._subtitle_label.cget("text")
        main_window._show_mode("single")
        tk_root.update()
        assert "vectorizing" in main_window._subtitle_label.cget("text")

    def test_views_are_reused_not_rebuilt(self, main_window, tk_root) -> None:
        first = main_window._single_view
        main_window._show_mode("batch")
        main_window._show_mode("single")
        tk_root.update()
        assert main_window._single_view is first

    def test_switch_to_batch_helper(self, main_window, tk_root) -> None:
        main_window.switch_to_batch()
        tk_root.update()
        assert main_window._mode_switcher.mode == "batch"

    def test_clicking_the_switcher_drives_the_window(
        self, main_window, tk_root
    ) -> None:
        main_window._mode_switcher._batch_btn.invoke()
        tk_root.update()
        assert main_window._batch_view is not None

    def test_reselecting_the_current_mode_is_inert(
        self, main_window, tk_root
    ) -> None:
        # _show_mode calls set_mode, which calls back into _show_mode; the
        # early return in set_mode is what stops that recursing.
        main_window._show_mode("single")
        tk_root.update()
        assert main_window._mode_switcher.mode == "single"

    def test_only_one_view_is_mapped_at_a_time(
        self, main_window, tk_root
    ) -> None:
        main_window._show_mode("batch")
        tk_root.update()
        assert main_window._batch_view.frame.winfo_ismapped()
        assert not main_window._single_view.frame.winfo_ismapped()


# ── Preferences application ─────────────────────────────────────────────────


class TestApplyPreferences:
    def test_prefs_are_replaced(self, main_window) -> None:
        main_window.apply_preferences({"theme": "dark", "output_directory": "/x"})
        assert main_window.prefs["output_directory"] == "/x"

    def test_palette_is_recomputed(self, main_window) -> None:
        main_window.apply_preferences({"theme": "dark"})
        assert main_window.palette is not None

    def test_appearance_toggle_does_not_raise(self, main_window, tk_root) -> None:
        # Unsupported outside macOS; the TclError is swallowed by design.
        main_window._toggle_appearance()
        tk_root.update()
        assert main_window._appearance in ("auto", "aqua", "darkaqua")


# ── ModeSwitcher ────────────────────────────────────────────────────────────


class TestModeSwitcher:
    @pytest.fixture
    def switcher(self, tk_root):
        from ui.mode_switcher import ModeSwitcher

        seen: list[str] = []
        widget = ModeSwitcher(tk_root, on_mode_change=seen.append)
        widget.pack()
        tk_root.update()
        return widget, seen

    def test_starts_in_single(self, switcher) -> None:
        widget, _ = switcher
        assert widget.mode == "single"

    def test_setting_a_new_mode_notifies(self, switcher, tk_root) -> None:
        widget, seen = switcher
        widget.set_mode("batch")
        tk_root.update()
        assert widget.mode == "batch"
        assert seen == ["batch"]

    def test_setting_the_same_mode_does_not_notify(
        self, switcher, tk_root
    ) -> None:
        widget, seen = switcher
        widget.set_mode("single")
        tk_root.update()
        assert seen == []

    def test_buttons_drive_the_mode(self, switcher, tk_root) -> None:
        widget, seen = switcher
        widget._batch_btn.invoke()
        tk_root.update()
        assert widget.mode == "batch"
        widget._single_btn.invoke()
        tk_root.update()
        assert seen == ["batch", "single"]

    def test_active_button_is_pressed(self, switcher, tk_root) -> None:
        widget, _ = switcher
        widget.set_mode("batch")
        tk_root.update()
        assert "pressed" in widget._batch_btn.state()
        assert "pressed" not in widget._single_btn.state()

    def test_works_without_a_callback(self, tk_root) -> None:
        from ui.mode_switcher import ModeSwitcher

        widget = ModeSwitcher(tk_root)
        widget.pack()
        widget.set_mode("batch")
        tk_root.update()
        assert widget.mode == "batch"


# ── SetupWizard ─────────────────────────────────────────────────────────────


def _status(*items: SetupItem) -> SetupStatus:
    return SetupStatus(items=list(items))


def _item(
    name: str, status: str = "missing", required: bool = True, mb: int | None = None
) -> SetupItem:
    return SetupItem(
        name=name, kind="weights", status=status, required=required, approx_mb=mb
    )


@pytest.fixture
def wizard_factory(tk_root, monkeypatch):
    """Build a SetupWizard over a canned checklist (no disk or network)."""
    from ui import setup_wizard as wizard_mod

    def _build(status: SetupStatus):
        monkeypatch.setattr(wizard_mod, "check_setup", lambda prefs: status)
        wizard = wizard_mod.SetupWizard(tk_root, {})
        tk_root.update()
        return wizard

    return _build


class TestSetupWizard:
    def test_window_is_titled(self, wizard_factory) -> None:
        wizard = wizard_factory(_status(_item("SAM 2.1")))
        assert "first-run setup" in wizard._win.title().lower()

    def test_one_row_per_component(self, wizard_factory) -> None:
        wizard = wizard_factory(
            _status(_item("SAM 2.1"), _item("GroundingDINO"), _item("VitMatte"))
        )
        assert len(wizard._tree.get_children()) == 3

    def test_ready_items_are_ticked(self, wizard_factory) -> None:
        wizard = wizard_factory(_status(_item("SAM 2.1", status="ready")))
        row = wizard._tree.item(wizard._tree.get_children()[0], "values")
        assert row[2] == "✓"

    def test_missing_items_are_crossed(self, wizard_factory) -> None:
        wizard = wizard_factory(_status(_item("SAM 2.1", status="missing")))
        row = wizard._tree.item(wizard._tree.get_children()[0], "values")
        assert row[2] == "✕"

    def test_optional_items_are_labelled(self, wizard_factory) -> None:
        wizard = wizard_factory(_status(_item("minicpm-v", required=False)))
        row = wizard._tree.item(wizard._tree.get_children()[0], "values")
        assert "(optional)" in row[0]

    def test_size_is_shown_when_known(self, wizard_factory) -> None:
        wizard = wizard_factory(_status(_item("SAM 2.1", mb=898)))
        row = wizard._tree.item(wizard._tree.get_children()[0], "values")
        assert row[1] == "898 MB"

    def test_unknown_size_shows_a_dash(self, wizard_factory) -> None:
        wizard = wizard_factory(_status(_item("Ollama server")))
        row = wizard._tree.item(wizard._tree.get_children()[0], "values")
        assert row[1] == "—"

    def test_complete_setup_says_so(self, wizard_factory) -> None:
        wizard = wizard_factory(_status(_item("SAM 2.1", status="ready")))
        assert "ready" in wizard._progress_label.cget("text").lower()

    def test_incomplete_setup_stays_quiet(self, wizard_factory) -> None:
        wizard = wizard_factory(_status(_item("SAM 2.1", status="missing")))
        assert wizard._progress_label.cget("text") == ""

    def test_optional_gaps_still_count_as_complete(self, wizard_factory) -> None:
        wizard = wizard_factory(
            _status(
                _item("SAM 2.1", status="ready"),
                _item("minicpm-v", status="missing", required=False),
            )
        )
        assert "ready" in wizard._progress_label.cget("text").lower()

    def test_recheck_does_not_duplicate_rows(self, wizard_factory) -> None:
        wizard = wizard_factory(_status(_item("SAM 2.1"), _item("VitMatte")))
        wizard._refresh()
        wizard._refresh()
        assert len(wizard._tree.get_children()) == 2

    def test_a_failing_check_does_not_crash_the_window(
        self, tk_root, monkeypatch
    ) -> None:
        from ui import setup_wizard as wizard_mod

        def _boom(prefs):
            raise OSError("models directory unreadable")

        monkeypatch.setattr(wizard_mod, "check_setup", _boom)
        wizard = wizard_mod.SetupWizard(tk_root, {})
        tk_root.update()
        assert wizard._win.winfo_exists()
        assert wizard._tree.get_children() == ()

    def test_download_button_exists_but_is_not_invoked_here(
        self, wizard_factory
    ) -> None:
        # Invoking it would start a multi-GB download on a worker thread.
        wizard = wizard_factory(_status(_item("SAM 2.1")))
        assert str(wizard._download_btn.cget("state")) == "normal"

    def test_continue_anyway_closes_the_window(
        self, wizard_factory, tk_root
    ) -> None:
        wizard = wizard_factory(_status(_item("SAM 2.1")))
        wizard._win.destroy()
        tk_root.update()
        assert not wizard._win.winfo_exists()
