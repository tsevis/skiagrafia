"""test_gui_batch_view.py  --  windowed tests for the Batch wizard shell.

Construction, wizard navigation and the bottom bar. The six step screens
have their own module, test_gui_batch_steps.py.

Marked `gui_integration` automatically (via tk_root) and excluded from a
plain pytest run. A small explicit `gui` smoke set runs with `pytest -m gui`;
run every windowed test with `pytest -m "gui or gui_integration"`.

Nothing here starts a batch. The two methods that spawn work --
StepInterrogate._on_start and StepProgress._on_start -- launch daemon
threads that load models, so they are never called; every test drives the
data-facing methods those threads eventually feed (_add_tags, populate,
init_thumbnails, update_summary) directly instead.

Nothing here opens a native dialog either. _browse_folder, _load_guide,
_reveal_in_finder and the export buttons all block on a modal chooser, so
folder selection goes through _set_folder.

All filesystem access stays in tmp_path: the app stub points
`output_directory` at a temporary directory so StepImport's recent-batch
scan cannot reach ~/Desktop/skiagrafia_out.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conftest import _write_image

# ── Construction ────────────────────────────────────────────────────────────


class TestBatchViewConstruction:
    def test_starts_on_the_import_step(self, batch_view) -> None:
        assert batch_view.current_step == 0

    def test_sidebar_and_bottom_bar_exist(self, batch_view) -> None:
        assert batch_view._sidebar.frame.winfo_exists()
        assert batch_view._bottom_bar.frame.winfo_exists()

    def test_sidebar_lists_every_step(self, batch_view) -> None:
        assert len(batch_view._sidebar._step_frames) == len(
            batch_view.STEP_TITLES
        )

    def test_only_the_first_step_is_built_eagerly(self, batch_view) -> None:
        built = [v is not None for v in batch_view._step_views]
        assert built == [True, False, False, False, False, False]

    def test_visiting_a_step_builds_it_once(self, batch_view, tk_root) -> None:
        batch_view.go_to_step(1)
        tk_root.update()
        first = batch_view._step_views[1]
        batch_view.go_to_step(0)
        batch_view.go_to_step(1)
        tk_root.update()
        assert batch_view._step_views[1] is first

    def test_every_step_constructs(self, batch_view, tk_root) -> None:
        # Walking the whole wizard is the cheapest way to prove no step
        # raises on construction.
        for index in range(len(batch_view.STEP_TITLES)):
            batch_view.go_to_step(index)
            tk_root.update()
        assert all(v is not None for v in batch_view._step_views)

    def test_approved_labels_are_intersected_with_each_images_candidates(
        self, batch_view
    ) -> None:
        batch_view.confirmed_labels = ["iMac", "iPhone"]
        batch_view.interrogation_records = {
            "/input/one.jpg": [
                {"canonical_label": "iMac", "selection": "largest"},
                {"canonical_label": "Apple logo", "selection": "all"},
            ],
            "/input/two.jpg": [
                {"canonical_label": "iPhone", "selection": "all"},
            ],
        }

        labels_one, selections_one = batch_view.labels_for_image("/input/one.jpg")
        labels_two, selections_two = batch_view.labels_for_image("/input/two.jpg")

        assert labels_one == ["iMac"]
        assert selections_one == {"iMac": "largest"}
        assert labels_two == ["iPhone"]
        assert selections_two == {"iPhone": "all"}

    def test_image_specific_triage_exclusion_preserves_the_label_elsewhere(
        self, batch_view
    ) -> None:
        batch_view.confirmed_labels = ["Apple computer"]
        batch_view.excluded_labels_by_image = {
            "/input/rubble.jpg": ["Apple computer"]
        }
        batch_view.interrogation_records = {
            "/input/rubble.jpg": [{"canonical_label": "Apple computer", "selection": "all"}],
            "/input/mac.jpg": [{"canonical_label": "Apple computer", "selection": "largest"}],
        }

        skipped_labels, skipped_selections = batch_view.labels_for_image("/input/rubble.jpg")
        kept_labels, kept_selections = batch_view.labels_for_image("/input/mac.jpg")

        assert skipped_labels == []
        assert skipped_selections == {}
        assert kept_labels == ["Apple computer"]
        assert kept_selections == {"Apple computer": "largest"}

    def test_freezing_processing_manifest_preserves_per_image_triage(
        self, batch_view, tmp_path
    ) -> None:
        from core.batch_session import BatchRunSettings, load_processing_snapshot

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        kept = _write_image(input_dir / "kept.png")
        skipped = _write_image(input_dir / "skipped.png")
        batch_view.run_settings = BatchRunSettings(
            batch_id="frozen-triage",
            input_folder=str(input_dir),
            output_directory=batch_view.app.prefs["output_directory"],
            selection_request="Select Apple computers.",
            interrogation_settings={
                "output_mode": "vector+bitmap",
                "interrogation_profile": "balanced",
            },
        )
        batch_view.interrogation_records = {
            str(kept): [{"canonical_label": "Apple computer", "selection": "largest"}],
            str(skipped): [{"canonical_label": "Apple computer", "selection": "all"}],
        }
        batch_view.confirmed_labels = ["Apple computer"]
        batch_view.excluded_labels_by_image = {str(skipped): ["Apple computer"]}

        config = batch_view.freeze_processing_config([str(kept), str(skipped)])
        snapshot = load_processing_snapshot(
            batch_view.run_settings.run_dir / "processing.json"
        )

        assert config.input_images == [str(kept), str(skipped)]
        assert config.labels_by_image == {
            str(kept): ["Apple computer"],
            str(skipped): [],
        }
        assert config.selections_by_image == {
            str(kept): {"Apple computer": "largest"},
            str(skipped): {},
        }
        assert snapshot.config == config.model_dump()


# ── Navigation ──────────────────────────────────────────────────────────────


class TestWizardNavigation:
    def test_next_advances(self, batch_view, tk_root) -> None:
        batch_view.go_next()
        tk_root.update()
        assert batch_view.current_step == 1

    def test_back_returns(self, batch_view, tk_root) -> None:
        batch_view.go_to_step(2)
        batch_view.go_back()
        tk_root.update()
        assert batch_view.current_step == 1

    def test_back_is_clamped_at_the_first_step(self, batch_view, tk_root) -> None:
        batch_view.go_back()
        tk_root.update()
        assert batch_view.current_step == 0

    def test_next_is_clamped_at_the_last_step(self, batch_view, tk_root) -> None:
        batch_view.go_to_step(5)
        batch_view.go_next()
        tk_root.update()
        assert batch_view.current_step == 5

    def test_out_of_range_jump_is_ignored(self, batch_view, tk_root) -> None:
        batch_view.go_to_step(99)
        batch_view.go_to_step(-3)
        tk_root.update()
        assert batch_view.current_step == 0

    def test_advancing_marks_the_step_completed(self, batch_view, tk_root) -> None:
        batch_view.go_next()
        tk_root.update()
        assert 0 in batch_view._completed_steps

    def test_jumping_does_not_mark_completion(self, batch_view, tk_root) -> None:
        batch_view.go_to_step(3)
        tk_root.update()
        assert batch_view._completed_steps == set()

    def test_sidebar_click_navigates(self, batch_view, tk_root) -> None:
        batch_view._sidebar._on_click(2)
        tk_root.update()
        assert batch_view.current_step == 2

    def test_sidebar_tracks_the_active_step(self, batch_view, tk_root) -> None:
        batch_view.go_to_step(4)
        tk_root.update()
        assert batch_view._sidebar._active_index == 4

    def test_mark_completed_is_recorded(self, batch_view, tk_root) -> None:
        batch_view._sidebar.mark_completed(1)
        tk_root.update()
        assert 1 in batch_view._sidebar._completed_indices


# ── Bottom bar ──────────────────────────────────────────────────────────────


class TestBottomBar:
    @pytest.mark.parametrize("step", [0, 4, 5])
    def test_back_is_hidden_where_it_makes_no_sense(
        self, batch_view, tk_root, step: int
    ) -> None:
        batch_view.go_to_step(step)
        tk_root.update()
        assert not batch_view._bottom_bar._back_btn.winfo_ismapped()

    @pytest.mark.parametrize("step", [1, 2, 3])
    def test_back_is_shown_mid_wizard(
        self, batch_view, tk_root, step: int
    ) -> None:
        batch_view.go_to_step(step)
        tk_root.update()
        assert batch_view._bottom_bar._back_btn.winfo_ismapped()

    def test_next_is_hidden_while_processing(self, batch_view, tk_root) -> None:
        # Step 4 auto-advances when the batch finishes.
        batch_view.go_to_step(4)
        tk_root.update()
        assert not batch_view._bottom_bar._next_btn.winfo_ismapped()

    def test_status_text_tracks_the_step(self, batch_view, tk_root) -> None:
        batch_view.go_to_step(4)
        tk_root.update()
        assert "Processing" in batch_view._bottom_bar._status_text.cget("text")

    def test_progress_resets_off_the_progress_step(
        self, batch_view, tk_root
    ) -> None:
        batch_view._bottom_bar.set_progress(50, 100)
        batch_view.go_to_step(1)
        tk_root.update()
        assert batch_view._bottom_bar._progress["value"] == 0

    def test_set_progress_updates_the_bar(self, batch_view) -> None:
        batch_view._bottom_bar.set_progress(7, 21)
        assert batch_view._bottom_bar._progress["value"] == 7
        assert batch_view._bottom_bar._progress["maximum"] == 21


# ── Step 1: Import ──────────────────────────────────────────────────────────


