"""test_gui_batch_view.py  --  Windowed tests for the Batch wizard.

Marked `gui` automatically (via tk_root) and excluded from a plain pytest
run. Run deliberately with: pytest -m gui

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
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _write_image(path: Path) -> Path:
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(path)
    return path


@pytest.fixture
def batch_view(tk_root, tmp_path):
    """A real BatchView with all filesystem access sandboxed to tmp_path."""
    from tkinter import ttk

    from ui.batch.batch_view import BatchView

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    app = SimpleNamespace(
        root=tk_root,
        prefs={"output_directory": str(output_dir)},
        switch_to_batch=lambda *a, **k: None,
    )
    container = ttk.Frame(tk_root)
    container.pack(fill=tk.BOTH, expand=True)
    view = BatchView(container, app)
    view.frame.pack(fill=tk.BOTH, expand=True)
    tk_root.update()
    return view


@pytest.fixture
def image_folder(tmp_path):
    """A folder holding three images and two files that must be ignored."""
    folder = tmp_path / "input"
    folder.mkdir()
    for name in ("c.png", "a.jpg", "b.tiff"):
        _write_image(folder / name)
    (folder / "notes.txt").write_text("not an image")
    (folder / "sub").mkdir()
    return folder


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


class TestStepImport:
    def test_no_folder_yields_no_images(self, batch_view) -> None:
        step = batch_view._step_views[0]
        assert step.get_image_paths() == []
        assert step.input_folder is None

    def test_selecting_a_folder_records_it(
        self, batch_view, tk_root, image_folder
    ) -> None:
        step = batch_view._step_views[0]
        step._set_folder(str(image_folder))
        tk_root.update()
        assert step.input_folder == str(image_folder)

    def test_image_paths_are_filtered_and_sorted(
        self, batch_view, tk_root, image_folder
    ) -> None:
        step = batch_view._step_views[0]
        step._set_folder(str(image_folder))
        tk_root.update()
        names = [Path(p).name for p in step.get_image_paths()]
        assert names == ["a.jpg", "b.tiff", "c.png"]

    def test_non_images_and_subdirs_are_excluded(
        self, batch_view, tk_root, image_folder
    ) -> None:
        step = batch_view._step_views[0]
        step._set_folder(str(image_folder))
        tk_root.update()
        names = [Path(p).name for p in step.get_image_paths()]
        assert "notes.txt" not in names
        assert "sub" not in names

    def test_folder_label_reports_the_count(
        self, batch_view, tk_root, image_folder
    ) -> None:
        step = batch_view._step_views[0]
        step._set_folder(str(image_folder))
        tk_root.update()
        assert "3 images" in step._count_label.cget("text")

    def test_folder_without_a_guide_clears_knowledge_state(
        self, batch_view, tk_root, image_folder
    ) -> None:
        step = batch_view._step_views[0]
        step._set_folder(str(image_folder))
        tk_root.update()
        assert batch_view.knowledge_pack_path is None
        assert batch_view.knowledge_guidance_active is False

    def test_drop_of_a_non_directory_is_ignored(
        self, batch_view, tk_root, tmp_path
    ) -> None:
        step = batch_view._step_views[0]
        stray = tmp_path / "stray.png"
        _write_image(stray)
        step._on_drop(SimpleNamespace(data=str(stray)))
        tk_root.update()
        assert step.input_folder is None

    def test_legacy_state_database_is_not_presented_as_a_gui_resume_action(
        self, batch_view, tmp_path
    ) -> None:
        from core.state_manager import JobRecord, JobStatus, StateManager

        state = StateManager(tmp_path / "out" / "legacy-run" / "state.db")
        state.put(
            "unfinished",
            JobRecord(image_path="/input/unfinished.jpg", status=JobStatus.PENDING),
        )
        state.close()

        step = batch_view._step_views[0]
        step._scan_recent_batches()

        statuses = [
            step._recent_list.item(item_id, "values")[1]
            for item_id in step._recent_list.get_children()
        ]
        assert "Incomplete" not in statuses

    def test_verified_batch_resume_restores_triage_before_starting_runner(
        self, batch_view, tk_root, tmp_path, monkeypatch
    ) -> None:
        from core.batch_runner import BatchConfig
        from core.batch_session import (
            BatchInterrogationSnapshot,
            BatchProcessingSnapshot,
            BatchRunSettings,
            BatchTriageSnapshot,
            triage_labels_for_image,
            write_processing_snapshot,
            write_snapshot,
        )
        from core.state_manager import JobRecord, JobStatus, StateManager
        from ui.batch.steps.step_progress import StepProgress

        input_dir = tmp_path / "resume-input"
        input_dir.mkdir()
        complete = _write_image(input_dir / "complete.png")
        pending = _write_image(input_dir / "pending.png")
        output_dir = Path(batch_view.app.prefs["output_directory"])
        run = BatchRunSettings(
            batch_id="gui-resume",
            input_folder=str(input_dir),
            output_directory=str(output_dir),
            selection_request="Select Apple computers.",
            interrogation_settings={"output_mode": "vector+bitmap"},
        )
        candidates = {
            str(complete): [{"canonical_label": "Apple computer", "selection": "largest"}],
            str(pending): [{"canonical_label": "Apple computer", "selection": "all"}],
        }
        triage = BatchTriageSnapshot(
            batch_id=run.batch_id,
            selection_request=run.selection_request,
            approved_labels=["Apple computer"],
            excluded_labels_by_image={str(pending): ["Apple computer"]},
        )
        labels_by_image: dict[str, list[str]] = {}
        selections_by_image: dict[str, dict[str, str]] = {}
        for image_path, image_candidates in candidates.items():
            labels, selections = triage_labels_for_image(
                image_candidates,
                triage.approved_labels,
                triage.excluded_labels_by_image.get(image_path, []),
            )
            labels_by_image[image_path] = labels
            selections_by_image[image_path] = selections
        config = BatchConfig(
            batch_id=run.batch_id,
            input_folder=run.input_folder,
            output_dir=run.output_directory,
            confirmed_labels=triage.approved_labels,
            input_images=[str(complete), str(pending)],
            labels_by_image=labels_by_image,
            selections_by_image=selections_by_image,
        )
        write_snapshot(run.run_dir / "run.json", run)
        write_snapshot(
            run.run_dir / "interrogation.json",
            BatchInterrogationSnapshot(
                batch_id=run.batch_id,
                selection_request=run.selection_request,
                candidates_by_image=candidates,
            ),
        )
        write_snapshot(run.run_dir / "triage.json", triage)
        write_processing_snapshot(
            run.run_dir / "processing.json",
            BatchProcessingSnapshot(batch_id=run.batch_id, config=config.model_dump()),
        )
        all_objects = run.run_dir / "complete_all-objects.tiff"
        all_objects.write_bytes(b"durable output marker")
        state = StateManager(run.run_dir / "state.db")
        state.put(
            "complete",
            JobRecord(
                image_path=str(complete),
                status=JobStatus.COMPLETE,
                output_all_objects_tiff=str(all_objects),
            ),
        )
        state.put("pending", JobRecord(image_path=str(pending), status=JobStatus.FAILED))
        state.close()

        calls: list[str] = []
        monkeypatch.setattr(StepProgress, "resume_existing", lambda self: calls.append("resume"))
        step = batch_view._step_views[0]
        step._scan_recent_batches()
        item = next(
            item_id
            for item_id in step._recent_list.get_children()
            if step._recent_list.item(item_id, "values")[1] == "Resume ready"
        )
        step._recent_list.selection_set(item)
        step._update_resume_button()

        assert str(step._resume_btn.cget("state")) == "normal"
        step._resume_selected_run()
        tk_root.update()

        assert calls == ["resume"]
        assert batch_view.current_step == 4
        assert batch_view.run_settings.batch_id == "gui-resume"
        assert batch_view.confirmed_labels == ["Apple computer"]
        assert batch_view.excluded_labels_by_image == {str(pending): ["Apple computer"]}
        assert batch_view.interrogation_records == candidates


# ── Step 3: Interrogate ─────────────────────────────────────────────────────


class TestStepInterrogate:
    def test_tags_render_as_pills(self, batch_view, tk_root) -> None:
        batch_view.go_to_step(2)
        step = batch_view._step_views[2]
        step._add_tags([{"label": "chalice", "role": "parent"}])
        tk_root.update()
        assert step._tag_cloud.winfo_children()

    def test_tags_are_recorded_by_canonical_key(self, batch_view, tk_root) -> None:
        batch_view.go_to_step(2)
        step = batch_view._step_views[2]
        step._add_tags(
            [{"label": "a chalice", "canonical_label": "chalice", "role": "parent"}]
        )
        tk_root.update()
        assert "chalice" in step.get_all_tags()

    def test_duplicate_tags_are_not_added_twice(self, batch_view, tk_root) -> None:
        batch_view.go_to_step(2)
        step = batch_view._step_views[2]
        step._add_tags([{"label": "chalice", "role": "parent"}])
        step._add_tags([{"label": "chalice", "role": "parent"}])
        tk_root.update()
        assert len(step.get_all_tags()) == 1

    def test_blank_labels_are_skipped(self, batch_view, tk_root) -> None:
        batch_view.go_to_step(2)
        step = batch_view._step_views[2]
        step._add_tags([{"label": "", "role": "parent"}])
        tk_root.update()
        assert step.get_all_tags() == {}

    def test_get_all_tags_returns_a_copy(self, batch_view, tk_root) -> None:
        batch_view.go_to_step(2)
        step = batch_view._step_views[2]
        step._add_tags([{"label": "chalice", "role": "parent"}])
        tk_root.update()
        got = step.get_all_tags()
        got["injected"] = {}
        assert "injected" not in step.get_all_tags()


# ── Step 4: Triage ──────────────────────────────────────────────────────────


class TestStepTriage:
    @staticmethod
    def _tags() -> dict[str, dict]:
        return {
            "chalice": {"label": "chalice", "role": "parent"},
            "paten": {"label": "paten", "role": "parent"},
            "stem": {"label": "stem", "role": "child", "parent": "chalice"},
        }

    def test_populate_builds_a_card_per_parent(self, batch_view, tk_root) -> None:
        batch_view.go_to_step(3)
        step = batch_view._step_views[3]
        step.populate(self._tags())
        tk_root.update()
        assert len(step._cards_frame.winfo_children()) == 2

    def test_all_parents_start_included(self, batch_view, tk_root) -> None:
        batch_view.go_to_step(3)
        step = batch_view._step_views[3]
        step.populate(self._tags())
        tk_root.update()
        assert sorted(step.get_confirmed_labels()) == ["chalice", "paten"]

    def test_excluding_a_parent_drops_it(self, batch_view, tk_root) -> None:
        batch_view.go_to_step(3)
        step = batch_view._step_views[3]
        step.populate(self._tags())
        tk_root.update()
        step._include_vars["chalice"].set(False)
        step._on_toggle("chalice")
        assert step.get_confirmed_labels() == ["paten"]

    def test_canonical_label_is_confirmed_when_present(
        self, batch_view, tk_root
    ) -> None:
        batch_view.go_to_step(3)
        step = batch_view._step_views[3]
        step.populate(
            {
                "cup": {
                    "label": "cup",
                    "canonical_label": "chalice",
                    "role": "parent",
                }
            }
        )
        tk_root.update()
        assert step.get_confirmed_labels() == ["chalice"]

    def test_repopulating_replaces_previous_cards(
        self, batch_view, tk_root
    ) -> None:
        batch_view.go_to_step(3)
        step = batch_view._step_views[3]
        step.populate(self._tags())
        step.populate({"urn": {"label": "urn", "role": "parent"}})
        tk_root.update()
        assert step.get_confirmed_labels() == ["urn"]
        assert len(step._cards_frame.winfo_children()) == 1

    def test_navigating_to_triage_pulls_tags_from_interrogate(
        self, batch_view, tk_root
    ) -> None:
        batch_view.go_to_step(2)
        batch_view._step_views[2]._add_tags(
            [{"label": "chalice", "role": "parent"}]
        )
        tk_root.update()
        batch_view.go_to_step(3)
        tk_root.update()
        assert batch_view._step_views[3].get_confirmed_labels() == ["chalice"]

    def test_confirming_stores_labels_and_advances(
        self, batch_view, tk_root
    ) -> None:
        batch_view.go_to_step(3)
        step = batch_view._step_views[3]
        step.populate(self._tags())
        tk_root.update()
        step._on_confirm()
        tk_root.update()
        assert sorted(batch_view.confirmed_labels) == ["chalice", "paten"]
        assert batch_view.current_step == 4

    def test_per_image_exception_is_stored_with_the_triage_decision(
        self, batch_view, tk_root
    ) -> None:
        from core.batch_session import BatchRunSettings, load_triage_snapshot

        batch_view.go_to_step(3)
        step = batch_view._step_views[3]
        batch_view.run_settings = BatchRunSettings(
            batch_id="triage-exception",
            output_directory=batch_view.app.prefs["output_directory"],
            selection_request="Select Apple computers; exclude scenery.",
        )
        batch_view.interrogation_records = {
            "/input/rubble.jpg": [{"canonical_label": "Apple computer", "role": "parent"}],
            "/input/mac.jpg": [{"canonical_label": "Apple computer", "role": "parent"}],
        }
        step.populate(
            {"Apple computer": {"label": "Apple computer", "role": "parent"}}
        )
        tk_root.update()

        step._image_exclude_vars[("/input/rubble.jpg", "Apple computer")].set(True)
        step._on_confirm()

        assert batch_view.confirmed_labels == ["Apple computer"]
        assert batch_view.excluded_labels_by_image == {
            "/input/rubble.jpg": ["Apple computer"]
        }
        snapshot = load_triage_snapshot(batch_view.run_settings.run_dir / "triage.json")
        assert snapshot.excluded_labels_by_image == {
            "/input/rubble.jpg": ["Apple computer"]
        }

    def test_confirming_nothing_does_not_advance(
        self, batch_view, tk_root
    ) -> None:
        batch_view.go_to_step(3)
        step = batch_view._step_views[3]
        step.populate({})
        tk_root.update()
        step._on_confirm()
        tk_root.update()
        assert batch_view.current_step == 3

    def test_triage_shows_request_and_domain_guide_context(
        self, batch_view, tk_root
    ) -> None:
        batch_view.selection_request = "Select Apple products; exclude captions."
        batch_view.run_settings = SimpleNamespace(
            selection_request=batch_view.selection_request,
            guide_name="Apple — The First 50 Years",
        )
        batch_view.go_to_step(3)
        step = batch_view._step_views[3]
        step.populate(self._tags())
        tk_root.update()

        assert "Select Apple products" in step._request_label.cget("text")
        assert "Apple — The First 50 Years" in step._guide_label.cget("text")


# ── Step 5: Progress ────────────────────────────────────────────────────────


class TestStepProgress:
    def test_thumbnails_are_created_per_image(self, batch_view, tk_root) -> None:
        batch_view.go_to_step(4)
        step = batch_view._step_views[4]
        step.init_thumbnails(["a", "b", "c"])
        tk_root.update()
        assert len(step._thumb_labels) == 3

    def test_source_thumbnail_is_shown_when_image_path_is_available(
        self, batch_view, tk_root, tmp_path
    ) -> None:
        image_path = _write_image(tmp_path / "source.png")
        batch_view.go_to_step(4)
        step = batch_view._step_views[4]

        step.init_thumbnails([str(image_path)])
        tk_root.update()

        assert "source" in step._thumb_labels
        assert step._thumb_labels["source"].cget("image")
        assert "source" in step._thumb_refs

    def test_missing_source_image_keeps_the_status_cell_fallback(
        self, batch_view, tk_root, tmp_path
    ) -> None:
        batch_view.go_to_step(4)
        step = batch_view._step_views[4]

        step.init_thumbnails([str(tmp_path / "missing.png")])
        tk_root.update()

        label = step._thumb_labels[str(tmp_path / "missing.png")]
        assert label.cget("text") == "—"
        assert not label.cget("image")

    def test_thumbnail_status_updates(self, batch_view, tk_root) -> None:
        batch_view.go_to_step(4)
        step = batch_view._step_views[4]
        step.init_thumbnails(["a"])
        before = step._thumb_labels["a"].cget("text")
        step.update_thumbnail_status("a", "complete")
        tk_root.update()
        assert step._thumb_labels["a"].cget("text") != before

    def test_batch_runner_phases_render_as_running_thumbnail_status(
        self, batch_view, tk_root
    ) -> None:
        batch_view.go_to_step(4)
        step = batch_view._step_views[4]
        step.init_thumbnails(["a"])

        step.update_thumbnail_status("a", "masking")
        tk_root.update()

        assert step._thumb_labels["a"].cget("text") == "·"

    def test_progress_counts_terminal_failures_in_the_percentage(
        self, batch_view, tk_root
    ) -> None:
        batch_view.go_to_step(4)
        step = batch_view._step_views[4]

        step.update_progress(
            SimpleNamespace(
                total=4,
                completed=1,
                failed=2,
                remaining=1,
                images_per_min=3.0,
                eta_seconds=20.0,
            )
        )
        tk_root.update()

        assert step._progress_bar["value"] == 3
        assert step._progress_label.cget("text").startswith("75%")

    def test_status_for_an_unknown_image_is_ignored(
        self, batch_view, tk_root
    ) -> None:
        batch_view.go_to_step(4)
        step = batch_view._step_views[4]
        step.init_thumbnails(["a"])
        step.update_thumbnail_status("nonexistent", "complete")
        tk_root.update()
        assert len(step._thumb_labels) == 1

    def test_reinitialising_replaces_the_previous_set(
        self, batch_view, tk_root
    ) -> None:
        batch_view.go_to_step(4)
        step = batch_view._step_views[4]
        step.init_thumbnails(["a", "b"])
        step.init_thumbnails(["c"])
        tk_root.update()
        assert len(step._thumb_labels) == 1


# ── Step 6: Output ──────────────────────────────────────────────────────────


class TestStepOutput:
    def test_summary_updates_the_cards(self, batch_view, tk_root) -> None:
        batch_view.go_to_step(5)
        step = batch_view._step_views[5]
        step.update_summary(
            svg_count=12, avg_layers=3.5, failed_count=2, all_objects_count=11
        )
        tk_root.update()
        assert step._svg_card._value_label.cget("text") == "12"
        assert step._all_objects_card._value_label.cget("text") == "11"
        assert step._layers_card._value_label.cget("text") == "3.5"

    def test_summary_is_stored_on_the_view(self, batch_view, tk_root) -> None:
        batch_view.go_to_step(5)
        step = batch_view._step_views[5]
        step.update_summary(svg_count=12, avg_layers=3.5, failed_count=2)
        assert batch_view.output_summary["svg_count"] == 12

    def test_retry_is_enabled_when_something_failed(
        self, batch_view, tk_root
    ) -> None:
        batch_view.go_to_step(5)
        step = batch_view._step_views[5]
        step.update_summary(svg_count=1, avg_layers=1.0, failed_count=3)
        tk_root.update()
        assert str(step._retry_btn.cget("state")) == "normal"
        assert "3" in step._retry_btn.cget("text")

    def test_retry_is_disabled_with_no_failures(
        self, batch_view, tk_root
    ) -> None:
        batch_view.go_to_step(5)
        step = batch_view._step_views[5]
        step.update_summary(svg_count=5, avg_layers=2.0, failed_count=0)
        tk_root.update()
        assert str(step._retry_btn.cget("state")) == "disabled"

    def test_retry_failed_restarts_only_the_failed_images(
        self, batch_view, monkeypatch
    ) -> None:
        batch_view.failed_image_paths = ["/input/failed-one.jpg", "/input/failed-two.jpg"]
        batch_view.go_to_step(5)
        step = batch_view._step_views[5]
        calls: list[list[str]] = []
        batch_view.go_to_step(4)
        progress = batch_view._step_views[4]
        monkeypatch.setattr(progress, "start_retry", lambda paths: calls.append(paths))
        batch_view.go_to_step(5)

        step._retry_failed()

        assert calls == [["/input/failed-one.jpg", "/input/failed-two.jpg"]]
        assert batch_view.current_step == 4

    def test_saved_summary_is_rehydrated_on_revisit(
        self, batch_view, tk_root
    ) -> None:
        batch_view.go_to_step(5)
        batch_view._step_views[5].update_summary(
            svg_count=9, avg_layers=4.0, failed_count=1
        )
        batch_view.go_to_step(0)
        batch_view.go_to_step(5)
        tk_root.update()
        assert batch_view._step_views[5]._svg_card._value_label.cget("text") == "9"

    def test_tiff_bundle_copies_all_objects_and_layer_exports(
        self, batch_view, tmp_path, monkeypatch
    ) -> None:
        batch_view.go_to_step(5)
        step = batch_view._step_views[5]
        source = tmp_path / "out"
        all_objects = source / "sample_all-objects.tiff"
        layer = source / "sample_object-001.tiff"
        all_objects.write_bytes(b"foreground")
        layer.write_bytes(b"layer")
        destination = tmp_path / "exported"
        destination.mkdir()
        monkeypatch.setattr("tkinter.filedialog.askdirectory", lambda **_: str(destination))
        monkeypatch.setattr("tkinter.messagebox.showinfo", lambda *_, **__: None)

        step._export_tiff_bundle()

        assert (destination / all_objects.name).read_bytes() == b"foreground"
        assert (destination / layer.name).read_bytes() == b"layer"

    def test_svg_bundle_copies_only_svg_files(self, batch_view, tmp_path, monkeypatch) -> None:
        batch_view.go_to_step(5)
        step = batch_view._step_views[5]
        source = tmp_path / "out"
        svg = source / "sample.svg"
        svg.write_text("<svg/>", encoding="utf-8")
        (source / "sample.tiff").write_bytes(b"tiff")
        destination = tmp_path / "exported"
        destination.mkdir()
        monkeypatch.setattr("tkinter.filedialog.askdirectory", lambda **_: str(destination))
        monkeypatch.setattr("tkinter.messagebox.showinfo", lambda *_, **__: None)

        step._export_svg_bundle()

        assert (destination / svg.name).read_text(encoding="utf-8") == "<svg/>"
        assert not (destination / "sample.tiff").exists()


# ── Step 2: Configure ───────────────────────────────────────────────────────


class TestStepConfigure:
    def test_config_exposes_the_expected_keys(self, batch_view, tk_root) -> None:
        batch_view.go_to_step(1)
        config = batch_view._step_views[1].get_config()
        assert {
            "output_mode",
            "recursion_depth",
            "vtracer_quality",
            "interrogation_profile",
            "fallback_mode",
        } <= set(config)

    def test_config_is_published_to_the_view(self, batch_view, tk_root) -> None:
        batch_view.go_to_step(1)
        config = batch_view._step_views[1].get_config()
        assert batch_view.interrogation_settings == config

    def test_multiline_selection_request_is_committed_to_config(
        self, batch_view, tk_root
    ) -> None:
        batch_view.go_to_step(1)
        step = batch_view._step_views[1]
        step._set_selection_request("Select products.\nExclude captions.")

        assert step.get_config()["selection_request"] == "Select products.\nExclude captions."

    def test_request_edit_clears_existing_interrogation_and_triage(
        self, batch_view, tk_root
    ) -> None:
        batch_view.go_to_step(1)
        step = batch_view._step_views[1]
        batch_view.run_settings = SimpleNamespace(selection_request="Original request")
        batch_view.selection_request = "Original request"
        batch_view.interrogation_records = {"/input/a.jpg": [{"canonical_label": "iMac"}]}
        batch_view.confirmed_labels = ["iMac"]
        batch_view.excluded_labels_by_image = {"/input/a.jpg": ["iMac"]}
        step._set_selection_request("Changed request")
        tk_root.update()

        assert batch_view.interrogation_stale is True
        assert batch_view.interrogation_records == {}
        assert batch_view.confirmed_labels == []
        assert batch_view.excluded_labels_by_image == {}

    def test_template_and_previous_run_restore_selection_request(
        self, batch_view, tk_root
    ) -> None:
        from core.batch_session import BatchRunSettings
        from core.batch_template import BatchTemplate

        template = BatchTemplate(
            name="Apple", source_image="", confirmed_labels=[], confirmed_children={},
            output_mode="vector", recursion_depth=2, corner_threshold=60,
            speckle=8, smoothing=5, length_threshold=4.0, vtracer_quality="balanced",
            selection_request="Template request",
        )
        batch_view.load_template(template)
        batch_view.go_to_step(1)
        tk_root.update()
        step = batch_view._step_views[1]
        assert step.get_selection_request() == "Template request"
        assert "remain editable" in step._template_status_label.cget("text")

        batch_view.load_run_settings(
            BatchRunSettings(
                input_folder="", output_directory="/tmp/out",
                selection_request="Restored run request",
                guide_name="Apple — The First 50 Years",
                interrogation_settings={
                    "output_mode": "vector",
                    "recursion_depth": 3,
                    "vtracer_quality": "maximum",
                },
            )
        )
        batch_view.go_to_step(1)
        tk_root.update()
        restored = batch_view._step_views[1]
        assert restored.get_selection_request() == "Restored run request"
        assert restored.get_config()["output_mode"] == "vector"
        assert restored.get_config()["recursion_depth"] == 3
        assert restored.get_config()["vtracer_quality"] == "maximum"

    def test_output_mode_falls_back_to_vector(self, batch_view, tk_root) -> None:
        batch_view.go_to_step(1)
        step = batch_view._step_views[1]
        for var in step._mode_vars.values():
            var.set(False)
        assert step.get_config()["output_mode"] == "vector"

    def test_output_mode_joins_selected_options(self, batch_view, tk_root) -> None:
        batch_view.go_to_step(1)
        step = batch_view._step_views[1]
        for var in step._mode_vars.values():
            var.set(True)
        assert "+" in step.get_config()["output_mode"]

    def test_guide_path_is_omitted_when_guidance_is_off(
        self, batch_view, tk_root
    ) -> None:
        batch_view.go_to_step(1)
        step = batch_view._step_views[1]
        batch_view.knowledge_pack_path = "/tmp/guide.toml"
        step._guide_mode_var.set(False)
        assert step.get_config()["guide_path"] is None

    def test_guide_path_is_used_when_guidance_is_on(
        self, batch_view, tk_root
    ) -> None:
        batch_view.go_to_step(1)
        step = batch_view._step_views[1]
        batch_view.knowledge_pack_path = "/tmp/guide.toml"
        step._guide_mode_var.set(True)
        assert step.get_config()["guide_path"] == "/tmp/guide.toml"
