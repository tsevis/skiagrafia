"""test_gui_batch_steps.py  --  windowed tests for the six wizard steps.

Import, Interrogate, Triage, Progress, Output and Configure. Marked
`gui_integration` automatically (via tk_root) and excluded from a plain
pytest run.

NOTHING HERE STARTS A BATCH. The two methods that spawn work --
StepInterrogate._on_start and StepProgress._on_start -- launch daemon
threads that load models, so they are never called; every test drives the
data-facing methods those threads eventually feed (_add_tags, populate,
init_thumbnails, update_summary) directly instead.

Nothing here opens a native dialog either. _browse_folder, _load_guide,
_reveal_in_finder and the export buttons all block on a modal chooser, so
folder selection goes through _set_folder.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conftest import _write_image


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
            output_mode="vector", corner_threshold=60,
            speckle=8, length_threshold=4.0, vtracer_quality="balanced",
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
                    "vtracer_quality": "maximum",
                },
            )
        )
        batch_view.go_to_step(1)
        tk_root.update()
        restored = batch_view._step_views[1]
        assert restored.get_selection_request() == "Restored run request"
        assert restored.get_config()["output_mode"] == "vector"
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
