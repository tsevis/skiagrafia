from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ui.main_window import MainWindow


class BatchView:
    """Six-step wizard layout manager for Batch mode.

    Left sidebar (210px) + main content area + bottom bar.
    """

    STEP_TITLES = [
        "Import",
        "Configure",
        "Interrogate",
        "Triage",
        "Progress",
        "Output",
    ]

    def __init__(self, parent: tk.Widget, app: MainWindow) -> None:
        self.app = app
        self.root = app.root
        self.frame = ttk.Frame(parent)

        self._current_step = 0
        self._completed_steps: set[int] = set()
        self._template: object | None = None
        self.confirmed_labels: list[str] = []
        self.selection_request: str = ""
        self.interrogation_records: dict[str, list[dict]] = {}
        self.interrogation_stale = False
        self.run_settings = None
        self.knowledge_pack_path: str | None = None
        self.knowledge_pack_name: str | None = None
        self.knowledge_pack_notes_path: str | None = None
        self.knowledge_pack_defaults: dict[str, object] = {}
        self.knowledge_guidance_active = False
        self.interrogation_settings: dict[str, object] = {}
        self.output_summary = {
            "svg_count": 0,
            "avg_layers": 0.0,
            "failed_count": 0,
        }

        # Bottom bar must pack BEFORE body so it gets allocated space
        from ui.batch.bottom_bar import BottomBar

        self._bottom_bar = BottomBar(self.frame, self)
        self._bottom_bar.frame.pack(fill=tk.X, side=tk.BOTTOM)

        # Layout: sidebar | content
        self._body = ttk.Frame(self.frame)
        self._body.pack(fill=tk.BOTH, expand=True)

        # Sidebar
        from ui.batch.sidebar import BatchSidebar

        self._sidebar = BatchSidebar(self._body, self)
        self._sidebar.frame.pack(side=tk.LEFT, fill=tk.Y)

        # Content area
        self._content = ttk.Frame(self._body)
        self._content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Build step views (lazy)
        self._step_views: list[object | None] = [None] * 6

        # Show step 1
        self._show_step(0)

    def _show_step(self, index: int) -> None:
        """Show the given step view."""
        # Clear content
        for child in self._content.winfo_children():
            child.pack_forget()

        self._current_step = index
        self._sidebar.set_active_step(index)
        self._bottom_bar.update_for_step(index)
        self._sync_bottom_status(index)

        # Lazy-load step view
        if self._step_views[index] is None:
            self._step_views[index] = self._create_step_view(index)

        view = self._step_views[index]
        if view and hasattr(view, "frame"):
            view.frame.pack(fill=tk.BOTH, expand=True)

        # Refresh data when navigating to Triage (step 3)
        if index == 3 and hasattr(view, "_load_from_interrogation"):
            view._load_from_interrogation()

    def _sync_bottom_status(self, index: int) -> None:
        status_map = {
            0: ("Ready", "#34C759"),
            1: ("Configure batch", "#34C759"),
            2: ("Ready to analyse", "#007AFF"),
            3: ("Review labels in Triage", "#007AFF"),
            4: ("Processing batch...", "#FF9F0A"),
            5: ("Output ready", "#34C759"),
        }
        text, colour = status_map.get(index, ("Ready", "#34C759"))
        self._bottom_bar.set_status(text, colour)
        if index != 4:
            self._bottom_bar.set_progress(0, 1)

    def _create_step_view(self, index: int) -> object:
        """Create a step view by index."""
        from ui.batch.steps.step_import import StepImport
        from ui.batch.steps.step_configure import StepConfigure
        from ui.batch.steps.step_interrogate import StepInterrogate
        from ui.batch.steps.step_triage import StepTriage
        from ui.batch.steps.step_progress import StepProgress
        from ui.batch.steps.step_output import StepOutput

        step_classes = [
            StepImport,
            StepConfigure,
            StepInterrogate,
            StepTriage,
            StepProgress,
            StepOutput,
        ]
        return step_classes[index](self._content, self)

    def go_next(self) -> None:
        """Advance to the next step."""
        if self._current_step < 5:
            self._completed_steps.add(self._current_step)
            self._sidebar.mark_completed(self._current_step)
            self._show_step(self._current_step + 1)

    def go_back(self) -> None:
        """Go to the previous step."""
        if self._current_step > 0:
            self._show_step(self._current_step - 1)

    def go_to_step(self, index: int) -> None:
        """Navigate directly to a step."""
        if 0 <= index <= 5:
            self._show_step(index)

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def template(self) -> object | None:
        return self._template

    @template.setter
    def template(self, value: object) -> None:
        self._template = value

    def load_template(self, template: object) -> None:
        """Start a new batch from a saved template without bypassing Triage."""
        self._template = template
        self.selection_request = str(getattr(template, "selection_request", "") or "")
        self.confirmed_labels = []
        self.interrogation_records = {}
        self.interrogation_stale = False
        self.run_settings = None
        self.interrogation_settings = {}
        guide_path = getattr(template, "guide_path", None)
        if guide_path and Path(guide_path).is_file():
            from core.knowledge import KnowledgePack

            pack = KnowledgePack.load(guide_path)
            self.knowledge_pack_path = str(guide_path)
            self.knowledge_pack_name = pack.name
            self.knowledge_pack_notes_path = (
                str(Path(guide_path).with_suffix(".md"))
                if Path(guide_path).with_suffix(".md").exists()
                else None
            )
            self.knowledge_pack_defaults = pack.batch_defaults.model_dump()
            self.knowledge_guidance_active = True
        elif guide_path:
            self.knowledge_pack_path = None
            self.knowledge_pack_name = getattr(template, "guide_name", None) or Path(guide_path).stem
            self.knowledge_pack_notes_path = None
            self.knowledge_pack_defaults = {}
            self.knowledge_guidance_active = False

        # Rebuild views that render template/session values.
        for index in (0, 1, 2, 3):
            existing = self._step_views[index]
            if existing and hasattr(existing, "frame"):
                existing.frame.destroy()
            self._step_views[index] = None
        self._show_step(0)

    def begin_run(self, config: dict) -> object:
        """Freeze the configuration that produces an interrogation result."""
        from core.batch_session import BatchRunSettings, capture_guide, write_snapshot

        step_import = self._step_views[0]
        input_folder = getattr(step_import, "input_folder", None) if step_import else None
        output_directory = str(
            self.app.prefs.get(
                "output_directory", str(Path.home() / "Desktop" / "skiagrafia_out")
            )
        )
        guide_path = config.get("guide_path")
        guide_name, guide_fingerprint, guide_toml = capture_guide(guide_path)
        self.run_settings = BatchRunSettings(
            input_folder=input_folder or "",
            output_directory=output_directory,
            selection_request=str(config.get("selection_request", "")),
            guide_path=guide_path,
            guide_name=self.knowledge_pack_name or guide_name,
            guide_fingerprint=guide_fingerprint,
            guide_toml=guide_toml,
            interrogation_settings=dict(config),
            output_settings={
                key: config.get(key)
                for key in ("output_mode", "recursion_depth", "vtracer_quality")
            },
        )
        self.interrogation_stale = False
        write_snapshot(self.run_settings.run_dir / "run.json", self.run_settings)
        return self.run_settings

    def invalidate_interrogation(self) -> None:
        """Invalidate candidates and Triage approval after request edits."""
        self.interrogation_stale = True
        self.interrogation_records = {}
        self.confirmed_labels = []
        step_interrogate = self._step_views[2]
        if step_interrogate and hasattr(step_interrogate, "_clear_results"):
            step_interrogate._clear_results()
        step_triage = self._step_views[3]
        if step_triage and hasattr(step_triage, "populate"):
            step_triage.populate({})

    def store_interrogation_records(self, records: dict[str, list[dict]]) -> None:
        self.interrogation_records = {path: list(items) for path, items in records.items()}
        self.interrogation_stale = False
        if self.run_settings is None:
            return
        from core.batch_session import BatchInterrogationSnapshot, write_snapshot

        write_snapshot(
            self.run_settings.run_dir / "interrogation.json",
            BatchInterrogationSnapshot(
                batch_id=self.run_settings.batch_id,
                selection_request=self.run_settings.selection_request,
                candidates_by_image=self.interrogation_records,
            ),
        )

    def store_triage_decision(self, approved_labels: list[str]) -> None:
        self.confirmed_labels = list(approved_labels)
        if self.run_settings is None:
            return
        from core.batch_session import BatchTriageSnapshot, write_snapshot

        write_snapshot(
            self.run_settings.run_dir / "triage.json",
            BatchTriageSnapshot(
                batch_id=self.run_settings.batch_id,
                selection_request=self.run_settings.selection_request,
                approved_labels=approved_labels,
            ),
        )

    def labels_for_image(self, image_path: str) -> tuple[list[str], dict[str, str]]:
        """Intersect human-approved labels with candidates from this image only."""
        approved = set(self.confirmed_labels)
        labels: list[str] = []
        selections: dict[str, str] = {}
        for item in self.interrogation_records.get(image_path, []):
            canonical = str(item.get("canonical_label") or item.get("label") or "")
            if canonical and canonical in approved and canonical not in labels:
                labels.append(canonical)
                selections[canonical] = str(item.get("selection", "all"))
        return labels, selections

    def load_run_settings(self, settings: object) -> None:
        """Restore a historical request and guide context into a new editable batch."""
        self._template = None
        self.selection_request = str(getattr(settings, "selection_request", "") or "")
        self.confirmed_labels = []
        self.interrogation_records = {}
        self.interrogation_stale = True
        # Restoring a run starts a new, editable batch; it must never append
        # outputs to the historical run directory.
        self.run_settings = None
        self.interrogation_settings = dict(
            getattr(settings, "interrogation_settings", {}) or {}
        )
        guide_path = getattr(settings, "guide_path", None)
        self.knowledge_pack_name = getattr(settings, "guide_name", None)
        if guide_path and Path(guide_path).is_file():
            from core.knowledge import KnowledgePack

            pack = KnowledgePack.load(guide_path)
            self.knowledge_pack_path = str(guide_path)
            self.knowledge_pack_name = pack.name
            self.knowledge_pack_defaults = pack.batch_defaults.model_dump()
            self.knowledge_guidance_active = True
        else:
            self.knowledge_pack_path = None
            self.knowledge_pack_defaults = {}
            self.knowledge_guidance_active = False
        for index in (1, 2, 3):
            existing = self._step_views[index]
            if existing and hasattr(existing, "frame"):
                existing.frame.destroy()
            self._step_views[index] = None
