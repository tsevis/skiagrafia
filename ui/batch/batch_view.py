from __future__ import annotations

import tkinter as tk
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
