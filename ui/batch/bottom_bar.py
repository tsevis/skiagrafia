from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ui.batch.batch_view import BatchView


class BottomBar:
    """Bottom bar: status dot, status text, progress bar, Back/Next buttons."""

    def __init__(self, parent: tk.Widget, view: BatchView) -> None:
        self._view = view

        # Separator above
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, side=tk.BOTTOM)

        self.frame = ttk.Frame(parent, padding=(8, 4))

        # Status dot
        self._status_dot = tk.Label(
            self.frame,
            text="\u25CF",
            fg="#34C759",
            font=("", 8),
        )
        self._status_dot.pack(side=tk.LEFT, padx=(0, 4))

        # Status text
        self._status_text = ttk.Label(self.frame, text="Ready")
        self._status_text.pack(side=tk.LEFT, padx=(0, 8))

        # Progress bar
        self._progress = ttk.Progressbar(
            self.frame, mode="determinate", length=140
        )
        self._progress.pack(side=tk.LEFT, padx=(0, 8))

        # Next button (right-aligned)
        self._next_btn = ttk.Button(
            self.frame,
            text="Next \u2192",
            command=self._view.go_next,
        )
        self._next_btn.pack(side=tk.RIGHT, padx=(4, 0))

        # Back button
        self._back_btn = ttk.Button(
            self.frame,
            text="Back",
            command=self._view.go_back,
        )
        self._back_btn.pack(side=tk.RIGHT)

    def update_for_step(self, step: int) -> None:
        """Update button visibility based on current step."""
        # Hide Back on steps 0, 4, 5
        if step in (0, 4, 5):
            self._back_btn.pack_forget()
        else:
            self._back_btn.pack(side=tk.RIGHT)

        # Hide Next on step 4 (auto-advance when complete)
        if step == 4:
            self._next_btn.pack_forget()
        else:
            self._next_btn.pack(side=tk.RIGHT, padx=(4, 0))

    def set_status(self, text: str, colour: str = "#34C759") -> None:
        """Update status dot and text."""
        self._status_dot.config(fg=colour)
        self._status_text.config(text=text)

    def set_progress(self, value: float, maximum: float = 100.0) -> None:
        """Update progress bar."""
        self._progress["maximum"] = maximum
        self._progress["value"] = value
