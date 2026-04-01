from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

from ui.theme import is_macos

if TYPE_CHECKING:
    from ui.batch.batch_view import BatchView

# Duplicated from BatchView.STEP_TITLES to avoid circular import at runtime.
_STEP_TITLES = [
    "Import",
    "Configure",
    "Interrogate",
    "Triage",
    "Progress",
    "Output",
]


class BatchSidebar:
    """Numbered step list sidebar (210px) for batch wizard."""

    WIDTH = 210

    def __init__(self, parent: tk.Widget, view: BatchView) -> None:
        self._view = view
        self.frame = ttk.Frame(parent, width=self.WIDTH)
        self.frame.pack_propagate(False)

        self._step_frames: list[tk.Frame] = []
        self._badge_labels: list[tk.Label] = []
        self._title_labels: list[tk.Label] = []
        self._active_index = 0
        self._completed_indices: set[int] = set()

        self._build()

    def _build(self) -> None:
        """Build the step list."""
        ttk.Label(
            self.frame,
            text="Steps",
            font=("SF Pro Text", 11, "bold") if is_macos() else ("Segoe UI", 10, "bold"),
        ).pack(padx=12, pady=(12, 8), anchor=tk.W)

        for i, title in enumerate(_STEP_TITLES):
            row = tk.Frame(self.frame, bg="white")
            row.pack(fill=tk.X, padx=6, pady=1)
            row.bind("<Button-1>", lambda e, idx=i: self._on_click(idx))

            # Number badge
            badge_text = "!" if i == 3 else str(i + 1)
            badge_bg = "#FF9F0A" if i == 3 else "#E5E5EA"
            badge_fg = "white" if i == 3 else "#1D1D1F"

            badge = tk.Label(
                row,
                text=badge_text,
                bg=badge_bg,
                fg=badge_fg,
                width=3,
                height=1,
                font=("SF Pro Text", 9, "bold") if is_macos() else ("Segoe UI", 8, "bold"),
            )
            badge.pack(side=tk.LEFT, padx=(4, 8), pady=4)
            badge.bind("<Button-1>", lambda e, idx=i: self._on_click(idx))
            self._badge_labels.append(badge)

            # Title
            title_label = tk.Label(
                row,
                text=title,
                bg="white",
                fg="#1D1D1F",
                anchor=tk.W,
                font=("SF Pro Text", 12) if is_macos() else ("Segoe UI", 10),
            )
            title_label.pack(side=tk.LEFT, fill=tk.X)
            title_label.bind("<Button-1>", lambda e, idx=i: self._on_click(idx))
            self._title_labels.append(title_label)

            self._step_frames.append(row)

        # Separator
        ttk.Separator(self.frame, orient=tk.VERTICAL).pack(
            side=tk.RIGHT, fill=tk.Y, padx=0
        )
        self.set_active_step(0)

    def set_active_step(self, index: int) -> None:
        """Highlight the active step."""
        self._active_index = index
        for i, frame in enumerate(self._step_frames):
            active = i == index
            completed = i in self._completed_indices
            frame_bg = "#EAF2FF" if active else "white"
            frame.configure(bg=frame_bg)
            self._title_labels[i].configure(
                bg=frame_bg,
                fg="#0A4FB3" if active else "#1D1D1F",
                font=(
                    ("SF Pro Text", 12, "bold") if is_macos() else ("Segoe UI", 10, "bold")
                ) if active else (
                    ("SF Pro Text", 12) if is_macos() else ("Segoe UI", 10)
                ),
            )
            if completed and i != 3:
                self._badge_labels[i].config(text="\u2713", bg="#34C759", fg="white")
            elif i == 3:
                self._badge_labels[i].config(
                    text="!" if not completed else "\u2713",
                    bg="#FF9F0A" if not completed else "#34C759",
                    fg="white",
                )
            else:
                self._badge_labels[i].config(
                    text=str(i + 1),
                    bg="#007AFF" if active else "#E5E5EA",
                    fg="white" if active else "#1D1D1F",
                )

    def mark_completed(self, index: int) -> None:
        """Show green checkmark on completed step badge."""
        if 0 <= index < len(self._badge_labels):
            self._completed_indices.add(index)
            self.set_active_step(self._active_index)

    def _on_click(self, index: int) -> None:
        """Handle click on a step."""
        self._view.go_to_step(index)
