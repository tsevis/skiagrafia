"""The destination folder, shown where a run is set up.

One row, used by the single-image left panel and by the batch Import step,
so the place you choose the input is the place you choose where its output
goes. Both write through `ui.destination`, which is also what the
Preferences window sets, so the three never disagree.
"""
from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import filedialog, ttk
from typing import Any

from ui.destination import DestinationError, choose_destination, current_destination


class DestinationRow:
    """A labelled destination with a Choose button and an inline refusal."""

    def __init__(
        self,
        parent: tk.Widget,
        prefs: dict[str, Any],
        on_change: Callable[[], None] | None = None,
        wraplength: int = 260,
    ) -> None:
        self._prefs = prefs
        self._on_change = on_change

        self.frame = ttk.Frame(parent)

        self._path_label = ttk.Label(
            self.frame, text="", foreground="gray", wraplength=wraplength, justify=tk.LEFT
        )
        self._path_label.pack(anchor=tk.W, fill=tk.X)

        ttk.Button(
            self.frame, text="Choose Destination Folder…", command=self._browse
        ).pack(anchor=tk.W, fill=tk.X, pady=(4, 0))

        # Packed only when there is something to say, so an untouched panel
        # carries no empty warning row.
        self._refusal_label = ttk.Label(
            self.frame, text="", foreground="#B25000", wraplength=wraplength, justify=tk.LEFT
        )

        self.refresh()

    def refresh(self) -> None:
        """Show the destination the next run will use."""
        self._path_label.config(text=str(current_destination(self._prefs)))

    def _browse(self) -> None:
        path = filedialog.askdirectory(
            parent=self.frame.winfo_toplevel(),
            title="Choose where finished work is written",
            initialdir=str(current_destination(self._prefs).parent),
        )
        if not path:
            return
        try:
            choose_destination(self._prefs, path)
        except DestinationError as exc:
            self._show_refusal(str(exc))
            return
        self._clear_refusal()
        self.refresh()
        if self._on_change is not None:
            self._on_change()

    def _show_refusal(self, message: str) -> None:
        self._refusal_label.config(text=message)
        self._refusal_label.pack(anchor=tk.W, fill=tk.X, pady=(4, 0))

    def _clear_refusal(self) -> None:
        self._refusal_label.pack_forget()
