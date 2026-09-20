from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk
from typing import Any


class ModeSwitcher(ttk.Frame):
    """Custom segmented control for switching between Single and Batch modes.

    Two ttk.Buttons styled as a segmented control — active button gets
    a white background card appearance.
    """

    def __init__(
        self,
        parent: tk.Widget,
        on_mode_change: Callable[[str], None] | None = None,
        # Any: ttk.Frame's constructor accepts a heterogeneous mix of widget
        # options (str, int, bool, callables, ...); no single concrete type
        # covers a passthrough **kwargs without reproducing that whole stub.
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, **kwargs)
        self._mode = "single"
        self._on_mode_change = on_mode_change

        # Container with subtle border
        self._container = ttk.Frame(self)
        self._container.pack()

        self._single_btn = ttk.Button(
            self._container,
            text="Single Image",
            command=lambda: self.set_mode("single"),
            width=14,
        )
        self._single_btn.pack(side=tk.LEFT, padx=(0, 1))

        self._batch_btn = ttk.Button(
            self._container,
            text="Batch",
            command=lambda: self.set_mode("batch"),
            width=10,
        )
        self._batch_btn.pack(side=tk.LEFT)

        self._update_appearance()

    def set_mode(self, mode: str) -> None:
        """Switch to the given mode ('single' or 'batch')."""
        if mode == self._mode:
            return
        self._mode = mode
        self._update_appearance()
        if self._on_mode_change:
            self._on_mode_change(mode)

    @property
    def mode(self) -> str:
        return self._mode

    def _update_appearance(self) -> None:
        """Update button styles to reflect active mode."""
        if self._mode == "single":
            self._single_btn.state(["pressed"])
            self._batch_btn.state(["!pressed"])
        else:
            self._single_btn.state(["!pressed"])
            self._batch_btn.state(["pressed"])
