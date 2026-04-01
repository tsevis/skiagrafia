from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

from ui.theme import TAG_COLOURS, is_macos

if TYPE_CHECKING:
    from ui.batch.batch_view import BatchView


class StepTriage:
    """Step 4 — Triage (human gate): confirm labels before GPU pipeline."""

    def __init__(self, parent: tk.Widget, view: BatchView) -> None:
        self._view = view
        self._app = view.app

        self.frame = ttk.Frame(parent, padding=16)

        # Amber warning banner
        banner = tk.Frame(self.frame, bg="#FFF3CD", padx=10, pady=8)
        banner.pack(fill=tk.X, pady=(0, 12))
        tk.Label(
            banner,
            text="\u26a0  Confirm labels before the GPU pipeline begins.",
            bg="#FFF3CD",
            fg="#7A4F00",
            font=("SF Pro Text", 11) if is_macos() else ("Segoe UI", 10),
        ).pack(anchor=tk.W)

        ttk.Label(
            self.frame,
            text="Review Labels",
            font=("SF Pro Display", 16, "bold"),
        ).pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(
            self.frame,
            text="Toggle labels to include or skip in the processing pipeline.",
            foreground="gray",
        ).pack(anchor=tk.W, pady=(0, 12))

        # Cards container (scrollable)
        canvas = tk.Canvas(self.frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=canvas.yview)
        self._cards_frame = ttk.Frame(canvas)

        self._cards_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self._cards_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._include_vars: dict[str, tk.BooleanVar] = {}
        self._tag_data: dict[str, dict] = {}
        self._card_frames: dict[str, ttk.Frame] = {}

        # Auto-populate from interrogation results
        self._load_from_interrogation()

        # Confirm button (bottom)
        self._confirm_btn = ttk.Button(
            self.frame,
            text="Confirm Labels & Continue",
            command=self._on_confirm,
        )
        self._confirm_btn.pack(anchor=tk.E, pady=(12, 0))

    def _load_from_interrogation(self) -> None:
        """Pull tags from the Interrogate step and populate cards."""
        step_interrogate = self._view._step_views[2]
        if step_interrogate and hasattr(step_interrogate, "get_all_tags"):
            tags = step_interrogate.get_all_tags()
            if tags:
                self.populate(tags)

    def _on_confirm(self) -> None:
        """Confirm selected labels and advance to Progress step."""
        confirmed = self.get_confirmed_labels()
        if not confirmed:
            self._confirm_btn.config(text="Select at least one label")
            self._view.root.after(
                2000,
                lambda: self._confirm_btn.config(text="Confirm Labels & Continue"),
            )
            return
        # Store confirmed labels on the view for the Progress step
        self._view.confirmed_labels = confirmed
        self._view.go_next()

    def populate(self, tags: dict[str, dict]) -> None:
        """Populate triage cards from interrogation results."""
        for w in self._cards_frame.winfo_children():
            w.destroy()
        self._include_vars.clear()
        self._tag_data = dict(tags)
        self._card_frames.clear()

        parents = {k: v for k, v in tags.items() if v.get("role") == "parent"}
        children = {k: v for k, v in tags.items() if v.get("role") == "child"}

        for parent_label, parent_data in parents.items():
            display_label = parent_data.get("label", parent_label)
            card = ttk.LabelFrame(
                self._cards_frame, text=display_label, padding=8
            )
            card.pack(fill=tk.X, pady=4, padx=4)
            self._card_frames[parent_label] = card

            # Include/skip toggle
            include_var = tk.BooleanVar(value=True)
            self._include_vars[parent_label] = include_var

            header = ttk.Frame(card)
            header.pack(fill=tk.X)

            ttk.Checkbutton(
                header,
                text="Include",
                variable=include_var,
                command=lambda pl=parent_label: self._on_toggle(pl),
            ).pack(side=tk.LEFT)

            # Child tag pills
            child_frame = ttk.Frame(card)
            child_frame.pack(fill=tk.X, pady=(4, 0))

            parent_children = {
                k: v
                for k, v in children.items()
                if v.get("parent") in (parent_label, display_label)
            }
            for child_label in parent_children:
                colours = TAG_COLOURS["child"]
                tk.Label(
                    child_frame,
                    text=parent_children[child_label].get("label", child_label),
                    bg=colours["bg"],
                    fg=colours["fg"],
                    highlightbackground=colours["border"],
                    highlightthickness=1,
                    padx=6,
                    pady=2,
                ).pack(side=tk.LEFT, padx=2)

    def _on_toggle(self, parent_label: str) -> None:
        """Handle include/skip toggle for a parent card."""
        included = self._include_vars[parent_label].get()
        card = self._card_frames.get(parent_label)
        if card:
            # Dim the card when skipped — unfortunately ttk frames
            # don't support opacity, so we just collapse children
            for child in card.winfo_children():
                if isinstance(child, ttk.Frame) and child != card.winfo_children()[0]:
                    if included:
                        child.pack(fill=tk.X, pady=(4, 0))
                    else:
                        child.pack_forget()

    def get_confirmed_labels(self) -> list[str]:
        """Return labels that are included (not skipped)."""
        return [
            self._tag_data.get(label, {}).get("canonical_label", label)
            for label, var in self._include_vars.items()
            if var.get()
        ]
