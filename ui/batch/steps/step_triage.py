from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import TYPE_CHECKING, cast

from ui.theme import TAG_COLOURS, is_macos

if TYPE_CHECKING:
    from ui.batch.batch_view import BatchView
    from ui.batch.steps.step_interrogate import StepInterrogate


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

        # This context is deliberately visible at the human approval gate:
        # a label is meaningful only in relation to the request and guide
        # that created it.
        context = ttk.LabelFrame(self.frame, text="Analysis context", padding=8)
        context.pack(fill=tk.X, pady=(0, 12))
        self._request_label = ttk.Label(
            context,
            text="Selection request: —",
            justify=tk.LEFT,
            wraplength=540,
        )
        self._request_label.pack(anchor=tk.W)
        self._guide_label = ttk.Label(context, text="Domain Guide: —", foreground="gray")
        self._guide_label.pack(anchor=tk.W, pady=(4, 0))

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
        self._image_exclude_vars: dict[tuple[str, str], tk.BooleanVar] = {}
        self._tag_data: dict[str, dict] = {}
        self._card_frames: dict[str, ttk.LabelFrame] = {}
        self._detail_frames: dict[str, ttk.Frame] = {}

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
            tags = cast("StepInterrogate", step_interrogate).get_all_tags()
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
        # Persist the human gate alongside the immutable interrogation record.
        self._view.store_triage_decision(
            confirmed,
            self.get_excluded_labels_by_image(confirmed),
        )
        self._view.go_next()

    def populate(self, tags: dict[str, dict]) -> None:
        """Populate triage cards from interrogation results."""
        for w in self._cards_frame.winfo_children():
            w.destroy()
        self._include_vars.clear()
        self._image_exclude_vars.clear()
        self._tag_data = dict(tags)
        self._card_frames.clear()
        self._detail_frames.clear()

        run = self._view.run_settings
        request = (
            getattr(run, "selection_request", "")
            or self._view.selection_request
            or "No selection request was supplied."
        )
        guide_name = (
            getattr(run, "guide_name", None)
            or self._view.knowledge_pack_name
            or "No Domain Guide"
        )
        self._request_label.config(text=f"Selection request: {request}")
        self._guide_label.config(text=f"Domain Guide: {guide_name}")

        parents = {k: v for k, v in tags.items() if v.get("role") == "parent"}
        children = {k: v for k, v in tags.items() if v.get("role") == "child"}
        suggested = {
            str(value).casefold()
            for value in getattr(self._view.template, "confirmed_labels", [])
        }
        prior_approval = {
            str(value).casefold()
            for value in getattr(self._view, "confirmed_labels", [])
        }
        prior_exclusions = getattr(self._view, "excluded_labels_by_image", {})

        for parent_label, parent_data in parents.items():
            display_label = parent_data.get("label", parent_label)
            card = ttk.LabelFrame(
                self._cards_frame, text=display_label, padding=8
            )
            card.pack(fill=tk.X, pady=4, padx=4)
            self._card_frames[parent_label] = card

            # Include/skip toggle
            canonical = str(parent_data.get("canonical_label", parent_label)).strip()
            include_var = tk.BooleanVar(
                value=(
                    canonical.casefold() in prior_approval
                    if prior_approval
                    else not suggested or canonical.casefold() in suggested
                )
            )
            self._include_vars[parent_label] = include_var

            header = ttk.Frame(card)
            header.pack(fill=tk.X)

            ttk.Checkbutton(
                header,
                text="Include",
                variable=include_var,
                command=lambda pl=parent_label: self._on_toggle(pl),
            ).pack(side=tk.LEFT)

            image_count = int(parent_data.get("image_count", 0))
            ttk.Label(
                header,
                text=f"Seen in {image_count} image{'s' if image_count != 1 else ''}",
                foreground="gray",
            ).pack(side=tk.LEFT, padx=(10, 0))

            detail_frame = ttk.Frame(card)
            detail_frame.pack(fill=tk.X, pady=(4, 0))
            self._detail_frames[parent_label] = detail_frame

            # Child tag pills
            child_frame = ttk.Frame(detail_frame)
            child_frame.pack(fill=tk.X)

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

            # Global approval handles batch-wide vocabulary.  These controls
            # make it possible to reject a false positive in one image while
            # retaining that same label for the inputs where it is correct.
            image_paths = self._candidate_image_paths(canonical)
            if image_paths:
                ttk.Label(
                    detail_frame,
                    text="Skip this label in individual images:",
                    foreground="gray",
                ).pack(anchor=tk.W, pady=(6, 1))
                image_frame = ttk.Frame(detail_frame)
                image_frame.pack(fill=tk.X)
                for image_path in image_paths:
                    exclude_var = tk.BooleanVar(
                        value=canonical.casefold()
                        in {
                            str(label).casefold()
                            for label in prior_exclusions.get(image_path, [])
                        }
                    )
                    self._image_exclude_vars[(image_path, canonical)] = exclude_var
                    ttk.Checkbutton(
                        image_frame,
                        text=f"Skip · {Path(image_path).name}",
                        variable=exclude_var,
                    ).pack(anchor=tk.W)

    def _on_toggle(self, parent_label: str) -> None:
        """Handle include/skip toggle for a parent card."""
        included = self._include_vars[parent_label].get()
        card = self._card_frames.get(parent_label)
        detail_frame = self._detail_frames.get(parent_label)
        if card and detail_frame:
            # ttk frames do not support opacity, so collapsing the details
            # makes a batch-wide skip immediately clear without losing a
            # previously chosen per-image exception.
            if included:
                detail_frame.pack(fill=tk.X, pady=(4, 0))
            else:
                detail_frame.pack_forget()

    def get_confirmed_labels(self) -> list[str]:
        """Return labels that are included (not skipped)."""
        return [
            self._tag_data.get(label, {}).get("canonical_label", label)
            for label, var in self._include_vars.items()
            if var.get()
        ]

    def get_excluded_labels_by_image(
        self, confirmed_labels: list[str] | None = None
    ) -> dict[str, list[str]]:
        """Return only scoped exclusions that still have a global approval."""
        approved = {
            str(label).casefold()
            for label in (
                confirmed_labels if confirmed_labels is not None else self.get_confirmed_labels()
            )
        }
        exclusions: dict[str, list[str]] = {}
        for (image_path, canonical), var in self._image_exclude_vars.items():
            if var.get() and canonical.casefold() in approved:
                exclusions.setdefault(image_path, []).append(canonical)
        return {
            image_path: sorted(set(labels))
            for image_path, labels in exclusions.items()
        }

    def _candidate_image_paths(self, canonical: str) -> list[str]:
        """List source images that proposed this parent candidate."""
        paths: list[str] = []
        for image_path, candidates in self._view.interrogation_records.items():
            if any(
                str(item.get("canonical_label") or item.get("label") or "").casefold()
                == canonical.casefold()
                and item.get("role", "parent") == "parent"
                for item in candidates
            ):
                paths.append(image_path)
        return sorted(paths)
