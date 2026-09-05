"""left_panel_labels.py  --  Label discovery and editing for the left panel.

The scan/confirm/edit half of LeftPanel: running interrogation on a worker
thread, polling its queue, and the label-pill UI built from the result.
Split out of left_panel.py to keep each file within a readable size; mixed
into LeftPanel, so `self` is the panel and behaviour is unchanged.
"""
from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from tkinter import ttk

from ui.single.scan_dedup import dedup_scan_detections
from ui.theme import is_macos, TAG_COLOURS

logger = logging.getLogger(__name__)


class LabelsSectionMixin:
    """Label scanning, rendering and editing. Requires LeftPanel's attributes."""

    def _build_labels_section(self) -> None:
        section = ttk.LabelFrame(self._inner, text="Labels", padding=6)
        section.pack(fill=tk.X, padx=6, pady=3)

        self._labels_hint = ttk.Label(
            section,
            text="Scan to detect objects",
            foreground="gray",
        )
        self._labels_hint.pack(anchor=tk.W)

        self._labels_container = ttk.Frame(section)

        # Scan button
        self._scan_btn = ttk.Button(
            section,
            text="Scan with Moondream",
            command=self._scan_labels,
        )
        self._scan_btn.pack(fill=tk.X, pady=(4, 0))

        # Progress bar for scanning
        self._scan_progress = ttk.Progressbar(
            section, mode="indeterminate", length=200
        )
        # Status label shown between labels-ready and boxes-ready
        self._scan_status = ttk.Label(section, text="", foreground="gray")

        # Box opacity slider (shown once scan preview boxes are available)
        self._box_opacity_frame = ttk.Frame(section)
        ttk.Label(self._box_opacity_frame, text="Box opacity").pack(
            side=tk.LEFT, padx=(0, 6)
        )
        self._box_opacity_var = tk.IntVar(
            value=int(self._app.prefs.get("scan_preview_box_opacity", 40))
        )
        ttk.Scale(
            self._box_opacity_frame,
            variable=self._box_opacity_var,
            from_=0,
            to=100,
            command=self._on_box_opacity_changed,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Add label button
        add_btn = ttk.Button(
            section, text="+ Add label", command=self._add_label_dialog
        )
        add_btn.pack(fill=tk.X, pady=(2, 0))

    def _scan_labels(self) -> None:
        """Run Moondream interrogation in background thread."""
        if not self._image_path:
            return

        self._scan_btn.config(state="disabled")
        self._scan_progress.pack(fill=tk.X, pady=(2, 0))
        self._scan_progress.start(20)
        self._box_opacity_frame.pack_forget()
        self._scan_status.pack_forget()

        def _worker() -> None:
            try:
                from core.factory import build_interrogation_settings
                from core.interrogation import GuidedInterrogator
                from core.knowledge import KnowledgePack
                import cv2

                image = cv2.imread(self._image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                knowledge_pack = (
                    KnowledgePack.load(self._knowledge_pack_path)
                    if self._knowledge_pack_path
                    else None
                )
                interrogator = GuidedInterrogator(
                    build_interrogation_settings(
                        self._app.prefs,
                        kp_defaults=self._knowledge_pack_defaults,
                    )
                )
                detected = interrogator.interrogate(image, knowledge_pack=knowledge_pack)
                label_dicts = [
                    {
                        "label": candidate.display_label,
                        "canonical_label": candidate.canonical_label,
                        "role": candidate.role,
                        "confidence": candidate.confidence,
                        "source_model": candidate.source_model,
                    }
                    for candidate in detected.candidates
                ]
                self._progress_queue.put(("labels", label_dicts))
                preview_detections: list[dict] = []
                try:
                    from models.grounded_sam import GroundedSAM

                    detector = GroundedSAM()
                    parent_candidates = [
                        candidate for candidate in detected.candidates if candidate.role == "parent"
                    ][:4]
                    for candidate in parent_candidates:
                        detection = None
                        for phrase in (candidate.detector_phrases or [candidate.display_label])[:2]:
                            detection = detector.detect_box(image, phrase, skip_synonyms=True)
                            if detection is not None:
                                break
                        if detection is None:
                            continue
                        preview_detections.append(
                            {
                                "label": candidate.display_label,
                                "role": candidate.role,
                                "bbox": detection.bbox,
                                "confidence": candidate.confidence,
                            }
                        )
                    detector.clear_cache()
                    preview_detections = dedup_scan_detections(preview_detections)
                except Exception:
                    logger.info("Scan preview detections unavailable", exc_info=True)
                self._progress_queue.put(("scan_preview", preview_detections))
            except Exception as exc:
                logger.error("Moondream scan failed: %s", exc, exc_info=True)
                self._progress_queue.put(("error", str(exc)))

        threading.Thread(target=_worker, daemon=True).start()
        self._poll_scan_queue()

    def _poll_scan_queue(self) -> None:
        """Poll the scan queue from the main thread."""
        try:
            msg_type, data = self._progress_queue.get_nowait()
            if msg_type == "labels":
                # Show labels immediately but keep progress bar spinning
                # until box detections are ready
                self._labels = data
                self._render_label_pills()
                self._view.on_labels_updated(data)
                self._scan_status.config(text="Detecting boxes…")
                self._scan_status.pack(fill=tk.X, pady=(2, 0))
                self._root.after(50, self._poll_scan_queue)
            elif msg_type == "scan_preview":
                # Boxes are ready — now stop progress and show results
                self._scan_progress.stop()
                self._scan_progress.pack_forget()
                self._scan_status.pack_forget()
                self._scan_btn.config(state="normal", text="Re-scan")
                # Remove labels that were dropped by scan dedup
                kept_labels = {d.get("label", "").lower() for d in data}
                if kept_labels and self._labels:
                    before = len(self._labels)
                    self._labels = [
                        lbl for lbl in self._labels
                        if lbl.get("label", "").lower() in kept_labels
                        or lbl.get("role") != "parent"
                    ]
                    if len(self._labels) < before:
                        self._render_label_pills()
                        self._view.on_labels_updated(self._labels)
                self._view.on_scan_preview_ready(data)
                if data:
                    self._box_opacity_frame.pack(fill=tk.X, pady=(4, 0))
                # Don't poll further — scan is fully complete
            elif msg_type == "error":
                self._scan_progress.stop()
                self._scan_progress.pack_forget()
                self._scan_status.pack_forget()
                self._scan_btn.config(state="normal", text="Re-scan")
                self._labels_hint.config(text=f"Scan failed: {data}", foreground="red")
        except queue.Empty:
            self._root.after(100, self._poll_scan_queue)

    def _on_box_opacity_changed(self, value: str) -> None:
        """Update box opacity preference and refresh canvas preview."""
        opacity = int(float(value))
        self._app.prefs["scan_preview_box_opacity"] = opacity
        self._view.canvas_panel.refresh_scan_preview()

    def _render_label_pills(self) -> None:
        """Render coloured pill buttons for detected labels."""
        for w in self._labels_container.winfo_children():
            w.destroy()

        if not self._labels:
            self._labels_container.pack_forget()
            self._labels_hint.config(text="Scan to detect objects", foreground="gray")
            self._labels_hint.pack(anchor=tk.W)
            return

        self._labels_hint.pack_forget()
        self._labels_container.pack(fill=tk.X, pady=(4, 0))

        for label_data in self._labels:
            role = label_data.get("role", "parent")
            colours = TAG_COLOURS.get(role, TAG_COLOURS["parent"])

            pill = tk.Label(
                self._labels_container,
                text=label_data.get("label", ""),
                bg=colours["bg"],
                fg=colours["fg"],
                highlightbackground=colours["border"],
                highlightthickness=1,
                padx=6,
                pady=2,
                cursor="hand2",
            )
            pill.pack(anchor=tk.W, pady=1)
            pill.bind(
                "<Button-1>",
                lambda e, lbl=label_data: self._toggle_label(lbl, e.widget),
            )
            # Option+click (Alt+click) opens edit/delete context menu
            pill.bind(
                "<Option-Button-1>" if is_macos() else "<Alt-Button-1>",
                lambda e, lbl=label_data: self._show_label_context_menu(e, lbl),
            )

    def _toggle_label(self, label_data: dict, widget: tk.Label) -> None:
        """Toggle a label on/off."""
        off_colours = TAG_COLOURS["off"]
        current_bg = str(widget.cget("bg"))
        if current_bg == off_colours["bg"]:
            role = label_data.get("role", "parent")
            colours = TAG_COLOURS.get(role, TAG_COLOURS["parent"])
            widget.config(bg=colours["bg"], fg=colours["fg"])
        else:
            widget.config(bg=off_colours["bg"], fg=off_colours["fg"])

    def _show_label_context_menu(self, event: tk.Event, label_data: dict) -> None:
        """Show a context menu with Edit and Delete options for a label pill."""
        menu = tk.Menu(self._root, tearoff=0)
        menu.add_command(
            label="Edit name",
            command=lambda: self._edit_label(label_data),
        )
        menu.add_command(
            label="Delete",
            command=lambda: self._delete_label(label_data),
        )
        menu.tk_popup(event.x_root, event.y_root)

    def _edit_label(self, label_data: dict) -> None:
        """Open a dialog to rename a label."""
        dialog = tk.Toplevel(self._root)
        dialog.title("Edit Label")
        dialog.geometry("280x120")
        dialog.transient(self._root)
        dialog.grab_set()

        ttk.Label(dialog, text="New name:").pack(padx=10, pady=(10, 2), anchor=tk.W)
        entry_var = tk.StringVar(value=label_data.get("label", ""))
        entry = ttk.Entry(dialog, textvariable=entry_var, width=30)
        entry.pack(padx=10, pady=2)
        entry.focus_set()
        entry.select_range(0, tk.END)

        def _apply() -> None:
            new_name = entry_var.get().strip()
            if not new_name:
                dialog.destroy()
                return
            old_label = label_data.get("label", "")
            try:
                self._view.rename_scan_detection(old_label, new_name)
            except Exception:
                logger.exception("Failed to rename detection in canvas")
            label_data["label"] = new_name
            label_data["canonical_label"] = new_name
            self._render_label_pills()
            dialog.destroy()

        entry.bind("<Return>", lambda e: _apply())
        ttk.Button(dialog, text="Save", command=_apply).pack(pady=8)

    def _delete_label(self, label_data: dict) -> None:
        """Remove a label from the list and its scan preview detection."""
        label = label_data.get("label", "")
        self._labels = [lbl for lbl in self._labels if lbl is not label_data]
        self._view.remove_scan_detection(label)
        self._render_label_pills()

    def _add_label_dialog(self) -> None:
        """Open dialog to manually add a label."""
        dialog = tk.Toplevel(self._root)
        dialog.title("Add Label")
        dialog.geometry("280x120")
        dialog.transient(self._root)
        dialog.grab_set()

        ttk.Label(dialog, text="Label name:").pack(padx=10, pady=(10, 2), anchor=tk.W)
        entry = ttk.Entry(dialog, width=30)
        entry.pack(padx=10, pady=2)
        entry.focus_set()

        def _add() -> None:
            name = entry.get().strip()
            if name:
                self._labels.append({"label": name, "role": "parent"})
                self._render_label_pills()
            dialog.destroy()

        entry.bind("<Return>", lambda e: _add())
        ttk.Button(dialog, text="Add", command=_add).pack(pady=8)

    def add_manual_label(self, label: str) -> None:
        """Add a user-defined label if it is not already present."""
        cleaned = label.strip()
        if not cleaned:
            return
        existing = {
            item.get("canonical_label", item.get("label", "")).lower()
            for item in self._labels
        }
        if cleaned.lower() in existing:
            return
        self._labels.append(
            {
                "label": cleaned,
                "canonical_label": cleaned,
                "role": "parent",
                "confidence": 1.0,
                "source_model": "manual-box",
            }
        )
        self._render_label_pills()

    # ── Parameters section ─────────────────────────────────────
