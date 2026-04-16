from __future__ import annotations

import logging
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import TYPE_CHECKING

from PIL import Image

from core.knowledge import load_knowledge_pack
from ui.theme import is_macos, TAG_COLOURS

if TYPE_CHECKING:
    from ui.single.single_view import SingleView

logger = logging.getLogger(__name__)

# ── Scan-preview bbox dedup ──────────────────────────────────────────

_SCAN_BBOX_IOU_THRESHOLD = 0.50
_SCAN_CONTAINMENT_THRESHOLD = 0.70


def _bbox_iou_scan(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    """IoU of two (x0, y0, x1, y1) bounding boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union else 0.0


def _bbox_containment_scan(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    """Fraction of the smaller bbox's area that overlaps the larger one."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    smaller = min(area_a, area_b)
    return float(inter / smaller) if smaller else 0.0


def _dedup_scan_detections(detections: list[dict]) -> list[dict]:
    """Remove duplicate scan detections whose bboxes overlap heavily.

    When two detections cover the same physical object (e.g. 'guitar'
    and 'urn'), keep only the one with the higher confidence.
    """
    if len(detections) <= 1:
        return detections

    # Sort by confidence descending so higher-confidence detections win
    ranked = sorted(detections, key=lambda d: d.get("confidence", 0), reverse=True)
    kept: list[dict] = []
    for det in ranked:
        bbox = det.get("bbox")
        if bbox is None:
            kept.append(det)
            continue
        is_dup = False
        for existing in kept:
            ex_bbox = existing.get("bbox")
            if ex_bbox is None:
                continue
            iou = _bbox_iou_scan(bbox, ex_bbox)
            containment = _bbox_containment_scan(bbox, ex_bbox)
            if iou > _SCAN_BBOX_IOU_THRESHOLD or containment > _SCAN_CONTAINMENT_THRESHOLD:
                logger.info(
                    "Scan dedup: dropping '%s' (iou=%.2f, contain=%.2f with '%s')",
                    det.get("label"), iou, containment, existing.get("label"),
                )
                is_dup = True
                break
        if not is_dup:
            kept.append(det)
    return kept


class LeftPanel:
    """Left panel: drop zone, labels, parameters, process button.

    Width: 308px fixed. Entire panel is scrollable via mousewheel.
    """

    PANEL_WIDTH = 324

    def __init__(self, parent: tk.Widget, view: SingleView) -> None:
        self._view = view
        self._app = view.app
        self._root = view.root

        self.frame = ttk.Frame(parent, width=self.PANEL_WIDTH)
        self.frame.pack_propagate(False)

        self._image_path: str | None = None
        self._knowledge_pack_path: str | None = None
        self._knowledge_pack_name: str | None = None
        self._knowledge_pack_defaults: dict[str, object] = {}
        self._labels: list[dict] = []
        self._progress_queue: queue.Queue = queue.Queue()
        self._parameter_entry_vars: dict[str, tk.StringVar] = {}

        # ── Scrollable wrapper ─────────────────────────────────
        self._scroll_canvas = tk.Canvas(
            self.frame, highlightthickness=0, width=self.PANEL_WIDTH
        )
        self._scrollbar = ttk.Scrollbar(
            self.frame, orient=tk.VERTICAL, command=self._scroll_canvas.yview
        )
        self._inner = ttk.Frame(self._scroll_canvas)

        self._inner.bind(
            "<Configure>",
            lambda e: self._scroll_canvas.configure(
                scrollregion=self._scroll_canvas.bbox("all")
            ),
        )
        self._canvas_window = self._scroll_canvas.create_window(
            (0, 0), window=self._inner, anchor=tk.NW, width=self.PANEL_WIDTH - 14
        )

        self._scroll_canvas.configure(yscrollcommand=self._scrollbar.set)
        self._scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Mousewheel binding
        self._scroll_canvas.bind("<Enter>", self._bind_mousewheel)
        self._scroll_canvas.bind("<Leave>", self._unbind_mousewheel)

        self._build_image_section()
        self._build_labels_section()
        self._build_parameters_section()
        self._build_process_section()

    # ── Mousewheel scrolling ───────────────────────────────────

    def _bind_mousewheel(self, event: object = None) -> None:
        self._scroll_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, event: object = None) -> None:
        self._scroll_canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event: tk.Event) -> None:
        self._scroll_canvas.yview_scroll(-1 * (event.delta // 120 or event.delta), "units")

    # ── Image section ──────────────────────────────────────────

    def _build_image_section(self) -> None:
        section = ttk.LabelFrame(self._inner, text="Image", padding=6)
        section.pack(fill=tk.X, padx=6, pady=(6, 3))

        # Drop zone
        self._drop_zone = ttk.Label(
            section,
            text="Drop image here\nor click to browse",
            anchor=tk.CENTER,
            justify=tk.CENTER,
            relief="groove",
            padding=20,
        )
        self._drop_zone.pack(fill=tk.X, pady=(0, 4))

        guide_row = ttk.Frame(section)
        guide_row.pack(fill=tk.X, pady=(0, 2))
        self._guide_label = ttk.Label(
            guide_row,
            text="Domain guide: none",
            foreground="gray",
            wraplength=280,
            justify=tk.LEFT,
        )
        self._guide_label.pack(fill=tk.X, anchor=tk.W)

        guide_actions = ttk.Frame(section)
        guide_actions.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(
            guide_actions,
            text="Load guide",
            command=self._browse_guide,
            width=12,
        ).pack(side=tk.LEFT)
        self._unload_guide_btn = ttk.Button(
            guide_actions,
            text="Unload guide",
            command=self._clear_guide,
            width=12,
            state=tk.DISABLED,
        )
        self._unload_guide_btn.pack(side=tk.LEFT, padx=(6, 0))

        # Register DnD
        try:
            self._drop_zone.drop_target_register("DND_Files")
            self._drop_zone.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            logger.warning("tkinterdnd2 not available — drag-and-drop disabled")

        # Click to browse
        self._drop_zone.bind("<Button-1>", self._browse_image)

        # File info labels (hidden until image loaded)
        self._info_frame = ttk.Frame(section)
        self._filename_label = ttk.Label(self._info_frame, text="", wraplength=280)
        self._filename_label.pack(anchor=tk.W)
        self._dims_label = ttk.Label(self._info_frame, text="")
        self._dims_label.pack(anchor=tk.W)
        self._size_label = ttk.Label(self._info_frame, text="")
        self._size_label.pack(anchor=tk.W)

    def _on_drop(self, event: object) -> None:
        """Handle file drop via tkinterdnd2."""
        path = event.data.strip().strip("{}")
        if Path(path).is_file():
            self._load_image(path)

    def _browse_image(self, event: object = None) -> None:
        """Open file dialog to select an image."""
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.tiff *.tif *.bmp *.webp"),
                ("All files", "*.*"),
            ]
        )
        if path:
            self._load_image(path)

    def _load_image(self, path: str) -> None:
        """Load an image and update the UI."""
        self._image_path = path
        self._labels = []
        self._render_label_pills()
        p = Path(path)

        # Update file info
        self._filename_label.config(text=p.name)
        try:
            img = Image.open(path)
            w, h = img.size
            self._dims_label.config(text=f"{w} x {h} px")
        except Exception:
            self._dims_label.config(text="")

        size_mb = p.stat().st_size / (1024 * 1024)
        self._size_label.config(text=f"{size_mb:.1f} MB")

        self._info_frame.pack(fill=tk.X, pady=(4, 0))
        self._drop_zone.config(text=p.name)

        # Notify view
        self._view.on_image_loaded(path)
        self._auto_load_guide(p.parent)
        logger.info("Image loaded: %s", path)

    def _auto_load_guide(self, folder: Path) -> None:
        pack = load_knowledge_pack(folder)
        if pack is None:
            self._clear_guide()
            return
        self._knowledge_pack_path = pack.path
        self._knowledge_pack_name = pack.name
        self._knowledge_pack_defaults = pack.batch_defaults.model_dump()
        self._guide_label.config(text=f"Domain guide: {pack.name}")
        self._unload_guide_btn.config(state=tk.NORMAL)

    def _browse_guide(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")]
        )
        if path:
            pack = load_knowledge_pack(Path(path).parent)
            self._knowledge_pack_path = path
            self._knowledge_pack_name = pack.name if pack else Path(path).stem
            self._knowledge_pack_defaults = pack.batch_defaults.model_dump() if pack else {}
            self._guide_label.config(text=f"Domain guide: {self._knowledge_pack_name}")
            self._unload_guide_btn.config(state=tk.NORMAL)

    def _clear_guide(self) -> None:
        self._knowledge_pack_path = None
        self._knowledge_pack_name = None
        self._knowledge_pack_defaults = {}
        self._guide_label.config(text="Domain guide: none")
        self._unload_guide_btn.config(state=tk.DISABLED)

    # ── Labels section ─────────────────────────────────────────

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
                from core.interrogation import GuidedInterrogator, InterrogationSettings
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
                    InterrogationSettings(
                        host=self._app.prefs.get("ollama_url", "http://localhost:11434"),
                        primary_vlm=str(
                            self._knowledge_pack_defaults.get(
                                "preferred_vlm",
                                self._app.prefs.get("ollama_model", "moondream"),
                            )
                        ),
                        fallback_vlms=[
                            self._app.prefs.get("preferred_fallback_vlm", "minicpm-v"),
                            "llava:7b",
                        ],
                        reasoner_model=self._app.prefs.get(
                            "preferred_text_reasoner", "qwen3.5"
                        ),
                        profile=self._app.prefs.get("interrogation_profile", "balanced"),
                        fallback_mode=self._app.prefs.get(
                            "interrogation_fallback_mode", "adaptive_auto"
                        ),
                        enable_tiling=self._app.prefs.get("enable_tiled_fallback", True),
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
                    preview_detections = _dedup_scan_detections(preview_detections)
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
                        l for l in self._labels
                        if l.get("label", "").lower() in kept_labels
                        or l.get("role") != "parent"
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
        self._labels = [l for l in self._labels if l is not label_data]
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

    def _build_parameters_section(self) -> None:
        section = ttk.LabelFrame(self._inner, text="Parameters", padding=6)
        section.pack(fill=tk.X, padx=6, pady=3)

        # Output modes (checkboxes — user can select multiple)
        ttk.Label(section, text="Output Modes").pack(anchor=tk.W)
        self._mode_structural_svg_var = tk.BooleanVar(value=True)
        self._mode_bitmap_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(
            section, text="Structural SVG (VTracer)", variable=self._mode_structural_svg_var
        ).pack(anchor=tk.W)
        ttk.Checkbutton(
            section, text="Bitmap (TIFF alpha)", variable=self._mode_bitmap_var
        ).pack(anchor=tk.W, pady=(0, 4))

        # Sliders
        self._depth_var = tk.IntVar(value=2)
        self._corner_var = tk.IntVar(value=60)
        self._speckle_var = tk.IntVar(value=8)
        self._smoothing_var = tk.IntVar(value=5)
        self._length_var = tk.DoubleVar(value=4.0)

        sliders = [
            ("Depth", self._depth_var, 1, 3, False),
            ("Corner thr.", self._corner_var, 30, 90, False),
            ("Speckle", self._speckle_var, 2, 20, False),
            ("Smoothing", self._smoothing_var, 1, 10, False),
            ("Length thr.", self._length_var, 2.0, 8.0, True),
        ]

        for label_text, var, from_, to_, is_float in sliders:
            self._build_parameter_row(section, label_text, var, from_, to_, is_float)

    # ── Process section ────────────────────────────────────────

    def _build_process_section(self) -> None:
        section = ttk.Frame(self._inner, padding=6)
        section.pack(fill=tk.X, padx=6, pady=(3, 6))

        self._process_btn = ttk.Button(
            section,
            text="Process image",
            command=self._process_image,
        )
        self._process_btn.pack(fill=tk.X)

        # Staged progress
        self._stage_label = ttk.Label(section, text="", foreground="gray")
        self._process_progress = ttk.Progressbar(
            section, mode="determinate", length=200, maximum=10
        )

    def _process_image(self) -> None:
        """Launch the 10-step orchestrator in a background thread."""
        if not self._image_path:
            return

        self._process_btn.config(state="disabled")
        self._stage_label.pack(fill=tk.X, pady=(4, 0))
        self._process_progress.pack(fill=tk.X, pady=(2, 0))
        self._process_progress["value"] = 0

        # Collect parent labels only — these are used as a whitelist + additions
        # for Moondream detection. Children are auto-discovered per parent.
        active_labels = [
            l.get("canonical_label", l["label"]) for l in self._labels
            if l.get("role") == "parent"
        ]

        def _progress_callback(step: int, msg: str) -> None:
            self._progress_queue.put(("progress", (step, msg)))

        def _worker() -> None:
            try:
                from core.factory import build_capabilities, build_knowledge_pack
                from core.orchestrator import Orchestrator

                caps = build_capabilities(
                    self._app.prefs,
                    corner_threshold=self._corner_var.get(),
                    length_threshold=self._length_var.get(),
                    filter_speckle=self._speckle_var.get(),
                    knowledge_pack_path=self._knowledge_pack_path,
                    knowledge_pack_defaults=self._knowledge_pack_defaults,
                )
                orch = Orchestrator(
                    capabilities=caps,
                    output_dir=Path(self._app.prefs.get(
                        "output_directory",
                        str(Path.home() / "Desktop" / "skiagrafia_out"),
                    )),
                    output_mode=self._get_output_mode(),
                    bilateral_d=int(self._app.prefs.get("bilateral_filter_d", 9)),
                    box_threshold=float(self._app.prefs.get("sam_box_threshold", 0.35)),
                    text_threshold=float(self._app.prefs.get("sam_text_threshold", 0.25)),
                    progress_callback=_progress_callback,
                    knowledge_pack=build_knowledge_pack(self._knowledge_pack_path),
                )
                result = orch.process(
                    self._image_path,
                    active_labels or None,
                    manual_detections=self._view.canvas_panel.get_manual_detections(),
                )
                self._progress_queue.put(("complete", result))
            except Exception as exc:
                logger.error("Processing failed: %s", exc, exc_info=True)
                self._progress_queue.put(("error", str(exc)))

        threading.Thread(target=_worker, daemon=True).start()
        self._poll_process_queue()

    def _poll_process_queue(self) -> None:
        """Poll processing progress from the main thread."""
        try:
            msg_type, data = self._progress_queue.get_nowait()
            if msg_type == "progress":
                step, msg = data
                self._process_progress["value"] = step + 1
                self._stage_label.config(text=msg)
                self._root.after(100, self._poll_process_queue)
            elif msg_type == "complete":
                self._stage_label.config(text="Complete")
                self._process_progress["value"] = self._process_progress["maximum"]
                self._process_btn.config(state="normal")
                self._view.on_processing_complete(data)
            elif msg_type == "error":
                self._stage_label.config(text=f"Failed: {data}", foreground="red")
                self._process_btn.config(state="normal")
        except queue.Empty:
            self._root.after(100, self._poll_process_queue)

    def get_confirmed_labels(self) -> list[str]:
        """Return list of active (non-toggled-off) labels."""
        return [l.get("canonical_label", l["label"]) for l in self._labels]

    def _get_output_mode(self) -> str:
        """Build output mode string from checkbox state."""
        parts: list[str] = []
        if self._mode_structural_svg_var.get():
            parts.append("vector")
        if self._mode_bitmap_var.get():
            parts.append("bitmap")
        return "+".join(parts) if parts else "vector"

    def get_parameters(self) -> dict:
        """Return current parameter values."""
        return {
            "output_mode": self._get_output_mode(),
            "structural_svg": self._mode_structural_svg_var.get(),
            "recursion_depth": self._depth_var.get(),
            "corner_threshold": self._corner_var.get(),
            "speckle": self._speckle_var.get(),
            "smoothing": self._smoothing_var.get(),
            "length_threshold": self._length_var.get(),
        }

    def wants_structural_svg(self) -> bool:
        return self._mode_structural_svg_var.get()

    def _build_parameter_row(
        self,
        parent: ttk.Widget,
        label_text: str,
        var: tk.Variable,
        from_: float,
        to_: float,
        is_float: bool,
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text=label_text, width=10).pack(side=tk.LEFT)

        entry_var = tk.StringVar(
            value=f"{float(var.get()):.1f}" if is_float else str(int(float(var.get())))
        )
        self._parameter_entry_vars[label_text] = entry_var

        entry = ttk.Entry(
            row,
            textvariable=entry_var,
            width=7,
            justify=tk.RIGHT,
        )
        entry.pack(side=tk.RIGHT)

        def _apply_entry(event: object = None) -> None:
            try:
                raw = float(entry_var.get())
            except ValueError:
                raw = float(var.get())
            clamped = min(max(raw, from_), to_)
            if is_float:
                var.set(round(clamped, 1))
                entry_var.set(f"{float(var.get()):.1f}")
            else:
                var.set(int(round(clamped)))
                entry_var.set(str(int(float(var.get()))))

        entry.bind("<Return>", _apply_entry)
        entry.bind("<FocusOut>", _apply_entry)

        def _on_scale(val: str) -> None:
            numeric = float(val)
            if is_float:
                var.set(round(numeric, 1))
                entry_var.set(f"{float(var.get()):.1f}")
            else:
                var.set(int(round(numeric)))
                entry_var.set(str(int(float(var.get()))))

        scale = ttk.Scale(
            row,
            variable=var,
            from_=from_,
            to=to_,
            command=_on_scale,
        )
        scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(6, 6))
