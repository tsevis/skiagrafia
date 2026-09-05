from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import TYPE_CHECKING

from PIL import Image

from core.knowledge import load_knowledge_pack
from ui.single.left_panel_labels import LabelsSectionMixin

if TYPE_CHECKING:
    from ui.single.single_view import SingleView

logger = logging.getLogger(__name__)

class LeftPanel(LabelsSectionMixin):
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
            lbl.get("canonical_label", lbl["label"]) for lbl in self._labels
            if lbl.get("role") == "parent"
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
        return [lbl.get("canonical_label", lbl["label"]) for lbl in self._labels]

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
