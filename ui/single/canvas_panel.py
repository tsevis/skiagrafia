from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import TYPE_CHECKING

from PIL import Image, ImageTk

from ui.single.canvas_drawing import CanvasDrawingMixin
from ui.single.canvas_events import CanvasEventsMixin
from ui.theme import is_macos

if TYPE_CHECKING:
    from ui.single.single_view import SingleView

logger = logging.getLogger(__name__)


class CanvasPanel(CanvasDrawingMixin, CanvasEventsMixin):
    """Dark canvas with zoom/pan, toolbar, and overlay rendering.

    Central panel of the Single Image mode.
    """

    MIN_ZOOM = 0.1
    MAX_ZOOM = 8.0
    ZOOM_STEP = 1.15

    def __init__(self, parent: tk.Widget, view: SingleView) -> None:
        self._view = view
        self._app = view.app
        self._root = view.root

        self.frame = ttk.Frame(parent)

        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._source_size = (1, 1)
        self._source_image: Image.Image | None = None
        self._photo_image: ImageTk.PhotoImage | None = None
        self._overlay_photo: ImageTk.PhotoImage | None = None
        self._compare_photo: ImageTk.PhotoImage | None = None
        self._layer_photos: list[ImageTk.PhotoImage] = []
        self._scan_preview_image: Image.Image | None = None
        self._scan_preview_detections: list[dict] = []
        self._compare_ratio = 0.5
        self._compare_dragging = False
        self._draw_box_mode = tk.BooleanVar(value=False)
        self._show_scan_boxes = tk.BooleanVar(
            value=bool(self._app.prefs.get("scan_preview_show_boxes", True))
        )
        self._show_scan_labels = tk.BooleanVar(
            value=bool(self._app.prefs.get("scan_preview_show_labels", True))
        )
        self._show_scan_heatmap = tk.BooleanVar(
            value=bool(self._app.prefs.get("scan_preview_show_heatmap", True))
        )
        self._box_drag_start: tuple[float, float] | None = None
        self._box_drag_current: tuple[float, float] | None = None
        self._drag_start: tuple[float, float] | None = None
        self._space_held = False
        self._view_mode = tk.StringVar(value="original")
        self._debounce_id: str | None = None
        self._auto_cover = True

        self._build_toolbar()
        self._build_canvas()
        self._build_preview_controls()
        self._build_status_bar()
        self._bind_events()

    def _build_toolbar(self) -> None:
        """Toolbar with view mode radio buttons."""
        toolbar = ttk.Frame(self.frame)
        toolbar.pack(fill=tk.X, padx=4, pady=(4, 0))

        modes = [
            ("Original", "original"),
            ("Masks", "masks"),
            ("Vectors", "vectors"),
            ("Composite", "composite"),
        ]
        for text, value in modes:
            rb = ttk.Radiobutton(
                toolbar,
                text=text,
                variable=self._view_mode,
                value=value,
                command=self._on_view_mode_change,
            )
            rb.pack(side=tk.LEFT, padx=2)

        # Zoom controls (right side): [Fit] [-] 100% [+]
        ttk.Button(
            toolbar, text="+", width=2,
            command=lambda: self._zoom_at(self.ZOOM_STEP),
        ).pack(side=tk.RIGHT, padx=1)

        self._zoom_label = ttk.Label(toolbar, text="100%")
        self._zoom_label.pack(side=tk.RIGHT, padx=4)

        ttk.Button(
            toolbar, text="-", width=2,
            command=lambda: self._zoom_at(1 / self.ZOOM_STEP),
        ).pack(side=tk.RIGHT, padx=1)

        ttk.Button(
            toolbar, text="Fit", width=3,
            command=self.zoom_to_fit,
        ).pack(side=tk.RIGHT, padx=(0, 4))

    def _build_canvas(self) -> None:
        """Build the main zoomable canvas with scrollbars."""
        canvas_frame = ttk.Frame(self.frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(
            canvas_frame,
            bg="#E9E7E2",
            highlightthickness=0,
        )

        self._h_scroll = ttk.Scrollbar(
            canvas_frame, orient=tk.HORIZONTAL, command=self._on_scrollbar_x,
        )
        self._v_scroll = ttk.Scrollbar(
            canvas_frame, orient=tk.VERTICAL, command=self._on_scrollbar_y,
        )

        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._v_scroll.grid(row=0, column=1, sticky="ns")
        self._h_scroll.grid(row=1, column=0, sticky="ew")

        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

    def _build_preview_controls(self) -> None:
        controls = ttk.Frame(self.frame)
        controls.pack(fill=tk.X, padx=6, pady=(4, 2))

        ttk.Checkbutton(
            controls,
            text="Boxes",
            variable=self._show_scan_boxes,
            command=self.refresh_overlays,
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            controls,
            text="Labels",
            variable=self._show_scan_labels,
            command=self.refresh_overlays,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Checkbutton(
            controls,
            text="Heatmap",
            variable=self._show_scan_heatmap,
            command=self.refresh_overlays,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Checkbutton(
            controls,
            text="Add Box",
            variable=self._draw_box_mode,
            command=self._on_draw_box_toggle,
        ).pack(side=tk.RIGHT)

    def _build_status_bar(self) -> None:
        """Status bar below canvas: dimensions, file size, active layer."""
        status = ttk.Frame(self.frame)
        status.pack(fill=tk.X, padx=4, pady=(0, 4))

        self._dims_status = ttk.Label(status, text="")
        self._dims_status.pack(side=tk.LEFT, padx=(0, 12))

        self._size_status = ttk.Label(status, text="")
        self._size_status.pack(side=tk.LEFT, padx=(0, 12))

        self._layer_status = ttk.Label(status, text="")
        self._layer_status.pack(side=tk.LEFT)

    def _bind_events(self) -> None:
        """Set up zoom, pan, and keyboard bindings."""
        self._canvas.bind("<MouseWheel>", self._on_scroll)
        self._canvas.bind("<Button-2>", self._on_pan_start)
        self._canvas.bind("<B2-Motion>", self._on_pan_drag)
        self._canvas.bind("<ButtonRelease-2>", self._on_pan_end)

        # Space + left drag for pan
        self._root.bind("<KeyPress-space>", self._on_space_press)
        self._root.bind("<KeyRelease-space>", self._on_space_release)
        self._canvas.bind("<Button-1>", self._on_left_click)
        self._canvas.bind("<B1-Motion>", self._on_left_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_left_release)

        # Keyboard shortcuts
        modifier = "Command" if is_macos() else "Control"
        self._root.bind(f"<{modifier}-0>", self.zoom_to_fit)
        self._root.bind(f"<{modifier}-equal>", lambda e: self._zoom_at(self.ZOOM_STEP))
        self._root.bind(f"<{modifier}-minus>", lambda e: self._zoom_at(1 / self.ZOOM_STEP))

        # Redraw on resize
        self._canvas.bind("<Configure>", self._on_canvas_resize)

    def load_image(self, path: str) -> None:
        """Load an image file onto the canvas."""
        try:
            self._source_image = Image.open(path).convert("RGB")
            self._source_size = self._source_image.size

            p = Path(path)
            w, h = self._source_size
            self._dims_status.config(text=f"{w} x {h} px")
            size_mb = p.stat().st_size / (1024 * 1024)
            self._size_status.config(text=f"{size_mb:.1f} MB")

            # Cover the preview after layout so the viewport never shows empty gutters.
            self._auto_cover = True
            self._root.after(50, self.zoom_to_cover)
        except Exception:
            logger.error("Failed to load image: %s", path, exc_info=True)

    def _zoom_at(
        self,
        factor: float,
        cx: float | None = None,
        cy: float | None = None,
    ) -> None:
        """Zoom centred on (cx, cy) — cursor or window centre."""
        if cx is None:
            cx = self._canvas.winfo_width() / 2
        if cy is None:
            cy = self._canvas.winfo_height() / 2

        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self._zoom * factor))
        if new_zoom == self._zoom:
            return
        self._auto_cover = False
        ratio = new_zoom / self._zoom
        self._pan_x = cx - ratio * (cx - self._pan_x)
        self._pan_y = cy - ratio * (cy - self._pan_y)
        self._zoom = new_zoom
        self._redraw()
        self._zoom_label.config(text=f"{int(self._zoom * 100)}%")

    def zoom_to_fit(self, event: object = None) -> None:
        """Reset the preview to the default auto-cover framing."""
        self.zoom_to_cover(event)

    def zoom_to_cover(self, event: object = None) -> None:
        """Scale the image so the preview window is completely filled."""
        if self._source_image is None:
            return
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return
        iw, ih = self._source_size
        self._auto_cover = True
        self._zoom = max(cw / iw, ch / ih)
        self._pan_x = (cw - iw * self._zoom) / 2
        self._pan_y = (ch - ih * self._zoom) / 2
        self._redraw()
        self._zoom_label.config(text=f"{int(self._zoom * 100)}%")

    def _redraw(self) -> None:
        """Redraw the canvas with current zoom/pan."""
        if self._source_image is None:
            return

        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return

        iw, ih = self._source_size
        display_w = int(iw * self._zoom)
        display_h = int(ih * self._zoom)

        if display_w < 1 or display_h < 1:
            return

        self._canvas.delete("all")
        self._draw_checkerboard(cw, ch)
        self._draw_view_mode(display_w, display_h)
        self._sync_scrollbars(cw, ch, display_w, display_h)

    def refresh_overlays(self) -> None:
        """Re-render overlays after processing or zoom change."""
        if self._scan_preview_detections:
            self._scan_preview_image = self._build_scan_preview_image(self._scan_preview_detections)
        self._redraw()

    def clear_scan_preview(self) -> None:
        self._scan_preview_image = None
        self._scan_preview_detections = []
        self._compare_ratio = 0.5
        self._draw_box_mode.set(False)
        self._box_drag_start = None
        self._box_drag_current = None
        self.refresh_overlays()

    def set_scan_preview(self, detections: list[dict]) -> None:
        manual = [d for d in self._scan_preview_detections if d.get("source") == "manual"]
        self._scan_preview_detections = list(detections) + manual
        self._show_scan_boxes.set(bool(self._app.prefs.get("scan_preview_show_boxes", True)))
        self._show_scan_labels.set(bool(self._app.prefs.get("scan_preview_show_labels", True)))
        self._show_scan_heatmap.set(bool(self._app.prefs.get("scan_preview_show_heatmap", True)))
        self._view_mode.set("original")
        self._scan_preview_image = self._build_scan_preview_image(self._scan_preview_detections)
        self._compare_ratio = 0.5
        self.refresh_overlays()

    def refresh_scan_preview(self) -> None:
        """Rebuild scan preview image (e.g. after opacity change) and redraw."""
        if self._scan_preview_detections:
            self._scan_preview_image = self._build_scan_preview_image(self._scan_preview_detections)
            self.refresh_overlays()

    def add_manual_detection(self, label: str, bbox: tuple[int, int, int, int]) -> None:
        self._scan_preview_detections.append(
            {
                "label": label,
                "role": "parent",
                "bbox": bbox,
                "confidence": 1.0,
                "source": "manual",
            }
        )
        self._scan_preview_image = self._build_scan_preview_image(self._scan_preview_detections)
        self._compare_ratio = 0.5
        self._draw_box_mode.set(False)
        self._box_drag_start = None
        self._box_drag_current = None
        self.refresh_overlays()

    def rename_detection(self, old_label: str, new_label: str) -> None:
        """Rename a detection's label in the scan preview."""
        self._scan_preview_detections = [
            {**d, "label": new_label} if d.get("label", "").lower() == old_label.lower() else d
            for d in self._scan_preview_detections
        ]
        if self._source_image is not None:
            self._scan_preview_image = self._build_scan_preview_image(self._scan_preview_detections)
        self._redraw()

    def remove_detection(self, label: str) -> None:
        """Remove a detection from the scan preview by label."""
        self._scan_preview_detections = [
            d for d in self._scan_preview_detections
            if d.get("label", "").lower() != label.lower()
        ]
        if self._scan_preview_detections:
            self._scan_preview_image = self._build_scan_preview_image(self._scan_preview_detections)
        else:
            self._scan_preview_image = None
        self._redraw()

    def get_manual_detections(self) -> list[dict]:
        return [
            dict(detection)
            for detection in self._scan_preview_detections
            if detection.get("source") == "manual"
        ]

    def get_scan_preview_detections(self) -> list[dict]:
        """Return a copy of all scan preview detections (auto + manual)."""
        return [dict(d) for d in self._scan_preview_detections]

    def set_active_layer(self, label: str) -> None:
        """Update status bar with active layer name."""
        self._layer_status.config(text=label)

    def _update_compare_ratio_from_canvas_x(self, canvas_x: float) -> None:
        if self._source_image is None:
            return
        display_w = self._source_size[0] * self._zoom
        if display_w <= 1:
            return
        ratio = (canvas_x - self._pan_x) / display_w
        self._compare_ratio = max(0.0, min(1.0, ratio))
        self._redraw()

    def _on_draw_box_toggle(self) -> None:
        if self._draw_box_mode.get():
            self._compare_dragging = False
            self._canvas.config(cursor="crosshair")
        else:
            self._box_drag_start = None
            self._box_drag_current = None
            self._canvas.config(cursor="" if not self._space_held else "fleur")
            self._redraw()

    def _finish_manual_box(self) -> None:
        if self._source_image is None or self._box_drag_start is None or self._box_drag_current is None:
            return
        bbox = self._canvas_rect_to_image_bbox(self._box_drag_start, self._box_drag_current)
        self._box_drag_start = None
        self._box_drag_current = None
        if bbox is None:
            self._redraw()
            return
        label = self._prompt_manual_label()
        if not label:
            self._redraw()
            return
        self._view.on_manual_detection_added(label, bbox)
        self._canvas.config(cursor="")

    def _prompt_manual_label(self) -> str | None:
        dialog = tk.Toplevel(self._root)
        dialog.title("Name Object")
        dialog.geometry("280x120")
        dialog.transient(self._root)
        dialog.grab_set()
        ttk.Label(dialog, text="Object label:").pack(anchor=tk.W, padx=10, pady=(10, 4))
        value = tk.StringVar()
        entry = ttk.Entry(dialog, textvariable=value)
        entry.pack(fill=tk.X, padx=10)
        entry.focus_set()
        result = {"label": None}

        def _confirm() -> None:
            result["label"] = value.get().strip() or None
            dialog.destroy()

        entry.bind("<Return>", lambda _e: _confirm())
        ttk.Button(dialog, text="Add", command=_confirm).pack(pady=10)
        self._root.wait_window(dialog)
        return result["label"]

    def _canvas_rect_to_image_bbox(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> tuple[int, int, int, int] | None:
        if self._zoom <= 0:
            return None
        sx, sy = start
        ex, ey = end
        x0 = int((min(sx, ex) - self._pan_x) / self._zoom)
        y0 = int((min(sy, ey) - self._pan_y) / self._zoom)
        x1 = int((max(sx, ex) - self._pan_x) / self._zoom)
        y1 = int((max(sy, ey) - self._pan_y) / self._zoom)
        iw, ih = self._source_size
        x0 = max(0, min(iw, x0))
        y0 = max(0, min(ih, y0))
        x1 = max(0, min(iw, x1))
        y1 = max(0, min(ih, y1))
        if (x1 - x0) < 8 or (y1 - y0) < 8:
            return None
        return (x0, y0, x1, y1)
