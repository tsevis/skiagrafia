from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageTk

from ui.single.canvas_overlays import (
    render_layer_masks,
    render_layer_vectors,
    render_mask_overlay,
    render_vector_overlay,
)

from ui.theme import is_macos

if TYPE_CHECKING:
    from ui.single.single_view import SingleView

logger = logging.getLogger(__name__)


class CanvasPanel:
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

    def _draw_checkerboard(self, width: int, height: int) -> None:
        tile = 16
        colours = ("#F3F1EC", "#E9E6DF")
        for y in range(0, height, tile):
            for x in range(0, width, tile):
                colour = colours[((x // tile) + (y // tile)) % 2]
                self._canvas.create_rectangle(
                    x,
                    y,
                    min(x + tile, width),
                    min(y + tile, height),
                    fill=colour,
                    outline="",
                )

    def _draw_view_mode(self, display_w: int, display_h: int) -> None:
        mode = self._view_mode.get()
        self._layer_photos = []
        self._overlay_photo = None

        # Always draw the original image as the background layer
        self._draw_original_layer(display_w, display_h, mode)

        if mode in {"masks", "composite"}:
            self._draw_mask_overlays()
        if mode in {"vectors", "composite"}:
            self._draw_vector_overlay(display_w, display_h)

    def _draw_mask_overlays(self) -> None:
        result = getattr(self._view, "_last_result", None)
        if result is None or not hasattr(result, "layers"):
            return

        layers_with_svg = [l for l in result.layers if getattr(l, "svg_data", "")]
        if layers_with_svg:
            mask_opacity = float(self._app.prefs.get("mask_overlay_opacity", 30)) / 100.0
            self._layer_photos.extend(render_layer_masks(
                self._canvas,
                layers_with_svg,
                self._source_size[0],
                self._source_size[1],
                self._zoom,
                self._pan_x,
                self._pan_y,
                opacity=mask_opacity,
            ))
        else:
            # Fallback: draw bbox rectangles when no SVG data available
            from ui.single.canvas_overlays import OVERLAY_PALETTE
            for i, layer in enumerate(result.layers):
                bbox = getattr(layer, "bbox", None)
                if not bbox:
                    continue
                render_mask_overlay(
                    self._canvas,
                    bbox,
                    OVERLAY_PALETTE[i % len(OVERLAY_PALETTE)],
                    self._zoom,
                    self._pan_x,
                    self._pan_y,
                )

    def _draw_vector_overlay(self, display_w: int, display_h: int) -> None:
        result = getattr(self._view, "_last_result", None)
        if result is None or not hasattr(result, "layers"):
            return

        layers_with_svg = [l for l in result.layers if getattr(l, "svg_data", "")]
        if layers_with_svg:
            self._layer_photos.extend(render_layer_vectors(
                self._canvas,
                layers_with_svg,
                self._source_size[0],
                self._source_size[1],
                self._zoom,
                self._pan_x,
                self._pan_y,
            ))
        else:
            # Fallback: render assembled SVG file
            svg_path = getattr(result, "svg_path", None)
            if not svg_path:
                return
            rendered = render_vector_overlay(
                self._canvas,
                svg_path,
                zoom=self._zoom,
                pan_x=self._pan_x,
                pan_y=self._pan_y,
                target_width=self._source_size[0],
                target_height=self._source_size[1],
            )
            self._overlay_photo = rendered[1] if rendered else None

    def _draw_original_layer(self, display_w: int, display_h: int, mode: str) -> None:
        if self._source_image is None:
            return
        resized = self._source_image.resize((display_w, display_h), Image.LANCZOS)
        self._photo_image = ImageTk.PhotoImage(resized)
        self._canvas.create_image(
            self._pan_x,
            self._pan_y,
            anchor=tk.NW,
            image=self._photo_image,
        )
        if mode == "original" and self._scan_preview_image is not None:
            preview = self._scan_preview_image.resize((display_w, display_h), Image.LANCZOS)
            split_px = int(display_w * self._compare_ratio)
            split_px = max(0, min(display_w, split_px))
            if split_px < display_w:
                crop = preview.crop((split_px, 0, display_w, display_h))
                self._compare_photo = ImageTk.PhotoImage(crop)
                self._canvas.create_image(
                    self._pan_x + split_px,
                    self._pan_y,
                    anchor=tk.NW,
                    image=self._compare_photo,
                )
            self._draw_compare_handle(split_px, display_h)
        if mode == "original" and self._show_scan_labels.get() and self._scan_preview_detections:
            self._draw_canvas_labels()
        if self._draw_box_mode.get() and self._box_drag_start and self._box_drag_current:
            x0, y0 = self._box_drag_start
            x1, y1 = self._box_drag_current
            self._canvas.create_rectangle(
                x0,
                y0,
                x1,
                y1,
                outline="#FF9F0A",
                width=2,
                dash=(6, 4),
            )

    def _build_scan_preview_image(self, detections: list[dict]) -> Image.Image | None:
        if self._source_image is None:
            return None
        base = self._source_image.convert("RGBA")
        # Use a separate transparent overlay for proper alpha compositing
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        box_opacity = max(
            0,
            min(255, int(float(self._app.prefs.get("scan_preview_box_opacity", 40)) * 255 / 100)),
        )
        palette = [
            (0, 122, 255),
            (255, 99, 71),
            (52, 199, 89),
            (255, 159, 10),
            (175, 82, 222),
            (90, 200, 250),
            (255, 45, 85),
            (48, 209, 88),
        ]
        for idx, detection in enumerate(detections):
            x0, y0, x1, y1 = detection.get("bbox", (0, 0, 0, 0))
            colour = palette[idx % len(palette)]
            fill = (*colour, box_opacity)
            box_stroke = (*colour, min(255, box_opacity + 60))
            label_stroke = (*colour, 230)
            if self._show_scan_heatmap.get():
                draw.rectangle((x0, y0, x1, y1), fill=fill)
            if self._show_scan_boxes.get():
                draw.rectangle((x0, y0, x1, y1), outline=box_stroke, width=4)
        composed = Image.alpha_composite(base, overlay)
        return composed.convert("RGB")

    def _draw_canvas_labels(self) -> None:
        """Draw detection labels as canvas items at a fixed screen font size."""
        palette = [
            "#007AFF", "#FF6347", "#34C759", "#FF9F0A",
            "#AF52DE", "#5AC8FA", "#FF2D55", "#30D158",
        ]
        font_size = 12
        pad_x, pad_y = 6, 3
        for idx, det in enumerate(self._scan_preview_detections):
            label = det.get("label", f"object {idx + 1}")
            x0, y0 = det.get("bbox", (0, 0, 0, 0))[:2]
            # Convert image coords → canvas coords
            cx = self._pan_x + x0 * self._zoom
            cy = self._pan_y + y0 * self._zoom
            colour = palette[idx % len(palette)]
            # Place label above the box, anchored at bottom-left
            label_y = cy - 4
            tag = f"_lbl_{idx}"
            tid = self._canvas.create_text(
                cx + pad_x, label_y,
                text=label, anchor=tk.SW,
                font=("Helvetica", font_size),
                fill="#181819",
                tags=(tag,),
            )
            # Measure text to draw background pill
            bx0, by0, bx1, by1 = self._canvas.bbox(tid)
            self._canvas.create_rectangle(
                bx0 - pad_x, by0 - pad_y,
                bx1 + pad_x, by1 + pad_y,
                fill="white", stipple="",
                outline=colour, width=2,
                tags=(tag,),
            )
            # Raise text above its background
            self._canvas.tag_raise(tid)

    def _draw_compare_handle(self, split_px: int, display_h: int) -> None:
        line_x = self._pan_x + split_px
        top_y = self._pan_y
        bottom_y = self._pan_y + display_h
        self._canvas.create_line(
            line_x,
            top_y,
            line_x,
            bottom_y,
            fill="white",
            width=2,
        )
        handle_w = 22
        handle_h = 18
        cx = line_x
        cy = self._pan_y + display_h / 2
        self._canvas.create_rectangle(
            cx - handle_w / 2,
            cy - handle_h / 2,
            cx + handle_w / 2,
            cy + handle_h / 2,
            fill="white",
            outline="#B7B7C2",
            width=1,
        )
        self._canvas.create_text(
            cx,
            cy,
            text="↔",
            fill="#3B3B45",
            font=("SF Pro Text", 10, "bold") if is_macos() else ("Segoe UI", 9, "bold"),
        )

    # ── Event handlers ─────────────────────────────────────────

    def _on_scroll(self, event: tk.Event) -> None:
        """Zoom on scroll wheel, centred on cursor."""
        if event.delta > 0:
            factor = self.ZOOM_STEP
        else:
            factor = 1 / self.ZOOM_STEP
        self._zoom_at(factor, event.x, event.y)

    def _on_pan_start(self, event: tk.Event) -> None:
        self._drag_start = (event.x, event.y)

    def _on_pan_drag(self, event: tk.Event) -> None:
        if self._drag_start is None:
            return
        self._auto_cover = False
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._pan_x += dx
        self._pan_y += dy
        self._drag_start = (event.x, event.y)
        self._redraw()

    def _on_pan_end(self, event: tk.Event) -> None:
        self._drag_start = None

    def _on_space_press(self, event: tk.Event) -> None:
        self._space_held = True
        self._canvas.config(cursor="fleur")

    def _on_space_release(self, event: tk.Event) -> None:
        self._space_held = False
        self._canvas.config(cursor="")

    def _on_left_click(self, event: tk.Event) -> None:
        if self._draw_box_mode.get() and self._view_mode.get() == "original":
            self._box_drag_start = (event.x, event.y)
            self._box_drag_current = (event.x, event.y)
            self._redraw()
            return
        if self._view_mode.get() == "original" and self._scan_preview_image is not None:
            self._compare_dragging = True
            self._update_compare_ratio_from_canvas_x(event.x)
            return
        if self._space_held:
            self._drag_start = (event.x, event.y)

    def _on_left_drag(self, event: tk.Event) -> None:
        if self._draw_box_mode.get() and self._box_drag_start is not None:
            self._box_drag_current = (event.x, event.y)
            self._redraw()
            return
        if self._compare_dragging and self._view_mode.get() == "original":
            self._update_compare_ratio_from_canvas_x(event.x)
            return
        if self._space_held and self._drag_start:
            self._auto_cover = False
            dx = event.x - self._drag_start[0]
            dy = event.y - self._drag_start[1]
            self._pan_x += dx
            self._pan_y += dy
            self._drag_start = (event.x, event.y)
            self._redraw()

    def _on_left_release(self, event: tk.Event) -> None:
        if self._draw_box_mode.get() and self._box_drag_start and self._box_drag_current:
            self._finish_manual_box()
            return
        self._compare_dragging = False
        self._drag_start = None

    def _on_view_mode_change(self) -> None:
        """Handle view mode toolbar change."""
        self._redraw()

    def _on_canvas_resize(self, event: tk.Event) -> None:
        """Keep the default preview framing covering the viewport during resizes."""
        if self._source_image is None:
            return
        if self._auto_cover:
            self.zoom_to_cover()
        else:
            self._redraw()

    def _sync_scrollbars(
        self, cw: int, ch: int, display_w: int, display_h: int,
    ) -> None:
        """Update scrollbar thumb positions and sizes based on pan/zoom."""
        if display_w <= cw:
            self._h_scroll.set(0.0, 1.0)
        else:
            # visible fraction and offset within the full image width
            thumb = cw / display_w
            left = -self._pan_x / display_w
            self._h_scroll.set(
                max(0.0, min(1.0 - thumb, left)),
                min(1.0, left + thumb),
            )

        if display_h <= ch:
            self._v_scroll.set(0.0, 1.0)
        else:
            thumb = ch / display_h
            top = -self._pan_y / display_h
            self._v_scroll.set(
                max(0.0, min(1.0 - thumb, top)),
                min(1.0, top + thumb),
            )

    def _on_scrollbar_x(self, *args: str) -> None:
        """Handle horizontal scrollbar interaction."""
        if self._source_image is None:
            return
        display_w = self._source_size[0] * self._zoom
        cw = self._canvas.winfo_width()
        if display_w <= cw:
            return
        self._auto_cover = False
        if args[0] == "moveto":
            fraction = float(args[1])
            self._pan_x = -fraction * display_w
        elif args[0] == "scroll":
            amount = int(args[1])
            if args[2] == "units":
                self._pan_x -= amount * 20
            else:  # pages
                self._pan_x -= amount * cw * 0.9
        self._redraw()

    def _on_scrollbar_y(self, *args: str) -> None:
        """Handle vertical scrollbar interaction."""
        if self._source_image is None:
            return
        display_h = self._source_size[1] * self._zoom
        ch = self._canvas.winfo_height()
        if display_h <= ch:
            return
        self._auto_cover = False
        if args[0] == "moveto":
            fraction = float(args[1])
            self._pan_y = -fraction * display_h
        elif args[0] == "scroll":
            amount = int(args[1])
            if args[2] == "units":
                self._pan_y -= amount * 20
            else:  # pages
                self._pan_y -= amount * ch * 0.9
        self._redraw()

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
