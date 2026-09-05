"""canvas_drawing.py  --  Canvas rendering for the Single Image view.

Everything that paints onto the Tk canvas: the transparency checkerboard,
mask and vector overlays, the scan-preview composite, canvas labels and the
compare-mode handle. Split out of canvas_panel.py for readability; mixed
into CanvasPanel, so `self` is the panel and behaviour is unchanged.
"""
from __future__ import annotations

import logging
import tkinter as tk

from PIL import Image, ImageDraw, ImageTk

from ui.single.canvas_overlays import (
    render_layer_masks,
    render_layer_vectors,
    render_mask_overlay,
    render_vector_overlay,
)
from ui.theme import is_macos

logger = logging.getLogger(__name__)


class CanvasDrawingMixin:
    """Canvas painting. Requires CanvasPanel's attributes."""

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

        layers_with_svg = [layer for layer in result.layers if getattr(layer, "svg_data", "")]
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

        layers_with_svg = [layer for layer in result.layers if getattr(layer, "svg_data", "")]
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
