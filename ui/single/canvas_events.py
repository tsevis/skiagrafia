"""canvas_events.py  --  Pointer, keyboard and scrollbar handling.

Zoom, pan, space-drag, manual box drawing and scrollbar synchronisation for
the Single Image canvas. Split out of canvas_panel.py for readability; mixed
into CanvasPanel, so `self` is the panel and behaviour is unchanged.
"""
from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


class CanvasEventsMixin:
    """Input handling. Requires CanvasPanel's attributes."""

    # Provided by CanvasPanel, which mixes this in. Annotations only:
    # no assignment, so nothing exists at runtime and the MRO is untouched.
    ZOOM_STEP: ClassVar[float]
    _canvas: tk.Canvas
    _draw_box_mode: tk.BooleanVar
    _h_scroll: ttk.Scrollbar
    _scan_preview_image: Image.Image | None
    _source_image: Image.Image | None
    _source_size: tuple[int, int]
    _v_scroll: ttk.Scrollbar
    _view_mode: tk.StringVar
    _zoom: float

    # Methods provided by CanvasPanel. CanvasPanel defines the real
    # implementations directly, so they always shadow these stubs in the
    # MRO (CanvasPanel, CanvasDrawingMixin, CanvasEventsMixin) -- these
    # exist only so pyright can see the calls below; if one is ever
    # reached it means the host stopped providing it, so it fails loudly.
    def _redraw(self) -> None:
        raise NotImplementedError

    def _zoom_at(
        self,
        factor: float,
        cx: float | None = None,
        cy: float | None = None,
    ) -> None:
        raise NotImplementedError

    def zoom_to_cover(self, event: object = None) -> None:
        raise NotImplementedError

    def _update_compare_ratio_from_canvas_x(self, canvas_x: float) -> None:
        raise NotImplementedError

    def _finish_manual_box(self) -> None:
        raise NotImplementedError

    def _on_scroll(self, event: tk.Event) -> None:
        """Zoom on scroll wheel, centred on cursor."""
        factor = self.ZOOM_STEP if event.delta > 0 else 1 / self.ZOOM_STEP
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
