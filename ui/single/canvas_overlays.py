from __future__ import annotations

import io
import logging
import re
import tkinter as tk

from PIL import Image, ImageTk

_cairosvg = None
_cairosvg_checked = False


def _get_cairosvg():
    global _cairosvg, _cairosvg_checked
    if not _cairosvg_checked:
        _cairosvg_checked = True
        try:
            import cairosvg as _mod
            _cairosvg = _mod
        except OSError:
            pass
    return _cairosvg

logger = logging.getLogger(__name__)

# Per-layer colour palette (8 distinct colours).
OVERLAY_PALETTE = [
    "#007AFF",  # blue
    "#FF6347",  # coral
    "#34C759",  # green
    "#FF9F0A",  # orange
    "#AF52DE",  # purple
    "#5AC8FA",  # cyan
    "#FF2D55",  # red-pink
    "#30D158",  # emerald
]


# ── SVG helpers ──────────────────────────────────────────────────────────


def _extract_svg_inner(svg_str: str) -> str:
    """Extract inner content (paths) from a full VTracer SVG string."""
    content = svg_str
    if "<?xml" in content:
        content = content[content.index("?>") + 2:]
    if "<svg" in content and "</svg>" in content:
        start = content.index(">", content.index("<svg")) + 1
        end = content.rindex("</svg>")
        content = content[start:end]
    return content.strip()


def _recolor_paths(svg_inner: str, fill_color: str) -> str:
    """Remove white background paths and recolor object paths.

    VTracer binary mode emits ``fill="#000000"`` (object) and
    ``fill="#ffffff"`` (background).  We drop the background paths and
    replace all remaining fill attrs with the desired colour.
    """
    # Remove white / background paths
    result = re.sub(
        r"<path\s[^>]*?\bfill\s*=\s*\"#[fF]{6}\"[^>]*?/>",
        "",
        svg_inner,
    )
    # Replace all remaining fill attrs with the desired colour
    result = re.sub(r'\sfill\s*=\s*"[^"]*"', f' fill="{fill_color}"', result)
    return result


def _build_layer_svg(
    inner: str,
    source_w: int,
    source_h: int,
    render_w: int,
    render_h: int,
    *,
    fill_opacity: float = 1.0,
    stroke: str | None = None,
    stroke_width: float = 0,
) -> str:
    """Wrap recolored inner SVG content in a sized ``<svg>`` element."""
    stroke_attr = ""
    if stroke:
        stroke_attr = f' stroke="{stroke}" stroke-width="{stroke_width}"'
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {source_w} {source_h}" '
        f'width="{render_w}" height="{render_h}">'
        f'<g fill-opacity="{fill_opacity:.2f}"{stroke_attr}>'
        f'{inner}'
        f'</g></svg>'
    )


def _render_svg_to_photo(svg_bytes: bytes, width: int, height: int) -> ImageTk.PhotoImage | None:
    """Render SVG bytes to a transparent PhotoImage via cairosvg."""
    cairo = _get_cairosvg()
    if cairo is None:
        return None
    try:
        png_bytes = cairo.svg2png(
            bytestring=svg_bytes,
            output_width=width,
            output_height=height,
        )
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        return ImageTk.PhotoImage(img)
    except Exception:
        logger.error("SVG→PNG render failed", exc_info=True)
        return None


# ── Public overlay renderers ─────────────────────────────────────────────


def render_layer_masks(
    canvas: tk.Canvas,
    layers: list,
    source_width: int,
    source_height: int,
    zoom: float,
    pan_x: float,
    pan_y: float,
    opacity: float = 0.40,
) -> list[ImageTk.PhotoImage]:
    """Render per-layer SVG masks as coloured semi-transparent filled shapes.

    Returns a list of PhotoImage references the caller **must** keep alive
    to prevent garbage collection.
    """
    render_w = int(source_width * zoom)
    render_h = int(source_height * zoom)
    if render_w < 1 or render_h < 1:
        return []

    photos: list[ImageTk.PhotoImage] = []
    for i, layer in enumerate(layers):
        svg_data = getattr(layer, "svg_data", "")
        if not svg_data:
            continue

        colour = OVERLAY_PALETTE[i % len(OVERLAY_PALETTE)]
        inner = _recolor_paths(_extract_svg_inner(svg_data), colour)
        if not inner.strip():
            continue

        svg_str = _build_layer_svg(
            inner, source_width, source_height, render_w, render_h,
            fill_opacity=opacity,
        )
        photo = _render_svg_to_photo(svg_str.encode("utf-8"), render_w, render_h)
        if photo is None:
            continue

        canvas.create_image(pan_x, pan_y, anchor=tk.NW, image=photo, tags=("overlay",))
        photos.append(photo)

    return photos


def render_layer_vectors(
    canvas: tk.Canvas,
    layers: list,
    source_width: int,
    source_height: int,
    zoom: float,
    pan_x: float,
    pan_y: float,
) -> list[ImageTk.PhotoImage]:
    """Render per-layer SVG vectors as coloured stroke outlines with light fill.

    Returns a list of PhotoImage references the caller **must** keep alive.
    """
    render_w = int(source_width * zoom)
    render_h = int(source_height * zoom)
    if render_w < 1 or render_h < 1:
        return []

    photos: list[ImageTk.PhotoImage] = []
    for i, layer in enumerate(layers):
        svg_data = getattr(layer, "svg_data", "")
        if not svg_data:
            continue

        colour = OVERLAY_PALETTE[i % len(OVERLAY_PALETTE)]
        inner = _recolor_paths(_extract_svg_inner(svg_data), colour)
        if not inner.strip():
            continue

        svg_str = _build_layer_svg(
            inner, source_width, source_height, render_w, render_h,
            fill_opacity=0.15,
            stroke=colour,
            stroke_width=1.5,
        )
        photo = _render_svg_to_photo(svg_str.encode("utf-8"), render_w, render_h)
        if photo is None:
            continue

        canvas.create_image(pan_x, pan_y, anchor=tk.NW, image=photo, tags=("overlay",))
        photos.append(photo)

    return photos


# ── Legacy helpers (kept for backward compat) ────────────────────────────


def render_mask_overlay(
    canvas: tk.Canvas,
    bbox: tuple[int, int, int, int],
    colour_hex: str,
    zoom: float,
    pan_x: float,
    pan_y: float,
    selected: bool = False,
    opacity: int = 30,
) -> int:
    """Draw a bbox rectangle fallback when SVG data is unavailable."""
    x0, y0, x1, y1 = bbox
    cx0 = x0 * zoom + pan_x
    cy0 = y0 * zoom + pan_y
    cx1 = x1 * zoom + pan_x
    cy1 = y1 * zoom + pan_y

    item = canvas.create_rectangle(
        cx0, cy0, cx1, cy1,
        fill=colour_hex,
        stipple="gray50",
        outline=colour_hex if selected else "",
        width=1.5 if selected else 0,
    )
    return item


def render_vector_overlay(
    canvas: tk.Canvas,
    svg_path: str,
    zoom: float,
    pan_x: float,
    pan_y: float,
    target_width: int,
    target_height: int,
) -> tuple[int, ImageTk.PhotoImage] | None:
    """Render a single assembled SVG file (fallback)."""
    cairo = _get_cairosvg()
    if cairo is None:
        return None
    try:
        render_w = int(target_width * zoom)
        render_h = int(target_height * zoom)
        if render_w < 1 or render_h < 1:
            return None

        png_bytes = cairo.svg2png(
            url=svg_path,
            output_width=render_w,
            output_height=render_h,
            background_color="transparent",
        )
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        photo = ImageTk.PhotoImage(img)

        item = canvas.create_image(pan_x, pan_y, anchor=tk.NW, image=photo)
        return item, photo
    except Exception:
        logger.error("Failed to render vector overlay", exc_info=True)
        return None


def clear_overlays(canvas: tk.Canvas, tag: str = "overlay") -> None:
    """Remove all overlay items from canvas."""
    canvas.delete(tag)
