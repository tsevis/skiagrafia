from __future__ import annotations

import io
import logging
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageTk

logger = logging.getLogger(__name__)

_cairosvg = None
_cairosvg_checked = False


def _get_cairosvg():
    """Lazy-load cairosvg, returning None if libcairo is missing."""
    global _cairosvg, _cairosvg_checked
    if not _cairosvg_checked:
        _cairosvg_checked = True
        try:
            import cairosvg as _mod
            _cairosvg = _mod
        except OSError:
            logger.warning(
                "libcairo not found — SVG thumbnails will use placeholders. "
                "Fix: brew install cairo"
            )
    return _cairosvg


@lru_cache(maxsize=256)
def render_svg_thumbnail(
    svg_path: str,
    colour_hex: str,
    size: int = 32,
) -> ImageTk.PhotoImage:
    """Render an SVG to a tinted thumbnail PhotoImage.

    Falls back to a solid-colour square if cairosvg/libcairo is unavailable.
    """
    r = int(colour_hex[1:3], 16)
    g = int(colour_hex[3:5], 16)
    b = int(colour_hex[5:7], 16)

    cairo = _get_cairosvg()
    if cairo is not None:
        try:
            png_bytes = cairo.svg2png(
                url=svg_path,
                output_width=size,
                output_height=size,
                background_color="transparent",
            )
            img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
            tinted = Image.new("RGBA", img.size, (r, g, b, 180))
            tinted.putalpha(img.split()[3])
        except Exception:
            logger.debug("SVG thumbnail render failed, using placeholder", exc_info=True)
            tinted = Image.new("RGBA", (size, size), (r, g, b, 180))
    else:
        # Placeholder: solid colour square
        tinted = Image.new("RGBA", (size, size), (r, g, b, 180))

    draw = ImageDraw.Draw(tinted)
    draw.rectangle([0, 0, size - 1, size - 1], outline=(r, g, b, 255), width=1)

    return ImageTk.PhotoImage(tinted)


def invalidate_thumbnail_cache() -> None:
    """Call before re-process to force re-render of all thumbnails."""
    render_svg_thumbnail.cache_clear()
