from __future__ import annotations

import io
import logging
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np
import vtracer
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

LAYER_PALETTE = [
    "#4A90D9",  # blue
    "#E67E22",  # orange
    "#27AE60",  # green
    "#E74C3C",  # red
    "#9B59B6",  # purple
    "#F39C12",  # yellow
    "#1ABC9C",  # teal
    "#E91E63",  # pink
]


def trace_mask(
    mask: NDArray[np.uint8],
    mode: str = "spline",
    corner_threshold: int = 60,
    length_threshold: float = 4.0,
    splice_threshold: int = 45,
    filter_speckle: int = 8,
) -> str:
    """Trace a binary mask to SVG path data using VTracer.

    The mask is inverted (object→black, bg→white) before tracing so that
    VTracer's binary mode produces filled paths for the object silhouette,
    not the background.

    Returns raw SVG string (single layer, no viewBox wrapper).
    """
    # Invert: object (255) → black (0), background (0) → white (255)
    inverted = cv2.bitwise_not(mask)

    # VTracer expects file paths for both input and output.
    with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as tmp_in, \
         tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_out:
        cv2.imwrite(tmp_in.name, inverted)
        in_path = tmp_in.name
        out_path = tmp_out.name

    try:
        vtracer.convert_image_to_svg_py(
            in_path,
            out_path,
            colormode="binary",
            mode=mode,
            corner_threshold=corner_threshold,
            length_threshold=length_threshold,
            splice_threshold=splice_threshold,
            filter_speckle=filter_speckle,
        )
        svg_str = Path(out_path).read_text()
    finally:
        Path(in_path).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)

    logger.info(
        "VTracer: traced mask %s, output %d chars",
        mask.shape,
        len(svg_str),
    )
    return svg_str


def assemble_svg(
    width: int,
    height: int,
    layers: list[dict[str, str | int | float]],
) -> str:
    """Assemble a multi-layer SVG from traced path data.

    Each layer dict: {"id": str, "svg_data": str, "dx": int, "dy": int}
    Layer 0 is the parent silhouette, layers 1..N are children.
    """
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">',
    ]

    for i, layer in enumerate(layers):
        layer_id = layer.get("id", f"layer_{i}")
        svg_data = layer.get("svg_data", "")
        dx = layer.get("dx", 0)
        dy = layer.get("dy", 0)
        fill = LAYER_PALETTE[i % len(LAYER_PALETTE)]

        if dx or dy:
            parts.append(
                f'  <g id="{layer_id}" fill="{fill}" transform="translate({dx},{dy})">'
            )
        else:
            parts.append(f'  <g id="{layer_id}" fill="{fill}">')

        # Extract just the path data from VTracer output
        # VTracer wraps in full SVG — extract inner content
        inner = _strip_vtracer_fills(_extract_svg_content(svg_data))
        parts.append(f"    {inner}")
        parts.append("  </g>")

    parts.append("</svg>")
    return "\n".join(parts)


class VTracerVectorizer:
    """Wraps VTracer module functions into the Vectorizer protocol."""

    def __init__(
        self,
        corner_threshold: int = 60,
        length_threshold: float = 4.0,
        splice_threshold: int = 45,
        filter_speckle: int = 8,
    ) -> None:
        self._corner = corner_threshold
        self._length = length_threshold
        self._splice = splice_threshold
        self._speckle = filter_speckle

    def trace(self, mask: NDArray[np.uint8]) -> str:
        """Trace a binary mask to SVG path data (Vectorizer protocol)."""
        return trace_mask(
            mask,
            corner_threshold=self._corner,
            length_threshold=self._length,
            splice_threshold=self._splice,
            filter_speckle=self._speckle,
        )


def _extract_svg_content(svg_str: str) -> str:
    """Extract inner path elements from a full SVG string."""
    # Remove xml declaration and svg wrapper
    content = svg_str
    # Strip XML declaration
    if "<?xml" in content:
        content = content[content.index("?>") + 2 :]
    # Strip <svg ...> and </svg>
    if "<svg" in content:
        start = content.index(">", content.index("<svg")) + 1
        end = content.rindex("</svg>")
        content = content[start:end]
    return content.strip()


def _strip_vtracer_fills(svg_content: str) -> str:
    """Remove white background paths and strip fills from object paths.

    VTracer binary mode outputs paths with explicit fill="#000000" (object)
    and fill="#ffffff" (background).  This removes the background paths
    entirely and strips fill attributes from the remaining object paths so
    that a parent ``<g fill="...">`` colour applies correctly.
    """
    # Remove white / background path elements
    result = re.sub(
        r"<path\s[^>]*?\bfill\s*=\s*\"#[fF]{6}\"[^>]*?/>",
        "",
        svg_content,
    )
    # Strip explicit fill attrs from remaining (object) elements
    result = re.sub(r'\sfill\s*=\s*"[^"]*"', "", result)
    return result
