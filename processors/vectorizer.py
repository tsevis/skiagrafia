from __future__ import annotations

import logging
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np
import vtracer
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_MAX_VTRACER_SVG_BYTES = 32 * 1024 * 1024
_PATH_DATA_RE = re.compile(r"^[0-9a-zA-Z, .+\-]*$")
_TRANSLATE_RE = re.compile(
    r"^translate\(\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*,?\s*"
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*\)$"
)
_XML_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_ATTRIBUTE_ESCAPE = {'"': "&quot;"}

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
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("VTracer requires a non-empty two-dimensional mask.")
    # Invert: object (255) → black (0), background (0) → white (255)
    inverted = cv2.bitwise_not(mask)

    # VTracer only accepts file paths.  A private temporary directory keeps
    # cleanup bounded even if conversion fails part way through.
    with tempfile.TemporaryDirectory(prefix="skiagrafia-vtracer-") as temp_dir:
        in_path = str(Path(temp_dir) / "mask.bmp")
        out_path = str(Path(temp_dir) / "mask.svg")
        if not cv2.imwrite(in_path, inverted):
            raise RuntimeError("VTracer could not write its temporary input image.")
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
        try:
            svg_str = Path(out_path).read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError("VTracer did not produce an SVG output file.") from exc

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
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError("SVG dimensions must be positive integers.")
    if len(layers) > 256:
        raise ValueError("SVG export contains too many layers.")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">',
    ]

    used_ids: set[str] = set()
    for i, layer in enumerate(layers):
        # Layer ids come from VLM-authored labels; escape before they are
        # embedded as an attribute value or they can inject markup.
        raw_id = str(layer.get("id", f"layer_{i}"))
        unique_id = raw_id
        suffix = 2
        while unique_id in used_ids:
            unique_id = f"{raw_id}_{suffix}"
            suffix += 1
        used_ids.add(unique_id)
        layer_id = escape(unique_id, {'"': "&quot;"})
        svg_data = str(layer.get("svg_data", ""))
        dx = _safe_svg_number(layer.get("dx", 0))
        dy = _safe_svg_number(layer.get("dy", 0))
        fill = LAYER_PALETTE[i % len(LAYER_PALETTE)]

        if float(dx) != 0 or float(dy) != 0:
            parts.append(
                f'  <g id="{layer_id}" fill="{fill}" transform="translate({dx},{dy})">'
            )
        else:
            parts.append(f'  <g id="{layer_id}" fill="{fill}">')

        if layer.get("label"):
            label = _XML_CONTROL_RE.sub(" ", str(layer["label"]))
            parts.append(f"    <title>{escape(label)}</title>")
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
        preserve_detail: bool = False,
    ) -> None:
        self._corner = corner_threshold
        self._length = length_threshold
        self._splice = splice_threshold
        self._speckle = filter_speckle
        self._preserve_detail = preserve_detail

    def trace(self, mask: NDArray[np.uint8]) -> str:
        """Trace a binary mask to SVG path data (Vectorizer protocol)."""
        return trace_mask(
            mask,
            corner_threshold=self._corner,
            length_threshold=min(self._length, 1.0) if self._preserve_detail else self._length,
            splice_threshold=self._splice,
            filter_speckle=0 if self._preserve_detail else self._speckle,
        )


def _extract_svg_content(svg_str: str) -> str:
    """Extract only validated VTracer path elements from an SVG fragment."""
    if len(svg_str.encode("utf-8")) > _MAX_VTRACER_SVG_BYTES:
        raise ValueError("VTracer SVG exceeds the maximum supported size.")
    if "<!DOCTYPE" in svg_str.upper() or "<!ENTITY" in svg_str.upper():
        raise ValueError("VTracer SVG must not contain DTDs or entities.")
    content = svg_str.strip()
    if not content:
        return ""
    try:
        root = ET.fromstring(content if "<svg" in content else f"<svg>{content}</svg>")  # noqa: S314 — DTDs and entities are rejected and the input is size-capped above
    except ET.ParseError as exc:
        raise ValueError("VTracer returned malformed SVG.") from exc

    paths: list[str] = []
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "svg":
            continue
        if tag != "path":
            raise ValueError(f"VTracer returned unsupported SVG element: {tag}.")
        paths.append(_safe_path_element(element))
    return "\n".join(paths)


def _strip_vtracer_fills(svg_content: str) -> str:
    """Remove white background paths and strip fills from object paths.

    VTracer binary mode outputs paths with explicit fill="#000000" (object)
    and fill="#ffffff" (background).  This removes the background paths
    entirely and strips fill attributes from the remaining object paths so
    that a parent ``<g fill="...">`` colour applies correctly.
    """
    # Parse again so direct callers receive the same validation as assembly.
    content = _extract_svg_content(svg_content)
    if not content:
        return ""
    root = ET.fromstring(f"<svg>{content}</svg>")  # noqa: S314 — content already passed _extract_svg_content()
    visible: list[str] = []
    for element in root:
        fill = element.attrib.get("fill", "").lower()
        if fill == "#ffffff":
            continue
        element.attrib.pop("fill", None)
        visible.append(_safe_path_element(element))
    return "\n".join(visible)


def _safe_path_element(element: ET.Element) -> str:
    """Serialize the small VTracer path subset without carrying active SVG."""
    attrs: list[str] = []
    d = element.attrib.get("d")
    if d is not None:
        if not _PATH_DATA_RE.fullmatch(d) or len(d) > 2_000_000:
            raise ValueError("VTracer returned invalid path data.")
        attrs.append(f'd="{escape(d, _ATTRIBUTE_ESCAPE)}"')
    transform = element.attrib.get("transform")
    if transform is not None:
        if not _TRANSLATE_RE.fullmatch(transform):
            raise ValueError("VTracer returned an unsupported path transform.")
        attrs.append(f'transform="{escape(transform, _ATTRIBUTE_ESCAPE)}"')
    fill = element.attrib.get("fill")
    if fill is not None:
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", fill):
            raise ValueError("VTracer returned an invalid fill colour.")
        attrs.append(f'fill="{fill}"')
    return "<path" + (" " + " ".join(attrs) if attrs else "") + "/>"


def _safe_svg_number(value: object) -> str:
    """Format numeric transforms without allowing attribute injection."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(  # noqa: TRY004 — every SVG validation failure is a ValueError
            "SVG translation must be numeric."
        )
    if not np.isfinite(value) or abs(value) > 10_000_000:
        raise ValueError("SVG translation is out of range.")
    return f"{value:g}"
