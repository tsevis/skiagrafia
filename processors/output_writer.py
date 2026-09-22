from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from utils.cairo_support import load_cairosvg
from utils.security import SecurityError, atomic_write_bytes, temporary_output_path

logger = logging.getLogger(__name__)

_MAX_SVG_BYTES = 32 * 1024 * 1024
_FORBIDDEN_SVG_TAGS = {"script", "foreignobject", "iframe", "object", "embed", "image"}


def _validate_svg(svg_content: str) -> None:
    """Reject malformed SVG and content that can load or execute external data."""
    if not svg_content.strip():
        raise ValueError("SVG export content is empty.")
    encoded = svg_content.encode("utf-8")
    if len(encoded) > _MAX_SVG_BYTES:
        raise ValueError("SVG export exceeds the maximum supported size.")
    if "<!DOCTYPE" in svg_content.upper() or "<!ENTITY" in svg_content.upper():
        raise ValueError("SVG export must not contain DTDs or entities.")
    try:
        root = ET.fromstring(encoded)  # noqa: S314 — DTDs and entities are rejected and the input is size-capped above
    except ET.ParseError as exc:
        raise ValueError("SVG export content is malformed.") from exc
    if root.tag.rsplit("}", 1)[-1].lower() != "svg":
        raise ValueError("SVG export must have an SVG root element.")
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1].lower()
        if tag in _FORBIDDEN_SVG_TAGS:
            raise SecurityError(f"SVG export contains forbidden <{tag}> content.")
        for name, value in element.attrib.items():
            attribute = name.rsplit("}", 1)[-1].lower()
            normalized = value.strip().lower()
            if attribute.startswith("on") or attribute in {"href", "xlink:href"}:
                raise SecurityError("SVG export contains an executable or external reference.")
            if "url(" in normalized or "javascript:" in normalized or "data:" in normalized:
                raise SecurityError("SVG export contains an external resource reference.")


def write_svg(svg_content: str, output_path: Path) -> Path:
    """Write a validated SVG atomically to a non-symlink target."""
    _validate_svg(svg_content)
    atomic_write_bytes(output_path, svg_content.encode("utf-8"))
    logger.info("SVG written: %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
    return output_path


# How much real colour to keep around an object before the rest is cleared.
# Compositing that grows or blurs a matte reads the colour under the
# transparency; black there shows up as a dark fringe. Eight pixels costs
# about one percentage point of the saving and removes that risk.
COLOUR_BLEED_PIXELS = 8


def _clear_colour_far_from_the_object(
    rgba: NDArray[np.uint8], alpha: NDArray[np.uint8]
) -> NDArray[np.uint8]:
    """Zero the RGB channels well outside the mask, in place.

    A layer used to carry the whole source photograph in RGB and mask it
    only in alpha. LZW cannot compress a photograph, so a layer covering
    0.5% of the page still cost a full uncompressed frame -- the 9,779
    layers of the APPLE50 book came to 19 GB at a median coverage of
    0.46%. Clearing colour where nothing is visible leaves a flat field
    the compression collapses, and leaves the canvas, the position and
    the alpha exactly as they were.
    """
    if COLOUR_BLEED_PIXELS <= 0:
        return rgba
    span = 2 * COLOUR_BLEED_PIXELS + 1
    near = cv2.dilate(
        (alpha > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (span, span)),
    )
    rgba[..., :3][near == 0] = 0
    return rgba


def write_tiff(
    image: NDArray[np.uint8],
    output_path: Path,
    alpha: NDArray[np.uint8] | None = None,
    *,
    icc_profile: bytes | None = None,
) -> Path:
    """Write image as TIFF with optional alpha channel (4-channel RGBA)."""
    if image.ndim not in {2, 3}:
        raise ValueError("TIFF export image must be grayscale or RGB.")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("TIFF export image must have exactly three RGB channels.")
    if alpha is not None and alpha.shape != image.shape[:2]:
        raise ValueError("TIFF export alpha dimensions must match the image.")

    if alpha is not None:
        if len(image.shape) == 2:
            rgba = np.dstack([image, image, image, alpha])
        else:
            rgba = np.dstack([image, alpha])
        rgba = _clear_colour_far_from_the_object(rgba, alpha)
        pil_img = Image.fromarray(rgba, mode="RGBA")
    else:
        if len(image.shape) == 2:
            pil_img = Image.fromarray(image, mode="L")
        else:
            pil_img = Image.fromarray(image, mode="RGB")

    temporary = temporary_output_path(output_path)
    try:
        pil_img.save(str(temporary), format="TIFF", compression="tiff_lzw",
                     **({"icc_profile": icc_profile} if icc_profile else {}))
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    with Image.open(output_path) as decoded:
        expected_mode = "RGBA" if alpha is not None else ("L" if image.ndim == 2 else "RGB")
        if decoded.size != (image.shape[1], image.shape[0]) or decoded.mode != expected_mode:
            raise RuntimeError("TIFF export failed integrity validation.")
    logger.info("TIFF written: %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
    return output_path


def write_png(
    image: NDArray[np.uint8],
    output_path: Path,
    alpha: NDArray[np.uint8] | None = None,
) -> Path:
    """Write image as PNG with optional alpha channel."""
    if image.ndim not in {2, 3}:
        raise ValueError("PNG export image must be grayscale or RGB.")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("PNG export image must have exactly three RGB channels.")
    if alpha is not None and alpha.shape != image.shape[:2]:
        raise ValueError("PNG export alpha dimensions must match the image.")

    if alpha is not None:
        if len(image.shape) == 2:
            rgba = np.dstack([image, image, image, alpha])
        else:
            rgba = np.dstack([image, alpha])
        rgba = _clear_colour_far_from_the_object(rgba, alpha)
        pil_img = Image.fromarray(rgba, mode="RGBA")
    else:
        if len(image.shape) == 2:
            pil_img = Image.fromarray(image, mode="L")
        else:
            pil_img = Image.fromarray(image, mode="RGB")

    temporary = temporary_output_path(output_path)
    try:
        pil_img.save(str(temporary), format="PNG")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    with Image.open(output_path) as decoded:
        expected_mode = "RGBA" if alpha is not None else ("L" if image.ndim == 2 else "RGB")
        if decoded.size != (image.shape[1], image.shape[0]) or decoded.mode != expected_mode:
            raise RuntimeError("PNG export failed integrity validation.")
    logger.info("PNG written: %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
    return output_path


def write_pdf(svg_content: str, output_path: Path) -> Path:
    """Convert SVG to PDF via cairosvg."""
    _validate_svg(svg_content)
    cairosvg = load_cairosvg(logger)
    if cairosvg is None:
        raise RuntimeError(
            "PDF export requires CairoSVG and its native Cairo library. "
            "Install Cairo with 'brew install cairo' and restart Skiagrafia."
        )

    temporary = temporary_output_path(output_path)
    try:
        cairosvg.svg2pdf(
            bytestring=svg_content.encode("utf-8"), write_to=str(temporary), unsafe=False
        )
        if not temporary.read_bytes().startswith(b"%PDF"):
            raise RuntimeError("PDF export failed integrity validation.")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    logger.info("PDF written: %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
    return output_path
