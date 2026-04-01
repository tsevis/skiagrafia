from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

logger = logging.getLogger(__name__)


def write_svg(svg_content: str, output_path: Path) -> Path:
    """Write SVG string to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg_content, encoding="utf-8")
    logger.info("SVG written: %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
    return output_path


def write_tiff(
    image: NDArray[np.uint8],
    output_path: Path,
    alpha: NDArray[np.uint8] | None = None,
) -> Path:
    """Write image as TIFF with optional alpha channel (4-channel RGBA)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if alpha is not None:
        if len(image.shape) == 2:
            rgba = np.dstack([image, image, image, alpha])
        else:
            rgba = np.dstack([image, alpha])
        pil_img = Image.fromarray(rgba, mode="RGBA")
    else:
        if len(image.shape) == 2:
            pil_img = Image.fromarray(image, mode="L")
        else:
            pil_img = Image.fromarray(image, mode="RGB")

    pil_img.save(str(output_path), format="TIFF", compression="tiff_lzw")
    logger.info("TIFF written: %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
    return output_path


def write_png(
    image: NDArray[np.uint8],
    output_path: Path,
    alpha: NDArray[np.uint8] | None = None,
) -> Path:
    """Write image as PNG with optional alpha channel."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if alpha is not None:
        if len(image.shape) == 2:
            rgba = np.dstack([image, image, image, alpha])
        else:
            rgba = np.dstack([image, alpha])
        pil_img = Image.fromarray(rgba, mode="RGBA")
    else:
        if len(image.shape) == 2:
            pil_img = Image.fromarray(image, mode="L")
        else:
            pil_img = Image.fromarray(image, mode="RGB")

    pil_img.save(str(output_path), format="PNG")
    logger.info("PNG written: %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
    return output_path


def write_pdf(svg_content: str, output_path: Path) -> Path:
    """Convert SVG to PDF via cairosvg."""
    import os
    import sys

    # Ensure cairocffi can find Homebrew's libcairo on macOS
    if sys.platform == "darwin":
        brew_lib = "/opt/homebrew/lib"
        if os.path.isdir(brew_lib):
            ld = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
            if brew_lib not in ld:
                os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
                    f"{brew_lib}:{ld}" if ld else brew_lib
                )

    import cairosvg

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cairosvg.svg2pdf(bytestring=svg_content.encode("utf-8"), write_to=str(output_path))
    logger.info("PDF written: %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
    return output_path
