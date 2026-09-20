"""Load oriented RGB and preserve any existing transparency."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps


def load_source_image(
    path: str | Path,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], bytes | None]:
    try:
        with Image.open(path) as source:
            source = ImageOps.exif_transpose(source)
            rgba = source.convert("RGBA")
            pixels = np.array(rgba)
            icc = source.info.get("icc_profile") if source.mode in {"RGB", "RGBA", "P"} else None
            return pixels[..., :3].copy(), pixels[..., 3].copy(), icc
    except (OSError, ValueError) as exc:
        raise FileNotFoundError(f"Cannot read image: {path}") from exc


def detection_image(rgb: NDArray[np.uint8], alpha: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Composite transparency over white for recognition, preserving source RGB for export."""
    if np.all(alpha == 255):
        return rgb
    weight = alpha[..., None].astype(np.float32) / 255
    return np.rint(rgb * weight + 255 * (1 - weight)).astype(np.uint8)
