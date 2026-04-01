"""model_manager.py  --  v5.0 ModelManager class

User-configurable model directory, registry-based resolution, and download.
Backward-compat shims (module-level functions) kept for transition period.
"""
from __future__ import annotations

import logging
import urllib.request
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── Registry of known models ────────────────────────────────────────────────

REGISTRY: dict[str, dict[str, str]] = {
    "groundingdino_swint_ogc.pth": {
        "subpath": "Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth",
        "display_name": "GroundingDINO SwinT-OGC",
        "url": (
            "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
            "v0.1.0-alpha/groundingdino_swint_ogc.pth"
        ),
    },
    "sam2.1_hiera_large.pt": {
        "subpath": "Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt",
        "display_name": "SAM 2.1 Hiera Large",
        "url": (
            "https://dl.fbaipublicfiles.com/segment_anything_2/"
            "092824/sam2.1_hiera_large.pt"
        ),
    },
    "vitmatte-base-composition-1k": {
        "subpath": "vitmatte-base-composition-1k",
        "display_name": "VitMatte ViT-B Composition-1K",
        "url": "https://huggingface.co/hustvl/vitmatte-base-composition-1k",
    },
}


# ── ModelInfo Pydantic model ────────────────────────────────────────────────

class ModelInfo(BaseModel):
    """Scan result for a single model entry."""

    name: str
    display_name: str
    subpath: str
    size_bytes: int | None = None
    status: str  # "ready" | "missing" | "incomplete"
    download_url: str | None = None
    manual_install: bool = False


# ── ModelManager class ──────────────────────────────────────────────────────

class ModelManager:
    """User-configurable model manager.

    Owns model lifecycle: discovery, download, path resolution.
    The models_dir is passed in at construction -- never hardcoded.
    """

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir
        self._models_dir.mkdir(parents=True, exist_ok=True)

    @property
    def models_dir(self) -> Path:
        """Read-only access to the configured model directory."""
        return self._models_dir

    def resolve(self, name: str) -> Path:
        """Resolve logical model name to absolute path via registry subpath."""
        entry = REGISTRY.get(name)
        if entry and "subpath" in entry:
            return self._models_dir / entry["subpath"]
        return self._models_dir / name

    def is_available(self, name: str) -> bool:
        """Check if model exists on disk."""
        return self.resolve(name).exists()

    def ensure(
        self,
        name: str,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> Path:
        """Download model to library if not already present. Returns path."""
        path = self.resolve(name)
        if path.exists():
            return path

        entry = REGISTRY.get(name)
        if entry is None:
            raise KeyError(f"Unknown model '{name}'")
        url = entry.get("url")
        if not url:
            raise FileNotFoundError(
                f"Model '{name}' must be installed manually at {path}. "
                "This model does not support automatic download."
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s -> %s", name, path)

        if progress_callback is not None:
            def _reporthook(
                block_num: int, block_size: int, total_size: int,
            ) -> None:
                progress_callback(block_num * block_size, total_size if total_size > 0 else None)

            urllib.request.urlretrieve(url, path, reporthook=_reporthook)
        else:
            urllib.request.urlretrieve(url, path)

        logger.info("Downloaded %s (%.1f MB)", name, path.stat().st_size / 1e6)
        return path

    def scan(self) -> list[ModelInfo]:
        """List all known models with their status."""
        results: list[ModelInfo] = []
        for name, entry in REGISTRY.items():
            subpath = entry.get("subpath", name)
            full_path = self._models_dir / subpath
            url = entry.get("url")
            manual = url is None

            if full_path.exists():
                if full_path.is_file():
                    size = full_path.stat().st_size
                else:
                    # Directory-based model: sum all files
                    size = sum(f.stat().st_size for f in full_path.rglob("*") if f.is_file())
                status = "ready"
            else:
                size = None
                status = "missing"

            results.append(
                ModelInfo(
                    name=name,
                    display_name=entry.get("display_name", name),
                    subpath=subpath,
                    size_bytes=size,
                    status=status,
                    download_url=url,
                    manual_install=manual,
                )
            )
        return results


# ── Backward-compatibility shims ────────────────────────────────────────────
# Used by concrete model clients (grounded_sam, vitmatte_refiner)
# as fallback when no explicit path is passed to their constructors.
# Normal operation via the factory always passes explicit paths.

_default_manager: ModelManager | None = None

# Module-level constant for import by model clients that need a default dir.
# No eager mkdir -- ModelManager.__init__ handles directory creation.
from utils.preferences import DEFAULT_MODELS_DIR
MODELS_DIR = DEFAULT_MODELS_DIR


def _get_default() -> ModelManager:
    global _default_manager
    if _default_manager is None:
        try:
            from utils.preferences import get_models_dir
            models_dir = get_models_dir()
        except Exception:
            models_dir = DEFAULT_MODELS_DIR
        _default_manager = ModelManager(models_dir)
    return _default_manager


def model_path(name: str) -> Path:
    """Resolve model name to path. Used as fallback by model client constructors."""
    mgr = _get_default()
    path = mgr.resolve(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Model '{name}' not found at {path}. "
            "Use Preferences -> Models to download it."
        )
    return path
