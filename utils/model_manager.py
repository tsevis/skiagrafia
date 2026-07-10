"""model_manager.py  --  ModelManager class

User-configurable model directory, registry-based resolution, and download.

The registry covers every component the ML pipeline needs so a fresh install
can bootstrap itself (first-run setup wizard), while an existing machine with
the models already on disk is used as-is — nothing is re-downloaded or moved.

Registry entry kinds:
- "file"       -- single weight file fetched from a direct URL.
- "hf_files"   -- a directory of files fetched from a HuggingFace repo via
                  direct resolve URLs (no login, no hub cache involved).
- "github_zip" -- a source checkout restored from a GitHub archive zip.

Backward-compat shims (module-level functions) kept for transition period.
"""
from __future__ import annotations

import io
import logging
import shutil
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── Registry of known models ────────────────────────────────────────────────

REGISTRY: dict[str, dict[str, object]] = {
    "grounded-sam-2-source": {
        "kind": "github_zip",
        "subpath": "Grounded-SAM-2",
        "display_name": "Grounded-SAM-2 source (code + configs)",
        "url": (
            "https://github.com/IDEA-Research/Grounded-SAM-2/"
            "archive/refs/heads/main.zip"
        ),
        "zip_root": "Grounded-SAM-2-main",
        "approx_mb": 30,
    },
    "groundingdino_swint_ogc.pth": {
        "kind": "file",
        "subpath": "Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth",
        "display_name": "GroundingDINO SwinT-OGC",
        "url": (
            "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
            "v0.1.0-alpha/groundingdino_swint_ogc.pth"
        ),
        "approx_mb": 694,
    },
    "sam2.1_hiera_large.pt": {
        "kind": "file",
        "subpath": "Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt",
        "display_name": "SAM 2.1 Hiera Large",
        "url": (
            "https://dl.fbaipublicfiles.com/segment_anything_2/"
            "092824/sam2.1_hiera_large.pt"
        ),
        "approx_mb": 898,
    },
    "vitmatte-base-composition-1k": {
        "kind": "hf_files",
        "subpath": "vitmatte-base-composition-1k",
        "display_name": "VitMatte ViT-B Composition-1K",
        "url": "https://huggingface.co/hustvl/vitmatte-base-composition-1k",
        "hf_files": [
            "config.json",
            "preprocessor_config.json",
        ],
        # transformers loads either format; existing installs may have
        # only the .bin. Fresh downloads fetch the first entry.
        "hf_weight_alternatives": ["model.safetensors", "pytorch_model.bin"],
        "approx_mb": 380,
    },
}


def _entry_files(entry: dict[str, object], key: str = "hf_files") -> list[str]:
    """File list of an "hf_files" registry entry (empty when absent)."""
    files = entry.get(key, [])
    return [str(f) for f in files] if isinstance(files, list) else []


def _hf_dir_available(entry: dict[str, object], path: Path) -> bool:
    """True when all required files and at least one weight file exist."""
    if not path.is_dir():
        return False
    if not all((path / f).is_file() for f in _entry_files(entry)):
        return False
    alternatives = _entry_files(entry, "hf_weight_alternatives")
    if not alternatives:
        return True
    return any((path / f).is_file() for f in alternatives)


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
    approx_mb: int | None = None


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
            return self._models_dir / str(entry["subpath"])
        return self._models_dir / name

    def is_available(self, name: str) -> bool:
        """Check if model exists on disk (all expected files for hf_files)."""
        path = self.resolve(name)
        entry = REGISTRY.get(name)
        if entry and entry.get("kind") == "hf_files":
            return _hf_dir_available(entry, path)
        return path.exists()

    def missing(self) -> list[str]:
        """Names of registry entries not present on disk."""
        return [name for name in REGISTRY if not self.is_available(name)]

    def ensure(
        self,
        name: str,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> Path:
        """Download model to library if not already present. Returns path.

        Existing models are returned as-is — never re-downloaded or moved.
        """
        path = self.resolve(name)
        if self.is_available(name):
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

        kind = str(entry.get("kind", "file"))
        logger.info("Downloading %s (%s) -> %s", name, kind, path)
        if kind == "github_zip":
            self._download_github_zip(entry, path, progress_callback)
        elif kind == "hf_files":
            self._download_hf_files(entry, path, progress_callback)
        else:
            self._download_file(str(url), path, progress_callback)
        logger.info("Downloaded %s", name)
        return path

    # ── Download helpers ────────────────────────────────────────────────

    @staticmethod
    def _download_file(
        url: str,
        path: Path,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> None:
        """Fetch a single file from a direct URL with optional progress."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".part")

        def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
            if progress_callback is not None:
                progress_callback(
                    block_num * block_size,
                    total_size if total_size > 0 else None,
                )

        try:
            urllib.request.urlretrieve(url, tmp_path, reporthook=_reporthook)
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)

    @classmethod
    def _download_hf_files(
        cls,
        entry: dict[str, object],
        target_dir: Path,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> None:
        """Fetch a model directory file-by-file from HuggingFace resolve URLs."""
        base_url = str(entry["url"]).rstrip("/")
        files = list(_entry_files(entry))
        alternatives = _entry_files(entry, "hf_weight_alternatives")
        if alternatives and not any((target_dir / f).is_file() for f in alternatives):
            files.append(alternatives[0])
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in files:
            dest = target_dir / filename
            if dest.is_file():
                continue
            cls._download_file(
                f"{base_url}/resolve/main/{filename}", dest, progress_callback
            )

    @staticmethod
    def _download_github_zip(
        entry: dict[str, object],
        target_dir: Path,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> None:
        """Restore a source checkout from a GitHub archive zip."""
        url = str(entry["url"])
        zip_root = str(entry.get("zip_root", ""))

        request = urllib.request.Request(url)
        with urllib.request.urlopen(request, timeout=120) as response:
            total = response.headers.get("Content-Length")
            total_size = int(total) if total else None
            chunks: list[bytes] = []
            read = 0
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                chunks.append(chunk)
                read += len(chunk)
                if progress_callback is not None:
                    progress_callback(read, total_size)
        payload = b"".join(chunks)

        extract_parent = target_dir.parent
        extract_parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extractall(extract_parent)

        extracted = extract_parent / zip_root
        if zip_root and extracted.is_dir() and not target_dir.exists():
            extracted.rename(target_dir)
        elif zip_root and extracted.is_dir():
            # Merge into an existing partial checkout, then clean up
            shutil.copytree(extracted, target_dir, dirs_exist_ok=True)
            shutil.rmtree(extracted)

    def scan(self) -> list[ModelInfo]:
        """List all known models with their status."""
        results: list[ModelInfo] = []
        for name, entry in REGISTRY.items():
            subpath = str(entry.get("subpath", name))
            full_path = self._models_dir / subpath
            url = entry.get("url")
            manual = url is None

            if self.is_available(name):
                if full_path.is_file():
                    size = full_path.stat().st_size
                else:
                    # Directory-based model: sum all files
                    size = sum(
                        f.stat().st_size for f in full_path.rglob("*") if f.is_file()
                    )
                status = "ready"
            else:
                size = None
                status = "missing"

            approx = entry.get("approx_mb")
            results.append(
                ModelInfo(
                    name=name,
                    display_name=str(entry.get("display_name", name)),
                    subpath=subpath,
                    size_bytes=size,
                    status=status,
                    download_url=str(url) if url else None,
                    manual_install=manual,
                    approx_mb=approx if isinstance(approx, int) else None,
                )
            )
        return results


# ── Backward-compatibility shims ────────────────────────────────────────────
# Used by concrete model clients (grounded_sam, vitmatte_refiner)
# as fallback when no explicit path is passed to their constructors.
# Normal operation via the factory always passes explicit paths.

from utils.preferences import DEFAULT_MODELS_DIR  # noqa: E402

_default_manager: ModelManager | None = None

# Module-level constant for import by model clients that need a default dir.
# No eager mkdir -- ModelManager.__init__ handles directory creation.
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
