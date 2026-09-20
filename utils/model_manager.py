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

import hashlib
import io
import logging
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from utils.security import SecurityError, safe_child_path, validate_download_url

logger = logging.getLogger(__name__)

_MODEL_DOWNLOAD_HOSTS = {
    "github.com",
    "dl.fbaipublicfiles.com",
    "huggingface.co",
}
_MAX_SOURCE_ARCHIVE_BYTES = 256 * 1024 * 1024
_HASH_CHUNK_BYTES = 1024 * 1024
_MAX_SOURCE_ARCHIVE_MEMBERS = 20_000

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
        # NO PIN IS POSSIBLE HERE, and that is a property of the URL. It
        # names refs/heads/main, so the archive changes whenever that branch
        # does; a digest recorded today would reject every later fetch. What
        # bounds this entry is _extract_archive_safely() plus the size and
        # member caps, not integrity.
        "sha256": None,
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
        # UPSTREAM PUBLISHES NO DIGEST. Checked 2026-09-20: the GitHub
        # releases API reports `digest: null` for this asset, and the
        # repository documents no checksum anywhere. A hash computed from a
        # copy already on a developer's disk would be trust-on-first-use
        # wearing the costume of an integrity check, so this stays unpinned
        # and honest about it.
        "sha256": None,
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
        # UPSTREAM PUBLISHES NO DIGEST. Checked 2026-09-20: Meta's own
        # checkpoints/download_ckpts.sh verifies nothing, and the only
        # server-side value is an S3 multipart ETag ("...-108"), which is a
        # hash of part hashes and cannot be compared against the file. Same
        # reasoning as the GroundingDINO entry above.
        "sha256": None,
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
        # transformers loads either format, and an existing install may
        # already carry a .safetensors from elsewhere, so both stay
        # acceptable. A FRESH DOWNLOAD FETCHES THE FIRST ENTRY, and
        # hustvl/vitmatte-base-composition-1k publishes no model.safetensors
        # at all -- its resolve URL answers 404. The .bin leads for that
        # reason, not as a preference.
        "hf_weight_alternatives": ["pytorch_model.bin", "model.safetensors"],
        # PUBLISHED BY UPSTREAM, not computed from this machine. The weight
        # digest is the sha256 the HuggingFace API reports for the LFS blob
        # (/api/models/<repo>?blobs=true); the two configs are not LFS, so
        # the API reports no digest and these are the sha256 of the bytes
        # their resolve URLs served. Read 2026-09-20.
        #
        # model.safetensors carries no pin: it does not exist upstream, so
        # there is nothing to pin it to. It is only ever accepted, never
        # fetched.
        "sha256": {
            "pytorch_model.bin": (
                "b2521bcc4b719fb24611c39605b6642162fd7502e69b3cc846506ca921757b41"
            ),
            "config.json": (
                "6c88774b9be97a236203be0d978b1ba8121bc6e585d10ca69ff3e90f921458be"
            ),
            "preprocessor_config.json": (
                "05b1234eb3f939dca65743521503302c8bace56d07d52b6c6e993104f216c85a"
            ),
        },
        "approx_mb": 380,
    },
}


def _entry_sha256(entry: dict[str, object], filename: str | None = None) -> str | None:
    """Declared digest for an entry, or for one file inside an "hf_files" one.

    Returns None when the entry declares no pin, which is not an error: see
    the registry comments for the three entries upstream publishes no digest
    for. A pin that is present is always enforced.
    """
    declared = entry.get("sha256")
    if filename is not None:
        if not isinstance(declared, dict):
            return None
        value = declared.get(filename)
        return str(value) if value else None
    return str(declared) if isinstance(declared, str) and declared else None


def _verify_sha256(path: Path, expected: str, label: str) -> None:
    """Raise SecurityError unless `path` hashes to `expected`.

    Reads in chunks: these files reach ~900 MB and the check must not need
    the whole payload in memory.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected.lower():
        raise SecurityError(
            f"Integrity check failed for {label}: expected sha256 {expected}, "
            f"got {actual}. The download was discarded."
        )


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
        self._models_dir = models_dir.expanduser()
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._models_dir = self._models_dir.resolve(strict=True)

    @property
    def models_dir(self) -> Path:
        """Read-only access to the configured model directory."""
        return self._models_dir

    def resolve(self, name: str) -> Path:
        """Resolve logical model name to absolute path via registry subpath."""
        entry = REGISTRY.get(name)
        relative = str(entry["subpath"]) if entry and "subpath" in entry else name
        candidate = (self._models_dir / relative).resolve(strict=False)
        try:
            candidate.relative_to(self._models_dir)
        except ValueError as exc:
            raise SecurityError("Model path escapes the configured model directory.") from exc
        return candidate

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
        validate_download_url(str(url), _MODEL_DOWNLOAD_HOSTS)
        logger.info("Downloading %s (%s) -> %s", name, kind, path)
        if kind == "github_zip":
            self._download_github_zip(entry, path, progress_callback)
        elif kind == "hf_files":
            self._download_hf_files(entry, path, progress_callback)
        else:
            self._download_file(
                str(url),
                path,
                progress_callback,
                expected_sha256=_entry_sha256(entry),
            )
        logger.info("Downloaded %s", name)
        return path

    # ── Download helpers ────────────────────────────────────────────────

    @staticmethod
    def _download_file(
        url: str,
        path: Path,
        progress_callback: Callable[[int, int | None], None] | None = None,
        expected_sha256: str | None = None,
    ) -> None:
        """Fetch a single file from a direct URL with optional progress.

        When `expected_sha256` is given the payload is verified while it is
        still the `.part` file, so a mismatch never reaches the destination
        name and the `finally` below removes it.
        """
        validate_download_url(url, _MODEL_DOWNLOAD_HOSTS)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".part")
        if path.is_symlink() or tmp_path.is_symlink():
            raise SecurityError("Refusing to write a model download through a symlink.")

        def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
            if progress_callback is not None:
                progress_callback(
                    block_num * block_size,
                    total_size if total_size > 0 else None,
                )

        try:
            urllib.request.urlretrieve(url, tmp_path, reporthook=_reporthook)  # noqa: S310 — validate_download_url() enforces HTTPS and the host allowlist
            if expected_sha256 is not None:
                _verify_sha256(tmp_path, expected_sha256, path.name)
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
        validate_download_url(base_url, _MODEL_DOWNLOAD_HOSTS)
        files = list(_entry_files(entry))
        alternatives = _entry_files(entry, "hf_weight_alternatives")
        if alternatives and not any((target_dir / f).is_file() for f in alternatives):
            files.append(alternatives[0])
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in files:
            dest = safe_child_path(target_dir, filename)
            if dest.is_file():
                continue
            cls._download_file(
                f"{base_url}/resolve/main/{filename}",
                dest,
                progress_callback,
                expected_sha256=_entry_sha256(entry, filename),
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
        validate_download_url(url, _MODEL_DOWNLOAD_HOSTS)
        if not zip_root or Path(zip_root).name != zip_root:
            raise SecurityError("Model archive has an unsafe expected root directory.")
        if target_dir.is_symlink():
            raise SecurityError("Refusing to extract a model archive through a symlink.")

        request = urllib.request.Request(url)  # noqa: S310 — validate_download_url() enforces HTTPS and the host allowlist
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 — validate_download_url() enforces HTTPS and the host allowlist
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
                if read > _MAX_SOURCE_ARCHIVE_BYTES:
                    raise SecurityError("Model source archive exceeds the safety size limit.")
        payload = b"".join(chunks)

        extract_parent = target_dir.parent
        extract_parent.mkdir(parents=True, exist_ok=True)
        with (
            zipfile.ZipFile(io.BytesIO(payload)) as archive,
            tempfile.TemporaryDirectory(
                prefix="skiagrafia-model-", dir=extract_parent
            ) as staging_name,
        ):
            staging = Path(staging_name)
            extracted = staging / zip_root
            ModelManager._extract_archive_safely(archive, staging, zip_root)
            if not extracted.is_dir():
                raise SecurityError("Model archive did not contain its expected root directory.")
            if target_dir.exists():
                # Preserve a partial user-owned checkout while only copying
                # the validated archive tree into the configured model root.
                shutil.copytree(extracted, target_dir, dirs_exist_ok=True)
            else:
                extracted.rename(target_dir)

    @staticmethod
    def _extract_archive_safely(
        archive: zipfile.ZipFile,
        destination: Path,
        expected_root: str,
    ) -> None:
        """Extract a zip without traversal, symlinks, zip bombs, or extra roots."""
        members = archive.infolist()
        if len(members) > _MAX_SOURCE_ARCHIVE_MEMBERS:
            raise SecurityError("Model source archive has too many files.")
        total_size = 0
        canonical_destination = destination.resolve(strict=True)
        for member in members:
            filename = member.filename.replace("\\", "/")
            parts = [part for part in filename.split("/") if part]
            if not parts or parts[0] != expected_root or any(part in {".", ".."} for part in parts):
                raise SecurityError("Model source archive contains an unsafe file path.")
            if filename.startswith("/") or ":" in parts[0]:
                raise SecurityError("Model source archive contains an absolute file path.")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise SecurityError("Model source archive contains a symbolic link.")
            total_size += member.file_size
            if total_size > _MAX_SOURCE_ARCHIVE_BYTES:
                raise SecurityError("Model source archive expands beyond the safety size limit.")
            target = (destination.joinpath(*parts)).resolve(strict=False)
            try:
                target.relative_to(canonical_destination)
            except ValueError as exc:
                raise SecurityError("Model source archive escapes its staging directory.") from exc
            if member.is_dir() or filename.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)

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

from utils.preferences import DEFAULT_MODELS_DIR  # noqa: E402 — shim kept below the class it backs

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
        except (OSError, RuntimeError, TypeError, ValueError):
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
