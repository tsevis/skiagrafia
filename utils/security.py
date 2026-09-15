"""Small, dependency-free guards for local-only I/O boundaries.

These helpers deliberately validate at the boundary where untrusted strings
become URLs or filesystem paths.  They do not try to decide which user-owned
directory is appropriate; callers supply that root explicitly.
"""
from __future__ import annotations

import ipaddress
import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit


class SecurityError(ValueError):
    """Raised when external input violates an I/O safety boundary."""


def validate_loopback_url(url: str) -> str:
    """Return a normalized local HTTP URL or reject a non-local endpoint."""
    try:
        parsed = urlsplit(url.strip())
        port = parsed.port
    except ValueError as exc:
        raise SecurityError("The local model URL has an invalid port.") from exc

    if parsed.scheme not in {"http", "https"}:
        raise SecurityError("The local model URL must use HTTP or HTTPS.")
    if not parsed.hostname or parsed.username or parsed.password:
        raise SecurityError("The local model URL must not contain credentials.")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise SecurityError("The local model URL must be a server root URL.")
    if port is not None and not 1 <= port <= 65535:
        raise SecurityError("The local model URL has an invalid port.")

    hostname = parsed.hostname.rstrip(".").lower()
    if hostname != "localhost":
        try:
            if not ipaddress.ip_address(hostname).is_loopback:
                raise SecurityError("Model services must use a loopback address.")
        except ValueError as exc:
            raise SecurityError("Model services must use localhost or a loopback address.") from exc
    return parsed.geturl().rstrip("/")


def validate_download_url(url: str, allowed_hosts: set[str]) -> str:
    """Reject non-HTTPS or untrusted model-download endpoints."""
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise SecurityError("The model download URL has an invalid port.") from exc
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or not host
        or host not in allowed_hosts
        or parsed.username
        or parsed.password
        or parsed.fragment
        or (port is not None and port != 443)
    ):
        raise SecurityError("The model download URL is not an approved HTTPS source.")
    return parsed.geturl()


def safe_child_path(root: Path, filename: str) -> Path:
    """Resolve a flat output filename under *root* without symlink escape."""
    if not filename or "\x00" in filename:
        raise SecurityError("Output filename is empty or invalid.")
    component = Path(filename)
    if component.is_absolute() or component.name != filename or filename in {".", ".."}:
        raise SecurityError("Output filename must be a single relative path component.")

    root.mkdir(parents=True, exist_ok=True)
    canonical_root = root.resolve(strict=True)
    target = canonical_root / filename
    try:
        target.parent.resolve(strict=True).relative_to(canonical_root)
    except ValueError as exc:
        raise SecurityError("Output path escapes the configured output directory.") from exc
    if target.is_symlink():
        raise SecurityError("Refusing to overwrite a symlinked output file.")
    return target


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically replace a regular file, never following a target symlink."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise SecurityError("Refusing to overwrite a symlinked output file.")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if path.is_symlink():
            raise SecurityError("Refusing to overwrite a symlinked output file.")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def temporary_output_path(path: Path) -> Path:
    """Create a same-directory temporary filename for format writers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise SecurityError("Refusing to overwrite a symlinked output file.")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=path.suffix
    )
    os.close(descriptor)
    return Path(temporary_name)
