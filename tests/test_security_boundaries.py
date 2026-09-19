"""Regression tests for filesystem, archive, SVG, and local-network boundaries."""
from __future__ import annotations

import io
import stat
import zipfile
from pathlib import Path

import numpy as np
import pytest

from core.batch_runner import BatchConfig
from models.vlm_client import LlamaCppVLMClient, OllamaVLMClient
from processors.output_writer import write_svg, write_tiff
from processors.vectorizer import assemble_svg
from utils.model_manager import ModelManager
from utils.security import SecurityError, safe_child_path, validate_loopback_url


def test_loopback_validator_rejects_remote_and_credentialed_urls() -> None:
    assert validate_loopback_url("http://localhost:11434/") == "http://localhost:11434"
    with pytest.raises(SecurityError):
        validate_loopback_url("https://example.com")
    with pytest.raises(SecurityError):
        validate_loopback_url("http://user:secret@127.0.0.1:8080")


def test_vlm_clients_reject_non_local_servers() -> None:
    with pytest.raises(SecurityError):
        LlamaCppVLMClient("http://192.0.2.10:8080")
    with pytest.raises(SecurityError):
        OllamaVLMClient("https://example.com", "test")


def test_output_path_rejects_traversal_and_target_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    with pytest.raises(SecurityError):
        safe_child_path(root, "../outside.svg")

    outside = tmp_path / "outside.svg"
    outside.write_text("keep")
    root.mkdir()
    target = root / "layer.svg"
    target.symlink_to(outside)
    with pytest.raises(SecurityError):
        write_svg("<svg/>", target)
    assert outside.read_text() == "keep"


def test_model_resolution_rejects_traversal(tmp_path: Path) -> None:
    manager = ModelManager(tmp_path / "models")
    with pytest.raises(SecurityError):
        manager.resolve("../../outside.bin")


def test_archive_extraction_rejects_traversal_and_symlinks(tmp_path: Path) -> None:
    traversal = io.BytesIO()
    with zipfile.ZipFile(traversal, "w") as archive:
        archive.writestr("root/../../outside.txt", "bad")
    with (
        zipfile.ZipFile(io.BytesIO(traversal.getvalue())) as archive,
        pytest.raises(SecurityError),
    ):
        ModelManager._extract_archive_safely(archive, tmp_path, "root")

    symlink = io.BytesIO()
    with zipfile.ZipFile(symlink, "w") as archive:
        entry = zipfile.ZipInfo("root/link")
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(entry, "../../outside")
    with (
        zipfile.ZipFile(io.BytesIO(symlink.getvalue())) as archive,
        pytest.raises(SecurityError),
    ):
        ModelManager._extract_archive_safely(archive, tmp_path, "root")


def test_svg_assembly_and_writer_reject_active_markup(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        assemble_svg(8, 8, [{"svg_data": '<script>alert(1)</script>'}])
    with pytest.raises(SecurityError):
        write_svg('<svg><image href="https://example.com/a.png"/></svg>', tmp_path / "bad.svg")


def test_tiff_writer_rejects_invalid_alpha_dimensions(tmp_path: Path) -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="dimensions"):
        write_tiff(image, tmp_path / "bad.tiff", np.zeros((7, 8), dtype=np.uint8))


def test_batch_id_cannot_escape_output_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Batch ID"):
        BatchConfig(
            input_folder=str(tmp_path), output_dir=str(tmp_path), confirmed_labels=[], batch_id="../escape"
        )
