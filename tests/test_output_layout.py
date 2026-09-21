"""test_output_layout.py  --  one folder per format under a run's root.

A run wrote every file it produced into one flat directory. For a single
image that is a handful; for a 376-image batch it is one folder holding
every SVG and every layer TIFF of every page at once.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.output_layout import collect_by_format, output_path


def test_each_format_is_written_to_its_own_folder(tmp_path: Path) -> None:
    svg = output_path(tmp_path, "page-abc123.svg")
    tiff = output_path(tmp_path, "page-abc123_layer-00-cat.tiff")

    assert svg == tmp_path / "svg" / "page-abc123.svg"
    assert tiff == tmp_path / "tiff" / "page-abc123_layer-00-cat.tiff"


def test_the_folder_exists_before_anything_tries_to_write_into_it(tmp_path: Path) -> None:
    output_path(tmp_path, "page.svg")

    assert (tmp_path / "svg").is_dir()


def test_a_path_that_escapes_the_run_is_refused(tmp_path: Path) -> None:
    from utils.security import SecurityError

    with pytest.raises(SecurityError):
        output_path(tmp_path, "../escaped.svg")


def test_the_export_bundle_finds_files_now_that_they_are_shelved(tmp_path: Path) -> None:
    # The bundle export globbed the run root directly. Shelving the files
    # without this would have turned every export into "Nothing to export".
    (tmp_path / "svg").mkdir()
    (tmp_path / "tiff").mkdir()
    (tmp_path / "svg" / "one.svg").write_text("<svg/>")
    (tmp_path / "svg" / "two.svg").write_text("<svg/>")
    (tmp_path / "tiff" / "one.tiff").write_bytes(b"II*\x00")

    found = collect_by_format(tmp_path, "svg")

    assert [p.name for p in found] == ["one.svg", "two.svg"]


def test_a_bundle_still_finds_the_flat_files_an_older_run_left(tmp_path: Path) -> None:
    # Runs finished before this change are still on disk and still openable.
    (tmp_path / "old.svg").write_text("<svg/>")

    assert [p.name for p in collect_by_format(tmp_path, "svg")] == ["old.svg"]


def test_a_symlinked_file_is_not_offered_for_export(tmp_path: Path) -> None:
    (tmp_path / "svg").mkdir()
    real = tmp_path / "real.svg"
    real.write_text("<svg/>")
    (tmp_path / "svg" / "link.svg").symlink_to(real)

    assert [p.name for p in collect_by_format(tmp_path, "svg")] == ["real.svg"]
