"""test_batch_given_nothing.py  --  a batch told to do nothing must not do everything.

Three faults found by running the 376-page APPLE50 set. Each is silent:
the run finishes, reports no failures, and produces the wrong thing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.batch_runner import BatchConfig, BatchRunner, _process_single


def _config(tmp_path: Path, **overrides: object) -> BatchConfig:
    source = tmp_path / "in"
    source.mkdir(exist_ok=True)
    for name in ("a.png", "b.png", "c.png"):
        (source / name).write_bytes(b"\x89PNG\r\n\x1a\n")
    defaults: dict = {
        "batch_id": "test-batch",
        "input_folder": str(source),
        "output_dir": str(tmp_path / "out"),
        "confirmed_labels": ["cat"],
    }
    defaults.update(overrides)
    return BatchConfig(**defaults)


def test_an_unfrozen_batch_still_discovers_the_folder(tmp_path: Path) -> None:
    runner = BatchRunner(_config(tmp_path))
    try:
        assert len(runner.discover_images()) == 3
    finally:
        runner.close()


def test_a_batch_frozen_to_no_images_refuses_instead_of_taking_the_folder(
    tmp_path: Path,
) -> None:
    # Triage that excluded every image means none, and the runner read it as
    # all: an empty list fell through to scanning the whole input folder.
    runner = BatchRunner(_config(tmp_path, input_images=[]))
    try:
        with pytest.raises(ValueError, match="no images"):
            runner.discover_images()
    finally:
        runner.close()


def test_an_image_with_no_confirmed_labels_is_refused_not_quietly_reduced(
    tmp_path: Path,
) -> None:
    # With an empty label list the interrogator returns nothing and the run
    # still writes an all-objects file: less than was asked for, silently.
    with pytest.raises(ValueError, match="No labels"):
        _process_single("page.png", {"confirmed_labels": [], "labels_by_image": {}})


def test_a_summary_can_still_be_read_after_the_runner_is_closed(
    tmp_path: Path,
) -> None:
    # Closing shuts the state database, and the summary is reconstructed
    # from it. Reading the result of a finished run is the ordinary thing
    # to do next, and it raised.
    runner = BatchRunner(_config(tmp_path))
    runner.discover_images()
    runner.close()

    assert runner.summary().total == 3
