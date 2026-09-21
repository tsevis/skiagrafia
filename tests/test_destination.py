"""test_destination.py  --  where finished work is written.

No window is constructed here. The destination was reachable only from the
Preferences window, which is not where anyone decides what a run produces;
this covers the plumbing the run-setup panels now use.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui.destination import DestinationError, choose_destination, current_destination


def _prefs(tmp_path: Path) -> dict[str, object]:
    return {"output_directory": str(tmp_path / "already-here")}


def test_a_chosen_destination_is_what_the_next_run_writes_to(tmp_path, monkeypatch) -> None:
    saved: list[dict[str, object]] = []
    monkeypatch.setattr("ui.destination.save_preferences", saved.append)
    prefs = _prefs(tmp_path)
    chosen = tmp_path / "chosen"
    chosen.mkdir()

    result = choose_destination(prefs, str(chosen))

    assert result == chosen
    assert current_destination(prefs) == chosen
    assert saved == [prefs], "a choice that is not written down is lost on quit"


def test_a_destination_that_cannot_be_written_to_is_refused(tmp_path, monkeypatch) -> None:
    # Otherwise the refusal arrives from the pipeline, after the wait.
    saved: list[dict[str, object]] = []
    monkeypatch.setattr("ui.destination.save_preferences", saved.append)
    prefs = _prefs(tmp_path)
    before = prefs["output_directory"]
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)

    try:
        with pytest.raises(DestinationError, match="written to"):
            choose_destination(prefs, str(locked))
    finally:
        locked.chmod(0o755)

    assert prefs["output_directory"] == before, "a refusal must not half-apply"
    assert saved == []


def test_a_file_where_a_folder_should_be_is_refused(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("ui.destination.save_preferences", lambda prefs: None)
    prefs = _prefs(tmp_path)
    not_a_folder = tmp_path / "file.txt"
    not_a_folder.write_text("x")

    with pytest.raises(DestinationError, match="not a folder"):
        choose_destination(prefs, str(not_a_folder))


def test_a_folder_that_does_not_exist_yet_is_accepted(tmp_path, monkeypatch) -> None:
    # The default destination is created on first use, so refusing a path
    # that is merely absent would refuse the default itself.
    monkeypatch.setattr("ui.destination.save_preferences", lambda prefs: None)
    prefs = _prefs(tmp_path)
    fresh = tmp_path / "not" / "there" / "yet"

    assert choose_destination(prefs, str(fresh)) == fresh


def test_a_folder_under_an_unwritable_parent_is_refused(tmp_path, monkeypatch) -> None:
    # It cannot be created when the run starts, so accepting it only defers
    # the failure to the moment work would otherwise have been saved.
    monkeypatch.setattr("ui.destination.save_preferences", lambda prefs: None)
    prefs = _prefs(tmp_path)
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)

    try:
        with pytest.raises(DestinationError, match="written to"):
            choose_destination(prefs, str(locked / "child"))
    finally:
        locked.chmod(0o755)
