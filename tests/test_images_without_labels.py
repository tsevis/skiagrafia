"""test_images_without_labels.py  --  an image triage left with nothing.

Triage intersects the approved labels with each image's own candidates, so
an image can come out of it with an empty list. That list was frozen into
the run anyway. The image then failed at processing time, one worker and
one model load later, for something knowable before the run started.

This is the fourth appearance of one pattern: an empty collection standing
in silently for something it is not. The others were an empty frozen image
list meaning the whole folder, an empty confirmed-label list meaning "the
user confirmed nothing", and an empty matte written as a layer file.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.batch_session import partition_by_labelled


def test_images_with_labels_are_kept_in_order() -> None:
    labelled, bare = partition_by_labelled(
        {"a.png": ["screen"], "b.png": ["keyboard", "cable"]}
    )

    assert labelled == ["a.png", "b.png"]
    assert bare == []


def test_an_image_triage_left_bare_is_separated_out() -> None:
    labelled, bare = partition_by_labelled(
        {"a.png": ["screen"], "b.png": [], "c.png": ["cable"]}
    )

    assert labelled == ["a.png", "c.png"]
    assert bare == ["b.png"]


def test_everything_bare_is_reported_as_such() -> None:
    # Not an empty run: a run with nothing to do, which the caller must be
    # able to tell apart from a run it simply has not configured yet.
    labelled, bare = partition_by_labelled({"a.png": [], "b.png": []})

    assert labelled == []
    assert bare == ["a.png", "b.png"]


def test_nothing_at_all_gives_two_empty_lists() -> None:
    assert partition_by_labelled({}) == ([], [])
