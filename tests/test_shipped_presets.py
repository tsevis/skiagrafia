"""test_shipped_presets.py  --  every guide shipped with the app must load.

A malformed preset is not caught by anything else: it parses at the
moment someone picks it, in the middle of setting up a run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.knowledge import KnowledgePack

PRESETS = sorted((Path(__file__).resolve().parent.parent / "core" / "presets").glob("*.toml"))


@pytest.mark.parametrize("path", PRESETS, ids=lambda p: p.stem)
def test_a_shipped_guide_loads_and_names_its_objects(path: Path) -> None:
    pack = KnowledgePack.load(path)

    assert pack.objects, f"{path.name} lists no objects"
    assert all(obj.canonical.strip() for obj in pack.objects)


@pytest.mark.parametrize("path", PRESETS, ids=lambda p: p.stem)
def test_no_shipped_guide_names_the_same_object_twice(path: Path) -> None:
    # Two objects with one canonical term make the second unreachable:
    # find_object returns the first and the rest of its aliases are dead.
    canonicals = [obj.canonical.strip().lower() for obj in KnowledgePack.load(path).objects]

    assert len(canonicals) == len(set(canonicals)), sorted(
        c for c in canonicals if canonicals.count(c) > 1
    )


def _closed_apple_guide() -> KnowledgePack:
    return KnowledgePack.load(
        Path(__file__).resolve().parent.parent
        / "core" / "presets" / "apple_the_first_50_years_closed.toml"
    )


def test_the_closed_apple_guide_refuses_a_term_it_does_not_list() -> None:
    pack = _closed_apple_guide()

    assert pack.domain.closed_vocabulary
    assert pack.admits("shirt")
    assert not pack.admits("trombone")


def test_a_plural_resolves_to_its_singular() -> None:
    found = _closed_apple_guide().find_object("cables")

    assert found is not None
    assert found.canonical == "cable"


def test_nobody_is_named_from_the_fact_that_they_are_a_person() -> None:
    # The guide lists Steve Jobs with "man" and "person" among the terms a
    # detector should search for. Those must not name him.
    pack = _closed_apple_guide()

    for observed in ("man", "woman", "people", "person"):
        found = pack.find_object(observed)
        assert found is not None, observed
        assert found.canonical == "person", f"{observed} named {found.canonical}"


def test_a_body_part_is_a_part_of_a_person_not_an_object_of_its_own() -> None:
    pack = _closed_apple_guide()
    person = pack.find_object("person")

    assert person is not None
    assert "hand" in person.parts
    assert not pack.admits("hand"), "a loose hand is not a thing to cut out"


def test_a_screen_and_a_monitor_stay_different_things() -> None:
    # A screen is a surface, a monitor is a device. The guide already
    # carries a CRT monitor; merging them would lose that.
    pack = _closed_apple_guide()
    screen = pack.find_object("computer screen")
    monitor = pack.find_object("monitor")

    assert screen is not None and screen.canonical == "screen"
    assert monitor is not None and monitor.canonical == "monitor"
