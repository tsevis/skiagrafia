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


def test_the_closed_apple_guide_admits_its_own_terms_and_refuses_others() -> None:
    pack = KnowledgePack.load(
        Path(__file__).resolve().parent.parent
        / "core" / "presets" / "apple_the_first_50_years_closed.toml"
    )

    assert pack.domain.closed_vocabulary
    assert pack.admits("shirt")
    assert pack.admits("hands"), "a plural should resolve to its singular"
    assert not pack.admits("trombone")
