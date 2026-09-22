"""test_guide_term_roles.py  --  an alias names a thing, a generic term finds it.

A guide entry carries both. `aliases` are other names for the same thing:
"Woz" is Steve Wozniak. `generic_terms` and `detector_phrases` are what to
show a detector so it can locate the thing: to find Steve Jobs, look for a
person.

`find_object` matched all of them alike, so an observation of "man" or
"person" was canonicalised to "Steve Jobs". The APPLE50 book saw `man` 72
times and `person` 35 times. Every one of them would have been named as a
specific living-or-dead individual in a layer name and a file name, on the
evidence of there being a man in the photograph.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.knowledge import KnowledgeDomain, KnowledgePack, ObjectKnowledge

JOBS = ObjectKnowledge(
    canonical="Steve Jobs",
    aliases=["Steven Jobs"],
    generic_terms=["person", "man"],
    detector_phrases=["Steve Jobs", "person"],
)
PACK = KnowledgePack(domain=KnowledgeDomain(name="apple"), objects=[JOBS])


def test_a_generic_term_does_not_name_a_specific_person() -> None:
    assert PACK.find_object("man") is None
    assert PACK.find_object("person") is None


def test_a_detector_phrase_does_not_name_a_specific_person_either() -> None:
    # "person" is listed as a detector phrase too; that is how the thing is
    # found, not proof of who it is.
    assert PACK.find_object("Person") is None


def test_an_alias_is_still_a_name_for_the_same_thing() -> None:
    found = PACK.find_object("Steven Jobs")

    assert found is not None
    assert found.canonical == "Steve Jobs"


def test_the_canonical_term_still_resolves_whatever_its_case() -> None:
    found = PACK.find_object("steve jobs")

    assert found is not None
    assert found.canonical == "Steve Jobs"


def test_a_generic_term_is_still_offered_to_the_detector() -> None:
    # Removing it from naming must not remove it from finding.
    assert "person" in JOBS.ranked_detector_phrases(4)
