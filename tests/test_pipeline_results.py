"""test_pipeline_results.py  --  how a run reports what it could not do."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pipeline_results import collapse_repeats


class TestCollapseRepeats:
    def test_identical_messages_are_counted_not_repeated(self) -> None:
        """A 231-image run produced the same line ten times for one image.

        Ten copies of "Empty or tiny mask for 'button'" say nothing that one
        copy and a count do not, and they bury the warnings that differ.
        """
        assert collapse_repeats(["a", "a", "a"]) == ["a (x3)"]

    def test_distinct_messages_all_survive_in_order(self) -> None:
        """Collapsing must never drop a warning that differs."""
        assert collapse_repeats(["b", "a", "b", "c"]) == ["b (x2)", "a", "c"]

    def test_a_single_message_is_left_exactly_as_it_was(self) -> None:
        """No count suffix when there is nothing to count."""
        assert collapse_repeats(["only one"]) == ["only one"]

    def test_empty_stays_empty(self) -> None:
        assert collapse_repeats([]) == []
