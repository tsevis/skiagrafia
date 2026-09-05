"""Shared pytest configuration.

Keeps window-opening tests out of the default run. A test that constructs a
real Tk window puts it on the developer's desktop, so such tests are marked
`gui` automatically — based on the fixtures they request — and excluded by
the `addopts` in pyproject.toml. Run them deliberately with `pytest -m gui`.
"""
from __future__ import annotations

import pytest

# Any test requesting one of these fixtures builds a real window, so it is
# marked `gui` without the author having to remember the marker.
GUI_FIXTURE_NAMES = frozenset({"tk_root", "gui_root", "gui_app", "main_window"})


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Mark every test that depends on a window-creating fixture as `gui`."""
    for item in items:
        fixtures = frozenset(getattr(item, "fixturenames", ()))
        if fixtures & GUI_FIXTURE_NAMES:
            item.add_marker("gui")
