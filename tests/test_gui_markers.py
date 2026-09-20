"""Deterministic coverage for the fast/full GUI test split."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import pytest_collection_modifyitems, pytest_ignore_collect


class _Marker:
    def __init__(self, name: str) -> None:
        self.name = name


class _MarkexprOption:
    def __init__(self, markexpr: str) -> None:
        self.markexpr = markexpr


class _Config:
    """Structurally satisfies conftest._ConfigLike without depending on it."""

    def __init__(self, markexpr: str) -> None:
        self.option = _MarkexprOption(markexpr)


class _Item:
    def __init__(self, fixtures: tuple[str, ...], *, fast: bool = False) -> None:
        self.fixturenames = fixtures
        self._fast = fast
        self.added_markers: list[str] = []

    def get_closest_marker(self, name: str) -> _Marker | None:
        return _Marker(name) if name == "gui" and self._fast else None

    def add_marker(self, name: str) -> None:
        self.added_markers.append(name)


def test_real_window_collection_separates_fast_smoke_from_full_integration() -> None:
    fast_window = _Item(("tk_root",), fast=True)
    full_window = _Item(("main_window",))
    non_window = _Item(("tmp_path",))

    pytest_collection_modifyitems(None, [fast_window, full_window, non_window])

    assert fast_window.added_markers == []
    assert full_window.added_markers == ["gui_integration"]
    assert non_window.added_markers == []


def test_fast_gui_collection_skips_unselected_test_modules() -> None:
    fast_config = _Config("gui")
    full_config = _Config("gui or gui_integration")
    smoke_module = Path("/repo/tests/test_gui_smoke.py")
    batch_module = Path("/repo/tests/test_gui_batch_view.py")
    single_module = Path("/repo/tests/test_gui_single_view.py")
    preferences_module = Path("/repo/tests/test_gui_preferences.py")

    assert not pytest_ignore_collect(smoke_module, fast_config)
    assert pytest_ignore_collect(batch_module, fast_config)
    assert pytest_ignore_collect(single_module, fast_config)
    assert pytest_ignore_collect(preferences_module, fast_config)
    assert not pytest_ignore_collect(preferences_module, full_config)
