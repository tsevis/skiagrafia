"""Shared pytest configuration.

Keeps window-opening tests out of the default run. A test that constructs a
real Tk window puts it on the developer's desktop, so it is normally marked
`gui_integration` automatically — based on its fixtures — and excluded by the
`addopts` in pyproject.toml. A deliberately small, explicitly marked `gui`
smoke set remains available for fast local feedback.
"""
from __future__ import annotations

import contextlib
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import pytest
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ui.batch.batch_view import BatchView
    from ui.main_window import MainWindow
    from ui.single.single_view import SingleView

# Any test requesting one of these fixtures builds a real window, so it is
# integration coverage unless an author deliberately marks it as fast `gui`.
GUI_FIXTURE_NAMES = frozenset({"tk_root", "gui_root", "gui_app", "main_window"})
FAST_GUI_MODULE_NAMES = frozenset({"test_gui_smoke.py"})


class _MarkerLike(Protocol):
    """The `pytest.Mark` surface `pytest_collection_modifyitems` reads."""

    name: str


class _ItemLike(Protocol):
    """The `pytest.Item` surface `pytest_collection_modifyitems` touches.

    Narrow on purpose so tests/test_gui_markers.py can exercise the hook
    against a lightweight fake instead of a real, expensively-collected
    pytest.Item -- both satisfy this structurally.
    """

    def get_closest_marker(self, name: str) -> _MarkerLike | None: ...
    def add_marker(self, name: str) -> None: ...


class _MarkexprOptionLike(Protocol):
    markexpr: str


class _ConfigLike(Protocol):
    """The `pytest.Config` surface `pytest_ignore_collect` touches.

    `option` is declared as a read-only property so structurally-similar but
    not identical `option` types (mutable-attribute protocol members are
    matched invariantly) still satisfy this Protocol.
    """

    @property
    def option(self) -> _MarkexprOptionLike: ...


def pytest_ignore_collect(collection_path: Path, config: _ConfigLike) -> bool:
    """Avoid importing the full test tree for the deliberately tiny smoke gate."""
    if config.option.markexpr.strip() != "gui":
        return False
    return (
        collection_path.parent.name == "tests"
        and collection_path.name.startswith("test_")
        and collection_path.name not in FAST_GUI_MODULE_NAMES
    )


def pytest_collection_modifyitems(
    config: _ConfigLike | None, items: list[_ItemLike]
) -> None:
    """Classify real-window tests as fast smoke or full integration coverage."""
    for item in items:
        fixtures = frozenset(getattr(item, "fixturenames", ()))
        if fixtures & GUI_FIXTURE_NAMES and item.get_closest_marker("gui") is None:
            item.add_marker("gui_integration")

# ── GUI fixtures ────────────────────────────────────────────────────────────
# Requesting any of these auto-marks the test `gui_integration` (see the hook
# above), so it stays out of a plain pytest run. Fast smoke tests explicitly
# use `@pytest.mark.gui`; run every windowed test with:
# pytest -m "gui or gui_integration".


@pytest.fixture
def tk_root() -> Iterator[tk.Tk]:
    """A real Tk root window. Destroyed even if the test fails."""
    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError as exc:  # no display available
        pytest.skip(f"Tk unavailable: {exc}")
    root.geometry("1100x760")
    root.update_idletasks()
    try:
        yield root
    finally:
        # Drop pending after() callbacks before teardown so a scheduled
        # redraw cannot fire against half-destroyed widgets.
        for after_id in root.tk.eval("after info").split():
            # The callback already fired or the id went stale; either way
            # there is nothing left to cancel.
            with contextlib.suppress(tk.TclError):
                root.after_cancel(after_id)
        root.destroy()


@pytest.fixture
def stub_app(tk_root) -> SimpleNamespace:
    """Minimal stand-in for MainWindow.

    The panels only ever reach for `root`, `prefs` and `switch_to_batch`
    (verified by grepping ui/single), so a real MainWindow -- which would
    build the top bar, mode switcher and Batch view too -- is unnecessary.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        root=tk_root,
        prefs={},
        switch_to_batch=lambda *a, **k: None,
    )


@pytest.fixture
def single_view(tk_root, stub_app) -> SingleView:
    """A real three-panel SingleView, laid out and realised."""
    import tkinter as tk
    from tkinter import ttk

    from ui.single.single_view import SingleView

    container = ttk.Frame(tk_root)
    container.pack(fill=tk.BOTH, expand=True)
    view = SingleView(container, stub_app)
    view.frame.pack(fill=tk.BOTH, expand=True)
    tk_root.update()  # realise geometry so the canvas has a real size
    return view


# ── Batch wizard fixtures ───────────────────────────────────────────────────
# Here rather than in a module the test files import, because a fixture is
# requested by NAME: an imported one is flagged as unused, and a test
# parameter of the same name is flagged as redefining it. conftest is how
# pytest shares a fixture, and it is where tk_root already lives.


def _write_image(path: Path) -> Path:
    """A tiny valid image on disk, for the steps that scan a folder."""
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(path)
    return path


@pytest.fixture
def batch_view(tk_root, tmp_path) -> BatchView:
    """A real BatchView with all filesystem access sandboxed to tmp_path."""
    from tkinter import ttk

    from ui.batch.batch_view import BatchView

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    app = SimpleNamespace(
        root=tk_root,
        prefs={"output_directory": str(output_dir)},
        switch_to_batch=lambda *a, **k: None,
    )
    container = ttk.Frame(tk_root)
    container.pack(fill=tk.BOTH, expand=True)
    # `app` is a SimpleNamespace exposing only the MainWindow surface BatchView
    # actually touches (root/prefs/switch_to_batch); cast for the type checker.
    view = BatchView(container, cast("MainWindow", app))
    view.frame.pack(fill=tk.BOTH, expand=True)
    tk_root.update()
    return view


@pytest.fixture
def image_folder(tmp_path) -> Path:
    """A folder holding three images and two files that must be ignored."""
    folder = tmp_path / "input"
    folder.mkdir()
    for name in ("c.png", "a.jpg", "b.tiff"):
        _write_image(folder / name)
    (folder / "notes.txt").write_text("not an image")
    (folder / "sub").mkdir()
    return folder
