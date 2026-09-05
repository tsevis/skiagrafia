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

# ── GUI fixtures ────────────────────────────────────────────────────────────
# Requesting any of these auto-marks the test `gui` (see the hook above), so
# it stays out of a plain `pytest` run. Run them deliberately: pytest -m gui


@pytest.fixture
def tk_root():
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
            try:
                root.after_cancel(after_id)
            except Exception:
                pass
        root.destroy()


@pytest.fixture
def stub_app(tk_root):
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
def single_view(tk_root, stub_app):
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
