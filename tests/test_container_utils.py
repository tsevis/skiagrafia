"""test_container_utils.py  --  clearing a Tk container's children.

No window is constructed anywhere here: the children are MagicMocks built
with `spec=`, which satisfy isinstance against the real classes.
"""
from __future__ import annotations

import logging
import sys
import tkinter as tk
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui.container_utils import unpack_children


def _container(*children: object) -> MagicMock:
    container = MagicMock(spec=tk.Frame)
    container.winfo_children.return_value = list(children)
    return container


class TestUnpackChildren:
    def test_forgets_every_widget_child(self) -> None:
        first = MagicMock(spec=tk.Frame)
        second = MagicMock(spec=tk.Frame)

        unpack_children(_container(first, second))

        first.pack_forget.assert_called_once_with()
        second.pack_forget.assert_called_once_with()

    def test_a_child_with_no_geometry_manager_is_reported_not_skipped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The whole point of the guard.

        A Toplevel is a BaseWidget but not a Widget -- the window manager
        places it, so it has no pack_forget and calling one raises. Guarding
        the call is right; guarding it SILENTLY is not, because the container
        then fails to clear and nothing says why.
        """
        stray = MagicMock(spec=tk.Toplevel)

        with caplog.at_level(logging.WARNING, logger="ui.container_utils"):
            unpack_children(_container(stray))

        assert "Toplevel" in caplog.text
        assert caplog.records and caplog.records[0].levelno == logging.WARNING

    def test_one_unmanageable_child_does_not_stop_the_others(self) -> None:
        """Clearing is best-effort: one odd child must not leave a view up."""
        stray = MagicMock(spec=tk.Toplevel)
        after = MagicMock(spec=tk.Frame)

        unpack_children(_container(stray, after))

        after.pack_forget.assert_called_once_with()
