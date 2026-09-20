"""container_utils.py  --  shared helpers for Tk container widgets.

One helper so far, used by both view shells to empty their content frame
before showing the next view. It lived twice, identically, which is how the
two copies would have drifted.
"""
from __future__ import annotations

import logging
import tkinter as tk

logger = logging.getLogger(__name__)


def unpack_children(container: tk.Misc) -> None:
    """Remove every child of `container` from its geometry manager.

    `winfo_children()` yields `Misc`, and a Toplevel really is one: it is a
    BaseWidget but NOT a Widget, because the window manager places it and it
    carries no pack/grid/place of its own. `pack_forget()` on one raises
    AttributeError.

    That does not happen with these containers, which hold frames. The case
    is nevertheless REPORTED rather than passed over: a silent skip would
    turn a structural surprise into a view that quietly failed to clear and
    then drew the next one on top of it. Clearing stays best-effort, so one
    odd child cannot leave the rest of the view on screen.
    """
    for child in container.winfo_children():
        if isinstance(child, tk.Widget):
            child.pack_forget()
        else:
            # %r as well as the class name: a Tk widget reprs as its own
            # path (".!frame.!toplevel"), which says WHICH child this was.
            # The class name alone would not.
            logger.warning(
                "Not clearing %s %r from %s: it has no geometry manager, so "
                "the container may still be showing it.",
                type(child).__name__,
                child,
                type(container).__name__,
            )
