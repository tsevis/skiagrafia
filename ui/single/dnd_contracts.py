"""dnd_contracts.py  --  the tkinterdnd2 surface this project relies on.

tkinterdnd2 adds `drop_target_register` and `dnd_bind` onto widget objects
at runtime, and delivers an event carrying the dropped paths in `.data`.
None of that exists on the tkinter classes themselves, so a type checker
reports every use as an unknown attribute.

These Protocols name exactly what is used and nothing more. They are
structural, so nothing imports tkinterdnd2 to satisfy them, and the calls
stay inside the try/except that already handles the library being absent.
"""
from __future__ import annotations

from typing import Protocol


class DndTarget(Protocol):
    """A widget after tkinterdnd2 has registered it as a drop target."""

    def drop_target_register(self, *args: str) -> None: ...

    def dnd_bind(self, sequence: str, func: object) -> None: ...


class DndEvent(Protocol):
    """The drop event. `data` is a Tcl list of paths, brace-quoted."""

    data: str
