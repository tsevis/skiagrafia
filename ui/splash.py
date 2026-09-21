"""The about window: what this is, what its name means, and who made it.

Shown once at launch and reachable afterwards from the Help menu.

The key art is the separation itself -- real layers this pipeline cut out of
one photograph, set down again with air between them. Not a decoration made
elsewhere: the program's own output, which is the only honest thing to put on
the front of it. The same picture is on the native macOS application.

This module opens a window, so nothing here may run during a plain test pass;
see `tests/test_splash_assets.py`, which checks the artwork without Tk.
"""
from __future__ import annotations

import contextlib
import logging
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import ImageTk

logger = logging.getLogger(__name__)

ASSETS = Path(__file__).resolve().parent / "assets"
STUDIO_URL = "https://tsevis.com"

KEY_ART = ASSETS / "SplashKeyArt.png"
STUDIO_MARK = ASSETS / "TVDLogo.png"
APP_ICON = ASSETS / "AppIcon.png"

#: Width of the window, and therefore of the key art above it.
WIDTH = 660
KEY_ART_HEIGHT = 208

EXPLANATION = (
    "Skiagrafia — σκιαγραφία — is the Greek for outlining: drawing a thing by "
    "its shadow. That is what this does. It reads one local photograph, finds "
    "the objects in it, and writes each one out as its own mask, matte and "
    "vector outline, so a picture becomes a stack you can edit rather than a "
    "single flat image.\n\n"
    "Everything happens on this machine. No image is uploaded and no service "
    "is called, and it will tell you when a check did not run rather than "
    "quietly giving you less than you asked for.\n\n"
    "The picture above is its own output: real layers cut from one photograph "
    "of a letterpress collage."
)


def _scaled(
    path: Path, width: int | None = None, height: int | None = None
) -> ImageTk.PhotoImage | None:
    """Load artwork at a drawn size, or return None if it is not there.

    Returns a Tk image the caller must keep a reference to: Tk does not own
    the bytes, and an image only referenced locally is collected and drawn as
    nothing at all.
    """
    try:
        from PIL import Image, ImageTk
    except ImportError:
        logger.info("Pillow unavailable; the about window will open without artwork")
        return None
    if not path.is_file():
        logger.warning("Missing about-window artwork: %s", path)
        return None
    try:
        image = Image.open(path).convert("RGBA")
    except OSError:
        logger.warning("Unreadable about-window artwork: %s", path, exc_info=True)
        return None
    if width and not height:
        height = max(1, round(image.height * width / image.width))
    elif height and not width:
        width = max(1, round(image.width * height / image.height))
    if width and height:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(image)


def apply_window_icon(root: tk.Tk) -> None:
    """Give the window the application icon, if it is present.

    Best effort by design: a missing or unreadable icon is a cosmetic loss and
    must never stop the application opening.
    """
    icon = _scaled(APP_ICON, width=512)
    if icon is None:
        return
    try:
        root.iconphoto(True, icon)  # type: ignore[arg-type]  # a PhotoImage is what Tk wants
    except tk.TclError:
        logger.info("Window manager refused the application icon", exc_info=True)
        return
    # Held on the widget: Tk keeps no strong reference of its own.
    root._skiagrafia_icon = icon  # type: ignore[attr-defined]


class SplashWindow:
    """The about window. Modal, dismissed by Continue or Escape."""

    def __init__(self, parent: tk.Tk, palette: dict[str, str] | None = None) -> None:
        self._palette = palette or {}
        self._images: list[object] = []

        self._win = tk.Toplevel(parent)
        self._win.title("About Skiagrafia")
        self._win.resizable(False, False)
        self._win.transient(parent)
        self._build()
        self._win.bind("<Escape>", lambda _event: self.close())
        self._centre(parent)
        self._win.grab_set()

    def _build(self) -> None:
        art = _scaled(KEY_ART, width=WIDTH)
        if art is not None:
            self._images.append(art)
            canvas = tk.Canvas(
                self._win, width=WIDTH, height=KEY_ART_HEIGHT,
                highlightthickness=0, bd=0,
            )
            canvas.pack(fill="x")
            canvas.create_image(0, 0, image=art, anchor="nw")
            self._draw_lockup(canvas)

        body = ttk.Frame(self._win, padding=(24, 18, 24, 0))
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=EXPLANATION, wraplength=WIDTH - 48, justify="left").pack(
            anchor="w"
        )

        ttk.Separator(self._win, orient="horizontal").pack(fill="x", pady=(18, 0))
        footer = ttk.Frame(self._win, padding=(24, 12))
        footer.pack(fill="x")
        mark = _scaled(STUDIO_MARK, height=22)
        if mark is not None:
            self._images.append(mark)
            corner = ttk.Label(footer, image=mark, cursor="pointinghand")
            corner.pack(side="left", padx=(0, 10))
            corner.bind("<Button-1>", lambda _event: self.open_studio())

        ttk.Label(
            footer, text="Created by Charis Tsevis, with the help of Claude Code."
        ).pack(side="left")
        link = ttk.Label(footer, text="tsevis.com", foreground="#3b7dd8", cursor="pointinghand")
        link.pack(side="left", padx=(12, 0))
        link.bind("<Button-1>", lambda _event: self.open_studio())
        ttk.Button(footer, text="Continue", command=self.close).pack(side="right")

    def _draw_lockup(self, canvas: tk.Canvas) -> None:
        """The studio mark bottom-left, with the name beside it.

        The mark is the link: clicking it opens tsevis.com. It sits on the
        same baseline as the title so the two read as one object.
        """
        mark_height = 42
        mark = _scaled(STUDIO_MARK, height=mark_height)
        left = 22
        baseline = KEY_ART_HEIGHT - 26
        if mark is not None:
            self._images.append(mark)
            item = canvas.create_image(left, baseline, image=mark, anchor="sw")
            canvas.tag_bind(item, "<Button-1>", lambda _event: self.open_studio())
            canvas.itemconfigure(item, tags=("studio-mark",))
            canvas.config(cursor="")
            left += mark.width() + 13

        canvas.create_text(
            left, baseline - 16, text="Skiagrafia", anchor="sw",
            fill="#ffffff", font=("Helvetica Neue", 30, "bold"),
        )
        canvas.create_text(
            left, baseline, text="Separates a photograph into real, editable layers.",
            anchor="sw", fill="#e8e8ea", font=("Helvetica Neue", 11),
        )

    def open_studio(self) -> None:
        webbrowser.open(STUDIO_URL)

    def _centre(self, parent: tk.Tk) -> None:
        self._win.update_idletasks()
        try:
            x = parent.winfo_rootx() + (parent.winfo_width() - self._win.winfo_width()) // 2
            y = parent.winfo_rooty() + (parent.winfo_height() - self._win.winfo_height()) // 3
        except tk.TclError:
            return
        self._win.geometry(f"+{max(0, x)}+{max(0, y)}")

    def close(self) -> None:
        with contextlib.suppress(tk.TclError):
            self._win.grab_release()
        self._win.destroy()


def makers_mark(parent: tk.Misc, height: int = 18) -> ttk.Label | None:
    """The studio mark as a clickable label, or None if the artwork is absent.

    Used by the main window's status bar as well as the about window, so the
    same mark and the same link appear in both.
    """
    image = _scaled(STUDIO_MARK, height=height)
    if image is None:
        return None
    label = ttk.Label(parent, image=image, cursor="pointinghand")
    # Tk keeps no strong reference to an image; the widget must hold it.
    label._skiagrafia_mark = image  # type: ignore[attr-defined]
    label.bind("<Button-1>", lambda _event: webbrowser.open(STUDIO_URL))
    return label
