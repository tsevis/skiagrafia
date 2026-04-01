from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk

from ui.theme import get_palette, is_macos
from ui.mode_switcher import ModeSwitcher
from utils.preferences import load_preferences, save_preferences

logger = logging.getLogger(__name__)


class MainWindow:
    """Main application window shell: titlebar, mode switcher, content area.

    Manages switching between Single and Batch views.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Skiagrafia")
        self._configure_default_window_size()
        self.root.minsize(1200, 760)

        self.prefs = load_preferences()
        self.palette = get_palette(self.prefs.get("theme"))
        self._appearance = "auto"

        # Configure ttk style
        self.style = ttk.Style()
        if is_macos():
            self.style.theme_use("aqua")
        else:
            self.style.theme_use("clam")

        # Build layout
        self._build_top_bar()
        self._build_content_area()

        # Lazy imports to avoid circular deps
        self._single_view: object | None = None
        self._batch_view: object | None = None

        # Show default mode
        default_mode = self.prefs.get("default_mode", "single")
        self._show_mode(default_mode)

    def _configure_default_window_size(self) -> None:
        """Open large and centered for a modern laptop display."""
        screen_w = max(self.root.winfo_screenwidth(), 1440)
        screen_h = max(self.root.winfo_screenheight(), 900)

        width = min(int(screen_w * 0.94), 1720)
        height = min(int(screen_h * 0.9), 1080)
        width = max(width, 1360)
        height = max(height, 860)

        offset_x = max((screen_w - width) // 2, 24)
        offset_y = max((screen_h - height) // 2, 24)
        self.root.geometry(f"{width}x{height}+{offset_x}+{offset_y}")

    def _build_top_bar(self) -> None:
        """Build the top bar: app name, mode switcher, dark mode, preferences."""
        self._top_bar = ttk.Frame(self.root)
        self._top_bar.pack(fill=tk.X, padx=12, pady=(8, 0))

        # Left: app name + subtitle
        title_frame = ttk.Frame(self._top_bar)
        title_frame.pack(side=tk.LEFT)

        self._title_label = ttk.Label(
            title_frame,
            text="Skiagrafia",
            font=("SF Pro Display", 16, "bold") if is_macos() else ("Segoe UI", 14, "bold"),
        )
        self._title_label.pack(anchor=tk.W)

        self._subtitle_label = ttk.Label(
            title_frame,
            text="Semantic vectorizing & masking creator",
            font=("SF Pro Text", 10) if is_macos() else ("Segoe UI", 9),
        )
        self._subtitle_label.pack(anchor=tk.W)

        # Right: Dark mode + Preferences buttons
        right_frame = ttk.Frame(self._top_bar)
        right_frame.pack(side=tk.RIGHT)

        prefs_btn = ttk.Button(
            right_frame,
            text="Preferences",
            command=self._open_preferences,
        )
        prefs_btn.pack(side=tk.RIGHT, padx=(4, 0))

        if is_macos():
            self._appearance_btn = ttk.Button(
                right_frame,
                text="Dark Mode",
                command=self._toggle_appearance,
            )
            self._appearance_btn.pack(side=tk.RIGHT, padx=(4, 0))

        # Centre: mode switcher
        self._mode_switcher = ModeSwitcher(
            self._top_bar,
            on_mode_change=self._on_mode_change,
        )
        self._mode_switcher.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Separator below top bar
        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(
            fill=tk.X, padx=0, pady=(8, 0)
        )

    def _build_content_area(self) -> None:
        """Build the main content area where views are swapped."""
        self._content = ttk.Frame(self.root)
        self._content.pack(fill=tk.BOTH, expand=True)

    def _on_mode_change(self, mode: str) -> None:
        """Handle mode switcher change."""
        self._show_mode(mode)

    def _show_mode(self, mode: str) -> None:
        """Show the appropriate view for the given mode."""
        # Clear content
        for child in self._content.winfo_children():
            child.pack_forget()

        if mode == "single":
            self._subtitle_label.config(
                text="Semantic vectorizing & masking creator"
            )
            self._show_single_view()
        else:
            self._subtitle_label.config(text="Batch semantic vectorizer")
            self._show_batch_view()

        self._mode_switcher.set_mode(mode)

    def _show_single_view(self) -> None:
        """Show the single image mode view."""
        if self._single_view is None:
            from ui.single.single_view import SingleView

            self._single_view = SingleView(self._content, self)
        self._single_view.frame.pack(fill=tk.BOTH, expand=True)

    def _show_batch_view(self) -> None:
        """Show the batch mode view."""
        if self._batch_view is None:
            from ui.batch.batch_view import BatchView

            self._batch_view = BatchView(self._content, self)
        self._batch_view.frame.pack(fill=tk.BOTH, expand=True)

    def _toggle_appearance(self) -> None:
        """Toggle dark/light mode on macOS."""
        try:
            if self._appearance in ("auto", "aqua"):
                self.root.tk.call(
                    "::tk::unsupported::MacWindowStyle",
                    "appearance",
                    ".",
                    "darkaqua",
                )
                self._appearance = "darkaqua"
                self._appearance_btn.config(text="Light Mode")
            else:
                self.root.tk.call(
                    "::tk::unsupported::MacWindowStyle",
                    "appearance",
                    ".",
                    "aqua",
                )
                self._appearance = "aqua"
                self._appearance_btn.config(text="Dark Mode")
        except tk.TclError:
            logger.warning("Dark mode toggle not supported", exc_info=True)

    def _open_preferences(self) -> None:
        """Open the preferences window."""
        from ui.preferences.preferences_window import PreferencesWindow

        PreferencesWindow(self)

    def apply_preferences(self, prefs: dict) -> None:
        """Apply updated preferences."""
        self.prefs = prefs
        self.palette = get_palette(prefs.get("theme"))

    @staticmethod
    def section_label(parent: tk.Widget, text: str) -> ttk.Label:
        """Create a bold section header label."""
        label = ttk.Label(
            parent,
            text=text,
            font=("SF Pro Text", 10, "bold") if is_macos() else ("Segoe UI", 9, "bold"),
        )
        return label

    def checkbox(
        self,
        parent: tk.Widget,
        text: str,
        variable: tk.BooleanVar,
        command: object = None,
    ) -> tk.Checkbutton | ttk.Checkbutton:
        """Create platform-appropriate checkbox."""
        if is_macos():
            return ttk.Checkbutton(
                parent, text=text, variable=variable, command=command
            )
        p = self.palette
        return tk.Checkbutton(
            parent,
            text=text,
            variable=variable,
            command=command,
            bg=p["entry_bg"],
            fg=p["entry_fg"],
            selectcolor=p["check_select"],
            activebackground=p["entry_bg"],
            activeforeground=p["entry_fg"],
        )

    def radio(
        self,
        parent: tk.Widget,
        text: str,
        variable: tk.StringVar,
        value: str,
        command: object = None,
    ) -> tk.Radiobutton | ttk.Radiobutton:
        """Create platform-appropriate radio button."""
        if is_macos():
            return ttk.Radiobutton(
                parent, text=text, variable=variable, value=value, command=command
            )
        p = self.palette
        return tk.Radiobutton(
            parent,
            text=text,
            variable=variable,
            value=value,
            command=command,
            bg=p["entry_bg"],
            fg=p["entry_fg"],
            selectcolor=p["check_select"],
            activebackground=p["entry_bg"],
            activeforeground=p["entry_fg"],
        )

    def switch_to_batch(self) -> None:
        """Switch to batch mode programmatically (from single mode template)."""
        self._show_mode("batch")
