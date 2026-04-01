from __future__ import annotations

import platform
import sys
from typing import Any


def is_macos() -> bool:
    """Check if running on macOS."""
    return platform.system() == "Darwin"


# macOS system colour names — resolved at render time, auto-adapt to dark/light
MACOS_PALETTE: dict[str, str] = {
    "entry_bg": "systemTextBackgroundColor",
    "entry_fg": "systemTextColor",
    "listbox_bg": "systemTextBackgroundColor",
    "listbox_fg": "systemTextColor",
    "text_bg": "systemTextBackgroundColor",
    "text_fg": "systemTextColor",
    "highlight_border": "systemSeparatorColor",
    "accent": "systemControlAccentColor",
    "light_text": "systemSecondaryLabelColor",
    "warning_fg": "#FF6B6B",
    "info_fg": "systemLinkColor",
    "drop_highlight_bg": "systemFindHighlightColor",
    "check_select": "systemTextBackgroundColor",
    "select_bg": "systemSelectedTextBackgroundColor",
    "select_fg": "systemSelectedTextColor",
    "insert_bg": "systemTextColor",
}

# Fallback palette for non-macOS platforms
FALLBACK_PALETTE: dict[str, str] = {
    "entry_bg": "#FFFFFF",
    "entry_fg": "#1D1D1F",
    "listbox_bg": "#FFFFFF",
    "listbox_fg": "#1D1D1F",
    "text_bg": "#FFFFFF",
    "text_fg": "#1D1D1F",
    "highlight_border": "#D1D1D6",
    "accent": "#007AFF",
    "light_text": "#8E8E93",
    "warning_fg": "#FF6B6B",
    "info_fg": "#007AFF",
    "drop_highlight_bg": "#FFFF00",
    "check_select": "#FFFFFF",
    "select_bg": "#0063E1",
    "select_fg": "#FFFFFF",
    "insert_bg": "#1D1D1F",
}

# Layer colours — consistent across overlay borders, swatches, thumbnails
LAYER_COLOURS: dict[str, str] = {
    "parent_0": "#007AFF",  # macOS blue
    "parent_1": "#5856D6",  # macOS purple
    "child_0": "#34C759",   # macOS green
    "child_1": "#FF9F0A",   # macOS amber
    "child_2": "#FF453A",   # macOS red
    "child_3": "#30D158",   # macOS mint
    "child_4": "#64D2FF",   # macOS cyan
}

# Tag cloud colours for batch interrogation
TAG_COLOURS: dict[str, dict[str, str]] = {
    "parent": {"bg": "#E6F1FB", "fg": "#0C447C", "border": "#B5D4F4"},
    "child": {"bg": "#E1F5EE", "fg": "#085041", "border": "#9FE1CB"},
    "off": {"bg": "#F2F2F7", "fg": "#8E8E93", "border": "#D1D1D6"},
    "warn": {"bg": "#FFF3CD", "fg": "#7A4F00", "border": "#FAC775"},
}


def get_palette(override: str | None = None) -> dict[str, str]:
    """Return the appropriate colour palette.

    On macOS, always uses system colours (override is ignored).
    On other platforms, returns the fallback palette.
    """
    if is_macos():
        return dict(MACOS_PALETTE)
    return dict(FALLBACK_PALETTE)


def get_layer_colour(role: str, index: int) -> str:
    """Get colour for a layer by role and index.

    Args:
        role: "parent" or "child"
        index: 0-based index within the role group
    """
    key = f"{role}_{index}"
    if key in LAYER_COLOURS:
        return LAYER_COLOURS[key]
    # Cycle through available colours
    role_colours = [v for k, v in LAYER_COLOURS.items() if k.startswith(role)]
    if role_colours:
        return role_colours[index % len(role_colours)]
    return "#007AFF"
