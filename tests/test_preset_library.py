"""Bundled Apple preset is valid, complete and independent of user folders."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.preset_library import APPLE_SELECTION_REQUEST, load_apple_preset


def test_apple_preset_contains_required_vocabulary_and_exclusions() -> None:
    pack, request = load_apple_preset()
    objects = {obj.canonical for obj in pack.objects}

    assert pack.domain.name == "Apple — The First 50 Years"
    assert request == APPLE_SELECTION_REQUEST
    assert {
        "Apple computer",
        "Macintosh computer",
        "Apple II computer",
        "iMac",
        "iPhone",
        "iPad",
        "iPod",
        "Apple Watch",
        "AirPods",
        "Newton device",
        "keyboard",
        "computer mouse",
        "CRT monitor",
        "printer",
        "circuit board",
        "microchip",
        "prototype device",
        "product box",
        "Apple logo",
        "Steve Jobs",
        "Steve Wozniak",
        "Apple employee",
        "Apple Store",
        "Apple building",
    } <= objects
    assert {
        "captions",
        "headlines",
        "printed text",
        "diagrams",
        "charts",
        "screenshots",
        "page backgrounds",
        "decorative borders",
    } <= set(pack.domain.exclusions)
    imac = pack.find_object("iMac")
    assert imac is not None
    assert "screen" in imac.parts
    keyboard = pack.find_object("keyboard")
    assert keyboard is not None
    assert "buttons" in keyboard.parts
