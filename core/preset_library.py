"""Discovery and loading of bundled Batch-mode presets."""
from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from core.knowledge import KnowledgePack

APPLE_PRESET_ID = "apple-the-first-50-years"
APPLE_SELECTION_REQUEST = (
    "Select all clearly visible Apple products, computers, peripherals, prototypes, "
    "circuit boards, product boxes, logos, and people. Exclude book pages, captions, "
    "headlines, printed text, diagrams, charts, screenshots, photographs inside other "
    "images, decorative borders, and unrelated background scenery."
)


def preset_path(preset_id: str) -> Path:
    if preset_id != APPLE_PRESET_ID:
        raise ValueError(f"Unknown bundled preset: {preset_id}")
    return Path(str(files("core.presets").joinpath("apple_the_first_50_years.toml")))


def load_apple_preset() -> tuple[KnowledgePack, str]:
    path = preset_path(APPLE_PRESET_ID)
    return KnowledgePack.load(path), APPLE_SELECTION_REQUEST
