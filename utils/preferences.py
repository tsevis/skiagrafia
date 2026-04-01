from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path.home() / ".config" / "skiagrafia"

# Single source of truth for the default model library path.
DEFAULT_MODELS_DIR = Path.home() / "ai" / "claudecode" / "mozaix" / "models"

DEFAULT_PREFERENCES: dict[str, Any] = {
    # General
    "output_directory": str(Path.home() / "Desktop" / "skiagrafia_out"),
    "default_mode": "single",
    "save_session_on_quit": True,
    "show_notifications": True,
    # Models & Ollama
    "ollama_url": "http://localhost:11434",
    "ollama_model": "moondream",
    "models_directory": "",  # empty = use default ~/ai/claudecode/mozaix/models
    "preferred_fallback_vlm": "minicpm-v",
    "preferred_text_reasoner": "qwen3.5",
    # Pipeline
    "sam_box_threshold": 0.35,
    "sam_text_threshold": 0.25,
    "vtracer_corner_threshold": 60,
    "vtracer_speckle": 8,
    "vtracer_length_threshold": 4.0,
    "bilateral_filter_d": 9,
    "max_cpu_workers": os.cpu_count() or 4,
    "interrogation_profile": "balanced",
    "interrogation_fallback_mode": "adaptive_auto",
    "enable_tiled_fallback": True,
    "enable_adaptive_interrogation": True,
    # Appearance
    "theme": "auto",
    "canvas_background": "#1a1a1a",
    "mask_overlay_opacity": 30,
    "vector_overlay_colour": "Blue",
    "scan_preview_show_boxes": True,
    "scan_preview_show_labels": True,
    "scan_preview_show_heatmap": True,
    "scan_preview_heatmap_opacity": 40,
    "scan_preview_box_opacity": 40,
}


def _prefs_path() -> Path:
    return _CONFIG_DIR / "preferences.json"


def load_preferences() -> dict[str, Any]:
    """Load preferences, merging saved values over defaults."""
    prefs = dict(DEFAULT_PREFERENCES)
    path = _prefs_path()
    if path.is_file():
        try:
            saved = json.loads(path.read_text())
            prefs.update(saved)
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to load preferences from %s, using defaults", path)
    else:
        # First run: persist defaults so the file exists for inspection/editing
        try:
            save_preferences(prefs)
            logger.info("Created default preferences at %s", path)
        except OSError:
            logger.debug("Could not write default preferences", exc_info=True)
    return prefs


def save_preferences(prefs: dict[str, Any]) -> None:
    """Write preferences to JSON config file."""
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = _prefs_path()
    path.write_text(json.dumps(prefs, indent=2))
    logger.info("Preferences saved to %s", path)


def get_models_dir(prefs: dict[str, Any] | None = None) -> Path:
    """Resolve the model library directory from preferences.

    Returns the user-configured path if set, otherwise the default
    ~/ai/claudecode/mozaix/models. Existing setups without the
    models_directory key behave identically to v4.0.
    """
    if prefs is None:
        prefs = load_preferences()
    custom = prefs.get("models_directory", "").strip()
    if custom:
        return Path(custom)
    return DEFAULT_MODELS_DIR
