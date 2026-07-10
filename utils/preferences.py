from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path.home() / ".config" / "skiagrafia"

# Pre-existing shared model library (development machine). When present it is
# used as-is so models stay exactly where they are; fresh installs get a
# standard per-user app-data directory instead and can download into it.
_SHARED_MODELS_DIR = Path.home() / "ai" / "claudecode" / "mozaix" / "models"


def _default_models_dir() -> Path:
    """Resolve the default model library path for this machine."""
    if _SHARED_MODELS_DIR.is_dir():
        return _SHARED_MODELS_DIR
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "skiagrafia" / "models"
    return Path.home() / ".local" / "share" / "skiagrafia" / "models"


# Single source of truth for the default model library path.
DEFAULT_MODELS_DIR = _default_models_dir()

DEFAULT_PREFERENCES: dict[str, Any] = {
    # General
    "output_directory": str(Path.home() / "Desktop" / "skiagrafia_out"),
    "default_mode": "single",
    "save_session_on_quit": True,
    "show_notifications": True,
    # Models & VLM backends
    "vlm_backend": "ollama",  # "ollama" | "llamacpp"
    "ollama_url": "http://localhost:11434",
    "ollama_model": "qwen2.5vl:3b",
    "llamacpp_url": "http://localhost:8080",
    # llama.cpp serves whatever model it was launched with; this name is
    # informational (shown in logs / sent in the request "model" field).
    "llamacpp_model": "Qwen3-VL-8B-Instruct",
    "models_directory": "",  # empty = use default ~/ai/claudecode/mozaix/models
    "preferred_fallback_vlm": "gemma4:e4b",
    "preferred_text_reasoner": "gemma4:e4b",
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


# Saved values that were shipped as defaults in older versions and have a
# strictly better local replacement now. Only exact old-default values are
# migrated -- a user's deliberate custom choice is never touched.
_LEGACY_DEFAULT_UPGRADES: dict[str, dict[str, str]] = {
    "ollama_model": {"moondream": "qwen2.5vl:3b"},
    "preferred_fallback_vlm": {"minicpm-v": "gemma4:e4b"},
    "preferred_text_reasoner": {"qwen3.5": "gemma4:e4b"},
}


def _migrate_legacy_defaults(saved: dict[str, Any]) -> dict[str, Any]:
    """Upgrade old shipped-default model names to the current best local models."""
    migrated = dict(saved)
    for key, upgrades in _LEGACY_DEFAULT_UPGRADES.items():
        old_value = migrated.get(key)
        if isinstance(old_value, str) and old_value in upgrades:
            migrated[key] = upgrades[old_value]
            logger.info(
                "Preferences migration: %s '%s' -> '%s'",
                key, old_value, upgrades[old_value],
            )
    return migrated


def _prefs_path() -> Path:
    return _CONFIG_DIR / "preferences.json"


def load_preferences() -> dict[str, Any]:
    """Load preferences, merging saved values over defaults."""
    prefs = dict(DEFAULT_PREFERENCES)
    path = _prefs_path()
    if path.is_file():
        try:
            saved = json.loads(path.read_text())
            migrated = _migrate_legacy_defaults(saved)
            prefs.update(migrated)
            if migrated != saved:
                # Persist once so the migration doesn't re-run on every load
                try:
                    save_preferences(prefs)
                except OSError:
                    logger.debug("Could not persist migrated preferences", exc_info=True)
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
