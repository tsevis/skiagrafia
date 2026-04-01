"""Skiagrafia — Semantic vectorizing & masking creator.

Entry point: TkinterDnD root window with rich logging.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# 100% local inference — block ALL network downloads from HuggingFace/transformers.
# Must be set before any transformers/huggingface import.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Prevent transformers from importing TensorFlow/Flax (not needed, causes crashes).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

# Ensure Homebrew libcairo is discoverable by cairocffi/cairosvg on macOS.
if sys.platform == "darwin":
    _brew_lib = "/opt/homebrew/lib"
    if os.path.isdir(_brew_lib):
        _ld = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        if _brew_lib not in _ld:
            os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
                f"{_brew_lib}:{_ld}" if _ld else _brew_lib
            )

from rich.logging import RichHandler


def setup_logging() -> None:
    """Configure logging with RichHandler + file output."""
    log_dir = Path.home() / ".config" / "skiagrafia"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "skiagrafia.log"

    handlers: list[logging.Handler] = [
        RichHandler(
            level=logging.INFO,
            show_time=True,
            show_path=False,
            markup=True,
        ),
        logging.FileHandler(str(log_file), encoding="utf-8"),
    ]

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(name)s - %(message)s",
        handlers=handlers,
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def check_ollama() -> None:
    """Verify Ollama is reachable at startup."""
    logger = logging.getLogger(__name__)
    try:
        from models.moondream_client import MoondreamClient
        from utils.preferences import load_preferences

        prefs = load_preferences()
        client = MoondreamClient(
            host=prefs.get("ollama_url", "http://localhost:11434"),
            model=prefs.get("ollama_model", "moondream"),
        )
        if client.health_check():
            logger.info("Ollama connected — model ready")
        else:
            logger.warning("Ollama model not found — scan will fail until resolved")
    except Exception:
        logger.warning("Ollama not reachable — scan features unavailable", exc_info=True)


def main() -> None:
    """Launch Skiagrafia."""
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Skiagrafia")

    try:
        from tkinterdnd2 import TkinterDnD

        root = TkinterDnD.Tk()
    except ImportError:
        logger.warning("tkinterdnd2 not available — drag-and-drop disabled")
        import tkinter as tk

        root = tk.Tk()

    # Check Ollama in background (don't block UI startup)
    root.after(500, check_ollama)

    from ui.main_window import MainWindow

    app = MainWindow(root)

    logger.info("Skiagrafia ready")
    root.mainloop()


if __name__ == "__main__":
    main()
