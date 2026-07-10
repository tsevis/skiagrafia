"""Skiagrafia — Semantic vectorizing & masking creator.

Entry point: TkinterDnD root window with rich logging.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tkinter as tk

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


def check_vlm_backend() -> None:
    """Verify the configured VLM backend (Ollama or llama.cpp) at startup."""
    logger = logging.getLogger(__name__)
    try:
        from core.factory import build_interrogation_settings
        from models.vlm_client import create_vlm_client
        from utils.preferences import load_preferences

        prefs = load_preferences()
        settings = build_interrogation_settings(prefs)
        client = create_vlm_client(
            backend=settings.backend,
            host=settings.host,
            model=settings.primary_vlm,
        )
        if client.health_check():
            logger.info(
                "VLM backend '%s' connected — model '%s' ready",
                settings.backend,
                settings.primary_vlm,
            )
        else:
            logger.warning(
                "VLM backend '%s' has no model '%s' — scan will fail until resolved",
                settings.backend,
                settings.primary_vlm,
            )
    except Exception:
        logger.warning(
            "VLM backend not reachable — scan features unavailable", exc_info=True
        )


def check_first_run(root: "tk.Misc") -> None:
    """Open the setup wizard when required components are missing.

    The readiness probe does disk and network I/O, so it runs on a worker
    thread; the wizard itself is created back on the Tk main loop.
    """
    logger = logging.getLogger(__name__)
    import threading

    def _probe() -> None:
        try:
            from utils.bootstrap import is_setup_complete
            from utils.preferences import load_preferences

            prefs = load_preferences()
            if is_setup_complete(prefs):
                return

            def _open() -> None:
                from ui.setup_wizard import SetupWizard

                SetupWizard(root, prefs)
                logger.info("First-run setup wizard opened")

            root.after(0, _open)
        except Exception:
            logger.warning("First-run check failed", exc_info=True)

    threading.Thread(target=_probe, daemon=True).start()


def main() -> None:
    """Launch Skiagrafia."""
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Skiagrafia")

    try:
        from tkinterdnd2 import TkinterDnD  # type: ignore[import-untyped]

        root = TkinterDnD.Tk()
    except ImportError:
        logger.warning("tkinterdnd2 not available — drag-and-drop disabled")
        import tkinter as tk

        root = tk.Tk()

    # Check VLM backend + first-run setup in background (don't block UI startup)
    root.after(500, check_vlm_backend)
    root.after(900, check_first_run, root)

    from ui.main_window import MainWindow

    _app = MainWindow(root)  # noqa: F841 — must stay referenced for Tk callbacks

    logger.info("Skiagrafia ready")
    root.mainloop()


if __name__ == "__main__":
    main()
