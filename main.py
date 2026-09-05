"""Skiagrafia — Semantic vectorizing & masking creator.

Entry point: TkinterDnD root window with rich logging.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import TYPE_CHECKING

from utils.cairo_support import configure_cairo_library_path

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
configure_cairo_library_path()

from rich.logging import RichHandler  # noqa: E402 — must follow configure_cairo_library_path()


# Log rotation — the file handler runs at DEBUG, so an unbounded file grows
# without limit. Cap total on-disk history at LOG_BACKUP_COUNT + 1 files.
LOG_MAX_BYTES = 2 * 1024 * 1024
LOG_BACKUP_COUNT = 5

# Third-party loggers that emit one record per HTTP frame or per draw call.
# At DEBUG these drown out the pipeline's own records, so they are pinned to
# WARNING; raise an individual one temporarily when debugging that layer.
NOISY_LOGGERS = ("PIL", "urllib3", "httpcore", "httpx", "matplotlib")


def build_file_handler(log_file: Path) -> logging.Handler:
    """Create the size-rotating file handler for the debug log."""
    return logging.handlers.RotatingFileHandler(
        str(log_file),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )


def setup_logging() -> None:
    """Configure logging with RichHandler + rotating file output."""
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
        build_file_handler(log_file),
    ]

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(name)s - %(message)s",
        handlers=handlers,
    )
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


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
