"""setup_wizard.py  --  First-run setup dialog.

Shown at startup only when required pipeline components are missing
(fresh install from GitHub). Machines that already have all models and a
reachable VLM backend never see this window.

Downloads run on a background thread; UI updates are marshalled back to
the Tk main loop via ``after``.
"""
from __future__ import annotations

import logging
import threading
import tkinter as tk
import webbrowser
from tkinter import ttk
from typing import Any

from utils.bootstrap import (
    OLLAMA_INSTALL_URL,
    SetupStatus,
    check_setup,
    download_missing_weights,
    pull_ollama_model,
)

logger = logging.getLogger(__name__)

_STATUS_ICONS = {"ready": "✓", "missing": "✕"}


class SetupWizard:
    """One-window checklist: weights, backend, Ollama models."""

    def __init__(self, parent: tk.Misc, prefs: dict[str, Any]) -> None:
        self._prefs = prefs
        self._busy = False

        self._win = tk.Toplevel(parent)
        self._win.title("Skiagrafia — First-run setup")
        self._win.geometry("640x520")
        if isinstance(parent, (tk.Tk, tk.Toplevel)):
            self._win.transient(parent)

        header = ttk.Label(
            self._win,
            text=(
                "Some components the ML pipeline needs are not on this machine yet.\n"
                "Download them below, or point Preferences → Models at an existing library."
            ),
            justify=tk.LEFT,
            padding=12,
        )
        header.pack(anchor=tk.W)

        self._tree = ttk.Treeview(
            self._win,
            columns=("component", "size", "status"),
            show="headings",
            height=10,
        )
        self._tree.heading("component", text="Component")
        self._tree.heading("size", text="Approx. size")
        self._tree.heading("status", text="Status")
        self._tree.column("component", width=380)
        self._tree.column("size", width=100, anchor=tk.E)
        self._tree.column("status", width=90, anchor=tk.CENTER)
        self._tree.pack(fill=tk.BOTH, expand=True, padx=12)

        self._progress_label = ttk.Label(self._win, text="", padding=(12, 6))
        self._progress_label.pack(anchor=tk.W)
        self._progress = ttk.Progressbar(self._win, mode="determinate")
        self._progress.pack(fill=tk.X, padx=12)

        buttons = ttk.Frame(self._win, padding=12)
        buttons.pack(fill=tk.X)
        self._download_btn = ttk.Button(
            buttons, text="Download missing", command=self._start_downloads
        )
        self._download_btn.pack(side=tk.LEFT)
        ttk.Button(
            buttons,
            text="Get Ollama…",
            command=lambda: webbrowser.open(OLLAMA_INSTALL_URL),
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(buttons, text="Recheck", command=self._refresh).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(buttons, text="Continue anyway", command=self._win.destroy).pack(
            side=tk.RIGHT
        )

        self._refresh()

    # ── Checklist ────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        try:
            status: SetupStatus = check_setup(self._prefs)
        except Exception:
            logger.error("Setup check failed", exc_info=True)
            return
        for row in self._tree.get_children():
            self._tree.delete(row)
        for item in status.items:
            size = f"{item.approx_mb} MB" if item.approx_mb else "—"
            label = item.name if item.required else f"{item.name} (optional)"
            self._tree.insert(
                "",
                tk.END,
                values=(label, size, _STATUS_ICONS.get(item.status, "?")),
            )
        if status.complete:
            self._progress_label.config(
                text="✓ Everything required is ready — you can close this window."
            )

    # ── Downloads ────────────────────────────────────────────────────────

    def _start_downloads(self) -> None:
        if self._busy:
            return
        self._busy = True
        self._download_btn.config(state="disabled")
        threading.Thread(target=self._download_worker, daemon=True).start()

    def _download_worker(self) -> None:
        def _report(name: str, done: int, total: int | None) -> None:
            self._win.after(0, self._update_progress, name, done, total)

        try:
            failures = download_missing_weights(self._prefs, _report)

            # Pull missing required/recommended Ollama models when the
            # Ollama backend is selected and the server is reachable.
            status = check_setup(self._prefs)
            host = str(self._prefs.get("ollama_url", "http://localhost:11434"))
            for item in status.items:
                if item.kind != "ollama_model" or item.status == "ready":
                    continue
                model_name = item.detail

                def _pull_report(
                    text: str, done: int, total: int | None, _m: str = model_name
                ) -> None:
                    _report(f"{_m}: {text}", done, total)

                try:
                    pull_ollama_model(host, model_name, _pull_report)
                except Exception:
                    logger.error("Ollama pull failed: %s", model_name, exc_info=True)
                    failures.append(model_name)

            message = (
                "Setup finished — restart the scan when ready."
                if not failures
                else f"Finished with errors: {', '.join(failures)}"
            )
        except Exception:
            logger.error("Setup downloads failed", exc_info=True)
            message = "Setup failed — see log for details."
        self._win.after(0, self._finish, message)

    def _update_progress(self, name: str, done: int, total: int | None) -> None:
        if total:
            self._progress.config(mode="determinate", maximum=total, value=done)
            self._progress_label.config(
                text=f"Downloading {name} — {done / 1e6:.0f} / {total / 1e6:.0f} MB"
            )
        else:
            self._progress_label.config(text=f"Downloading {name}…")

    def _finish(self, message: str) -> None:
        self._busy = False
        self._download_btn.config(state="normal")
        self._progress_label.config(text=message)
        self._refresh()


def maybe_show_setup_wizard(parent: tk.Misc, prefs: dict[str, Any]) -> bool:
    """Open the wizard only when required components are missing.

    Returns True when the wizard was shown.
    """
    from utils.bootstrap import is_setup_complete

    if is_setup_complete(prefs):
        return False
    SetupWizard(parent, prefs)
    return True
