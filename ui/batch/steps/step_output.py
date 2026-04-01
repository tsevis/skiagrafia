from __future__ import annotations

import logging
import subprocess
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import TYPE_CHECKING

from ui.theme import is_macos

if TYPE_CHECKING:
    from ui.batch.batch_view import BatchView

logger = logging.getLogger(__name__)


class StepOutput:
    """Step 6 — Output: summary metrics, output folder, export actions."""

    def __init__(self, parent: tk.Widget, view: BatchView) -> None:
        self._view = view
        self._app = view.app

        self.frame = ttk.Frame(parent, padding=16)

        ttk.Label(
            self.frame,
            text="Output Summary",
            font=("SF Pro Display", 16, "bold"),
        ).pack(anchor=tk.W, pady=(0, 12))

        # Metric cards (3 across)
        metrics_frame = ttk.Frame(self.frame)
        metrics_frame.pack(fill=tk.X, pady=(0, 12))

        self._svg_card = self._metric_card(metrics_frame, "SVG Files", "0")
        self._svg_card.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))

        self._layers_card = self._metric_card(metrics_frame, "Avg Layers", "0")
        self._layers_card.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)

        self._failed_card = self._metric_card(metrics_frame, "Failed", "0")
        self._failed_card.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        # Output folder
        ttk.Separator(self.frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

        folder_frame = ttk.Frame(self.frame)
        folder_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(folder_frame, text="Output folder:").pack(side=tk.LEFT)
        self._folder_label = ttk.Label(
            folder_frame,
            text=self._app.prefs.get(
                "output_directory",
                str(Path.home() / "Desktop" / "skiagrafia_out"),
            ),
            foreground="gray",
        )
        self._folder_label.pack(side=tk.LEFT, padx=(4, 0))

        ttk.Button(
            folder_frame,
            text="Reveal in Finder",
            command=self._reveal_in_finder,
        ).pack(side=tk.RIGHT)

        # Export buttons
        ttk.Separator(self.frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

        btn_frame = ttk.Frame(self.frame)
        btn_frame.pack(fill=tk.X)

        ttk.Button(
            btn_frame,
            text="Export SVG bundle",
            command=self._export_svg_bundle,
        ).pack(side=tk.LEFT, padx=(0, 4))

        ttk.Button(
            btn_frame,
            text="Export TIFF bundle",
            command=self._export_tiff_bundle,
        ).pack(side=tk.LEFT, padx=4)

        self._retry_btn = ttk.Button(
            btn_frame,
            text="Retry 0 failed",
            command=self._retry_failed,
            state="disabled",
        )
        self._retry_btn.pack(side=tk.LEFT, padx=(4, 0))

        self._apply_saved_summary()

    def _metric_card(
        self, parent: tk.Widget, title: str, value: str
    ) -> ttk.Frame:
        card = ttk.LabelFrame(parent, text=title, padding=8)
        label = ttk.Label(
            card,
            text=value,
            font=("SF Pro Display", 20, "bold") if is_macos() else ("Segoe UI", 18, "bold"),
        )
        label.pack()
        card._value_label = label  # type: ignore[attr-defined]
        return card

    def update_summary(
        self, svg_count: int, avg_layers: float, failed_count: int
    ) -> None:
        """Update summary metrics."""
        self._view.output_summary = {
            "svg_count": svg_count,
            "avg_layers": avg_layers,
            "failed_count": failed_count,
        }
        self._svg_card._value_label.config(text=str(svg_count))  # type: ignore[attr-defined]
        self._layers_card._value_label.config(text=f"{avg_layers:.1f}")  # type: ignore[attr-defined]
        self._failed_card._value_label.config(text=str(failed_count))  # type: ignore[attr-defined]

        if failed_count > 0:
            self._retry_btn.config(
                text=f"Retry {failed_count} failed", state="normal"
            )
        else:
            self._retry_btn.config(text="Retry 0 failed", state="disabled")

    def _apply_saved_summary(self) -> None:
        """Hydrate the view from the latest batch summary, if one exists."""
        summary = getattr(self._view, "output_summary", None) or {}
        self.update_summary(
            int(summary.get("svg_count", 0)),
            float(summary.get("avg_layers", 0.0)),
            int(summary.get("failed_count", 0)),
        )

    def _reveal_in_finder(self) -> None:
        output_dir = self._app.prefs.get(
            "output_directory",
            str(Path.home() / "Desktop" / "skiagrafia_out"),
        )
        if is_macos():
            subprocess.run(["open", output_dir], check=False)
        else:
            subprocess.run(["xdg-open", output_dir], check=False)

    def _export_svg_bundle(self) -> None:
        logger.info("Exporting SVG bundle")

    def _export_tiff_bundle(self) -> None:
        logger.info("Exporting TIFF bundle")

    def _retry_failed(self) -> None:
        logger.info("Retrying failed images")
