from __future__ import annotations

import logging
import subprocess
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import TYPE_CHECKING, cast

from ui.theme import MACOS_OPEN, is_macos
from utils.security import SecurityError, atomic_write_bytes, safe_child_path

if TYPE_CHECKING:
    from ui.batch.batch_view import BatchView
    from ui.batch.steps.step_progress import StepProgress

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

        # Metric cards
        metrics_frame = ttk.Frame(self.frame)
        metrics_frame.pack(fill=tk.X, pady=(0, 12))

        self._svg_card = self._metric_card(metrics_frame, "SVG Files", "0")
        self._svg_card.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))

        self._all_objects_card = self._metric_card(metrics_frame, "Foreground TIFFs", "0")
        self._all_objects_card.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=3)

        self._layers_card = self._metric_card(metrics_frame, "Avg Layers", "0")
        self._layers_card.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=3)

        self._failed_card = self._metric_card(metrics_frame, "Failed", "0")
        self._failed_card.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0))

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
    ) -> ttk.LabelFrame:
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
        self,
        svg_count: int,
        avg_layers: float,
        failed_count: int,
        all_objects_count: int = 0,
    ) -> None:
        """Update summary metrics."""
        self._view.output_summary = {
            "svg_count": svg_count,
            "all_objects_count": all_objects_count,
            "avg_layers": avg_layers,
            "failed_count": failed_count,
        }
        self._svg_card._value_label.config(text=str(svg_count))  # type: ignore[attr-defined]
        self._all_objects_card._value_label.config(text=str(all_objects_count))  # type: ignore[attr-defined]
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
        self._folder_label.config(text=str(self._output_dir()))
        self.update_summary(
            int(summary.get("svg_count", 0)),
            float(summary.get("avg_layers", 0.0)),
            int(summary.get("failed_count", 0)),
            int(summary.get("all_objects_count", 0)),
        )

    def _output_dir(self) -> Path:
        """Return the run-specific output directory when a Batch run exists."""
        run = getattr(self._view, "run_settings", None)
        if run is not None:
            return Path(run.run_dir)
        return Path(
            self._app.prefs.get(
                "output_directory",
                str(Path.home() / "Desktop" / "skiagrafia_out"),
            )
        )

    def _reveal_in_finder(self) -> None:
        output_dir = self._output_dir()
        if is_macos():
            subprocess.run([MACOS_OPEN, str(output_dir)], check=False)  # noqa: S603 — fixed argv, no shell
        else:
            # No absolute path exists for xdg-open across distributions.
            subprocess.run(["xdg-open", str(output_dir)], check=False)  # noqa: S603,S607

    def _export_svg_bundle(self) -> None:
        self._export_bundle("*.svg", "SVG")

    def _export_tiff_bundle(self) -> None:
        self._export_bundle("*.tiff", "TIFF")

    def _export_bundle(self, pattern: str, format_name: str) -> None:
        """Copy generated files to a user-selected folder without symlink writes."""
        from tkinter import filedialog, messagebox

        source_dir = self._output_dir()
        files = [
            path for path in sorted(source_dir.glob(pattern))
            if path.is_file() and not path.is_symlink()
        ]
        if not files:
            messagebox.showinfo(
                "Nothing to export",
                f"No {format_name} files are available in this Batch output.",
                parent=self.frame.winfo_toplevel(),
            )
            return

        destination_text = filedialog.askdirectory(
            parent=self.frame.winfo_toplevel(),
            title=f"Export {format_name} bundle",
            initialdir=str(source_dir),
        )
        if not destination_text:
            return

        destination = Path(destination_text)
        copied = 0
        failures: list[str] = []
        for source in files:
            try:
                target = safe_child_path(destination, source.name)
                atomic_write_bytes(target, source.read_bytes())
                copied += 1
            except (OSError, SecurityError) as exc:
                logger.warning("Could not export %s: %s", source, exc)
                failures.append(source.name)

        if failures:
            messagebox.showwarning(
                "Export incomplete",
                f"Copied {copied} {format_name} file(s); could not export {len(failures)}.",
                parent=self.frame.winfo_toplevel(),
            )
        else:
            messagebox.showinfo(
                "Export complete",
                f"Copied {copied} {format_name} file(s) to {destination}.",
                parent=self.frame.winfo_toplevel(),
            )

    def _retry_failed(self) -> None:
        failed_paths = list(getattr(self._view, "failed_image_paths", []))
        if not failed_paths:
            self._retry_btn.config(text="Retry 0 failed", state="disabled")
            return
        self._view.go_to_step(4)
        progress_step = self._view._step_views[4]
        if progress_step and hasattr(progress_step, "start_retry"):
            cast("StepProgress", progress_step).start_retry(failed_paths)
