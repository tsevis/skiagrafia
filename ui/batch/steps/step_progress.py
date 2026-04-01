from __future__ import annotations

import logging
import queue
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import TYPE_CHECKING

from ui.theme import is_macos

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ui.batch.batch_view import BatchView


class StepProgress:
    """Step 5 — Progress: metrics cards, progress bar, thumbnail strip."""

    def __init__(self, parent: tk.Widget, view: BatchView) -> None:
        self._view = view
        self._app = view.app
        self._root = view.root
        self._queue: queue.Queue = queue.Queue()
        self._total = 0
        self._completed = 0
        self._failed = 0
        self._start_time = 0.0
        self._successful_svgs = 0
        self._total_layers = 0

        self.frame = ttk.Frame(parent, padding=16)

        ttk.Label(
            self.frame,
            text="Processing",
            font=("SF Pro Display", 16, "bold"),
        ).pack(anchor=tk.W, pady=(0, 12))

        # Metric cards (3 across)
        metrics_frame = ttk.Frame(self.frame)
        metrics_frame.pack(fill=tk.X, pady=(0, 12))

        self._complete_card = self._metric_card(metrics_frame, "Complete", "0")
        self._complete_card.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))

        self._remaining_card = self._metric_card(metrics_frame, "Remaining", "0")
        self._remaining_card.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)

        self._speed_card = self._metric_card(metrics_frame, "Speed", "0 img/min")
        self._speed_card.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        # Global progress
        self._progress_label = ttk.Label(self.frame, text="0%  \u00b7  ETA: --")
        self._progress_label.pack(anchor=tk.W, pady=(0, 4))

        self._progress_bar = ttk.Progressbar(
            self.frame, mode="determinate", length=500
        )
        self._progress_bar.pack(fill=tk.X, pady=(0, 16))

        # Thumbnail strip
        ttk.Label(
            self.frame,
            text="Images",
            font=("SF Pro Text", 11, "bold"),
        ).pack(anchor=tk.W, pady=(0, 6))

        thumb_canvas = tk.Canvas(self.frame, height=56, highlightthickness=0)
        thumb_scrollbar = ttk.Scrollbar(
            self.frame, orient=tk.HORIZONTAL, command=thumb_canvas.xview
        )
        self._thumb_frame = ttk.Frame(thumb_canvas)
        self._thumb_frame.bind(
            "<Configure>",
            lambda e: thumb_canvas.configure(scrollregion=thumb_canvas.bbox("all")),
        )
        thumb_canvas.create_window((0, 0), window=self._thumb_frame, anchor=tk.NW)
        thumb_canvas.configure(xscrollcommand=thumb_scrollbar.set)

        thumb_canvas.pack(fill=tk.X)
        thumb_scrollbar.pack(fill=tk.X)

        self._thumb_labels: dict[str, tk.Label] = {}

        # Start Processing button
        self._start_btn = ttk.Button(
            self.frame,
            text="Start Processing",
            command=self._on_start,
        )
        self._start_btn.pack(anchor=tk.W, pady=(12, 0))

    def _on_start(self) -> None:
        """Collect images and confirmed labels, then launch batch processing."""
        self._start_btn.config(state="disabled", text="Processing...")

        # Get image paths from Step 1
        step_import = self._view._step_views[0]
        image_paths: list[str] = []
        if step_import and hasattr(step_import, "get_image_paths"):
            image_paths = step_import.get_image_paths()

        if not image_paths:
            self._start_btn.config(state="normal", text="Start Processing")
            self._progress_label.config(text="No images — go back to Step 1")
            return

        # Get confirmed labels from Triage (Step 4) or fallback to all
        confirmed_labels = getattr(self._view, "confirmed_labels", []) or None

        # Get config from Step 2
        step_configure = self._view._step_views[1]
        config: dict = {}
        if step_configure and hasattr(step_configure, "get_config"):
            config = step_configure.get_config()

        # Init thumbnails
        image_ids = [Path(p).stem for p in image_paths]
        self.init_thumbnails(image_ids)

        # Launch processing in background thread
        self._total = len(image_paths)
        self._completed = 0
        self._failed = 0
        self._successful_svgs = 0
        self._total_layers = 0
        self._start_time = time.time()

        import threading
        threading.Thread(
            target=self._process_batch,
            args=(image_paths, confirmed_labels, config),
            daemon=True,
        ).start()
        self._poll_progress()

    def _process_batch(
        self,
        image_paths: list[str],
        confirmed_labels: list[str] | None,
        config: dict,
    ) -> None:
        """Run orchestrator on each image (background thread)."""
        from core.factory import build_capabilities, build_knowledge_pack
        from core.orchestrator import Orchestrator

        output_dir = Path(
            self._app.prefs.get(
                "output_directory",
                str(Path.home() / "Desktop" / "skiagrafia_out"),
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        caps = build_capabilities(
            self._app.prefs,
            knowledge_pack_path=config.get("guide_path"),
        )
        orchestrator = Orchestrator(
            capabilities=caps,
            output_dir=output_dir,
            output_mode=config.get("output_mode", "vector+bitmap"),
            bilateral_d=int(self._app.prefs.get("bilateral_filter_d", 9)),
            box_threshold=float(self._app.prefs.get("sam_box_threshold", 0.35)),
            text_threshold=float(self._app.prefs.get("sam_text_threshold", 0.25)),
            knowledge_pack=build_knowledge_pack(config.get("guide_path")),
        )

        for i, path in enumerate(image_paths):
            img_id = Path(path).stem
            self._queue.put(("status", (img_id, "running")))
            try:
                result = orchestrator.process(path, confirmed_labels)
                status = "failed" if result.error else "complete"
            except Exception as exc:
                logger.error("Batch failed for %s: %s", path, exc)
                result = None
                status = "failed"

            if status == "failed":
                self._failed += 1
            else:
                if result and getattr(result, "svg_path", None):
                    self._successful_svgs += 1
                if result:
                    self._total_layers += len(getattr(result, "layers", []))
            self._completed += 1
            self._queue.put(("status", (img_id, status)))
            self._queue.put(("progress", (self._completed, self._total, self._failed)))

        self._queue.put(("done", None))

    def _poll_progress(self) -> None:
        """Poll the queue for progress updates from the worker thread."""
        try:
            while True:
                msg_type, data = self._queue.get_nowait()
                if msg_type == "status":
                    img_id, status = data
                    self.update_thumbnail_status(img_id, status)
                elif msg_type == "progress":
                    completed, total, failed = data
                    remaining = total - completed
                    elapsed = time.time() - self._start_time
                    speed = completed / (elapsed / 60) if elapsed > 0 else 0
                    eta = (remaining / speed * 60) if speed > 0 else 0
                    pct = int(completed / total * 100) if total > 0 else 0

                    self._complete_card._value_label.config(text=str(completed))
                    self._remaining_card._value_label.config(text=str(remaining))
                    self._speed_card._value_label.config(text=f"{speed:.1f} img/min")
                    self._progress_bar["maximum"] = total
                    self._progress_bar["value"] = completed
                    eta_min, eta_sec = int(eta // 60), int(eta % 60)
                    self._progress_label.config(
                        text=f"{pct}%  \u00b7  ETA: {eta_min}m {eta_sec}s"
                    )
                elif msg_type == "done":
                    self._start_btn.config(text="Complete", state="disabled")
                    self._progress_label.config(text="100%  \u00b7  Complete")

                    svg_count = self._successful_svgs
                    avg_layers = (
                        self._total_layers / svg_count if svg_count > 0 else 0.0
                    )
                    self._view.output_summary = {
                        "svg_count": svg_count,
                        "avg_layers": avg_layers,
                        "failed_count": self._failed,
                    }

                    # Update output step summary if the step is already instantiated.
                    step_output = self._view._step_views[5]
                    if step_output and hasattr(step_output, "update_summary"):
                        step_output.update_summary(svg_count, avg_layers, self._failed)

                    self._root.after(1000, lambda: self._view.go_next())
                    return
        except queue.Empty:
            pass
        self._root.after(200, self._poll_progress)

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

    def init_thumbnails(self, image_ids: list[str]) -> None:
        """Create placeholder thumbnails for all images."""
        for w in self._thumb_frame.winfo_children():
            w.destroy()
        self._thumb_labels.clear()

        for img_id in image_ids:
            lbl = tk.Label(
                self._thumb_frame,
                text="\u2014",
                bg="#3a3a3a",
                fg="gray",
                width=5,
                height=3,
                relief="solid",
                borderwidth=1,
            )
            lbl.pack(side=tk.LEFT, padx=1, pady=2)
            self._thumb_labels[img_id] = lbl

    def update_progress(self, progress: object) -> None:
        """Update metrics from a BatchProgress object."""
        total = getattr(progress, "total", 0)
        completed = getattr(progress, "completed", 0)
        failed = getattr(progress, "failed", 0)
        remaining = getattr(progress, "remaining", 0)
        speed = getattr(progress, "images_per_min", 0)
        eta = getattr(progress, "eta_seconds", 0)

        self._complete_card._value_label.config(text=str(completed))  # type: ignore[attr-defined]
        self._remaining_card._value_label.config(text=str(remaining))  # type: ignore[attr-defined]
        self._speed_card._value_label.config(text=f"{speed:.1f} img/min")  # type: ignore[attr-defined]

        if total > 0:
            pct = int(completed / total * 100)
            self._progress_bar["maximum"] = total
            self._progress_bar["value"] = completed
            eta_min = int(eta // 60)
            eta_sec = int(eta % 60)
            self._progress_label.config(
                text=f"{pct}%  \u00b7  ETA: {eta_min}m {eta_sec}s"
            )

    def update_thumbnail_status(
        self, image_id: str, status: str
    ) -> None:
        """Update a single thumbnail's appearance based on status."""
        lbl = self._thumb_labels.get(image_id)
        if lbl is None:
            return

        status_styles = {
            "pending": {"bg": "#3a3a3a", "fg": "gray", "highlightbackground": "#3a3a3a"},
            "running": {"bg": "#1a1a1a", "fg": "#007AFF", "highlightbackground": "#007AFF"},
            "complete": {"bg": "#1a1a1a", "fg": "#34C759", "highlightbackground": "#34C759"},
            "failed": {"bg": "#1a1a1a", "fg": "#FF453A", "highlightbackground": "#FF453A"},
        }
        style = status_styles.get(status, status_styles["pending"])
        text_map = {"pending": "\u2014", "running": "\u00b7", "complete": "\u2713", "failed": "\u2715"}
        lbl.config(text=text_map.get(status, "\u2014"), **style)

    def on_batch_complete(self) -> None:
        """Auto-advance to step 6 when batch is done."""
        self._root.after(500, lambda: self._view.go_next())
