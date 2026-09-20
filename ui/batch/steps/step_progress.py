from __future__ import annotations

import logging
import queue
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import TYPE_CHECKING, cast

from PIL import Image, ImageOps, ImageTk

from ui.theme import is_macos

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from core.batch_runner import BatchConfig
    from ui.batch.batch_view import BatchView
    from ui.batch.steps.step_import import StepImport
    from ui.batch.steps.step_output import StepOutput


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
        self._all_objects_tiffs = 0
        self._total_layers = 0
        self._summary_svg_base = 0
        self._summary_all_objects_base = 0
        self._summary_layer_total_base = 0.0
        self._runner = None
        self._runner_finished = False

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
        # Tk does not retain PhotoImage instances itself.  Keep explicit
        # references for as long as the corresponding status cells exist.
        self._thumb_refs: dict[str, ImageTk.PhotoImage] = {}
        self._thumb_key_by_source: dict[str, str] = {}

        # Start Processing button
        self._start_btn = ttk.Button(
            self.frame,
            text="Start Processing",
            command=self._on_start,
        )
        self._start_btn.pack(anchor=tk.W, pady=(12, 0))

    def _on_start(self) -> None:
        """Freeze the approved batch, then hand processing to BatchRunner."""
        self._start_btn.config(state="disabled", text="Processing...")

        # Get image paths from Step 1
        step_import = self._view._step_views[0]
        image_paths: list[str] = []
        if step_import and hasattr(step_import, "get_image_paths"):
            image_paths = cast("StepImport", step_import).get_image_paths()

        if not image_paths:
            self._start_btn.config(state="normal", text="Start Processing")
            self._progress_label.config(text="No images — go back to Step 1")
            return

        try:
            batch_config = self._view.freeze_processing_config(image_paths)
        except (OSError, ValueError) as exc:
            self._start_btn.config(state="normal", text="Start Processing")
            self._progress_label.config(text=str(exc))
            self._view._bottom_bar.set_status("Batch needs review", "#FF9F0A")
            return
        self._start_processing(batch_config)

    def start_retry(self, image_paths: list[str]) -> None:
        """Retry exactly the failed inputs while preserving prior successes."""
        if not image_paths:
            return
        self.resume_existing()

    def resume_existing(self) -> None:
        """Continue a verified frozen manifest through the shared runner."""
        run = self._view.run_settings
        if run is None:
            self._progress_label.config(text="No verified batch is available to resume")
            return
        try:
            from core.batch_runner import BatchConfig
            from core.batch_session import load_resumable_batch

            resumable = load_resumable_batch(run.run_dir)
            if resumable is None:
                self._progress_label.config(
                    text="Cannot resume: saved inputs, guide, or state no longer match"
                )
                return
            batch_config = BatchConfig.model_validate(resumable.processing.config)
        except (OSError, ValueError) as exc:
            self._progress_label.config(text=f"Cannot resume batch: {exc}")
            return
        self._start_processing(batch_config)

    def _start_processing(
        self,
        batch_config: BatchConfig,
    ) -> None:
        """Initialize GUI state and launch the common durable BatchRunner."""
        from core.batch_runner import BatchRunner

        self._start_btn.config(state="disabled", text="Processing...")
        image_paths = list(getattr(batch_config, "input_images", []))
        self.init_thumbnails(image_paths)
        self._total = len(image_paths)
        self._runner_finished = False
        self._start_time = time.time()
        self._runner = BatchRunner(
            batch_config,
            progress_callback=lambda progress: self._queue.put(("runner_progress", progress)),
            completion_callback=lambda progress: self._queue.put(("runner_complete", progress)),
            job_callback=lambda _id, image_path, status: self._queue.put(
                ("runner_job", (image_path, status.value))
            ),
        )
        try:
            self._runner.start()
        except (OSError, ValueError, RuntimeError) as exc:
            self._runner.close()
            self._runner = None
            self._start_btn.config(state="normal", text="Start Processing")
            self._progress_label.config(text=f"Cannot start batch: {exc}")
            return
        self._poll_progress()

    def _poll_progress(self) -> None:
        """Poll the queue for progress updates from the worker thread."""
        try:
            while True:
                msg_type, data = self._queue.get_nowait()
                if msg_type == "runner_job":
                    image_path, status = data
                    self.update_thumbnail_status(image_path, status)
                elif msg_type == "runner_progress":
                    self.update_progress(data)
                    self._view._bottom_bar.set_progress(
                        getattr(data, "completed", 0) + getattr(data, "failed", 0),
                        max(getattr(data, "total", 0), 1),
                    )
                elif msg_type == "runner_complete":
                    self._finish_runner()
                    return
        except queue.Empty:
            pass
        self._root.after(200, self._poll_progress)

    def _finish_runner(self) -> None:
        """Render durable state metrics only after the common runner finishes."""
        if self._runner_finished or self._runner is None:
            return
        self._runner_finished = True
        summary = self._runner.summary()
        self._view.failed_image_paths = list(summary.failed_image_paths)
        self._view.output_summary = {
            "svg_count": summary.svg_count,
            "all_objects_count": summary.all_objects_count,
            "avg_layers": summary.avg_layers,
            "failed_count": summary.failed,
        }
        for image_path, status in summary.status_by_image.items():
            self.update_thumbnail_status(image_path, status)
        self._start_btn.config(text="Complete", state="disabled")
        self._progress_label.config(text="100%  \u00b7  Complete")
        self._view._bottom_bar.set_status("Batch complete", "#34C759")
        self._view._bottom_bar.set_progress(summary.completed + summary.failed, max(summary.total, 1))

        step_output = self._view._step_views[5]
        if step_output and hasattr(step_output, "update_summary"):
            cast("StepOutput", step_output).update_summary(
                summary.svg_count,
                summary.avg_layers,
                summary.failed,
                summary.all_objects_count,
            )
        self._runner.close()
        self._runner = None
        self._root.after(1000, lambda: self._view.go_next())

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

    def init_thumbnails(self, image_paths_or_ids: list[str]) -> None:
        """Create source-image thumbnails, with status-cell fallbacks.

        The string-only identifier form is retained for callers that do not
        have local files available yet (and for the safe empty-state view).
        """
        for w in self._thumb_frame.winfo_children():
            w.destroy()
        self._thumb_labels.clear()
        self._thumb_refs.clear()
        self._thumb_key_by_source.clear()

        for image_path_or_id in image_paths_or_ids:
            source_path = Path(image_path_or_id)
            img_id = source_path.stem if source_path.is_file() else image_path_or_id
            if img_id in self._thumb_labels:
                # A batch may legitimately contain a.jpg and a.png.  Keep
                # the established compact stem for the first cell while
                # giving later source paths an unambiguous status target.
                img_id = image_path_or_id
            photo = self._load_source_thumbnail(source_path) if source_path.is_file() else None
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
            if photo is not None:
                lbl.config(
                    image=photo,
                    text="\u2014",
                    compound=tk.CENTER,
                    width=0,
                    height=0,
                    highlightthickness=1,
                    highlightbackground="#3a3a3a",
                )
                self._thumb_refs[img_id] = photo
            lbl.pack(side=tk.LEFT, padx=1, pady=2)
            self._thumb_labels[img_id] = lbl
            if source_path.is_file():
                self._thumb_key_by_source[str(source_path)] = img_id

    @staticmethod
    def _load_source_thumbnail(source_path: Path) -> ImageTk.PhotoImage | None:
        """Render a small, orientation-correct source preview for progress."""
        try:
            with Image.open(source_path) as image:
                preview = ImageOps.exif_transpose(image).convert("RGB")
                preview.thumbnail((48, 48), Image.Resampling.LANCZOS)
        except (OSError, ValueError) as exc:
            logger.warning("Could not create batch thumbnail for %s: %s", source_path, exc)
            return None

        canvas = Image.new("RGB", (48, 48), "#3a3a3a")
        x = (canvas.width - preview.width) // 2
        y = (canvas.height - preview.height) // 2
        canvas.paste(preview, (x, y))
        return ImageTk.PhotoImage(canvas)

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
            done = completed + failed
            pct = int(done / total * 100)
            self._progress_bar["maximum"] = total
            self._progress_bar["value"] = done
            eta_min = int(eta // 60)
            eta_sec = int(eta % 60)
            self._progress_label.config(
                text=f"{pct}%  \u00b7  ETA: {eta_min}m {eta_sec}s"
            )

    def update_thumbnail_status(
        self, image_id: str, status: str
    ) -> None:
        """Update a single thumbnail's appearance based on status."""
        thumb_key = self._thumb_key_by_source.get(image_id, image_id)
        lbl = self._thumb_labels.get(thumb_key)
        if lbl is None:
            return

        status_styles = {
            "pending": {"bg": "#3a3a3a", "fg": "gray", "highlightbackground": "#3a3a3a"},
            "running": {"bg": "#1a1a1a", "fg": "#007AFF", "highlightbackground": "#007AFF"},
            "interrogating": {"bg": "#1a1a1a", "fg": "#007AFF", "highlightbackground": "#007AFF"},
            "masking": {"bg": "#1a1a1a", "fg": "#007AFF", "highlightbackground": "#007AFF"},
            "vectorizing": {"bg": "#1a1a1a", "fg": "#007AFF", "highlightbackground": "#007AFF"},
            "complete": {"bg": "#1a1a1a", "fg": "#34C759", "highlightbackground": "#34C759"},
            "failed": {"bg": "#1a1a1a", "fg": "#FF453A", "highlightbackground": "#FF453A"},
        }
        style = status_styles.get(status, status_styles["pending"])
        text_map = {
            "pending": "\u2014",
            "running": "\u00b7",
            "interrogating": "\u00b7",
            "masking": "\u00b7",
            "vectorizing": "\u00b7",
            "complete": "\u2713",
            "failed": "\u2715",
        }
        # Unpacking a generic dict[str, str] via **style makes pyright check
        # its values against every keyword tkinter's configure() accepts
        # (anchor, relief, ...), not just the three keys this dict has \u2014
        # naming them explicitly keeps the real (str-typed) keys intact.
        lbl.config(
            text=text_map.get(status, "\u2014"),
            bg=style["bg"],
            fg=style["fg"],
            highlightbackground=style["highlightbackground"],
        )

    def on_batch_complete(self) -> None:
        """Auto-advance to step 6 when batch is done."""
        self._root.after(500, lambda: self._view.go_next())
