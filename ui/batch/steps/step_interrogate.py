from __future__ import annotations

import logging
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

from ui.theme import TAG_COLOURS

if TYPE_CHECKING:
    from ui.batch.batch_view import BatchView

logger = logging.getLogger(__name__)


class StepInterrogate:
    """Step 3 — Interrogate: progress bar + live tag cloud."""

    def __init__(self, parent: tk.Widget, view: BatchView) -> None:
        self._view = view
        self._app = view.app
        self._root = view.root
        self._queue: queue.Queue = queue.Queue()
        self._tags: dict[str, dict] = {}

        self.frame = ttk.Frame(parent, padding=16)

        ttk.Label(
            self.frame,
            text="Interrogating Images",
            font=("SF Pro Display", 16, "bold"),
        ).pack(anchor=tk.W, pady=(0, 8))

        # Progress
        self._progress_label = ttk.Label(
            self.frame, text="0 / 0 \u00b7 Waiting to start..."
        )
        self._progress_label.pack(anchor=tk.W, pady=(0, 4))

        self._progress_bar = ttk.Progressbar(
            self.frame, mode="determinate", length=400
        )
        self._progress_bar.pack(fill=tk.X, pady=(0, 8))

        # Start button
        self._start_btn = ttk.Button(
            self.frame,
            text="Start Analysis",
            command=self._on_start,
        )
        self._start_btn.pack(anchor=tk.W, pady=(0, 12))

        # Tag cloud
        ttk.Label(
            self.frame,
            text="Detected Labels",
            font=("SF Pro Text", 11, "bold"),
        ).pack(anchor=tk.W, pady=(0, 6))

        self._tag_cloud = ttk.Frame(self.frame)
        self._tag_cloud.pack(fill=tk.BOTH, expand=True)

        # Legend
        legend = ttk.Frame(self.frame)
        legend.pack(fill=tk.X, pady=(12, 0))
        for role, desc in [
            ("parent", "Blue = parent"),
            ("child", "Green = child"),
            ("off", "Gray = background"),
            ("warn", "Amber = rare"),
        ]:
            colours = TAG_COLOURS[role]
            tk.Label(
                legend,
                text=desc,
                bg=colours["bg"],
                fg=colours["fg"],
                padx=6,
                pady=2,
            ).pack(side=tk.LEFT, padx=2)

    def _on_start(self) -> None:
        """Gather image paths from the import step and launch interrogation."""
        self._start_btn.config(state="disabled", text="Analysing...")
        self._view._bottom_bar.set_status("Analysing batch...", "#FF9F0A")
        self._view._bottom_bar.set_progress(0, 1)

        # Collect image paths from step 1 (StepImport)
        step_import = self._view._step_views[0]
        image_paths: list[str] = []
        if step_import and hasattr(step_import, "get_image_paths"):
            image_paths = step_import.get_image_paths()

        if not image_paths:
            self._progress_label.config(text="No images imported — go back to Step 1")
            self._start_btn.config(state="normal", text="Start Analysis")
            self._view._bottom_bar.set_status("Import images first", "#FF9F0A")
            self._view._bottom_bar.set_progress(0, 1)
            return

        self.start_interrogation(image_paths)

    def start_interrogation(
        self, image_paths: list[str], confirmed_labels: list[str] | None = None
    ) -> None:
        """Launch Moondream interrogation for all images in background."""
        total = len(image_paths)
        self._progress_bar["maximum"] = total

        def _worker() -> None:
            from core.interrogation import GuidedInterrogator, InterrogationSettings
            from core.knowledge import KnowledgePack
            import cv2

            settings_config = getattr(self._view, "interrogation_settings", {}) or {}
            guide_path = settings_config.get("guide_path") or self._view.knowledge_pack_path
            knowledge_pack = KnowledgePack.load(guide_path) if guide_path else None
            interrogator = GuidedInterrogator(
                InterrogationSettings(
                    host=self._app.prefs.get("ollama_url", "http://localhost:11434"),
                    primary_vlm=self._app.prefs.get("ollama_model", "moondream"),
                    fallback_vlms=[
                        str(settings_config.get("preferred_vlm", self._app.prefs.get("preferred_fallback_vlm", "minicpm-v"))),
                        "llava:7b",
                    ],
                    reasoner_model=str(
                        settings_config.get(
                            "text_reasoner_model",
                            self._app.prefs.get("preferred_text_reasoner", "qwen3.5"),
                        )
                    ),
                    profile=str(
                        settings_config.get(
                            "interrogation_profile",
                            self._app.prefs.get("interrogation_profile", "balanced"),
                        )
                    ),
                    fallback_mode=str(
                        settings_config.get(
                            "fallback_mode",
                            self._app.prefs.get("interrogation_fallback_mode", "adaptive_auto"),
                        )
                    ),
                    enable_tiling=bool(
                        settings_config.get(
                            "enable_tiled_fallback",
                            self._app.prefs.get("enable_tiled_fallback", True),
                        )
                    ),
                )
            )

            for i, path in enumerate(image_paths):
                try:
                    image = cv2.imread(path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    detected = interrogator.interrogate(
                        image,
                        confirmed_labels=confirmed_labels,
                        knowledge_pack=knowledge_pack,
                    )
                    labels = [
                        {
                            "label": candidate.display_label,
                            "canonical_label": candidate.canonical_label,
                            "role": candidate.role,
                            "confidence": candidate.confidence,
                            "source_model": candidate.source_model,
                        }
                        for candidate in detected.candidates
                    ]
                    self._queue.put(("progress", (i + 1, total, labels)))
                except Exception as exc:
                    logger.error("Interrogation failed for %s: %s", path, exc)
                    self._queue.put(("progress", (i + 1, total, [])))

            self._queue.put(("complete", None))

        threading.Thread(target=_worker, daemon=True).start()
        self._poll_queue()

    def _poll_queue(self) -> None:
        try:
            msg_type, data = self._queue.get_nowait()
            if msg_type == "progress":
                done, total, labels = data
                elapsed_text = f"~{max(1, (total - done) * 150 // 1000)}s remaining"
                self._progress_label.config(
                    text=f"{done} / {total} \u00b7 {elapsed_text}"
                )
                self._progress_bar["value"] = done
                self._view._bottom_bar.set_status(
                    f"Analysing {done}/{total} images...",
                    "#FF9F0A",
                )
                self._view._bottom_bar.set_progress(done, total)
                self._add_tags(labels)
            elif msg_type == "complete":
                self._progress_label.config(text="Interrogation complete")
                self._start_btn.config(state="normal", text="Re-run Analysis")
                self._view._bottom_bar.set_status("Review labels in Triage", "#007AFF")
                self._view._bottom_bar.set_progress(1, 1)
                return
        except queue.Empty:
            pass
        self._root.after(100, self._poll_queue)

    def _add_tags(self, labels: list[dict]) -> None:
        """Add new tags to the tag cloud."""
        for label_data in labels:
            label = label_data.get("label", "")
            key = label_data.get("canonical_label", label)
            role = label_data.get("role", "parent")
            if label and key not in self._tags:
                self._tags[key] = label_data
                colours = TAG_COLOURS.get(role, TAG_COLOURS["parent"])
                pill = tk.Label(
                    self._tag_cloud,
                    text=label,
                    bg=colours["bg"],
                    fg=colours["fg"],
                    highlightbackground=colours["border"],
                    highlightthickness=1,
                    padx=6,
                    pady=2,
                )
                pill.pack(side=tk.LEFT, padx=2, pady=2)

    def get_all_tags(self) -> dict[str, dict]:
        return dict(self._tags)
