from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ui.main_window import MainWindow


class SingleView:
    """Three-panel layout manager for Single Image mode.

    Left panel (308px) | Canvas (fills remaining) | Right panel (308px)
    """

    def __init__(self, parent: tk.Widget, app: MainWindow) -> None:
        self.app = app
        self.root = app.root
        self.frame = ttk.Frame(parent)
        self._last_result = None

        # Three-column layout via PanedWindow
        self._paned = ttk.PanedWindow(self.frame, orient=tk.HORIZONTAL)
        self._paned.pack(fill=tk.BOTH, expand=True)

        # Left panel
        from ui.single.left_panel import LeftPanel

        self._left_panel = LeftPanel(self._paned, self)
        self._paned.add(self._left_panel.frame, weight=0)

        # Canvas panel (centre)
        from ui.single.canvas_panel import CanvasPanel

        self._canvas_panel = CanvasPanel(self._paned, self)
        self._paned.add(self._canvas_panel.frame, weight=1)

        # Right panel
        from ui.single.right_panel import RightPanel

        self._right_panel = RightPanel(self._paned, self)
        self._paned.add(self._right_panel.frame, weight=0)

    @property
    def left_panel(self) -> object:
        return self._left_panel

    @property
    def canvas_panel(self) -> object:
        return self._canvas_panel

    @property
    def right_panel(self) -> object:
        return self._right_panel

    def on_image_loaded(self, image_path: str) -> None:
        """Called when an image is dropped / loaded."""
        self._last_result = None
        self._canvas_panel.load_image(image_path)
        self._canvas_panel.clear_scan_preview()
        self._right_panel.update_layers([])

    def on_labels_updated(self, labels: list[dict]) -> None:
        """Called when Moondream scan returns labels."""
        self._right_panel.update_layers([])

    def on_scan_preview_ready(self, detections: list[dict]) -> None:
        """Called when lightweight post-scan detections are available."""
        self._canvas_panel.set_scan_preview(detections)
        self._sync_preview_layers()

    def on_manual_detection_added(self, label: str, bbox: tuple[int, int, int, int]) -> None:
        """Called when the user draws an extra object box on the preview."""
        self._left_panel.add_manual_label(label)
        self._canvas_panel.add_manual_detection(label, bbox)
        self._sync_preview_layers()

    def rename_scan_detection(self, old_label: str, new_label: str) -> None:
        """Rename a detection in the canvas scan preview."""
        self._canvas_panel.rename_detection(old_label, new_label)
        self._sync_preview_layers()

    def remove_scan_detection(self, label: str) -> None:
        """Remove a detection from the canvas scan preview."""
        self._canvas_panel.remove_detection(label)
        self._sync_preview_layers()

    def _sync_preview_layers(self) -> None:
        """Push current scan preview detections to the right panel as
        provisional layers so the user sees them before processing."""
        if self._last_result is not None:
            return  # real results already shown — don't overwrite
        detections = self._canvas_panel.get_scan_preview_detections()
        layers = [
            {
                "label": d.get("label", "unknown"),
                "role": d.get("role", "parent"),
                "parent_label": "",
                "bbox": d.get("bbox"),
            }
            for d in detections
            if d.get("label")
        ]
        self._right_panel.update_layers(layers)

    def on_processing_complete(self, result: object) -> None:
        """Called when the 10-step pipeline finishes."""
        self._last_result = result

        # Convert PipelineResult.layers → list[dict] for the right panel
        layers_data: list[dict] = []
        if hasattr(result, "layers"):
            for layer in result.layers:
                layers_data.append({
                    "label": layer.label,
                    "role": layer.role,
                    "parent_label": layer.parent_label,
                    "bbox": layer.bbox,
                })

        if hasattr(result, "error") and result.error:
            import logging
            logging.getLogger(__name__).warning("Pipeline error: %s", result.error)

        self._right_panel.update_layers(layers_data)
        self._canvas_panel.refresh_overlays()
