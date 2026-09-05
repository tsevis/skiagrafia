"""test_gui_single_view.py  --  Windowed tests for the Single Image view.

These construct real Tk windows and are therefore marked `gui` (automatically,
via the tk_root fixture) and excluded from a plain `pytest` run. Run them
deliberately:

    pytest -m gui

They exist to cover what launching the app cannot: the drawing and event
methods extracted into canvas_drawing.py, canvas_events.py and
left_panel_labels.py only execute once an image is loaded and the user
interacts, so startup alone proves nothing about them.

No model weights, network calls or subprocesses are involved.
"""
from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _write_image(path: Path, size: tuple[int, int] = (320, 240)) -> Path:
    """A small non-uniform RGB image (uniform ones hide scaling bugs)."""
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[: h // 2, : w // 2] = (200, 30, 30)
    arr[h // 2 :, w // 2 :] = (30, 30, 200)
    Image.fromarray(arr).save(path)
    return path


def _event(**kwargs: object) -> tk.Event:
    event = tk.Event()
    for key, value in kwargs.items():
        setattr(event, key, value)
    return event


@pytest.fixture
def loaded_canvas(single_view, tk_root, tmp_path):
    """A canvas panel with an image loaded and framed."""
    canvas = single_view.canvas_panel
    canvas.load_image(str(_write_image(tmp_path / "sample.png")))
    tk_root.update()
    canvas.zoom_to_cover()
    tk_root.update()
    return canvas


# ── Construction ────────────────────────────────────────────────────────────


class TestSingleViewConstruction:
    def test_builds_all_three_panels(self, single_view) -> None:
        assert single_view.left_panel is not None
        assert single_view.canvas_panel is not None
        assert single_view.right_panel is not None

    def test_panels_are_real_mapped_widgets(self, single_view, tk_root) -> None:
        tk_root.update()
        for panel in (
            single_view.left_panel,
            single_view.canvas_panel,
            single_view.right_panel,
        ):
            assert panel.frame.winfo_exists()

    def test_canvas_has_usable_geometry(self, single_view, tk_root) -> None:
        tk_root.update()
        canvas = single_view.canvas_panel._canvas
        assert canvas.winfo_width() > 1
        assert canvas.winfo_height() > 1


# ── canvas_drawing.py ───────────────────────────────────────────────────────


class TestCanvasDrawing:
    def test_load_image_records_source_size(self, loaded_canvas) -> None:
        assert loaded_canvas._source_image is not None
        assert loaded_canvas._source_size == (320, 240)

    def test_load_image_reports_dimensions_in_status_bar(
        self, loaded_canvas
    ) -> None:
        assert "320" in loaded_canvas._dims_status.cget("text")

    def test_load_image_ignores_a_missing_file(self, single_view, tmp_path) -> None:
        canvas = single_view.canvas_panel
        canvas.load_image(str(tmp_path / "nope.png"))
        assert canvas._source_image is None

    def test_redraw_paints_canvas_items(self, loaded_canvas) -> None:
        loaded_canvas._redraw()
        assert loaded_canvas._canvas.find_all(), "canvas painted nothing"

    def test_redraw_is_a_noop_without_an_image(self, single_view, tk_root) -> None:
        canvas = single_view.canvas_panel
        tk_root.update()
        canvas._redraw()
        assert canvas._canvas.find_all() == ()

    def test_checkerboard_is_drawn(self, loaded_canvas) -> None:
        loaded_canvas._canvas.delete("all")
        loaded_canvas._draw_checkerboard(200, 150)
        assert loaded_canvas._canvas.find_all()

    def test_zoom_to_cover_fills_the_viewport(self, loaded_canvas) -> None:
        cw = loaded_canvas._canvas.winfo_width()
        ch = loaded_canvas._canvas.winfo_height()
        iw, ih = loaded_canvas._source_size
        assert iw * loaded_canvas._zoom >= cw - 1
        assert ih * loaded_canvas._zoom >= ch - 1

    def test_zoom_label_tracks_zoom(self, loaded_canvas) -> None:
        loaded_canvas._zoom_at(2.0)
        assert "%" in loaded_canvas._zoom_label.cget("text")

    def test_scan_preview_image_is_built(self, loaded_canvas, tk_root) -> None:
        loaded_canvas.set_scan_preview(
            [{"label": "guitar", "bbox": (10, 10, 200, 200), "confidence": 0.9}]
        )
        tk_root.update()
        assert loaded_canvas._scan_preview_image is not None

    def test_build_scan_preview_image_returns_an_image(self, loaded_canvas) -> None:
        result = loaded_canvas._build_scan_preview_image(
            [{"label": "body", "bbox": (0, 0, 100, 100), "confidence": 0.8}]
        )
        assert isinstance(result, Image.Image)

    def test_scan_preview_survives_a_redraw(self, loaded_canvas, tk_root) -> None:
        loaded_canvas.set_scan_preview(
            [{"label": "guitar", "bbox": (5, 5, 120, 120), "confidence": 0.7}]
        )
        tk_root.update()
        loaded_canvas._redraw()
        assert loaded_canvas._canvas.find_all()


# ── canvas_events.py ────────────────────────────────────────────────────────


class TestCanvasEvents:
    def test_zoom_at_magnifies(self, loaded_canvas) -> None:
        before = loaded_canvas._zoom
        loaded_canvas._zoom_at(2.0)
        assert loaded_canvas._zoom > before

    def test_zoom_is_clamped_at_the_maximum(self, loaded_canvas) -> None:
        for _ in range(60):
            loaded_canvas._zoom_at(2.0)
        assert loaded_canvas._zoom <= loaded_canvas.MAX_ZOOM

    def test_zoom_is_clamped_at_the_minimum(self, loaded_canvas) -> None:
        for _ in range(60):
            loaded_canvas._zoom_at(0.5)
        assert loaded_canvas._zoom >= loaded_canvas.MIN_ZOOM

    def test_scroll_up_zooms_in(self, loaded_canvas) -> None:
        before = loaded_canvas._zoom
        loaded_canvas._on_scroll(_event(delta=1, x=100, y=100))
        assert loaded_canvas._zoom > before

    def test_scroll_down_zooms_out(self, loaded_canvas) -> None:
        loaded_canvas._zoom_at(4.0)
        before = loaded_canvas._zoom
        loaded_canvas._on_scroll(_event(delta=-1, x=100, y=100))
        assert loaded_canvas._zoom < before

    def test_middle_drag_pans(self, loaded_canvas) -> None:
        start_x = loaded_canvas._pan_x
        loaded_canvas._on_pan_start(_event(x=100, y=100))
        loaded_canvas._on_pan_drag(_event(x=140, y=120))
        assert loaded_canvas._pan_x == start_x + 40

    def test_pan_drag_without_a_start_is_ignored(self, loaded_canvas) -> None:
        loaded_canvas._on_pan_end(_event(x=0, y=0))
        before = loaded_canvas._pan_x
        loaded_canvas._on_pan_drag(_event(x=999, y=999))
        assert loaded_canvas._pan_x == before

    def test_pan_end_clears_the_drag(self, loaded_canvas) -> None:
        loaded_canvas._on_pan_start(_event(x=10, y=10))
        loaded_canvas._on_pan_end(_event(x=10, y=10))
        assert loaded_canvas._drag_start is None

    def test_space_toggles_pan_mode(self, loaded_canvas) -> None:
        loaded_canvas._on_space_press(_event())
        assert loaded_canvas._space_held is True
        loaded_canvas._on_space_release(_event())
        assert loaded_canvas._space_held is False

    def test_canvas_rect_maps_to_image_coordinates(self, loaded_canvas) -> None:
        loaded_canvas._zoom = 1.0
        loaded_canvas._pan_x = 0.0
        loaded_canvas._pan_y = 0.0
        bbox = loaded_canvas._canvas_rect_to_image_bbox((10, 20), (110, 140))
        assert bbox == (10, 20, 110, 140)

    def test_canvas_rect_is_normalised_when_dragged_backwards(
        self, loaded_canvas
    ) -> None:
        loaded_canvas._zoom = 1.0
        loaded_canvas._pan_x = 0.0
        loaded_canvas._pan_y = 0.0
        bbox = loaded_canvas._canvas_rect_to_image_bbox((110, 140), (10, 20))
        assert bbox == (10, 20, 110, 140)

    def test_canvas_rect_is_clipped_to_the_image(self, loaded_canvas) -> None:
        loaded_canvas._zoom = 1.0
        loaded_canvas._pan_x = 0.0
        loaded_canvas._pan_y = 0.0
        bbox = loaded_canvas._canvas_rect_to_image_bbox((-500, -500), (9999, 9999))
        assert bbox == (0, 0, 320, 240)

    def test_tiny_rect_is_rejected(self, loaded_canvas) -> None:
        loaded_canvas._zoom = 1.0
        loaded_canvas._pan_x = 0.0
        loaded_canvas._pan_y = 0.0
        assert loaded_canvas._canvas_rect_to_image_bbox((10, 10), (12, 12)) is None

    def test_manual_box_release_uses_the_prompted_label(
        self, loaded_canvas, tk_root, monkeypatch
    ) -> None:
        # _finish_manual_box opens a modal dialog and blocks on wait_window,
        # so the prompt must be stubbed or the suite would hang here.
        monkeypatch.setattr(
            type(loaded_canvas), "_prompt_manual_label", lambda self: "headstock"
        )
        loaded_canvas._zoom = 1.0
        loaded_canvas._pan_x = 0.0
        loaded_canvas._pan_y = 0.0
        loaded_canvas._draw_box_mode.set(True)
        loaded_canvas._view_mode.set("original")

        loaded_canvas._on_left_click(_event(x=20, y=20))
        loaded_canvas._on_left_drag(_event(x=140, y=130))
        loaded_canvas._on_left_release(_event(x=140, y=130))
        tk_root.update()

        labels = [d["label"] for d in loaded_canvas.get_manual_detections()]
        assert "headstock" in labels

    def test_cancelled_prompt_adds_nothing(
        self, loaded_canvas, tk_root, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            type(loaded_canvas), "_prompt_manual_label", lambda self: None
        )
        loaded_canvas._zoom = 1.0
        loaded_canvas._pan_x = 0.0
        loaded_canvas._pan_y = 0.0
        loaded_canvas._draw_box_mode.set(True)
        loaded_canvas._view_mode.set("original")

        loaded_canvas._on_left_click(_event(x=20, y=20))
        loaded_canvas._on_left_drag(_event(x=140, y=130))
        loaded_canvas._on_left_release(_event(x=140, y=130))
        tk_root.update()

        assert loaded_canvas.get_manual_detections() == []


# ── Scan-preview detection lifecycle ────────────────────────────────────────


class TestScanPreviewLifecycle:
    def test_set_and_read_back_detections(self, loaded_canvas, tk_root) -> None:
        loaded_canvas.set_scan_preview(
            [
                {"label": "guitar", "bbox": (0, 0, 100, 100), "confidence": 0.9},
                {"label": "amp", "bbox": (150, 100, 300, 200), "confidence": 0.8},
            ]
        )
        tk_root.update()
        labels = [d["label"] for d in loaded_canvas.get_scan_preview_detections()]
        assert labels == ["guitar", "amp"]

    def test_returned_detections_are_copies(self, loaded_canvas, tk_root) -> None:
        loaded_canvas.set_scan_preview(
            [{"label": "guitar", "bbox": (0, 0, 100, 100), "confidence": 0.9}]
        )
        tk_root.update()
        got = loaded_canvas.get_scan_preview_detections()
        got[0]["label"] = "mutated"
        assert loaded_canvas.get_scan_preview_detections()[0]["label"] == "guitar"

    def test_manual_detection_is_tagged_and_listed(
        self, loaded_canvas, tk_root
    ) -> None:
        loaded_canvas.add_manual_detection("neck", (10, 10, 90, 90))
        tk_root.update()
        manual = loaded_canvas.get_manual_detections()
        assert len(manual) == 1
        assert manual[0]["source"] == "manual"

    def test_manual_detections_survive_a_rescan(self, loaded_canvas, tk_root) -> None:
        loaded_canvas.add_manual_detection("neck", (10, 10, 90, 90))
        loaded_canvas.set_scan_preview(
            [{"label": "guitar", "bbox": (0, 0, 100, 100), "confidence": 0.9}]
        )
        tk_root.update()
        labels = [d["label"] for d in loaded_canvas.get_scan_preview_detections()]
        assert "neck" in labels and "guitar" in labels

    def test_rename_is_case_insensitive(self, loaded_canvas, tk_root) -> None:
        loaded_canvas.set_scan_preview(
            [{"label": "Guitar", "bbox": (0, 0, 100, 100), "confidence": 0.9}]
        )
        tk_root.update()
        loaded_canvas.rename_detection("guitar", "bass")
        labels = [d["label"] for d in loaded_canvas.get_scan_preview_detections()]
        assert labels == ["bass"]

    def test_remove_drops_the_detection(self, loaded_canvas, tk_root) -> None:
        loaded_canvas.set_scan_preview(
            [
                {"label": "guitar", "bbox": (0, 0, 100, 100), "confidence": 0.9},
                {"label": "amp", "bbox": (150, 100, 300, 200), "confidence": 0.8},
            ]
        )
        tk_root.update()
        loaded_canvas.remove_detection("GUITAR")
        labels = [d["label"] for d in loaded_canvas.get_scan_preview_detections()]
        assert labels == ["amp"]

    def test_removing_the_last_detection_clears_the_preview(
        self, loaded_canvas, tk_root
    ) -> None:
        loaded_canvas.set_scan_preview(
            [{"label": "guitar", "bbox": (0, 0, 100, 100), "confidence": 0.9}]
        )
        tk_root.update()
        loaded_canvas.remove_detection("guitar")
        assert loaded_canvas._scan_preview_image is None

    def test_clear_resets_everything(self, loaded_canvas, tk_root) -> None:
        loaded_canvas.set_scan_preview(
            [{"label": "guitar", "bbox": (0, 0, 100, 100), "confidence": 0.9}]
        )
        tk_root.update()
        loaded_canvas.clear_scan_preview()
        assert loaded_canvas.get_scan_preview_detections() == []
        assert loaded_canvas._scan_preview_image is None


# ── left_panel_labels.py ────────────────────────────────────────────────────


class TestLeftPanelLabels:
    def test_manual_label_renders_a_pill(self, single_view, tk_root) -> None:
        left = single_view.left_panel
        left.add_manual_label("chalice")
        tk_root.update()
        assert left._labels_container.winfo_children()

    def test_manual_label_is_recorded(self, single_view) -> None:
        left = single_view.left_panel
        left.add_manual_label("chalice")
        assert [item["label"] for item in left._labels] == ["chalice"]

    def test_duplicate_label_is_ignored_case_insensitively(
        self, single_view
    ) -> None:
        left = single_view.left_panel
        left.add_manual_label("chalice")
        left.add_manual_label("CHALICE")
        assert len(left._labels) == 1

    def test_blank_label_is_ignored(self, single_view) -> None:
        left = single_view.left_panel
        left.add_manual_label("   ")
        assert left._labels == []

    def test_pill_count_matches_label_count(self, single_view, tk_root) -> None:
        left = single_view.left_panel
        for name in ("chalice", "stem", "base"):
            left.add_manual_label(name)
        tk_root.update()
        assert len(left._labels_container.winfo_children()) == 3

    def test_empty_state_shows_the_hint(self, single_view, tk_root) -> None:
        left = single_view.left_panel
        left._labels = []
        left._render_label_pills()
        tk_root.update()
        assert left._labels_hint.winfo_ismapped()

    def test_toggling_a_pill_changes_its_colour(self, single_view, tk_root) -> None:
        left = single_view.left_panel
        left.add_manual_label("chalice")
        tk_root.update()
        pill = left._labels_container.winfo_children()[0]
        before = str(pill.cget("bg"))
        left._toggle_label(left._labels[0], pill)
        assert str(pill.cget("bg")) != before

    def test_toggling_twice_restores_the_original_colour(
        self, single_view, tk_root
    ) -> None:
        left = single_view.left_panel
        left.add_manual_label("chalice")
        tk_root.update()
        pill = left._labels_container.winfo_children()[0]
        before = str(pill.cget("bg"))
        left._toggle_label(left._labels[0], pill)
        left._toggle_label(left._labels[0], pill)
        assert str(pill.cget("bg")) == before

    def test_confirmed_labels_reflect_added_labels(self, single_view, tk_root) -> None:
        left = single_view.left_panel
        left.add_manual_label("chalice")
        tk_root.update()
        assert "chalice" in left.get_confirmed_labels()


class TestLeftPanelOutputMode:
    """get_svg_mode_label feeds the batch-template summary in the right panel."""

    def test_both_modes_selected(self, single_view) -> None:
        left = single_view.left_panel
        left._mode_structural_svg_var.set(True)
        left._mode_bitmap_var.set(True)
        assert left.get_svg_mode_label() == "Structural SVG + bitmap"

    def test_svg_only(self, single_view) -> None:
        left = single_view.left_panel
        left._mode_structural_svg_var.set(True)
        left._mode_bitmap_var.set(False)
        assert left.get_svg_mode_label() == "Structural SVG"

    def test_bitmap_only(self, single_view) -> None:
        left = single_view.left_panel
        left._mode_structural_svg_var.set(False)
        left._mode_bitmap_var.set(True)
        assert left.get_svg_mode_label() == "Bitmap only"

    def test_neither_falls_back_like_get_output_mode(self, single_view) -> None:
        # _get_output_mode falls back to "vector" when nothing is ticked, so
        # the human-readable label must not claim otherwise.
        left = single_view.left_panel
        left._mode_structural_svg_var.set(False)
        left._mode_bitmap_var.set(False)
        assert left._get_output_mode() == "vector"
        assert left.get_svg_mode_label() == "Structural SVG"

    def test_label_agrees_with_the_machine_readable_mode(
        self, single_view
    ) -> None:
        left = single_view.left_panel
        for svg in (True, False):
            for bitmap in (True, False):
                left._mode_structural_svg_var.set(svg)
                left._mode_bitmap_var.set(bitmap)
                mode = left._get_output_mode()
                label = left.get_svg_mode_label()
                assert ("bitmap" in mode) == ("bitmap" in label.lower())
