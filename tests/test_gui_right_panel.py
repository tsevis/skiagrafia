"""test_gui_right_panel.py  --  Windowed tests for the layers/export panel.

Marked `gui` automatically (via tk_root) and excluded from a plain pytest
run. Run deliberately with: pytest -m gui

The panel is built by the shared `single_view` fixture, so these drive the
real widget alongside a real canvas and left panel.

Never called from here:
  _export / _export_svg / _export_tiff / _open_export_dialog
      all open a modal dialog containing a native folder chooser
"""
from __future__ import annotations

import sys
import tkinter as tk
import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _layer(label: str, role: str = "parent", **extra: object) -> dict:
    layer = {"label": label, "role": role}
    layer.update(extra)
    return layer


def _write_image(path: Path, size: tuple[int, int] = (120, 90)) -> Path:
    w, h = size
    Image.fromarray(np.full((h, w, 3), 128, dtype=np.uint8)).save(path)
    return path


@pytest.fixture
def right_panel(single_view, tk_root):
    tk_root.update()
    return single_view.right_panel


@pytest.fixture
def populated(right_panel, tk_root):
    """A panel showing one parent and two children."""
    right_panel.update_layers(
        [
            _layer("chalice"),
            _layer("stem", role="child", parent_label="chalice"),
            _layer("base", role="child", parent_label="chalice"),
        ]
    )
    tk_root.update()
    return right_panel


# ── Layer list rendering ────────────────────────────────────────────────────


class TestLayerRendering:
    def test_starts_empty_with_a_placeholder(self, right_panel, tk_root) -> None:
        right_panel.update_layers([])
        tk_root.update()
        assert right_panel._layers_placeholder.winfo_exists()

    def test_empty_state_hides_the_controls(self, right_panel, tk_root) -> None:
        right_panel.update_layers([])
        tk_root.update()
        assert not right_panel._controls_frame.winfo_ismapped()

    def test_one_row_per_layer(self, populated) -> None:
        assert len(populated._layer_widgets) == 3

    def test_rows_are_real_widgets(self, populated) -> None:
        assert all(row.winfo_exists() for row in populated._layer_widgets)

    def test_updating_replaces_previous_rows(self, populated, tk_root) -> None:
        populated.update_layers([_layer("urn")])
        tk_root.update()
        assert len(populated._layer_widgets) == 1

    def test_updating_clears_the_selection(self, populated, tk_root) -> None:
        populated._select_layer(0)
        populated.update_layers([_layer("urn")])
        tk_root.update()
        assert populated._selected_index is None

    def test_refresh_rebuilds_without_changing_the_data(
        self, populated, tk_root
    ) -> None:
        populated.refresh_layers()
        tk_root.update()
        assert len(populated._layer_widgets) == 3

    def test_visibility_var_per_layer(self, populated) -> None:
        assert set(populated._visibility) == {0, 1, 2}

    def test_layers_start_visible(self, populated) -> None:
        assert all(var.get() for var in populated._visibility.values())

    def test_going_back_to_empty_restores_the_placeholder(
        self, populated, tk_root
    ) -> None:
        populated.update_layers([])
        tk_root.update()
        assert populated._layers_placeholder.winfo_exists()
        assert populated._layer_widgets == []


# ── Row contents ────────────────────────────────────────────────────────────


class TestLayerRows:
    @staticmethod
    def _labels_in(row: tk.Widget) -> list[str]:
        texts: list[str] = []
        for child in row.winfo_children():
            for widget in (child, *child.winfo_children()):
                try:
                    texts.append(str(widget.cget("text")))
                except tk.TclError:
                    pass
        return texts

    def test_parent_row_is_badged_p(self, populated) -> None:
        assert "P" in self._labels_in(populated._layer_widgets[0])

    def test_child_row_is_badged_c(self, populated) -> None:
        assert "C" in self._labels_in(populated._layer_widgets[1])

    def test_row_shows_the_label(self, populated) -> None:
        assert "chalice" in self._labels_in(populated._layer_widgets[0])

    def test_parent_subtitle(self, populated) -> None:
        assert "Parent silhouette" in self._labels_in(populated._layer_widgets[0])

    def test_child_subtitle_names_its_parent(self, populated) -> None:
        texts = self._labels_in(populated._layer_widgets[1])
        assert any("Child of chalice" in t for t in texts)

    def test_layer_without_a_label_still_renders(
        self, right_panel, tk_root
    ) -> None:
        right_panel.update_layers([{"role": "parent"}])
        tk_root.update()
        assert len(right_panel._layer_widgets) == 1


# ── Thumbnails ──────────────────────────────────────────────────────────────


class TestThumbnails:
    def test_thumbnail_renders_without_a_loaded_image(
        self, populated
    ) -> None:
        # No image is loaded, so the thumbnail falls back to a blank swatch.
        assert populated._layer_widgets[0].winfo_exists()

    def test_thumbnail_uses_the_loaded_image_when_a_bbox_is_given(
        self, single_view, tk_root, tmp_path
    ) -> None:
        path = _write_image(tmp_path / "src.png")
        single_view.left_panel._image_path = str(path)
        right = single_view.right_panel
        right.update_layers([_layer("chalice", bbox=(10, 10, 80, 70))])
        tk_root.update()
        assert right._thumb_refs, "no thumbnail image was produced"

    def test_no_thumbnail_image_without_a_bbox(
        self, single_view, tk_root, tmp_path
    ) -> None:
        path = _write_image(tmp_path / "src.png")
        single_view.left_panel._image_path = str(path)
        right = single_view.right_panel
        right.update_layers([_layer("chalice")])
        tk_root.update()
        assert right._thumb_refs == []

    def test_out_of_bounds_bbox_does_not_raise(
        self, single_view, tk_root, tmp_path
    ) -> None:
        # _build_thumbnail hands the bbox straight to Image.crop, which pads
        # rather than clamping, so the crop can exceed the source. Kept
        # modestly out of range on purpose: an extreme bbox really does
        # allocate a crop that large (see the note in the commit message).
        path = _write_image(tmp_path / "src.png")
        single_view.left_panel._image_path = str(path)
        right = single_view.right_panel
        right.update_layers([_layer("chalice", bbox=(-20, -20, 300, 260))])
        tk_root.update()
        assert len(right._layer_widgets) == 1
        assert right._thumb_refs, "padded crop should still yield a thumbnail"

    def test_extreme_bbox_is_clamped_to_the_source(
        self, single_view, tk_root, tmp_path, monkeypatch
    ) -> None:
        # Image.crop pads rather than clamps, so an unclamped bbox allocates a
        # buffer the size of the bbox. Spy on the box actually handed to crop.
        path = _write_image(tmp_path / "src.png")  # 120x90
        single_view.left_panel._image_path = str(path)
        boxes: list[tuple[int, int, int, int]] = []
        original = Image.Image.crop

        def _spy(self, box=None):  # noqa: ANN001, ANN202
            boxes.append(box)
            return original(self, box)

        monkeypatch.setattr(Image.Image, "crop", _spy)
        single_view.right_panel.update_layers(
            [_layer("chalice", bbox=(-500, -500, 99999, 99999))]
        )
        tk_root.update()

        assert boxes, "crop was never called"
        x0, y0, x1, y1 = boxes[0]
        assert x0 >= 0 and y0 >= 0
        assert x1 <= 120 and y1 <= 90

    def test_extreme_bbox_does_not_trip_the_decompression_guard(
        self, single_view, tk_root, tmp_path
    ) -> None:
        path = _write_image(tmp_path / "src.png")
        single_view.left_panel._image_path = str(path)
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            single_view.right_panel.update_layers(
                [_layer("chalice", bbox=(-500, -500, 99999, 99999))]
            )
            tk_root.update()
        assert single_view.right_panel._thumb_refs

    def test_bbox_entirely_outside_the_image_is_survivable(
        self, single_view, tk_root, tmp_path
    ) -> None:
        path = _write_image(tmp_path / "src.png")
        single_view.left_panel._image_path = str(path)
        single_view.right_panel.update_layers(
            [_layer("chalice", bbox=(500, 500, 600, 600))]
        )
        tk_root.update()
        assert len(single_view.right_panel._layer_widgets) == 1

    def test_missing_image_file_does_not_raise(
        self, single_view, tk_root, tmp_path
    ) -> None:
        single_view.left_panel._image_path = str(tmp_path / "gone.png")
        right = single_view.right_panel
        right.update_layers([_layer("chalice", bbox=(0, 0, 10, 10))])
        tk_root.update()
        assert len(right._layer_widgets) == 1


# ── Selection and controls ──────────────────────────────────────────────────


class TestSelection:
    def test_selecting_records_the_index(self, populated) -> None:
        populated._select_layer(1)
        assert populated._selected_index == 1

    def test_selecting_shows_the_controls(self, populated, tk_root) -> None:
        populated._select_layer(0)
        tk_root.update()
        assert populated._controls_frame.winfo_ismapped()

    def test_selecting_fills_the_name_field(self, populated) -> None:
        populated._select_layer(1)
        assert populated._name_var.get() == "stem"

    def test_selecting_resets_the_sliders(self, populated) -> None:
        populated._opacity_var.set(12)
        populated._edge_var.set(9)
        populated._select_layer(0)
        assert populated._opacity_var.get() == 100
        assert populated._edge_var.get() == 0

    def test_selecting_updates_the_canvas_status(
        self, populated, single_view, tk_root
    ) -> None:
        populated._select_layer(2)
        tk_root.update()
        status = single_view.canvas_panel._layer_status.cget("text")
        assert status == "base"

    def test_visibility_toggle_refreshes_overlays(
        self, populated, tk_root
    ) -> None:
        populated._visibility[0].set(False)
        populated._on_visibility_change()
        tk_root.update()
        assert populated._visibility[0].get() is False


# ── Deletion ────────────────────────────────────────────────────────────────


class TestDeletion:
    def test_delete_removes_the_selected_layer(self, populated, tk_root) -> None:
        populated._select_layer(1)
        populated._delete_layer()
        tk_root.update()
        assert [layer["label"] for layer in populated._layers] == [
            "chalice",
            "base",
        ]

    def test_delete_rerenders_the_rows(self, populated, tk_root) -> None:
        populated._select_layer(0)
        populated._delete_layer()
        tk_root.update()
        assert len(populated._layer_widgets) == 2

    def test_delete_clears_the_selection(self, populated, tk_root) -> None:
        populated._select_layer(0)
        populated._delete_layer()
        tk_root.update()
        assert populated._selected_index is None

    def test_delete_hides_the_controls(self, populated, tk_root) -> None:
        populated._select_layer(0)
        populated._delete_layer()
        tk_root.update()
        assert not populated._controls_frame.winfo_ismapped()

    def test_delete_without_a_selection_is_a_noop(
        self, populated, tk_root
    ) -> None:
        populated._delete_layer()
        tk_root.update()
        assert len(populated._layers) == 3

    def test_deleting_every_layer_restores_the_placeholder(
        self, populated, tk_root
    ) -> None:
        for _ in range(3):
            populated._select_layer(0)
            populated._delete_layer()
        tk_root.update()
        assert populated._layers == []
        assert populated._layers_placeholder.winfo_exists()

    def test_resegment_without_a_selection_is_a_noop(
        self, populated, tk_root
    ) -> None:
        populated._resegment_layer()
        tk_root.update()
        assert len(populated._layers) == 3


# ── Scrolling ───────────────────────────────────────────────────────────────


class TestScrolling:
    def test_mousewheel_binding_round_trips(self, right_panel, tk_root) -> None:
        right_panel._bind_mousewheel()
        right_panel._unbind_mousewheel()
        tk_root.update()
        assert right_panel._scroll_canvas.winfo_exists()

    def test_scrolling_does_not_raise(self, populated, tk_root) -> None:
        event = tk.Event()
        event.delta = -120
        populated._on_mousewheel(event)
        tk_root.update()
        assert populated._scroll_canvas.winfo_exists()


# ── Batch template ──────────────────────────────────────────────────────────


class TestCreateTemplate:
    def test_template_dialog_opens_fully(
        self, populated, single_view, tk_root, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        single_view.left_panel._image_path = str(_write_image(tmp_path / "s.png"))

        before = set(tk_root.winfo_children())
        populated._create_template()
        tk_root.update()

        dialog = next(
            w for w in tk_root.winfo_children() if w not in before
        )
        buttons = [
            child
            for frame in dialog.winfo_children()
            for child in frame.winfo_children()
            if child.winfo_class() == "TButton"
        ]
        assert len(buttons) == 2, "expected Save-only and Save-and-switch"

    def test_saving_the_template_writes_it(
        self, populated, single_view, tk_root, tmp_path, monkeypatch
    ) -> None:
        from core.batch_template import BatchTemplate

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        single_view.left_panel._image_path = str(_write_image(tmp_path / "s.png"))

        before = set(tk_root.winfo_children())
        populated._create_template()
        tk_root.update()
        dialog = next(w for w in tk_root.winfo_children() if w not in before)
        save_only = next(
            child
            for frame in dialog.winfo_children()
            for child in frame.winfo_children()
            if child.winfo_class() == "TButton"
            and "only" in str(child.cget("text")).lower()
        )
        save_only.invoke()
        tk_root.update()

        saved = BatchTemplate.list_all()
        assert [t.name for t in saved] == ["s"]
        assert saved[0].confirmed_children == {"chalice": ["stem", "base"]}

    def test_save_and_switch_hands_off_to_batch_mode(
        self, populated, single_view, tk_root, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        single_view.left_panel._image_path = str(_write_image(tmp_path / "s.png"))
        switched: list[bool] = []
        single_view.app.switch_to_batch = lambda: switched.append(True)

        before = set(tk_root.winfo_children())
        populated._create_template()
        tk_root.update()
        dialog = next(w for w in tk_root.winfo_children() if w not in before)
        switch_btn = next(
            child
            for frame in dialog.winfo_children()
            for child in frame.winfo_children()
            if child.winfo_class() == "TButton"
            and "switch" in str(child.cget("text")).lower()
        )
        switch_btn.invoke()
        tk_root.update()

        assert switched == [True]
