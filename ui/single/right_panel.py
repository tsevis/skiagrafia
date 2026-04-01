from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import TYPE_CHECKING

from PIL import Image, ImageTk

from ui.theme import get_layer_colour, is_macos

if TYPE_CHECKING:
    from ui.single.single_view import SingleView

logger = logging.getLogger(__name__)


class RightPanel:
    """Right panel: layers list, per-layer controls, export.

    Width: 308px fixed. Entire panel is scrollable via mousewheel.
    """

    PANEL_WIDTH = 308

    def __init__(self, parent: tk.Widget, view: SingleView) -> None:
        self._view = view
        self._app = view.app
        self._root = view.root

        self.frame = ttk.Frame(parent, width=self.PANEL_WIDTH)
        self.frame.pack_propagate(False)

        self._layers: list[dict] = []
        self._selected_index: int | None = None
        self._layer_widgets: list[ttk.Frame] = []
        self._visibility: dict[int, tk.BooleanVar] = {}
        self._thumb_refs: list[ImageTk.PhotoImage] = []
        self._control_entry_vars: dict[str, tk.StringVar] = {}

        # ── Scrollable wrapper ─────────────────────────────────
        self._scroll_canvas = tk.Canvas(
            self.frame, highlightthickness=0, width=self.PANEL_WIDTH
        )
        self._scrollbar = ttk.Scrollbar(
            self.frame, orient=tk.VERTICAL, command=self._scroll_canvas.yview
        )
        self._inner = ttk.Frame(self._scroll_canvas)

        self._inner.bind(
            "<Configure>",
            lambda e: self._scroll_canvas.configure(
                scrollregion=self._scroll_canvas.bbox("all")
            ),
        )
        self._scroll_canvas.create_window(
            (0, 0), window=self._inner, anchor=tk.NW, width=self.PANEL_WIDTH - 14
        )
        self._scroll_canvas.configure(yscrollcommand=self._scrollbar.set)
        self._scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Mousewheel binding
        self._scroll_canvas.bind("<Enter>", self._bind_mousewheel)
        self._scroll_canvas.bind("<Leave>", self._unbind_mousewheel)

        self._build_layers_section()
        self._build_layer_controls()
        self._build_export_section()

    # ── Mousewheel scrolling ───────────────────────────────────

    def _bind_mousewheel(self, event: object = None) -> None:
        self._scroll_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, event: object = None) -> None:
        self._scroll_canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event: tk.Event) -> None:
        self._scroll_canvas.yview_scroll(-1 * (event.delta // 120 or event.delta), "units")

    # ── Layers section ─────────────────────────────────────────

    def _build_layers_section(self) -> None:
        section = ttk.LabelFrame(self._inner, text="Layers", padding=4)
        section.pack(fill=tk.X, padx=6, pady=(6, 3))

        self._layers_frame = ttk.Frame(section)
        self._layers_frame.pack(fill=tk.X)

        # Placeholder
        self._layers_placeholder = ttk.Label(
            self._layers_frame,
            text="No layers yet\nProcess an image first",
            foreground="gray",
            justify=tk.CENTER,
        )
        self._layers_placeholder.pack(pady=20)

    def _build_layer_controls(self) -> None:
        """Per-layer controls (collapsed until a layer is selected)."""
        self._controls_frame = ttk.LabelFrame(
            self._inner, text="Layer Controls", padding=6
        )

        # Name entry
        ttk.Label(self._controls_frame, text="Name").pack(anchor=tk.W)
        self._name_var = tk.StringVar()
        self._name_entry = ttk.Entry(
            self._controls_frame, textvariable=self._name_var
        )
        self._name_entry.pack(fill=tk.X, pady=(0, 4))

        # Opacity slider
        ttk.Label(self._controls_frame, text="Opacity").pack(anchor=tk.W)
        self._opacity_var = tk.IntVar(value=100)
        self._build_numeric_control(
            self._controls_frame,
            "Opacity",
            self._opacity_var,
            0,
            100,
        )

        # Edge refinement slider
        ttk.Label(self._controls_frame, text="Edge refinement").pack(anchor=tk.W)
        self._edge_var = tk.IntVar(value=0)
        self._build_numeric_control(
            self._controls_frame,
            "Edge refinement",
            self._edge_var,
            0,
            10,
        )

        # Buttons
        btn_frame = ttk.Frame(self._controls_frame)
        btn_frame.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(
            btn_frame, text="Re-segment", command=self._resegment_layer
        ).pack(fill=tk.X, pady=1)
        ttk.Button(
            btn_frame, text="Delete layer", command=self._delete_layer
        ).pack(fill=tk.X, pady=1)

    def _build_export_section(self) -> None:
        """Export buttons at the bottom of the scrollable area."""
        section = ttk.Frame(self._inner, padding=6)
        section.pack(fill=tk.X, padx=6, pady=(3, 6))

        self._export_btn = ttk.Button(
            section, text="Export", command=self._export
        )
        self._export_btn.pack(fill=tk.X, pady=1)

        btn_row = ttk.Frame(section)
        btn_row.pack(fill=tk.X, pady=1)
        ttk.Button(btn_row, text="SVG", command=self._export_svg, width=8).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2)
        )
        ttk.Button(btn_row, text="TIFF", command=self._export_tiff, width=8).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0)
        )

        self._template_btn = ttk.Button(
            section,
            text="Use as batch template \u2192",
            command=self._create_template,
        )
        self._template_btn.pack(fill=tk.X, pady=(4, 0))

    # ── Layer management ───────────────────────────────────────

    def update_layers(self, layers: list[dict]) -> None:
        """Update the layers list from pipeline results."""
        self._layers = layers
        self._selected_index = None
        self._render_layers()

    def refresh_layers(self) -> None:
        """Re-render the layers list."""
        self._render_layers()

    def _render_layers(self) -> None:
        """Render layer rows in the layers frame."""
        for w in self._layers_frame.winfo_children():
            w.destroy()
        self._layer_widgets.clear()
        self._visibility.clear()
        self._thumb_refs.clear()

        if not self._layers:
            self._layers_placeholder = ttk.Label(
                self._layers_frame,
                text="No layers yet\nProcess an image first",
                foreground="gray",
                justify=tk.CENTER,
            )
            self._layers_placeholder.pack(pady=20)
            self._controls_frame.pack_forget()
            return

        for i, layer in enumerate(self._layers):
            row = self._create_layer_row(i, layer)
            row.pack(fill=tk.X, pady=1)
            self._layer_widgets.append(row)

    def _create_layer_row(self, index: int, layer: dict) -> ttk.Frame:
        """Create a single layer row widget."""
        role = layer.get("role", "parent")
        colour = get_layer_colour(role, index)

        row = ttk.Frame(self._layers_frame)
        row.bind("<Button-1>", lambda e, i=index: self._select_layer(i))

        # Eye toggle
        vis_var = tk.BooleanVar(value=True)
        self._visibility[index] = vis_var
        eye = ttk.Checkbutton(row, variable=vis_var, command=self._on_visibility_change)
        eye.pack(side=tk.LEFT, padx=(2, 4))

        thumb = self._build_thumbnail(row, layer, colour)
        thumb.pack(side=tk.LEFT, padx=(0, 4))

        # Label info
        info = ttk.Frame(row)
        info.pack(side=tk.LEFT, fill=tk.X, expand=True)
        info.bind("<Button-1>", lambda e, i=index: self._select_layer(i))

        ttk.Label(
            info,
            text=layer.get("label", f"Layer {index}"),
            wraplength=180,
            justify=tk.LEFT,
        ).pack(anchor=tk.W)
        subtitle = "Parent silhouette" if role == "parent" else f"Child of {layer.get('parent_label', '')}"
        ttk.Label(
            info,
            text=subtitle,
            foreground="gray",
            wraplength=180,
            justify=tk.LEFT,
        ).pack(anchor=tk.W)

        # Role badge
        badge_bg = "#007AFF" if role == "parent" else "#34C759"
        badge = tk.Label(
            row,
            text="P" if role == "parent" else "C",
            bg=badge_bg,
            fg="white",
            width=2,
            font=("SF Pro Text", 8, "bold") if is_macos() else ("Segoe UI", 7, "bold"),
        )
        badge.pack(side=tk.RIGHT, padx=4)

        return row

    def _build_thumbnail(self, parent: ttk.Frame, layer: dict, colour: str) -> tk.Canvas:
        thumb = tk.Canvas(
            parent,
            width=56,
            height=48,
            bg="#F3F1EC",
            highlightbackground=colour,
            highlightthickness=1,
            bd=0,
        )
        thumb.create_rectangle(0, 0, 56, 48, fill="#F3F1EC", outline="")

        image_path = getattr(self._view.left_panel, "_image_path", None)
        bbox = layer.get("bbox")
        if not image_path or not bbox:
            return thumb

        try:
            with Image.open(image_path) as source:
                x0, y0, x1, y1 = bbox
                crop = source.convert("RGB").crop((x0, y0, x1, y1))
                crop.thumbnail((56, 48), Image.LANCZOS)
                preview = Image.new("RGB", (56, 48), "#F3F1EC")
                offset = ((56 - crop.width) // 2, (48 - crop.height) // 2)
                preview.paste(crop, offset)
                photo = ImageTk.PhotoImage(preview)
                thumb.create_image(28, 24, image=photo)
                self._thumb_refs.append(photo)
        except Exception:
            logger.warning("Failed to render layer thumbnail for %s", layer.get("label"), exc_info=True)

        return thumb

    def _build_numeric_control(
        self,
        parent: ttk.Widget,
        label: str,
        var: tk.IntVar,
        minimum: int,
        maximum: int,
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=(0, 4))

        entry_var = tk.StringVar(value=str(var.get()))
        self._control_entry_vars[label] = entry_var

        entry = ttk.Entry(row, textvariable=entry_var, width=7, justify=tk.RIGHT)
        entry.pack(side=tk.RIGHT)

        def _apply_entry(event: object = None) -> None:
            try:
                raw = int(float(entry_var.get()))
            except ValueError:
                raw = int(var.get())
            clamped = min(max(raw, minimum), maximum)
            var.set(clamped)
            entry_var.set(str(clamped))

        entry.bind("<Return>", _apply_entry)
        entry.bind("<FocusOut>", _apply_entry)

        scale = ttk.Scale(
            row,
            variable=var,
            from_=minimum,
            to=maximum,
            command=lambda val: entry_var.set(str(int(round(float(val))))),
        )
        scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(6, 6))

    def _select_layer(self, index: int) -> None:
        """Select a layer and show its controls."""
        self._selected_index = index
        layer = self._layers[index]

        self._name_var.set(layer.get("label", ""))
        self._opacity_var.set(100)
        self._edge_var.set(0)

        self._controls_frame.pack(fill=tk.X, padx=6, pady=3)

        # Update canvas active layer
        self._view.canvas_panel.set_active_layer(layer.get("label", ""))

    def _on_visibility_change(self) -> None:
        """Handle layer visibility toggle."""
        self._view.canvas_panel.refresh_overlays()

    def _resegment_layer(self) -> None:
        """Re-run SAM for the selected layer."""
        if self._selected_index is None:
            return
        logger.info("Re-segmenting layer %d", self._selected_index)

    def _delete_layer(self) -> None:
        """Delete the selected layer."""
        if self._selected_index is None:
            return
        del self._layers[self._selected_index]
        self._selected_index = None
        self._controls_frame.pack_forget()
        self._render_layers()

    # ── Export ─────────────────────────────────────────────────

    def _export(self) -> None:
        """Open export dialog with all options."""
        self._open_export_dialog()

    def _export_svg(self) -> None:
        """Quick SVG export via save dialog."""
        self._open_export_dialog(preselect="svg")

    def _export_tiff(self) -> None:
        """Quick TIFF export via save dialog."""
        self._open_export_dialog(preselect="tiff")

    def _open_export_dialog(self, preselect: str | None = None) -> None:
        """Open a modal export dialog with format checkboxes and folder picker."""
        from tkinter import filedialog
        from processors.output_writer import write_pdf

        result = getattr(self._view, "_last_result", None)
        left = self._view.left_panel
        wants_structural = left.wants_structural_svg()
        has_structural = bool(result and getattr(result, "svg_path", None))

        dialog = tk.Toplevel(self._root)
        dialog.title("Export")
        dialog.geometry("420x390")
        dialog.transient(self._root)
        dialog.grab_set()

        ttk.Label(
            dialog,
            text="Export Options",
            font=("SF Pro Display", 14, "bold") if is_macos() else ("Segoe UI", 12, "bold"),
        ).pack(padx=16, pady=(12, 8), anchor=tk.W)

        # Format checkboxes
        fmt_frame = ttk.LabelFrame(dialog, text="Formats", padding=8)
        fmt_frame.pack(fill=tk.X, padx=16, pady=(0, 8))

        structural_svg_var = tk.BooleanVar(
            value=preselect in (None, "svg") and wants_structural and has_structural
        )
        tiff_var = tk.BooleanVar(value=preselect in (None, "tiff"))
        png_var = tk.BooleanVar(value=False)
        pdf_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(
            fmt_frame,
            text="Structural SVG (VTracer)",
            variable=structural_svg_var,
        ).pack(anchor=tk.W)
        ttk.Checkbutton(fmt_frame, text="TIFF (alpha matte)", variable=tiff_var).pack(anchor=tk.W)
        ttk.Checkbutton(fmt_frame, text="PNG (transparent)", variable=png_var).pack(anchor=tk.W)
        ttk.Checkbutton(fmt_frame, text="PDF (from SVG)", variable=pdf_var).pack(anchor=tk.W)

        if not wants_structural:
            ttk.Label(
                fmt_frame,
                text="Structural SVG was not selected in Output Modes.",
                foreground="gray",
            ).pack(anchor=tk.W, pady=(2, 0))
        elif wants_structural and not has_structural:
            ttk.Label(
                fmt_frame,
                text="Process the image to generate a structural SVG.",
                foreground="gray",
            ).pack(anchor=tk.W, pady=(2, 0))

        # Output folder
        folder_frame = ttk.LabelFrame(dialog, text="Output Folder", padding=8)
        folder_frame.pack(fill=tk.X, padx=16, pady=(0, 8))

        default_dir = self._app.prefs.get(
            "output_directory",
            str(Path.home() / "Desktop" / "skiagrafia_out"),
        )
        folder_var = tk.StringVar(value=default_dir)
        folder_entry = ttk.Entry(folder_frame, textvariable=folder_var)
        folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        def _browse() -> None:
            path = filedialog.askdirectory(initialdir=folder_var.get())
            if path:
                folder_var.set(path)

        ttk.Button(folder_frame, text="Browse", command=_browse).pack(
            side=tk.RIGHT, padx=(4, 0)
        )

        # Status label
        status_label = ttk.Label(dialog, text="", foreground="gray")
        status_label.pack(padx=16, anchor=tk.W)

        # Buttons
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(padx=16, pady=8, fill=tk.X)

        def _do_export() -> None:
            import shutil

            out_dir = Path(folder_var.get())
            out_dir.mkdir(parents=True, exist_ok=True)

            stem = Path(left._image_path).stem if left._image_path else "export"
            exported: list[str] = []
            first_svg_content: str | None = None

            def _copy_if_exists(src_path: str | None, label: str) -> None:
                nonlocal first_svg_content
                if not src_path:
                    return
                src = Path(src_path)
                if not src.exists():
                    return
                dst = out_dir / src.name
                if src.resolve() != dst.resolve():
                    shutil.copy2(src, dst)
                exported.append(f"{label} → {dst.name}")
                if first_svg_content is None and src.suffix.lower() == ".svg":
                    first_svg_content = src.read_text(encoding="utf-8")

            if result and structural_svg_var.get():
                _copy_if_exists(getattr(result, "svg_path", None), "Structural SVG")

            if result and hasattr(result, "tiff_path") and result.tiff_path and tiff_var.get():
                tiff_dir = Path(result.tiff_path)
                if tiff_dir.is_dir():
                    for f in tiff_dir.glob("*.tiff"):
                        tiff_dst = out_dir / f.name
                        if f.resolve() == tiff_dst.resolve():
                            exported.append(f"TIFF → {f.name}")
                        else:
                            shutil.copy2(f, tiff_dst)
                            exported.append(f"TIFF → {f.name}")
                elif tiff_dir.is_file():
                    tiff_dst = out_dir / tiff_dir.name
                    if tiff_dir.resolve() == tiff_dst.resolve():
                        exported.append(f"TIFF → {tiff_dir.name}")
                    else:
                        import shutil
                        shutil.copy2(tiff_dir, tiff_dst)
                        exported.append(f"TIFF → {tiff_dir.name}")

            if pdf_var.get() and first_svg_content:
                pdf_path = write_pdf(first_svg_content, out_dir / f"{stem}.pdf")
                exported.append(f"PDF → {pdf_path.name}")

            if not exported:
                status_label.config(
                    text="Nothing to export — process an image first.",
                    foreground="red",
                )
                return

            status_label.config(
                text=f"Exported {len(exported)} file(s) to {out_dir.name}",
                foreground="green",
            )
            logger.info("Exported: %s", exported)

            # Auto-close after 1.5s
            dialog.after(1500, dialog.destroy)

        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(
            side=tk.RIGHT, padx=(4, 0)
        )
        ttk.Button(btn_frame, text="Export", command=_do_export, default="active").pack(
            side=tk.RIGHT,
        )

    def _create_template(self) -> None:
        """Open the 'Use as batch template' dialog."""
        from core.batch_template import BatchTemplate

        left = self._view.left_panel
        params = left.get_parameters()
        labels = left.get_confirmed_labels()

        # Build parent/child structure
        parents = [l for l in self._layers if l.get("role") == "parent"]
        children_map: dict[str, list[str]] = {}
        for l in self._layers:
            if l.get("role") == "child" and l.get("parent_label"):
                children_map.setdefault(l["parent_label"], []).append(l["label"])

        # Dialog
        dialog = tk.Toplevel(self._root)
        dialog.title("Save as Batch Template")
        dialog.geometry("340x180")
        dialog.transient(self._root)
        dialog.grab_set()

        ttk.Label(dialog, text="Template name:").pack(padx=12, pady=(12, 2), anchor=tk.W)
        name_var = tk.StringVar(value=Path(left._image_path or "template").stem if left._image_path else "template")
        ttk.Entry(dialog, textvariable=name_var, width=35).pack(padx=12, pady=2)

        svg_summary = left.get_svg_mode_label()
        summary = (
            f"{len(parents)} parent labels \u00b7 "
            f"{sum(len(v) for v in children_map.values())} children \u00b7 "
            f"{svg_summary}"
        )
        ttk.Label(dialog, text=summary, foreground="gray").pack(padx=12, pady=4)

        def _save_and_switch() -> None:
            template = BatchTemplate(
                name=name_var.get(),
                source_image=left._image_path or "",
                confirmed_labels=labels,
                confirmed_children=children_map,
                output_mode=params.get("output_mode", "vector+bitmap").lower().replace(" ", ""),
                recursion_depth=params.get("recursion_depth", 2),
                corner_threshold=params.get("corner_threshold", 60),
                speckle=params.get("speckle", 8),
                smoothing=params.get("smoothing", 5),
                length_threshold=params.get("length_threshold", 4.0),
                vtracer_quality="balanced",
            )
            template.save()
            dialog.destroy()
            self._app.switch_to_batch()

        def _save_only() -> None:
            template = BatchTemplate(
                name=name_var.get(),
                source_image=left._image_path or "",
                confirmed_labels=labels,
                confirmed_children=children_map,
                output_mode=params.get("output_mode", "vector+bitmap").lower().replace(" ", ""),
                recursion_depth=params.get("recursion_depth", 2),
                corner_threshold=params.get("corner_threshold", 60),
                speckle=params.get("speckle", 8),
                smoothing=params.get("smoothing", 5),
                length_threshold=params.get("length_threshold", 4.0),
                vtracer_quality="balanced",
            )
            template.save()
            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=8)
        ttk.Button(btn_frame, text="Save & switch to batch", command=_save_and_switch).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(btn_frame, text="Save only", command=_save_only).pack(
            side=tk.LEFT, padx=4
        )
