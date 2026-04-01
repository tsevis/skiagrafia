from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import TYPE_CHECKING

from core.batch_template import BatchTemplate
from core.knowledge import load_knowledge_pack
from core.state_manager import StateManager

if TYPE_CHECKING:
    from ui.batch.batch_view import BatchView

logger = logging.getLogger(__name__)


class StepImport:
    """Step 1 — Import: drop zone for folder + recent batches list."""

    def __init__(self, parent: tk.Widget, view: BatchView) -> None:
        self._view = view
        self._app = view.app
        self._root = view.root
        self._input_folder: str | None = None

        self.frame = ttk.Frame(parent, padding=16)

        # Template banner (hidden unless template active)
        self._template_banner = tk.Frame(self.frame, bg="#D1FAE5", padx=10, pady=6)
        self._template_banner_label = tk.Label(
            self._template_banner,
            text="",
            bg="#D1FAE5",
            fg="#065F46",
            anchor=tk.W,
        )
        self._template_banner_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        dismiss_btn = tk.Label(
            self._template_banner,
            text="\u00d7",
            bg="#D1FAE5",
            fg="#065F46",
            cursor="hand2",
            font=("", 14),
        )
        dismiss_btn.pack(side=tk.RIGHT)
        dismiss_btn.bind("<Button-1>", lambda e: self._dismiss_template())

        if view.template is not None:
            self._show_template_banner(view.template)

        # Title
        ttk.Label(
            self.frame,
            text="Import Images",
            font=("SF Pro Display", 16, "bold"),
        ).pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(
            self.frame,
            text="Drop a folder of images to begin batch processing.",
            foreground="gray",
        ).pack(anchor=tk.W, pady=(0, 12))

        # Drop zone
        self._drop_zone = ttk.Label(
            self.frame,
            text="Drop folder here\nor click to browse",
            anchor=tk.CENTER,
            justify=tk.CENTER,
            relief="groove",
            padding=40,
        )
        self._drop_zone.pack(fill=tk.X, pady=(0, 12))

        try:
            self._drop_zone.drop_target_register("DND_Files")
            self._drop_zone.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            logger.warning("tkinterdnd2 not available — drag-and-drop disabled")

        self._drop_zone.bind("<Button-1>", self._browse_folder)

        # Folder info (hidden until folder selected)
        self._folder_info = ttk.Frame(self.frame)
        self._folder_label = ttk.Label(self._folder_info, text="")
        self._folder_label.pack(anchor=tk.W)
        self._count_label = ttk.Label(self._folder_info, text="", foreground="gray")
        self._count_label.pack(anchor=tk.W)
        self._guide_label = ttk.Label(self._folder_info, text="", foreground="gray")
        self._guide_label.pack(anchor=tk.W)

        # Recent batches section
        ttk.Separator(self.frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)
        ttk.Label(
            self.frame,
            text="Recent Batches",
            font=("SF Pro Text", 11, "bold"),
        ).pack(anchor=tk.W, pady=(0, 6))

        self._recent_list = ttk.Treeview(
            self.frame,
            columns=("folder", "status", "date"),
            show="headings",
            height=4,
        )
        self._recent_list.heading("folder", text="Folder")
        self._recent_list.heading("status", text="Status")
        self._recent_list.heading("date", text="Date")
        self._recent_list.column("folder", width=200)
        self._recent_list.column("status", width=80)
        self._recent_list.column("date", width=120)
        self._recent_list.pack(fill=tk.X)

        self._scan_recent_batches()

    def _on_drop(self, event: object) -> None:
        path = event.data.strip().strip("{}")
        if Path(path).is_dir():
            self._set_folder(path)

    def _browse_folder(self, event: object = None) -> None:
        from tkinter import filedialog

        path = filedialog.askdirectory()
        if path:
            self._set_folder(path)

    def _set_folder(self, path: str) -> None:
        self._input_folder = path
        folder = Path(path)

        extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
        images = [p for p in folder.iterdir() if p.suffix.lower() in extensions and p.is_file()]

        self._folder_label.config(text=folder.name)
        self._count_label.config(text=f"{len(images)} images found")
        self._update_knowledge_pack(folder)
        self._folder_info.pack(fill=tk.X, pady=(0, 8))
        self._drop_zone.config(text=folder.name)

        logger.info("Batch folder selected: %s (%d images)", path, len(images))

    def _show_template_banner(self, template: object) -> None:
        if hasattr(template, "name"):
            n_parents = len(getattr(template, "confirmed_labels", []))
            n_children = sum(
                len(v) for v in getattr(template, "confirmed_children", {}).values()
            )
            mode = getattr(template, "output_mode", "")
            self._template_banner_label.config(
                text=f"\u25cf  Template: {template.name}  \u00b7  "
                f"{n_parents} parents  \u00b7  {n_children} children  \u00b7  {mode}"
            )
            self._template_banner.pack(fill=tk.X, pady=(0, 8))

    def _dismiss_template(self) -> None:
        self._template_banner.pack_forget()
        self._view.template = None

    def _scan_recent_batches(self) -> None:
        output_dir = Path(
            self._app.prefs.get(
                "output_directory",
                str(Path.home() / "Desktop" / "skiagrafia_out"),
            )
        )
        incomplete = StateManager.find_incomplete_batches(output_dir)
        for db_path in incomplete[:5]:
            batch_dir = db_path.parent
            self._recent_list.insert(
                "",
                tk.END,
                values=(batch_dir.name, "Incomplete", ""),
            )

    def _update_knowledge_pack(self, folder: Path) -> None:
        pack = load_knowledge_pack(folder)
        if pack is None:
            self._view.knowledge_pack_path = None
            self._view.knowledge_pack_name = None
            self._view.knowledge_pack_notes_path = None
            self._view.knowledge_pack_defaults = {}
            self._view.knowledge_guidance_active = False
            self._guide_label.config(text="Knowledge guide: none")
            return

        notes_path = folder / "skiagrafia_guide.md"
        self._view.knowledge_pack_path = pack.path
        self._view.knowledge_pack_name = pack.name
        self._view.knowledge_pack_notes_path = str(notes_path) if notes_path.exists() else None
        self._view.knowledge_pack_defaults = pack.batch_defaults.model_dump()
        self._view.knowledge_guidance_active = True
        self._guide_label.config(
            text=f"Knowledge guide: {pack.name} ({len(pack.objects)} objects)"
        )

    @property
    def input_folder(self) -> str | None:
        return self._input_folder

    def get_image_paths(self) -> list[str]:
        """Return sorted list of image file paths from the selected folder."""
        if not self._input_folder:
            return []
        folder = Path(self._input_folder)
        extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
        return sorted(
            str(p) for p in folder.iterdir()
            if p.suffix.lower() in extensions and p.is_file()
        )
