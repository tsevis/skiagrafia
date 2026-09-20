from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import TYPE_CHECKING, Protocol, cast

from core.knowledge import load_knowledge_pack

if TYPE_CHECKING:
    from core.batch_template import BatchTemplate
    from ui.batch.batch_view import BatchView
    from ui.batch.steps.step_progress import StepProgress

    class _DnDWidget(Protocol):
        """Shape tkinterdnd2 monkey-patches onto every BaseWidget instance.

        tkinterdnd2 adds these methods to `tkinter.BaseWidget` at import
        time (see `TkinterDnD._require`), so a plain `ttk.Label` gains them
        only at runtime — this Protocol expresses that real, dynamically
        acquired shape for the type checker.
        """

        def drop_target_register(self, *dndtypes: str) -> None: ...
        def dnd_bind(self, sequence: str, func: object) -> object: ...

    class _DropEvent(Protocol):
        """Shape of a tkinterdnd2 `<<Drop>>` event: `DnDEvent` is declared
        with no attributes (they're set dynamically), so a Protocol
        expresses the one field this module actually reads."""

        data: str

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
            self._show_template_banner(cast("BatchTemplate", view.template))

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
            drop_target = cast("_DnDWidget", self._drop_zone)
            drop_target.drop_target_register("DND_Files")
            drop_target.dnd_bind("<<Drop>>", self._on_drop)
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
        self._recent_paths: dict[str, Path] = {}
        self._resumable_items: set[str] = set()
        self._recent_list.bind("<Double-1>", self._load_selected_run)
        self._recent_list.bind("<<TreeviewSelect>>", self._update_resume_button)
        ttk.Button(
            self.frame,
            text="Load selected run into Configure",
            command=self._load_selected_run,
        ).pack(anchor=tk.W, pady=(6, 0))
        self._resume_btn = ttk.Button(
            self.frame,
            text="Resume selected batch",
            command=self._resume_selected_run,
            state="disabled",
        )
        self._resume_btn.pack(anchor=tk.W, pady=(4, 0))

        self._scan_recent_batches()

    def _on_drop(self, event: _DropEvent) -> None:
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

    def _show_template_banner(self, template: BatchTemplate) -> None:
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
        for item in self._recent_list.get_children():
            self._recent_list.delete(item)
        self._recent_paths.clear()
        self._resumable_items.clear()
        self._resume_btn.config(state="disabled")
        try:
            from core.batch_session import load_resumable_batch, load_run_settings

            run_files = sorted(
                output_dir.glob("*/run.json"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            for run_file in run_files[:5]:
                try:
                    settings = load_run_settings(run_file)
                except Exception:
                    logger.warning("Skipping unreadable batch run %s", run_file, exc_info=True)
                    continue
                resumable = load_resumable_batch(run_file.parent)
                status = "Resume ready" if resumable is not None else "Saved run"
                item = self._recent_list.insert(
                    "",
                    tk.END,
                    values=(
                        Path(settings.input_folder).name or settings.batch_id,
                        status,
                        settings.created_at[:19],
                    ),
                )
                self._recent_paths[item] = run_file
                if resumable is not None:
                    self._resumable_items.add(item)
        except OSError:
            logger.warning("Could not scan recent batch runs", exc_info=True)

    def _load_selected_run(self, _event: object = None) -> None:
        selected = self._recent_list.selection()
        if not selected:
            return
        run_path = self._recent_paths.get(selected[0])
        if run_path is None:
            return
        from core.batch_session import load_run_settings

        try:
            settings = load_run_settings(run_path)
        except Exception:
            logger.warning("Could not load batch run %s", run_path, exc_info=True)
            return
        # Folder discovery runs first, then the historical context takes
        # precedence so request and guide metadata are restored faithfully.
        if settings.input_folder and Path(settings.input_folder).is_dir():
            self._set_folder(settings.input_folder)
        self._view.load_run_settings(settings)
        self._view.go_to_step(1)

    def _update_resume_button(self, _event: object = None) -> None:
        selected = self._recent_list.selection()
        state = "normal" if selected and selected[0] in self._resumable_items else "disabled"
        self._resume_btn.config(state=state)

    def _resume_selected_run(self) -> None:
        """Resume only a freshly revalidated, fully triaged GUI batch."""
        selected = self._recent_list.selection()
        if not selected:
            return
        run_path = self._recent_paths.get(selected[0])
        if run_path is None:
            return
        from core.batch_session import load_resumable_batch

        resumable = load_resumable_batch(run_path.parent)
        if resumable is None:
            self._resume_btn.config(state="disabled")
            logger.warning("Selected run is no longer safe to resume: %s", run_path)
            return
        if resumable.run_settings.input_folder and Path(
            resumable.run_settings.input_folder
        ).is_dir():
            self._set_folder(resumable.run_settings.input_folder)
        self._view.resume_run(resumable)
        self._view.go_to_step(4)
        progress_step = self._view._step_views[4]
        if progress_step and hasattr(progress_step, "resume_existing"):
            cast("StepProgress", progress_step).resume_existing()

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
