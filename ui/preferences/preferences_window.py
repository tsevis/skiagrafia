from __future__ import annotations

import logging
import os
import subprocess
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import TYPE_CHECKING

from ui.preferences.guide_editor import GuideEditorTab
from ui.theme import is_macos
from utils.model_manager import ModelManager, REGISTRY
from utils.preferences import DEFAULT_MODELS_DIR, get_models_dir, load_preferences, save_preferences

if TYPE_CHECKING:
    from ui.main_window import MainWindow

logger = logging.getLogger(__name__)


class PreferencesWindow:
    """Five-tab preferences modal (520x480)."""

    def __init__(self, app: MainWindow) -> None:
        self._app = app
        self._prefs = dict(app.prefs)

        self._win = tk.Toplevel(app.root)
        self._win.title("Preferences")
        self._win.geometry("820x560")
        self._win.resizable(False, False)
        self._win.transient(app.root)
        self._win.grab_set()

        self._notebook = ttk.Notebook(self._win)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 0))

        self._build_general_tab()
        self._build_models_tab()
        self._build_pipeline_tab()
        self._build_appearance_tab()
        self._build_templates_tab()
        self._build_domain_guides_tab()

        # Bottom buttons
        btn_frame = ttk.Frame(self._win)
        btn_frame.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(btn_frame, text="Cancel", command=self._win.destroy).pack(
            side=tk.RIGHT, padx=(4, 0)
        )
        ttk.Button(btn_frame, text="Save", command=self._save, default="active").pack(
            side=tk.RIGHT
        )

    # ── Tab 1: General ─────────────────────────────────────────

    def _build_general_tab(self) -> None:
        tab = ttk.Frame(self._notebook, padding=12)
        self._notebook.add(tab, text="  General  ")

        # Output directory
        ttk.Label(tab, text="Default output directory").pack(anchor=tk.W, pady=(0, 2))
        dir_frame = ttk.Frame(tab)
        dir_frame.pack(fill=tk.X, pady=(0, 8))
        self._output_dir_var = tk.StringVar(value=self._prefs.get("output_directory", ""))
        ttk.Entry(dir_frame, textvariable=self._output_dir_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(dir_frame, text="Browse", command=self._browse_output_dir).pack(
            side=tk.RIGHT, padx=(4, 0)
        )

        # Default mode
        ttk.Label(tab, text="Default mode on launch").pack(anchor=tk.W, pady=(8, 2))
        self._mode_var = tk.StringVar(value=self._prefs.get("default_mode", "single"))
        ttk.Combobox(
            tab,
            textvariable=self._mode_var,
            values=["single", "batch"],
            state="readonly",
            width=20,
        ).pack(anchor=tk.W, pady=(0, 8))

        # Checkboxes
        self._save_session_var = tk.BooleanVar(
            value=self._prefs.get("save_session_on_quit", True)
        )
        ttk.Checkbutton(
            tab, text="Save session on quit", variable=self._save_session_var
        ).pack(anchor=tk.W, pady=2)

        self._notifications_var = tk.BooleanVar(
            value=self._prefs.get("show_notifications", True)
        )
        ttk.Checkbutton(
            tab, text="Show processing notifications", variable=self._notifications_var
        ).pack(anchor=tk.W, pady=2)

    def _browse_output_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self._output_dir_var.set(path)

    # ── Tab 2: Models & Ollama ─────────────────────────────────

    def _build_models_tab(self) -> None:
        tab = ttk.Frame(self._notebook, padding=12)
        self._notebook.add(tab, text="  Models  ")

        # Ollama URL
        ttk.Label(tab, text="Ollama server URL").pack(anchor=tk.W, pady=(0, 2))
        self._ollama_url_var = tk.StringVar(
            value=self._prefs.get("ollama_url", "http://localhost:11434")
        )
        ttk.Entry(tab, textvariable=self._ollama_url_var).pack(
            fill=tk.X, pady=(0, 4)
        )

        # Ollama model
        ttk.Label(tab, text="Ollama model name").pack(anchor=tk.W, pady=(4, 2))
        self._ollama_model_var = tk.StringVar(
            value=self._prefs.get("ollama_model", "moondream")
        )
        ttk.Combobox(
            tab,
            textvariable=self._ollama_model_var,
            values=["moondream", "minicpm-v", "llava:7b"],
            width=20,
        ).pack(anchor=tk.W, pady=(0, 4))

        ttk.Label(tab, text="Preferred fallback VLM").pack(anchor=tk.W, pady=(4, 2))
        self._fallback_vlm_var = tk.StringVar(
            value=self._prefs.get("preferred_fallback_vlm", "minicpm-v")
        )
        ttk.Combobox(
            tab,
            textvariable=self._fallback_vlm_var,
            values=["minicpm-v", "llava:7b", "moondream"],
            state="readonly",
            width=20,
        ).pack(anchor=tk.W, pady=(0, 4))

        ttk.Label(tab, text="Text reasoner").pack(anchor=tk.W, pady=(4, 2))
        self._reasoner_var = tk.StringVar(
            value=self._prefs.get("preferred_text_reasoner", "qwen3.5")
        )
        ttk.Combobox(
            tab,
            textvariable=self._reasoner_var,
            values=["qwen3.5", "gpt-oss:20b", "qwen3-coder:30b"],
            state="readonly",
            width=20,
        ).pack(anchor=tk.W, pady=(0, 4))

        # Test connection
        test_frame = ttk.Frame(tab)
        test_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(
            test_frame, text="Test connection", command=self._test_ollama
        ).pack(side=tk.LEFT)
        self._test_result = ttk.Label(test_frame, text="")
        self._test_result.pack(side=tk.LEFT, padx=8)

        # Model library directory
        ttk.Separator(tab, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        ttk.Label(tab, text="Model library directory").pack(anchor=tk.W, pady=(0, 2))
        path_frame = ttk.Frame(tab)
        path_frame.pack(fill=tk.X, pady=(0, 4))
        self._models_dir_var = tk.StringVar(
            value=self._prefs.get("models_directory", "")
            or str(get_models_dir(self._prefs))
        )
        ttk.Entry(
            path_frame,
            textvariable=self._models_dir_var,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(
            path_frame,
            text="Browse\u2026",
            command=self._browse_models_dir,
        ).pack(side=tk.RIGHT, padx=(4, 0))
        btn_frame2 = ttk.Frame(tab)
        btn_frame2.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(
            btn_frame2,
            text="Reset to default",
            command=self._reset_models_dir,
        ).pack(side=tk.LEFT)
        ttk.Button(
            btn_frame2,
            text="Open in Finder",
            command=lambda: subprocess.run(
                ["open", str(self._get_current_models_dir())], check=False
            ),
        ).pack(side=tk.LEFT, padx=(4, 0))

        # Installed models treeview
        ttk.Label(tab, text="Installed models").pack(anchor=tk.W, pady=(8, 2))
        self._models_tree = ttk.Treeview(
            tab,
            columns=("name", "size", "status"),
            show="headings",
            height=4,
        )
        self._models_tree.heading("name", text="Name")
        self._models_tree.heading("size", text="Size")
        self._models_tree.heading("status", text="Status")
        self._models_tree.column("name", width=220)
        self._models_tree.column("size", width=80)
        self._models_tree.column("status", width=80)
        self._models_tree.pack(fill=tk.X)

        self._scan_models()

        ttk.Button(
            tab, text="Download missing", command=self._download_missing
        ).pack(anchor=tk.W, pady=(4, 0))

    def _get_current_models_dir(self) -> Path:
        """Resolve the current models directory from the UI entry."""
        custom = self._models_dir_var.get().strip()
        if custom:
            return Path(custom)
        return get_models_dir(self._prefs)

    def _browse_models_dir(self) -> None:
        path = filedialog.askdirectory(initialdir=str(self._get_current_models_dir()))
        if path:
            self._models_dir_var.set(path)
            self._scan_models()

    def _reset_models_dir(self) -> None:
        self._models_dir_var.set(str(DEFAULT_MODELS_DIR))
        self._scan_models()

    def _scan_models(self) -> None:
        for item in self._models_tree.get_children():
            self._models_tree.delete(item)

        mgr = ModelManager(self._get_current_models_dir())
        for info in mgr.scan():
            if info.size_bytes is not None:
                size = f"{info.size_bytes / 1e6:.0f} MB"
            else:
                size = "\u2014"
            status_text = "Installed" if info.status == "ready" else "Missing"
            self._models_tree.insert(
                "", tk.END, values=(info.display_name, size, status_text)
            )

        # Also scan for any extra .pth/.pt/.bin files
        models_dir = self._get_current_models_dir()
        for ext in ("*.pth", "*.pt", "*.bin"):
            for p in models_dir.glob(ext):
                if p.name not in REGISTRY:
                    size = f"{p.stat().st_size / 1e6:.0f} MB"
                    self._models_tree.insert(
                        "", tk.END, values=(p.name, size, "Extra")
                    )

    def _test_ollama(self) -> None:
        from models.moondream_client import MoondreamClient

        client = MoondreamClient(
            host=self._ollama_url_var.get(),
            model=self._ollama_model_var.get(),
        )
        if client.health_check():
            self._test_result.config(text="\u2713 Connected", foreground="green")
        else:
            self._test_result.config(text="\u2715 Failed", foreground="red")

    def _download_missing(self) -> None:
        mgr = ModelManager(self._get_current_models_dir())
        for name in REGISTRY:
            if not mgr.is_available(name):
                logger.info("Downloading %s...", name)
                try:
                    mgr.ensure(name)
                except Exception:
                    logger.error("Failed to download %s", name, exc_info=True)
        self._scan_models()

    # ── Tab 3: Pipeline ────────────────────────────────────────

    def _build_pipeline_tab(self) -> None:
        tab = ttk.Frame(self._notebook, padding=12)
        self._notebook.add(tab, text="  Pipeline  ")

        self._sam_box_var = tk.DoubleVar(
            value=self._prefs.get("sam_box_threshold", 0.35)
        )
        self._sam_text_var = tk.DoubleVar(
            value=self._prefs.get("sam_text_threshold", 0.25)
        )
        self._vt_corner_var = tk.IntVar(
            value=self._prefs.get("vtracer_corner_threshold", 60)
        )
        self._vt_speckle_var = tk.IntVar(
            value=self._prefs.get("vtracer_speckle", 8)
        )
        self._vt_length_var = tk.DoubleVar(
            value=self._prefs.get("vtracer_length_threshold", 4.0)
        )
        self._bilateral_var = tk.IntVar(
            value=self._prefs.get("bilateral_filter_d", 9)
        )
        self._workers_var = tk.IntVar(
            value=self._prefs.get("max_cpu_workers", os.cpu_count() or 4)
        )
        self._adaptive_var = tk.BooleanVar(
            value=self._prefs.get("enable_adaptive_interrogation", True)
        )
        self._interrogation_profile_var = tk.StringVar(
            value=self._prefs.get("interrogation_profile", "balanced")
        )
        self._fallback_mode_var = tk.StringVar(
            value=self._prefs.get("interrogation_fallback_mode", "adaptive_auto")
        )
        self._tiled_fallback_var = tk.BooleanVar(
            value=self._prefs.get("enable_tiled_fallback", True)
        )

        sliders: list[tuple[str, tk.Variable, float, float, bool]] = [
            ("SAM box threshold", self._sam_box_var, 0.1, 0.9, True),
            ("SAM text threshold", self._sam_text_var, 0.1, 0.9, True),
            ("VTracer corner threshold", self._vt_corner_var, 30, 90, False),
            ("VTracer speckle", self._vt_speckle_var, 2, 20, False),
            ("VTracer length threshold", self._vt_length_var, 2.0, 8.0, True),
            ("Bilateral filter d", self._bilateral_var, 3, 15, False),
        ]

        for label_text, var, from_, to_, is_float in sliders:
            row = ttk.Frame(tab)
            row.pack(fill=tk.X, pady=3)
            ttk.Label(row, text=label_text, width=24).pack(side=tk.LEFT)
            fmt = f"{var.get():.2f}" if is_float else str(int(var.get()))
            value_label = ttk.Label(
                row,
                text=fmt,
                width=5,
                font=("Menlo", 10) if is_macos() else ("Consolas", 9),
                anchor=tk.E,
            )
            value_label.pack(side=tk.RIGHT)
            ttk.Scale(
                row,
                variable=var,
                from_=from_,
                to=to_,
                command=lambda val, vl=value_label, fl=is_float: vl.config(
                    text=f"{float(val):.2f}" if fl else str(int(float(val)))
                ),
            ).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=4)

        # Max CPU workers (spinbox)
        workers_row = ttk.Frame(tab)
        workers_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(workers_row, text="Max CPU workers", width=24).pack(side=tk.LEFT)
        ttk.Spinbox(
            workers_row,
            textvariable=self._workers_var,
            from_=1,
            to=os.cpu_count() or 20,
            width=5,
        ).pack(side=tk.RIGHT)

        ttk.Separator(tab, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Checkbutton(
            tab,
            text="Enable adaptive interrogation",
            variable=self._adaptive_var,
        ).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(
            tab,
            text="Enable tiled fallback",
            variable=self._tiled_fallback_var,
        ).pack(anchor=tk.W, pady=2)
        profile_row = ttk.Frame(tab)
        profile_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(profile_row, text="Interrogation profile", width=24).pack(side=tk.LEFT)
        ttk.Combobox(
            profile_row,
            textvariable=self._interrogation_profile_var,
            values=["fast", "balanced", "deep"],
            state="readonly",
            width=18,
        ).pack(side=tk.RIGHT)
        mode_row = ttk.Frame(tab)
        mode_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(mode_row, text="Fallback mode", width=24).pack(side=tk.LEFT)
        ttk.Combobox(
            mode_row,
            textvariable=self._fallback_mode_var,
            values=["adaptive_auto", "always_enrich", "moondream_only"],
            state="readonly",
            width=18,
        ).pack(side=tk.RIGHT)

    # ── Tab 4: Appearance ──────────────────────────────────────

    def _build_appearance_tab(self) -> None:
        tab = ttk.Frame(self._notebook, padding=12)
        self._notebook.add(tab, text="  Appearance  ")

        # Theme
        ttk.Label(tab, text="Theme").pack(anchor=tk.W, pady=(0, 2))
        self._theme_var = tk.StringVar(value=self._prefs.get("theme", "auto"))
        ttk.Combobox(
            tab,
            textvariable=self._theme_var,
            values=["auto", "light", "dark"],
            state="readonly",
            width=12,
        ).pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(tab, text="Scan preview defaults").pack(anchor=tk.W, pady=(6, 2))
        self._scan_boxes_var = tk.BooleanVar(
            value=self._prefs.get("scan_preview_show_boxes", True)
        )
        self._scan_labels_var = tk.BooleanVar(
            value=self._prefs.get("scan_preview_show_labels", True)
        )
        self._scan_heatmap_var = tk.BooleanVar(
            value=self._prefs.get("scan_preview_show_heatmap", True)
        )
        ttk.Checkbutton(tab, text="Show boxes after scan", variable=self._scan_boxes_var).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(tab, text="Show labels after scan", variable=self._scan_labels_var).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(tab, text="Show heatmap after scan", variable=self._scan_heatmap_var).pack(anchor=tk.W, pady=2)

        ttk.Label(tab, text="Box opacity").pack(anchor=tk.W, pady=(8, 2))
        self._scan_box_opacity_var = tk.IntVar(
            value=self._prefs.get("scan_preview_box_opacity", 40)
        )
        box_opacity_row = ttk.Frame(tab)
        box_opacity_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Scale(
            box_opacity_row,
            variable=self._scan_box_opacity_var,
            from_=0,
            to=100,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(
            box_opacity_row,
            textvariable=self._scan_box_opacity_var,
            width=4,
        ).pack(side=tk.RIGHT)

        ttk.Label(tab, text="Heatmap opacity").pack(anchor=tk.W, pady=(0, 2))
        self._scan_heatmap_opacity_var = tk.IntVar(
            value=self._prefs.get("scan_preview_heatmap_opacity", 40)
        )
        heatmap_opacity_row = ttk.Frame(tab)
        heatmap_opacity_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Scale(
            heatmap_opacity_row,
            variable=self._scan_heatmap_opacity_var,
            from_=0,
            to=100,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(
            heatmap_opacity_row,
            textvariable=self._scan_heatmap_opacity_var,
            width=4,
        ).pack(side=tk.RIGHT)

        # Canvas background
        ttk.Label(tab, text="Canvas background").pack(anchor=tk.W, pady=(0, 2))
        canvas_frame = ttk.Frame(tab)
        canvas_frame.pack(fill=tk.X, pady=(0, 8))
        self._canvas_bg_var = tk.StringVar(
            value=self._prefs.get("canvas_background", "#1a1a1a")
        )
        ttk.Entry(canvas_frame, textvariable=self._canvas_bg_var, width=10).pack(
            side=tk.LEFT
        )
        self._canvas_swatch = tk.Label(
            canvas_frame,
            text="  ",
            bg=self._canvas_bg_var.get(),
            width=3,
            relief="solid",
            borderwidth=1,
        )
        self._canvas_swatch.pack(side=tk.LEFT, padx=4)
        self._canvas_bg_var.trace_add(
            "write",
            lambda *_: self._update_swatch(),
        )

        # Mask overlay opacity
        ttk.Label(tab, text="Mask overlay opacity").pack(anchor=tk.W, pady=(0, 2))
        self._mask_opacity_var = tk.IntVar(
            value=self._prefs.get("mask_overlay_opacity", 30)
        )
        opacity_row = ttk.Frame(tab)
        opacity_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Scale(
            opacity_row,
            variable=self._mask_opacity_var,
            from_=0,
            to=100,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(
            opacity_row,
            textvariable=self._mask_opacity_var,
            width=4,
        ).pack(side=tk.RIGHT)

        # Vector overlay colour
        ttk.Label(tab, text="Vector overlay colour").pack(anchor=tk.W, pady=(0, 2))
        self._vector_colour_var = tk.StringVar(
            value=self._prefs.get("vector_overlay_colour", "Blue")
        )
        ttk.Combobox(
            tab,
            textvariable=self._vector_colour_var,
            values=["Blue", "Red", "Green", "White", "Yellow"],
            state="readonly",
            width=12,
        ).pack(anchor=tk.W)

    def _update_swatch(self) -> None:
        try:
            self._canvas_swatch.config(bg=self._canvas_bg_var.get())
        except tk.TclError:
            pass

    # ── Tab 5: Templates ───────────────────────────────────────

    def _build_templates_tab(self) -> None:
        tab = ttk.Frame(self._notebook, padding=12)
        self._notebook.add(tab, text="  Templates  ")

        self._templates_tree = ttk.Treeview(
            tab,
            columns=("name", "labels", "mode", "created"),
            show="headings",
            height=8,
        )
        self._templates_tree.heading("name", text="Name")
        self._templates_tree.heading("labels", text="Labels")
        self._templates_tree.heading("mode", text="Mode")
        self._templates_tree.heading("created", text="Created")
        self._templates_tree.column("name", width=120)
        self._templates_tree.column("labels", width=60)
        self._templates_tree.column("mode", width=100)
        self._templates_tree.column("created", width=140)
        self._templates_tree.pack(fill=tk.BOTH, expand=True)

        self._load_templates()

        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(btn_frame, text="Load", command=self._load_template).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(btn_frame, text="Delete", command=self._delete_template).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(
            btn_frame, text="Reveal in Finder", command=self._reveal_templates
        ).pack(side=tk.LEFT, padx=(4, 0))

    def _load_templates(self) -> None:
        from core.batch_template import BatchTemplate

        for item in self._templates_tree.get_children():
            self._templates_tree.delete(item)

        for t in BatchTemplate.list_all():
            n_labels = len(t.confirmed_labels)
            self._templates_tree.insert(
                "",
                tk.END,
                values=(t.name, n_labels, t.output_mode, t.created_at[:19]),
            )

    def _load_template(self) -> None:
        logger.info("Loading template")

    def _delete_template(self) -> None:
        logger.info("Deleting template")

    def _reveal_templates(self) -> None:
        templates_dir = Path.home() / ".config" / "skiagrafia" / "templates"
        templates_dir.mkdir(parents=True, exist_ok=True)
        if is_macos():
            subprocess.run(["open", str(templates_dir)], check=False)

    # ── Tab 6: Domain Guides ──────────────────────────────────

    def _build_domain_guides_tab(self) -> None:
        tab = ttk.Frame(self._notebook, padding=0)
        self._notebook.add(tab, text="  Domain Guides  ")
        self._guide_editor = GuideEditorTab(tab)

    # ── Save ───────────────────────────────────────────────────

    def _save(self) -> None:
        # Resolve models_directory: store empty string if it matches the default
        models_dir_val = self._models_dir_var.get().strip()
        if models_dir_val == str(DEFAULT_MODELS_DIR):
            models_dir_val = ""

        prefs = {
            "output_directory": self._output_dir_var.get(),
            "default_mode": self._mode_var.get(),
            "save_session_on_quit": self._save_session_var.get(),
            "show_notifications": self._notifications_var.get(),
            "ollama_url": self._ollama_url_var.get(),
            "ollama_model": self._ollama_model_var.get(),
            "models_directory": models_dir_val,
            "preferred_fallback_vlm": self._fallback_vlm_var.get(),
            "preferred_text_reasoner": self._reasoner_var.get(),
            "sam_box_threshold": self._sam_box_var.get(),
            "sam_text_threshold": self._sam_text_var.get(),
            "vtracer_corner_threshold": self._vt_corner_var.get(),
            "vtracer_speckle": self._vt_speckle_var.get(),
            "vtracer_length_threshold": self._vt_length_var.get(),
            "bilateral_filter_d": self._bilateral_var.get(),
            "max_cpu_workers": self._workers_var.get(),
            "enable_adaptive_interrogation": self._adaptive_var.get(),
            "interrogation_profile": self._interrogation_profile_var.get(),
            "interrogation_fallback_mode": self._fallback_mode_var.get(),
            "enable_tiled_fallback": self._tiled_fallback_var.get(),
            "theme": self._theme_var.get(),
            "canvas_background": self._canvas_bg_var.get(),
            "mask_overlay_opacity": self._mask_opacity_var.get(),
            "vector_overlay_colour": self._vector_colour_var.get(),
            "scan_preview_show_boxes": self._scan_boxes_var.get(),
            "scan_preview_show_labels": self._scan_labels_var.get(),
            "scan_preview_show_heatmap": self._scan_heatmap_var.get(),
            "scan_preview_box_opacity": self._scan_box_opacity_var.get(),
            "scan_preview_heatmap_opacity": self._scan_heatmap_opacity_var.get(),
        }
        save_preferences(prefs)
        self._app.apply_preferences(prefs)
        self._win.destroy()
        logger.info("Preferences saved")
