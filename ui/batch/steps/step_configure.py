from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import TYPE_CHECKING

from core.knowledge import (
    KnowledgePack,
    build_knowledge_pack,
    default_guide_markdown,
)
from ui.theme import is_macos

if TYPE_CHECKING:
    from ui.batch.batch_view import BatchView


class StepConfigure:
    """Step 2 — Configure: output mode cards, recursion depth, VTracer quality."""

    def __init__(self, parent: tk.Widget, view: BatchView) -> None:
        self._view = view
        self._app = view.app

        self.frame = ttk.Frame(parent, padding=16)

        ttk.Label(
            self.frame,
            text="Configure Pipeline",
            font=("SF Pro Display", 16, "bold"),
        ).pack(anchor=tk.W, pady=(0, 8))

        # Template override banner
        if view.template is not None:
            override_frame = ttk.Frame(self.frame)
            override_frame.pack(fill=tk.X, pady=(0, 8))
            ttk.Label(
                override_frame,
                text="Using template settings.",
                foreground="gray",
            ).pack(side=tk.LEFT)
            ttk.Button(
                override_frame,
                text="Edit / override",
                command=self._enable_editing,
            ).pack(side=tk.RIGHT)

        # Output mode cards (checkboxes — user can enable multiple)
        ttk.Label(self.frame, text="Output Mode").pack(anchor=tk.W, pady=(8, 4))

        modes_frame = ttk.Frame(self.frame)
        modes_frame.pack(fill=tk.X, pady=(0, 12))

        mode_options = [
            ("Vector (SVG)", "vector", "Multi-layer SVG with spline paths"),
            ("Bitmap (TIFF)", "bitmap", "TIFF with VitMatte alpha mattes"),
        ]

        self._mode_vars: dict[str, tk.BooleanVar] = {}
        for i, (title, key, desc) in enumerate(mode_options):
            var = tk.BooleanVar(value=True)
            self._mode_vars[key] = var

            card = ttk.Frame(modes_frame, relief="groove", padding=8)
            card.grid(row=0, column=i, padx=4, pady=4, sticky="nsew")
            modes_frame.columnconfigure(i, weight=1)

            cb = ttk.Checkbutton(card, text=title, variable=var)
            cb.pack(anchor=tk.W)
            ttk.Label(card, text=desc, foreground="gray", wraplength=180).pack(
                anchor=tk.W, pady=(2, 0)
            )

        # Recursion depth
        depth_frame = ttk.Frame(self.frame)
        depth_frame.pack(fill=tk.X, pady=4)
        ttk.Label(depth_frame, text="Recursion Depth").pack(side=tk.LEFT)
        self._depth_var = tk.IntVar(value=2)
        depth_value = ttk.Label(
            depth_frame,
            text="2",
            font=("Menlo", 10) if is_macos() else ("Consolas", 9),
        )
        depth_value.pack(side=tk.RIGHT)
        ttk.Scale(
            depth_frame,
            variable=self._depth_var,
            from_=1,
            to=3,
            command=lambda v: depth_value.config(text=str(int(float(v)))),
        ).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=8)

        # VTracer quality
        quality_frame = ttk.Frame(self.frame)
        quality_frame.pack(fill=tk.X, pady=4)
        ttk.Label(quality_frame, text="VTracer Quality").pack(side=tk.LEFT)
        self._quality_var = tk.StringVar(value="balanced")
        ttk.Combobox(
            quality_frame,
            textvariable=self._quality_var,
            values=["draft", "balanced", "maximum"],
            state="readonly",
            width=12,
        ).pack(side=tk.RIGHT)

        self._build_interrogation_section()

        # Apply template values if available
        if view.template is not None:
            self._apply_template(view.template)

    def _apply_template(self, template: object) -> None:
        if hasattr(template, "output_mode"):
            mode = template.output_mode
            for key, var in self._mode_vars.items():
                var.set(key in mode)
        if hasattr(template, "recursion_depth"):
            self._depth_var.set(template.recursion_depth)
        if hasattr(template, "vtracer_quality"):
            self._quality_var.set(template.vtracer_quality)
        if hasattr(template, "guide_path") and template.guide_path:
            self._guide_mode_var.set(True)
        if hasattr(template, "fallback_mode"):
            self._fallback_mode_var.set(template.fallback_mode)
        if hasattr(template, "interrogation_profile") and template.interrogation_profile:
            self._profile_var.set(template.interrogation_profile)
        if hasattr(template, "preferred_vlm") and template.preferred_vlm:
            self._preferred_vlm_var.set(template.preferred_vlm)
        if hasattr(template, "text_reasoner_model") and template.text_reasoner_model:
            self._reasoner_var.set(template.text_reasoner_model)
        if hasattr(template, "enable_tiled_fallback"):
            self._tiled_fallback_var.set(template.enable_tiled_fallback)

    def _enable_editing(self) -> None:
        """Enable editing when overriding template."""
        pass

    def _build_interrogation_section(self) -> None:
        section = ttk.LabelFrame(self.frame, text="Advanced Interrogation", padding=8)
        section.pack(fill=tk.X, pady=(12, 0))

        guide_text = self._view.knowledge_pack_name or "No guide loaded"
        self._guide_status_label = ttk.Label(
            section,
            text=f"Guide: {guide_text}",
            foreground="gray",
            wraplength=520,
            justify=tk.LEFT,
        )
        self._guide_status_label.pack(anchor=tk.W, pady=(0, 6))

        guide_actions = ttk.Frame(section)
        guide_actions.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(
            guide_actions,
            text="Load guide",
            command=self._load_guide,
        ).pack(side=tk.LEFT)
        ttk.Button(
            guide_actions,
            text="Create guide",
            command=self._create_guide,
        ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(
            guide_actions,
            text="Clear guide",
            command=self._clear_guide,
        ).pack(side=tk.LEFT, padx=(6, 0))

        prefs = self._app.prefs
        guide_defaults = getattr(self._view, "knowledge_pack_defaults", {}) or {}
        self._guide_mode_var = tk.BooleanVar(
            value=bool(self._view.knowledge_pack_path)
        )
        self._fallback_mode_var = tk.StringVar(
            value=str(prefs.get("interrogation_fallback_mode", "adaptive_auto"))
        )
        self._profile_var = tk.StringVar(
            value=str(prefs.get("interrogation_profile", "balanced"))
        )
        self._preferred_vlm_var = tk.StringVar(
            value=str(
                guide_defaults.get(
                    "preferred_vlm",
                    prefs.get("preferred_fallback_vlm", "minicpm-v"),
                )
            )
        )
        self._reasoner_var = tk.StringVar(
            value=str(prefs.get("preferred_text_reasoner", "qwen3.5"))
        )
        self._tiled_fallback_var = tk.BooleanVar(
            value=bool(
                guide_defaults.get(
                    "enable_tiling",
                    prefs.get("enable_tiled_fallback", True),
                )
            )
        )

        ttk.Checkbutton(
            section,
            text="Enable guide-aware interrogation",
            variable=self._guide_mode_var,
        ).pack(anchor=tk.W, pady=1)

        row = ttk.Frame(section)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Fallback mode").pack(side=tk.LEFT)
        ttk.Combobox(
            row,
            textvariable=self._fallback_mode_var,
            values=["adaptive_auto", "always_enrich", "moondream_only"],
            state="readonly",
            width=18,
        ).pack(side=tk.RIGHT)

        row = ttk.Frame(section)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Interrogation profile").pack(side=tk.LEFT)
        ttk.Combobox(
            row,
            textvariable=self._profile_var,
            values=["fast", "balanced", "deep"],
            state="readonly",
            width=18,
        ).pack(side=tk.RIGHT)

        row = ttk.Frame(section)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Preferred fallback VLM").pack(side=tk.LEFT)
        ttk.Combobox(
            row,
            textvariable=self._preferred_vlm_var,
            values=["minicpm-v", "llava:7b", "moondream"],
            state="readonly",
            width=18,
        ).pack(side=tk.RIGHT)

        row = ttk.Frame(section)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Text reasoner").pack(side=tk.LEFT)
        ttk.Combobox(
            row,
            textvariable=self._reasoner_var,
            values=["qwen3.5", "gpt-oss:20b", "qwen3-coder:30b"],
            state="readonly",
            width=18,
        ).pack(side=tk.RIGHT)

        ttk.Checkbutton(
            section,
            text="Enable tiled fallback for hard images",
            variable=self._tiled_fallback_var,
        ).pack(anchor=tk.W, pady=(4, 0))
        self._refresh_guide_status()

    def _refresh_guide_status(self) -> None:
        if self._view.knowledge_pack_name:
            self._guide_status_label.config(
                text=f"Guide: {self._view.knowledge_pack_name}",
                foreground="gray",
            )
            self._guide_mode_var.set(True)
        else:
            self._guide_status_label.config(
                text="Guide: No guide loaded",
                foreground="gray",
            )
            self._guide_mode_var.set(False)

    def _load_guide(self) -> None:
        from tkinter import filedialog

        initial_dir = self._view._step_views[0].input_folder if self._view._step_views[0] else None
        path = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            pack = KnowledgePack.load(path)
        except Exception:
            self._guide_status_label.config(
                text="Guide: Failed to load selected TOML guide",
                foreground="red",
            )
            return
        self._view.knowledge_pack_path = path
        self._view.knowledge_pack_name = pack.name
        self._view.knowledge_pack_notes_path = (
            str(Path(path).with_suffix(".md")) if Path(path).with_suffix(".md").exists() else None
        )
        self._view.knowledge_pack_defaults = pack.batch_defaults.model_dump()
        self._view.knowledge_guidance_active = True
        self._refresh_guide_status()

    def _clear_guide(self) -> None:
        self._view.knowledge_pack_path = None
        self._view.knowledge_pack_name = None
        self._view.knowledge_pack_notes_path = None
        self._view.knowledge_pack_defaults = {}
        self._view.knowledge_guidance_active = False
        self._refresh_guide_status()

    def _create_guide(self) -> None:
        step_import = self._view._step_views[0]
        batch_folder = (
            Path(step_import.input_folder)
            if step_import and getattr(step_import, "input_folder", None)
            else None
        )
        if batch_folder is None:
            self._guide_status_label.config(
                text="Guide: Select a batch folder first in Step 1",
                foreground="#FF9F0A",
            )
            return

        dialog = tk.Toplevel(self.frame)
        dialog.title("Create Guide")
        dialog.geometry("620x520")
        dialog.transient(self.frame.winfo_toplevel())
        dialog.grab_set()

        ttk.Label(dialog, text="Domain name").pack(anchor=tk.W, padx=12, pady=(12, 4))
        domain_var = tk.StringVar(value=batch_folder.name)
        ttk.Entry(dialog, textvariable=domain_var).pack(fill=tk.X, padx=12)

        ttk.Label(dialog, text="Description").pack(anchor=tk.W, padx=12, pady=(10, 4))
        description_var = tk.StringVar(
            value="Collection-specific objects expected in this batch."
        )
        ttk.Entry(dialog, textvariable=description_var).pack(fill=tk.X, padx=12)

        ttk.Label(
            dialog,
            text="Objects, one per line: canonical | aliases | generic terms | description | parts",
            foreground="gray",
            wraplength=580,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=12, pady=(10, 4))

        objects_text = tk.Text(dialog, height=12, wrap="word")
        objects_text.pack(fill=tk.BOTH, expand=True, padx=12)
        objects_text.insert(
            "1.0",
            "chalice | communion cup, goblet | cup, vessel | ornate metal cup on stem | rim, stem, base\n"
            "cross | blessing cross, processional cross | cross, metal cross | ornate cross with central vertical shaft | top, arms, base\n",
        )

        ttk.Label(
            dialog,
            text="Optional notes for skiagrafia_guide.md",
            foreground="gray",
        ).pack(anchor=tk.W, padx=12, pady=(10, 4))
        notes_text = tk.Text(dialog, height=6, wrap="word")
        notes_text.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))
        notes_text.insert("1.0", default_guide_markdown(domain_var.get()))

        status_label = ttk.Label(dialog, text="", foreground="gray")
        status_label.pack(anchor=tk.W, padx=12, pady=(0, 8))

        def _parse_objects(raw_text: str) -> list[dict[str, object]]:
            specs: list[dict[str, object]] = []
            for line in raw_text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                parts = [part.strip() for part in stripped.split("|")]
                canonical = parts[0]
                aliases = [v.strip() for v in parts[1].split(",")] if len(parts) > 1 and parts[1] else []
                generic_terms = [v.strip() for v in parts[2].split(",")] if len(parts) > 2 and parts[2] else []
                description = parts[3] if len(parts) > 3 else ""
                object_parts = [v.strip() for v in parts[4].split(",")] if len(parts) > 4 and parts[4] else []
                detector_phrases = list(dict.fromkeys([*generic_terms, *aliases]))
                specs.append(
                    {
                        "canonical": canonical,
                        "aliases": aliases,
                        "generic_terms": generic_terms,
                        "description": description,
                        "parts": object_parts,
                        "detector_phrases": detector_phrases,
                    }
                )
            return specs

        def _save() -> None:
            domain_name = domain_var.get().strip()
            if not domain_name:
                status_label.config(text="Enter a domain name.", foreground="red")
                return
            object_specs = _parse_objects(objects_text.get("1.0", tk.END))
            if not object_specs:
                status_label.config(text="Add at least one object.", foreground="red")
                return
            guide_path = batch_folder / "skiagrafia_guide.toml"
            pack = build_knowledge_pack(
                path=guide_path,
                domain_name=domain_name,
                domain_description=description_var.get().strip(),
                object_specs=object_specs,
                preferred_vlm=self._preferred_vlm_var.get(),
                fallback_vlms=[self._preferred_vlm_var.get(), "llava:7b"],
                enable_tiling=self._tiled_fallback_var.get(),
                max_aliases_per_object=4,
                notes_markdown=notes_text.get("1.0", tk.END).strip() + "\n",
            )
            pack.save(guide_path)
            self._view.knowledge_pack_path = str(guide_path)
            self._view.knowledge_pack_name = pack.name
            self._view.knowledge_pack_notes_path = str(guide_path.with_suffix(".md"))
            self._view.knowledge_pack_defaults = pack.batch_defaults.model_dump()
            self._view.knowledge_guidance_active = True
            self._refresh_guide_status()
            status_label.config(text=f"Saved guide to {guide_path.name}", foreground="green")
            dialog.after(700, dialog.destroy)

        buttons = ttk.Frame(dialog)
        buttons.pack(fill=tk.X, padx=12, pady=(0, 12))
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Save guide", command=_save).pack(side=tk.RIGHT, padx=(0, 6))

    def get_config(self) -> dict:
        # Build output_mode string from checked options
        active = [k for k, v in self._mode_vars.items() if v.get()]
        output_mode = "+".join(active) if active else "vector"
        config = {
            "output_mode": output_mode,
            "recursion_depth": self._depth_var.get(),
            "vtracer_quality": self._quality_var.get(),
            "guide_path": self._view.knowledge_pack_path if self._guide_mode_var.get() else None,
            "guide_mode": self._guide_mode_var.get(),
            "fallback_mode": self._fallback_mode_var.get(),
            "interrogation_profile": self._profile_var.get(),
            "preferred_vlm": self._preferred_vlm_var.get(),
            "text_reasoner_model": self._reasoner_var.get(),
            "enable_tiled_fallback": self._tiled_fallback_var.get(),
        }
        self._view.interrogation_settings = dict(config)
        return config
