from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import TYPE_CHECKING

from ui.theme import is_macos

if TYPE_CHECKING:
    from core.knowledge import KnowledgePack

logger = logging.getLogger(__name__)


class GuideEditorTab:
    """Domain Guide editor — two-pane form + live TOML preview."""

    def __init__(self, parent: tk.Widget) -> None:
        self._current_path: Path | None = None
        self._objects: list[dict] = []
        self._fallback_vlms: list[str] = []

        # ── tk vars (must exist before any widget that traces them) ──
        self._dom_name_var = tk.StringVar()
        self._dom_desc_var = tk.StringVar()
        self._def_vlm_var = tk.StringVar()
        self._def_tiling_var = tk.StringVar()
        self._def_aliases_var = tk.StringVar()

        for var in (
            self._dom_name_var, self._dom_desc_var,
            self._def_vlm_var, self._def_tiling_var, self._def_aliases_var,
        ):
            var.trace_add("write", self._render_toml)

        # ── Layout ──
        outer = ttk.Frame(parent)
        outer.pack(fill=tk.BOTH, expand=True)

        self._build_toolbar(outer)

        body = ttk.Frame(outer)
        body.pack(fill=tk.BOTH, expand=True)

        self._build_form_pane(body)
        ttk.Separator(body, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y)
        self._build_preview_pane(body)

        self._render_toml()

    # ── Toolbar ───────────────────────────────────────────────────

    def _build_toolbar(self, parent: tk.Widget) -> None:
        bar = ttk.Frame(parent)
        bar.pack(fill=tk.X, padx=8, pady=(6, 2))

        ttk.Button(bar, text="New", command=self._new_guide).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(bar, text="Open\u2026", command=self._open_guide).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(bar, text="Save", command=self._save_guide).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(bar, text="Save As\u2026", command=self._save_guide_as).pack(
            side=tk.LEFT, padx=(0, 4)
        )

        self._path_label = ttk.Label(bar, text="Unsaved", foreground="gray")
        self._path_label.pack(side=tk.LEFT, padx=(12, 0))

    # ── Form pane (left, scrollable) ──────────────────────────────

    def _build_form_pane(self, parent: tk.Widget) -> None:
        container = ttk.Frame(parent, width=400)
        container.pack(side=tk.LEFT, fill=tk.BOTH)
        container.pack_propagate(False)

        canvas = tk.Canvas(container, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        self._form_frame = ttk.Frame(canvas, padding=(12, 8))

        self._form_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self._form_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Mousewheel scrolling — bind to canvas only, not globally
        def _on_mousewheel(event: tk.Event) -> None:
            canvas.yview_scroll(-1 * (event.delta // 120 or event.delta), "units")

        canvas.bind("<MouseWheel>", _on_mousewheel)

        self._build_domain_section()
        self._build_defaults_section()
        self._build_objects_section()

    # ── Preview pane (right, read-only TOML) ──────────────────────

    def _build_preview_pane(self, parent: tk.Widget) -> None:
        mono_font = ("Menlo", 10) if is_macos() else ("Consolas", 10)
        self._toml_text = tk.Text(
            parent,
            font=mono_font,
            state=tk.DISABLED,
            wrap=tk.NONE,
            borderwidth=0,
            padx=8,
            pady=8,
        )
        self._toml_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # ── Domain section ────────────────────────────────────────────

    def _build_domain_section(self) -> None:
        f = self._form_frame

        ttk.Label(f, text="Domain", font=("TkDefaultFont", 13, "bold")).pack(
            anchor=tk.W, pady=(0, 6)
        )

        row_name = ttk.Frame(f)
        row_name.pack(fill=tk.X, pady=2)
        ttk.Label(row_name, text="Name", width=14).pack(side=tk.LEFT)
        ttk.Entry(row_name, textvariable=self._dom_name_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

        row_desc = ttk.Frame(f)
        row_desc.pack(fill=tk.X, pady=2)
        ttk.Label(row_desc, text="Description", width=14).pack(side=tk.LEFT)
        ttk.Entry(row_desc, textvariable=self._dom_desc_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

    # ── Batch defaults section ────────────────────────────────────

    def _build_defaults_section(self) -> None:
        f = self._form_frame

        ttk.Separator(f, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(12, 8))
        ttk.Label(f, text="Batch defaults", font=("TkDefaultFont", 13, "bold")).pack(
            anchor=tk.W, pady=(0, 6)
        )

        # Preferred VLM
        row_vlm = ttk.Frame(f)
        row_vlm.pack(fill=tk.X, pady=2)
        ttk.Label(row_vlm, text="Preferred VLM", width=14).pack(side=tk.LEFT)
        ttk.Combobox(
            row_vlm,
            textvariable=self._def_vlm_var,
            values=["", "moondream", "moondream2", "minicpm-v", "llava:7b"],
            width=18,
        ).pack(side=tk.LEFT)

        # Enable tiling
        row_tiling = ttk.Frame(f)
        row_tiling.pack(fill=tk.X, pady=2)
        ttk.Label(row_tiling, text="Enable tiling", width=14).pack(side=tk.LEFT)

        tiling_frame = ttk.Frame(row_tiling)
        tiling_frame.pack(side=tk.LEFT)
        for text, val in [("Inherit", ""), ("Yes", "true"), ("No", "false")]:
            ttk.Radiobutton(
                tiling_frame,
                text=text,
                variable=self._def_tiling_var,
                value=val,
            ).pack(side=tk.LEFT, padx=(0, 6))

        # Max aliases
        row_aliases = ttk.Frame(f)
        row_aliases.pack(fill=tk.X, pady=2)
        ttk.Label(row_aliases, text="Max aliases/obj", width=14).pack(side=tk.LEFT)
        aliases_spin = ttk.Spinbox(
            row_aliases,
            textvariable=self._def_aliases_var,
            from_=1,
            to=12,
            width=5,
        )
        aliases_spin.pack(side=tk.LEFT)

        # Fallback VLMs tag list
        ttk.Label(f, text="Fallback VLMs", width=14).pack(anchor=tk.W, pady=(6, 2))
        self._fallback_frame = ttk.Frame(f)
        self._fallback_frame.pack(fill=tk.X, pady=(0, 4))
        self._rebuild_fallback_tags()

    def _rebuild_fallback_tags(self) -> None:
        for w in self._fallback_frame.winfo_children():
            w.destroy()

        tags_row = ttk.Frame(self._fallback_frame)
        tags_row.pack(fill=tk.X)

        for i, tag in enumerate(self._fallback_vlms):
            pill = ttk.Frame(tags_row)
            pill.pack(side=tk.LEFT, padx=(0, 4), pady=2)
            ttk.Label(pill, text=tag).pack(side=tk.LEFT)
            remove_btn = ttk.Button(
                pill,
                text="\u00d7",
                width=2,
                command=lambda idx=i: self._remove_fallback_vlm(idx),
            )
            remove_btn.pack(side=tk.LEFT)

        add_row = ttk.Frame(self._fallback_frame)
        add_row.pack(fill=tk.X, pady=(2, 0))
        add_var = tk.StringVar()
        entry = ttk.Entry(add_row, textvariable=add_var, width=16)
        entry.pack(side=tk.LEFT)

        def _add_fallback() -> None:
            val = add_var.get().strip()
            if val and val not in self._fallback_vlms:
                self._fallback_vlms.append(val)
                add_var.set("")
                self._rebuild_fallback_tags()
                self._render_toml()

        ttk.Button(add_row, text="+", width=2, command=_add_fallback).pack(
            side=tk.LEFT, padx=(4, 0)
        )
        entry.bind("<Return>", lambda e: _add_fallback())

    def _remove_fallback_vlm(self, idx: int) -> None:
        if 0 <= idx < len(self._fallback_vlms):
            self._fallback_vlms.pop(idx)
            self._rebuild_fallback_tags()
            self._render_toml()

    # ── Objects section ───────────────────────────────────────────

    def _build_objects_section(self) -> None:
        f = self._form_frame

        ttk.Separator(f, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(12, 8))
        ttk.Label(f, text="Objects", font=("TkDefaultFont", 13, "bold")).pack(
            anchor=tk.W, pady=(0, 6)
        )

        self._objects_container = ttk.Frame(f)
        self._objects_container.pack(fill=tk.X)

        ttk.Button(f, text="+ Add object", command=self._add_object).pack(
            anchor=tk.W, pady=(8, 4)
        )

    def _add_object(self) -> None:
        self._objects.append({
            "canonical": "",
            "aliases": [],
            "generic_terms": [],
            "description": "",
            "parts": [],
            "detector_phrases": [],
            "open": True,
        })
        self._rebuild_objects_ui()
        self._render_toml()

    def _remove_object(self, idx: int) -> None:
        if 0 <= idx < len(self._objects):
            self._objects.pop(idx)
            self._rebuild_objects_ui()
            self._render_toml()

    def _toggle_object(self, idx: int) -> None:
        if 0 <= idx < len(self._objects):
            self._objects[idx]["open"] = not self._objects[idx]["open"]
            self._rebuild_objects_ui()

    def _rebuild_objects_ui(self) -> None:
        for w in self._objects_container.winfo_children():
            w.destroy()

        for idx, obj in enumerate(self._objects):
            self._build_object_widget(self._objects_container, idx, obj)

    def _build_object_widget(
        self, parent: tk.Widget, idx: int, obj: dict
    ) -> None:
        frame = ttk.Frame(parent, relief="groove", borderwidth=1)
        frame.pack(fill=tk.X, pady=(0, 6))

        # Header row
        header = ttk.Frame(frame)
        header.pack(fill=tk.X, padx=6, pady=4)

        chevron = "\u25bc" if obj["open"] else "\u25b6"
        label_text = obj["canonical"] or "(unnamed)"
        header_btn = ttk.Label(
            header,
            text=f"{chevron}  {label_text}",
            cursor="hand2",
        )
        header_btn.pack(side=tk.LEFT)
        header_btn.bind("<Button-1>", lambda e, i=idx: self._toggle_object(i))

        ttk.Button(
            header, text="Remove", command=lambda i=idx: self._remove_object(i)
        ).pack(side=tk.RIGHT)

        if not obj["open"]:
            return

        # Expanded body
        body = ttk.Frame(frame, padding=(12, 4, 12, 8))
        body.pack(fill=tk.X)

        # Canonical
        row_can = ttk.Frame(body)
        row_can.pack(fill=tk.X, pady=2)
        ttk.Label(row_can, text="Canonical", width=14).pack(side=tk.LEFT)
        can_var = tk.StringVar(value=obj["canonical"])

        def _on_canonical_change(*_, i: int = idx, v: tk.StringVar = can_var) -> None:
            self._objects[i]["canonical"] = v.get()
            chev = "\u25bc" if self._objects[i]["open"] else "\u25b6"
            header_btn.config(text=f"{chev}  {v.get() or '(unnamed)'}")
            self._render_toml()

        can_var.trace_add("write", _on_canonical_change)
        ttk.Entry(row_can, textvariable=can_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

        # Description
        row_desc = ttk.Frame(body)
        row_desc.pack(fill=tk.X, pady=2)
        ttk.Label(row_desc, text="Description", width=14).pack(side=tk.LEFT)
        desc_var = tk.StringVar(value=obj["description"])

        def _on_desc_change(*_, i: int = idx, v: tk.StringVar = desc_var) -> None:
            self._objects[i]["description"] = v.get()
            self._render_toml()

        desc_var.trace_add("write", _on_desc_change)
        ttk.Entry(row_desc, textvariable=desc_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

        # Tag-list fields
        self._make_tag_field(body, obj["aliases"], "Aliases", idx)
        self._make_tag_field(body, obj["generic_terms"], "Generic terms", idx)
        self._make_tag_field(body, obj["parts"], "Parts", idx)
        self._make_tag_field(body, obj["detector_phrases"], "Detector phrases", idx)

    # ── Tag-list helper ───────────────────────────────────────────

    def _make_tag_field(
        self,
        parent: tk.Widget,
        tags_list: list[str],
        label_text: str,
        obj_idx: int,
    ) -> ttk.Frame:
        wrapper = ttk.Frame(parent)
        wrapper.pack(fill=tk.X, pady=2)

        ttk.Label(wrapper, text=label_text, width=14).pack(side=tk.LEFT, anchor=tk.N)

        right = ttk.Frame(wrapper)
        right.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tags_row = ttk.Frame(right)
        tags_row.pack(fill=tk.X)

        for i, tag in enumerate(tags_list):
            pill = ttk.Frame(tags_row)
            pill.pack(side=tk.LEFT, padx=(0, 4), pady=1)
            ttk.Label(pill, text=tag).pack(side=tk.LEFT)

            def _remove(idx: int = i, lst: list[str] = tags_list) -> None:
                lst.pop(idx)
                self._rebuild_objects_ui()
                self._render_toml()

            ttk.Button(pill, text="\u00d7", width=2, command=_remove).pack(
                side=tk.LEFT
            )

        add_row = ttk.Frame(right)
        add_row.pack(fill=tk.X, pady=(1, 0))
        add_var = tk.StringVar()
        entry = ttk.Entry(add_row, textvariable=add_var, width=14)
        entry.pack(side=tk.LEFT)

        def _add_tag() -> None:
            val = add_var.get().strip()
            if val and val not in tags_list:
                tags_list.append(val)
                add_var.set("")
                self._rebuild_objects_ui()
                self._render_toml()

        ttk.Button(add_row, text="+", width=2, command=_add_tag).pack(
            side=tk.LEFT, padx=(4, 0)
        )
        entry.bind("<Return>", lambda e: _add_tag())

        return wrapper

    # ── TOML preview ──────────────────────────────────────────────

    def _render_toml(self, *_: object) -> None:
        pack = self._pack_to_model()
        toml_str = pack.to_toml()
        self._toml_text.config(state=tk.NORMAL)
        self._toml_text.delete("1.0", tk.END)
        self._toml_text.insert("1.0", toml_str)
        self._toml_text.config(state=tk.DISABLED)

    # ── Model conversion ──────────────────────────────────────────

    def _pack_to_model(self) -> KnowledgePack:
        from core.knowledge import (
            BatchGuideDefaults,
            KnowledgeDomain,
            KnowledgePack,
            ObjectKnowledge,
        )

        vlm = self._def_vlm_var.get().strip() or None
        tiling_raw = self._def_tiling_var.get()
        tiling = {"true": True, "false": False}.get(tiling_raw)
        aliases_raw = self._def_aliases_var.get().strip()
        try:
            max_aliases = int(aliases_raw) if aliases_raw else None
        except ValueError:
            max_aliases = None

        objects = [
            ObjectKnowledge(
                canonical=obj["canonical"],
                aliases=list(obj["aliases"]),
                generic_terms=list(obj["generic_terms"]),
                description=obj["description"],
                parts=list(obj["parts"]),
                detector_phrases=list(obj["detector_phrases"]),
            )
            for obj in self._objects
            if obj["canonical"].strip()
        ]

        return KnowledgePack(
            path=str(self._current_path) if self._current_path else "",
            domain=KnowledgeDomain(
                name=self._dom_name_var.get().strip(),
                description=self._dom_desc_var.get().strip(),
            ),
            objects=objects,
            batch_defaults=BatchGuideDefaults(
                preferred_vlm=vlm,
                fallback_vlms=list(self._fallback_vlms),
                enable_tiling=tiling,
                max_aliases_per_object=max_aliases,
            ),
        )

    def _load_from_model(self, pack: KnowledgePack) -> None:
        self._dom_name_var.set(pack.domain.name)
        self._dom_desc_var.set(pack.domain.description)

        self._def_vlm_var.set(pack.batch_defaults.preferred_vlm or "")
        self._fallback_vlms = list(pack.batch_defaults.fallback_vlms)
        tiling = pack.batch_defaults.enable_tiling
        self._def_tiling_var.set("" if tiling is None else ("true" if tiling else "false"))
        aliases = pack.batch_defaults.max_aliases_per_object
        self._def_aliases_var.set("" if aliases is None else str(aliases))

        self._objects = [
            {
                "canonical": obj.canonical,
                "aliases": list(obj.aliases),
                "generic_terms": list(obj.generic_terms),
                "description": obj.description,
                "parts": list(obj.parts),
                "detector_phrases": list(obj.detector_phrases),
                "open": False,
            }
            for obj in pack.objects
        ]

        self._rebuild_fallback_tags()
        self._rebuild_objects_ui()
        self._render_toml()

    # ── File operations ───────────────────────────────────────────

    def _new_guide(self) -> None:
        self._current_path = None
        self._dom_name_var.set("")
        self._dom_desc_var.set("")
        self._def_vlm_var.set("")
        self._fallback_vlms = []
        self._def_tiling_var.set("")
        self._def_aliases_var.set("")
        self._objects = []
        self._rebuild_fallback_tags()
        self._rebuild_objects_ui()
        self._render_toml()
        self._update_path_label()

    def _open_guide(self) -> None:
        from tkinter import filedialog

        from core.knowledge import KnowledgePack

        path = filedialog.askopenfilename(
            title="Open domain guide",
            filetypes=[("TOML guide", "*.toml"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            pack = KnowledgePack.load(path)
            self._current_path = Path(path)
            self._load_from_model(pack)
            self._update_path_label()
        except Exception as exc:
            from tkinter import messagebox

            messagebox.showerror("Load error", str(exc))

    def _save_guide(self) -> None:
        if self._current_path is None:
            self._save_guide_as()
            return
        pack = self._pack_to_model()
        try:
            pack.save(self._current_path)
            self._update_path_label()
        except Exception as exc:
            from tkinter import messagebox

            messagebox.showerror("Save error", str(exc))

    def _save_guide_as(self) -> None:
        from tkinter import filedialog

        path = filedialog.asksaveasfilename(
            title="Save domain guide",
            defaultextension=".toml",
            filetypes=[("TOML guide", "*.toml")],
            initialfile="skiagrafia_guide.toml",
        )
        if not path:
            return
        self._current_path = Path(path)
        self._save_guide()

    def _update_path_label(self) -> None:
        if self._current_path:
            self._path_label.config(text=str(self._current_path), foreground="")
        else:
            self._path_label.config(text="Unsaved", foreground="gray")
