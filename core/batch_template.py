from __future__ import annotations

import datetime
import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel, ValidationError

from utils.security import atomic_write_bytes, safe_child_path

logger = logging.getLogger(__name__)

# macOS caps a filename at 255 bytes; leave room for the .json suffix.
MAX_TEMPLATE_SLUG_LEN = 120
_UNSAFE_SLUG_CHARS = re.compile(r"[^a-z0-9._-]+")


def _slugify(name: str) -> str:
    """Reduce a template name to a safe, flat filename stem.

    Path separators and traversal segments must not survive: the stem is
    joined onto the templates directory and must stay inside it.
    """
    slug = _UNSAFE_SLUG_CHARS.sub("_", name.strip().lower())
    slug = slug.strip("._")[:MAX_TEMPLATE_SLUG_LEN].strip("._")
    return slug or "untitled"


class BatchTemplate(BaseModel):
    """Serialisable template for promoting a Single session to Batch."""

    name: str
    created_at: str = ""
    source_image: str
    confirmed_labels: list[str]
    confirmed_children: dict[str, list[str]]
    output_mode: str  # "vector+bitmap" | "vector" | "bitmap"
    recursion_depth: int  # 1 | 2 | 3
    corner_threshold: int  # 30–90
    speckle: int  # 2–20
    smoothing: int  # 1–10
    length_threshold: float  # 2.0–8.0
    vtracer_quality: str  # "draft" | "balanced" | "maximum"
    guide_path: str | None = None
    interrogation_profile: str = "balanced"
    fallback_mode: str = "adaptive_auto"
    preferred_vlm: str | None = None
    text_reasoner_model: str | None = None
    enable_tiled_fallback: bool = True
    selection_request: str = ""
    guide_name: str | None = None

    def stamped(self) -> BatchTemplate:
        """A copy carrying the current local time; the receiver is unchanged.

        The offset is recorded alongside the wall clock, so a template saved
        in one timezone still says when it was written.
        """
        return self.model_copy(
            update={"created_at": datetime.datetime.now().astimezone().isoformat()}
        )

    def save(self) -> Path:
        """Write a timestamped copy of this template and return its path.

        The copy is what gets written; THIS OBJECT IS NOT MODIFIED. It used
        to stamp `created_at` onto the caller's own instance, so writing a
        template to disk silently edited an object the caller still held --
        and saving twice changed it twice. Use `stamped()` when the written
        form is what you want in hand.
        """
        d = Path.home() / ".config" / "skiagrafia" / "templates"
        d.mkdir(parents=True, exist_ok=True)
        record = self.stamped()
        path = safe_child_path(d, f"{_slugify(record.name)}.json")
        atomic_write_bytes(path, record.model_dump_json(indent=2).encode("utf-8"))
        return path

    @classmethod
    def load(cls, path: Path) -> BatchTemplate:
        return cls.model_validate_json(path.read_text())

    @classmethod
    def list_all(cls) -> list[BatchTemplate]:
        return [template for _path, template in cls.list_all_with_paths()]

    @classmethod
    def list_all_with_paths(cls) -> list[tuple[Path, BatchTemplate]]:
        """Return usable templates with their source paths for UI actions."""
        d = Path.home() / ".config" / "skiagrafia" / "templates"
        if not d.exists():
            return []
        templates: list[tuple[Path, BatchTemplate]] = []
        for p in sorted(
            d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            try:
                templates.append((p, cls.load(p)))
            except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError, ValidationError):
                # One unreadable template must not hide every other one.
                logger.warning("Skipping unreadable template %s", p, exc_info=True)
        return templates
