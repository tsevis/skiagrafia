from __future__ import annotations

import datetime
import logging
import re
from pathlib import Path

from pydantic import BaseModel

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

    def save(self) -> Path:
        d = Path.home() / ".config" / "skiagrafia" / "templates"
        d.mkdir(parents=True, exist_ok=True)
        self.created_at = datetime.datetime.now().isoformat()
        path = d / f"{_slugify(self.name)}.json"
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> BatchTemplate:
        return cls.model_validate_json(path.read_text())

    @classmethod
    def list_all(cls) -> list[BatchTemplate]:
        d = Path.home() / ".config" / "skiagrafia" / "templates"
        if not d.exists():
            return []
        templates: list[BatchTemplate] = []
        for p in sorted(
            d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            try:
                templates.append(cls.load(p))
            except Exception:
                # One unreadable template must not hide every other one.
                logger.warning("Skipping unreadable template %s", p, exc_info=True)
        return templates
