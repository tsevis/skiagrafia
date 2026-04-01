from __future__ import annotations

import datetime
from pathlib import Path

from pydantic import BaseModel


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
        path = d / f"{self.name.lower().replace(' ', '_')}.json"
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
        return [
            cls.load(p)
            for p in sorted(
                d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
            )
        ]
