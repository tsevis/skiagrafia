"""interrogation_types.py  --  what an interrogation says, as data.

The candidate labels a vision pass proposes, the result it returns, the
settings that shape it, and the vague terms that never count as an answer.
No behaviour, no model calls.

Separated so the stages that produce these -- core/interrogation.py and
core/reasoner_stage.py -- can both name them without importing each other.
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from models.vlm_client import BACKEND_OLLAMA

_VAGUE_TERMS = {
    "object",
    "item",
    "artifact",
    "decoration",
    "ornament",
    "thing",
    "metal object",
}


class InterrogationCandidate(BaseModel):
    canonical_label: str
    display_label: str
    detector_phrases: list[str] = Field(default_factory=list)
    source_model: str = "vlm"
    confidence: float = 0.5
    role: str = "parent"
    parent: str | None = None
    selection: str = "all"


class InterrogationResult(BaseModel):
    candidates: list[InterrogationCandidate] = Field(default_factory=list)
    children_by_parent: dict[str, list[str]] = Field(default_factory=dict)
    raw_responses: dict[str, str] = Field(default_factory=dict)
    escalation_stage: str = "primary"
    confidence_summary: str = ""


@dataclass
class InterrogationSettings:
    host: str
    primary_vlm: str
    fallback_vlms: list[str]
    reasoner_model: str
    backend: str = BACKEND_OLLAMA  # "ollama" | "llamacpp"
    profile: str = "balanced"
    fallback_mode: str = "adaptive_auto"
    composition_first: bool = True
    enable_tiling: bool = True
    max_aliases_per_object: int = 4
    selection_request: str = ""
    # Legacy alias retained for callers from the first Single-mode release.
    user_prompt: str = ""
    discover_parts: bool = True
    selections: dict[str, str] | None = None
