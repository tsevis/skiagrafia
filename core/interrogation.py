from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, Field

from core.knowledge import KnowledgePack, ObjectKnowledge
from models.moondream_client import MAX_PARENTS, MoondreamClient

logger = logging.getLogger(__name__)

_VAGUE_TERMS = {
    "object",
    "item",
    "artifact",
    "decoration",
    "ornament",
    "thing",
    "metal object",
}
_PART_BLACKLIST = {
    "coffee",
    "tea",
    "smoothie",
    "hot chocolate",
    "iced tea",
    "drink",
    "beverage",
    "food",
    "background",
    "scene",
}
_SPLIT_RE = re.compile(r"[\n,;•\-]+")
_GARBAGE_RE = re.compile(
    r"/"               # contains slashes
    r"|^[\d\.\-,\s]+$" # pure numbers / decimals / punctuation
    r"|^\W+$"          # pure non-word characters
)
_LEADIN_RE = re.compile(
    r"^(the image (shows|features)|there is|there are|visible objects include|objects?:)\s+",
    re.IGNORECASE,
)


class InterrogationCandidate(BaseModel):
    canonical_label: str
    display_label: str
    detector_phrases: list[str] = Field(default_factory=list)
    source_model: str = "moondream"
    confidence: float = 0.5
    role: str = "parent"
    parent: str | None = None


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
    profile: str = "balanced"
    fallback_mode: str = "adaptive_auto"
    composition_first: bool = True
    enable_tiling: bool = True
    max_aliases_per_object: int = 4


def parse_label_candidates(raw: str, limit: int = MAX_PARENTS) -> list[str]:
    text = raw.strip()
    if not text:
        return []
    text = _LEADIN_RE.sub("", text)

    chunks = [c.strip(" .:") for c in _SPLIT_RE.split(text) if c.strip()]
    candidates: list[str] = []
    for chunk in chunks:
        lowered = chunk.lower()
        if lowered.startswith(("a ", "an ", "the ")):
            chunk = chunk.split(" ", 1)[1]
        if len(chunk) <= 1:
            continue
        if _GARBAGE_RE.search(chunk):
            continue
        if any(tok.isdigit() for tok in chunk.split()):
            continue
        if chunk.lower() not in {c.lower() for c in candidates}:
            candidates.append(chunk)
        if len(candidates) >= limit:
            break
    return candidates


def rank_detector_phrases(
    canonical: str,
    aliases: list[str],
    generic_terms: list[str],
    description: str = "",
    limit: int = 4,
) -> list[str]:
    phrases: list[str] = []
    for value in [*generic_terms, *aliases, canonical, description]:
        cleaned = value.strip()
        if cleaned and cleaned not in phrases:
            phrases.append(cleaned)
    return phrases[:limit]


class GuidedInterrogator:
    def __init__(self, settings: InterrogationSettings) -> None:
        self._settings = settings
        self._clients: dict[str, MoondreamClient] = {}

    def interrogate(
        self,
        image: NDArray[np.uint8],
        confirmed_labels: list[str] | None = None,
        knowledge_pack: KnowledgePack | None = None,
    ) -> InterrogationResult:
        if confirmed_labels:
            candidates = self._candidates_from_confirmed_labels(
                confirmed_labels, knowledge_pack
            )
            return InterrogationResult(
                candidates=candidates,
                children_by_parent=self._children_map(candidates, knowledge_pack, image),
                escalation_stage="confirmed",
                confidence_summary="user-confirmed labels",
            )

        stage = "primary"
        raw_responses: dict[str, str] = {}
        candidates: list[InterrogationCandidate] = []

        if self._settings.composition_first:
            stage = "composition"
            composition = self._run_vision_stage(
                image=image,
                model=self._settings.primary_vlm,
                knowledge_pack=knowledge_pack,
                raw_responses=raw_responses,
                prompt_style="composition",
            )
            candidates = self._merge_candidates(candidates, composition)

        if self._should_escalate(candidates) or self._should_force_primary_pass():
            stage = "primary"
            primary = self._run_vision_stage(
                image=image,
                model=self._settings.primary_vlm,
                knowledge_pack=knowledge_pack,
                raw_responses=raw_responses,
                prompt_style="primary",
            )
            candidates = self._merge_candidates(candidates, primary)

        if self._should_escalate(candidates):
            stage = "guided"
            candidates = self._merge_candidates(
                candidates,
                self._knowledge_seed_candidates(knowledge_pack),
            )
            guided = self._run_vision_stage(
                image=image,
                model=self._settings.primary_vlm,
                knowledge_pack=knowledge_pack,
                raw_responses=raw_responses,
                prompt_style="guided",
            )
            candidates = self._merge_candidates(candidates, guided)

        if self._settings.fallback_mode != "moondream_only" and self._should_escalate(candidates):
            for model in self._settings.fallback_vlms:
                stage = f"fallback:{model}"
                fallback_candidates = self._run_vision_stage(
                    image=image,
                    model=model,
                    knowledge_pack=knowledge_pack,
                    raw_responses=raw_responses,
                    prompt_style="guided",
                )
                candidates = self._merge_candidates(candidates, fallback_candidates)
                if not self._should_escalate(candidates):
                    break

        if self._settings.enable_tiling and self._allows_tiling() and self._should_escalate(candidates):
            stage = "tiled"
            for tile in self._iter_tiles(image):
                tiled = self._run_vision_stage(
                    image=tile,
                    model=self._settings.primary_vlm,
                    knowledge_pack=knowledge_pack,
                    raw_responses=raw_responses,
                    prompt_style="guided",
                )
                candidates = self._merge_candidates(candidates, tiled)

        candidates = self._reason_and_rank(candidates, knowledge_pack, raw_responses, stage)
        confidence_summary = (
            "high-confidence results"
            if not self._should_escalate(candidates)
            else "low-confidence results"
        )
        return InterrogationResult(
            candidates=candidates,
            children_by_parent=self._children_map(candidates, knowledge_pack, image),
            raw_responses=raw_responses,
            escalation_stage=stage,
            confidence_summary=confidence_summary,
        )

    def _run_vision_stage(
        self,
        image: NDArray[np.uint8],
        model: str,
        knowledge_pack: KnowledgePack | None,
        raw_responses: dict[str, str],
        prompt_style: str,
    ) -> list[InterrogationCandidate]:
        prompt = self._build_prompt(knowledge_pack, prompt_style)
        prepared_image = self._prepare_image(image)
        try:
            client = self._get_client(model)
            response = client.query_vision(prepared_image, prompt)
        except Exception:
            logger.warning("Vision interrogation failed for model %s", model, exc_info=True)
            return []

        raw_responses[f"{prompt_style}:{model}"] = response
        labels = parse_label_candidates(response)
        return [
            self._candidate_from_label(
                label=label,
                knowledge=knowledge_pack.find_object(label) if knowledge_pack else None,
                source_model=model,
                confidence=0.8 if prompt_style == "primary" else 0.72,
            )
            for label in labels
        ]

    def _get_client(self, model: str) -> MoondreamClient:
        client = self._clients.get(model)
        if client is None:
            client = MoondreamClient(host=self._settings.host, model=model)
            self._clients[model] = client
        return client

    def _build_prompt(
        self,
        knowledge_pack: KnowledgePack | None,
        prompt_style: str,
    ) -> str:
        domain_prefix = ""
        if knowledge_pack and knowledge_pack.domain.name:
            domain_prefix = (
                f"The image belongs to the domain '{knowledge_pack.domain.name}'. "
            )
        domain_desc = ""
        if knowledge_pack and knowledge_pack.domain.description:
            domain_desc = f"{knowledge_pack.domain.description} "
        exemplars = ""
        if knowledge_pack and knowledge_pack.objects:
            sample_terms: list[str] = []
            for obj in knowledge_pack.objects[:6]:
                sample_terms.append(obj.canonical)
                sample_terms.extend(obj.aliases[:1])
            exemplar_text = ", ".join(dict.fromkeys(sample_terms))
            exemplars = f"Possible object families include: {exemplar_text}. "

        if prompt_style == "composition":
            return (
                f"{domain_prefix}{domain_desc}{exemplars}"
                "Describe the whole foreground composition first, then name the main objects that define it. "
                "Prefer the dominant object or grouped objects over tiny details. "
                "If the exact specialist term is unknown, use short visual nouns based on shape, material, or purpose. "
                "Reply only as a comma-separated list of the main whole objects."
            )
        if prompt_style == "guided":
            return (
                f"{domain_prefix}{domain_desc}{exemplars}"
                "Identify the main foreground objects. "
                "If the exact specialist term is unknown, reply with concrete visual nouns. "
                "Prefer simple detector-friendly phrases that describe the visible object plainly. "
                "Reply only as a comma-separated object list."
            )
        return (
            f"{domain_prefix}{domain_desc}"
            "List the main foreground objects in this image. "
            "Reply only as a comma-separated list of object names. "
            "If the exact name is unknown, use simple visual nouns."
        )

    def _knowledge_seed_candidates(
        self, knowledge_pack: KnowledgePack | None
    ) -> list[InterrogationCandidate]:
        if knowledge_pack is None or self._settings.fallback_mode != "always_enrich":
            return []
        return [
            self._candidate_from_label(
                label=obj.canonical,
                knowledge=obj,
                source_model="knowledge-pack",
                confidence=0.45,
            )
            for obj in knowledge_pack.objects[: min(6, MAX_PARENTS)]
        ]

    def _candidates_from_confirmed_labels(
        self,
        confirmed_labels: list[str],
        knowledge_pack: KnowledgePack | None,
    ) -> list[InterrogationCandidate]:
        candidates: list[InterrogationCandidate] = []
        for label in confirmed_labels:
            knowledge = knowledge_pack.find_object(label) if knowledge_pack else None
            candidates.append(
                self._candidate_from_label(
                    label=label,
                    knowledge=knowledge,
                    source_model="confirmed",
                    confidence=1.0,
                )
            )
        return candidates

    def _candidate_from_label(
        self,
        label: str,
        knowledge: ObjectKnowledge | None,
        source_model: str,
        confidence: float,
    ) -> InterrogationCandidate:
        canonical = knowledge.canonical if knowledge else label.strip().lower()
        display = knowledge.canonical if knowledge else label.strip().lower()
        detector_phrases = (
            knowledge.ranked_detector_phrases(self._settings.max_aliases_per_object)
            if knowledge
            else rank_detector_phrases(
                canonical=canonical,
                aliases=[],
                generic_terms=self._generic_terms_from_label(label),
                limit=self._settings.max_aliases_per_object,
            )
        )
        return InterrogationCandidate(
            canonical_label=canonical,
            display_label=display,
            detector_phrases=detector_phrases,
            source_model=source_model,
            confidence=confidence,
        )

    def _generic_terms_from_label(self, label: str) -> list[str]:
        label = label.strip().lower()
        terms = [label]
        normalized = label.replace("_", " ").replace("-", " ").strip()
        if normalized and normalized != label:
            terms.append(normalized)
        broad_aliases = {
            "iphone": ["phone", "smartphone", "mobile phone"],
            "phone": ["smartphone", "mobile phone"],
            "monitor": ["computer monitor", "screen", "display"],
            "keyboard": ["computer keyboard"],
            "mouse": ["computer mouse"],
            "laptop": ["computer", "notebook computer"],
            "tablet": ["tablet computer", "screen device"],
        }
        for needle, aliases in broad_aliases.items():
            if needle in normalized:
                terms.extend(aliases)
                break
        if not any("object" in term for term in terms):
            terms.append(f"{normalized or label} object")
        return list(dict.fromkeys(terms))

    def _should_escalate(self, candidates: list[InterrogationCandidate]) -> bool:
        if not candidates:
            return True
        high_conf = [c for c in candidates if c.confidence >= 0.65]
        if not high_conf:
            return True
        if all(c.display_label.lower() in _VAGUE_TERMS for c in candidates):
            return True
        if len(candidates) < 2:
            return True
        return False

    def _merge_candidates(
        self,
        left: list[InterrogationCandidate],
        right: list[InterrogationCandidate],
    ) -> list[InterrogationCandidate]:
        merged: dict[str, InterrogationCandidate] = {c.canonical_label: c for c in left}
        for candidate in right:
            existing = merged.get(candidate.canonical_label)
            if existing is None or candidate.confidence > existing.confidence:
                merged[candidate.canonical_label] = candidate
            elif existing:
                existing.detector_phrases = list(
                    dict.fromkeys(existing.detector_phrases + candidate.detector_phrases)
                )
        return list(merged.values())

    def _children_map(
        self,
        candidates: list[InterrogationCandidate],
        knowledge_pack: KnowledgePack | None,
        image: NDArray[np.uint8],
    ) -> dict[str, list[str]]:
        children: dict[str, list[str]] = {}
        ranked_candidates = sorted(
            candidates,
            key=lambda c: c.confidence,
            reverse=True,
        )[: min(self._max_child_query_parents(), len(candidates))]
        for candidate in ranked_candidates:
            knowledge = knowledge_pack.find_object(candidate.canonical_label) if knowledge_pack else None
            if knowledge and knowledge.parts:
                children[candidate.display_label] = knowledge.parts
                continue
            try:
                if candidate.confidence < 0.55 or candidate.display_label.lower() in _VAGUE_TERMS:
                    continue
                client = self._get_client(self._settings.primary_vlm)
                parts = client.get_children(image, candidate.display_label)
            except Exception:
                parts = []
            parts = self._filter_child_parts(parts, candidate.display_label)
            if parts:
                children[candidate.display_label] = parts
        return children

    def _iter_tiles(self, image: NDArray[np.uint8]) -> list[NDArray[np.uint8]]:
        h, w = image.shape[:2]
        mid_y = max(1, h // 2)
        mid_x = max(1, w // 2)
        return [
            image[:mid_y, :mid_x],
            image[:mid_y, mid_x:],
            image[mid_y:, :mid_x],
            image[mid_y:, mid_x:],
        ]

    def _prepare_image(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Downscale very large photos before VLM interrogation for speed."""
        h, w = image.shape[:2]
        max_edge = max(h, w)
        if max_edge <= 1280:
            return image
        scale = 1280 / max_edge
        resized = Image.fromarray(image).resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.LANCZOS,
        )
        return np.array(resized)

    def _reason_and_rank(
        self,
        candidates: list[InterrogationCandidate],
        knowledge_pack: KnowledgePack | None,
        raw_responses: dict[str, str],
        stage: str,
    ) -> list[InterrogationCandidate]:
        if not candidates:
            return candidates
        if not self._should_run_reasoner(candidates, knowledge_pack, stage):
            return sorted(candidates, key=lambda c: c.confidence, reverse=True)
        try:
            client = self._get_client(self._settings.reasoner_model)
            prompt = self._build_reasoner_prompt(candidates, knowledge_pack, raw_responses)
            response = client.query_text(prompt)
            raw_responses[f"reasoner:{self._settings.reasoner_model}"] = response
            ranked = self._parse_reasoner_response(response, candidates)
            if ranked:
                return ranked
        except Exception:
            logger.info("Reasoner model unavailable, using heuristic ranking", exc_info=True)
        return sorted(candidates, key=lambda c: c.confidence, reverse=True)

    def _should_run_reasoner(
        self,
        candidates: list[InterrogationCandidate],
        knowledge_pack: KnowledgePack | None,
        stage: str,
    ) -> bool:
        if self._settings.profile == "fast":
            return False
        if self._settings.profile == "deep":
            return True
        if stage not in {"guided", "tiled"} and knowledge_pack is None:
            return False
        if any(c.confidence < 0.72 for c in candidates):
            return True
        if any(c.display_label.lower() in _VAGUE_TERMS for c in candidates):
            return True
        return knowledge_pack is not None and stage != "composition"

    def _should_force_primary_pass(self) -> bool:
        return self._settings.profile == "deep" or self._settings.fallback_mode == "always_enrich"

    def _allows_tiling(self) -> bool:
        return self._settings.profile in {"balanced", "deep"}

    def _max_child_query_parents(self) -> int:
        if self._settings.profile == "fast":
            return 1
        if self._settings.profile == "deep":
            return 5
        return 3

    def _filter_child_parts(self, parts: list[str], parent_label: str) -> list[str]:
        cleaned: list[str] = []
        parent_words = set(parent_label.lower().split())
        max_parts = 3 if self._settings.profile == "fast" else 8 if self._settings.profile == "deep" else 6
        for part in parts:
            item = part.strip().lower()
            if not item or item == parent_label.lower():
                continue
            if item in _PART_BLACKLIST:
                continue
            if set(item.split()) == parent_words:
                continue
            if any(bad in item for bad in _PART_BLACKLIST):
                continue
            if item not in cleaned:
                cleaned.append(item)
            if len(cleaned) >= max_parts:
                break
        return cleaned

    def _build_reasoner_prompt(
        self,
        candidates: list[InterrogationCandidate],
        knowledge_pack: KnowledgePack | None,
        raw_responses: dict[str, str],
    ) -> str:
        domain = knowledge_pack.domain.name if knowledge_pack else "generic objects"
        payload = [
            {
                "canonical_label": c.canonical_label,
                "display_label": c.display_label,
                "detector_phrases": c.detector_phrases,
                "confidence": c.confidence,
            }
            for c in candidates
        ]
        return (
            f"You are ranking visual detector labels for the domain '{domain}'. "
            "Keep canonical labels precise, but prefer generic detector phrases that GroundingDINO can understand. "
            "Return JSON with key 'candidates' containing the same candidates in ranked order. "
            f"Candidates: {json.dumps(payload)} "
            f"Raw model responses: {json.dumps(raw_responses)}"
        )

    def _parse_reasoner_response(
        self,
        response: str,
        existing: list[InterrogationCandidate],
    ) -> list[InterrogationCandidate]:
        try:
            start = response.index("{")
            end = response.rindex("}") + 1
            payload = json.loads(response[start:end])
        except Exception:
            return []
        ranked: list[InterrogationCandidate] = []
        by_label = {c.canonical_label: c for c in existing}
        for item in payload.get("candidates", []):
            label = item.get("canonical_label")
            existing_candidate = by_label.get(label)
            if existing_candidate is None:
                continue
            phrases = item.get("detector_phrases")
            if isinstance(phrases, list) and phrases:
                existing_candidate.detector_phrases = [
                    str(p).strip() for p in phrases if str(p).strip()
                ][: self._settings.max_aliases_per_object]
            ranked.append(existing_candidate)
        return ranked
