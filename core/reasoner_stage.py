"""reasoner_stage.py  --  the text pass that re-ranks vision candidates.

After the vision stages have proposed labels, an optional text model is
asked to judge them against the domain guide and the raw replies. Split out
of interrogation.py, which had grown past this project's 800-line limit.

A mixin, for the same reason as core/detection_policy.py: these need the
settings and the client pool the interrogator was built with, and threading
those through each call would say less than naming them once.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from core.interrogation_types import _VAGUE_TERMS, InterrogationCandidate
from models.vlm_client import VLMResponseError

if TYPE_CHECKING:
    from core.knowledge import KnowledgePack

logger = logging.getLogger(__name__)


class ReasonerStageMixin:
    """The reasoner pass for GuidedInterrogator, which mixes this in.

    Provided by the interrogator: annotations with no assignment, so nothing
    exists at runtime and the MRO is untouched.
    """

    _settings: Any

    def _get_client(self, model: str) -> Any: ...

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
        except (OSError, TimeoutError, ConnectionError, RuntimeError, VLMResponseError):
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
        except (ValueError, json.JSONDecodeError, TypeError):
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
            if existing_candidate not in ranked:
                ranked.append(existing_candidate)
        if ranked:
            ranked.extend(candidate for candidate in existing if candidate not in ranked)
        return ranked
