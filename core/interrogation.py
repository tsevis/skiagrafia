from __future__ import annotations

import json
import logging
import re
from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from core.interrogation_types import (
    _VAGUE_TERMS,
    InterrogationCandidate,
    InterrogationResult,
    InterrogationSettings,
)
from core.knowledge import KnowledgePack, ObjectKnowledge
from core.reasoner_stage import ReasonerStageMixin
from core.typography_labels import (
    GlyphInspection,
    is_individual_glyph_label,
    is_typography_label,
    parse_typography_observation,
)
from models.vlm_client import (
    MAX_PARENTS,
    BaseVLMClient,
    VLMResponseError,
    create_vlm_client,
)

logger = logging.getLogger(__name__)

# Smallest quadrant worth sending to a vision model on its own. Tiling splits
# an image in half on both axes, so an image must be twice this on its shorter
# side before the four extra passes can show anything the whole image did not.
MIN_TILE_EDGE_PX = 256

# MAX_TOKENS is sized for a short list answer and is far too small for one
# JSON object per visible glyph. Measured with Qwen3-VL-8B-Instruct on
# synthetic grids: 8 glyphs answered in 325 characters and fitted inside the
# 200-token default, while 12 glyphs was truncated. At 800 tokens the same
# model returned 12, 16 and 24 glyphs complete, the largest of those in 939
# characters. This does not rescue a dense collage: on a 600x450 crop of
# Pete1_6000_on_grey.jpg the model degenerates into repeating one word with
# out-of-range coordinates and never closes the JSON, at any budget tried up
# to 4000 tokens.
GLYPH_READING_MAX_TOKENS = 800

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







@overload
def parse_label_candidates(raw: str, limit: int = ...) -> list[str]: ...


@overload
def parse_label_candidates(
    raw: str, limit: int = ..., *, report_dropped: Literal[True]
) -> tuple[list[str], int]: ...


def parse_label_candidates(
    raw: str, limit: int = MAX_PARENTS, *, report_dropped: bool = False
) -> list[str] | tuple[list[str], int]:
    """Object names from a model's free-text answer, capped at `limit`.

    With `report_dropped`, also returns how many distinct names were named and
    then discarded by the cap -- information the caller previously had no way
    to recover.
    """
    text = raw.strip()
    if not text:
        return ([], 0) if report_dropped else []
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
    if report_dropped:
        # How many distinct names were named and then thrown away. A model
        # listing twelve object types silently lost four.
        seen: list[str] = []
        for chunk in chunks:
            lowered = chunk.lower()
            if lowered and lowered not in {value.lower() for value in seen}:
                seen.append(chunk)
        return candidates, max(0, len(seen) - len(candidates))
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


class GuidedInterrogator(ReasonerStageMixin):
    def __init__(self, settings: InterrogationSettings) -> None:
        self._settings = settings
        self._clients: dict[str, BaseVLMClient] = {}
        self._unreachable_models: list[str] = []
        self._models_answered = 0

    def interrogate(
        self,
        image: NDArray[np.uint8],
        confirmed_labels: list[str] | None = None,
        knowledge_pack: KnowledgePack | None = None,
    ) -> InterrogationResult:
        if confirmed_labels is not None:
            candidates = self._candidates_from_confirmed_labels(
                confirmed_labels, knowledge_pack
            )
            return InterrogationResult(
                candidates=candidates,
                children_by_parent=self._known_parts(candidates, knowledge_pack),
                escalation_stage="confirmed",
                confidence_summary="user-confirmed labels",
            )

        stage = "primary"
        raw_responses: dict[str, str] = {}
        candidates = []
        # Reset per run: this records whether THIS image was looked at.
        self._unreachable_models = []
        self._models_answered = 0

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

        if self._settings.fallback_mode != "moondream_only" and (self._should_escalate(candidates) or self._settings.profile == "deep"):
            for model in self._settings.fallback_vlms:
                if model == self._settings.primary_vlm:
                    continue  # same model would just repeat the guided pass
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

        if (
            self._settings.enable_tiling
            and self._allows_tiling()
            and self._tiles_are_useful(image)
            and self._should_escalate(candidates)
        ):
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
            "object proposals; detection still required"
            if not self._should_escalate(candidates)
            else "uncertain object proposals; review required"
        )
        unavailable = bool(self._unreachable_models) and self._models_answered == 0
        if unavailable:
            confidence_summary = (
                "could not reach any vision model: "
                + ", ".join(dict.fromkeys(self._unreachable_models))
            )
        return InterrogationResult(
            candidates=candidates,
            children_by_parent=self._known_parts(candidates, knowledge_pack),
            raw_responses=raw_responses,
            escalation_stage=stage,
            confidence_summary=confidence_summary,
            vision_unavailable=unavailable,
        )

    def set_confirmed_selections(self, selections: dict[str, str]) -> None:
        """Apply a per-image instance policy for the next confirmed-label pass."""
        self._settings.selections = dict(selections)

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
        except (OSError, TimeoutError, ConnectionError, RuntimeError, VLMResponseError) as exc:
            logger.warning("Vision interrogation failed for model %s", model, exc_info=True)
            # Distinguished from an empty answer: the caller must be able to
            # say "could not look" rather than "found nothing".
            self._unreachable_models.append(f"{model} ({type(exc).__name__})")
            return []

        self._models_answered += 1
        raw_responses[f"{prompt_style}:{model}"] = response
        selections = {}
        if self._selection_request:
            try:
                content = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                objects = json.loads(content)["objects"]
                labels = []
                for obj in objects[:MAX_PARENTS]:
                    label = obj["label"].strip().lower()
                    selection = obj.get("selection", "all")
                    if label and selection in {"all", "leftmost", "rightmost", "largest", "smallest"}:
                        labels.append(label)
                        selections[label] = selection
            except (ValueError, KeyError, TypeError, AttributeError):
                logger.warning("Rejected incomplete or malformed object request response")
                return []
        else:
            labels = parse_label_candidates(response)
        candidates = [
            self._candidate_from_label(
                label=label,
                knowledge=knowledge_pack.find_object(label) if knowledge_pack else None,
                source_model=model,
                confidence=0.8 if prompt_style == "primary" else 0.72,
            )
            for label in labels
        ]
        for source_label, candidate in zip(labels, candidates, strict=True):
            candidate.selection = selections.get(
                source_label,
                selections.get(
                    candidate.display_label,
                    selections.get(candidate.canonical_label, "all"),
                ),
            )
        return candidates

    def _get_client(self, model: str) -> BaseVLMClient:
        client = self._clients.get(model)
        if client is None:
            client = create_vlm_client(
                backend=self._settings.backend,
                host=self._settings.host,
                model=model,
            )
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
            for obj in knowledge_pack.objects:
                sample_terms.append(obj.canonical)
                sample_terms.extend(obj.aliases[:1])
            exemplar_text = ", ".join(dict.fromkeys(sample_terms))
            exemplars = f"Possible object families include: {exemplar_text}. "

        exclusions = ""
        if knowledge_pack and knowledge_pack.domain.exclusions:
            exclusions = (
                "Never select these domain exclusions: "
                + ", ".join(knowledge_pack.domain.exclusions)
                + ". "
            )

        if self._selection_request:
            return (
                f"{domain_prefix}{domain_desc}{exemplars}{exclusions}"
                "Select only visible objects that satisfy the batch Selection Request. "
                "The Domain Guide provides naming and detector vocabulary; the Selection Request "
                "controls what is in or out for this batch. Apply both sets of exclusions. "
                "Use short concrete noun phrases retaining useful color or material attributes. "
                "Do not include counts in labels. Separate distinct object types; repeated instances share a label. "
                "For left/right/largest/smallest requests use selection leftmost/rightmost/largest/smallest; otherwise all. "
                'Return ONLY JSON: {"objects":[{"label":"object name","selection":"all"}]}. '
                'Return {"objects":[]} when nothing requested is visible. '
                f"Selection Request: {json.dumps(self._selection_request)}"
            )

        if prompt_style == "composition":
            return (
                f"{domain_prefix}{domain_desc}{exemplars}{exclusions}"
                "Name each separate visible foreground object type. Do not merge touching or stacked objects into a group. "
                "Include small recognizable objects. Exclude scenery and empty spaces. "
                "If the exact specialist term is unknown, use short visual nouns based on shape, material, or purpose. "
                "Reply only as a comma-separated list of the main whole objects."
            )
        if prompt_style == "guided":
            return (
                f"{domain_prefix}{domain_desc}{exemplars}{exclusions}"
                "Identify the main foreground objects. "
                "If the exact specialist term is unknown, reply with concrete visual nouns. "
                "Prefer simple detector-friendly phrases that describe the visible object plainly. "
                "Reply only as a comma-separated object list."
            )
        return (
            f"{domain_prefix}{domain_desc}{exclusions}"
            "List the main foreground objects in this image. "
            "Reply only as a comma-separated list of object names. "
            "If the exact name is unknown, use simple visual nouns."
        )

    @property
    def _selection_request(self) -> str:
        """Return the explicit request, with the legacy Single alias as fallback."""
        return self._settings.selection_request.strip() or self._settings.user_prompt.strip()

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
        for candidate in candidates:
            candidate.selection = (self._settings.selections or {}).get(candidate.display_label, "all")
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
        # Text-to-instance models localize an individual ``letter`` prompt
        # much more reliably than a plural ``letters`` prompt, which can also
        # yield a whole-word region. Keep the original label as an alias for
        # other detector backends, but always query the atomic term first.
        glyph_singulars = {
            "letters": "letter",
            "glyphs": "glyph",
            "characters": "character",
            "digits": "digit",
            "numerals": "numeral",
            "numbers": "digit",
        }
        words = normalized.split()
        if is_individual_glyph_label(normalized) and words:
            atomic = glyph_singulars.get(words[-1], words[-1])
            terms.insert(0, atomic)
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
        return bool(all(c.display_label.lower() in _VAGUE_TERMS for c in candidates))

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

    @property
    def part_query_limit(self) -> int:
        return self._max_child_query_parents() if self._settings.discover_parts else 0

    def _known_parts(
        self,
        candidates: list[InterrogationCandidate],
        knowledge_pack: KnowledgePack | None,
    ) -> dict[str, list[str]]:
        if not self._settings.discover_parts or knowledge_pack is None:
            return {}
        return {c.display_label: obj.parts for c in candidates
                if (obj := knowledge_pack.find_object(c.canonical_label)) and obj.parts}

    def discover_parts(
        self,
        image: NDArray[np.uint8],
        candidate: InterrogationCandidate,
        knowledge_pack: KnowledgePack | None = None,
    ) -> list[str]:
        """Called only after localization, on an individual object's crop."""
        if not self._settings.discover_parts:
            return []
        # A glyph is already an atomic foreground object for this pipeline.
        # Asking a generative VLM for its "physical sub-parts" produces
        # invented stems/bars/counters and corrupts the parent/child tree.
        # Guide-declared parts remain available through _known_parts(), but
        # automatic visual part discovery is deliberately not used here.
        if is_typography_label(candidate.display_label):
            return []
        return self._children_map([candidate], knowledge_pack, self._prepare_image(image)).get(candidate.display_label, [])

    def inspect_individual_glyphs(
        self,
        image: NDArray[np.uint8],
        candidate: InterrogationCandidate,
    ) -> GlyphInspection:
        """Return a local-VLM reading usable to validate glyph proposals.

        This applies only to an explicit individual-glyph category such as
        ``letters`` or ``digits``.  It asks for approximate, normalized boxes
        so the orchestrator can require agreement with independently detected
        boxes.  The VLM result never becomes a mask or an output box itself.

        Returns a GlyphInspection rather than an observation, so a caller can
        tell "not a glyph label" from "asked, and got nothing usable" and warn
        about the second.
        """
        if not is_individual_glyph_label(candidate.display_label):
            return GlyphInspection()
        prompt = (
            "Inspect the image as individual typography. Return ONLY JSON in this exact shape: "
            '{"elements":[{"glyph":"visible character","bbox":[x0,y0,x1,y1]}]}. '
            "List every visible glyph exactly once in reading order, including repeats. "
            "Each bbox is an approximate 0-to-1000 coordinate box relative to the full image, "
            "with origin at top left. Do not include background, decorative strokes, word-level "
            "regions, neighbouring objects, inferred characters, or partial composite regions."
        )
        try:
            client = self._get_client(self._settings.primary_vlm)
            observation = parse_typography_observation(
                client.query_vision(
                    self._prepare_image(image),
                    prompt,
                    num_predict=GLYPH_READING_MAX_TOKENS,
                )
            )
        except (OSError, TimeoutError, ConnectionError, RuntimeError, VLMResponseError) as exc:
            logger.info("Typography validation unavailable for '%s'", candidate.display_label, exc_info=True)
            return GlyphInspection(unavailable_reason=str(exc) or type(exc).__name__)
        if observation:
            logger.info(
                "%s typography observation for '%s': %s",
                self._settings.primary_vlm,
                candidate.display_label,
                observation.glyphs,
            )
            return GlyphInspection(observation=observation)
        logger.warning("Rejected malformed typography observation for '%s'", candidate.display_label)
        return GlyphInspection(
            unavailable_reason="the model's reading could not be read as individual glyphs"
        )

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
                if (
                    candidate.confidence < 0.55
                    or candidate.display_label.lower() in _VAGUE_TERMS
                    or is_typography_label(candidate.display_label)
                ):
                    continue
                client = self._get_client(self._settings.primary_vlm)
                parts = client.get_children(image, candidate.display_label)
            except (OSError, TimeoutError, ConnectionError, RuntimeError, VLMResponseError):
                logger.warning("Part interrogation failed for '%s'", candidate.display_label, exc_info=True)
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
            Image.Resampling.LANCZOS,
        )
        return np.array(resized)



    def _should_force_primary_pass(self) -> bool:
        return self._settings.profile == "deep" or self._settings.fallback_mode == "always_enrich"

    def _allows_tiling(self) -> bool:
        return self._settings.profile in {"balanced", "deep"}

    def _tiles_are_useful(self, image: NDArray[np.uint8]) -> bool:
        """True when splitting this image into quadrants can add detail.

        A tile is half the width and half the height, so below this size
        each quadrant carries LESS than the whole picture already showed --
        and the run pays four full model round-trips to learn nothing. The
        threshold keeps a quadrant at or above MIN_TILE_EDGE_PX.
        """
        return min(image.shape[:2]) >= MIN_TILE_EDGE_PX * 2

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


