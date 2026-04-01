from __future__ import annotations

import base64
import logging
import re
from itertools import chain
from pathlib import Path

import ollama
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Hard limits to prevent Moondream's repetitive token generation
MAX_PARENTS = 8
MAX_CHILDREN = 10
MAX_TOKENS = 150  # cap Ollama output length


class DetectedLabel(BaseModel):
    """A label detected by Moondream semantic interrogation."""

    label: str
    role: str  # "parent" | "child"
    parent: str | None = None  # parent label if role == "child"
    confidence: float = 1.0


_JUNK_RE = re.compile(
    r"^[\d\.\-/,\s]+$"  # pure numbers, decimals, slashes, commas
    r"|^.{0,1}$"         # single char or empty
    r"|/"                # contains slashes (e.g. "cd/mov/x")
)


def _is_valid_label(label: str) -> bool:
    """Reject garbage labels: numbers, coordinates, slash-separated, too short."""
    return not _JUNK_RE.search(label)


def _is_word_subset(short: str, long: str) -> bool:
    """True if every word in *short* appears in *long* (order-independent)."""
    return set(short.split()).issubset(set(long.split()))


def _dedupe(items: list[str], limit: int) -> list[str]:
    """Deduplicate, strip quotes/brackets, lowercase, cap at limit.

    Two-pass semantic dedup using word-subset matching:
    Pass 1: collect all cleaned labels, rejecting garbage (no limit).
    Pass 2: remove labels whose words are a subset of a more-specific label
             (e.g. "monitor" is dropped when "crt monitor" exists), then cap.
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in items:
        clean = re.sub(r"^[\s\[\]'\"]+|[\s\[\]'\"]+$", "", raw).lower()
        if not clean or clean in seen:
            continue
        if not _is_valid_label(clean):
            logger.debug("Rejected garbage label: '%s'", clean)
            continue
        seen.add(clean)
        cleaned.append(clean)

    # Remove labels that are word-subsets of a longer, more-specific label
    to_remove: set[str] = set()
    for i, a in enumerate(cleaned):
        for j, b in enumerate(cleaned):
            if i == j:
                continue
            # If a's words are a strict subset of b's words, drop a
            if len(a.split()) < len(b.split()) and _is_word_subset(a, b):
                to_remove.add(a)
    result = [label for label in cleaned if label not in to_remove]
    return result[:limit]


class MoondreamClient:
    """Ollama HTTP client for Moondream 2 semantic interrogation."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "moondream",
    ) -> None:
        self._client = ollama.Client(host=host)
        self._model = model

    def health_check(self) -> bool:
        """Check if Ollama is reachable and model is available."""
        try:
            models = self._client.list()
            available = [m.model for m in models.models]
            found = any(self._model in name for name in available)
            if not found:
                logger.warning(
                    "Model '%s' not found. Available: %s", self._model, available
                )
            return found
        except Exception:
            logger.error("Ollama health check failed", exc_info=True)
            return False

    def _encode_image(self, image: NDArray[np.uint8]) -> str:
        """Encode numpy RGB image as base64 PNG for Ollama."""
        from PIL import Image
        import io

        pil_img = Image.fromarray(image)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _query(self, image: NDArray[np.uint8], prompt: str) -> str:
        """Send image + prompt to Moondream, return text response.

        Uses num_predict to hard-cap token output and prevent runaway repetition.
        """
        b64 = self._encode_image(image)
        response = self._client.chat(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [b64],
                }
            ],
            options={"num_predict": MAX_TOKENS},
        )
        return response.message.content.strip()

    def query_vision(self, image: NDArray[np.uint8], prompt: str) -> str:
        """Public wrapper for vision-capable Ollama models."""
        return self._query(image, prompt)

    def query_text(self, prompt: str, *, num_predict: int = 256) -> str:
        """Send a text-only prompt to any Ollama chat model."""
        response = self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": num_predict},
        )
        return response.message.content.strip()

    def get_parents(self, image: NDArray[np.uint8]) -> list[str]:
        """Detect main foreground objects (parent labels) in the image."""
        prompts = [
            (
                f"List up to {MAX_PARENTS} main foreground objects in this image. "
                "Reply ONLY with a comma-separated list of object names. "
                "No numbering, no coordinates, no bounding boxes, no explanation. "
                "Use descriptive English names (e.g., 'CRT monitor', 'computer mouse', 'keyboard'). "
                "Do not output numbers or coordinates. "
                "Do not repeat any item. Do not include background."
            ),
            (
                "What objects do you see in this image? "
                "Reply with a comma-separated list of object names only."
            ),
            (
                "Briefly describe the visible objects in this image. "
                "Focus on concrete object nouns."
            ),
        ]

        labels: list[str] = []
        last_raw: list[str] = []
        for idx, prompt in enumerate(prompts):
            response = self._query(image, prompt)
            raw = []
            if response:
                parts = [response]
                for splitter in (",", "\n", ";"):
                    parts = list(chain.from_iterable(part.split(splitter) for part in parts))
                raw = [part.strip() for part in parts if part.strip()]
            last_raw = raw
            labels = _dedupe(raw, MAX_PARENTS)
            if labels:
                break
            if idx == 0:
                logger.warning(
                    "Moondream returned no usable parents, retrying with simpler prompts"
                )
            elif raw:
                logger.warning(
                    "All Moondream labels rejected as garbage (%s), retrying",
                    raw[:5],
                )

        logger.info("Moondream parents: %s", labels)
        if not labels and not last_raw:
            logger.warning("Moondream returned an empty response for parent detection")
        return labels

    def get_children(
        self, image: NDArray[np.uint8], parent_label: str
    ) -> list[str]:
        """Detect sub-parts of a parent object (deduplicated, capped)."""
        prompt = (
            f"Name up to {MAX_CHILDREN} distinct visible parts of the '{parent_label}' ONLY. "
            "Do NOT list parts belonging to other objects in the image. "
            "Reply ONLY with a comma-separated list. No numbering, no explanation. "
            "Do NOT list colours. Do NOT repeat any item."
        )
        response = self._query(image, prompt)
        raw = [c.strip() for c in response.split(",") if c.strip()]
        children = _dedupe(raw, MAX_CHILDREN)
        logger.info("Moondream children of '%s': %s", parent_label, children)
        return children

    def interrogate(
        self,
        image: NDArray[np.uint8],
        confirmed_labels: list[str] | None = None,
    ) -> list[DetectedLabel]:
        """Full interrogation: parents then children for each parent.

        If confirmed_labels is provided:
        - Moondream-detected parents are filtered to only confirmed ones.
        - User-added labels NOT detected by Moondream are added as extra
          parents (so the user can manually specify objects Moondream missed).
        """
        parents = self.get_parents(image)
        if confirmed_labels is not None:
            detected_set = set(parents)
            # Keep only confirmed labels that Moondream detected
            parents = [p for p in parents if p in confirmed_labels]
            # Add user-specified labels that Moondream missed
            for label in confirmed_labels:
                if label.lower() not in {p.lower() for p in detected_set}:
                    logger.info("User-added parent (not detected by Moondream): '%s'", label)
                    parents.append(label.lower())

        # Dedupe parents again (interrogate path)
        parents = _dedupe(parents, MAX_PARENTS)

        results: list[DetectedLabel] = []
        for parent in parents:
            results.append(DetectedLabel(label=parent, role="parent"))
            children = self.get_children(image, parent)
            for child in children:
                results.append(
                    DetectedLabel(
                        label=child, role="child", parent=parent
                    )
                )
        return results
