"""vlm_client.py  --  Backend-agnostic local VLM clients.

Two interchangeable transports for semantic interrogation:

- ``OllamaVLMClient``    -- Ollama server (default http://localhost:11434)
- ``LlamaCppVLMClient``  -- llama.cpp server, OpenAI-compatible API
                            (default http://localhost:8080)

Both share the same prompt strategy, label parsing, and dedup logic via
``BaseVLMClient``; only the HTTP transport differs. Use ``create_vlm_client``
to construct the right one from preferences.

llama.cpp serves a single loaded model, e.g.:

    llama-server -hf Qwen/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M --port 8080 -c 8192
"""
from __future__ import annotations

import base64
import json
import logging
import re
import urllib.error
import urllib.request
from itertools import chain

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Backend identifiers (stored in preferences under "vlm_backend")
BACKEND_OLLAMA = "ollama"
BACKEND_LLAMACPP = "llamacpp"
VALID_BACKENDS = (BACKEND_OLLAMA, BACKEND_LLAMACPP)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_LLAMACPP_URL = "http://localhost:8080"

# Hard limits to prevent repetitive token generation from small VLMs
MAX_PARENTS = 8
MAX_CHILDREN = 10
MAX_TOKENS = 200  # cap chat output length for list-style answers

# llama.cpp vision inference on Apple Silicon can take minutes for the
# first request while the model warms up.
LLAMACPP_TIMEOUT_S = 300.0
LLAMACPP_HEALTH_TIMEOUT_S = 5.0


class DetectedLabel(BaseModel):
    """A label detected by VLM semantic interrogation."""

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


_NUMBERING_RE = re.compile(
    r"^\s*\d+[\].):\-]\s*"  # strips "1] ", "2. ", "3) ", "4- " etc.
)


def _split_children_response(response: str) -> list[str]:
    """Split and clean a VLM children response.

    Handles comma-separated, newline-separated, and numbered lists.
    Strips numbering artifacts like '1]', '2.', '3)' that small VLMs
    sometimes produce.
    """
    parts: list[str] = [response]
    for splitter in (",", "\n", ";"):
        parts = list(chain.from_iterable(p.split(splitter) for p in parts))
    cleaned: list[str] = []
    for part in parts:
        item = _NUMBERING_RE.sub("", part).strip()
        if item:
            cleaned.append(item)
    return cleaned


class BaseVLMClient:
    """Shared prompt strategy and label parsing for all VLM backends.

    Subclasses implement the transport: ``_chat`` and ``health_check``.
    """

    backend: str = "base"

    def __init__(self, host: str, model: str) -> None:
        self._host = host
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @property
    def host(self) -> str:
        return self._host

    # ── Transport interface (implemented by subclasses) ────────────────

    def _chat(
        self,
        prompt: str,
        images_b64: list[str] | None = None,
        num_predict: int = MAX_TOKENS,
    ) -> str:
        """Send a chat request, return the assistant's text response."""
        raise NotImplementedError

    def health_check(self) -> bool:
        """Check the server is reachable and the model is available."""
        raise NotImplementedError

    # ── Shared helpers ──────────────────────────────────────────────────

    def _encode_image(self, image: NDArray[np.uint8]) -> str:
        """Encode numpy RGB image as base64 PNG."""
        import io

        from PIL import Image

        pil_img = Image.fromarray(image)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def query_vision(self, image: NDArray[np.uint8], prompt: str) -> str:
        """Send image + prompt to the VLM, return text response."""
        return self._chat(prompt, images_b64=[self._encode_image(image)])

    def query_text(self, prompt: str, *, num_predict: int = 256) -> str:
        """Send a text-only prompt to the model."""
        return self._chat(prompt, num_predict=num_predict)

    # ── Interrogation logic (backend-independent) ───────────────────────

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
            response = self.query_vision(image, prompt)
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
                    "VLM returned no usable parents, retrying with simpler prompts"
                )
            elif raw:
                logger.warning(
                    "All VLM labels rejected as garbage (%s), retrying",
                    raw[:5],
                )

        logger.info("%s parents: %s", self._model, labels)
        if not labels and not last_raw:
            logger.warning("VLM returned an empty response for parent detection")
        return labels

    def get_children(
        self, image: NDArray[np.uint8], parent_label: str
    ) -> list[str]:
        """Detect sub-parts of a parent object (deduplicated, capped).

        Runs two prompt strategies and merges results for better coverage.
        """
        prompts = [
            (
                f"What are the visible parts and components of this {parent_label}? "
                "List specific physical parts like buttons, knobs, panels, handles. "
                "Reply ONLY as a comma-separated list. No numbering."
            ),
            (
                f"Name the distinct physical sub-parts of the '{parent_label}' visible in this image. "
                "Focus on parts that have clear outlines: controls, hardware, structural elements. "
                "Reply ONLY as a comma-separated list. No numbering, no explanation."
            ),
        ]

        all_raw: list[str] = []
        for prompt in prompts:
            response = self.query_vision(image, prompt)
            raw = _split_children_response(response)
            all_raw.extend(raw)
            if len(_dedupe(all_raw, MAX_CHILDREN)) >= 4:
                break

        children = _dedupe(all_raw, MAX_CHILDREN)
        logger.info("%s children of '%s': %s", self._model, parent_label, children)
        return children

    def interrogate(
        self,
        image: NDArray[np.uint8],
        confirmed_labels: list[str] | None = None,
    ) -> list[DetectedLabel]:
        """Full interrogation: parents then children for each parent.

        If confirmed_labels is provided:
        - VLM-detected parents are filtered to only confirmed ones.
        - User-added labels NOT detected by the VLM are added as extra
          parents (so the user can manually specify objects the VLM missed).
        """
        parents = self.get_parents(image)
        if confirmed_labels is not None:
            detected_set = set(parents)
            # Keep only confirmed labels that the VLM detected
            parents = [p for p in parents if p in confirmed_labels]
            # Add user-specified labels that the VLM missed
            for label in confirmed_labels:
                if label.lower() not in {p.lower() for p in detected_set}:
                    logger.info("User-added parent (not detected by VLM): '%s'", label)
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


class OllamaVLMClient(BaseVLMClient):
    """Ollama HTTP client for local VLM semantic interrogation."""

    backend = BACKEND_OLLAMA

    def __init__(
        self,
        host: str = DEFAULT_OLLAMA_URL,
        model: str = "qwen2.5vl:3b",
    ) -> None:
        import ollama

        super().__init__(host=host, model=model)
        self._client = ollama.Client(host=host)

    def health_check(self) -> bool:
        """Check if Ollama is reachable and the model is available."""
        try:
            models = self._client.list()
            available = [m.model or "" for m in models.models]
            found = any(self._model in name for name in available)
            if not found:
                logger.warning(
                    "Model '%s' not found. Available: %s", self._model, available
                )
            return found
        except Exception:
            logger.error("Ollama health check failed", exc_info=True)
            return False

    def _chat(
        self,
        prompt: str,
        images_b64: list[str] | None = None,
        num_predict: int = MAX_TOKENS,
    ) -> str:
        message: dict[str, object] = {"role": "user", "content": prompt}
        if images_b64:
            message["images"] = images_b64
        response = self._client.chat(
            model=self._model,
            messages=[message],
            options={"num_predict": num_predict},
        )
        return (response.message.content or "").strip()


class LlamaCppVLMClient(BaseVLMClient):
    """llama.cpp server client (OpenAI-compatible chat completions API).

    A llama.cpp server hosts ONE loaded model; the ``model`` name here is
    informational (the server answers with whatever it has loaded). Vision
    requires the server to be started with a multimodal model + projector,
    e.g.::

        llama-server -hf Qwen/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M --port 8080 -c 8192
    """

    backend = BACKEND_LLAMACPP

    def __init__(
        self,
        host: str = DEFAULT_LLAMACPP_URL,
        model: str = "loaded-model",
        timeout: float = LLAMACPP_TIMEOUT_S,
    ) -> None:
        super().__init__(host=host, model=model)
        self._timeout = timeout

    # ── HTTP helpers ─────────────────────────────────────────────────────

    def _request_json(
        self,
        path: str,
        payload: dict | None = None,
        timeout: float | None = None,
    ) -> dict:
        """GET (payload=None) or POST JSON to the llama.cpp server."""
        url = f"{self._host.rstrip('/')}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=timeout or self._timeout
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            # Surface the server's error body — llama.cpp explains failures
            # (model loading, OOM, missing mmproj) in the response payload.
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            logger.error("llama.cpp HTTP %s from %s: %s", exc.code, path, detail)
            raise
        parsed = json.loads(body)
        if not isinstance(parsed, dict):
            raise ValueError(f"llama.cpp returned non-object JSON from {path}")
        return parsed

    def loaded_model(self) -> str | None:
        """Ask the server which model it has loaded (via /v1/models)."""
        try:
            payload = self._request_json(
                "/v1/models", timeout=LLAMACPP_HEALTH_TIMEOUT_S
            )
            models = payload.get("data", [])
            if models and isinstance(models, list):
                return str(models[0].get("id", "")) or None
        except Exception:
            logger.debug("Could not query llama.cpp /v1/models", exc_info=True)
        return None

    def health_check(self) -> bool:
        """True when the llama.cpp server is up and finished loading."""
        try:
            status = self._request_json("/health", timeout=LLAMACPP_HEALTH_TIMEOUT_S)
        except urllib.error.HTTPError as exc:
            # llama.cpp answers 503 while the model is still loading
            logger.warning("llama.cpp server not ready (HTTP %s)", exc.code)
            return False
        except Exception:
            logger.error("llama.cpp health check failed", exc_info=True)
            return False
        ok = status.get("status") == "ok"
        if ok:
            loaded = self.loaded_model()
            if loaded:
                logger.info("llama.cpp server ready — loaded model: %s", loaded)
        else:
            logger.warning("llama.cpp server status: %s", status)
        return ok

    # ── Chat transport ───────────────────────────────────────────────────

    def _chat(
        self,
        prompt: str,
        images_b64: list[str] | None = None,
        num_predict: int = MAX_TOKENS,
    ) -> str:
        content: str | list[dict]
        if images_b64:
            content = [{"type": "text", "text": prompt}]
            for b64 in images_b64:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )
        else:
            content = prompt

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": num_predict,
        }
        response = self._request_json("/v1/chat/completions", payload)
        choices = response.get("choices") or []
        if not choices:
            raise ValueError(f"llama.cpp returned no choices: {response}")
        message = choices[0].get("message") or {}
        text = message.get("content")
        if not isinstance(text, str):
            raise ValueError(f"llama.cpp returned non-text content: {message}")
        return text.strip()


def create_vlm_client(
    backend: str,
    host: str,
    model: str,
) -> BaseVLMClient:
    """Build the right VLM client for the configured backend.

    Unknown backend values fall back to Ollama with a warning so a corrupt
    preferences file can never make interrogation unavailable.
    """
    if backend == BACKEND_LLAMACPP:
        return LlamaCppVLMClient(host=host, model=model)
    if backend != BACKEND_OLLAMA:
        logger.warning("Unknown VLM backend '%s' — falling back to Ollama", backend)
    return OllamaVLMClient(host=host, model=model)


# Backward-compatibility alias: MoondreamClient was the historical name of
# the Ollama transport before multi-backend support.
MoondreamClient = OllamaVLMClient
