"""bootstrap.py  --  First-run setup detection and model bootstrapping.

Decides whether this machine already has everything the pipeline needs
(weights on disk + a reachable VLM backend with its models) and, when it
does not, provides the download/pull primitives the setup wizard uses.

On a machine where all models are already present this module is a cheap
no-op check — nothing is downloaded, nothing is moved.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from utils.model_manager import REGISTRY, ModelManager
from utils.preferences import get_models_dir
from utils.security import SecurityError, validate_loopback_url

logger = logging.getLogger(__name__)

# Ollama models the default pipeline configuration uses.
# First entry is the primary interrogation VLM; the rest are the fallback
# chain and the text reasoner (see utils/preferences.py defaults).
REQUIRED_OLLAMA_MODELS = ["qwen2.5vl:3b"]
RECOMMENDED_OLLAMA_MODELS = ["gemma4:e4b", "minicpm-v"]

OLLAMA_INSTALL_URL = "https://ollama.com/download"
# -c 8192 keeps the KV cache small; the server's default context can
# exhaust Metal GPU memory on vision models.
LLAMACPP_LAUNCH_HINT = (
    "llama-server -hf Qwen/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M --port 8080 -c 8192"
)


class SetupItem(BaseModel):
    """One component of the first-run checklist."""

    name: str
    kind: str  # "weights" | "ollama_model" | "backend"
    status: str  # "ready" | "missing"
    required: bool = True
    detail: str = ""
    approx_mb: int | None = None


class SetupStatus(BaseModel):
    """Aggregated first-run status."""

    items: list[SetupItem]

    @property
    def missing_required(self) -> list[SetupItem]:
        return [i for i in self.items if i.required and i.status != "ready"]

    @property
    def complete(self) -> bool:
        return not self.missing_required


def _list_ollama_models(host: str) -> list[str] | None:
    """Model names on the Ollama server, or None when unreachable."""
    import ollama

    try:
        response = ollama.Client(host=validate_loopback_url(host)).list()
        return [m.model or "" for m in response.models]
    except (OSError, TimeoutError, SecurityError, ollama.RequestError, ollama.ResponseError):
        return None


def check_setup(prefs: dict[str, Any]) -> SetupStatus:
    """Inspect weights on disk and the configured VLM backend."""
    items: list[SetupItem] = []

    # 1. Pipeline weights (GroundingDINO, SAM 2.1, VitMatte, GSAM source)
    manager = ModelManager(get_models_dir(prefs))
    missing = set(manager.missing())
    for name, entry in REGISTRY.items():
        approx = entry.get("approx_mb")
        items.append(
            SetupItem(
                name=str(entry.get("display_name", name)),
                kind="weights",
                status="missing" if name in missing else "ready",
                detail=name,
                approx_mb=approx if isinstance(approx, int) else None,
            )
        )

    # 2. VLM backend
    backend = str(prefs.get("vlm_backend", "ollama"))
    if backend == "local":
        from models.local_vlm import resolve_local_model, server_binary, LOCAL_PRIMARY, LOCAL_FALLBACK
        for model in dict.fromkeys([prefs.get("local_primary_model", LOCAL_PRIMARY), prefs.get("local_fallback_model", LOCAL_FALLBACK)]):
            try:
                binary = server_binary()
                weights, projector = resolve_local_model(model)
                detail = f"{weights.name} + {projector.name}; {binary}"
                ready = True
            except (OSError, ValueError) as exc:
                ready, detail = False, str(exc)
            items.append(SetupItem(name=f"Local {model}", kind="backend", status="ready" if ready else "missing", detail=detail))
        sam3_root = get_models_dir(prefs) / "mlx_sam3"
        if prefs.get("segmentation_backend") == "mlx-sam3":
            present = (sam3_root / "sam3-mod-weights/model.safetensors").is_file()
            items.append(SetupItem(name="MLX SAM 3", kind="backend", status="ready" if present else "missing",
                                   required=False, detail=str(sam3_root)))
        return SetupStatus(items=items)
    if backend == "llamacpp":
        from models.vlm_client import DEFAULT_LLAMACPP_URL, LlamaCppVLMClient

        host = str(prefs.get("llamacpp_url", DEFAULT_LLAMACPP_URL))
        try:
            reachable = LlamaCppVLMClient(host=host).health_check()
        except SecurityError:
            reachable = False
        items.append(
            SetupItem(
                name=f"llama.cpp server ({host})",
                kind="backend",
                status="ready" if reachable else "missing",
                detail=f"Start it with: {LLAMACPP_LAUNCH_HINT}",
            )
        )
        return SetupStatus(items=items)

    host = str(prefs.get("ollama_url", "http://localhost:11434"))
    available = _list_ollama_models(host)
    items.append(
        SetupItem(
            name=f"Ollama server ({host})",
            kind="backend",
            status="ready" if available is not None else "missing",
            detail=f"Install from {OLLAMA_INSTALL_URL} and run 'ollama serve'",
        )
    )

    def _has(model: str) -> bool:
        return available is not None and any(model in name for name in available)

    primary = str(prefs.get("ollama_model", REQUIRED_OLLAMA_MODELS[0]))
    fallback = str(prefs.get("preferred_fallback_vlm", ""))
    reasoner = str(prefs.get("preferred_text_reasoner", ""))
    required_models = list(dict.fromkeys([primary, *REQUIRED_OLLAMA_MODELS]))
    recommended_models = [
        m
        for m in dict.fromkeys([fallback, reasoner, *RECOMMENDED_OLLAMA_MODELS])
        if m and m not in required_models
    ]

    for model in required_models:
        items.append(
            SetupItem(
                name=f"Ollama model {model}",
                kind="ollama_model",
                status="ready" if _has(model) else "missing",
                detail=model,
            )
        )
    for model in recommended_models:
        items.append(
            SetupItem(
                name=f"Ollama model {model}",
                kind="ollama_model",
                status="ready" if _has(model) else "missing",
                required=False,
                detail=model,
            )
        )
    return SetupStatus(items=items)


def is_setup_complete(prefs: dict[str, Any]) -> bool:
    """True when every required component is ready on this machine."""
    try:
        return check_setup(prefs).complete
    except (ConnectionError, OSError, RuntimeError, SecurityError, TimeoutError, ValueError):
        logger.warning("First-run setup check failed", exc_info=True)
        # Do not tell callers the machine is ready when the verification was
        # inconclusive: the setup wizard makes the failure visible and avoids
        # an opaque later pipeline error.
        return False


def pull_ollama_model(
    host: str,
    model: str,
    progress_callback: Callable[[str, int, int | None], None] | None = None,
) -> None:
    """Pull a model onto the local Ollama server, streaming progress.

    progress_callback receives (status_text, completed_bytes, total_bytes).
    """
    import ollama

    client = ollama.Client(host=validate_loopback_url(host))
    for update in client.pull(model, stream=True):
        if progress_callback is None:
            continue
        progress_callback(
            str(getattr(update, "status", "")),
            int(getattr(update, "completed", 0) or 0),
            getattr(update, "total", None),
        )
    logger.info("Ollama pull complete: %s", model)


def download_missing_weights(
    prefs: dict[str, Any],
    progress_callback: Callable[[str, int, int | None], None] | None = None,
) -> list[str]:
    """Download every missing registry entry. Returns names that failed."""
    manager = ModelManager(get_models_dir(prefs))
    failures: list[str] = []
    for name in manager.missing():
        try:
            def _cb(done: int, total: int | None) -> None:
                if progress_callback is not None:
                    progress_callback(name, done, total)

            manager.ensure(name, progress_callback=_cb)
        except (
            ConnectionError,
            KeyError,
            OSError,
            RuntimeError,
            SecurityError,
            TimeoutError,
            ValueError,
        ):
            logger.error("Failed to download %s", name, exc_info=True)
            failures.append(name)
    return failures
