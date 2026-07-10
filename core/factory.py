"""factory.py  --  v5.0 Capability Wiring

Builds a fully configured CapabilitySet from user preferences.
Called by UI code (left_panel, step_progress, batch_runner) just before
creating the Orchestrator.

This is where all "which model, which path, which threshold" decisions live.
Safe to call inside a ProcessPoolExecutor worker -- no global mutable state,
no singletons, no tkinter references.
"""
from __future__ import annotations

from typing import Any

from core.contracts import CapabilitySet
from core.interrogation import GuidedInterrogator, InterrogationSettings
from core.knowledge import KnowledgePack
from models.grounded_sam import GroundedSAM
from models.vitmatte_refiner import VitMatteRefiner
from models.vlm_client import (
    BACKEND_LLAMACPP,
    BACKEND_OLLAMA,
    DEFAULT_LLAMACPP_URL,
    DEFAULT_OLLAMA_URL,
)
from processors.vectorizer import VTracerVectorizer
from utils.model_manager import ModelManager
from utils.preferences import get_models_dir


def build_interrogation_settings(
    prefs: dict[str, Any],
    *,
    kp_defaults: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> InterrogationSettings:
    """Build InterrogationSettings from preferences (single source of truth).

    Backend resolution:
    - "ollama"   -- primary + fallback chain + separate text reasoner.
    - "llamacpp" -- one llama.cpp server hosts ONE loaded model, so there is
      no fallback chain and the same model acts as the text reasoner.

    Parameters
    ----------
    prefs : dict
        User preferences (same shape as load_preferences() output).
    kp_defaults : dict, optional
        Knowledge-pack batch defaults (may set "preferred_vlm").
    overrides : dict, optional
        Per-run overrides: preferred_vlm, text_reasoner_model, profile,
        fallback_mode, enable_tiling.
    """
    kp_defaults = kp_defaults or {}
    overrides = overrides or {}

    backend = str(prefs.get("vlm_backend", BACKEND_OLLAMA))
    if backend == BACKEND_LLAMACPP:
        host = str(prefs.get("llamacpp_url", DEFAULT_LLAMACPP_URL))
        default_model = str(prefs.get("llamacpp_model", "Qwen3-VL-8B-Instruct"))
        fallback_vlms: list[str] = []
        default_reasoner = default_model
    else:
        backend = BACKEND_OLLAMA
        host = str(prefs.get("ollama_url", DEFAULT_OLLAMA_URL))
        default_model = str(prefs.get("ollama_model", "qwen2.5vl:3b"))
        fallback_pool = [
            str(prefs.get("preferred_fallback_vlm", "gemma4:e4b")),
            "minicpm-v",
        ]
        fallback_vlms = list(dict.fromkeys(m for m in fallback_pool if m))
        default_reasoner = str(prefs.get("preferred_text_reasoner", "gemma4:e4b"))

    primary_vlm = str(
        overrides.get("preferred_vlm")
        or kp_defaults.get("preferred_vlm")
        or default_model
    )
    fallback_vlms = [m for m in fallback_vlms if m != primary_vlm]

    return InterrogationSettings(
        host=host,
        primary_vlm=primary_vlm,
        fallback_vlms=fallback_vlms,
        reasoner_model=str(overrides.get("text_reasoner_model") or default_reasoner),
        backend=backend,
        profile=str(
            overrides.get("profile")
            or prefs.get("interrogation_profile", "balanced")
        ),
        fallback_mode=str(
            overrides.get("fallback_mode")
            or prefs.get("interrogation_fallback_mode", "adaptive_auto")
        ),
        enable_tiling=bool(
            overrides.get(
                "enable_tiling",
                prefs.get("enable_tiled_fallback", True),
            )
        ),
        max_aliases_per_object=int(prefs.get("max_aliases_per_object", 4)),
    )


def build_capabilities(
    prefs: dict[str, Any],
    *,
    corner_threshold: int | None = None,
    length_threshold: float | None = None,
    filter_speckle: int | None = None,
    splice_threshold: int | None = None,
    knowledge_pack_path: str | None = None,
    knowledge_pack_defaults: dict[str, Any] | None = None,
) -> CapabilitySet:
    """Build a fully wired CapabilitySet from user preferences.

    Parameters
    ----------
    prefs : dict
        User preferences (same shape as load_preferences() output).
    corner_threshold, length_threshold, filter_speckle, splice_threshold :
        Overrides for VTracer parameters (from Single mode sliders).
    knowledge_pack_path : str, optional
        Path to knowledge pack JSON for the interrogator.
    knowledge_pack_defaults : dict, optional
        Override defaults from the knowledge pack (e.g. preferred_vlm).
    """
    kp_defaults = knowledge_pack_defaults or {}

    # 1. Resolve models_dir and create ModelManager
    models_dir = get_models_dir(prefs)
    mgr = ModelManager(models_dir)

    # 2. Build GuidedInterrogator
    interrogator = GuidedInterrogator(
        build_interrogation_settings(prefs, kp_defaults=kp_defaults)
    )

    # 3. Build GroundedSAM (serves as both Detector and Segmenter)
    gsam_root = mgr.resolve("groundingdino_swint_ogc.pth").parent.parent
    sam = GroundedSAM(
        dino_weights=mgr.resolve("groundingdino_swint_ogc.pth"),
        sam_weights=mgr.resolve("sam2.1_hiera_large.pt"),
        gsam_root=gsam_root,
    )

    # 4. Build VitMatteRefiner
    alpha_refiner = VitMatteRefiner(
        model_dir=mgr.resolve("vitmatte-base-composition-1k"),
    )

    # 5. Build VTracerVectorizer
    vectorizer = VTracerVectorizer(
        corner_threshold=corner_threshold or int(prefs.get("vtracer_corner_threshold", 60)),
        length_threshold=length_threshold or float(prefs.get("vtracer_length_threshold", 4.0)),
        splice_threshold=splice_threshold or 45,
        filter_speckle=filter_speckle or int(prefs.get("vtracer_speckle", 8)),
    )

    return CapabilitySet(
        interrogator=interrogator,
        detector=sam,
        segmenter=sam,
        alpha_refiner=alpha_refiner,
        vectorizer=vectorizer,
    )


def build_knowledge_pack(
    knowledge_pack_path: str | None,
) -> KnowledgePack | None:
    """Load a KnowledgePack from path, or return None."""
    if knowledge_pack_path:
        return KnowledgePack.load(knowledge_pack_path)
    return None
