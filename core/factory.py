"""factory.py  --  v5.0 Capability Wiring

Builds a fully configured CapabilitySet from user preferences.
Called by UI code (left_panel, step_progress, batch_runner) just before
creating the Orchestrator.

This is where all "which model, which path, which threshold" decisions live.
Safe to call inside a ProcessPoolExecutor worker -- no global mutable state,
no singletons, no tkinter references.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.contracts import CapabilitySet
from core.interrogation import GuidedInterrogator, InterrogationSettings
from core.knowledge import KnowledgePack
from models.grounded_sam import GroundedSAM
from models.vitmatte_refiner import VitMatteRefiner
from processors.vectorizer import VTracerVectorizer
from utils.model_manager import ModelManager
from utils.preferences import get_models_dir


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
        InterrogationSettings(
            host=prefs.get("ollama_url", "http://localhost:11434"),
            primary_vlm=str(
                kp_defaults.get(
                    "preferred_vlm",
                    prefs.get("ollama_model", "moondream"),
                )
            ),
            fallback_vlms=[
                prefs.get("preferred_fallback_vlm", "minicpm-v"),
                "llava:7b",
            ],
            reasoner_model=prefs.get("preferred_text_reasoner", "qwen3.5"),
            profile=prefs.get("interrogation_profile", "balanced"),
            fallback_mode=prefs.get("interrogation_fallback_mode", "adaptive_auto"),
            enable_tiling=prefs.get("enable_tiled_fallback", True),
            max_aliases_per_object=prefs.get("max_aliases_per_object", 4),
        )
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
