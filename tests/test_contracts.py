"""test_contracts.py  --  v5.0 Protocol conformance & factory smoke tests.

These tests verify structural conformance only -- no model loading needed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Ensure project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.contracts import (
    AlphaRefiner,
    CapabilitySet,
    Detector,
    Interrogator,
    Segmenter,
    Vectorizer,
)
from utils.model_manager import ModelManager, ModelInfo, REGISTRY


# ── Protocol conformance ──────────────────────────────────────────────────


class TestProtocolConformance:
    """Verify each concrete client structurally satisfies its Protocol."""

    def test_guided_interrogator_satisfies_interrogator(self) -> None:
        from core.interrogation import GuidedInterrogator, InterrogationSettings

        settings = InterrogationSettings(
            host="http://localhost:11434",
            primary_vlm="moondream",
            fallback_vlms=["minicpm-v"],
            reasoner_model="qwen3.5",
        )
        instance = GuidedInterrogator(settings)
        assert isinstance(instance, Interrogator)

    def test_grounded_sam_satisfies_detector(self) -> None:
        from models.grounded_sam import GroundedSAM

        instance = GroundedSAM()
        assert isinstance(instance, Detector)

    def test_grounded_sam_satisfies_segmenter(self) -> None:
        from models.grounded_sam import GroundedSAM

        instance = GroundedSAM()
        assert isinstance(instance, Segmenter)

    def test_vitmatte_satisfies_alpha_refiner(self) -> None:
        from models.vitmatte_refiner import VitMatteRefiner

        instance = VitMatteRefiner()
        assert isinstance(instance, AlphaRefiner)

    def test_vtracer_vectorizer_satisfies_vectorizer(self) -> None:
        from processors.vectorizer import VTracerVectorizer

        instance = VTracerVectorizer()
        assert isinstance(instance, Vectorizer)



# ── Helpers for creating protocol-conforming mocks ────────────────────────


def _mock_interrogator() -> MagicMock:
    m = MagicMock()
    m.interrogate = MagicMock()
    return m


def _mock_detector() -> MagicMock:
    m = MagicMock()
    m.detect_box = MagicMock()
    return m


def _mock_segmenter() -> MagicMock:
    m = MagicMock()
    m.segment = MagicMock()
    m.clear_cache = MagicMock()
    return m


def _mock_alpha_refiner() -> MagicMock:
    m = MagicMock()
    m.predict = MagicMock()
    return m


def _mock_vectorizer() -> MagicMock:
    m = MagicMock()
    m.trace = MagicMock()
    return m


def _mock_caps() -> CapabilitySet:
    return CapabilitySet(
        interrogator=_mock_interrogator(),
        detector=_mock_detector(),
        segmenter=_mock_segmenter(),
        alpha_refiner=_mock_alpha_refiner(),
        vectorizer=_mock_vectorizer(),
    )


# ── CapabilitySet construction ────────────────────────────────────────────


class TestCapabilitySet:
    """Verify CapabilitySet bundles can be created and inspected."""

    def test_create_with_all_capabilities(self) -> None:
        caps = _mock_caps()
        assert caps.vectorizer is not None


# ── ModelManager ──────────────────────────────────────────────────────────


class TestModelManager:
    """Verify ModelManager resolves paths under its configured directory."""

    def test_resolve_uses_custom_directory(self, tmp_path: Path) -> None:
        mgr = ModelManager(tmp_path)
        resolved = mgr.resolve("groundingdino_swint_ogc.pth")
        assert str(resolved).startswith(str(tmp_path))

    def test_is_available_false_when_missing(self, tmp_path: Path) -> None:
        mgr = ModelManager(tmp_path)
        assert mgr.is_available("groundingdino_swint_ogc.pth") is False

    def test_scan_returns_model_info_list(self, tmp_path: Path) -> None:
        mgr = ModelManager(tmp_path)
        results = mgr.scan()
        assert len(results) == len(REGISTRY)
        assert all(isinstance(r, ModelInfo) for r in results)
        assert all(r.status == "missing" for r in results)

    def test_models_dir_property(self, tmp_path: Path) -> None:
        mgr = ModelManager(tmp_path)
        assert mgr.models_dir == tmp_path


# ── Orchestrator instantiation ────────────────────────────────────────────


class TestOrchestratorInstantiation:
    """Verify the Orchestrator can be created with a CapabilitySet."""

    def test_create_orchestrator_with_caps(self) -> None:
        from core.orchestrator import Orchestrator

        caps = _mock_caps()
        orch = Orchestrator(capabilities=caps)
        assert orch is not None

    def test_process_signature_unchanged(self) -> None:
        """Verify process() still accepts (image_path, confirmed_labels, manual_detections)."""
        from core.orchestrator import Orchestrator
        import inspect

        sig = inspect.signature(Orchestrator.process)
        params = list(sig.parameters.keys())
        assert params == ["self", "image_path", "confirmed_labels", "manual_detections"]
