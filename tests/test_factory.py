"""test_factory.py  --  build_capabilities wiring & build_knowledge_pack.

build_interrogation_settings is already thoroughly covered by
test_backend_config.py; this file focuses on the remaining, uncovered
factory responsibilities: wiring build_capabilities() into a CapabilitySet,
and the small build_knowledge_pack() loader in core/factory.py.

HARD CONSTRAINT: build_capabilities() would otherwise resolve real model
weight paths via ModelManager and construct GroundedSAM / VitMatteRefiner /
VTracerVectorizer / GuidedInterrogator. Every test below replaces those
factory-module names with lightweight fakes via monkeypatch, so no real
weights are ever touched or loaded, and no network I/O happens.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import factory
from core.contracts import CapabilitySet
from core.knowledge import KnowledgePack


class _FakeModelManager:
    """Records construction/resolve calls; never touches disk beyond a Path."""

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir

    def resolve(self, name: str) -> Path:
        return Path("/fake-models") / name


def _install_fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch every model-touching symbol in core.factory with a fake and
    return a dict this test can inspect for what was passed in."""
    captured: dict[str, Any] = {}

    def fake_grounded_sam(**kwargs: Any) -> object:
        captured["gsam_kwargs"] = kwargs
        # Explicit attribute assignment (not bare MagicMock() auto-attrs) is
        # required: runtime_checkable Protocol.__instancecheck__ uses
        # getattr_static, which does not see MagicMock's lazily
        # auto-created child mocks.
        m = MagicMock()
        m.detect_box = MagicMock()
        m.segment = MagicMock()
        m.clear_cache = MagicMock()
        return m

    def fake_vitmatte(**kwargs: Any) -> object:
        captured["vitmatte_kwargs"] = kwargs
        m = MagicMock()
        m.predict = MagicMock()
        return m

    def fake_vectorizer(**kwargs: Any) -> object:
        captured["vectorizer_kwargs"] = kwargs
        m = MagicMock()
        m.trace = MagicMock()
        return m

    def fake_interrogator(settings: Any) -> object:
        captured["interrogator_settings"] = settings
        m = MagicMock()
        m.interrogate = MagicMock()
        return m

    monkeypatch.setattr(factory, "ModelManager", _FakeModelManager)
    monkeypatch.setattr(factory, "GroundedSAM", fake_grounded_sam)
    monkeypatch.setattr(factory, "VitMatteRefiner", fake_vitmatte)
    monkeypatch.setattr(factory, "VTracerVectorizer", fake_vectorizer)
    monkeypatch.setattr(factory, "GuidedInterrogator", fake_interrogator)
    return captured


def _prefs(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    base = {
        "models_directory": str(tmp_path / "models"),
        "vlm_backend": "ollama",
        "ollama_url": "http://localhost:11434",
        "ollama_model": "qwen2.5vl:3b",
        "vtracer_corner_threshold": 60,
        "vtracer_length_threshold": 4.0,
        "vtracer_speckle": 8,
    }
    base.update(overrides)
    return base


class TestBuildCapabilities:
    def test_returns_capability_set_wired_from_fakes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _install_fakes(monkeypatch)
        prefs = _prefs(tmp_path)

        caps = factory.build_capabilities(prefs)

        assert isinstance(caps, CapabilitySet)
        # GroundedSAM instance is reused as both detector and segmenter.
        assert caps.detector is caps.segmenter
        assert "interrogator_settings" in captured
        assert "vitmatte_kwargs" in captured
        assert "vectorizer_kwargs" in captured

    def test_resolves_gsam_root_from_dino_weights_parent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _install_fakes(monkeypatch)
        prefs = _prefs(tmp_path)

        factory.build_capabilities(prefs)

        gsam_kwargs = captured["gsam_kwargs"]
        assert gsam_kwargs["dino_weights"] == Path(
            "/fake-models/groundingdino_swint_ogc.pth"
        )
        assert gsam_kwargs["sam_weights"] == Path("/fake-models/sam2.1_hiera_large.pt")
        # dino_weights.parent.parent: "/fake-models/x".parent is "/fake-models",
        # and .parent again climbs one more level to "/".
        assert gsam_kwargs["gsam_root"] == Path("/")

    def test_vitmatte_uses_resolved_model_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _install_fakes(monkeypatch)
        prefs = _prefs(tmp_path)

        factory.build_capabilities(prefs)

        assert captured["vitmatte_kwargs"]["model_dir"] == Path(
            "/fake-models/vitmatte-base-composition-1k"
        )

    def test_vectorizer_uses_prefs_defaults_when_no_overrides_given(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _install_fakes(monkeypatch)
        prefs = _prefs(
            tmp_path,
            vtracer_corner_threshold=60,
            vtracer_length_threshold=4.0,
            vtracer_speckle=8,
        )

        factory.build_capabilities(prefs)

        kwargs = captured["vectorizer_kwargs"]
        assert kwargs["corner_threshold"] == 60
        assert kwargs["length_threshold"] == 4.0
        assert kwargs["filter_speckle"] == 8
        assert kwargs["splice_threshold"] == 45

    def test_vectorizer_explicit_overrides_win_over_prefs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _install_fakes(monkeypatch)
        prefs = _prefs(
            tmp_path,
            vtracer_corner_threshold=60,
            vtracer_length_threshold=4.0,
            vtracer_speckle=8,
        )

        factory.build_capabilities(
            prefs,
            corner_threshold=77,
            length_threshold=6.5,
            filter_speckle=3,
            splice_threshold=20,
        )

        kwargs = captured["vectorizer_kwargs"]
        assert kwargs["corner_threshold"] == 77
        assert kwargs["length_threshold"] == 6.5
        assert kwargs["filter_speckle"] == 3
        assert kwargs["splice_threshold"] == 20

    def test_knowledge_pack_defaults_flow_into_interrogation_settings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _install_fakes(monkeypatch)
        prefs = _prefs(tmp_path)

        factory.build_capabilities(
            prefs, knowledge_pack_defaults={"preferred_vlm": "minicpm-v"}
        )

        settings = captured["interrogator_settings"]
        assert settings.primary_vlm == "minicpm-v"

    def test_model_manager_constructed_with_resolved_models_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fakes(monkeypatch)
        models_dir = tmp_path / "custom-models"
        prefs = _prefs(tmp_path, models_directory=str(models_dir))

        # Sanity: FakeModelManager records the dir it was constructed with.
        instances: list[_FakeModelManager] = []
        original_init = _FakeModelManager.__init__

        def recording_init(self: _FakeModelManager, models_dir: Path) -> None:
            original_init(self, models_dir)
            instances.append(self)

        monkeypatch.setattr(_FakeModelManager, "__init__", recording_init)

        factory.build_capabilities(prefs)

        assert len(instances) == 1
        assert instances[0].models_dir == models_dir


class TestFactoryBuildKnowledgePack:
    def test_returns_none_when_no_path_given(self) -> None:
        assert factory.build_knowledge_pack(None) is None

    def test_returns_none_for_empty_string_path(self) -> None:
        assert factory.build_knowledge_pack("") is None

    def test_loads_knowledge_pack_from_given_path(self, tmp_path: Path) -> None:
        guide = tmp_path / "skiagrafia_guide.toml"
        guide.write_text(
            textwrap.dedent(
                """
                [domain]
                name = "Test Domain"
                """
            ),
            encoding="utf-8",
        )

        pack = factory.build_knowledge_pack(str(guide))

        assert isinstance(pack, KnowledgePack)
        assert pack.domain.name == "Test Domain"
