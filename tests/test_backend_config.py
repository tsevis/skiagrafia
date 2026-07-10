"""test_backend_config.py  --  Backend wiring, preferences migration, bootstrap.

Covers build_interrogation_settings resolution, legacy-default migration,
the extended ModelManager registry, and first-run setup detection.
All tests are offline: network probes are mocked.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.factory import build_interrogation_settings
from utils import bootstrap
from utils.model_manager import REGISTRY, ModelManager
from utils.preferences import DEFAULT_PREFERENCES, _migrate_legacy_defaults


# ── build_interrogation_settings ────────────────────────────────────────────


class TestBuildInterrogationSettings:
    def test_ollama_defaults(self) -> None:
        settings = build_interrogation_settings(DEFAULT_PREFERENCES)
        assert settings.backend == "ollama"
        assert settings.host == "http://localhost:11434"
        assert settings.primary_vlm == "qwen2.5vl:3b"
        assert settings.fallback_vlms == ["gemma4:e4b", "minicpm-v"]
        assert settings.reasoner_model == "gemma4:e4b"

    def test_llamacpp_backend_has_no_fallback_chain(self) -> None:
        prefs = dict(DEFAULT_PREFERENCES, vlm_backend="llamacpp")
        settings = build_interrogation_settings(prefs)
        assert settings.backend == "llamacpp"
        assert settings.host == "http://localhost:8080"
        assert settings.primary_vlm == "Qwen3-VL-8B-Instruct"
        assert settings.fallback_vlms == []
        # The single loaded model doubles as the text reasoner
        assert settings.reasoner_model == "Qwen3-VL-8B-Instruct"

    def test_knowledge_pack_preferred_vlm_wins(self) -> None:
        settings = build_interrogation_settings(
            DEFAULT_PREFERENCES, kp_defaults={"preferred_vlm": "minicpm-v"}
        )
        assert settings.primary_vlm == "minicpm-v"

    def test_overrides_beat_knowledge_pack(self) -> None:
        settings = build_interrogation_settings(
            DEFAULT_PREFERENCES,
            kp_defaults={"preferred_vlm": "minicpm-v"},
            overrides={"preferred_vlm": "gemma4:e4b", "profile": "deep"},
        )
        assert settings.primary_vlm == "gemma4:e4b"
        assert settings.profile == "deep"

    def test_primary_excluded_from_fallbacks(self) -> None:
        prefs = dict(DEFAULT_PREFERENCES, ollama_model="gemma4:e4b")
        settings = build_interrogation_settings(prefs)
        assert "gemma4:e4b" not in settings.fallback_vlms

    def test_unknown_backend_normalized_to_ollama(self) -> None:
        prefs = dict(DEFAULT_PREFERENCES, vlm_backend="banana")
        settings = build_interrogation_settings(prefs)
        assert settings.backend == "ollama"


# ── Preferences migration ───────────────────────────────────────────────────


class TestLegacyDefaultMigration:
    def test_old_defaults_upgraded(self) -> None:
        saved = {
            "ollama_model": "moondream",
            "preferred_fallback_vlm": "minicpm-v",
            "preferred_text_reasoner": "qwen3.5",
        }
        migrated = _migrate_legacy_defaults(saved)
        assert migrated["ollama_model"] == "qwen2.5vl:3b"
        assert migrated["preferred_fallback_vlm"] == "gemma4:e4b"
        assert migrated["preferred_text_reasoner"] == "gemma4:e4b"

    def test_custom_choices_untouched(self) -> None:
        saved = {"ollama_model": "llava:7b", "preferred_text_reasoner": "my-model"}
        migrated = _migrate_legacy_defaults(saved)
        assert migrated == saved

    def test_input_not_mutated(self) -> None:
        saved = {"ollama_model": "moondream"}
        _migrate_legacy_defaults(saved)
        assert saved["ollama_model"] == "moondream"

    def test_new_defaults_include_backend_keys(self) -> None:
        assert DEFAULT_PREFERENCES["vlm_backend"] == "ollama"
        assert DEFAULT_PREFERENCES["llamacpp_url"] == "http://localhost:8080"
        assert "llamacpp_model" in DEFAULT_PREFERENCES


# ── ModelManager registry & bootstrap ───────────────────────────────────────


class TestModelManagerBootstrap:
    def test_registry_covers_full_pipeline(self) -> None:
        assert "grounded-sam-2-source" in REGISTRY
        assert "groundingdino_swint_ogc.pth" in REGISTRY
        assert "sam2.1_hiera_large.pt" in REGISTRY
        assert "vitmatte-base-composition-1k" in REGISTRY
        assert all("url" in entry for entry in REGISTRY.values())

    def test_missing_lists_everything_in_empty_dir(self, tmp_path: Path) -> None:
        mgr = ModelManager(tmp_path)
        assert set(mgr.missing()) == set(REGISTRY)

    def test_hf_files_availability_requires_all_files(self, tmp_path: Path) -> None:
        mgr = ModelManager(tmp_path)
        vitmatte_dir = tmp_path / "vitmatte-base-composition-1k"
        vitmatte_dir.mkdir()
        (vitmatte_dir / "config.json").write_text("{}")
        assert mgr.is_available("vitmatte-base-composition-1k") is False
        # Configs alone are not enough — a weight file is required
        (vitmatte_dir / "preprocessor_config.json").write_text("{}")
        assert mgr.is_available("vitmatte-base-composition-1k") is False
        (vitmatte_dir / "model.safetensors").write_bytes(b"\0")
        assert mgr.is_available("vitmatte-base-composition-1k") is True

    def test_hf_files_accepts_bin_weights(self, tmp_path: Path) -> None:
        """Existing installs with pytorch_model.bin (no safetensors) are valid."""
        mgr = ModelManager(tmp_path)
        vitmatte_dir = tmp_path / "vitmatte-base-composition-1k"
        vitmatte_dir.mkdir()
        (vitmatte_dir / "config.json").write_text("{}")
        (vitmatte_dir / "preprocessor_config.json").write_text("{}")
        (vitmatte_dir / "pytorch_model.bin").write_bytes(b"\0")
        assert mgr.is_available("vitmatte-base-composition-1k") is True

    def test_ensure_skips_existing_file(self, tmp_path: Path) -> None:
        mgr = ModelManager(tmp_path)
        target = mgr.resolve("sam2.1_hiera_large.pt")
        target.parent.mkdir(parents=True)
        target.write_bytes(b"weights")
        # Must return without any network access (would raise otherwise)
        assert mgr.ensure("sam2.1_hiera_large.pt") == target
        assert target.read_bytes() == b"weights"

    def test_scan_reports_approx_size_for_missing(self, tmp_path: Path) -> None:
        mgr = ModelManager(tmp_path)
        infos = {i.name: i for i in mgr.scan()}
        assert infos["sam2.1_hiera_large.pt"].status == "missing"
        assert infos["sam2.1_hiera_large.pt"].approx_mb == 898


class TestSetupCheck:
    def test_complete_when_weights_and_models_present(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            bootstrap.ModelManager, "missing", lambda self: []
        )
        monkeypatch.setattr(
            bootstrap,
            "_list_ollama_models",
            lambda host: ["qwen2.5vl:3b", "gemma4:e4b", "minicpm-v"],
        )
        prefs = dict(DEFAULT_PREFERENCES, models_directory=str(tmp_path))
        status = bootstrap.check_setup(prefs)
        assert status.complete is True

    def test_incomplete_when_ollama_unreachable(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(bootstrap.ModelManager, "missing", lambda self: [])
        monkeypatch.setattr(bootstrap, "_list_ollama_models", lambda host: None)
        prefs = dict(DEFAULT_PREFERENCES, models_directory=str(tmp_path))
        status = bootstrap.check_setup(prefs)
        assert status.complete is False
        kinds = {i.kind for i in status.missing_required}
        assert "backend" in kinds

    def test_missing_weights_are_required(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(
            bootstrap,
            "_list_ollama_models",
            lambda host: ["qwen2.5vl:3b"],
        )
        prefs = dict(DEFAULT_PREFERENCES, models_directory=str(tmp_path))
        status = bootstrap.check_setup(prefs)
        missing_names = {i.detail for i in status.missing_required if i.kind == "weights"}
        assert "sam2.1_hiera_large.pt" in missing_names

    def test_recommended_models_are_optional(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(bootstrap.ModelManager, "missing", lambda self: [])
        monkeypatch.setattr(
            bootstrap, "_list_ollama_models", lambda host: ["qwen2.5vl:3b"]
        )
        prefs = dict(DEFAULT_PREFERENCES, models_directory=str(tmp_path))
        status = bootstrap.check_setup(prefs)
        # gemma4/minicpm missing but only recommended -> still complete
        assert status.complete is True

    def test_llamacpp_backend_checks_server_only(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(bootstrap.ModelManager, "missing", lambda self: [])
        from models import vlm_client

        monkeypatch.setattr(
            vlm_client.LlamaCppVLMClient, "health_check", lambda self: True
        )
        prefs = dict(
            DEFAULT_PREFERENCES,
            models_directory=str(tmp_path),
            vlm_backend="llamacpp",
        )
        status = bootstrap.check_setup(prefs)
        assert status.complete is True
        assert not any(i.kind == "ollama_model" for i in status.items)
