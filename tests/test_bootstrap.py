"""test_bootstrap.py  --  First-run setup detection and bootstrap primitives.

All Ollama / llama.cpp / HTTP access is mocked. ModelManager always operates
over a pytest tmp_path via a monkeypatched get_models_dir, never the user's
real model library.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import bootstrap
from utils.model_manager import ModelManager


# ── _list_ollama_models ──────────────────────────────────────────────────────


class TestListOllamaModels:
    def test_returns_model_names_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import ollama

        class FakeClient:
            def __init__(self, host: str) -> None:
                self.host = host

            def list(self):
                return SimpleNamespace(
                    models=[SimpleNamespace(model="qwen2.5vl:3b"), SimpleNamespace(model="gemma4:e4b")]
                )

        monkeypatch.setattr(ollama, "Client", FakeClient)

        result = bootstrap._list_ollama_models("http://localhost:11434")

        assert result == ["qwen2.5vl:3b", "gemma4:e4b"]

    def test_none_model_name_becomes_empty_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import ollama

        class FakeClient:
            def __init__(self, host: str) -> None:
                pass

            def list(self):
                return SimpleNamespace(models=[SimpleNamespace(model=None)])

        monkeypatch.setattr(ollama, "Client", FakeClient)

        assert bootstrap._list_ollama_models("http://localhost:11434") == [""]

    def test_unreachable_server_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import ollama

        class FakeClient:
            def __init__(self, host: str) -> None:
                raise ConnectionError("no server")

        monkeypatch.setattr(ollama, "Client", FakeClient)

        assert bootstrap._list_ollama_models("http://localhost:11434") is None


# ── check_setup ──────────────────────────────────────────────────────────────


def _patch_models_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(bootstrap, "get_models_dir", lambda prefs: tmp_path / "models")


class TestCheckSetupOllamaBackend:
    def test_all_missing_when_nothing_on_disk_or_server(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_models_dir(monkeypatch, tmp_path)
        monkeypatch.setattr(bootstrap, "_list_ollama_models", lambda host: None)

        status = bootstrap.check_setup({"vlm_backend": "ollama"})

        assert status.complete is False
        weight_items = [i for i in status.items if i.kind == "weights"]
        assert len(weight_items) == 4
        assert all(i.status == "missing" for i in weight_items)
        backend_item = next(i for i in status.items if i.kind == "backend")
        assert backend_item.status == "missing"

    def test_complete_when_weights_present_and_models_available(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_models_dir(monkeypatch, tmp_path)
        mgr = ModelManager(tmp_path / "models")

        # Materialize every registry entry so nothing is missing.
        from utils.model_manager import REGISTRY

        for name, entry in REGISTRY.items():
            path = mgr.resolve(name)
            if entry.get("kind") == "hf_files":
                path.mkdir(parents=True, exist_ok=True)
                for f in entry.get("hf_files", []):
                    (path / f).touch()
                alternatives = entry.get("hf_weight_alternatives", [])
                if alternatives:
                    (path / alternatives[0]).touch()
            elif entry.get("kind") == "github_zip":
                # Shares a parent directory with nested "file" entries below
                # it (e.g. Grounded-SAM-2/gdino_checkpoints/...), so it must
                # itself be a directory rather than a plain file.
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

        monkeypatch.setattr(
            bootstrap,
            "_list_ollama_models",
            lambda host: ["qwen2.5vl:3b", "gemma4:e4b"],
        )

        status = bootstrap.check_setup(
            {
                "vlm_backend": "ollama",
                "ollama_model": "qwen2.5vl:3b",
                "preferred_fallback_vlm": "gemma4:e4b",
                "preferred_text_reasoner": "gemma4:e4b",
            }
        )

        assert status.complete is True
        assert status.missing_required == []

    def test_recommended_models_are_not_required(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_models_dir(monkeypatch, tmp_path)
        monkeypatch.setattr(
            bootstrap, "_list_ollama_models", lambda host: ["qwen2.5vl:3b"]
        )

        status = bootstrap.check_setup(
            {"vlm_backend": "ollama", "ollama_model": "qwen2.5vl:3b"}
        )

        recommended = [i for i in status.items if not i.required and i.kind == "ollama_model"]
        assert recommended  # RECOMMENDED_OLLAMA_MODELS entries present
        assert all(i.status == "missing" for i in recommended)
        # Missing recommended models must not block completeness on their own.
        assert all(i not in status.missing_required for i in recommended)

    def test_default_host_used_when_prefs_omit_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_models_dir(monkeypatch, tmp_path)
        captured = {}

        def fake_list(host: str):
            captured["host"] = host
            return None

        monkeypatch.setattr(bootstrap, "_list_ollama_models", fake_list)

        bootstrap.check_setup({})

        assert captured["host"] == "http://localhost:11434"


class TestCheckSetupLlamaCppBackend:
    def test_ready_when_health_check_true(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_models_dir(monkeypatch, tmp_path)
        from models.vlm_client import LlamaCppVLMClient

        monkeypatch.setattr(LlamaCppVLMClient, "health_check", lambda self: True)

        status = bootstrap.check_setup(
            {"vlm_backend": "llamacpp", "llamacpp_url": "http://localhost:8080"}
        )

        backend_item = next(i for i in status.items if i.kind == "backend")
        assert backend_item.status == "ready"
        assert "8080" in backend_item.name

    def test_missing_when_health_check_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_models_dir(monkeypatch, tmp_path)
        from models.vlm_client import LlamaCppVLMClient

        monkeypatch.setattr(LlamaCppVLMClient, "health_check", lambda self: False)

        status = bootstrap.check_setup({"vlm_backend": "llamacpp"})

        backend_item = next(i for i in status.items if i.kind == "backend")
        assert backend_item.status == "missing"
        assert status.complete is False

    def test_llamacpp_backend_does_not_query_ollama_models(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_models_dir(monkeypatch, tmp_path)
        from models.vlm_client import LlamaCppVLMClient

        monkeypatch.setattr(LlamaCppVLMClient, "health_check", lambda self: True)

        def fail(host: str):
            raise AssertionError("should not query ollama for llamacpp backend")

        monkeypatch.setattr(bootstrap, "_list_ollama_models", fail)

        status = bootstrap.check_setup({"vlm_backend": "llamacpp"})
        assert any(i.kind == "backend" for i in status.items)


# ── is_setup_complete ─────────────────────────────────────────────────────


class TestIsSetupComplete:
    def test_true_when_check_setup_reports_complete(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            bootstrap,
            "check_setup",
            lambda prefs: bootstrap.SetupStatus(items=[]),
        )
        assert bootstrap.is_setup_complete({}) is True

    def test_false_when_required_item_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            bootstrap,
            "check_setup",
            lambda prefs: bootstrap.SetupStatus(
                items=[
                    bootstrap.SetupItem(
                        name="x", kind="weights", status="missing", required=True
                    )
                ]
            ),
        )
        assert bootstrap.is_setup_complete({}) is False

    def test_never_blocks_app_on_broken_check(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def broken(prefs):
            raise RuntimeError("boom")

        monkeypatch.setattr(bootstrap, "check_setup", broken)

        assert bootstrap.is_setup_complete({}) is True


# ── pull_ollama_model ─────────────────────────────────────────────────────


class TestPullOllamaModel:
    def test_streams_progress_updates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import ollama

        updates = [
            SimpleNamespace(status="downloading", completed=10, total=100),
            SimpleNamespace(status="verifying", completed=100, total=100),
        ]

        class FakeClient:
            def __init__(self, host: str) -> None:
                pass

            def pull(self, model: str, stream: bool = True):
                assert stream is True
                return iter(updates)

        monkeypatch.setattr(ollama, "Client", FakeClient)
        seen: list[tuple[str, int, int | None]] = []

        bootstrap.pull_ollama_model(
            "http://localhost:11434",
            "qwen2.5vl:3b",
            progress_callback=lambda status, done, total: seen.append((status, done, total)),
        )

        assert seen == [("downloading", 10, 100), ("verifying", 100, 100)]

    def test_missing_fields_default_gracefully(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import ollama

        class FakeClient:
            def __init__(self, host: str) -> None:
                pass

            def pull(self, model: str, stream: bool = True):
                return iter([SimpleNamespace()])

        monkeypatch.setattr(ollama, "Client", FakeClient)
        seen: list[tuple[str, int, int | None]] = []

        bootstrap.pull_ollama_model(
            "http://localhost:11434",
            "m",
            progress_callback=lambda status, done, total: seen.append((status, done, total)),
        )

        assert seen == [("", 0, None)]

    def test_works_without_progress_callback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import ollama

        class FakeClient:
            def __init__(self, host: str) -> None:
                pass

            def pull(self, model: str, stream: bool = True):
                return iter([SimpleNamespace(status="ok", completed=1, total=1)])

        monkeypatch.setattr(ollama, "Client", FakeClient)

        bootstrap.pull_ollama_model("http://localhost:11434", "m")


# ── download_missing_weights ─────────────────────────────────────────────


class TestDownloadMissingWeights:
    def test_reports_failures_and_forwards_progress(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_models_dir(monkeypatch, tmp_path)

        class FakeManager:
            def __init__(self, models_dir: Path) -> None:
                self.models_dir = models_dir

            def missing(self) -> list[str]:
                return ["good-model", "bad-model"]

            def ensure(self, name: str, progress_callback=None) -> Path:
                if name == "bad-model":
                    raise RuntimeError("download failed")
                if progress_callback is not None:
                    progress_callback(5, 10)
                return Path("/fake/path")

        monkeypatch.setattr(bootstrap, "ModelManager", FakeManager)
        seen: list[tuple[str, int, int | None]] = []

        failures = bootstrap.download_missing_weights(
            {},
            progress_callback=lambda name, done, total: seen.append((name, done, total)),
        )

        assert failures == ["bad-model"]
        assert seen == [("good-model", 5, 10)]

    def test_returns_empty_list_when_nothing_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_models_dir(monkeypatch, tmp_path)

        class FakeManager:
            def __init__(self, models_dir: Path) -> None:
                pass

            def missing(self) -> list[str]:
                return []

        monkeypatch.setattr(bootstrap, "ModelManager", FakeManager)

        assert bootstrap.download_missing_weights({}) == []


# ── SetupStatus / SetupItem models ──────────────────────────────────────────


class TestSetupStatusModel:
    def test_missing_required_excludes_optional_items(self) -> None:
        status = bootstrap.SetupStatus(
            items=[
                bootstrap.SetupItem(name="a", kind="weights", status="missing", required=True),
                bootstrap.SetupItem(name="b", kind="weights", status="missing", required=False),
                bootstrap.SetupItem(name="c", kind="weights", status="ready", required=True),
            ]
        )

        assert [i.name for i in status.missing_required] == ["a"]
        assert status.complete is False

    def test_complete_when_no_missing_required(self) -> None:
        status = bootstrap.SetupStatus(
            items=[
                bootstrap.SetupItem(name="a", kind="weights", status="ready", required=True),
                bootstrap.SetupItem(name="b", kind="weights", status="missing", required=False),
            ]
        )

        assert status.complete is True
