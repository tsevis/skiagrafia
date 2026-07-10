"""test_vlm_client.py  --  VLM backend clients (Ollama + llama.cpp).

All tests are offline: transports are mocked, no server required.
"""
from __future__ import annotations

import sys
import urllib.error
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.vlm_client import (
    BACKEND_LLAMACPP,
    BACKEND_OLLAMA,
    BaseVLMClient,
    LlamaCppVLMClient,
    OllamaVLMClient,
    _dedupe,
    _split_children_response,
    create_vlm_client,
)


# ── Shared parsing logic ────────────────────────────────────────────────────


class TestLabelParsing:
    def test_dedupe_rejects_garbage(self) -> None:
        labels = ["12.5", "a", "cd/mov/x", "keyboard", "  'mouse'  "]
        assert _dedupe(labels, 8) == ["keyboard", "mouse"]

    def test_dedupe_drops_word_subsets(self) -> None:
        labels = ["monitor", "crt monitor", "keyboard"]
        assert _dedupe(labels, 8) == ["crt monitor", "keyboard"]

    def test_dedupe_caps_at_limit(self) -> None:
        labels = [f"object {chr(97 + i)}" for i in range(10)]
        assert len(_dedupe(labels, 4)) == 4

    def test_split_children_strips_numbering(self) -> None:
        response = "1] screen, 2. stand\n3) bezel; 4- cable"
        assert _split_children_response(response) == [
            "screen", "stand", "bezel", "cable",
        ]


# ── Fake transport for BaseVLMClient behavior ───────────────────────────────


class _FakeVLM(BaseVLMClient):
    """Canned-response client to exercise shared interrogation logic."""

    backend = "fake"

    def __init__(self, responses: list[str]) -> None:
        super().__init__(host="http://fake", model="fake-model")
        self._responses = list(responses)
        self.prompts: list[str] = []

    def _chat(self, prompt, images_b64=None, num_predict=200):  # type: ignore[override]
        self.prompts.append(prompt)
        return self._responses.pop(0) if self._responses else ""

    def health_check(self) -> bool:
        return True


def _image() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.uint8)


class TestBaseClientLogic:
    def test_get_parents_first_prompt_success(self) -> None:
        client = _FakeVLM(["CRT monitor, keyboard, computer mouse"])
        assert client.get_parents(_image()) == [
            "crt monitor", "keyboard", "computer mouse",
        ]
        assert len(client.prompts) == 1

    def test_get_parents_falls_back_on_garbage(self) -> None:
        client = _FakeVLM(["1.2, 3/4", "desk lamp, mug"])
        assert client.get_parents(_image()) == ["desk lamp", "mug"]
        assert len(client.prompts) == 2

    def test_interrogate_adds_user_labels(self) -> None:
        # parents response, then children responses (2 prompts per parent max)
        client = _FakeVLM(["keyboard", "keycaps, cable", "", "", ""])
        results = client.interrogate(_image(), confirmed_labels=["keyboard", "vase"])
        parents = [r.label for r in results if r.role == "parent"]
        assert "keyboard" in parents
        assert "vase" in parents  # user-added even though not detected


# ── llama.cpp client ────────────────────────────────────────────────────────


class TestLlamaCppClient:
    def test_chat_builds_openai_payload_with_image(self) -> None:
        client = LlamaCppVLMClient(host="http://localhost:8080", model="qwen3-vl")
        captured: dict = {}

        def fake_request(path, payload=None, timeout=None):
            captured["path"] = path
            captured["payload"] = payload
            return {"choices": [{"message": {"content": "  mug, plate  "}}]}

        client._request_json = fake_request  # type: ignore[method-assign]
        result = client.query_vision(_image(), "List objects")

        assert result == "mug, plate"
        assert captured["path"] == "/v1/chat/completions"
        payload = captured["payload"]
        assert payload["model"] == "qwen3-vl"
        content = payload["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": "List objects"}
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_chat_text_only_uses_plain_content(self) -> None:
        client = LlamaCppVLMClient()
        captured: dict = {}

        def fake_request(path, payload=None, timeout=None):
            captured["payload"] = payload
            return {"choices": [{"message": {"content": "ok"}}]}

        client._request_json = fake_request  # type: ignore[method-assign]
        assert client.query_text("rank these") == "ok"
        assert captured["payload"]["messages"][0]["content"] == "rank these"

    def test_chat_raises_on_missing_choices(self) -> None:
        client = LlamaCppVLMClient()
        client._request_json = lambda *a, **k: {"choices": []}  # type: ignore[method-assign]
        with pytest.raises(ValueError):
            client.query_text("hello")

    def test_chat_raises_on_non_text_content(self) -> None:
        client = LlamaCppVLMClient()
        client._request_json = (  # type: ignore[method-assign]
            lambda *a, **k: {"choices": [{"message": {"content": None}}]}
        )
        with pytest.raises(ValueError):
            client.query_text("hello")

    def test_health_check_ok(self) -> None:
        client = LlamaCppVLMClient()

        def fake_request(path, payload=None, timeout=None):
            if path == "/health":
                return {"status": "ok"}
            return {"data": [{"id": "qwen3-vl-8b"}]}

        client._request_json = fake_request  # type: ignore[method-assign]
        assert client.health_check() is True

    def test_health_check_false_while_loading(self) -> None:
        client = LlamaCppVLMClient()

        def fake_request(path, payload=None, timeout=None):
            raise urllib.error.HTTPError(path, 503, "loading", {}, None)  # type: ignore[arg-type]

        client._request_json = fake_request  # type: ignore[method-assign]
        assert client.health_check() is False

    def test_health_check_false_when_down(self) -> None:
        client = LlamaCppVLMClient(host="http://localhost:1")

        def fake_request(path, payload=None, timeout=None):
            raise urllib.error.URLError("connection refused")

        client._request_json = fake_request  # type: ignore[method-assign]
        assert client.health_check() is False

    def test_loaded_model(self) -> None:
        client = LlamaCppVLMClient()
        client._request_json = (  # type: ignore[method-assign]
            lambda *a, **k: {"data": [{"id": "gemma-4-12b"}]}
        )
        assert client.loaded_model() == "gemma-4-12b"


# ── Factory function + backward compatibility ───────────────────────────────


class TestCreateClient:
    def test_ollama_backend(self) -> None:
        client = create_vlm_client(BACKEND_OLLAMA, "http://localhost:11434", "qwen2.5vl:3b")
        assert isinstance(client, OllamaVLMClient)
        assert client.model == "qwen2.5vl:3b"

    def test_llamacpp_backend(self) -> None:
        client = create_vlm_client(BACKEND_LLAMACPP, "http://localhost:8080", "any")
        assert isinstance(client, LlamaCppVLMClient)

    def test_unknown_backend_falls_back_to_ollama(self) -> None:
        client = create_vlm_client("nonsense", "http://localhost:11434", "m")
        assert isinstance(client, OllamaVLMClient)

    def test_moondream_client_alias_still_importable(self) -> None:
        from models.moondream_client import MoondreamClient

        assert MoondreamClient is OllamaVLMClient
