"""Managed, offline llama.cpp inference using existing Hugging Face cache files.

One app-owned server is shared by clients. Switching models is serialized,
and only our child process is stopped. No router, downloads or global config.
"""
from __future__ import annotations

import atexit
import logging
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from models.vlm_client import LlamaCppVLMClient

logger = logging.getLogger(__name__)
LOCAL_PRIMARY = "Qwen3-VL-8B-Instruct"
LOCAL_FALLBACK = "gemma-4-12B-it"
MODEL_FILES = {
    LOCAL_PRIMARY: (
        "Qwen--Qwen3-VL-8B-Instruct-GGUF",
        "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
        "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf",
    ),
    LOCAL_FALLBACK: (
        "ggml-org--gemma-4-12B-it-GGUF",
        "gemma-4-12B-it-Q4_K_M.gguf",
        "mmproj-gemma-4-12B-it-Q8_0.gguf",
    ),
}


def resolve_local_model(model: str, cache: Path | None = None) -> tuple[Path, Path]:
    if model not in MODEL_FILES:
        raise ValueError(f"Unsupported managed model: {model}. Choose {', '.join(MODEL_FILES)}.")
    repo, weights, projector = MODEL_FILES[model]
    cache = cache or Path.home() / ".cache/huggingface/hub"
    for snapshot in sorted((cache / f"models--{repo}" / "snapshots").glob("*"), reverse=True):
        pair = snapshot / weights, snapshot / projector
        if all(path.is_file() and path.stat().st_size > 1_000_000 for path in pair):
            return pair
    raise FileNotFoundError(f"Local weights/projector missing for {model} in {cache}. No files were downloaded.")


def server_binary() -> str:
    binary = shutil.which("llama-server")
    if not binary:
        candidate = Path.home() / ".local/bin/llama-server"
        binary = str(candidate) if candidate.is_file() else None
    if not binary:
        raise FileNotFoundError("llama-server is not installed. Select an existing Ollama or llama.cpp service.")
    return binary


class LocalServer:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.process: subprocess.Popen | None = None
        self.model = ""
        self.host = ""
        self.log = None
        self.timer: threading.Timer | None = None

    def stop(self) -> None:
        with self.lock:
            if self.timer:
                self.timer.cancel()
                self.timer = None
            if self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
                self.process = None
            if self.log:
                self.log.close()
                self.log = None
            self.model = ""

    def keep_alive(self) -> None:
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(180, self.stop)
        self.timer.daemon = True
        self.timer.start()

    def start(self, model: str) -> str:
        # Caller holds lock throughout the subsequent inference request.
        if self.process and self.process.poll() is None and self.model == model:
            return self.host
        weights, projector = resolve_local_model(model)
        binary = server_binary()
        self.stop()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.host = f"http://127.0.0.1:{port}"
        # Held for the lifetime of the server process and closed by stop().
        self.log = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115
        self.process = subprocess.Popen(  # noqa: S603 — argv is fixed; binary and weights come from resolve_local_model()
            [binary, "-m", str(weights), "--mmproj", str(projector),
             "--offline", "--host", "127.0.0.1", "--port", str(port),
             "-c", "8192", "-np", "1", "--alias", model,
             "--reasoning", "off", "--reasoning-budget", "0"],
            stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT,
        )
        probe = LlamaCppVLMClient(self.host, model)
        deadline = time.monotonic() + 120
        started = False
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    self.log.seek(0)
                    detail = self.log.read().decode(errors="replace")[-3000:]
                    raise RuntimeError(f"Local {model} server exited: {detail}")
                try:
                    if probe._request_json("/health", timeout=1).get("status") == "ok":
                        self.model = model
                        logger.info("Managed local model ready: %s", model)
                        started = True
                        return self.host
                except (OSError, ValueError):
                    pass
                time.sleep(0.2)
            raise TimeoutError(f"Local {model} did not become ready within 120 seconds")
        finally:
            # A failed or cancelled readiness probe must not leave a child
            # server behind.  This also handles exceptions not anticipated by
            # the HTTP polling loop without converting them into success.
            if not started:
                self.stop()


_SERVER = LocalServer()
atexit.register(_SERVER.stop)


class ManagedVLMClient(LlamaCppVLMClient):
    backend = "local"

    def __init__(self, model: str) -> None:
        super().__init__(host="", model=model)

    def health_check(self) -> bool:
        try:
            with _SERVER.lock:
                self._host = _SERVER.start(self._model)
                _SERVER.keep_alive()
                return True
        except (
            ConnectionError,
            OSError,
            RuntimeError,
            subprocess.SubprocessError,
            TimeoutError,
            ValueError,
        ):
            logger.exception("Managed local VLM unavailable")
            return False

    def _chat(
        self,
        prompt: str,
        images_b64: list[str] | None = None,
        num_predict: int = 200,
    ) -> str:
        with _SERVER.lock:
            if _SERVER.timer:
                _SERVER.timer.cancel()
            try:
                self._host = _SERVER.start(self._model)
                return super()._chat(prompt, images_b64, num_predict)
            finally:
                _SERVER.keep_alive()
