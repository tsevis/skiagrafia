"""test_model_manager.py  --  ModelManager registry resolution and downloads.

All network access is mocked: urllib.request.urlretrieve/urlopen are never
allowed to hit the real internet. Every ModelManager is constructed over a
pytest tmp_path, never the user's real model library.
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import model_manager
from utils.model_manager import ModelInfo, ModelManager, _entry_files, _hf_dir_available


# ── _entry_files / _hf_dir_available (pure helpers) ─────────────────────────


class TestEntryFiles:
    def test_returns_list_for_present_key(self) -> None:
        entry = {"hf_files": ["a.json", "b.json"]}
        assert _entry_files(entry) == ["a.json", "b.json"]

    def test_returns_empty_for_missing_key(self) -> None:
        assert _entry_files({}) == []

    def test_returns_empty_when_value_is_not_a_list(self) -> None:
        assert _entry_files({"hf_files": "not-a-list"}) == []

    def test_reads_alternate_key(self) -> None:
        entry = {"hf_weight_alternatives": ["model.safetensors"]}
        assert _entry_files(entry, "hf_weight_alternatives") == ["model.safetensors"]


class TestHfDirAvailable:
    ENTRY = {
        "hf_files": ["config.json", "preprocessor_config.json"],
        "hf_weight_alternatives": ["model.safetensors", "pytorch_model.bin"],
    }

    def test_missing_directory_is_unavailable(self, tmp_path: Path) -> None:
        assert _hf_dir_available(self.ENTRY, tmp_path / "nope") is False

    def test_missing_required_file_is_unavailable(self, tmp_path: Path) -> None:
        target = tmp_path / "model"
        target.mkdir()
        (target / "config.json").touch()
        # preprocessor_config.json missing
        assert _hf_dir_available(self.ENTRY, target) is False

    def test_missing_weight_alternative_is_unavailable(self, tmp_path: Path) -> None:
        target = tmp_path / "model"
        target.mkdir()
        (target / "config.json").touch()
        (target / "preprocessor_config.json").touch()
        assert _hf_dir_available(self.ENTRY, target) is False

    def test_all_required_files_and_one_alternative_is_available(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "model"
        target.mkdir()
        (target / "config.json").touch()
        (target / "preprocessor_config.json").touch()
        (target / "model.safetensors").touch()
        assert _hf_dir_available(self.ENTRY, target) is True

    def test_no_alternatives_declared_means_required_files_suffice(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "model"
        target.mkdir()
        (target / "config.json").touch()
        entry = {"hf_files": ["config.json"]}
        assert _hf_dir_available(entry, target) is True


# ── ModelManager: resolve / is_available / missing ──────────────────────────


class TestResolve:
    def test_known_entry_uses_registry_subpath(self, tmp_path: Path) -> None:
        mgr = ModelManager(tmp_path)
        resolved = mgr.resolve("groundingdino_swint_ogc.pth")
        assert resolved == tmp_path / "Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"

    def test_unknown_entry_falls_back_to_bare_name(self, tmp_path: Path) -> None:
        mgr = ModelManager(tmp_path)
        assert mgr.resolve("totally-unknown-model") == tmp_path / "totally-unknown-model"

    def test_constructor_creates_models_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "fresh" / "models"
        assert not target.exists()
        ModelManager(target)
        assert target.is_dir()


class TestIsAvailableAndMissing:
    def test_file_kind_available_when_file_exists(self, tmp_path: Path) -> None:
        mgr = ModelManager(tmp_path)
        path = mgr.resolve("groundingdino_swint_ogc.pth")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        assert mgr.is_available("groundingdino_swint_ogc.pth") is True

    def test_file_kind_missing_when_absent(self, tmp_path: Path) -> None:
        mgr = ModelManager(tmp_path)
        assert mgr.is_available("groundingdino_swint_ogc.pth") is False

    def test_hf_files_kind_uses_hf_dir_available(self, tmp_path: Path) -> None:
        mgr = ModelManager(tmp_path)
        path = mgr.resolve("vitmatte-base-composition-1k")
        path.mkdir(parents=True)
        (path / "config.json").touch()
        (path / "preprocessor_config.json").touch()
        assert mgr.is_available("vitmatte-base-composition-1k") is False
        (path / "model.safetensors").touch()
        assert mgr.is_available("vitmatte-base-composition-1k") is True

    def test_unknown_name_checked_via_plain_path_existence(
        self, tmp_path: Path
    ) -> None:
        mgr = ModelManager(tmp_path)
        assert mgr.is_available("nonexistent-thing") is False

    def test_missing_lists_only_absent_entries(self, tmp_path: Path) -> None:
        mgr = ModelManager(tmp_path)
        path = mgr.resolve("groundingdino_swint_ogc.pth")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

        missing = mgr.missing()

        assert "groundingdino_swint_ogc.pth" not in missing
        assert "sam2.1_hiera_large.pt" in missing
        assert "vitmatte-base-composition-1k" in missing


# ── ModelManager.ensure ──────────────────────────────────────────────────────


class TestEnsure:
    def test_already_available_skips_download(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mgr = ModelManager(tmp_path)
        path = mgr.resolve("groundingdino_swint_ogc.pth")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

        def fail(*a: object, **k: object) -> None:
            raise AssertionError("should not attempt download")

        monkeypatch.setattr(ModelManager, "_download_file", staticmethod(fail))

        result = mgr.ensure("groundingdino_swint_ogc.pth")
        assert result == path

    def test_unknown_name_raises_key_error(self, tmp_path: Path) -> None:
        mgr = ModelManager(tmp_path)
        with pytest.raises(KeyError):
            mgr.ensure("nope-not-registered")

    def test_entry_without_url_raises_file_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_registry = {
            "manual-only": {"kind": "file", "subpath": "manual-only.bin"},
        }
        monkeypatch.setattr(model_manager, "REGISTRY", fake_registry)
        mgr = ModelManager(tmp_path)

        with pytest.raises(FileNotFoundError):
            mgr.ensure("manual-only")

    def test_file_kind_downloads_via_download_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mgr = ModelManager(tmp_path)
        captured: dict = {}

        def fake_download(url: str, path: Path, progress_callback=None) -> None:
            captured["url"] = url
            captured["path"] = path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        monkeypatch.setattr(ModelManager, "_download_file", staticmethod(fake_download))

        result = mgr.ensure("groundingdino_swint_ogc.pth")

        assert captured["url"].startswith("https://github.com")
        assert result == mgr.resolve("groundingdino_swint_ogc.pth")

    def test_hf_files_kind_downloads_via_download_hf_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mgr = ModelManager(tmp_path)
        called = {"n": 0}

        def fake_download(cls, entry, target_dir, progress_callback=None) -> None:
            called["n"] += 1

        monkeypatch.setattr(
            ModelManager, "_download_hf_files", classmethod(fake_download)
        )

        mgr.ensure("vitmatte-base-composition-1k")

        assert called["n"] == 1

    def test_github_zip_kind_downloads_via_download_github_zip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mgr = ModelManager(tmp_path)
        called = {"n": 0}

        def fake_download(entry, target_dir, progress_callback=None) -> None:
            called["n"] += 1

        monkeypatch.setattr(
            ModelManager, "_download_github_zip", staticmethod(fake_download)
        )

        mgr.ensure("grounded-sam-2-source")

        assert called["n"] == 1


# ── ModelManager._download_file ─────────────────────────────────────────────


class TestDownloadFile:
    def test_retrieves_to_tmp_then_renames_into_place(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dest = tmp_path / "sub" / "weights.bin"
        progress: list[tuple[int, int | None]] = []

        def fake_urlretrieve(url: str, filename, reporthook=None) -> None:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            Path(filename).write_bytes(b"payload")
            if reporthook is not None:
                reporthook(1, 100, 700)
                reporthook(2, 100, 700)

        monkeypatch.setattr(
            model_manager.urllib.request, "urlretrieve", fake_urlretrieve
        )

        ModelManager._download_file(
            "https://example.invalid/f.bin",
            dest,
            progress_callback=lambda done, total: progress.append((done, total)),
        )

        assert dest.read_bytes() == b"payload"
        assert not dest.with_suffix(dest.suffix + ".part").exists()
        assert progress == [(100, 700), (200, 700)]

    def test_reports_none_total_when_size_unknown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dest = tmp_path / "weights.bin"
        progress: list[tuple[int, int | None]] = []

        def fake_urlretrieve(url: str, filename, reporthook=None) -> None:
            Path(filename).write_bytes(b"x")
            if reporthook is not None:
                reporthook(1, 10, 0)

        monkeypatch.setattr(
            model_manager.urllib.request, "urlretrieve", fake_urlretrieve
        )

        ModelManager._download_file(
            "https://example.invalid/f.bin",
            dest,
            progress_callback=lambda done, total: progress.append((done, total)),
        )

        assert progress == [(10, None)]

    def test_no_progress_callback_does_not_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dest = tmp_path / "weights.bin"

        def fake_urlretrieve(url: str, filename, reporthook=None) -> None:
            Path(filename).write_bytes(b"x")
            if reporthook is not None:
                reporthook(1, 10, 100)

        monkeypatch.setattr(
            model_manager.urllib.request, "urlretrieve", fake_urlretrieve
        )

        ModelManager._download_file("https://example.invalid/f.bin", dest)
        assert dest.read_bytes() == b"x"

    def test_part_file_cleaned_up_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dest = tmp_path / "weights.bin"
        part = dest.with_suffix(dest.suffix + ".part")

        def fake_urlretrieve(url: str, filename, reporthook=None) -> None:
            Path(filename).write_bytes(b"partial")
            raise OSError("connection dropped")

        monkeypatch.setattr(
            model_manager.urllib.request, "urlretrieve", fake_urlretrieve
        )

        with pytest.raises(OSError):
            ModelManager._download_file("https://example.invalid/f.bin", dest)

        assert not part.exists()
        assert not dest.exists()


# ── ModelManager._download_hf_files ──────────────────────────────────────────


class TestDownloadHfFiles:
    ENTRY = {
        "url": "https://huggingface.co/org/repo",
        "hf_files": ["config.json", "preprocessor_config.json"],
        "hf_weight_alternatives": ["model.safetensors", "pytorch_model.bin"],
    }

    def test_downloads_missing_files_and_first_alternative(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "model"
        (target).mkdir(parents=True)
        (target / "config.json").touch()  # already present, should be skipped
        downloaded: list[str] = []

        def fake_download_file(url: str, dest: Path, progress_callback=None) -> None:
            downloaded.append(url)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.touch()

        monkeypatch.setattr(ModelManager, "_download_file", staticmethod(fake_download_file))

        ModelManager._download_hf_files(self.ENTRY, target)

        assert downloaded == [
            "https://huggingface.co/org/repo/resolve/main/preprocessor_config.json",
            "https://huggingface.co/org/repo/resolve/main/model.safetensors",
        ]

    def test_skips_alternative_when_one_already_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "model"
        target.mkdir(parents=True)
        (target / "pytorch_model.bin").touch()
        downloaded: list[str] = []

        def fake_download_file(url: str, dest: Path, progress_callback=None) -> None:
            downloaded.append(url)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.touch()

        monkeypatch.setattr(ModelManager, "_download_file", staticmethod(fake_download_file))

        ModelManager._download_hf_files(self.ENTRY, target)

        assert "https://huggingface.co/org/repo/resolve/main/pytorch_model.bin" not in downloaded
        assert "https://huggingface.co/org/repo/resolve/main/model.safetensors" not in downloaded
        assert len(downloaded) == 2  # only the two required hf_files


# ── ModelManager._download_github_zip ───────────────────────────────────────


def _fake_zip_bytes(root: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{root}/README.md", "hello")
        zf.writestr(f"{root}/sub/file.txt", "world")
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, payload: bytes, content_length: str | None) -> None:
        self._chunks = [payload[i : i + 64] for i in range(0, len(payload), 64)] or [b""]
        self.headers = {"Content-Length": content_length} if content_length else {}

    def read(self, size: int) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class TestDownloadGithubZip:
    def test_extracts_and_renames_root_to_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = _fake_zip_bytes("Grounded-SAM-2-main")
        progress: list[tuple[int, int | None]] = []

        def fake_urlopen(request, timeout=None):
            return _FakeResponse(payload, str(len(payload)))

        monkeypatch.setattr(model_manager.urllib.request, "urlopen", fake_urlopen)

        target = tmp_path / "Grounded-SAM-2"
        entry = {
            "url": "https://github.com/example/repo/archive/refs/heads/main.zip",
            "zip_root": "Grounded-SAM-2-main",
        }

        ModelManager._download_github_zip(
            entry, target, progress_callback=lambda done, total: progress.append((done, total))
        )

        assert (target / "README.md").read_text() == "hello"
        assert (target / "sub" / "file.txt").read_text() == "world"
        assert progress[-1][0] == len(payload)
        assert progress[-1][1] == len(payload)

    def test_merges_into_existing_partial_checkout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = _fake_zip_bytes("Grounded-SAM-2-main")

        def fake_urlopen(request, timeout=None):
            return _FakeResponse(payload, None)

        monkeypatch.setattr(model_manager.urllib.request, "urlopen", fake_urlopen)

        target = tmp_path / "Grounded-SAM-2"
        target.mkdir(parents=True)
        (target / "existing.txt").write_text("keep me")

        entry = {
            "url": "https://github.com/example/repo/archive/refs/heads/main.zip",
            "zip_root": "Grounded-SAM-2-main",
        }

        ModelManager._download_github_zip(entry, target)

        assert (target / "existing.txt").read_text() == "keep me"
        assert (target / "README.md").read_text() == "hello"
        assert not (target.parent / "Grounded-SAM-2-main").exists()

    def test_no_content_length_reports_none_total(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = _fake_zip_bytes("root")

        def fake_urlopen(request, timeout=None):
            return _FakeResponse(payload, None)

        monkeypatch.setattr(model_manager.urllib.request, "urlopen", fake_urlopen)
        progress: list[tuple[int, int | None]] = []

        entry = {"url": "https://x/y.zip", "zip_root": "root"}
        target = tmp_path / "dest"

        ModelManager._download_github_zip(
            entry, target, progress_callback=lambda done, total: progress.append((done, total))
        )

        assert all(total is None for _, total in progress)


# ── ModelManager.scan ─────────────────────────────────────────────────────


class TestScan:
    def test_reports_ready_file_with_size(self, tmp_path: Path) -> None:
        mgr = ModelManager(tmp_path)
        path = mgr.resolve("groundingdino_swint_ogc.pth")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"0123456789")

        results = {r.name: r for r in mgr.scan()}
        info = results["groundingdino_swint_ogc.pth"]

        assert isinstance(info, ModelInfo)
        assert info.status == "ready"
        assert info.size_bytes == 10
        assert info.manual_install is False

    def test_reports_missing_entry_with_no_size(self, tmp_path: Path) -> None:
        mgr = ModelManager(tmp_path)
        results = {r.name: r for r in mgr.scan()}
        info = results["sam2.1_hiera_large.pt"]

        assert info.status == "missing"
        assert info.size_bytes is None
        assert info.download_url is not None

    def test_directory_based_model_sums_file_sizes(self, tmp_path: Path) -> None:
        mgr = ModelManager(tmp_path)
        path = mgr.resolve("vitmatte-base-composition-1k")
        path.mkdir(parents=True)
        (path / "config.json").write_bytes(b"a" * 5)
        (path / "preprocessor_config.json").write_bytes(b"b" * 7)
        (path / "model.safetensors").write_bytes(b"c" * 100)

        results = {r.name: r for r in mgr.scan()}
        info = results["vitmatte-base-composition-1k"]

        assert info.status == "ready"
        assert info.size_bytes == 5 + 7 + 100

    def test_entry_without_url_is_manual_install(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_registry = {
            "manual-model": {"kind": "file", "subpath": "manual-model.bin"},
        }
        monkeypatch.setattr(model_manager, "REGISTRY", fake_registry)
        mgr = ModelManager(tmp_path)

        results = mgr.scan()

        assert len(results) == 1
        assert results[0].manual_install is True
        assert results[0].download_url is None


# ── Backward-compat shims: model_path / _get_default ────────────────────────


class TestBackwardCompatShims:
    @pytest.fixture(autouse=True)
    def _reset_default_manager(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(model_manager, "_default_manager", None)

    def test_get_default_uses_preferences_models_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from utils import preferences

        calls = {"n": 0}

        def fake_get_models_dir() -> Path:
            calls["n"] += 1
            return tmp_path / "models"

        monkeypatch.setattr(preferences, "get_models_dir", fake_get_models_dir)

        first = model_manager._get_default()
        second = model_manager._get_default()

        assert first is second  # singleton reused
        assert calls["n"] == 1  # only resolved once
        assert first.models_dir == tmp_path / "models"

    def test_get_default_falls_back_when_preferences_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from utils import preferences

        fallback_dir = tmp_path / "fallback-models"
        monkeypatch.setattr(model_manager, "DEFAULT_MODELS_DIR", fallback_dir)

        def broken() -> Path:
            raise RuntimeError("prefs unavailable")

        monkeypatch.setattr(preferences, "get_models_dir", broken)

        mgr = model_manager._get_default()

        assert mgr.models_dir == fallback_dir

    def test_model_path_returns_existing_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from utils import preferences

        monkeypatch.setattr(preferences, "get_models_dir", lambda: tmp_path / "models")

        mgr = model_manager._get_default()
        path = mgr.resolve("groundingdino_swint_ogc.pth")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

        assert model_manager.model_path("groundingdino_swint_ogc.pth") == path

    def test_model_path_raises_when_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from utils import preferences

        monkeypatch.setattr(preferences, "get_models_dir", lambda: tmp_path / "models")

        with pytest.raises(FileNotFoundError):
            model_manager.model_path("groundingdino_swint_ogc.pth")
