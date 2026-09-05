"""test_preferences.py  --  Preferences load/save, migration, model dir resolution.

Every test redirects the config directory to a tmp_path via monkeypatch --
the user's real ~/.config/skiagrafia is never touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import preferences


@pytest.fixture(autouse=True)
def _isolated_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect _CONFIG_DIR so no test can touch the real user config."""
    config_dir = tmp_path / "config" / "skiagrafia"
    monkeypatch.setattr(preferences, "_CONFIG_DIR", config_dir)
    return config_dir


class TestPrefsPath:
    def test_prefs_path_is_under_config_dir(self, _isolated_config_dir: Path) -> None:
        assert preferences._prefs_path() == _isolated_config_dir / "preferences.json"


class TestLoadPreferencesFirstRun:
    def test_creates_defaults_file_when_absent(self, _isolated_config_dir: Path) -> None:
        assert not _isolated_config_dir.exists()

        prefs = preferences.load_preferences()

        assert prefs == preferences.DEFAULT_PREFERENCES
        saved_path = _isolated_config_dir / "preferences.json"
        assert saved_path.is_file()
        assert json.loads(saved_path.read_text()) == preferences.DEFAULT_PREFERENCES


class TestLoadPreferencesExistingFile:
    def test_saved_values_override_defaults(self, _isolated_config_dir: Path) -> None:
        _isolated_config_dir.mkdir(parents=True)
        saved = {"theme": "dark", "mask_overlay_opacity": 55}
        (_isolated_config_dir / "preferences.json").write_text(json.dumps(saved))

        prefs = preferences.load_preferences()

        assert prefs["theme"] == "dark"
        assert prefs["mask_overlay_opacity"] == 55
        # Untouched defaults still present.
        assert prefs["default_mode"] == preferences.DEFAULT_PREFERENCES["default_mode"]

    def test_corrupt_json_falls_back_to_defaults(
        self, _isolated_config_dir: Path
    ) -> None:
        _isolated_config_dir.mkdir(parents=True)
        (_isolated_config_dir / "preferences.json").write_text("{not valid json")

        prefs = preferences.load_preferences()

        assert prefs == preferences.DEFAULT_PREFERENCES

    def test_unreadable_file_falls_back_to_defaults(
        self, _isolated_config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolated_config_dir.mkdir(parents=True)
        prefs_path = _isolated_config_dir / "preferences.json"
        prefs_path.write_text("{}")

        real_read_text = Path.read_text

        def flaky_read_text(self: Path, *args: object, **kwargs: object) -> str:
            if self == prefs_path:
                raise OSError("disk error")
            return real_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", flaky_read_text)

        prefs = preferences.load_preferences()

        assert prefs == preferences.DEFAULT_PREFERENCES


class TestLoadPreferencesSaveFailures:
    def test_first_run_save_failure_still_returns_defaults(
        self, _isolated_config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken_save(prefs: dict) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(preferences, "save_preferences", broken_save)

        prefs = preferences.load_preferences()

        assert prefs == preferences.DEFAULT_PREFERENCES
        assert not _isolated_config_dir.exists()

    def test_migration_persist_failure_still_returns_migrated_values(
        self, _isolated_config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolated_config_dir.mkdir(parents=True)
        prefs_path = _isolated_config_dir / "preferences.json"
        prefs_path.write_text(json.dumps({"ollama_model": "moondream"}))

        def broken_save(prefs: dict) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(preferences, "save_preferences", broken_save)

        prefs = preferences.load_preferences()

        assert prefs["ollama_model"] == "qwen2.5vl:3b"
        # The on-disk file is left as originally saved since persisting
        # the migration failed.
        assert json.loads(prefs_path.read_text())["ollama_model"] == "moondream"


class TestLegacyMigration:
    def test_migrates_known_legacy_default(self, _isolated_config_dir: Path) -> None:
        _isolated_config_dir.mkdir(parents=True)
        saved = {"ollama_model": "moondream"}
        prefs_path = _isolated_config_dir / "preferences.json"
        prefs_path.write_text(json.dumps(saved))

        prefs = preferences.load_preferences()

        assert prefs["ollama_model"] == "qwen2.5vl:3b"
        # Migration is persisted so it doesn't re-run on the next load.
        assert json.loads(prefs_path.read_text())["ollama_model"] == "qwen2.5vl:3b"

    def test_deliberate_custom_value_is_not_touched(self) -> None:
        saved = {"ollama_model": "my-custom-model"}

        migrated = preferences._migrate_legacy_defaults(saved)

        assert migrated["ollama_model"] == "my-custom-model"

    def test_migrates_fallback_and_reasoner_keys(self) -> None:
        saved = {
            "preferred_fallback_vlm": "minicpm-v",
            "preferred_text_reasoner": "qwen3.5",
        }

        migrated = preferences._migrate_legacy_defaults(saved)

        assert migrated["preferred_fallback_vlm"] == "gemma4:e4b"
        assert migrated["preferred_text_reasoner"] == "gemma4:e4b"

    def test_non_string_value_is_left_untouched(self) -> None:
        saved = {"ollama_model": 123}

        migrated = preferences._migrate_legacy_defaults(saved)

        assert migrated["ollama_model"] == 123

    def test_returns_new_dict_not_mutating_input(self) -> None:
        saved = {"ollama_model": "moondream"}

        migrated = preferences._migrate_legacy_defaults(saved)

        assert saved["ollama_model"] == "moondream"
        assert migrated is not saved


class TestSavePreferences:
    def test_writes_json_and_creates_parent_dirs(
        self, _isolated_config_dir: Path
    ) -> None:
        assert not _isolated_config_dir.exists()

        preferences.save_preferences({"a": 1, "b": "two"})

        saved_path = _isolated_config_dir / "preferences.json"
        assert json.loads(saved_path.read_text()) == {"a": 1, "b": "two"}

    def test_overwrites_previous_contents(self, _isolated_config_dir: Path) -> None:
        preferences.save_preferences({"a": 1})
        preferences.save_preferences({"a": 2})

        saved_path = _isolated_config_dir / "preferences.json"
        assert json.loads(saved_path.read_text()) == {"a": 2}


class TestGetModelsDir:
    def test_custom_directory_is_used_when_set(self) -> None:
        prefs = {"models_directory": "/custom/models"}

        assert preferences.get_models_dir(prefs) == Path("/custom/models")

    def test_whitespace_only_custom_directory_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fallback = tmp_path / "default-models"
        monkeypatch.setattr(preferences, "DEFAULT_MODELS_DIR", fallback)
        prefs = {"models_directory": "   "}

        assert preferences.get_models_dir(prefs) == fallback

    def test_missing_key_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fallback = tmp_path / "default-models"
        monkeypatch.setattr(preferences, "DEFAULT_MODELS_DIR", fallback)

        assert preferences.get_models_dir({}) == fallback

    def test_none_prefs_loads_preferences_first(
        self, _isolated_config_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fallback = tmp_path / "default-models"
        monkeypatch.setattr(preferences, "DEFAULT_MODELS_DIR", fallback)

        assert preferences.get_models_dir(None) == fallback
        assert (_isolated_config_dir / "preferences.json").is_file()

    def test_custom_directory_strips_surrounding_whitespace(self) -> None:
        prefs = {"models_directory": "  /custom/models  "}

        assert preferences.get_models_dir(prefs) == Path("/custom/models")


class TestDefaultModelsDir:
    def test_uses_shared_dir_when_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        shared = tmp_path / "shared-models"
        shared.mkdir()
        monkeypatch.setattr(preferences, "_SHARED_MODELS_DIR", shared)

        assert preferences._default_models_dir() == shared

    def test_darwin_fallback_when_shared_dir_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(preferences, "_SHARED_MODELS_DIR", tmp_path / "nope")
        monkeypatch.setattr(preferences.sys, "platform", "darwin")
        fake_home = tmp_path / "home"
        monkeypatch.setattr(preferences.Path, "home", classmethod(lambda cls: fake_home))

        result = preferences._default_models_dir()

        assert result == fake_home / "Library" / "Application Support" / "skiagrafia" / "models"

    def test_linux_fallback_when_shared_dir_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(preferences, "_SHARED_MODELS_DIR", tmp_path / "nope")
        monkeypatch.setattr(preferences.sys, "platform", "linux")
        fake_home = tmp_path / "home"
        monkeypatch.setattr(preferences.Path, "home", classmethod(lambda cls: fake_home))

        result = preferences._default_models_dir()

        assert result == fake_home / ".local" / "share" / "skiagrafia" / "models"
