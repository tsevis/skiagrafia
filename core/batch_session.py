"""Serializable Batch-mode context and immutable run artifacts."""
from __future__ import annotations

import datetime
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class BatchRunSettings(BaseModel):
    """The frozen user choices that affect semantic batch interpretation."""

    batch_id: str = ""
    input_folder: str = ""
    output_directory: str = ""
    selection_request: str = ""
    guide_path: str | None = None
    guide_name: str | None = None
    guide_fingerprint: str | None = None
    guide_toml: str | None = None
    interrogation_settings: dict[str, Any] = Field(default_factory=dict)
    output_settings: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""

    def model_post_init(self, __context: object) -> None:
        if not self.batch_id:
            self.batch_id = uuid.uuid4().hex[:12]
        if not self.created_at:
            self.created_at = datetime.datetime.now(datetime.UTC).isoformat()

    @property
    def run_dir(self) -> Path:
        return Path(self.output_directory) / self.batch_id


class BatchInterrogationSnapshot(BaseModel):
    """Candidates proposed for each image under a frozen BatchRunSettings."""

    batch_id: str
    selection_request: str = ""
    candidates_by_image: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    created_at: str = ""

    def model_post_init(self, __context: object) -> None:
        if not self.created_at:
            self.created_at = datetime.datetime.now(datetime.UTC).isoformat()


class BatchTriageSnapshot(BaseModel):
    """The human approval gate corresponding to one interrogation result."""

    batch_id: str
    selection_request: str = ""
    approved_labels: list[str] = Field(default_factory=list)
    created_at: str = ""

    def model_post_init(self, __context: object) -> None:
        if not self.created_at:
            self.created_at = datetime.datetime.now(datetime.UTC).isoformat()


def capture_guide(path: str | None) -> tuple[str | None, str | None, str | None]:
    """Return guide name, stable content fingerprint, and TOML snapshot."""
    if not path:
        return None, None, None
    guide_path = Path(path)
    try:
        content = guide_path.read_text(encoding="utf-8")
    except OSError:
        return guide_path.stem, None, None
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return guide_path.stem, digest, content


def write_snapshot(path: Path, model: BaseModel) -> Path:
    """Atomically write a one-time run artifact without partial JSON files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def load_run_settings(path: str | Path) -> BatchRunSettings:
    return BatchRunSettings.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_interrogation_snapshot(path: str | Path) -> BatchInterrogationSnapshot:
    return BatchInterrogationSnapshot.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def load_triage_snapshot(path: str | Path) -> BatchTriageSnapshot:
    return BatchTriageSnapshot.model_validate_json(Path(path).read_text(encoding="utf-8"))
