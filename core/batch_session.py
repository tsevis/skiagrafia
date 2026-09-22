"""Serializable Batch-mode context and immutable run artifacts."""
from __future__ import annotations

import datetime
import hashlib
import re
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from utils.security import SecurityError, atomic_write_bytes, safe_child_path

_BATCH_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


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
        elif not _BATCH_ID_RE.fullmatch(self.batch_id):
            raise ValueError("Batch ID must use only letters, numbers, underscores, or hyphens.")
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
    # A label can be appropriate for the batch while still being a false
    # positive in one source image.  Keep those human-reviewed exceptions
    # alongside the global approval rather than forcing the user to reject
    # the label everywhere.
    excluded_labels_by_image: dict[str, list[str]] = Field(default_factory=dict)
    created_at: str = ""

    def model_post_init(self, __context: object) -> None:
        if not self.created_at:
            self.created_at = datetime.datetime.now(datetime.UTC).isoformat()


class BatchProcessingSnapshot(BaseModel):
    """The exact effective BatchConfig used after human Triage approval."""

    batch_id: str
    config: dict[str, Any]
    created_at: str = ""

    def model_post_init(self, __context: object) -> None:
        if not self.created_at:
            self.created_at = datetime.datetime.now(datetime.UTC).isoformat()


@dataclass(frozen=True)
class ResumableBatch:
    """A fully verified GUI batch that may safely continue with BatchRunner."""

    run_settings: BatchRunSettings
    interrogation: BatchInterrogationSnapshot
    triage: BatchTriageSnapshot
    processing: BatchProcessingSnapshot
    state_db_path: Path


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
    atomic_write_bytes(path, model.model_dump_json(indent=2).encode("utf-8"))
    return path


def materialize_frozen_guide(settings: BatchRunSettings) -> Path | None:
    """Store the captured guide once inside the run directory without overwriting.

    A resume must never silently pick up a guide edited after interrogation.
    The run-local copy is also what the worker receives for the processing pass.
    """
    if settings.guide_toml is None:
        return None
    digest = hashlib.sha256(settings.guide_toml.encode("utf-8")).hexdigest()
    if settings.guide_fingerprint and digest != settings.guide_fingerprint:
        raise ValueError("The captured guide does not match its fingerprint.")

    target = safe_child_path(settings.run_dir, "guide.toml")
    if target.exists():
        try:
            existing = target.read_text(encoding="utf-8")
        except OSError as exc:
            raise OSError("Could not read the existing frozen guide.") from exc
        if existing != settings.guide_toml:
            raise FileExistsError("The run directory already contains a different guide.toml.")
        return target

    atomic_write_bytes(target, settings.guide_toml.encode("utf-8"))
    return target


def write_processing_snapshot(
    path: Path, snapshot: BatchProcessingSnapshot
) -> Path:
    """Write a processing manifest once, or prove an existing one is identical."""
    target = safe_child_path(path.parent, path.name)
    if target.exists():
        existing = load_processing_snapshot(target)
        if existing.batch_id != snapshot.batch_id or existing.config != snapshot.config:
            raise FileExistsError("The run already has a different processing manifest.")
        return target
    return write_snapshot(target, snapshot)


def load_run_settings(path: str | Path) -> BatchRunSettings:
    return BatchRunSettings.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_interrogation_snapshot(path: str | Path) -> BatchInterrogationSnapshot:
    return BatchInterrogationSnapshot.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def load_triage_snapshot(path: str | Path) -> BatchTriageSnapshot:
    return BatchTriageSnapshot.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_processing_snapshot(path: str | Path) -> BatchProcessingSnapshot:
    return BatchProcessingSnapshot.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def partition_by_labelled(
    labels_by_image: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    """Split images into those triage left with labels, and those it did not.

    Triage intersects the approved labels with each image's own candidates,
    so an image can come out of it with an empty list. Freezing that list
    into a run made the image fail later, one worker and one model load
    after the fact, for something knowable before the run began.
    """
    labelled = [path for path, labels in labels_by_image.items() if labels]
    bare = [path for path, labels in labels_by_image.items() if not labels]
    return labelled, bare


def triage_labels_for_image(
    candidates: list[dict[str, Any]],
    approved_labels: list[str],
    excluded_labels: list[str],
) -> tuple[list[str], dict[str, str]]:
    """Resolve approved candidates for one source image deterministically."""
    approved = {str(label).casefold() for label in approved_labels}
    excluded = {str(label).casefold() for label in excluded_labels}
    labels: list[str] = []
    selections: dict[str, str] = {}
    seen: set[str] = set()
    for item in candidates:
        canonical = str(item.get("canonical_label") or item.get("label") or "")
        key = canonical.casefold()
        if not canonical or key not in approved or key in excluded or key in seen:
            continue
        labels.append(canonical)
        selections[canonical] = str(item.get("selection", "all"))
        seen.add(key)
    return labels, selections


def load_resumable_batch(run_dir: str | Path) -> ResumableBatch | None:
    """Return a resume context only when every persisted decision agrees.

    A bare or corrupt ``state.db`` is intentionally not resumable through the
    GUI.  This boundary prevents a resume from bypassing Triage or applying
    changed input files, guides, or requests to an old state database.
    """
    directory = Path(run_dir)
    try:
        settings = load_run_settings(directory / "run.json")
        interrogation = load_interrogation_snapshot(directory / "interrogation.json")
        triage = load_triage_snapshot(directory / "triage.json")
        processing = load_processing_snapshot(directory / "processing.json")
        if {
            settings.batch_id,
            interrogation.batch_id,
            triage.batch_id,
            processing.batch_id,
        } != {directory.name}:
            return None
        if (
            interrogation.selection_request != settings.selection_request
            or triage.selection_request != settings.selection_request
        ):
            return None

        # Avoid an import cycle at module load time while still applying the
        # complete BatchConfig schema to an untrusted on-disk manifest.
        from core.batch_runner import BatchConfig
        from core.state_manager import JobStatus, StateManager

        config = BatchConfig.model_validate(processing.config)
        if (
            config.batch_id != settings.batch_id
            or Path(config.input_folder).resolve() != Path(settings.input_folder).resolve()
            or Path(config.output_dir).resolve() != Path(settings.output_directory).resolve()
            or config.confirmed_labels != triage.approved_labels
        ):
            return None
        input_images = list(config.input_images or [])
        if not input_images or len(input_images) != len(set(input_images)):
            return None
        if set(input_images) != set(interrogation.candidates_by_image):
            return None
        input_root = Path(config.input_folder).resolve()
        if not input_root.is_dir() or any(
            not Path(image_path).is_file()
            or Path(image_path).resolve().parent != input_root
            for image_path in input_images
        ):
            return None

        labels_by_image: dict[str, list[str]] = {}
        selections_by_image: dict[str, dict[str, str]] = {}
        for image_path in input_images:
            labels, selections = triage_labels_for_image(
                interrogation.candidates_by_image[image_path],
                triage.approved_labels,
                triage.excluded_labels_by_image.get(image_path, []),
            )
            labels_by_image[image_path] = labels
            selections_by_image[image_path] = selections
        if (
            config.labels_by_image != labels_by_image
            or config.selections_by_image != selections_by_image
        ):
            return None

        if settings.guide_toml is not None:
            if not config.guide_path or not Path(config.guide_path).is_file():
                return None
            frozen_guide = Path(config.guide_path).read_text(encoding="utf-8")
            digest = hashlib.sha256(settings.guide_toml.encode("utf-8")).hexdigest()
            if (
                frozen_guide != settings.guide_toml
                or (
                    settings.guide_fingerprint is not None
                    and digest != settings.guide_fingerprint
                )
            ):
                return None
        elif config.guide_path is not None:
            # Older runs without a captured copy cannot prove guide parity.
            return None

        state_db_path = directory / "state.db"
        # This validator is also used while populating the Recent Batches
        # list.  Do not let StateManager create an empty DB merely because a
        # saved, not-yet-started run happens to have a processing manifest.
        if not state_db_path.is_file() or state_db_path.is_symlink():
            return None
        state = StateManager(state_db_path)
        try:
            records = state.all_records()
            record_paths = [record.image_path for record in records.values()]
            if (
                not state.incomplete_ids()
                or len(record_paths) != len(input_images)
                or set(record_paths) != set(input_images)
            ):
                return None
            for record in records.values():
                if record.status != JobStatus.COMPLETE:
                    continue
                output = record.output_all_objects_tiff
                if not output or not Path(output).is_file() or Path(output).is_symlink():
                    return None
                if record.output_svg and (
                    not Path(record.output_svg).is_file()
                    or Path(record.output_svg).is_symlink()
                ):
                    return None
        finally:
            state.close()
    except (OSError, UnicodeError, ValueError, sqlite3.DatabaseError, SecurityError):
        return None

    return ResumableBatch(
        run_settings=settings,
        interrogation=interrogation,
        triage=triage,
        processing=processing,
        state_db_path=state_db_path,
    )
