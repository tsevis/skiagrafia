from __future__ import annotations

import json
import logging
import sqlite3
import threading
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class JobStatus(StrEnum):
    PENDING = "pending"
    INTERROGATING = "interrogating"
    AWAITING_USER = "awaiting_user"
    MASKING = "masking"
    VECTORIZING = "vectorizing"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


class JobRecord(BaseModel):
    """Per-image job state stored in SQLite."""

    image_path: str
    status: JobStatus = JobStatus.PENDING
    labels: list[str] = []
    children: dict[str, list[str]] = {}
    error: str | None = None
    output_svg: str | None = None
    output_tiff: str | None = None
    output_all_objects_tiff: str | None = None
    # Stored rather than inferred from output files so a later resume retains
    # the original processing metrics without scanning or trusting artefacts.
    layer_count: int = 0


class StateManager:
    """SQLite-backed batch state persistence.

    Records are stored as JSON text, never as pickles: a ``state.db`` can
    arrive from a copied or shared batch folder, and reading one must not be
    able to execute anything it contains.
    """

    _SCHEMA = "CREATE TABLE IF NOT EXISTS jobs (image_id TEXT PRIMARY KEY, record TEXT NOT NULL)"

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # The batch runner drives this from whichever thread owns the run, so
        # the connection is shared under an explicit lock rather than pinned.
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        with self._lock:
            self._db.execute(self._SCHEMA)
            self._db.commit()
        logger.info("State DB opened at %s", db_path)

    @staticmethod
    def _decode(image_id: str, raw: str) -> JobRecord:
        """Parse one stored row, failing loudly on anything that is not ours."""
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"State DB record for {image_id!r} is not valid JSON."
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(  # noqa: TRY004 — callers catch ValueError for every bad record
                f"State DB record for {image_id!r} is not an object."
            )
        try:
            return JobRecord.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(
                f"State DB record for {image_id!r} is not a job record."
            ) from exc

    def _rows(self) -> list[tuple[str, str]]:
        with self._lock:
            return self._db.execute("SELECT image_id, record FROM jobs").fetchall()

    def get(self, image_id: str) -> JobRecord | None:
        with self._lock:
            row = self._db.execute(
                "SELECT record FROM jobs WHERE image_id = ?", (image_id,)
            ).fetchone()
        if row is None:
            return None
        return self._decode(image_id, row[0])

    def put(self, image_id: str, record: JobRecord) -> None:
        payload = json.dumps(record.model_dump(mode="json"))
        with self._lock:
            self._db.execute(
                "INSERT INTO jobs (image_id, record) VALUES (?, ?) "
                "ON CONFLICT(image_id) DO UPDATE SET record = excluded.record",
                (image_id, payload),
            )
            self._db.commit()

    def update_status(
        self, image_id: str, status: JobStatus, error: str | None = None
    ) -> None:
        record = self.get(image_id)
        if record is None:
            logger.warning("Cannot update status for unknown image %s", image_id)
            return
        record = record.model_copy(update={"status": status, "error": error})
        self.put(image_id, record)

    def all_records(self) -> dict[str, JobRecord]:
        return {k: self._decode(k, raw) for k, raw in self._rows()}

    def count_by_status(self) -> dict[JobStatus, int]:
        counts: dict[JobStatus, int] = dict.fromkeys(JobStatus, 0)
        for image_id, raw in self._rows():
            status = self._decode(image_id, raw).status
            counts[status] = counts.get(status, 0) + 1
        return counts

    def incomplete_ids(self) -> list[str]:
        return [
            image_id
            for image_id, raw in self._rows()
            if self._decode(image_id, raw).status
            not in (JobStatus.COMPLETE, JobStatus.SKIPPED)
        ]

    def close(self) -> None:
        self._db.close()

    @staticmethod
    def find_incomplete_batches(output_dir: Path) -> list[Path]:
        """Scan output directory for state.db files with incomplete jobs."""
        results: list[Path] = []
        if not output_dir.exists():
            return results
        for db_path in output_dir.glob("*/state.db"):
            sm: StateManager | None = None
            try:
                sm = StateManager(db_path)
                if sm.incomplete_ids():
                    results.append(db_path)
            except (OSError, sqlite3.DatabaseError, ValidationError, ValueError):
                logger.warning("Could not read %s", db_path, exc_info=True)
            finally:
                if sm is not None:
                    sm.close()
        return results
