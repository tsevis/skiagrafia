from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from sqlitedict import SqliteDict

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


class StateManager:
    """SQLiteDict-backed batch state persistence."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = SqliteDict(str(db_path), autocommit=True)
        logger.info("State DB opened at %s", db_path)

    def get(self, image_id: str) -> JobRecord | None:
        raw = self._db.get(image_id)
        if raw is None:
            return None
        return JobRecord.model_validate(raw)

    def put(self, image_id: str, record: JobRecord) -> None:
        self._db[image_id] = record.model_dump()

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
        return {
            k: JobRecord.model_validate(v) for k, v in self._db.items()
        }

    def count_by_status(self) -> dict[JobStatus, int]:
        counts: dict[JobStatus, int] = {s: 0 for s in JobStatus}
        for v in self._db.values():
            status = JobStatus(v.get("status", "pending"))
            counts[status] = counts.get(status, 0) + 1
        return counts

    def incomplete_ids(self) -> list[str]:
        return [
            k
            for k, v in self._db.items()
            if v.get("status") not in (JobStatus.COMPLETE, JobStatus.SKIPPED)
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
            try:
                sm = StateManager(db_path)
                if sm.incomplete_ids():
                    results.append(db_path)
                sm.close()
            except Exception:
                logger.warning("Could not read %s", db_path, exc_info=True)
        return results
