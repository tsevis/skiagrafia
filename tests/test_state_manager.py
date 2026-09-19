"""test_state_manager.py  --  StateManager persistence & scan behaviour.

All tests operate on SQLite files under pytest's tmp_path -- never on any
real user config or output directory.
"""
from __future__ import annotations

import json
import pickle
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.state_manager import JobRecord, JobStatus, StateManager


def _make_manager(tmp_path: Path, name: str = "batch") -> StateManager:
    return StateManager(tmp_path / name / "state.db")


class TestPutAndGet:
    def test_round_trips_a_record(self, tmp_path: Path) -> None:
        sm = _make_manager(tmp_path)
        record = JobRecord(image_path="/images/foo.png", status=JobStatus.PENDING)
        sm.put("foo", record)

        fetched = sm.get("foo")

        assert fetched is not None
        assert fetched.image_path == "/images/foo.png"
        assert fetched.status == JobStatus.PENDING
        sm.close()

    def test_get_unknown_id_returns_none(self, tmp_path: Path) -> None:
        sm = _make_manager(tmp_path)
        assert sm.get("does-not-exist") is None
        sm.close()

    def test_put_overwrites_existing_record(self, tmp_path: Path) -> None:
        sm = _make_manager(tmp_path)
        sm.put("foo", JobRecord(image_path="/a.png", status=JobStatus.PENDING))
        sm.put("foo", JobRecord(image_path="/a.png", status=JobStatus.COMPLETE))

        assert sm.get("foo").status == JobStatus.COMPLETE
        sm.close()

    def test_survives_reopen_of_same_db_path(self, tmp_path: Path) -> None:
        db_path = tmp_path / "batch" / "state.db"
        sm1 = StateManager(db_path)
        sm1.put("foo", JobRecord(image_path="/a.png", status=JobStatus.MASKING))
        sm1.close()

        sm2 = StateManager(db_path)
        record = sm2.get("foo")
        assert record is not None
        assert record.status == JobStatus.MASKING
        sm2.close()


class TestOnDiskFormat:
    """A state.db may arrive from anywhere -- a copied or shared batch folder.

    Reading one must never deserialise executable payloads, so the stored
    values are JSON text and are parsed as JSON only.
    """

    def test_records_are_stored_as_json_text(self, tmp_path: Path) -> None:
        db_path = tmp_path / "batch" / "state.db"
        sm = StateManager(db_path)
        sm.put("foo", JobRecord(image_path="/a.png", status=JobStatus.MASKING))
        sm.close()

        with sqlite3.connect(db_path) as conn:
            rows = conn.execute("SELECT image_id, record FROM jobs").fetchall()

        assert len(rows) == 1
        image_id, record = rows[0]
        assert image_id == "foo"
        assert isinstance(record, str)
        assert json.loads(record)["image_path"] == "/a.png"

    def test_a_pickle_payload_is_rejected_instead_of_deserialised(
        self, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "batch" / "state.db"
        sm = StateManager(db_path)
        sm.put("foo", JobRecord(image_path="/a.png"))
        sm.close()

        payload = pickle.dumps({"image_path": "/evil.png"})
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE jobs SET record = ? WHERE image_id = ?", (payload, "foo")
            )

        sm = StateManager(db_path)
        with pytest.raises(ValueError):
            sm.get("foo")
        sm.close()

    def test_a_malformed_record_fails_loudly_rather_than_silently(
        self, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "batch" / "state.db"
        sm = StateManager(db_path)
        sm.put("foo", JobRecord(image_path="/a.png"))
        sm.close()

        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE jobs SET record = ? WHERE image_id = ?", ("{not json", "foo")
            )

        sm = StateManager(db_path)
        with pytest.raises(ValueError):
            sm.all_records()
        sm.close()


class TestUpdateStatus:
    def test_updates_status_and_clears_error_by_default(self, tmp_path: Path) -> None:
        sm = _make_manager(tmp_path)
        sm.put(
            "foo",
            JobRecord(image_path="/a.png", status=JobStatus.FAILED, error="boom"),
        )

        sm.update_status("foo", JobStatus.PENDING)

        record = sm.get("foo")
        assert record is not None
        assert record.status == JobStatus.PENDING
        assert record.error is None
        sm.close()

    def test_updates_status_with_error_message(self, tmp_path: Path) -> None:
        sm = _make_manager(tmp_path)
        sm.put("foo", JobRecord(image_path="/a.png", status=JobStatus.PENDING))

        sm.update_status("foo", JobStatus.FAILED, error="network timeout")

        record = sm.get("foo")
        assert record is not None
        assert record.status == JobStatus.FAILED
        assert record.error == "network timeout"
        sm.close()

    def test_updating_unknown_image_id_is_a_noop(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        sm = _make_manager(tmp_path)

        with caplog.at_level("WARNING"):
            sm.update_status("ghost", JobStatus.COMPLETE)

        assert sm.get("ghost") is None
        assert any("Cannot update status" in msg for msg in caplog.messages)
        sm.close()


class TestAllRecordsAndCounts:
    def test_all_records_returns_every_stored_job(self, tmp_path: Path) -> None:
        sm = _make_manager(tmp_path)
        sm.put("a", JobRecord(image_path="/a.png", status=JobStatus.PENDING))
        sm.put("b", JobRecord(image_path="/b.png", status=JobStatus.COMPLETE))

        records = sm.all_records()

        assert set(records) == {"a", "b"}
        assert all(isinstance(r, JobRecord) for r in records.values())
        sm.close()

    def test_all_records_empty_db(self, tmp_path: Path) -> None:
        sm = _make_manager(tmp_path)
        assert sm.all_records() == {}
        sm.close()

    def test_count_by_status_with_mixed_statuses(self, tmp_path: Path) -> None:
        sm = _make_manager(tmp_path)
        sm.put("a", JobRecord(image_path="/a.png", status=JobStatus.COMPLETE))
        sm.put("b", JobRecord(image_path="/b.png", status=JobStatus.COMPLETE))
        sm.put("c", JobRecord(image_path="/c.png", status=JobStatus.FAILED))
        sm.put("d", JobRecord(image_path="/d.png", status=JobStatus.PENDING))

        counts = sm.count_by_status()

        assert counts[JobStatus.COMPLETE] == 2
        assert counts[JobStatus.FAILED] == 1
        assert counts[JobStatus.PENDING] == 1
        assert counts[JobStatus.SKIPPED] == 0
        # Every JobStatus member is represented, even with zero jobs.
        assert set(counts) == set(JobStatus)
        sm.close()

    def test_count_by_status_empty_db_has_all_zero_counts(
        self, tmp_path: Path
    ) -> None:
        sm = _make_manager(tmp_path)
        counts = sm.count_by_status()
        assert all(v == 0 for v in counts.values())
        sm.close()


class TestIncompleteIds:
    def test_incomplete_ids_excludes_complete_and_skipped(
        self, tmp_path: Path
    ) -> None:
        sm = _make_manager(tmp_path)
        sm.put("a", JobRecord(image_path="/a.png", status=JobStatus.COMPLETE))
        sm.put("b", JobRecord(image_path="/b.png", status=JobStatus.SKIPPED))
        sm.put("c", JobRecord(image_path="/c.png", status=JobStatus.PENDING))
        sm.put("d", JobRecord(image_path="/d.png", status=JobStatus.FAILED))

        incomplete = sm.incomplete_ids()

        assert set(incomplete) == {"c", "d"}
        sm.close()

    def test_incomplete_ids_empty_when_all_done(self, tmp_path: Path) -> None:
        sm = _make_manager(tmp_path)
        sm.put("a", JobRecord(image_path="/a.png", status=JobStatus.COMPLETE))
        sm.put("b", JobRecord(image_path="/b.png", status=JobStatus.SKIPPED))

        assert sm.incomplete_ids() == []
        sm.close()


class TestFindIncompleteBatches:
    def test_returns_empty_list_when_output_dir_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist"
        assert StateManager.find_incomplete_batches(missing) == []

    def test_returns_empty_list_when_output_dir_has_no_batches(
        self, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        assert StateManager.find_incomplete_batches(output_dir) == []

    def test_finds_batch_with_incomplete_jobs(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "out"
        batch_db = output_dir / "batch1" / "state.db"
        sm = StateManager(batch_db)
        sm.put("a", JobRecord(image_path="/a.png", status=JobStatus.PENDING))
        sm.close()

        results = StateManager.find_incomplete_batches(output_dir)

        assert results == [batch_db]

    def test_skips_batch_where_all_jobs_are_done(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "out"
        batch_db = output_dir / "batch1" / "state.db"
        sm = StateManager(batch_db)
        sm.put("a", JobRecord(image_path="/a.png", status=JobStatus.COMPLETE))
        sm.put("b", JobRecord(image_path="/b.png", status=JobStatus.SKIPPED))
        sm.close()

        assert StateManager.find_incomplete_batches(output_dir) == []

    def test_skips_unreadable_db_without_raising(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "out"
        corrupt_dir = output_dir / "corrupt_batch"
        corrupt_dir.mkdir(parents=True)
        # Not a valid SQLite file -- StateManager() construction will raise
        # inside find_incomplete_batches, which must swallow it.
        (corrupt_dir / "state.db").write_bytes(b"not a real sqlite database")

        good_db = output_dir / "good_batch" / "state.db"
        sm = StateManager(good_db)
        sm.put("a", JobRecord(image_path="/a.png", status=JobStatus.PENDING))
        sm.close()

        results = StateManager.find_incomplete_batches(output_dir)

        assert results == [good_db]
