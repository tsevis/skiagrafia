"""What a core batch run records about work it completed imperfectly.

`_on_complete` read `result.error` and the layer count and dropped
`result.warnings`. The state DB is the only durable record of a batch, so a
warning not written there can never be recovered: the PipelineResult is
discarded at the process boundary.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.state_manager import JobRecord, JobStatus


def test_a_job_record_can_carry_what_the_run_could_not_do() -> None:
    record = JobRecord(
        image_path="/in/a.png",
        status=JobStatus.COMPLETE,
        warnings=["Stopped at 64 object instances", "Could not locate 'keyboard'"],
    )

    assert record.warnings == [
        "Stopped at 64 object instances",
        "Could not locate 'keyboard'",
    ]


def test_a_record_written_before_warnings_existed_still_loads() -> None:
    """The SQLite rows are JSON; an older row simply has no such key."""
    record = JobRecord.model_validate(
        {"image_path": "/in/a.png", "status": "complete", "layer_count": 3}
    )

    assert record.warnings == []
