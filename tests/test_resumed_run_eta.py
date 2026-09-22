"""test_resumed_run_eta.py  --  the rate a resumed run reports.

The second APPLE50 attempt inherited 198 finished pages and re-ran 178.
Those 198 count as done the instant the run starts, and the rate divided
all of them by the seconds since start: the app reported 8.9 pages/min
and 8 minutes left while the real throughput was 2.15 pages/min and 34
minutes. An ETA that confident and that wrong is worse than none.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batch_runner_fakes import (
    FakeExecutor,
    _drain,
    _make_config,
    _ok_result,
    _write_image,
)

from core import batch_runner
from core.batch_runner import BatchProgress, BatchRunner
from core.state_manager import JobStatus


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(batch_runner, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(batch_runner, "_process_single",
                        lambda path, config: _ok_result(path))


def _runner_with_finished_pages(tmp_path: Path, total: int, finished: int) -> BatchRunner:
    source = tmp_path / "input"
    source.mkdir(parents=True, exist_ok=True)
    for index in range(total):
        _write_image(source / f"page{index}.png")
    runner = BatchRunner(_make_config(tmp_path))
    runner.discover_images()
    for index in range(finished):
        runner._state.update_status(f"page{index}", JobStatus.COMPLETE)
    return runner


def test_a_resumed_run_rates_only_the_work_it_did(tmp_path: Path) -> None:
    reported: list[BatchProgress] = []
    runner = _runner_with_finished_pages(tmp_path, total=5, finished=3)
    runner._progress_cb = reported.append
    try:
        runner.start()
        # One minute of wall clock, so the arithmetic is checkable.
        runner._start_time = time.time() - 60
        _drain(runner)

        assert reported, "a completion should report progress"
        # Two pages ran in this minute. Counting the three it inherited
        # would report 5.0.
        assert reported[-1].images_per_min == pytest.approx(2.0, abs=0.2)
    finally:
        runner.close()


def test_no_eta_is_offered_before_anything_has_finished(tmp_path: Path) -> None:
    # With inherited work in the numerator the old rate produced a
    # confident ETA before this run had processed a single page.
    runner = _runner_with_finished_pages(tmp_path, total=5, finished=3)
    try:
        runner.start()
        runner._start_time = time.time() - 60

        progress = runner._get_progress()

        assert progress.images_per_min == 0.0
        assert progress.eta_seconds == 0.0
    finally:
        runner.close()


def test_the_remaining_count_still_covers_the_whole_book(tmp_path: Path) -> None:
    # Only the rate changes: what is left to do is still measured against
    # every page, not just this run's share.
    runner = _runner_with_finished_pages(tmp_path, total=5, finished=3)
    try:
        runner.start()

        progress = runner._get_progress()

        assert progress.total == 5
        assert progress.completed == 3
        assert progress.remaining == 2
    finally:
        runner.close()
