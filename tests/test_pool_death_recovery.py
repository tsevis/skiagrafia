"""test_pool_death_recovery.py  --  one dead worker must cost one page.

On the 376-page APPLE50 run a worker died on page 199 during VitMatte.
ProcessPoolExecutor marks the pool broken, so every future queued against
it raised at once: 178 pages failed in four seconds and 177 had never been
read. The page was fine -- re-run alone it produced 44 layers in 31s.

The first attempt at this fix (#28) rebuilt the pool inside the executor's
own done-callback thread. `shutdown()` there waits for the machinery that
is running the callback, so it waited for itself: in the field the run sat
at 0% CPU for 65 minutes. Those tests passed because FakeExecutor runs
every job on the calling thread -- no callback thread, no deadlock to find.

So the tests here pin the *structure*, not only the outcome: the callback
must record and return, and the rebuild must happen on another thread,
within a deadline. A test that hangs is a failing test.
"""
from __future__ import annotations

import sys
import threading
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, ClassVar

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batch_runner_fakes import FakeExecutor, _make_config, _ok_result, _write_image

from core import batch_runner
from core.batch_runner import BatchRunner
from core.state_manager import JobStatus

DEADLINE = 10.0


class RecordingExecutor(FakeExecutor):
    """Notes which thread tore it down, and can break on a chosen job."""

    built: ClassVar[list[RecordingExecutor]] = []
    kill_on_call: ClassVar[int] = 0
    kills_remaining: ClassVar[int] = 0

    def __init__(self, max_workers: int | None = None, **kwargs: Any) -> None:
        super().__init__(max_workers=max_workers, **kwargs)
        self._calls = 0
        self.shutdown_threads: list[str] = []
        RecordingExecutor.built.append(self)

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        self.shutdown_threads.append(threading.current_thread().name)
        super().shutdown(wait=False, cancel_futures=False)

    def drain(self) -> None:
        pending, self._pending = self._pending, []
        for index, (future, fn, args, kwargs) in enumerate(pending):
            self._calls += 1
            if RecordingExecutor.kills_remaining > 0 and self._calls == RecordingExecutor.kill_on_call:
                RecordingExecutor.kills_remaining -= 1
                for broken, *_ in pending[index:]:
                    if not broken.done():
                        broken.set_exception(BrokenProcessPool("worker died"))
                return
            try:
                future.set_result(fn(*args, **kwargs))
            except Exception as exc:
                future.set_exception(exc)


@pytest.fixture
def killer(monkeypatch: pytest.MonkeyPatch) -> type[RecordingExecutor]:
    RecordingExecutor.built = []
    RecordingExecutor.kill_on_call = 3
    RecordingExecutor.kills_remaining = 1
    monkeypatch.setattr(batch_runner, "ProcessPoolExecutor", RecordingExecutor)
    monkeypatch.setattr(batch_runner, "_process_single", lambda path, config: _ok_result(path))
    return RecordingExecutor


def _runner(tmp_path: Path, count: int = 5) -> BatchRunner:
    source = tmp_path / "input"
    source.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        _write_image(source / f"page{index}.png")
    runner = BatchRunner(_make_config(tmp_path))
    runner.discover_images()
    return runner


def _drain_from_a_worker_thread(runner: BatchRunner, rounds: int = 12) -> None:
    """Drain on a thread that is not the caller's, as a real pool does.

    Any blocking in the completion callback shows up here as a thread that
    does not finish, and the join deadline turns that into a failure rather
    than a hung suite.
    """
    for _ in range(rounds):
        executor = runner._executor
        if executor is None:
            return
        assert isinstance(executor, RecordingExecutor)
        if not executor._pending:
            return
        worker = threading.Thread(target=executor.drain, name="pool-callback")
        worker.start()
        worker.join(timeout=DEADLINE)
        assert not worker.is_alive(), "the completion callback blocked"
        runner.wait_for_recovery(timeout=DEADLINE)


def test_the_completion_callback_never_tears_the_pool_down(tmp_path: Path, killer) -> None:
    # The deadlock in #28: shutdown() called from the thread the executor
    # is running its own callback on waits for itself.
    runner = _runner(tmp_path, count=5)
    try:
        runner.start()
        _drain_from_a_worker_thread(runner)

        torn_down_by = [t for ex in RecordingExecutor.built for t in ex.shutdown_threads]
        assert "pool-callback" not in torn_down_by, (
            f"the pool was shut down from its own callback thread: {torn_down_by}"
        )
    finally:
        runner.close()


def test_a_dead_worker_costs_one_page_not_the_queue(tmp_path: Path, killer) -> None:
    runner = _runner(tmp_path, count=5)
    try:
        runner.start()
        _drain_from_a_worker_thread(runner)

        summary = runner.summary()
        assert summary.completed == 4
        assert summary.failed == 1
    finally:
        runner.close()


def test_the_page_that_was_running_is_the_one_marked_failed(tmp_path: Path, killer) -> None:
    runner = _runner(tmp_path, count=5)
    try:
        runner.start()
        _drain_from_a_worker_thread(runner)

        failed = [
            image_id for image_id, record in runner._state.all_records().items()
            if record.status == JobStatus.FAILED
        ]
        assert failed == ["page2"]
    finally:
        runner.close()


def test_the_lost_page_says_a_worker_died_not_that_the_image_was_bad(
    tmp_path: Path, killer
) -> None:
    runner = _runner(tmp_path, count=5)
    try:
        runner.start()
        _drain_from_a_worker_thread(runner)

        record = runner._state.get("page2")
        assert record is not None
        assert "worker" in (record.error or "").lower()
    finally:
        runner.close()


def test_a_pool_that_never_makes_progress_gives_up(tmp_path: Path, killer) -> None:
    RecordingExecutor.kill_on_call = 1
    RecordingExecutor.kills_remaining = 99
    runner = _runner(tmp_path, count=8)
    try:
        runner.start()
        _drain_from_a_worker_thread(runner, rounds=30)

        assert not runner.is_running, "the run must end rather than rebuild for ever"
        assert len(RecordingExecutor.built) <= 5, "rebuilds must be capped"
        assert "unattempted" in runner.summary().stop_reason
    finally:
        runner.close()


def test_a_run_that_recovered_is_not_reported_as_stopped_early(
    tmp_path: Path, killer
) -> None:
    runner = _runner(tmp_path, count=5)
    try:
        runner.start()
        _drain_from_a_worker_thread(runner)

        assert runner.summary().stop_reason == ""
    finally:
        runner.close()
