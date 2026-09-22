"""test_worker_footprint.py  --  what a run keeps about a worker that dies.

A worker died on page 199 of the APPLE50 book and the run lost 53 minutes.
`BrokenProcessPool` reports that a worker died, never why. There was no
resident-memory figure, no exit signal, no record of which step was in
flight. The report for that run has a hypothesis where it should have had
a cause, and said so.

The worker is a separate process, so the only number the coordinator can
be sure of is the one the worker hands back with its result. It reports
its own peak resident memory per image; the runner keeps the last one and
says it when the pool breaks.
"""
from __future__ import annotations

import sys
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, ClassVar

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batch_runner_fakes import FakeExecutor, _make_config, _ok_result, _write_image

from core import batch_runner
from core.batch_runner import BatchRunner
from core.orchestrator import peak_resident_mb
from core.state_manager import JobStatus

DEADLINE = 10.0


def test_a_worker_can_measure_its_own_footprint() -> None:
    # Stdlib only: psutil is not a declared dependency of this project.
    measured = peak_resident_mb()

    assert measured > 0, "a running process occupies some memory"


class DyingExecutor(FakeExecutor):
    """Reports a footprint for a while, then loses its worker."""

    kill_on_call: ClassVar[int] = 3
    footprint_mb: ClassVar[float] = 1234.5

    def __init__(self, max_workers: int | None = None, **kwargs: Any) -> None:
        super().__init__(max_workers=max_workers, **kwargs)
        self._calls = 0

    def drain(self) -> None:
        pending, self._pending = self._pending, []
        for index, (future, _fn, args, _kwargs) in enumerate(pending):
            self._calls += 1
            if self._calls == DyingExecutor.kill_on_call:
                for broken, *_ in pending[index:]:
                    if not broken.done():
                        broken.set_exception(BrokenProcessPool("worker died"))
                return
            result = _ok_result(args[0])
            result.worker_peak_rss_mb = DyingExecutor.footprint_mb
            future.set_result(result)


@pytest.fixture
def dying(monkeypatch: pytest.MonkeyPatch) -> type[DyingExecutor]:
    DyingExecutor.kill_on_call = 3
    monkeypatch.setattr(batch_runner, "ProcessPoolExecutor", DyingExecutor)
    monkeypatch.setattr(batch_runner, "_process_single", lambda path, config: _ok_result(path))
    return DyingExecutor


def _runner(tmp_path: Path, count: int = 5) -> BatchRunner:
    source = tmp_path / "input"
    source.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        _write_image(source / f"page{index}.png")
    runner = BatchRunner(_make_config(tmp_path))
    runner.discover_images()
    return runner


def _drain(runner: BatchRunner) -> None:
    import threading

    for _ in range(12):
        executor = runner._executor
        if executor is None:
            return
        assert isinstance(executor, DyingExecutor)
        if not executor._pending:
            return
        worker = threading.Thread(target=executor.drain, name="pool-callback")
        worker.start()
        worker.join(timeout=DEADLINE)
        assert not worker.is_alive(), "the completion callback blocked"
        runner.wait_for_recovery(timeout=DEADLINE)


def test_the_lost_page_records_how_large_the_worker_had_grown(
    tmp_path: Path, dying: type[DyingExecutor]
) -> None:
    # The question the APPLE50 post-mortem could not answer.
    runner = _runner(tmp_path)
    try:
        runner.start()
        _drain(runner)

        failed = [
            record for record in runner._state.all_records().values()
            if record.status == JobStatus.FAILED
        ]
        assert failed, "one page should be charged for the death"
        assert "1234" in (failed[0].error or ""), failed[0].error
    finally:
        runner.close()


def test_a_death_before_any_image_finished_says_so_rather_than_inventing_a_figure(
    tmp_path: Path, dying: type[DyingExecutor]
) -> None:
    # Nothing came back, so there is no measurement. Saying "0 MB" would be
    # a number that looks like one and is not.
    DyingExecutor.kill_on_call = 1
    runner = _runner(tmp_path)
    try:
        runner.start()
        _drain(runner)

        failed = [
            record for record in runner._state.all_records().values()
            if record.status == JobStatus.FAILED
        ]
        assert failed
        assert "not known" in (failed[0].error or "").lower(), failed[0].error
    finally:
        runner.close()
