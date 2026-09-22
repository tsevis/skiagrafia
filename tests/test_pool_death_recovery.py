"""test_pool_death_recovery.py  --  one dead worker must cost one page.

On the 376-page APPLE50 run a worker died on page 199 during VitMatte.
ProcessPoolExecutor marks the pool broken, so every future still queued
against it raised at once: 178 pages were recorded as failed in four
seconds, and 177 of them had never been read. The page itself was fine --
re-run alone it produced 44 layers in 31 seconds.

Offline, like the other batch runner tests: no subprocess, no models.
"""
from __future__ import annotations

import sys
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, ClassVar

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batch_runner_fakes import (
    FakeExecutor,
    _make_config,
    _ok_result,
    _write_image,
)

from core import batch_runner
from core.batch_runner import BatchRunner
from core.state_manager import JobStatus


class PoolKillingExecutor(FakeExecutor):
    """Breaks the pool on the Nth drained job, as a dying worker does.

    Every future still queued gets BrokenProcessPool, including the one that
    was running -- the real executor cannot tell them apart either.
    """

    kill_on_call: int = 1
    kills_remaining: int = 1
    built: ClassVar[list[PoolKillingExecutor]] = []

    def __init__(self, max_workers: int | None = None, **kwargs: Any) -> None:
        super().__init__(max_workers=max_workers, **kwargs)
        self._calls = 0
        PoolKillingExecutor.built.append(self)

    def drain(self) -> None:
        pending, self._pending = self._pending, []
        for index, (future, fn, args, kwargs) in enumerate(pending):
            self._calls += 1
            should_kill = (
                PoolKillingExecutor.kills_remaining > 0
                and self._calls == PoolKillingExecutor.kill_on_call
            )
            if should_kill:
                PoolKillingExecutor.kills_remaining -= 1
                for broken, *_ in pending[index:]:
                    if not broken.done():
                        broken.set_exception(BrokenProcessPool("worker died"))
                return
            try:
                future.set_result(fn(*args, **kwargs))
            except Exception as exc:
                future.set_exception(exc)


@pytest.fixture
def pool_killer(monkeypatch: pytest.MonkeyPatch) -> type[PoolKillingExecutor]:
    PoolKillingExecutor.built = []
    PoolKillingExecutor.kill_on_call = 3
    PoolKillingExecutor.kills_remaining = 1
    monkeypatch.setattr(batch_runner, "ProcessPoolExecutor", PoolKillingExecutor)
    monkeypatch.setattr(batch_runner, "_process_single",
                        lambda path, config: _ok_result(path))
    return PoolKillingExecutor


def _runner_over(tmp_path: Path, count: int = 5) -> BatchRunner:
    source = tmp_path / "input"
    source.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        _write_image(source / f"page{index}.png")
    runner = BatchRunner(_make_config(tmp_path))
    runner.discover_images()
    return runner


def _drain_until_idle(runner: BatchRunner, limit: int = 12) -> None:
    """Drain each pool the runner builds, including the ones it rebuilds.

    `_executor` is typed as the real ProcessPoolExecutor; the isinstance
    check narrows it and asserts the fixture actually replaced it.
    """
    for _ in range(limit):
        executor = runner._executor
        if executor is None:
            return
        assert isinstance(executor, PoolKillingExecutor)
        if not executor._pending:
            return
        executor.drain()


def test_a_dead_worker_costs_one_page_not_the_queue(tmp_path: Path, pool_killer) -> None:
    runner = _runner_over(tmp_path, count=5)
    try:
        runner.start()
        _drain_until_idle(runner)

        summary = runner.summary()
        assert summary.completed == 4, "the four healthy pages should still finish"
        assert summary.failed == 1, "only the page that was running should fail"
    finally:
        runner.close()


def test_the_page_that_was_running_is_the_one_marked_failed(
    tmp_path: Path, pool_killer
) -> None:
    runner = _runner_over(tmp_path, count=5)
    try:
        runner.start()
        _drain_until_idle(runner)

        failed = [
            image_id for image_id, record in runner._state.all_records().items()
            if record.status == JobStatus.FAILED
        ]
        assert failed == ["page2"], "the third submitted page was the one in flight"
    finally:
        runner.close()


def test_a_pool_that_never_makes_progress_gives_up(tmp_path: Path, pool_killer) -> None:
    # A rebuild that cannot get anywhere must stop, not rebuild for ever.
    PoolKillingExecutor.kill_on_call = 1
    PoolKillingExecutor.kills_remaining = 99
    runner = _runner_over(tmp_path, count=5)
    try:
        runner.start()
        _drain_until_idle(runner, limit=30)

        assert not runner.is_running, "the run must end rather than rebuild for ever"
        assert len(PoolKillingExecutor.built) <= 5, "rebuilds must be capped"
    finally:
        runner.close()


def test_the_lost_page_says_a_worker_died_not_that_the_image_was_bad(
    tmp_path: Path, pool_killer
) -> None:
    # The page is usually fine -- _page_454_Picture_0 produced 44 layers when
    # re-run alone. Its record should say what actually happened to it.
    runner = _runner_over(tmp_path, count=5)
    try:
        runner.start()
        _drain_until_idle(runner)

        record = runner._state.get("page2")
        assert record is not None
        assert "worker" in (record.error or "").lower()
    finally:
        runner.close()


def test_a_run_that_recovered_is_not_reported_as_stopped_early(
    tmp_path: Path, pool_killer
) -> None:
    runner = _runner_over(tmp_path, count=5)
    try:
        runner.start()
        _drain_until_idle(runner)

        assert runner.summary().stop_reason == "", "it recovered; it did not stop"
    finally:
        runner.close()


def test_a_run_that_gave_up_says_how_many_pages_it_never_reached(
    tmp_path: Path, pool_killer
) -> None:
    # "178 failed" and "one worker kept dying and 177 were never read" are
    # different sentences, and only the second one is true.
    PoolKillingExecutor.kill_on_call = 1
    PoolKillingExecutor.kills_remaining = 99
    runner = _runner_over(tmp_path, count=8)
    try:
        runner.start()
        _drain_until_idle(runner, limit=30)

        reason = runner.summary().stop_reason
        assert "unattempted" in reason, reason
    finally:
        runner.close()
