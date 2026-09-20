"""batch_runner_fakes.py  --  offline stand-ins for the batch runner tests.

A ProcessPoolExecutor that runs everything in the calling thread, the canned
results a worker would return, and the patch that puts them in place of real
per-image processing. No subprocess is ever spawned and no model is loaded.

Shared by the test_batch_runner modules so they can be split by what they
cover without either copying this. These are plain callables, not fixtures:
a fixture shared across modules belongs in conftest.py, and the one autouse
fixture here stays with the tests that need it.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import batch_runner
from core.batch_runner import BatchConfig, BatchRunner
from core.orchestrator import PipelineResult
from core.state_manager import JobRecord


def _write_image(path: Path, size: int = 32) -> Path:
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


def _make_config(tmp_path: Path, **overrides: Any) -> BatchConfig:
    base: dict[str, Any] = {
        "input_folder": str(tmp_path / "input"),
        "output_dir": str(tmp_path / "output"),
        "confirmed_labels": ["cat"],
    }
    base.update(overrides)
    return BatchConfig(**base)


def _ok_result(image_path: str) -> PipelineResult:
    return PipelineResult(
        image_path=image_path,
        width=32,
        height=32,
        svg_path=image_path + ".svg",
        tiff_path=image_path + ".tiff",
        all_objects_tiff_path=image_path + ".all-objects.tiff",
    )


def _failing_result(image_path: str, message: str = "boom") -> PipelineResult:
    return PipelineResult(image_path=image_path, width=32, height=32, error=message)


class FakeExecutor:
    """Stand-in for ProcessPoolExecutor: never spawns a subprocess.

    `submit()` returns a *pending* Future and queues the call; the caller
    (BatchRunner.start) registers its done-callback on that pending Future,
    exactly as it would against a real ProcessPoolExecutor where the worker
    process has not finished yet. Tests then call `drain()` once `start()`
    has returned (i.e. once every job for this batch has been submitted and
    every future's done-callback registered) to run the queued callables
    in-process and complete the futures -- this is what fires
    `BatchRunner._on_complete` for each job, synchronously, with no thread
    or real subprocess involved.
    """

    def __init__(self, max_workers: int | None = None, **kwargs: Any) -> None:
        self.max_workers = max_workers
        self.submitted: list[tuple] = []
        self.shutdown_calls: list[tuple[bool, bool]] = []
        self._pending: list[tuple[Future, Callable, tuple, dict]] = []

    def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> Future:
        self.submitted.append((fn, args, kwargs))
        future: Future = Future()
        self._pending.append((future, fn, args, kwargs))
        return future

    def drain(self) -> None:
        """Run every queued job and complete its Future."""
        pending, self._pending = self._pending, []
        for future, fn, args, kwargs in pending:
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        self.shutdown_calls.append((wait, cancel_futures))
        if cancel_futures:
            # A real executor cancels only what has NOT started. Model the
            # first max_workers submissions as already picked up by a
            # worker, so a test can exercise a job that finishes after the
            # caller asked to stop.
            running = self.max_workers or 1
            for future, *_ in self._pending[running:]:
                future.cancel()
            self._pending = self._pending[:running]
        if wait:
            # A real executor's shutdown(wait=True) lets in-flight work
            # finish, which fires each future's done-callback. A fake that
            # returns without doing so cannot show what happens to a
            # callback that lands after the caller has torn things down.
            self.drain()


def _drain(runner: BatchRunner) -> None:
    """Drain the `FakeExecutor` the `_fake_executor` fixture installed.

    `BatchRunner._executor` is typed as `ProcessPoolExecutor | None` in
    production; the isinstance check both narrows that for the type checker
    and asserts the fixture actually did its job.
    """
    executor = runner._executor
    assert isinstance(executor, FakeExecutor)
    executor.drain()


def _record(runner: BatchRunner, image_id: str) -> JobRecord:
    """Fetch a state record that a test expects to exist."""
    record = runner._state.get(image_id)
    assert record is not None
    return record


def _patch_process_single(
    monkeypatch: pytest.MonkeyPatch,
    fn_by_image: dict[str, PipelineResult] | None = None,
    default: PipelineResult | Exception | None = None,
    fn: Callable[[str, dict], PipelineResult] | None = None,
) -> list[tuple[str, dict]]:
    """Monkeypatch the module-level worker function used by start().

    Replaces core.batch_runner._process_single so the real
    build_capabilities/build_knowledge_pack/Orchestrator machinery in that
    function is never reached.
    """
    calls: list[tuple[str, dict]] = []

    def fake(image_path: str, config_dict: dict) -> PipelineResult:
        calls.append((image_path, config_dict))
        if fn is not None:
            return fn(image_path, config_dict)
        stem = Path(image_path).stem
        if fn_by_image and stem in fn_by_image:
            outcome = fn_by_image[stem]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        if isinstance(default, Exception):
            raise default
        if default is not None:
            return default
        return _ok_result(image_path)

    monkeypatch.setattr(batch_runner, "_process_single", fake)
    return calls


class EagerExecutor(FakeExecutor):
    """Completes each Future inside submit(), before start() moves on.

    A real ProcessPoolExecutor leaves a window between submitting a job and
    the worker finishing it. That window is usually wide enough to hide
    ordering bugs in start(); completing eagerly collapses it to zero and
    makes them deterministic.
    """

    def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> Future:
        self.submitted.append((fn, args, kwargs))
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def drain(self) -> None:
        """Everything already ran in submit(); nothing left to drain."""
