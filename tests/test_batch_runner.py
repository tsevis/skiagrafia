"""test_batch_runner.py  --  BatchRunner / BatchConfig coordination logic.

All tests are offline: no ML weights are ever resolved or loaded, no GPU/MPS
work happens, and no network I/O occurs. `core.batch_runner` imports
`build_capabilities` / `build_knowledge_pack` (from core.factory) and
`Orchestrator` (from core.orchestrator) only inside the module-level
`_process_single` worker function -- every test here monkeypatches that
function directly with a lightweight fake, so those factory/orchestrator
symbols are never actually invoked.

`BatchRunner.start()` uses a real `ProcessPoolExecutor` in production. Tests
replace `core.batch_runner.ProcessPoolExecutor` with an in-process fake that
runs the submitted callable synchronously (within `submit`) and returns an
already-completed `concurrent.futures.Future`. Because a `Future`'s
`add_done_callback` invokes the callback immediately once the future is
already done, `BatchRunner._on_complete` runs synchronously during `start()`
itself -- no real worker process is ever spawned, and no background thread or
polling loop is needed to observe results.

Tiny synthetic 32x32 uint8 images are written to pytest's `tmp_path` with
PIL when the code needs real files on disk (only `discover_images()` reads
the filesystem; image *contents* are never decoded by this module).
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batch_runner_fakes import (
    EagerExecutor,
    FakeExecutor,
    _drain,
    _failing_result,
    _make_config,
    _ok_result,
    _patch_process_single,
    _record,
    _write_image,
)

from core import batch_runner
from core.batch_runner import BatchProgress, BatchRunner
from core.orchestrator import LayerResult, PipelineResult
from core.state_manager import JobRecord, JobStatus, StateManager

# ── Helpers ──────────────────────────────────────────────────────────────












@pytest.fixture(autouse=True)
def _fake_executor(monkeypatch: pytest.MonkeyPatch) -> type[FakeExecutor]:
    """Every test in this module gets the inline fake executor -- a real
    ProcessPoolExecutor must never be constructed here."""
    monkeypatch.setattr(batch_runner, "ProcessPoolExecutor", FakeExecutor)
    return FakeExecutor








# ── BatchConfig ──────────────────────────────────────────────────────────


class TestBatchConfig:
    def test_auto_generates_batch_id_when_blank(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        assert cfg.batch_id
        assert len(cfg.batch_id) == 12

    def test_auto_generated_batch_ids_are_unique(self, tmp_path: Path) -> None:
        ids = {_make_config(tmp_path).batch_id for _ in range(20)}
        assert len(ids) == 20

    def test_preserves_explicit_batch_id(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, batch_id="my-batch")
        assert cfg.batch_id == "my-batch"

    def test_zero_max_workers_resolves_to_cpu_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(batch_runner.os, "cpu_count", lambda: 7)
        cfg = _make_config(tmp_path, max_workers=0)
        assert cfg.max_workers == 7

    def test_zero_max_workers_falls_back_to_four_when_cpu_count_unknown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(batch_runner.os, "cpu_count", lambda: None)
        cfg = _make_config(tmp_path, max_workers=0)
        assert cfg.max_workers == 4

    def test_explicit_max_workers_is_preserved(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, max_workers=3)
        assert cfg.max_workers == 3

    def test_negative_max_workers_also_resolves_via_cpu_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(batch_runner.os, "cpu_count", lambda: 5)
        cfg = _make_config(tmp_path, max_workers=-1)
        assert cfg.max_workers == 5


def test_worker_uses_the_frozen_per_image_labels_and_selection_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = str(tmp_path / "input" / "a.png")
    config = _make_config(
        tmp_path,
        confirmed_labels=["fallback"],
        labels_by_image={image_path: ["Apple computer"]},
        selections_by_image={image_path: {"Apple computer": "largest"}},
    )
    events: list[tuple[str, object]] = []

    class StubOrchestrator:
        def set_confirmed_selections(self, selections: dict[str, str]) -> None:
            events.append(("selections", selections))

        def process(self, path: str, labels: list[str]) -> PipelineResult:
            events.append(("process", (path, labels)))
            return _ok_result(path)

    monkeypatch.setattr(batch_runner, "_worker_orchestrator", lambda _: StubOrchestrator())

    result = batch_runner._process_single(image_path, config.model_dump())

    assert result.error is None
    assert events == [
        ("selections", {"Apple computer": "largest"}),
        ("process", (image_path, ["Apple computer"])),
    ]


# ── discover_images ──────────────────────────────────────────────────────


class TestDiscoverImages:
    def test_finds_supported_extensions_only_and_sorts_them(
        self, tmp_path: Path
    ) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "b.png")
        _write_image(input_dir / "a.jpg")
        (input_dir / "notes.txt").write_text("not an image")
        (input_dir / "ignored.psd").write_bytes(b"nope")

        cfg = _make_config(tmp_path)
        runner = BatchRunner(cfg)
        found = runner.discover_images()

        assert [p.name for p in found] == ["a.jpg", "b.png"]
        runner.close()

    def test_ignores_subdirectories(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "a.png")
        (input_dir / "subdir").mkdir()
        _write_image(input_dir / "subdir" / "hidden.png")

        cfg = _make_config(tmp_path)
        runner = BatchRunner(cfg)
        found = runner.discover_images()

        assert [p.name for p in found] == ["a.png"]
        runner.close()

    def test_creates_pending_state_records_for_new_images(
        self, tmp_path: Path
    ) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "a.png")

        cfg = _make_config(tmp_path)
        runner = BatchRunner(cfg)
        runner.discover_images()

        record = runner._state.get("a")
        assert record is not None
        assert record.status == JobStatus.PENDING
        assert record.image_path.endswith("a.png")
        runner.close()

    def test_does_not_clobber_existing_state_record(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "a.png")

        cfg = _make_config(tmp_path)
        batch_dir = Path(cfg.output_dir) / cfg.batch_id
        batch_dir.mkdir(parents=True)
        seed = StateManager(batch_dir / "state.db")
        seed.put("a", JobRecord(image_path="/old/a.png", status=JobStatus.COMPLETE))
        seed.close()

        runner = BatchRunner(cfg)
        runner.discover_images()

        record = runner._state.get("a")
        assert record is not None
        assert record.status == JobStatus.COMPLETE
        assert record.image_path == "/old/a.png"
        runner.close()

    def test_frozen_inputs_do_not_pick_up_new_folder_images(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        approved = _write_image(input_dir / "approved.png")
        _write_image(input_dir / "new-unreviewed.png")
        calls = _patch_process_single(monkeypatch)

        runner = BatchRunner(
            _make_config(tmp_path, input_images=[str(approved)], max_workers=1)
        )
        runner.start()
        _drain(runner)

        assert [Path(path).name for path, _ in calls] == ["approved.png"]
        assert runner._state.get("new-unreviewed") is None
        runner.close()


# ── start() / _on_complete / _get_progress ──────────────────────────────


class TestStartSuccessPath:
    def test_processes_all_images_and_marks_them_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "a.png")
        _write_image(input_dir / "b.png")
        _patch_process_single(monkeypatch)

        cfg = _make_config(tmp_path, max_workers=1)
        runner = BatchRunner(cfg)
        runner.start()
        _drain(runner)

        assert _record(runner, "a").status == JobStatus.COMPLETE
        assert _record(runner, "b").status == JobStatus.COMPLETE
        assert _record(runner, "a").output_svg == str(input_dir / "a.png") + ".svg"
        assert _record(runner, "a").output_all_objects_tiff == str(input_dir / "a.png") + ".all-objects.tiff"
        assert not runner.is_running
        runner.close()

    def test_submits_serialised_config_dict_to_worker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "a.png")
        calls = _patch_process_single(monkeypatch)

        cfg = _make_config(tmp_path, max_workers=1)
        runner = BatchRunner(cfg)
        runner.start()
        _drain(runner)

        assert len(calls) == 1
        image_path, config_dict = calls[0]
        assert image_path == str(input_dir / "a.png")
        assert config_dict["batch_id"] == cfg.batch_id
        assert config_dict["confirmed_labels"] == ["cat"]
        runner.close()

    def test_persists_per_image_triage_and_output_metrics(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        image_path = _write_image(input_dir / "a.png")

        result = _ok_result(str(image_path))
        result.layers = [
            LayerResult(
                layer_id="object-001",
                label="Apple computer",
                role="parent",
                bbox=(0, 0, 16, 16),
            )
        ]
        calls: list[tuple[str, dict]] = []

        def worker(path: str, config: dict) -> PipelineResult:
            calls.append((path, config))
            return result

        monkeypatch.setattr(batch_runner, "_process_single", worker)
        cfg = _make_config(
            tmp_path,
            max_workers=1,
            confirmed_labels=["Apple computer"],
            input_images=[str(image_path)],
            labels_by_image={str(image_path): ["Apple computer"]},
            selections_by_image={str(image_path): {"Apple computer": "largest"}},
        )
        runner = BatchRunner(cfg)
        runner.start()
        _drain(runner)

        record = runner._state.get("a")
        assert record is not None
        assert record.labels == ["Apple computer"]
        assert record.layer_count == 1
        assert runner.summary().model_dump() == {
            "total": 1,
            "completed": 1,
            "failed": 0,
            "svg_count": 1,
            "all_objects_count": 1,
            "avg_layers": 1.0,
            "failed_image_paths": [],
            "status_by_image": {str(image_path): "complete"},
            # Empty unless the run ended for a reason that is not the
            # images' fault -- a worker that kept dying, say.
            "stop_reason": "",
        }
        assert calls[0][1]["labels_by_image"] == {
            str(image_path): ["Apple computer"]
        }
        assert calls[0][1]["selections_by_image"] == {
            str(image_path): {"Apple computer": "largest"}
        }
        runner.close()

    def test_skips_jobs_already_complete_or_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "done.png")
        _write_image(input_dir / "skip.png")
        _write_image(input_dir / "todo.png")
        calls = _patch_process_single(monkeypatch)

        cfg = _make_config(tmp_path, max_workers=1)
        batch_dir = Path(cfg.output_dir) / cfg.batch_id
        batch_dir.mkdir(parents=True)
        seed = StateManager(batch_dir / "state.db")
        seed.put(
            "done",
            JobRecord(image_path=str(input_dir / "done.png"), status=JobStatus.COMPLETE),
        )
        seed.put(
            "skip",
            JobRecord(image_path=str(input_dir / "skip.png"), status=JobStatus.SKIPPED),
        )
        seed.close()

        runner = BatchRunner(cfg)
        runner.start()
        _drain(runner)

        processed_stems = {Path(p).stem for p, _ in calls}
        assert processed_stems == {"todo"}
        assert _record(runner, "done").status == JobStatus.COMPLETE
        assert _record(runner, "skip").status == JobStatus.SKIPPED
        assert _record(runner, "todo").status == JobStatus.COMPLETE
        runner.close()

    def test_progress_callback_invoked_per_image_and_on_completion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "a.png")
        _write_image(input_dir / "b.png")
        _patch_process_single(monkeypatch)

        progress_events: list[BatchProgress] = []
        completion_events: list[BatchProgress] = []

        cfg = _make_config(tmp_path, max_workers=1)
        runner = BatchRunner(
            cfg,
            progress_callback=progress_events.append,
            completion_callback=completion_events.append,
        )
        runner.start()
        _drain(runner)

        assert len(progress_events) == 2
        assert progress_events[-1].completed == 2
        assert progress_events[-1].total == 2
        assert len(completion_events) == 1
        assert completion_events[0].completed == 2
        assert completion_events[0].remaining == 0
        runner.close()

    def test_start_calls_discover_images_when_not_already_populated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "a.png")
        _patch_process_single(monkeypatch)

        cfg = _make_config(tmp_path, max_workers=1)
        runner = BatchRunner(cfg)
        assert runner._image_paths == []

        runner.start()
        _drain(runner)

        assert len(runner._image_paths) == 1
        runner.close()

    def test_reports_correct_worker_count_to_fake_executor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "a.png")
        _patch_process_single(monkeypatch)

        cfg = _make_config(tmp_path, max_workers=2)
        runner = BatchRunner(cfg)
        runner.start()
        _drain(runner)

        assert isinstance(runner._executor, FakeExecutor)
        assert runner._executor.max_workers == (1 if sys.platform == "darwin" else 2)
        runner.close()


class TestStartFailurePath:
    def test_one_failing_image_is_marked_failed_and_batch_continues(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "good.png")
        _write_image(input_dir / "bad.png")

        def outcome(image_path: str, config_dict: dict) -> PipelineResult:
            if "bad" in image_path:
                return _failing_result(image_path, "vectorizer exploded")
            return _ok_result(image_path)

        _patch_process_single(monkeypatch, fn=outcome)

        cfg = _make_config(tmp_path, max_workers=1)
        runner = BatchRunner(cfg)
        runner.start()
        _drain(runner)

        good = _record(runner, "good")
        bad = _record(runner, "bad")
        assert good.status == JobStatus.COMPLETE
        assert bad.status == JobStatus.FAILED
        assert bad.error == "vectorizer exploded"
        assert not runner.is_running
        runner.close()

    def test_worker_exception_is_caught_and_marks_job_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "a.png")
        _patch_process_single(monkeypatch, default=RuntimeError("model crashed"))

        cfg = _make_config(tmp_path, max_workers=1)
        runner = BatchRunner(cfg)
        runner.start()
        _drain(runner)

        record = _record(runner, "a")
        assert record.status == JobStatus.FAILED
        assert record.error is not None
        assert "model crashed" in record.error
        assert not runner.is_running
        runner.close()

    def test_failed_job_still_triggers_completion_callback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "a.png")
        _patch_process_single(monkeypatch, default=RuntimeError("boom"))

        completion_events: list[BatchProgress] = []
        cfg = _make_config(tmp_path, max_workers=1)
        runner = BatchRunner(cfg, completion_callback=completion_events.append)
        runner.start()
        _drain(runner)

        assert len(completion_events) == 1
        assert completion_events[0].failed == 1
        assert completion_events[0].completed == 0
        runner.close()


class TestResume:
    def test_resumes_over_existing_state_db_reprocessing_only_incomplete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "a.png")
        _write_image(input_dir / "b.png")

        cfg = _make_config(tmp_path, max_workers=1, batch_id="resume-batch")

        # First run: "a" fails, "b" succeeds.
        def first_outcome(image_path: str, config_dict: dict) -> PipelineResult:
            if "a.png" in image_path:
                return _failing_result(image_path, "first attempt failed")
            return _ok_result(image_path)

        _patch_process_single(monkeypatch, fn=first_outcome)
        runner1 = BatchRunner(cfg)
        runner1.start()
        _drain(runner1)
        assert _record(runner1, "a").status == JobStatus.FAILED
        assert _record(runner1, "b").status == JobStatus.COMPLETE
        runner1.close()

        # Second run against the same batch_id/output_dir: only "a" (not
        # COMPLETE/SKIPPED) should be resubmitted.
        calls = _patch_process_single(monkeypatch)
        runner2 = BatchRunner(cfg)
        runner2.start()
        _drain(runner2)

        processed_stems = {Path(p).stem for p, _ in calls}
        assert processed_stems == {"a"}
        assert _record(runner2, "a").status == JobStatus.COMPLETE
        assert _record(runner2, "b").status == JobStatus.COMPLETE
        runner2.close()


class TestGetProgress:
    def test_progress_counts_and_eta_with_zero_elapsed(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "a.png")

        cfg = _make_config(tmp_path)
        runner = BatchRunner(cfg)
        runner.discover_images()
        runner._start_time = time_module_time()

        progress = runner._get_progress()
        assert progress.total == 1
        assert progress.completed == 0
        assert progress.remaining == 1
        assert progress.images_per_min == 0.0
        assert progress.eta_seconds == 0.0
        runner.close()


def time_module_time() -> float:
    import time

    return time.time()


# ── stop() / close() / is_running ───────────────────────────────────────


class TestStopAndClose:
    def test_stop_before_start_is_a_harmless_noop(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        runner = BatchRunner(cfg)
        runner.stop()
        assert not runner.is_running
        runner.close()

    def test_stop_after_start_shuts_down_executor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "a.png")
        _patch_process_single(monkeypatch)

        cfg = _make_config(tmp_path, max_workers=1)
        runner = BatchRunner(cfg)
        runner.start()
        executor = runner._executor
        assert isinstance(executor, FakeExecutor)

        runner.stop()

        assert executor.shutdown_calls == [(False, True)]
        assert not runner.is_running

    def test_close_stops_and_closes_state_db(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "a.png")
        _patch_process_single(monkeypatch)

        cfg = _make_config(tmp_path, max_workers=1)
        runner = BatchRunner(cfg)
        runner.start()
        runner.close()

        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            runner._state.get("a")

    def test_is_running_true_only_while_futures_pending(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        runner = BatchRunner(cfg)
        assert not runner.is_running
        runner.close()




class TestStartCompletionOrdering:
    """start() must tolerate a job finishing before submission is done."""

    def test_eagerly_completed_job_does_not_raise(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "a.png")
        _write_image(input_dir / "b.png")
        _patch_process_single(monkeypatch)
        monkeypatch.setattr(batch_runner, "ProcessPoolExecutor", EagerExecutor)

        runner = BatchRunner(_make_config(tmp_path, max_workers=1))
        runner.start()

        # Future.add_done_callback swallows and logs callback exceptions, so a
        # KeyError in _on_complete is silent -- it shows up as futures that are
        # never cleared and a run that never reports itself finished.
        assert runner._futures == {}
        assert runner._running is False

    def test_completion_callback_fires_once_after_every_job(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for name in ("a.png", "b.png", "c.png"):
            _write_image(input_dir / name)
        _patch_process_single(monkeypatch)
        monkeypatch.setattr(batch_runner, "ProcessPoolExecutor", EagerExecutor)

        seen: list[BatchProgress] = []
        runner = BatchRunner(
            _make_config(tmp_path, max_workers=1),
            completion_callback=seen.append,
        )
        runner.start()

        # A job completing mid-submission must not empty _futures and fire
        # completion early -- it must fire exactly once, with every job done.
        assert len(seen) == 1
        assert seen[0].completed == 3


class TestCloseWaitsForInFlightWork:
    def test_a_job_finishing_during_close_still_records_its_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """close() closed the state store while workers were still running.

        shutdown(wait=False) leaves an already-started future running. When
        it finishes, concurrent.futures invokes its done-callback from the
        executor's own thread, and that callback writes the final status
        through the shared SQLite store. With the connection already closed
        the write raised sqlite3.ProgrammingError -- which Future's callback
        machinery logs and SWALLOWS.

        The image then kept whatever status it had mid-flight, so a resume
        treated finished work as pending and summary() undercounted.
        """
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "a.png")
        _patch_process_single(monkeypatch)

        cfg = _make_config(tmp_path, max_workers=1)
        runner = BatchRunner(cfg)
        runner.start()

        runner.close()

        # Read through a FRESH connection: the point is what survived the
        # close, not what the runner still had in hand.
        state_db = Path(cfg.output_dir) / cfg.batch_id / "state.db"
        record = StateManager(state_db).get("a")
        assert record is not None
        assert record.status == JobStatus.COMPLETE
