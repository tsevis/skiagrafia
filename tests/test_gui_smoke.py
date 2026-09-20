"""Fast real-window smoke coverage for the Batch wizard.

This is intentionally the only module collected by ``pytest -m gui``.  The
complete windowed suite remains available through ``gui or gui_integration``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if TYPE_CHECKING:
    from ui.main_window import MainWindow


def _write_image(path: Path) -> Path:
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(path)
    return path


@pytest.fixture
def batch_smoke_view(tk_root, tmp_path):
    """A real BatchView whose reads and writes stay inside tmp_path."""
    from tkinter import ttk

    from ui.batch.batch_view import BatchView

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    app = SimpleNamespace(
        root=tk_root,
        prefs={"output_directory": str(output_dir)},
        switch_to_batch=lambda *args, **kwargs: None,
    )
    container = ttk.Frame(tk_root)
    container.pack(fill="both", expand=True)
    # `app` is a SimpleNamespace exposing only the MainWindow surface BatchView
    # actually touches (root/prefs/switch_to_batch); cast for the type checker.
    view = BatchView(container, cast("MainWindow", app))
    view.frame.pack(fill="both", expand=True)
    tk_root.update()
    return view


@pytest.mark.gui
def test_verified_batch_resume_restores_triage_before_starting_runner(
    batch_smoke_view, tk_root, tmp_path, monkeypatch
) -> None:
    from core.batch_runner import BatchConfig
    from core.batch_session import (
        BatchInterrogationSnapshot,
        BatchProcessingSnapshot,
        BatchRunSettings,
        BatchTriageSnapshot,
        triage_labels_for_image,
        write_processing_snapshot,
        write_snapshot,
    )
    from core.state_manager import JobRecord, JobStatus, StateManager
    from ui.batch.steps.step_progress import StepProgress

    assert batch_smoke_view.current_step == 0

    input_dir = tmp_path / "resume-input"
    input_dir.mkdir()
    complete = _write_image(input_dir / "complete.png")
    pending = _write_image(input_dir / "pending.png")
    output_dir = Path(batch_smoke_view.app.prefs["output_directory"])
    run = BatchRunSettings(
        batch_id="gui-resume",
        input_folder=str(input_dir),
        output_directory=str(output_dir),
        selection_request="Select Apple computers.",
        interrogation_settings={"output_mode": "vector+bitmap"},
    )
    candidates = {
        str(complete): [{"canonical_label": "Apple computer", "selection": "largest"}],
        str(pending): [{"canonical_label": "Apple computer", "selection": "all"}],
    }
    triage = BatchTriageSnapshot(
        batch_id=run.batch_id,
        selection_request=run.selection_request,
        approved_labels=["Apple computer"],
        excluded_labels_by_image={str(pending): ["Apple computer"]},
    )
    labels_by_image: dict[str, list[str]] = {}
    selections_by_image: dict[str, dict[str, str]] = {}
    for image_path, image_candidates in candidates.items():
        labels, selections = triage_labels_for_image(
            image_candidates,
            triage.approved_labels,
            triage.excluded_labels_by_image.get(image_path, []),
        )
        labels_by_image[image_path] = labels
        selections_by_image[image_path] = selections
    config = BatchConfig(
        batch_id=run.batch_id,
        input_folder=run.input_folder,
        output_dir=run.output_directory,
        confirmed_labels=triage.approved_labels,
        input_images=[str(complete), str(pending)],
        labels_by_image=labels_by_image,
        selections_by_image=selections_by_image,
    )
    write_snapshot(run.run_dir / "run.json", run)
    write_snapshot(
        run.run_dir / "interrogation.json",
        BatchInterrogationSnapshot(
            batch_id=run.batch_id,
            selection_request=run.selection_request,
            candidates_by_image=candidates,
        ),
    )
    write_snapshot(run.run_dir / "triage.json", triage)
    write_processing_snapshot(
        run.run_dir / "processing.json",
        BatchProcessingSnapshot(batch_id=run.batch_id, config=config.model_dump()),
    )
    all_objects = run.run_dir / "complete_all-objects.tiff"
    all_objects.write_bytes(b"durable output marker")
    state = StateManager(run.run_dir / "state.db")
    state.put(
        "complete",
        JobRecord(
            image_path=str(complete),
            status=JobStatus.COMPLETE,
            output_all_objects_tiff=str(all_objects),
        ),
    )
    state.put("pending", JobRecord(image_path=str(pending), status=JobStatus.FAILED))
    state.close()

    calls: list[str] = []
    monkeypatch.setattr(StepProgress, "resume_existing", lambda self: calls.append("resume"))
    step = batch_smoke_view._step_views[0]
    step._scan_recent_batches()
    item = next(
        item_id
        for item_id in step._recent_list.get_children()
        if step._recent_list.item(item_id, "values")[1] == "Resume ready"
    )
    step._recent_list.selection_set(item)
    step._update_resume_button()

    assert str(step._resume_btn.cget("state")) == "normal"
    step._resume_selected_run()
    tk_root.update()

    assert calls == ["resume"]
    assert batch_smoke_view.current_step == 4
    assert batch_smoke_view.run_settings.batch_id == "gui-resume"
    assert batch_smoke_view.confirmed_labels == ["Apple computer"]
    assert batch_smoke_view.excluded_labels_by_image == {
        str(pending): ["Apple computer"]
    }
    assert batch_smoke_view.interrogation_records == candidates
