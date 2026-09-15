"""Offline tests for immutable Batch-mode run artifacts."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.batch_session import (
    BatchInterrogationSnapshot,
    BatchProcessingSnapshot,
    BatchRunSettings,
    BatchTriageSnapshot,
    capture_guide,
    load_interrogation_snapshot,
    load_resumable_batch,
    load_run_settings,
    load_triage_snapshot,
    materialize_frozen_guide,
    triage_labels_for_image,
    write_processing_snapshot,
    write_snapshot,
)
from core.batch_runner import BatchConfig
from core.state_manager import JobRecord, JobStatus, StateManager


def test_capture_guide_preserves_content_and_fingerprint(tmp_path: Path) -> None:
    guide = tmp_path / "apple.toml"
    guide.write_text("[domain]\nname = 'Apple'\n", encoding="utf-8")

    name, fingerprint, contents = capture_guide(str(guide))

    assert name == "apple"
    assert fingerprint is not None and len(fingerprint) == 64
    assert contents == "[domain]\nname = 'Apple'\n"


def test_run_snapshots_round_trip_request_guide_candidates_and_triage(tmp_path: Path) -> None:
    run = BatchRunSettings(
        batch_id="apple-run",
        input_folder="/input/apple",
        output_directory=str(tmp_path / "output"),
        selection_request="Select hardware.\nExclude captions.",
        guide_path="/guides/apple.toml",
        guide_name="Apple — The First 50 Years",
        guide_fingerprint="a" * 64,
        guide_toml="[domain]\nname = 'Apple'\n",
    )
    write_snapshot(run.run_dir / "run.json", run)
    write_snapshot(
        run.run_dir / "interrogation.json",
        BatchInterrogationSnapshot(
            batch_id=run.batch_id,
            selection_request=run.selection_request,
            candidates_by_image={
                "/input/apple/001.jpg": [
                    {"canonical_label": "iMac", "selection": "largest"}
                ]
            },
        ),
    )
    write_snapshot(
        run.run_dir / "triage.json",
        BatchTriageSnapshot(
            batch_id=run.batch_id,
            selection_request=run.selection_request,
            approved_labels=["iMac"],
            excluded_labels_by_image={"/input/apple/002.jpg": ["iMac"]},
        ),
    )

    restored = load_run_settings(run.run_dir / "run.json")
    candidates = load_interrogation_snapshot(run.run_dir / "interrogation.json")
    triage = load_triage_snapshot(run.run_dir / "triage.json")

    assert restored.selection_request == run.selection_request
    assert restored.guide_fingerprint == "a" * 64
    assert restored.guide_toml == run.guide_toml
    assert candidates.candidates_by_image["/input/apple/001.jpg"][0]["selection"] == "largest"
    assert triage.approved_labels == ["iMac"]
    assert triage.excluded_labels_by_image == {"/input/apple/002.jpg": ["iMac"]}


def test_frozen_guide_is_created_once_without_overwriting_different_content(
    tmp_path: Path,
) -> None:
    run = BatchRunSettings(
        batch_id="guide-run",
        output_directory=str(tmp_path / "output"),
        guide_toml="[domain]\nname = 'Apple'\n",
    )

    target = materialize_frozen_guide(run)

    assert target is not None
    assert target.read_text(encoding="utf-8") == run.guide_toml
    target.write_text("[domain]\nname = 'Different'\n", encoding="utf-8")
    try:
        materialize_frozen_guide(run)
    except FileExistsError:
        pass
    else:  # pragma: no cover - keeps a destructive regression explicit
        raise AssertionError("A different existing guide must never be overwritten")


def test_load_resumable_batch_restores_only_coherent_frozen_triage_state(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    first = input_dir / "one.png"
    second = input_dir / "two.png"
    first.write_bytes(b"not decoded by this metadata-only check")
    second.write_bytes(b"not decoded by this metadata-only check")
    output_dir = tmp_path / "output"
    run = BatchRunSettings(
        batch_id="resume-run",
        input_folder=str(input_dir),
        output_directory=str(output_dir),
        selection_request="Select Apple computers.",
        interrogation_settings={"output_mode": "vector+bitmap"},
    )
    candidates = {
        str(first): [{"canonical_label": "Apple computer", "selection": "largest"}],
        str(second): [{"canonical_label": "Apple computer", "selection": "all"}],
    }
    triage = BatchTriageSnapshot(
        batch_id=run.batch_id,
        selection_request=run.selection_request,
        approved_labels=["Apple computer"],
        excluded_labels_by_image={str(second): ["Apple computer"]},
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
        input_images=[str(first), str(second)],
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
    state_db = run.run_dir / "state.db"
    assert load_resumable_batch(run.run_dir) is None
    assert not state_db.exists()

    all_objects = run.run_dir / "one_all-objects.tiff"
    state = StateManager(run.run_dir / "state.db")
    state.put(
        "one",
        JobRecord(
            image_path=str(first),
            status=JobStatus.COMPLETE,
            output_all_objects_tiff=str(all_objects),
            layer_count=3,
        ),
    )
    state.put("two", JobRecord(image_path=str(second), status=JobStatus.FAILED))
    state.close()

    assert load_resumable_batch(run.run_dir) is None
    all_objects.write_bytes(b"verified only for durable path presence")

    resumable = load_resumable_batch(run.run_dir)

    assert resumable is not None
    assert resumable.triage.excluded_labels_by_image == triage.excluded_labels_by_image
    assert resumable.processing.config["labels_by_image"] == {
        str(first): ["Apple computer"],
        str(second): [],
    }
    assert resumable.processing.config["selections_by_image"][str(first)] == {
        "Apple computer": "largest"
    }


def test_bare_or_mismatched_state_never_becomes_a_gui_resume(tmp_path: Path) -> None:
    state = StateManager(tmp_path / "legacy" / "state.db")
    state.put("pending", JobRecord(image_path="/input/a.png", status=JobStatus.PENDING))
    state.close()

    assert load_resumable_batch(tmp_path / "legacy") is None
