"""Offline tests for immutable Batch-mode run artifacts."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.batch_session import (
    BatchInterrogationSnapshot,
    BatchRunSettings,
    BatchTriageSnapshot,
    capture_guide,
    load_interrogation_snapshot,
    load_run_settings,
    load_triage_snapshot,
    write_snapshot,
)


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
