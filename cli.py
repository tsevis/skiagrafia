"""skiagrafia — batch from a command line.

The application had only a GUI, so a 376-page book had to be driven by a
script written for the occasion. That script carried four faults before it
ran correctly, and each one is closed here rather than left for the next
person to rediscover:

  * an empty confirmed-label list is NOT None. The interrogator reads it as
    "the user confirmed these, and there are none" and returns nothing
    without looking at an image, at a hundred images a second;
  * an empty frozen image list used to mean the whole folder, so a run that
    found no labels processed everything blind;
  * `BatchRunner.is_running` is a property;
  * `summary()` after `close()` reads a database that has been shut.

Long runs belong in a session of their own. `nohup` only ignores SIGHUP:
when the terminal is torn down as a process group the job goes with it,
which is how one run died at 134 of 376 pages. Use `--detach`.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.batch_runner import BatchConfig, BatchRunSummary

logger = logging.getLogger("skiagrafia")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
#: Exit codes a caller can branch on.
EXIT_OK = 0
EXIT_RUN_HAD_FAILURES = 1
EXIT_NOTHING_TO_DO = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skiagrafia",
        description="Separate every image in a folder into layers.",
        epilog=(
            "Runs resume: images already finished in the destination are "
            "skipped, so re-running after an interruption picks up where it "
            "stopped."
        ),
    )
    parser.add_argument("input", type=Path, help="folder of images to process")
    parser.add_argument(
        "--output", "-o", type=Path, required=True,
        help="where finished work is written",
    )
    parser.add_argument(
        "--labels", "-l", default="",
        help=(
            "comma-separated labels to separate. Omit to ask the vision "
            "model what is in each image."
        ),
    )
    parser.add_argument(
        "--guide", "-g", type=Path, default=None,
        help="a Domain Guide (.toml) naming the objects and their terms",
    )
    parser.add_argument(
        "--output-mode", default="vector+bitmap",
        help="what a run produces (default: vector+bitmap)",
    )
    parser.add_argument(
        "--limit", type=int, default=0, metavar="N",
        help=(
            "process only the first N images. A rehearsal on a few pages "
            "costs minutes and is how most faults are found."
        ),
    )
    parser.add_argument(
        "--batch-id", default="batch",
        help="names the run's folder inside the destination (default: batch)",
    )
    parser.add_argument(
        "--detach", action="store_true",
        help=(
            "run in a session of its own, surviving the terminal that "
            "started it, and print the process id"
        ),
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="errors only")
    return parser


def images_in(folder: Path, limit: int = 0) -> list[Path]:
    """Every supported image directly inside `folder`, in a stable order."""
    found = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    return found[:limit] if limit > 0 else found


def plan(args: Any, prefs: dict[str, Any]) -> BatchConfig:
    """Turn the arguments into a run, or exit saying what is wrong.

    Checked here, before a single model is loaded, because the alternative
    is discovering an empty folder after a minute of imports.
    """
    from core.batch_runner import BatchConfig

    if not args.input.is_dir():
        raise SystemExit(f"skiagrafia: not a folder: {args.input}")
    paths = images_in(args.input, args.limit)
    if not paths:
        raise SystemExit(
            f"skiagrafia: no supported images directly inside {args.input}. "
            f"Looked for: {', '.join(sorted(IMAGE_SUFFIXES))}"
        )
    if args.guide is not None and not args.guide.is_file():
        raise SystemExit(f"skiagrafia: no such guide: {args.guide}")

    labels = [part.strip() for part in args.labels.split(",") if part.strip()]
    return BatchConfig(
        batch_id=args.batch_id,
        input_folder=str(args.input),
        output_dir=str(args.output),
        confirmed_labels=labels,
        input_images=[str(path) for path in paths],
        output_mode=args.output_mode,
        guide_path=str(args.guide) if args.guide else None,
        vlm_backend=prefs.get("vlm_backend", "ollama"),
        ollama_url=prefs.get("ollama_url", "http://localhost:11434"),
        ollama_model=prefs.get("ollama_model", "qwen2.5vl:3b"),
        llamacpp_url=prefs.get("llamacpp_url", "http://localhost:8080"),
        llamacpp_model=prefs.get("llamacpp_model", "Qwen3-VL-8B-Instruct"),
        models_directory=str(prefs.get("models_directory", "")),
        segmentation_backend=prefs.get("segmentation_backend", "auto"),
        local_primary_model=prefs.get("local_primary_model", "Qwen3-VL-8B-Instruct"),
        local_fallback_model=prefs.get("local_fallback_model", "gemma-4-12B-it"),
        quality_profile=prefs.get("quality_profile", "balanced"),
    )


def interrogate(
    paths: list[Path], prefs: dict[str, Any], guide: Path | None
) -> dict[str, list[str]]:
    """Ask the vision model what is in each image.

    `confirmed_labels=None`, never `[]`: an empty list means "the user
    confirmed these and there are none", and the interrogator honours it by
    returning nothing without looking at the image.
    """
    from core.factory import build_interrogation_settings
    from core.interrogation import GuidedInterrogator
    from core.knowledge import KnowledgePack
    from processors.source_image import detection_image, load_source_image

    pack = KnowledgePack.load(guide) if guide else None
    interrogator = GuidedInterrogator(build_interrogation_settings(prefs, overrides={}))
    labels_by_image: dict[str, list[str]] = {}
    refused: dict[str, int] = {}
    started = time.monotonic()

    for index, path in enumerate(paths, start=1):
        try:
            rgb, alpha, _ = load_source_image(str(path))
            detected = interrogator.interrogate(
                detection_image(rgb, alpha), confirmed_labels=None, knowledge_pack=pack
            )
            labels_by_image[str(path)] = [c.display_label for c in detected.candidates]
            for term in detected.labels_outside_vocabulary:
                refused[term] = refused.get(term, 0) + 1
        except Exception as exc:
            logger.warning("could not read %s: %s", path.name, exc)
            labels_by_image[str(path)] = []
        rate = index / (time.monotonic() - started or 1)
        logger.info(
            "read %d/%d  %s  %s", index, len(paths), path.name,
            ", ".join(labels_by_image[str(path)]) or "nothing",
        )
        if index == len(paths):
            logger.info("read %d images at %.2f a second", index, rate)

    if refused:
        ranked = sorted(refused.items(), key=lambda kv: -kv[1])
        logger.info(
            "the guide does not list %d terms the model proposed: %s",
            len(ranked), ", ".join(f"{t} ({n})" for t, n in ranked[:10]),
        )
    return labels_by_image


def execute(config: BatchConfig, quiet: bool) -> BatchRunSummary:
    """Run the batch to completion and return what it produced."""
    from core.batch_runner import BatchProgress, BatchRunner

    def _report(progress: BatchProgress) -> None:
        if quiet:
            return
        logger.info(
            "%d/%d done, %d failed, %.1f a minute, %.0f min left",
            progress.completed, progress.total, progress.failed,
            progress.images_per_min, progress.eta_seconds / 60,
        )

    runner = BatchRunner(config, progress_callback=_report)
    runner.discover_images()
    runner.start()
    while runner.is_running:  # a property, not a method
        time.sleep(2)
    summary = runner.summary()  # before close(): it reads the state database
    runner.close()
    return summary


def _detach() -> None:
    """Put this process in a session of its own and carry on.

    `nohup` only ignores SIGHUP. A terminal torn down as a process group
    takes its jobs with it, which killed one run at 134 of 376 pages.
    """
    if os.fork():
        os._exit(EXIT_OK)
    os.setsid()
    if os.fork():
        os._exit(EXIT_OK)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    from utils.preferences import load_preferences

    prefs = load_preferences()
    config = plan(args, prefs)
    paths = [Path(p) for p in (config.input_images or [])]

    if args.detach:
        _detach()
        logger.info("running detached as pid %d", os.getpid())

    args.output.mkdir(parents=True, exist_ok=True)
    if not config.confirmed_labels:
        labels_by_image = interrogate(paths, prefs, args.guide)
        (args.output / "_labels.json").write_text(json.dumps(labels_by_image, indent=2))
        usable = [p for p in paths if labels_by_image.get(str(p))]
        if not usable:
            logger.error(
                "no image got a label, so there is nothing to separate. "
                "Name labels with --labels, or check that the vision model "
                "is reachable."
            )
            return EXIT_NOTHING_TO_DO
        # Frozen to what actually has labels. An empty list here used to
        # mean the whole folder.
        config = config.model_copy(update={
            "input_images": [str(p) for p in usable],
            "labels_by_image": {k: v for k, v in labels_by_image.items() if v},
            "confirmed_labels": sorted({lab for v in labels_by_image.values() for lab in v}),
        })
        logger.info("%d of %d images have at least one label", len(usable), len(paths))

    started = time.monotonic()
    summary = execute(config, args.quiet)
    logger.info(
        "%d of %d complete, %d failed, %.1f layers an image, %.1f min",
        summary.completed, summary.total, summary.failed,
        summary.avg_layers, (time.monotonic() - started) / 60,
    )
    if summary.stop_reason:
        logger.error("%s", summary.stop_reason)
    for path in summary.failed_image_paths[:10]:
        logger.error("failed: %s", path)
    return EXIT_RUN_HAD_FAILURES if summary.failed else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
