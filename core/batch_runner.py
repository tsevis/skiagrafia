from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path

from pydantic import BaseModel, Field

from core.factory import build_capabilities, build_knowledge_pack
from core.orchestrator import Orchestrator, PipelineResult
from core.state_manager import JobRecord, JobStatus, StateManager

logger = logging.getLogger(__name__)

_BATCH_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class BatchConfig(BaseModel):
    """Configuration for a batch run."""

    batch_id: str = ""
    input_folder: str
    output_dir: str
    confirmed_labels: list[str]
    confirmed_children: dict[str, list[str]] = {}
    output_mode: str = "vector+bitmap"
    corner_threshold: int = 60
    speckle: int = 8
    length_threshold: float = 4.0
    vtracer_quality: str = "balanced"
    vlm_backend: str = "ollama"  # "ollama" | "llamacpp"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5vl:3b"
    llamacpp_url: str = "http://localhost:8080"
    llamacpp_model: str = "Qwen3-VL-8B-Instruct"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    bilateral_d: int = 9
    max_workers: int = 0  # 0 = os.cpu_count()
    guide_path: str | None = None
    interrogation_profile: str = "balanced"
    fallback_mode: str = "adaptive_auto"
    preferred_vlm: str | None = None
    fallback_vlms: list[str] = Field(default_factory=lambda: ["gemma4:e4b", "minicpm-v"])
    text_reasoner_model: str = "gemma4:e4b"
    enable_tiled_fallback: bool = True
    max_aliases_per_object: int = 4
    models_directory: str = ""
    segmentation_backend: str = "auto"
    sam3_confidence: float = 0.2
    local_primary_model: str = "Qwen3-VL-8B-Instruct"
    local_fallback_model: str = "gemma-4-12B-it"
    quality_profile: str = "balanced"
    preserve_path_detail: bool = True
    object_prompt: str = ""
    selection_request: str = ""
    discover_parts: bool = True
    # GUI batches freeze this exact list after interrogation.  A resume must
    # never discover a newly-added folder file that did not pass Triage.
    input_images: list[str] | None = None
    # The human-approved labels and instance policies are image-specific.
    # Keeping them in the worker config preserves GUI Triage semantics when
    # BatchRunner resumes an interrupted run.
    labels_by_image: dict[str, list[str]] = Field(default_factory=dict)
    selections_by_image: dict[str, dict[str, str]] = Field(default_factory=dict)

    def model_post_init(self, __context: object) -> None:
        if not self.batch_id:
            self.batch_id = uuid.uuid4().hex[:12]
        elif not _BATCH_ID_RE.fullmatch(self.batch_id):
            raise ValueError("Batch ID must use only letters, numbers, underscores, or hyphens.")
        if self.max_workers <= 0:
            self.max_workers = os.cpu_count() or 4


class BatchProgress(BaseModel):
    """Progress snapshot for UI updates."""

    total: int
    completed: int
    failed: int
    remaining: int
    images_per_min: float
    current_image: str = ""
    eta_seconds: float = 0.0


class BatchRunSummary(BaseModel):
    """Persisted output metrics reconstructed from successful state records."""

    total: int = 0
    completed: int = 0
    failed: int = 0
    svg_count: int = 0
    all_objects_count: int = 0
    avg_layers: float = 0.0
    failed_image_paths: list[str] = Field(default_factory=list)
    status_by_image: dict[str, str] = Field(default_factory=dict)


def _process_single(
    image_path: str,
    config_dict: dict,
) -> PipelineResult:
    """Worker function -- runs in a separate process.

    Rebuilds model clients from scratch in each worker process (model weights
    cannot be shared across process boundaries).
    """
    labels = config_dict.get("labels_by_image", {}).get(
        image_path, config_dict.get("confirmed_labels")
    )
    # Checked before a worker builds its models: with an empty list the
    # interrogator reports "user-confirmed labels" and returns none of them,
    # and the run goes on to write an unlabelled all-objects file.
    if not labels:
        raise ValueError(
            f"No labels were confirmed for {Path(image_path).name}, so there is "
            "nothing to separate. Interrogate the batch, or confirm a label."
        )
    orchestrator = _worker_orchestrator(json.dumps(config_dict, sort_keys=True))
    selections = config_dict.get("selections_by_image", {}).get(image_path, {})
    orchestrator.set_confirmed_selections(selections)
    return orchestrator.process(image_path, labels)


@functools.lru_cache(maxsize=1)
def _worker_orchestrator(config_json: str) -> Orchestrator:
    """Keep model weights resident for successive images in this worker."""
    config = BatchConfig.model_validate_json(config_json)

    # Build prefs-like dict from BatchConfig for the factory
    prefs_from_config: dict = {
        **config.model_dump(),
        "vlm_backend": config.vlm_backend,
        "ollama_url": config.ollama_url,
        "ollama_model": config.ollama_model,
        "llamacpp_url": config.llamacpp_url,
        "llamacpp_model": config.llamacpp_model,
        "preferred_fallback_vlm": config.fallback_vlms[0] if config.fallback_vlms else "gemma4:e4b",
        "preferred_text_reasoner": config.text_reasoner_model,
        "interrogation_profile": config.interrogation_profile,
        "interrogation_fallback_mode": config.fallback_mode,
        "enable_tiled_fallback": config.enable_tiled_fallback,
        "max_aliases_per_object": config.max_aliases_per_object,
        "vtracer_corner_threshold": config.corner_threshold,
        "vtracer_speckle": config.speckle,
        "vtracer_length_threshold": config.length_threshold,
    }

    caps = build_capabilities(
        prefs_from_config,
        corner_threshold=config.corner_threshold,
        length_threshold=config.length_threshold,
        filter_speckle=config.speckle,
        knowledge_pack_path=config.guide_path,
        interrogation_overrides={
            "preferred_vlm": config.preferred_vlm,
            "selection_request": config.selection_request,
        },
    )
    return Orchestrator(
        capabilities=caps,
        quality=config.quality_profile,
        output_dir=Path(config.output_dir) / config.batch_id,
        output_mode=config.output_mode,
        bilateral_d=config.bilateral_d,
        box_threshold=config.box_threshold,
        text_threshold=config.text_threshold,
        knowledge_pack=build_knowledge_pack(config.guide_path),
    )


class BatchRunner:
    """ProcessPoolExecutor batch coordinator.

    Manages parallel processing of images with state persistence.
    """

    def __init__(
        self,
        config: BatchConfig,
        progress_callback: Callable[[BatchProgress], None] | None = None,
        completion_callback: Callable[[BatchProgress], None] | None = None,
        job_callback: Callable[[str, str, JobStatus], None] | None = None,
    ) -> None:
        self._config = config
        self._progress_cb = progress_callback
        self._completion_cb = completion_callback
        self._job_cb = job_callback

        batch_dir = Path(config.output_dir) / config.batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        self._state = StateManager(batch_dir / "state.db")

        self._image_paths: list[Path] = []
        self._image_ids: dict[str, str] = {}
        self._start_time: float = 0.0
        self._executor: ProcessPoolExecutor | None = None
        self._futures: dict[str, Future] = {}
        self._running = False
        self._closed_summary: BatchRunSummary | None = None

    def discover_images(self) -> list[Path]:
        """Scan input folder for supported image files."""
        folder = Path(self._config.input_folder)
        extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
        if self._config.input_images is not None:
            if not self._config.input_images:
                raise ValueError(
                    "This batch was frozen to no images. Triage kept none of them, "
                    "which is not the same as keeping all of them."
                )
            self._image_paths = [Path(path) for path in self._config.input_images]
            invalid = [
                path
                for path in self._image_paths
                if not path.is_file() or path.suffix.lower() not in extensions
            ]
            if invalid:
                raise ValueError(
                    "Frozen batch inputs are missing or are no longer supported images."
                )
        else:
            self._image_paths = sorted(
                p for p in folder.iterdir()
                if p.suffix.lower() in extensions and p.is_file()
            )
        logger.info("Discovered %d images in %s", len(self._image_paths), folder)

        stem_counts: dict[str, int] = {}
        for image_path in self._image_paths:
            stem_counts[image_path.stem] = stem_counts.get(image_path.stem, 0) + 1

        # Initialise state records for new images
        for img_path in self._image_paths:
            image_id = self._state_id_for(img_path, stem_counts)
            self._image_ids[str(img_path)] = image_id
            if self._state.get(image_id) is None:
                self._state.put(
                    image_id,
                    JobRecord(
                        image_path=str(img_path),
                        labels=list(self._config.labels_by_image.get(str(img_path), [])),
                    ),
                )
        return self._image_paths

    def _state_id_for(self, image_path: Path, stem_counts: dict[str, int]) -> str:
        """Keep legacy simple IDs unless same-stem source files need separation."""
        if stem_counts[image_path.stem] == 1:
            return image_path.stem
        digest = hashlib.sha256(str(image_path).encode("utf-8")).hexdigest()[:10]
        return f"{image_path.stem}-{digest}"

    def start(self) -> None:
        """Launch batch processing with ProcessPoolExecutor."""
        if not self._image_paths:
            self.discover_images()

        self._running = True
        self._start_time = time.time()
        config_dict = self._config.model_dump()

        # The model stages share one GPU; parallel model replicas multiply
        # memory and contend for Metal. Reuse one warm worker on this path.
        workers = 1 if sys.platform == "darwin" or self._config.vlm_backend == "local" else self._config.max_workers
        self._executor = ProcessPoolExecutor(max_workers=workers)

        for img_path in self._image_paths:
            image_id = self._image_ids[str(img_path)]
            record = self._state.get(image_id)
            if record and record.status in (JobStatus.COMPLETE, JobStatus.SKIPPED):
                continue

            self._state.update_status(image_id, JobStatus.MASKING)
            self._emit_job(image_id, str(img_path), JobStatus.MASKING)
            future = self._executor.submit(
                _process_single, str(img_path), config_dict
            )
            self._futures[image_id] = future

        logger.info(
            "Batch started: %d images, %d workers",
            len(self._futures),
            workers,
        )

        # Callbacks are attached only once every future is tracked. A job that
        # finishes while the loop is still submitting would otherwise fire
        # _on_complete against a half-filled _futures -- clearing the last
        # entry and reporting the whole batch finished before it had started.
        submitted_any = bool(self._futures)
        for image_id, future in list(self._futures.items()):
            future.add_done_callback(
                functools.partial(self._on_complete, image_id)
            )

        if not submitted_any:
            self._running = False
            progress = self._get_progress()
            if self._progress_cb:
                self._progress_cb(progress)
            if self._completion_cb:
                self._completion_cb(progress)

    def _on_complete(self, image_id: str, future: Future) -> None:
        """Handle completion of a single image."""
        try:
            result = future.result()
            # Kept for both outcomes: a run that errored late can still have
            # said something useful about what it managed first.
            warnings = [str(warning) for warning in getattr(result, "warnings", [])]
            record = self._state.get(image_id)
            if result.error:
                if record:
                    self._state.put(image_id, record.model_copy(update={
                        "status": JobStatus.FAILED,
                        "error": result.error,
                        "warnings": warnings,
                    }))
                else:
                    self._state.update_status(
                        image_id, JobStatus.FAILED, error=result.error
                    )
            elif record:
                updated = record.model_copy(
                    update={
                        "status": JobStatus.COMPLETE,
                        "output_svg": result.svg_path,
                        "output_tiff": result.tiff_path,
                        "output_all_objects_tiff": result.all_objects_tiff_path,
                        "layer_count": len(result.layers),
                        "warnings": warnings,
                    }
                )
                self._state.put(image_id, updated)
        except Exception as exc:
            self._state.update_status(
                image_id, JobStatus.FAILED, error=str(exc)
            )
            logger.exception("Image %s failed: %s", image_id, exc)

        self._futures.pop(image_id, None)

        record = self._state.get(image_id)
        if record is not None:
            self._emit_job(image_id, record.image_path, record.status)

        progress = self._get_progress()
        if self._progress_cb:
            self._progress_cb(progress)

        if not self._futures:
            self._running = False
            if self._completion_cb:
                self._completion_cb(progress)

    def _emit_job(self, image_id: str, image_path: str, status: JobStatus) -> None:
        if self._job_cb is None:
            return
        try:
            self._job_cb(image_id, image_path, status)
        except Exception:
            logger.exception("Batch job callback failed for %s", image_id)

    def _get_progress(self) -> BatchProgress:
        counts = self._state.count_by_status()
        total = len(self._image_paths)
        completed = counts.get(JobStatus.COMPLETE, 0)
        failed = counts.get(JobStatus.FAILED, 0)
        done = completed + failed + counts.get(JobStatus.SKIPPED, 0)
        remaining = total - done

        elapsed = time.time() - self._start_time
        rate = done / elapsed * 60 if elapsed > 0 else 0.0
        eta = remaining / (rate / 60) if rate > 0 else 0.0

        return BatchProgress(
            total=total,
            completed=completed,
            failed=failed,
            remaining=remaining,
            images_per_min=round(rate, 1),
            eta_seconds=round(eta, 1),
        )

    def stop(self) -> None:
        """Stop batch processing gracefully.

        Cancels what has not started and returns immediately, so a UI stays
        responsive. Work already running in a worker is NOT waited for --
        close() does that.
        """
        self._running = False
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
            logger.info("Batch stopped")

    def close(self) -> None:
        """Shut down the executor and close the state store, in that order.

        WAITS for in-flight work before closing the store. A future that is
        already running is not cancelled by stop(), and when it finishes
        concurrent.futures invokes its done-callback from the executor's own
        thread -- that callback writes the job's final status through this
        shared SQLite connection. Closing the connection first made every
        such write raise sqlite3.ProgrammingError, which Future's callback
        machinery logs and SWALLOWS: the image kept whatever status it had
        mid-flight, so a later resume treated finished work as pending and
        summary() undercounted it.
        """
        self.stop()
        if self._executor:
            self._executor.shutdown(wait=True)
        # Taken before the store shuts: reading the result of a finished run
        # is the ordinary next thing to do, and the records are only in the
        # database this is about to close.
        self._closed_summary = self.summary()
        self._state.close()

    def summary(self) -> BatchRunSummary:
        """Reconstruct output metrics and failed paths from durable state."""
        if self._closed_summary is not None:
            return self._closed_summary
        records = self._state.all_records()
        completed_records = [
            record for record in records.values() if record.status == JobStatus.COMPLETE
        ]
        failed_records = [
            record for record in records.values() if record.status == JobStatus.FAILED
        ]
        total_layers = sum(record.layer_count for record in completed_records)
        return BatchRunSummary(
            total=len(records),
            completed=len(completed_records),
            failed=len(failed_records),
            svg_count=sum(record.output_svg is not None for record in completed_records),
            all_objects_count=sum(
                record.output_all_objects_tiff is not None
                for record in completed_records
            ),
            avg_layers=(total_layers / len(completed_records)) if completed_records else 0.0,
            failed_image_paths=sorted(record.image_path for record in failed_records),
            status_by_image={
                record.image_path: record.status.value for record in records.values()
            },
        )

    @property
    def is_running(self) -> bool:
        return self._running
