from __future__ import annotations

import functools
import json
import sys
import logging
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, Future
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from core.factory import build_capabilities, build_knowledge_pack
from core.orchestrator import Orchestrator, PipelineResult
from core.state_manager import JobRecord, JobStatus, StateManager

logger = logging.getLogger(__name__)


class BatchConfig(BaseModel):
    """Configuration for a batch run."""

    batch_id: str = ""
    input_folder: str
    output_dir: str
    confirmed_labels: list[str]
    confirmed_children: dict[str, list[str]] = {}
    output_mode: str = "vector+bitmap"
    recursion_depth: int = 2
    corner_threshold: int = 60
    speckle: int = 8
    smoothing: int = 5
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
    sam3_confidence: float = 0.5
    local_primary_model: str = "Qwen3-VL-8B-Instruct"
    local_fallback_model: str = "gemma-4-12B-it"
    quality_profile: str = "balanced"
    preserve_path_detail: bool = True
    object_prompt: str = ""
    selection_request: str = ""
    discover_parts: bool = True

    def model_post_init(self, __context: object) -> None:
        if not self.batch_id:
            self.batch_id = uuid.uuid4().hex[:12]
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


def _process_single(
    image_path: str,
    config_dict: dict,
) -> PipelineResult:
    """Worker function -- runs in a separate process.

    Rebuilds model clients from scratch in each worker process (model weights
    cannot be shared across process boundaries).
    """
    orchestrator = _worker_orchestrator(json.dumps(config_dict, sort_keys=True))
    return orchestrator.process(image_path, config_dict.get("confirmed_labels"))


@functools.lru_cache(maxsize=1)
def _worker_orchestrator(config_json: str):
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
    orchestrator = Orchestrator(
        capabilities=caps,
        quality=config.quality_profile,
        output_dir=Path(config.output_dir) / config.batch_id,
        output_mode=config.output_mode,
        bilateral_d=config.bilateral_d,
        box_threshold=config.box_threshold,
        text_threshold=config.text_threshold,
        knowledge_pack=build_knowledge_pack(config.guide_path),
    )
    return orchestrator


class BatchRunner:
    """ProcessPoolExecutor batch coordinator.

    Manages parallel processing of images with state persistence.
    """

    def __init__(
        self,
        config: BatchConfig,
        progress_callback: Callable[[BatchProgress], None] | None = None,
        completion_callback: Callable[[BatchProgress], None] | None = None,
    ) -> None:
        self._config = config
        self._progress_cb = progress_callback
        self._completion_cb = completion_callback

        batch_dir = Path(config.output_dir) / config.batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        self._state = StateManager(batch_dir / "state.db")

        self._image_paths: list[Path] = []
        self._start_time: float = 0.0
        self._executor: ProcessPoolExecutor | None = None
        self._futures: dict[str, Future] = {}
        self._running = False

    def discover_images(self) -> list[Path]:
        """Scan input folder for supported image files."""
        folder = Path(self._config.input_folder)
        extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
        self._image_paths = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in extensions and p.is_file()
        )
        logger.info("Discovered %d images in %s", len(self._image_paths), folder)

        # Initialise state records for new images
        for img_path in self._image_paths:
            image_id = img_path.stem
            if self._state.get(image_id) is None:
                self._state.put(
                    image_id,
                    JobRecord(image_path=str(img_path)),
                )
        return self._image_paths

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
            image_id = img_path.stem
            record = self._state.get(image_id)
            if record and record.status in (JobStatus.COMPLETE, JobStatus.SKIPPED):
                continue

            self._state.update_status(image_id, JobStatus.MASKING)
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
        for image_id, future in list(self._futures.items()):
            future.add_done_callback(
                functools.partial(self._on_complete, image_id)
            )

    def _on_complete(self, image_id: str, future: Future) -> None:
        """Handle completion of a single image."""
        try:
            result = future.result()
            if result.error:
                self._state.update_status(
                    image_id, JobStatus.FAILED, error=result.error
                )
            else:
                record = self._state.get(image_id)
                if record:
                    updated = record.model_copy(
                        update={
                            "status": JobStatus.COMPLETE,
                            "output_svg": result.svg_path,
                            "output_tiff": result.tiff_path,
                        }
                    )
                    self._state.put(image_id, updated)
        except Exception as exc:
            self._state.update_status(
                image_id, JobStatus.FAILED, error=str(exc)
            )
            logger.error("Image %s failed: %s", image_id, exc, exc_info=True)

        self._futures.pop(image_id, None)

        progress = self._get_progress()
        if self._progress_cb:
            self._progress_cb(progress)

        if not self._futures:
            self._running = False
            if self._completion_cb:
                self._completion_cb(progress)

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
        """Stop batch processing gracefully."""
        self._running = False
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
            logger.info("Batch stopped")

    def close(self) -> None:
        """Shut down executor and close state DB."""
        self.stop()
        self._state.close()

    @property
    def is_running(self) -> bool:
        return self._running
