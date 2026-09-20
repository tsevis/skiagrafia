# Skiagrafia — File Structure

> **v0.5.0 · Python 3.13 · Apple Silicon**
>
> A local semantic vectorizing and masking desktop application.

[![Version](https://img.shields.io/badge/version-0.5.0-blue.svg)](https://github.com/tsevis/skiagrafia)
[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey.svg)](https://support.apple.com/en-us/116943)
[![Architecture](https://img.shields.io/badge/architecture-v5.2-orange.svg)](README.md#architecture)

## Overview

Skiagrafia has a contract-based pipeline: the orchestrator depends on five
capabilities, while the factory selects their concrete local implementations
from the frozen preferences or batch configuration. The `auto` segmentation
setting prefers MLX SAM 3 when its verified local checkpoint exists; otherwise
it retains GroundingDINO + SAM 2.1 as the working fallback.

The project is intentionally a flat Python layout. `uv.lock` and `run.sh`
define the supported Python `>=3.13,<3.14` runtime. Historical audit reports
under `docs/audits/` are evidence snapshots, not current operational
instructions.

## Repository Map

```text
skiagrafia/
├── main.py                         # Application bootstrap and Tk root
├── pyproject.toml                  # v0.5.0 metadata, Python/runtime dependencies
├── uv.lock                         # Locked Python 3.13 dependency resolution
├── run.sh                          # Validates and launches .venv/bin/python
├── scripts/
│   └── verify.sh                   # Locked-runtime test/check entry point
├── README.md                       # User guide, installation and troubleshooting
├── FILE_STRUCTURE.md               # This current module map
│
├── core/                           # Pipeline, contracts, batch and semantic policy
│   ├── contracts.py                # Five capability Protocols + CapabilitySet
│   ├── factory.py                  # Preferences → concrete capability wiring
│   ├── orchestrator.py             # Single-image structural pipeline and outputs
│   ├── pipeline_results.py         # Source, layer and run result models
│   ├── pipeline_geometry.py        # Pure mask/bbox maths and filename-safe labels
│   ├── interrogation.py            # VLM candidates and selection policy
│   ├── typography_labels.py        # Does a label name typography, or single glyphs
│   ├── typography_matching.py      # Semantic glyphs matched to detector boxes
│   ├── knowledge.py                # TOML domain-guide models and normalization
│   ├── batch_runner.py             # ProcessPoolExecutor runner and persisted metrics
│   ├── batch_session.py            # Frozen run/guide/interrogation/triage snapshots
│   ├── batch_template.py           # Reusable Single → Batch configuration template
│   ├── state_manager.py            # Per-image JSON-in-SQLite state records
│   ├── layer_editing.py            # Layer edits plus all-objects TIFF recomposition
│   ├── preset_library.py           # Bundled preset discovery
│   └── presets/
│       └── apple_the_first_50_years.toml
│
├── models/                         # Local model adapters
│   ├── vlm_client.py               # Ollama and existing llama.cpp HTTP clients
│   ├── local_vlm.py                # App-managed offline llama.cpp server
│   ├── mlx_sam3.py                 # MLX SAM 3 instance masks + SAM 2.1 fallback
│   ├── grounded_sam.py             # GroundingDINO detection and SAM 2.1 masks
│   ├── vitmatte_refiner.py         # Alpha-matte refinement
│   ├── vendored_contracts.py       # Protocols for model source not importable here
│   └── moondream_client.py         # Legacy import compatibility shim
│
├── processors/                     # Pure image, mask, vector and output operations
│   ├── source_image.py             # EXIF-aware RGB/alpha loading and detection view
│   ├── mask_ops.py                 # Mask metrics, cleanup and Boolean operations
│   ├── image_filter.py             # Image pre-processing helpers
│   ├── vectorizer.py               # VTracer and safe SVG assembly
│   └── output_writer.py            # Validated SVG/TIFF/PNG/PDF serialization
│
├── ui/                             # Tkinter desktop application
│   ├── main_window.py              # App shell and Single/Batch switching
│   ├── container_utils.py          # Clearing a container, reporting what it cannot
│   ├── setup_wizard.py             # Core-model and VLM setup checks/download UI
│   ├── single/                     # Editable three-panel Single Image workflow
│   │   ├── left_panel.py           # Labels, scan and pipeline parameters
│   │   ├── scan_dedup.py           # Detection candidate deduplication
│   │   ├── canvas_panel.py         # Zoom/pan canvas and overlays
│   │   ├── canvas_drawing.py       # Manual selection drawing
│   │   ├── canvas_events.py        # Canvas interaction bindings
│   │   ├── canvas_overlays.py      # Mask/vector/composite overlays
│   │   ├── dnd_contracts.py        # The tkinterdnd2 surface this project relies on
│   │   └── right_panel.py          # Layer edits and exports
│   ├── batch/                      # Six-step Batch workflow
│   │   ├── batch_view.py           # Session wiring, frozen artifacts and resume
│   │   ├── sidebar.py              # Step navigation
│   │   ├── bottom_bar.py           # Progress and navigation controls
│   │   └── steps/                  # Import, Configure, Interrogate, Triage,
│   │                               # Progress and Output screens
│   └── preferences/                # Preferences and domain-guide editor
│
├── utils/                          # Runtime, model and I/O safeguards
│   ├── preferences.py              # Defaults and exact-old-default migrations
│   ├── model_manager.py            # Core model registry, verified downloads/paths
│   ├── bootstrap.py                # First-run readiness detection
│   ├── security.py                 # URL, archive/output path and atomic-write guards
│   ├── array_types.py              # Narrowing OpenCV returns to their declared dtype
│   ├── coord_math.py               # Crop, remap and bounding-box transforms
│   ├── mps_utils.py                # PyTorch MPS/CPU selection
│   ├── cairo_support.py            # Optional Cairo export support
│   └── thumbnail.py                # Source and SVG thumbnail utilities
│
├── tests/                          # Deterministic unit, pipeline and GUI tests
│   ├── test_factory.py             # Backend selection and MLX wiring
│   ├── test_interrogation.py       # Candidate policy and glyph observations
│   ├── test_batch_*.py             # Batch runner/session/template behavior
│   ├── test_gui_*.py               # Fast smoke + opt-in window integration coverage
│   ├── conftest.py                 # Shared fixtures; the hook that marks windowed tests
│   ├── *_fakes.py                  # Offline capability stand-ins, shared per area
│   ├── test_security_boundaries.py # I/O, path, URL, archive and SVG guards
│   └── test_*.py                   # Model, output, state and pipeline regressions
│
└── docs/
    ├── MANUAL.md                   # User workflow manual
    ├── quality-pipeline.md         # Historical quality-pipeline report
    ├── audits/                     # Dated audit and QA evidence
    │   └── 2026-09-16/
    │       └── PYTHON313_MLX_SAM3_MIGRATION.md
    ├── skiagrafia-readme.jpg       # README interface image
    └── Skiagrafia.png              # Application icon
```

## Runtime and Models

| Concern | Source of truth | Behaviour |
|---|---|---|
| Python/runtime | `pyproject.toml`, `uv.lock`, `run.sh` | Python 3.13 only; `run.sh` refuses a missing or incompatible `.venv` |
| VLM | `models/vlm_client.py`, `models/local_vlm.py` | Ollama, existing llama.cpp, or app-managed local llama.cpp; service URLs must be loopback roots |
| Instance detection | `core/factory.py`, `models/mlx_sam3.py` | `auto` selects MLX SAM 3 only with `mlx_sam3/sam3-mod-weights/model.safetensors` |
| Fallback segmentation | `models/grounded_sam.py` | GroundingDINO produces boxes and SAM 2.1 produces masks when MLX is unavailable or unsuitable |
| Alpha/vector output | `models/vitmatte_refiner.py`, `processors/vectorizer.py` | Soft alpha mattes and structurally validated SVG paths |
| Core weights | `utils/model_manager.py`, `utils/bootstrap.py` | Wizard may acquire registered fallback weights; it does not download the optional MLX SAM 3 bundle |

The bundled MLX source currently needs `mlx==0.31.2`. Newer MLX releases must
not be substituted without independently updating and testing the bundle's
custom Metal kernel contract.

## Data Flow

```text
Single Image UI / Batch UI
            │ user settings, Selection Request, Domain Guide
            ▼
       core.factory
            │ CapabilitySet: Interrogator · Detector · Segmenter · AlphaRefiner · Vectorizer
            ▼
    core.orchestrator
      ├── VLM semantic interrogation
      ├── MLX SAM 3 native instances OR GroundingDINO + SAM 2.1 fallback
      ├── parent/child containment, repeated-instance and glyph validation
      ├── VitMatte alpha refinement and VTracer SVG assembly
      └── per-layer TIFF/SVG + RGBA all-objects TIFF
```

Every successful image writes an `*_all-objects.tiff` sidecar. Its RGB comes
from the original image and its alpha is the pixel-wise union of accepted layer
alpha mattes; an intentionally empty
selection remains transparent rather than fabricating foreground. Layer edits
recompute the sidecar through `core/layer_editing.py`.

## Batch Persistence and Resume

`ui/batch/batch_view.py` uses `core/batch_session.py` and `BatchRunner` as one
processing path. A resumable run directory contains:

| Artifact | Contents |
|---|---|
| `run.json` | Input/output locations, Selection Request and captured Domain Guide |
| `guide.toml` | Frozen guide copy, when a guide was used |
| `interrogation.json` | Per-image VLM candidates and instance policies |
| `triage.json` | Approved labels and per-image label exclusions |
| `processing.json` | Effective immutable `BatchConfig` after Triage |
| `state.db` | Per-image status, successful outputs and metrics |

The GUI exposes Resume only after validating all of these artifacts together.
A bare, stale or corrupt `state.db` is never shown as resumable; a saved run
that cannot prove this parity is restored only as a new editable batch.

## Verification Commands

Use the isolated project interpreter for every Python command:

```bash
./.venv/bin/python -m pytest -q
./.venv/bin/python -m pytest -q -m gui
./.venv/bin/python -m ruff check .
./.venv/bin/python -m pyright
./.venv/bin/python -m compileall -q .
./.venv/bin/python -m pip check
./.venv/bin/python -m pip_audit
uv lock --check
git diff --check
```

Or through `scripts/verify.sh`, which uses that interpreter for you:

| mode | what it runs |
| --- | --- |
| *(none)* | the default pytest run |
| `check` | **every gate except the windowed tests** — use this while working |
| `types` | pyright alone |
| `gui` | the one real-window smoke test |
| `gui-full` | all windowed tests |
| `all` | `check` plus every windowed test — for a release |

`check` exists because `all` runs `-m "gui or gui_integration"`, which opens
real windows on whoever's desktop is running it. That is correct before a
release and wrong while working, and the two should not share one command.

The `gui` marker is deliberately the fast real-window smoke gate. Full
windowed integration tests use `-m gui_integration` and are opt-in, rather
than part of ordinary development checks.

`pyright` runs in **basic** mode, configured in `[tool.pyright]` in
pyproject.toml. It needs `venvPath`/`venv` set there: without them it
resolves against its own interpreter and reports numpy, torch, pydantic and
cv2 as missing, with several hundred attribute errors cascading off that.

`pip-audit` reads the **installed environment**, not the lock file, so it is
only meaningful after `uv sync --frozen`. A `.venv` that has drifted from the
lock will report packages the project no longer declares.

## Operational Limits

- MLX SAM 3 improves text-to-instance segmentation, but semantic recognition
  remains model output. Ambiguous material needs Triage and per-image
  exceptions.
- The download registry restricts source hosts, the output paths/SVGs are
  validated, and a registry entry may now pin an immutable SHA-256 that is
  verified before the download is moved into place. **Three of the four
  entries are deliberately unpinned, because upstream publishes no digest to
  pin them to** (checked 2026-09-20): the GitHub releases API reports
  `digest: null` for `groundingdino_swint_ogc.pth`; `sam2.1_hiera_large.pt`
  offers only an S3 multipart ETag, which is a hash of part hashes and cannot
  be compared against the file; and `grounded-sam-2-source` points at
  `refs/heads/main.zip`, a moving target that no fixed digest can describe.
  Hashing the copies already on a developer's disk would record whatever
  those copies are — trust-on-first-use in the costume of an integrity check
  — so it is not done. Only `vitmatte-base-composition-1k` is pinned, from
  HuggingFace's published LFS digest and the bytes its resolve URLs served.
- Batch state is JSON in SQLite (`core/state_manager.py`). It is never
  unpickled, because a `state.db` can arrive from a copied or shared batch
  folder; a record that is not a valid job record raises instead of loading.
- `ruff check .` is configured in `pyproject.toml` so the gate does not vary
  by machine. Reviewed-and-accepted findings carry an inline `noqa` naming the
  guard that makes them safe. Bandit still reports some of these, since it
  does not read ruff directives: the `xml` parses are DTD/entity-rejected and
  size-capped, the `urlopen` calls run behind `validate_download_url` or
  `validate_loopback_url`, and VitMatte's `from_pretrained` needs no revision
  pin because it loads a local directory with `local_files_only=True`.

See [README.md](README.md) for installation and user-facing operation, and
[the Python 3.13 / MLX SAM 3 migration audit](docs/audits/2026-09-16/PYTHON313_MLX_SAM3_MIGRATION.md)
for the verified runtime evidence.
