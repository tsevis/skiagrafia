# Skiagrafia — File Structure

> **Semantic Vectorizing & Masking Creator**
> A desktop application for AI-powered image segmentation, masking, and vectorization.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey.svg)](https://developer.apple.com/documentation/techdocs/50056847)
[![Architecture](https://img.shields.io/badge/architecture-v5.0-orange.svg)](docs/CLAUDEv5.md)

---

## Overview

Skiagrafia is a Python desktop application that uses local ML models (GroundingDINO, SAM 2.1 HQ, VitMatte, Moondream 2) to semantically segment images and produce vector (SVG) and bitmap (TIFF/PNG) outputs. It features a Tkinter-based GUI with two operating modes: **Single Image** (designer workflow) and **Batch** (production pipeline).

**Architecture v5.0**: Contract-based dependency injection with five capability protocols for decoupled model management.

---

## Quick Links

| Document | Description |
|----------|-------------|
| **[README.md](README.md)** | Full documentation, installation, and usage guide |
| **[docs/CLAUDE.md](docs/CLAUDE.md)** | Implementation guide for AI assistants |
| **[docs/CLAUDEv4.md](docs/CLAUDEv4.md)** | v4 implementation notes |
| **[docs/CLAUDEv5.md](docs/CLAUDEv5.md)** | v5.0 architecture & contracts refactor guide |

---

## Directory Structure

```
skiagrafia/
│
├── main.py                          # Application entry point
├── pyproject.toml                   # Project dependencies and metadata
├── run.sh                           # Shell script launcher
├── README.md                        # Main documentation
├── FILE_STRUCTURE.md                # This file
│
├── .claude/                         # Claude Code assistant settings
│   └── settings.local.json
│
├── .qwen/                           # Qwen Code assistant settings
│   └── settings.json
│
├── core/                            # Core pipeline and state management
│   ├── __init__.py
│   ├── orchestrator.py              # 12-step pipeline with injected capabilities
│   ├── batch_runner.py              # ProcessPoolExecutor batch coordinator
│   ├── batch_template.py            # Single → Batch template serialization
│   ├── state_manager.py             # SQLiteDict job state persistence
│   ├── interrogation.py             # GuidedInterrogator with VLM fallback chain
│   ├── knowledge.py                 # KnowledgePack and ObjectKnowledge models
│   ├── contracts.py                 # v5.0: Five capability Protocol interfaces
│   └── factory.py                   # v5.0: CapabilitySet builder from preferences
│
├── models/                          # ML model wrappers (protocol implementations)
│   ├── __init__.py
│   ├── moondream_client.py          # Ollama HTTP client (Interrogator)
│   ├── grounded_sam.py              # GroundingDINO + SAM 2.1 HQ (Detector + Segmenter)
│   └── vitmatte_refiner.py          # VitMatte alpha matting (AlphaRefiner)
│
├── processors/                      # Image processing utilities
│   ├── __init__.py
│   ├── mask_ops.py                  # Boolean ops, bbox, mask refinement
│   ├── image_filter.py              # Bilateral filter, K-Means
│   ├── vectorizer.py                # VTracer wrapper (Vectorizer protocol)
│   ├── svg_assembler.py             # SVG composition and layer assembly
│   └── output_writer.py             # TIFF/PNG/SVG/PDF serialization
│
├── ui/                              # User interface components
│   ├── __init__.py
│   ├── theme.py                     # Color palette and styling
│   ├── main_window.py               # Main shell: titlebar, mode switcher
│   ├── mode_switcher.py             # Segmented control for Single/Batch
│   │
│   ├── single/                      # Single image mode UI
│   │   ├── __init__.py
│   │   ├── single_view.py           # Three-panel layout manager
│   │   ├── left_panel.py            # Drop zone, labels, parameters, Process
│   │   ├── canvas_panel.py          # Dark canvas, zoom/pan, overlays
│   │   ├── right_panel.py           # Layers list, controls, export
│   │   └── canvas_overlays.py       # Mask/vector overlay rendering
│   │
│   ├── batch/                       # Batch mode UI
│   │   ├── __init__.py
│   │   ├── batch_view.py            # Six-step wizard layout manager
│   │   ├── sidebar.py               # Numbered step navigation
│   │   ├── bottom_bar.py            # Status, progress, navigation
│   │   └── steps/
│   │       ├── __init__.py
│   │       ├── step_import.py       # Step 1: Drop zone + recent batches
│   │       ├── step_configure.py    # Step 2: Output mode + parameters
│   │       ├── step_interrogate.py  # Step 3: Progress + tag cloud
│   │       ├── step_triage.py       # Step 4: Human label confirmation
│   │       ├── step_progress.py     # Step 5: Metrics + thumbnails
│   │       └── step_output.py       # Step 6: Summary + export
│   │
│   └── preferences/                 # Preferences window
│       ├── __init__.py
│       └── preferences_window.py    # Five-tab modal (General, Models, Pipeline, Appearance, Templates)
│
├── utils/                           # Utility modules
│   ├── __init__.py
│   ├── mps_utils.py                 # MPS/CPU device detection
│   ├── model_manager.py             # v5.0: ModelManager class for lifecycle
│   ├── coord_math.py                # Affine remap, crop, bbox helpers
│   ├── thumbnail.py                 # 32×32 SVG thumbnail renderer
│   └── preferences.py               # JSON preferences load/save
│
├── docs/                            # Documentation and planning
│   ├── CLAUDE.md                    # Implementation guide for Claude Code
│   ├── CLAUDEv4.md                  # Claude Code v4 implementation notes
│   ├── CLAUDEv5.md                  # v5.0 Contracts & DI refactor guide
│   ├── CLAUDE_FIXES.md              # Bug fixes and troubleshooting guide
│   ├── CLAUDE_HEALTHCHECK.md        # System health check procedures
│   ├── skiaplan.ai                  # Architecture diagram (Adobe Illustrator)
│   ├── skiaplan.png                 # Architecture diagram preview
│   ├── skiaplan.txt                 # Architecture planning notes
│   ├── skiaplan2.jpg                # Alternative architecture sketch
│   ├── skiaplan2.png                # Alternative architecture sketch preview
│   ├── Strategic Optimization for Skiagrafia & Moondream.md
│   ├── Skiagrafia Health Report -- 2026-03-28.md
│   ├── Proposal for a Modular Plugin Architecture.md
│   └── Skiagrafia as a Plugin-Based Platform.md
│
└── skiagrafia_out/                  # Output directory for generated files
    └── (generated outputs)          # SVG, TIFF, PNG exports from processing
```

---

## Module Descriptions

### Entry Point

| File | Purpose |
|------|---------|
| `main.py` | Application bootstrap: environment setup (offline mode), logging, Ollama health check, TkinterDnD root window, MainWindow instantiation |
| `pyproject.toml` | Project metadata and dependencies (torch, torchvision, tkinterdnd2, pillow, opencv-python-headless, etc.) |
| `run.sh` | Shell script launcher that sets environment variables for offline inference |

### Core Pipeline (v5.0 Architecture)

| File | Purpose |
|------|---------|
| `orchestrator.py` | **12-step pipeline** with injected `CapabilitySet`. Knows only Protocol interfaces, never concrete model clients. Pipeline: Load → Interrogate → Detect → Segment (parent) → Segment (children) → Remap coords → Alpha refine → Mask refine → Vectorize → Assemble SVG → Export |
| `batch_runner.py` | Coordinates parallel processing via `ProcessPoolExecutor`; uses factory to build capabilities in worker processes |
| `batch_template.py` | Pydantic model for serializing Single mode parameters into reusable Batch templates |
| `state_manager.py` | SQLiteDict-based persistence for batch job state (pending, running, complete, failed) |
| `interrogation.py` | `GuidedInterrogator` with VLM fallback chain (primary → fallback → reasoner), tiled fallback for high-res images |
| `knowledge.py` | `KnowledgePack` and `ObjectKnowledge` Pydantic models for semantic label normalization and detector phrase ranking |
| `contracts.py` | **v5.0**: Five `@runtime_checkable` Protocol interfaces (Interrogator, Detector, Segmenter, AlphaRefiner, Vectorizer) + `CapabilitySet` bundle |
| `factory.py` | **v5.0**: `build_capabilities()` function that reads preferences and constructs wired `CapabilitySet` with concrete clients |

### ML Model Clients (Protocol Implementations)

| File | Purpose |
|------|---------|
| `moondream_client.py` | `MoondreamClient`: HTTP client for Ollama API; implements `Interrogator` protocol |
| `grounded_sam.py` | `GroundedSAM`: Wraps GroundingDINO (text→bbox) and SAM 2.1 HQ (bbox→mask); implements `Detector` + `Segmenter` protocols; lazy model loading |
| `vitmatte_refiner.py` | `VitMatteRefiner`: Alpha matting for fine edge detail; implements `AlphaRefiner` protocol |

### Processors

| File | Purpose |
|------|---------|
| `mask_ops.py` | Boolean operations (subtract, union), bounding box calculations, mask refinement (morphology, contour filtering) |
| `image_filter.py` | Bilateral filtering for edge-preserving smoothing, K-Means color quantization |
| `vectorizer.py` | `VTracerVectorizer` class (implements `Vectorizer` protocol) + `trace_mask()` function + `assemble_svg()` composition |
| `svg_assembler.py` | SVG composition and layer assembly utilities for hybrid raster+vector output |
| `output_writer.py` | Serialization to TIFF (4-channel), PNG, SVG, PDF formats |

### UI Components

| Directory/File | Purpose |
|----------------|---------|
| `ui/theme.py` | Color palette (mirrors macOS system colors), ttk style configuration |
| `ui/main_window.py` | Top-level shell: title bar, mode switcher, content area swap, preferences access |
| `ui/mode_switcher.py` | Custom ttk.Frame with segmented control buttons for Single/Batch mode |
| `ui/single/` | Single image mode: three-panel layout (left: controls, center: canvas, right: layers) |
| `ui/single/single_view.py` | Three-panel layout manager coordinating left/center/right panels |
| `ui/single/left_panel.py` | Drop zone, label list, VTracer parameters, Process/Export buttons |
| `ui/single/canvas_panel.py` | Dark canvas with zoom/pan, overlay rendering for masks/vectors |
| `ui/single/right_panel.py` | Hierarchical layer list with visibility toggles and export controls |
| `ui/single/canvas_overlays.py` | Mask and vector overlay rendering utilities |
| `ui/batch/` | Batch mode: six-step wizard with sidebar navigation and progress tracking |
| `ui/batch/batch_view.py` | Six-step wizard layout manager |
| `ui/batch/sidebar.py` | Numbered step navigation with completion indicators |
| `ui/batch/bottom_bar.py` | Status display, progress bar, navigation buttons |
| `ui/batch/steps/` | Individual wizard step implementations |
| `ui/batch/steps/step_import.py` | Step 1: Folder selection, template loading, recursion depth |
| `ui/batch/steps/step_configure.py` | Step 2: Output format, VTracer parameters, naming conventions |
| `ui/batch/steps/step_interrogate.py` | Step 3: Moondream scanning progress with tag cloud preview |
| `ui/batch/steps/step_triage.py` | Step 4: Human label review and confirmation (mandatory gate) |
| `ui/batch/steps/step_progress.py` | Step 5: Real-time progress with per-image status and thumbnails |
| `ui/batch/steps/step_output.py` | Step 6: Summary statistics, export bundles, retry failed jobs |
| `ui/preferences/` | Five-tab preferences modal |
| `ui/preferences/preferences_window.py` | Preferences window: General, Models & Ollama, Pipeline, Appearance, Templates |

### Utilities

| File | Purpose |
|------|---------|
| `mps_utils.py` | PyTorch MPS device detection with CPU fallback |
| `model_manager.py` | **v5.0**: `ModelManager` class for model lifecycle (discovery, download, path resolution, device residency tracking, memory-aware unload) |
| `coord_math.py` | Coordinate transformations: crop with padding, mask remapping, tight bounding box |
| `thumbnail.py` | CairoSVG-based 32×32 SVG thumbnail rendering with LRU cache |
| `preferences.py` | JSON file I/O for user preferences stored in `~/.config/skiagrafia/`; `get_models_dir()` helper |

---

## Data Flow

### v5.0 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     UI Layer                                    │
│  main_window.py · left_panel.py · step_progress.py              │
│  batch_runner.py                                                │
│                                                                 │
│  Reads preferences → builds concrete clients → injects          │
│  them into Orchestrator via CapabilitySet                       │
└────────────────────────┬────────────────────────────────────────┘
                         │ passes CapabilitySet
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Orchestrator                                  │
│  Knows ONLY the Protocol interfaces.                            │
│  Never imports a concrete model client.                         │
└────────────────────────┬────────────────────────────────────────┘
                         │ calls Protocol methods
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Capability Protocols (contracts.py)                │
│  Interrogator · Detector · Segmenter · AlphaRefiner ·           │
│  Vectorizer                                                     │
└─────────────────────────────────────────────────────────────────┘
                         │ implemented by
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Concrete Model Clients (models/)                   │
│  moondream_client.py · grounded_sam.py · vitmatte_refiner.py    │
│                                                                 │
│  Each receives its model path from ModelManager.                │
└────────────────────────┬────────────────────────────────────────┘
                         │ paths resolved by
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              ModelManager (model_manager.py)                    │
│  User-configurable models_dir from preferences.                 │
│  Registry of known models with download URLs.                   │
│  Device residency tracking and memory-aware unload.             │
└─────────────────────────────────────────────────────────────────┘
```

### Single Image Mode Pipeline

```
[Image Drop] ──► [Moondream Scan] ──► [Label Selection]
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                 ORCHESTRATOR (12 steps)                         │
│                                                                  │
│  0.  Initialize capabilities (injected via CapabilitySet)       │
│  1.  Load Image (RGB conversion, metadata)                      │
│  2.  Moondream Interrogation → parent/child labels              │
│  3.  GroundingDINO Detection → bounding boxes                   │
│  4.  SAM 2.1 HQ Parent Segmentation → parent mask               │
│  5.  SAM 2.1 HQ Child Segmentation (cropped) → child masks      │
│  6.  Coordinate Remapping (crop → full canvas)                  │
│  7.  VitMatte Alpha Refinement → soft alpha mattes              │
│  8.  Mask Refinement (bilateral, morphology, contour filter)    │
│  9.  VTracer Vectorization → SVG path data                      │
│  10. SVG Assembly (grouped layers, hierarchy)                   │
│  11. Export (TIFF/PNG/SVG/PDF)                                  │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
    [Canvas Preview] ◄──► [Layer Controls] ──► [Export]
```

### Batch Mode Pipeline

```
Step 1: Import    ──► Select folder or load template
Step 2: Configure ──► Output mode, recursion depth, VTracer params
Step 3: Interrogate ──► Moondream scans all images (parallel)
Step 4: Triage    ──► Human confirms/rejects labels (mandatory gate)
Step 5: Progress  ──► Orchestrator runs on all images (ProcessPoolExecutor)
Step 6: Output    ──► Summary, export bundles, retry failed
```

---

## Key Classes

| Class | Module | Description |
|-------|--------|-------------|
| `Orchestrator` | `core/orchestrator.py` | 12-step pipeline executor; receives `CapabilitySet` via constructor injection |
| `PipelineResult` | `core/orchestrator.py` | Pydantic model for pipeline output (layers, paths, errors) |
| `LayerResult` | `core/orchestrator.py` | Pydantic model for a single layer result (mask, alpha, svg_path, label) |
| `CapabilitySet` | `core/contracts.py` | Pydantic bundle of all five capability protocols for dependency injection |
| `Interrogator` | `core/contracts.py` | Protocol: `interrogate(image, confirmed_labels, knowledge_pack)` |
| `Detector` | `core/contracts.py` | Protocol: `detect_box(image, prompt, box_threshold, text_threshold)` |
| `Segmenter` | `core/contracts.py` | Protocol: `segment(image, boxes, remap_points)` + `clear_cache()` |
| `AlphaRefiner` | `core/contracts.py` | Protocol: `predict(image, mask)` |
| `Vectorizer` | `core/contracts.py` | Protocol: `trace(image, mask, params)` |
| `GuidedInterrogator` | `core/interrogation.py` | VLM interrogation with fallback chain and tiled processing |
| `KnowledgePack` | `core/knowledge.py` | Normalized label taxonomy with detector phrase rankings |
| `GroundedSAM` | `models/grounded_sam.py` | GroundingDINO + SAM 2.1 HQ wrapper; implements Detector + Segmenter |
| `MoondreamClient` | `models/moondream_client.py` | Ollama HTTP client for semantic interrogation |
| `VitMatteRefiner` | `models/vitmatte_refiner.py` | Alpha matting model wrapper |
| `VTracerVectorizer` | `processors/vectorizer.py` | VTracer wrapper implementing Vectorizer protocol |
| `ModelManager` | `utils/model_manager.py` | Model lifecycle management with user-configurable directory |
| `ModelInfo` | `utils/model_manager.py` | Model metadata (path, size, status, device residency) |
| `BatchTemplate` | `core/batch_template.py` | Pydantic model for reusable batch configurations |
| `MainWindow` | `ui/main_window.py` | Application shell; manages mode switching |
| `SingleView` | `ui/single/single_view.py` | Three-panel layout for single image mode |
| `BatchView` | `ui/batch/batch_view.py` | Six-step wizard for batch mode |
| `ZoomableCanvas` | `ui/single/canvas_panel.py` | Tkinter canvas with zoom/pan support |

---

## Configuration

| Location | Purpose |
|----------|---------|
| `.claude/settings.local.json` | Claude Code assistant project settings |
| `.qwen/settings.json` | Qwen Code assistant project settings |
| `~/.config/skiagrafia/preferences.json` | User preferences (theme, defaults, paths, models_directory) |
| `~/.config/skiagrafia/templates/*.json` | Saved batch templates |
| `~/.config/skiagrafia/skiagrafia.log` | Application log file |
| `~/ai/claudecode/mozaix/models/` | Default shared ML model weights directory (configurable) |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision` | PyTorch with MPS support |
| `tkinterdnd2` | Drag-and-drop support for Tkinter |
| `pillow`, `opencv-python-headless` | Image processing |
| `numpy`, `scipy` | Numerical operations |
| `pydantic` | Data validation and settings |
| `sqlitedict` | Persistent state storage |
| `dask[distributed]` | Parallel processing |
| `tqdm`, `rich` | Progress bars and logging |
| `cairosvg` | SVG rendering and PDF export |
| `vtracer` | Bitmap to vector conversion |
| `ollama` | Ollama Python client |
| `lxml` | XML/SVG parsing |
| `transformers`, `accelerate` | HuggingFace transformers (offline mode) |

---

## Model Weights

| Model | Filename | Purpose | Size |
|-------|----------|---------|------|
| GroundingDINO | `groundingdino_swint_ogc.pth` | Text → bounding box detection | ~660 MB |
| SAM 2.1 HQ | `sam2.1_hiera_large.pt` | Precision segmentation | ~2.4 GB |
| VitMatte ViT-B | `vitmatte-base-composition-1k/` | Alpha matting | ~350 MB |
| Moondream 2 | via Ollama | Semantic interrogation | ~1.5 GB |

---

## Execution

```bash
# Run the application
python main.py

# Or use the shell script
./run.sh
```

---

## Related Documentation

- **[README.md](README.md)** — Full user documentation with installation and usage guide
- **[docs/CLAUDE.md](docs/CLAUDE.md)** — Implementation guide for AI assistants
- **[docs/CLAUDEv5.md](docs/CLAUDEv5.md)** — v5.0 architecture and contracts refactor notes
- **[docs/CLAUDEv4.md](docs/CLAUDEv4.md)** — v4 implementation notes and historical reference
