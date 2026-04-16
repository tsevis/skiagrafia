# Skiagrafia — File Structure

> **Semantic Vectorizing & Masking Creator**
> A desktop application for AI-powered image segmentation, masking, and vectorization.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey.svg)](https://developer.apple.com/documentation/techdocs/50056847)
[![Architecture](https://img.shields.io/badge/architecture-v5.2-orange.svg)](docs/CLAUDEv5.md)

---

## Overview

Skiagrafia is a Python desktop application that uses local ML models (GroundingDINO, SAM 2.1 HQ, VitMatte, Moondream 2) to semantically segment images and produce vector (SVG) and bitmap (TIFF/PNG) outputs. It features a Tkinter-based GUI with two operating modes: **Single Image** (designer workflow) and **Batch** (production pipeline).

**Architecture v5.2**: Contract-based dependency injection with five capability protocols, multi-level deduplication (scan-stage bbox dedup + pipeline-stage mask/containment dedup), and smart child validation.

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
├── run.sh                           # Shell script launcher (sets offline env)
├── README.md                        # Main documentation
├── FILE_STRUCTURE.md                # This file
├── LICENSE                          # MIT license
│
├── core/                            # Core pipeline and state management
│   ├── __init__.py
│   ├── orchestrator.py              # 10-step pipeline with injected capabilities
│   ├── batch_runner.py              # ProcessPoolExecutor batch coordinator
│   ├── batch_template.py            # Single → Batch template serialization
│   ├── state_manager.py             # SQLiteDict job state persistence
│   ├── interrogation.py             # GuidedInterrogator with VLM fallback chain
│   ├── knowledge.py                 # KnowledgePack and ObjectKnowledge models
│   ├── contracts.py                 # Five capability Protocol interfaces
│   └── factory.py                   # CapabilitySet builder from preferences
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
│   ├── vectorizer.py                # VTracer wrapper (Vectorizer protocol) + SVG assembly
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
│   │   ├── left_panel.py            # Drop zone, labels, parameters, scan dedup
│   │   ├── canvas_panel.py          # Canvas with zoom/pan, scrollbars, overlays
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
│       ├── preferences_window.py    # Six-tab modal (General, Models, Pipeline, Appearance, Templates, Domain Guides)
│       └── guide_editor.py          # Domain guide TOML editor component
│
├── utils/                           # Utility modules
│   ├── __init__.py
│   ├── mps_utils.py                 # MPS/CPU device detection
│   ├── model_manager.py             # ModelManager class for lifecycle
│   ├── coord_math.py                # Affine remap, crop, bbox helpers
│   ├── thumbnail.py                 # 32×32 SVG thumbnail renderer
│   └── preferences.py               # JSON preferences load/save
│
├── tests/                           # Test suite
│   ├── test_contracts.py            # Protocol conformance and instantiation tests
│   └── test_knowledge_interrogation.py  # KnowledgePack, interrogation helpers
│
└── docs/                            # Documentation and planning
    ├── skiagrafia-readme.jpg        # README hero image
    └── Skiagrafia.png               # Application icon
```

---

## Module Descriptions

### Entry Point

| File | Purpose |
|------|---------|
| `main.py` | Application bootstrap: environment setup (offline mode), logging, Ollama health check, TkinterDnD root window, MainWindow instantiation |
| `pyproject.toml` | Project metadata and dependencies (torch, torchvision, tkinterdnd2, pillow, opencv-python-headless, etc.) |
| `run.sh` | Shell script launcher that sets environment variables for offline inference and disables bytecode caching |

### Core Pipeline

| File | Purpose |
|------|---------|
| `orchestrator.py` | **10-step pipeline** with injected `CapabilitySet`. Knows only Protocol interfaces. Includes multi-level parent dedup (mask IoU, mask containment, bbox IoU) and child validation (parent-similarity rejection, child-child dedup). Pipeline: Load → Interrogate → Detect → Segment parent → Segment children → Remap → Alpha refine → Mask refine → Vectorize → SVG assembly |
| `batch_runner.py` | Coordinates parallel processing via `ProcessPoolExecutor`; uses factory to build capabilities in worker processes |
| `batch_template.py` | Pydantic model for serializing Single mode parameters into reusable Batch templates |
| `state_manager.py` | SQLiteDict-based persistence for batch job state (pending, running, complete, failed) |
| `interrogation.py` | `GuidedInterrogator` with VLM fallback chain (primary → fallback → reasoner), tiled fallback for high-res images, configurable child parts cap per profile |
| `knowledge.py` | `KnowledgePack` and `ObjectKnowledge` Pydantic models for semantic label normalization and detector phrase ranking |
| `contracts.py` | Five `@runtime_checkable` Protocol interfaces (Interrogator, Detector, Segmenter, AlphaRefiner, Vectorizer) + `CapabilitySet` bundle |
| `factory.py` | `build_capabilities()` function that reads preferences and constructs wired `CapabilitySet` with concrete clients |

### ML Model Clients (Protocol Implementations)

| File | Purpose |
|------|---------|
| `moondream_client.py` | `MoondreamClient`: HTTP client for Ollama API; multi-prompt child detection with numbering cleanup; implements `Interrogator` protocol |
| `grounded_sam.py` | `GroundedSAM`: GroundingDINO (text→bbox) + SAM 2.1 HQ (bbox→mask); `prefer_full_box` multi-mask mode for manual bboxes; synonym retry for ambiguous labels; implements `Detector` + `Segmenter` protocols |
| `vitmatte_refiner.py` | `VitMatteRefiner`: Alpha matting for fine edge detail; implements `AlphaRefiner` protocol |

### Processors

| File | Purpose |
|------|---------|
| `mask_ops.py` | Boolean operations (subtract, union, intersect), bounding box calculations, mask refinement (bilateral, morphology, contour filtering), coverage metrics |
| `image_filter.py` | Bilateral filtering for edge-preserving smoothing, K-Means color quantization |
| `vectorizer.py` | `VTracerVectorizer` class (implements `Vectorizer` protocol) + `trace_mask()` + `assemble_svg()` multi-layer composition |
| `output_writer.py` | Serialization to TIFF (4-channel), PNG, SVG, PDF formats |

### UI Components

| Directory/File | Purpose |
|----------------|---------|
| `ui/theme.py` | Color palette (mirrors macOS system colors), ttk style configuration |
| `ui/main_window.py` | Top-level shell: title bar, mode switcher, content area swap, preferences access |
| `ui/mode_switcher.py` | Custom ttk.Frame with segmented control buttons for Single/Batch mode |
| `ui/single/single_view.py` | Three-panel layout manager coordinating left/center/right panels |
| `ui/single/left_panel.py` | Drop zone, label list, VTracer parameters, Process/Export buttons, **scan-stage bbox dedup** (filters duplicate detections before they reach the UI) |
| `ui/single/canvas_panel.py` | Canvas with zoom/pan, **horizontal and vertical scrollbars**, overlay rendering for masks/vectors, manual box drawing, compare slider |
| `ui/single/right_panel.py` | Hierarchical layer list with visibility toggles and export controls |
| `ui/single/canvas_overlays.py` | Mask and vector overlay rendering utilities |
| `ui/batch/batch_view.py` | Six-step wizard layout manager |
| `ui/batch/sidebar.py` | Numbered step navigation with completion indicators |
| `ui/batch/bottom_bar.py` | Status display, progress bar, navigation buttons |
| `ui/batch/steps/step_import.py` | Step 1: Folder selection, template loading, recursion depth |
| `ui/batch/steps/step_configure.py` | Step 2: Output format, VTracer parameters, naming conventions |
| `ui/batch/steps/step_interrogate.py` | Step 3: Moondream scanning progress with tag cloud preview |
| `ui/batch/steps/step_triage.py` | Step 4: Human label review and confirmation (mandatory gate) |
| `ui/batch/steps/step_progress.py` | Step 5: Real-time progress with per-image status and thumbnails |
| `ui/batch/steps/step_output.py` | Step 6: Summary statistics, export bundles, retry failed jobs |
| `ui/preferences/preferences_window.py` | Six-tab preferences modal: General, Models & Ollama, Pipeline, Appearance, Templates, Domain Guides |
| `ui/preferences/guide_editor.py` | Domain guide TOML editor component with scrollable form and live preview |

### Utilities

| File | Purpose |
|------|---------|
| `mps_utils.py` | PyTorch MPS device detection with CPU fallback |
| `model_manager.py` | `ModelManager` class for model lifecycle (discovery, download, path resolution, device residency tracking) |
| `coord_math.py` | Coordinate transformations: crop with padding, mask remapping, tight bounding box (`y0,x0,y1,x1` format) |
| `thumbnail.py` | CairoSVG-based 32×32 SVG thumbnail rendering with LRU cache |
| `preferences.py` | JSON file I/O for user preferences stored in `~/.config/skiagrafia/`; `get_models_dir()` helper |

### Tests

| File | Purpose |
|------|---------|
| `test_contracts.py` | Protocol conformance (all concrete classes satisfy their protocols), CapabilitySet construction, ModelManager, Orchestrator instantiation and signature tests |
| `test_knowledge_interrogation.py` | KnowledgePack loading/validation, child parts filtering, composition prompts, candidate parsing, detector phrase ranking, reasoner gating |

---

## Data Flow

### Architecture Overview

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
│  Multi-level dedup: mask IoU, containment, bbox IoU.            │
│  Child validation: parent-similarity, child-child dedup.        │
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
[Image Drop] ──► [Moondream Scan] ──► [GroundingDINO Boxes]
                                              │
                                    ┌─────────▼──────────┐
                                    │  Scan-Stage Dedup   │
                                    │  bbox IoU + contain │
                                    │  (left_panel.py)    │
                                    └─────────┬──────────┘
                                              │
                                    [Label Selection]
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 ORCHESTRATOR (10 steps)                          │
│                                                                  │
│  1.  Load Image (RGB conversion, metadata)                      │
│  2.  Moondream Interrogation → parent/child labels              │
│  3.  GroundingDINO Detection → bounding boxes                   │
│      ├── Manual bbox: prefer_full_box + clip to bbox            │
│      └── Pipeline-stage dedup: mask IoU, containment, bbox IoU  │
│  4.  SAM 2.1 HQ Parent Segmentation → parent mask               │
│  5.  SAM 2.1 HQ Child Segmentation → child masks                │
│      ├── Reject if child IoU > 0.85 with parent (same object)  │
│      └── Reject if child IoU > 0.80 with sibling (duplicate)   │
│  6.  Coordinate Remapping (crop → full canvas)                  │
│  7.  VitMatte Alpha Refinement → soft alpha mattes              │
│  8.  Mask Refinement (bilateral, morphology, contour filter)    │
│  9.  VTracer Vectorization → SVG path data                      │
│  10. SVG Assembly + Export (TIFF/SVG)                           │
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
| `Orchestrator` | `core/orchestrator.py` | 10-step pipeline executor; receives `CapabilitySet` via constructor injection; multi-level parent dedup and child validation |
| `PipelineResult` | `core/orchestrator.py` | Pydantic model for pipeline output (layers, paths, errors) |
| `LayerResult` | `core/orchestrator.py` | Pydantic model for a single layer result (label, role, bbox, svg_data) |
| `CapabilitySet` | `core/contracts.py` | Pydantic bundle of all five capability protocols for dependency injection |
| `Interrogator` | `core/contracts.py` | Protocol: `interrogate(image, confirmed_labels, knowledge_pack)` |
| `Detector` | `core/contracts.py` | Protocol: `detect_box(image, label, box_threshold, text_threshold)` |
| `Segmenter` | `core/contracts.py` | Protocol: `segment(image, bbox, label, prefer_full_box)` + `clear_cache()` |
| `AlphaRefiner` | `core/contracts.py` | Protocol: `predict(image, mask)` |
| `Vectorizer` | `core/contracts.py` | Protocol: `trace(mask)` |
| `GuidedInterrogator` | `core/interrogation.py` | VLM interrogation with fallback chain, tiled processing, configurable child parts cap |
| `InterrogationCandidate` | `core/interrogation.py` | Pydantic model: canonical/display labels, detector phrases, confidence, role |
| `KnowledgePack` | `core/knowledge.py` | TOML-based label taxonomy with detector phrase rankings and child parts |
| `GroundedSAM` | `models/grounded_sam.py` | GroundingDINO + SAM 2.1 HQ wrapper; `prefer_full_box` multi-mask selection; synonym retry; implements Detector + Segmenter |
| `MoondreamClient` | `models/moondream_client.py` | Ollama HTTP client; multi-prompt child detection with numbering cleanup |
| `VitMatteRefiner` | `models/vitmatte_refiner.py` | Alpha matting model wrapper |
| `VTracerVectorizer` | `processors/vectorizer.py` | VTracer wrapper implementing Vectorizer protocol |
| `CanvasPanel` | `ui/single/canvas_panel.py` | Tkinter canvas with zoom/pan, scrollbars, overlay rendering, manual box drawing |
| `ModelManager` | `utils/model_manager.py` | Model lifecycle management with user-configurable directory |
| `BatchTemplate` | `core/batch_template.py` | Pydantic model for reusable batch configurations |
| `MainWindow` | `ui/main_window.py` | Application shell; manages mode switching |
| `SingleView` | `ui/single/single_view.py` | Three-panel layout for single image mode |
| `BatchView` | `ui/batch/batch_view.py` | Six-step wizard for batch mode |

---

## Configuration

| Location | Purpose |
|----------|---------|
| `.claude/settings.local.json` | Claude Code assistant project settings |
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

# Or use the shell script (recommended — sets env vars)
./run.sh
```

---

## Related Documentation

- **[README.md](README.md)** — Full user documentation with installation and usage guide
- **[docs/CLAUDE.md](docs/CLAUDE.md)** — Implementation guide for AI assistants
- **[docs/CLAUDEv5.md](docs/CLAUDEv5.md)** — v5.0 architecture and contracts refactor notes
- **[docs/CLAUDEv4.md](docs/CLAUDEv4.md)** — v4 implementation notes and historical reference
