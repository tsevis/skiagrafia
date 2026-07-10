# Skiagrafia

> **Semantic Vectorizing and Masking Creator**
>
> A desktop application for AI-powered image segmentation, masking, and vectorization.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey.svg)](https://developer.apple.com/documentation/techdocs/50056847)
[![Architecture](https://img.shields.io/badge/architecture-v5.2-orange.svg)](docs/CLAUDEv5.md)

**Skiagrafia** uses local ML models to automatically segment images into semantic layers (objects and their parts), producing both vector (SVG) and bitmap (TIFF/PNG) outputs. Designed for designers and production workflows.

<p align="center">
  <img src="docs/skiagrafia-readme.jpg" alt="Skiagrafia desktop interface showing semantic segmentation and masking layers" width="1200">
</p>

---

## Quick Start

```bash
# Clone and install
git clone <repository-url> && cd skiagrafia
python -m venv venv && source venv/bin/activate
pip install -e .

# Backend option A — Ollama (default)
ollama serve &
ollama pull qwen2.5vl:3b

# Backend option B — llama.cpp server (OpenAI-compatible API)
llama-server -hf Qwen/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M --port 8080 -c 8192

# Run the application
python main.py
```

On first launch, a setup wizard checks for missing model weights
(GroundingDINO, SAM 2.1, VitMatte, Grounded-SAM-2 source) and downloads
them into your models directory. Machines that already have the models
skip the wizard entirely.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [System Requirements](#system-requirements)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Operating Modes](#operating-modes)
7. [Architecture v5.2](#architecture-v52)
8. [Pipeline Details](#pipeline-details)
9. [ML Models](#ml-models)
10. [User Interface](#user-interface)
11. [Configuration](#configuration)
12. [Output Formats](#output-formats)
13. [API Reference](#api-reference)
14. [Troubleshooting](#troubleshooting)
15. [License](#license)

---

## Overview

Skiagrafia addresses a common challenge in design and production workflows: converting raster images into clean, layered vector artwork with precise masks. Traditional approaches require manual tracing and masking, which is time-consuming and error-prone. Skiagrafia automates this process using state-of-the-art machine learning models running entirely locally on your machine.

### Key Capabilities

- **Semantic Understanding**: Automatically identifies objects and their constituent parts (e.g., a monitor with screen, stand, and bezel)
- **Precision Segmentation**: Uses SAM 2.1 HQ for high-quality mask generation
- **Alpha Matting**: VitMatte refinement for hair, fur, and transparent edges
- **Vector Output**: VTracer converts bitmaps to clean SVG paths
- **Batch Processing**: Process thousands of images with a wizard-driven workflow
- **100% Local Inference**: No cloud APIs; all models run on your machine

---

## Features

### Single Image Mode

- Interactive designer workflow with live preview
- Drag-and-drop image import
- Real-time layer visualization
- Iterative refinement with adjustable parameters
- Export to SVG, TIFF, PNG, and PDF

### Batch Mode

- Six-step wizard for production pipelines
- Process 2,000+ images in parallel
- Human-in-the-loop label triage (mandatory gate)
- Persistent state with resume capability
- Template system for reusable configurations
- Progress tracking and error recovery

### ML Pipeline (10-step)

- **Qwen2.5-VL / Qwen3-VL**: Semantic interrogation via Ollama or a llama.cpp server, with dual-prompt strategy and fallback chain
- **GroundingDINO**: Text-to-bounding-box detection with scan-stage deduplication
- **SAM 2.1 HQ**: State-of-the-art segmentation with multi-mask output for manual bboxes
- **VitMatte**: High-quality alpha matting
- **VTracer**: Bitmap-to-SVG spline fitting

### Domain Guides

- TOML-based knowledge packs that teach the VLM which objects to look for
- Built-in GUI editor inside Preferences (Domain Guides tab)
- Define canonical names, aliases, detector phrases, and child parts per object
- Live TOML preview as you edit
- Batch defaults (preferred VLM, tiling, fallback chain) per domain

### Technical Features

- Contract-based architecture with Protocol interfaces
- Dependency injection via CapabilitySet
- User-configurable model directory
- Apple Silicon optimized (MPS acceleration)
- Lazy model loading with memory residency
- ProcessPoolExecutor parallelization
- SQLiteDict state persistence

---

## System Requirements

### Hardware

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Platform** | Apple Silicon Mac | M1 Ultra or better |
| **Memory** | 32 GB | 64-128 GB unified memory |
| **Storage** | 20 GB free | 50 GB for model weights |
| **GPU** | MPS capable | Metal Performance Shaders |

### Software

- **Operating System**: macOS 12.0 (Monterey) or later
- **Python**: 3.11 or later
- **VLM backend** (one of):
  - **Ollama** running locally at `http://localhost:11434` (default), or
  - **llama.cpp server** (`llama-server`) at `http://localhost:8080`

### Model Weights

The following model weights must be available (the first-run wizard can
download all of them):

| Model | Filename | Size |
|-------|----------|------|
| Grounded-SAM-2 source | `Grounded-SAM-2/` (code + configs) | ~30 MB |
| GroundingDINO | `groundingdino_swint_ogc.pth` | ~660 MB |
| SAM 2.1 HQ | `sam2.1_hiera_large.pt` | ~900 MB |
| VitMatte ViT-B | `vitmatte-base-composition-1k/` | ~380 MB |
| Qwen2.5-VL 3B | Pulled via Ollama (`qwen2.5vl:3b`) | ~3.2 GB |
| Gemma 4 E4B (fallback + reasoner) | Pulled via Ollama (`gemma4:e4b`) | ~9.6 GB |
| Qwen3-VL 8B (llama.cpp alternative) | `Qwen/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M` | ~6.5 GB |

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd skiagrafia
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -e .
```

Or using the provided shell script:

```bash
./run.sh
```

### 4. Set Up a VLM Backend

**Option A — Ollama (default):**

```bash
# Install Ollama (if not already installed)
brew install ollama

# Start Ollama service
ollama serve

# Pull the interrogation models
ollama pull qwen2.5vl:3b      # primary VLM (required)
ollama pull gemma4:e4b        # fallback VLM + text reasoner (recommended)
ollama pull minicpm-v         # second fallback (optional)
```

**Option B — llama.cpp server:**

```bash
# Install llama.cpp (if not already installed)
brew install llama.cpp

# Start the server with a multimodal model (downloads to its cache on first use)
llama-server -hf Qwen/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M --port 8080 -c 8192
```

Select the backend under **Preferences → Models → VLM backend**. A llama.cpp
server hosts a single loaded model, so it also acts as the text reasoner and
no fallback chain is used.

### 5. Download Model Weights

Model weights live in the configured models directory (an existing shared
library is used when present; fresh installs default to
`~/Library/Application Support/skiagrafia/models/`). On first launch the
setup wizard lists anything missing and downloads it with one click — or use
**Preferences → Models → Download missing**.

---

## Quick Start

### Launch the Application

```bash
# Direct execution
python main.py

# Or use the launcher script
./run.sh
```

### Single Image Workflow

1. **Drop an image** onto the canvas area
2. **Wait for interrogation** - The VLM analyzes the image
3. **Review labels** - Confirm or adjust detected objects
4. **Run pipeline** - Click "Process" to generate masks
5. **Export results** - Save as SVG, TIFF, PNG, or PDF

### Batch Workflow

1. **Import** - Select a folder or load a template
2. **Configure** - Set output options and VTracer parameters
3. **Interrogate** - The VLM scans all images (parallel)
4. **Triage** - Review and confirm labels (mandatory gate)
5. **Process** - Run the full pipeline on all images
6. **Output** - Review results and retry any failures

---

## Operating Modes

### Single Image Mode

Single Image Mode provides an interactive designer workflow with a three-panel layout:

```
+------------------+------------------------+------------------+
|                  |                        |                  |
|   Controls       |       Canvas           |     Layers       |
|                  |                        |                  |
|   - Image info   |   - Live preview       |   - Layer list   |
|   - Labels       |   - Zoom/pan           |   - Visibility   |
|   - Parameters   |   - Overlay toggles    |   - Ordering     |
|   - Actions      |                        |   - Export       |
|                  |                        |                  |
+------------------+------------------------+------------------+
```

**Features:**

- Real-time preview with zoom, pan, and scrollbar navigation
- Layer visibility toggles
- Parameter adjustment without re-running pipeline
- Individual layer export

### Batch Mode

Batch Mode provides a six-step wizard for processing large image collections:

```
Step 1: Import
    └── Select folder or load template
    └── Configure recursion depth
    └── Filter by file extension

Step 2: Configure
    └── Output mode (SVG, TIFF, PNG, PDF)
    └── VTracer parameters
    └── Naming conventions

Step 3: Interrogate
    └── Moondream scans all images
    └── Parallel processing with progress bar
    └── Generates parent/child label pairs

Step 4: Triage (MANDATORY GATE)
    └── Human reviews all labels
    └── Accept, reject, or edit labels
    └── Cannot proceed without triage

Step 5: Progress
    └── Orchestrator runs on all images
    └── ProcessPoolExecutor parallelization
    └── Real-time progress tracking

Step 6: Output
    └── Summary statistics
    └── Export bundles
    └── Retry failed jobs
```

**Key Features:**

- **Human Gate**: Step 4 (Triage) is mandatory before GPU pipeline runs
- **Resume Capability**: State persisted in SQLite, can resume interrupted batches
- **Template System**: Save and load batch configurations

---

## Architecture v5.2

v5.2 implements contract-based dependency injection with five capability protocols for decoupled model management and a lean 10-step structural pipeline. v5.2 adds two-stage deduplication, child mask validation, multi-mask manual bbox handling, and scrollbar canvas navigation.

### Design Principles

1. **Capability Protocols**: Five `@runtime_checkable` Protocol interfaces define contracts
2. **Dependency Injection**: Orchestrator receives `CapabilitySet` via constructor
3. **User-Configurable Models**: Model directory configurable via preferences
4. **Model Lifecycle Management**: `ModelManager` class handles discovery, download, and residency
5. **Lean Core**: Structural pipeline only -- segmentation and vectorization, no neural stylization

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     UI Layer                                    │
│  main_window.py · left_panel.py · canvas_panel.py               │
│  step_progress.py · batch_runner.py                             │
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
└────────────────────────┴────────────────────────────────────────┘
                         │ implemented by
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Concrete Model Clients (models/)                   │
│  vlm_client.py · grounded_sam.py · vitmatte_refiner.py          │
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

### Capability Protocols

| Protocol | Method | Concrete Implementation |
|----------|--------|------------------------|
| `Interrogator` | `interrogate()` | `GuidedInterrogator` (Ollama / llama.cpp VLM + fallbacks) |
| `Detector` | `detect_box()` | `GroundedSAM` (GroundingDINO) |
| `Segmenter` | `segment(prefer_full_box)`, `clear_cache()` | `GroundedSAM` (SAM 2.1 HQ) |
| `AlphaRefiner` | `predict()` | `VitMatteRefiner` |
| `Vectorizer` | `trace()` | `VTracerVectorizer` |

---

## Pipeline Details

### Single Image Pipeline (10 Steps)

The `Orchestrator` class executes a 10-step structural pipeline:

```
Step 1:  Load image as numpy array (BGR -> RGB)
Step 2:  VLM interrogation — get parents, filter against confirmed labels,
         get children per confirmed parent
Step 3:  GroundingDINO detection — text-to-bounding-box per parent label,
         scan-stage bbox dedup (IoU + containment) removes duplicate detections
Step 4:  SAM 2.1 HQ parent segmentation — mask from bounding box prompt,
         multi-mask output for manual bboxes (prefer_full_box), tighten bbox
         from mask contour, pipeline-stage dedup (mask IoU + containment + bbox IoU)
Step 5:  SAM child segmentation — crop region, detect + segment per child,
         child mask validation (reject parent-similar >85% IoU, sibling dedup >80% IoU),
         boolean-subtract children from parent body mask
Step 6:  Coordinate remapping — transform child masks from crop space to
         full image space, validate dimensions
Step 7:  VitMatte alpha refinement — generate soft alpha matte per layer,
         save 4-channel TIFF
Step 8:  Mask cleanup — bilateral filter, morphological close,
         remove small contours (<64 px)
Step 9:  VTracer vectorization — bitmap-to-SVG spline fitting per mask
Step 10: SVG assembly — group paths by layer hierarchy, write final SVG,
         update state to COMPLETE
```

### Batch Processing Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        BatchRunner                             │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  ProcessPoolExecutor                    │   │
│  │                                                         │   │
│  │   Worker 1        Worker 2        Worker 3        ...   │   │
│  │   ┌─────────┐     ┌─────────┐     ┌─────────┐           │   │
│  │   │Orchestr.│     │Orchestr.│     │Orchestr.│           │   │
│  │   │ Image A │     │ Image B │     │ Image C │           │   │
│  │   └─────────┘     └─────────┘     └─────────┘           │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                 │
│                              v                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     StateManager                        │   │
│  │                                                         │   │
│  │   SQLiteDict: {job_id: JobRecord(status, result, ...)}  │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## ML Models

### VLM Semantic Interrogation (Ollama or llama.cpp)

**Purpose**: Analyzes images to identify objects and their constituent parts.

**Architecture**: Local vision-language models behind a pluggable transport
(`models/vlm_client.py`). Two interchangeable backends:

| Backend | API | Default URL | Default model |
|---------|-----|-------------|---------------|
| Ollama | Ollama HTTP API | `http://localhost:11434` | `qwen2.5vl:3b` |
| llama.cpp | OpenAI-compatible `/v1/chat/completions` | `http://localhost:8080` | whatever the server has loaded (e.g. Qwen3-VL-8B) |

**Input**: Image (base64 PNG; sent as a data URI on the llama.cpp backend)

**Output**: Structured label list with parent/child relationships

**Fallback Chain** (Ollama backend):
1. Primary VLM (`qwen2.5vl:3b` or configured)
2. Fallback VLMs (`gemma4:e4b`, `minicpm-v` or configured)
3. Reasoner model (`gemma4:e4b`) for ranking complex scenes

The llama.cpp backend serves one loaded model, so it has no fallback chain
and the same model doubles as the text reasoner.

**Configuration**: Preferences → Models (backend, URLs, model names)

### GroundingDINO (Object Detection)

**Purpose**: Converts text prompts to precise bounding boxes.

**Architecture**: Vision-language detection model (Swin-T backbone).

**Input**: Image + text prompt

**Output**: Bounding boxes with confidence scores

**Configuration**:
- Confidence threshold: 0.35 (adjustable)
- Text threshold: 0.25 (adjustable)
- Model: `groundingdino_swint_ogc.pth`

### SAM 2.1 HQ (Segmentation)

**Purpose**: Generates high-quality segmentation masks from bounding boxes.

**Architecture**: Segment Anything Model 2.1 with High Quality outputs.

**Input**: Image + bounding box prompt

**Output**: Binary segmentation mask

**Features**:
- Hiera Large backbone for precision
- MPS acceleration on Apple Silicon
- Automatic mask refinement
- Multi-mask output mode for manual bboxes (selects highest fill ratio)

**Configuration**:
- Model: `sam2.1_hiera_large.pt`
- Device: MPS (Metal Performance Shaders)

### VitMatte (Alpha Matting)

**Purpose**: Refines mask edges for hair, fur, and transparent objects.

**Architecture**: Vision Transformer for natural image matting.

**Input**: Image + trimap (rough mask)

**Output**: Alpha matte (soft mask)

**Configuration**:
- Model: `vitmatte-base-composition-1k`
- Device: MPS or CPU fallback

---

## User Interface

### Main Window

The main window provides the application shell with:

- **Title Bar**: Application branding and window controls
- **Mode Switcher**: Segmented control for Single/Batch modes
- **Content Area**: Swappable view for current mode
- **Preferences Access**: Settings modal via gear icon

### Single Image Mode

Three-panel layout:

**Left Panel (Controls)**:
- Image information display
- Label list with checkboxes
- Parameter controls (VTracer sliders)
- Action buttons (Process, Export)

**Center Panel (Canvas)**:
- Image preview with zoom/pan (Fit, +/- buttons, scroll wheel, keyboard shortcuts)
- Scrollbar navigation when zoomed in (horizontal and vertical)
- Layer overlay toggles (Original, Masks, Vectors, Composite)
- Real-time mask visualization
- Color-coded layer display
- Fixed-size detection labels (do not scale with zoom)

**Right Panel (Layers)**:
- Hierarchical layer list
- Visibility toggles
- Layer ordering
- Individual export buttons

### Batch Mode

Six-step wizard with sidebar navigation:

**Step 1 - Import**:
- Folder selection dialog
- Template loading
- Recursion depth setting
- File extension filter

**Step 2 - Configure**:
- Output format selection
- VTracer parameter tuning
- Naming convention options

**Step 3 - Interrogate**:
- Progress bar for Moondream scans
- Parallel processing indicator
- Estimated time remaining

**Step 4 - Triage**:
- Image thumbnail grid
- Label editing interface
- Accept/Reject buttons
- Cannot proceed without triage

**Step 5 - Progress**:
- Real-time progress tracking
- Per-image status display
- Error logging
- Pause/Resume controls

**Step 6 - Output**:
- Summary statistics
- Success/failure counts
- Export bundle options
- Retry failed jobs

### Preferences Modal

Six-tab preferences interface:

1. **General**: Default output directory, default mode, session and notification settings
2. **Models**: VLM backend selector (Ollama / llama.cpp), server URLs, model selection, connection test, model library directory, installed models list
3. **Pipeline**: SAM thresholds, VTracer parameters, bilateral filter, worker count, interrogation profile and fallback mode
4. **Appearance**: Theme, scan preview defaults, box/heatmap opacity, canvas background, mask overlay opacity, vector overlay colour
5. **Templates**: Saved batch templates management
6. **Domain Guides**: Create, edit, and save domain guide TOML files with live preview; two-pane editor with scrollable form and real-time TOML output

---

## Configuration

### Preferences File

Location: `~/.config/skiagrafia/preferences.json`

```json
{
    "output_directory": "~/Desktop/skiagrafia_out",
    "models_directory": "~/ai/claudecode/mozaix/models",
    "default_confidence_threshold": 0.35,
    "vtracer_params": {
        "colormode": "color",
        "hierarchical": "stacked",
        "mode": "spline",
        "filter_speckle": 4,
        "color_precision": 8,
        "layer_difference": 16,
        "corner_threshold": 60,
        "length_threshold": 4.0,
        "max_iterations": 10,
        "splice_threshold": 45,
        "path_precision": 3
    },
    "vlm_backend": "ollama",
    "ollama_url": "http://localhost:11434",
    "ollama_model": "qwen2.5vl:3b",
    "llamacpp_url": "http://localhost:8080",
    "llamacpp_model": "Qwen3-VL-8B-Instruct",
    "preferred_fallback_vlm": "gemma4:e4b",
    "preferred_text_reasoner": "gemma4:e4b",
    "device": "mps"
}
```

Legacy preferences that still name the old shipped defaults
(`moondream`, `qwen3.5`) are migrated automatically to the current
defaults on load; deliberate custom model choices are never changed.

### Environment Variables

The `run.sh` script sets:

```bash
# Force local inference (block HuggingFace downloads)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Disable TensorFlow/Flax imports
export USE_TF=0
export USE_FLAX=0

# PyTorch MPS fallback
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Prevent stale .pyc caches from masking source changes
export PYTHONDONTWRITEBYTECODE=1

# Cairo library path for PDF export (macOS)
export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib"
```

### Template Files

Templates are stored in `~/.config/skiagrafia/templates/` as JSON files:

```json
{
    "name": "Product Photography",
    "created": "2024-01-15T10:30:00Z",
    "config": {
        "output_formats": ["svg", "tiff"],
        "recursion_depth": 1,
        "file_extensions": [".jpg", ".png"],
        "vtracer_params": {...}
    }
}
```

---

## Output Formats

### SVG (Scalable Vector Graphics)

Multi-layer SVG with grouped paths:

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1920 1080">
  <g id="monitor">
    <path d="..." fill="#333333"/>
    <g id="monitor_screen">
      <path d="..." fill="#1a1a1a"/>
    </g>
    <g id="monitor_stand">
      <path d="..." fill="#666666"/>
    </g>
  </g>
</svg>
```

### TIFF (Tagged Image File Format)

4-channel RGBA with alpha matte:

- **Channels**: Red, Green, Blue, Alpha
- **Bit Depth**: 8-bit per channel
- **Compression**: LZW
- **Naming**: `{image_name}_{label}.tiff`

### PNG (Portable Network Graphics)

Standard image format with optional alpha:

- **Channels**: RGBA
- **Bit Depth**: 8-bit per channel
- **Compression**: Deflate
- **Naming**: `{image_name}_{label}.png`

### PDF (Portable Document Format)

Vector PDF export via CairoSVG:

- **Format**: Vector paths preserved
- **Compatibility**: PDF 1.4+
- **Naming**: `{image_name}.pdf`

---

## API Reference

### Core Classes

#### Orchestrator (v5.2)

```python
from core.orchestrator import Orchestrator
from core.factory import build_capabilities
from utils.preferences import load_preferences

# Build capabilities from preferences
prefs = load_preferences()
capabilities = build_capabilities(
    prefs,
    corner_threshold=60,
    length_threshold=4.0,
    filter_speckle=4,
)

# Initialize with injected capabilities
orchestrator = Orchestrator(capabilities=capabilities)

# Run pipeline
result = orchestrator.process(
    image_path=Path("/path/to/image.jpg"),
    confirmed_labels=["monitor", "keyboard"],
    manual_detections=[],
)

# Access results
for layer in result.layers:
    print(layer.label, layer.svg_path)
    layer.mask  # numpy array
    layer.alpha  # numpy array
```

#### CapabilitySet

```python
from core.contracts import CapabilitySet
from core.factory import build_capabilities

# Build from preferences
capabilities: CapabilitySet = build_capabilities(prefs)

# Access individual capabilities
interrogator = capabilities.interrogator
detector = capabilities.detector
segmenter = capabilities.segmenter
alpha_refiner = capabilities.alpha_refiner
vectorizer = capabilities.vectorizer
```

#### ModelManager

```python
from utils.model_manager import ModelManager
from utils.preferences import get_models_dir, load_preferences

prefs = load_preferences()
models_dir = get_models_dir(prefs)
manager = ModelManager(models_dir)

# Resolve model path
dino_path = manager.resolve("groundingdino_swint_ogc.pth")

# Ensure model is downloaded
sam_path = manager.ensure("sam2.1_hiera_large.pt")

# Check availability
if manager.is_available("vitmatte-base-composition-1k"):
    print("VitMatte ready")

# Scan all models
for info in manager.scan():
    print(f"{info.display_name}: {info.status}")
```

#### GuidedInterrogator

```python
from core.factory import build_interrogation_settings
from core.interrogation import GuidedInterrogator, InterrogationSettings
from utils.preferences import load_preferences

# Preferred: derive settings from preferences (backend-aware)
settings = build_interrogation_settings(load_preferences())

# Or construct explicitly — e.g. against a llama.cpp server:
settings = InterrogationSettings(
    host="http://localhost:8080",
    primary_vlm="Qwen3-VL-8B-Instruct",
    fallback_vlms=[],               # one server = one loaded model
    reasoner_model="Qwen3-VL-8B-Instruct",
    backend="llamacpp",             # or "ollama"
    profile="balanced",
    fallback_mode="adaptive_auto",
    enable_tiling=True,
    max_aliases_per_object=4,
)

interrogator = GuidedInterrogator(settings)

# Interrogate image
result = interrogator.interrogate(
    image=image_array,
    confirmed_labels=["monitor"],
    knowledge_pack=knowledge_pack,
)

# Access results
for candidate in result.candidates:
    print(f"{candidate.canonical_label} ({candidate.role})")
    print(f"  Detector phrases: {candidate.detector_phrases}")
    print(f"  Confidence: {candidate.confidence}")

# Access parent-child relationships
for parent, children in result.children_by_parent.items():
    print(f"Parent: {parent} -> Children: {children}")
```

---

## Troubleshooting

### Ollama Connection Failed

**Problem**: Application warns "VLM backend not reachable" (Ollama backend)

**Solution**:
```bash
# Verify Ollama is running
ollama serve

# Check model is pulled
ollama pull qwen2.5vl:3b

# Verify connectivity
curl http://localhost:11434/api/tags
```

### llama.cpp Server Not Ready

**Problem**: "Test connection" fails with the llama.cpp backend selected

**Solution**:
```bash
# Start the server with a multimodal model (needs vision projector)
llama-server -hf Qwen/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M --port 8080 -c 8192

# The server answers 503 while loading — wait for:
curl http://localhost:8080/health     # -> {"status":"ok"}

# Confirm the loaded model
curl http://localhost:8080/v1/models
```

Note: vision requires a model with a multimodal projector (mmproj). Plain
text-only GGUFs will answer text prompts but fail on images.

### Model Weights Missing

**Problem**: "Model file not found" error

**Solution**:
1. Check configured models directory in Preferences
2. Use the first-run setup wizard or Preferences → Models → Download missing
3. Fresh-install default location: `~/Library/Application Support/skiagrafia/models/`

### MPS Not Available

**Problem**: Slow inference on Apple Silicon

**Solution**:
```bash
# Enable MPS fallback
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Verify MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"
```

### Drag-and-Drop Not Working

**Problem**: Cannot drop images onto canvas

**Solution**:
```bash
# Install tkinterdnd2
pip install tkinterdnd2

# Verify Tk installation
python -m tkinter
```

### PDF Export Fails

**Problem**: CairoSVG errors on PDF generation

**Solution**:
```bash
# Install libcairo via Homebrew
brew install cairo libffi

# Set library path
export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib"
```

---

## License
MIT Licence

Built with ❤️ for the design community.
=======

Released under the MIT License. See [LICENSE].

