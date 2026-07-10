# Skiagrafia

> **Semantic Vectorizing and Masking Creator**
>
> A desktop application for AI-powered image segmentation, masking, and vectorization.

[![Version](https://img.shields.io/badge/version-0.3.1-blue.svg)](https://github.com/tsevis/skiagrafia)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey.svg)](https://support.apple.com/en-us/116943)
[![Architecture](https://img.shields.io/badge/architecture-v5.2-orange.svg)](FILE_STRUCTURE.md)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Skiagrafia** uses local ML models to automatically segment images into semantic layers (objects and their parts), producing both vector (SVG) and bitmap (TIFF/PNG) outputs. Designed for designers and production workflows. 100% local inference — no cloud APIs.

<p align="center">
  <img src="docs/skiagrafia-readme.jpg" alt="Skiagrafia desktop interface showing semantic segmentation and masking layers" width="1200">
</p>

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/tsevis/skiagrafia.git && cd skiagrafia
python -m venv venv && source venv/bin/activate
pip install -e .

# VLM backend, option A — Ollama (default)
ollama serve &
ollama pull qwen2.5vl:3b

# VLM backend, option B — llama.cpp server (OpenAI-compatible API)
llama-server -hf Qwen/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M --port 8080 -c 8192

# Run the application
./run.sh
```

On first launch, a **setup wizard** checks for missing model weights
(GroundingDINO, SAM 2.1, VitMatte, Grounded-SAM-2 source) and downloads them
into your models directory with one click. Machines that already have the
models skip the wizard entirely.

📖 **New here? Read the [User Manual](docs/MANUAL.md).**

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [System Requirements](#system-requirements)
4. [Installation](#installation)
5. [VLM Backends](#vlm-backends)
6. [Operating Modes](#operating-modes)
7. [Architecture](#architecture)
8. [ML Pipeline](#ml-pipeline)
9. [Configuration](#configuration)
10. [Output Formats](#output-formats)
11. [API Reference](#api-reference)
12. [Troubleshooting](#troubleshooting)
13. [License](#license)

---

## Overview

Skiagrafia addresses a common challenge in design and production workflows: converting raster images into clean, layered vector artwork with precise masks. Traditional approaches require manual tracing and masking — time-consuming and error-prone. Skiagrafia automates the process with state-of-the-art machine learning models running entirely on your machine.

### Key Capabilities

- **Semantic Understanding** — automatically identifies objects and their constituent parts (e.g., a monitor with screen, stand, and bezel)
- **Two Local VLM Backends** — Ollama (default) or a llama.cpp server, switchable in Preferences
- **Precision Segmentation** — SAM 2.1 HQ for high-quality mask generation
- **Alpha Matting** — VitMatte refinement for hair, fur, and soft edges
- **Vector Output** — VTracer converts bitmaps to clean SVG paths
- **Batch Processing** — process thousands of images with a wizard-driven workflow
- **First-Run Bootstrap** — fresh installs download everything they need automatically

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

- **Qwen2.5-VL / Qwen3-VL** — semantic interrogation via Ollama or llama.cpp, with dual-prompt strategy and fallback chain
- **GroundingDINO** — text-to-bounding-box detection with scan-stage deduplication
- **SAM 2.1 HQ** — state-of-the-art segmentation with multi-mask output for manual bboxes
- **VitMatte** — high-quality alpha matting
- **VTracer** — bitmap-to-SVG spline fitting

### Domain Guides

- TOML-based knowledge packs that teach the VLM which objects to look for
- Built-in GUI editor inside Preferences (Domain Guides tab)
- Canonical names, aliases, detector phrases, and child parts per object
- Live TOML preview as you edit
- Batch defaults (preferred VLM, tiling, fallback chain) per domain

### Technical

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
| **Memory** | 32 GB | 64–128 GB unified memory |
| **Storage** | 20 GB free | 50 GB for model weights |
| **GPU** | MPS capable | Metal Performance Shaders |

### Software

- **macOS** 12.0 (Monterey) or later
- **Python** 3.11 or later
- **One VLM backend**:
  - [Ollama](https://ollama.com) at `http://localhost:11434` (default), or
  - [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` at `http://localhost:8080`

### Model Weights

All downloadable by the first-run wizard (or Preferences → Models → Download missing):

| Model | Source | Size |
|-------|--------|-----:|
| Grounded-SAM-2 source (code + configs) | GitHub archive | ~30 MB |
| GroundingDINO SwinT-OGC | official release | ~660 MB |
| SAM 2.1 Hiera Large | official release | ~900 MB |
| VitMatte ViT-B Composition-1K | HuggingFace | ~380 MB |
| Qwen2.5-VL 3B (`qwen2.5vl:3b`) | Ollama pull | ~3.2 GB |
| Gemma 4 E4B (`gemma4:e4b`) — fallback + reasoner | Ollama pull | ~9.6 GB |
| Qwen3-VL 8B GGUF — llama.cpp alternative | llama.cpp `-hf` cache | ~6.5 GB |

---

## Installation

### 1. Clone and Install

```bash
git clone https://github.com/tsevis/skiagrafia.git
cd skiagrafia
python -m venv venv
source venv/bin/activate
pip install -e .
```

### 2. Set Up a VLM Backend

**Option A — Ollama (default):**

```bash
brew install ollama
ollama serve

ollama pull qwen2.5vl:3b      # primary VLM (required)
ollama pull gemma4:e4b        # fallback VLM + text reasoner (recommended)
ollama pull minicpm-v         # second fallback (optional)
```

**Option B — llama.cpp server:**

```bash
brew install llama.cpp

# Start with a multimodal model (cached after the first download)
llama-server -hf Qwen/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M --port 8080 -c 8192
```

### 3. First Launch

```bash
./run.sh
```

The setup wizard opens automatically **only if something is missing**: it
lists every required component with its size, downloads the weights, and can
pull the Ollama models. The models directory defaults to an existing shared
library when one is present, otherwise
`~/Library/Application Support/skiagrafia/models/` — configurable in
Preferences → Models.

---

## VLM Backends

Semantic interrogation runs on one of two interchangeable local backends,
selected in **Preferences → Models → VLM backend**:

| | Ollama | llama.cpp server |
|---|---|---|
| API | Ollama HTTP | OpenAI-compatible `/v1/chat/completions` |
| Default URL | `http://localhost:11434` | `http://localhost:8080` |
| Default model | `qwen2.5vl:3b` | whatever the server has loaded |
| Fallback chain | `gemma4:e4b` → `minicpm-v` | — (one loaded model) |
| Text reasoner | `gemma4:e4b` | the same loaded model |

Notes on llama.cpp:

- One server hosts **one** loaded model; the model name in preferences is informational.
- Vision requires a model with a multimodal projector (mmproj) — e.g. `Qwen/Qwen3-VL-8B-Instruct-GGUF`.
- Use `-c 8192` — the server's default context can exhaust Metal GPU memory on vision models.
- The server answers HTTP 503 while loading; the app's health check handles this.

---

## Operating Modes

### Single Image Workflow

1. **Drop an image** onto the canvas area
2. **Scan** — the VLM analyzes the image and proposes labels
3. **Review labels** — confirm, adjust, or add objects
4. **Process** — run the 10-step pipeline to generate masks
5. **Export** — SVG, TIFF, PNG, or PDF

### Batch Workflow

1. **Import** — select a folder or load a template
2. **Configure** — output options, VTracer parameters, interrogation settings
3. **Interrogate** — the VLM scans all images (parallel)
4. **Triage** — review and confirm labels (mandatory gate)
5. **Process** — run the full pipeline on all images
6. **Output** — review results and retry failures

See the [User Manual](docs/MANUAL.md) for a full walkthrough of both modes.

---

## Architecture

v5.2 implements contract-based dependency injection with five capability protocols:

```
┌─────────────────────────────────────────────────────────────────┐
│                     UI Layer                                    │
│  main_window.py · left_panel.py · canvas_panel.py               │
│  step_progress.py · batch_runner.py                             │
│                                                                 │
│  Reads preferences → builds concrete clients → injects          │
└────────────────────────┬────────────────────────────────────────┘
                         │ build_capabilities(prefs)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Factory (factory.py)                               │
│  build_interrogation_settings() · build_capabilities()          │
│  All "which model, which path, which threshold" decisions.      │
└────────────────────────┬────────────────────────────────────────┘
                         │ returns CapabilitySet
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Capability Protocols (contracts.py)                │
│  Interrogator · Detector · Segmenter · AlphaRefiner ·           │
│  Vectorizer                                                     │
└────────────────────────┬────────────────────────────────────────┘
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
└─────────────────────────────────────────────────────────────────┘
```

### Capability Protocols

| Protocol | Method | Concrete Implementation |
|----------|--------|------------------------|
| `Interrogator` | `interrogate()` | `GuidedInterrogator` (Ollama / llama.cpp VLM + fallbacks) |
| `Detector` | `detect_box()` | `GroundedSAM` (GroundingDINO) |
| `Segmenter` | `segment()`, `clear_cache()` | `GroundedSAM` (SAM 2.1 HQ) |
| `AlphaRefiner` | `predict()` | `VitMatteRefiner` |
| `Vectorizer` | `trace()` | `VTracerVectorizer` |

See [FILE_STRUCTURE.md](FILE_STRUCTURE.md) for the complete module map.

---

## ML Pipeline

The `Orchestrator` executes a 10-step structural pipeline:

```
Step 1:  Load image as numpy array (BGR → RGB)
Step 2:  VLM interrogation — get parents, filter against confirmed labels,
         get children per confirmed parent
Step 3:  GroundingDINO detection — text-to-bounding-box per parent label,
         scan-stage bbox dedup (IoU + containment)
Step 4:  SAM 2.1 HQ parent segmentation — mask from bbox prompt, multi-mask
         output for manual bboxes, tighten bbox from mask contour
Step 5:  SAM child segmentation — crop region, detect + segment per child,
         child mask validation, boolean-subtract children from parent body
Step 6:  Coordinate remapping — child masks from crop space to image space
Step 7:  VitMatte alpha refinement — soft alpha matte per layer, 4-ch TIFF
Step 8:  Mask cleanup — bilateral filter, morphological close, despeckle
Step 9:  VTracer vectorization — bitmap-to-SVG spline fitting per mask
Step 10: SVG assembly — group paths by layer hierarchy, write final SVG
```

### VLM Interrogation Detail

The `GuidedInterrogator` escalates through stages until it has confident labels:

1. **Composition pass** — whole-scene understanding first
2. **Primary pass** — direct object listing (`qwen2.5vl:3b` by default)
3. **Guided pass** — domain-guide exemplars injected into the prompt
4. **Fallback VLMs** — `gemma4:e4b`, then `minicpm-v` (Ollama backend only)
5. **Tiled pass** — quadrant crops for small objects
6. **Reasoner ranking** — `gemma4:e4b` ranks and refines detector phrases

Profiles: `fast` (skip reasoner, 1 child query) · `balanced` (default) · `deep` (always reason, 5 child queries).

---

## Configuration

### Preferences File

`~/.config/skiagrafia/preferences.json` — managed by the Preferences window:

```json
{
    "vlm_backend": "ollama",
    "ollama_url": "http://localhost:11434",
    "ollama_model": "qwen2.5vl:3b",
    "llamacpp_url": "http://localhost:8080",
    "llamacpp_model": "Qwen3-VL-8B-Instruct",
    "preferred_fallback_vlm": "gemma4:e4b",
    "preferred_text_reasoner": "gemma4:e4b",
    "models_directory": "",
    "output_directory": "~/Desktop/skiagrafia_out",
    "sam_box_threshold": 0.35,
    "sam_text_threshold": 0.25,
    "interrogation_profile": "balanced",
    "interrogation_fallback_mode": "adaptive_auto"
}
```

Legacy preferences naming the old shipped defaults (`moondream`, `qwen3.5`)
are migrated automatically to the current defaults once, on load. Deliberate
custom model choices are never changed.

### Environment Variables

`run.sh` sets these (recommended way to launch):

```bash
export HF_HUB_OFFLINE=1                  # force local inference
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export USE_TF=0 USE_FLAX=0               # no TensorFlow/Flax imports
export PYTORCH_ENABLE_MPS_FALLBACK=1     # MPS fallback for unsupported ops
export PYTHONDONTWRITEBYTECODE=1
export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib"   # cairo for PDF export
```

---

## Output Formats

| Format | Contents |
|--------|----------|
| **SVG** | Layered vector paths grouped by parent/child hierarchy |
| **TIFF** | 4-channel (RGBA) bitmap per layer with alpha matte |
| **PNG** | Flattened preview or per-layer export |
| **PDF** | Vector output via cairosvg (requires Homebrew libcairo) |

Batch runs write to `<output_dir>/<batch_id>/` with a `state.db` for resume.

---

## API Reference

### Building a pipeline programmatically

```python
from core.factory import build_capabilities
from core.orchestrator import Orchestrator
from utils.preferences import load_preferences

prefs = load_preferences()
caps = build_capabilities(prefs)
orchestrator = Orchestrator(capabilities=caps)
result = orchestrator.process("photo.jpg", confirmed_labels=["monitor"], manual_detections=None)
```

### Interrogation only

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
)

interrogator = GuidedInterrogator(settings)
result = interrogator.interrogate(image_array)
for candidate in result.candidates:
    print(candidate.canonical_label, candidate.detector_phrases, candidate.confidence)
```

### Talking to a VLM directly

```python
from models.vlm_client import create_vlm_client

client = create_vlm_client("llamacpp", "http://localhost:8080", "Qwen3-VL-8B-Instruct")
if client.health_check():
    labels = client.get_parents(image_array)          # list[str]
    reply = client.query_text("Rank these labels…")   # str
```

---

## Troubleshooting

### Ollama connection failed

```bash
ollama serve                        # verify Ollama is running
ollama pull qwen2.5vl:3b            # check the model is pulled
curl http://localhost:11434/api/tags
```

### llama.cpp server not ready

```bash
# Start with a multimodal model and a bounded context
llama-server -hf Qwen/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M --port 8080 -c 8192

curl http://localhost:8080/health     # 503 while loading → {"status":"ok"} when ready
curl http://localhost:8080/v1/models  # confirm the loaded model
```

Vision requires a model with a multimodal projector (mmproj); text-only GGUFs
fail on images. If requests return "Compute error", another model may be
holding GPU memory (check `ollama ps`) — free it and restart the server.

### Model weights missing

1. Check the models directory in Preferences → Models
2. Use the first-run wizard or **Download missing**
3. Fresh-install default: `~/Library/Application Support/skiagrafia/models/`

### Slow inference on Apple Silicon

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python -c "import torch; print(torch.backends.mps.is_available())"
```

### PDF export fails

```bash
brew install cairo libffi
export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib"
```

### Drag-and-drop not working

Install `tkinterdnd2` (`pip install tkinterdnd2`); the app falls back to
click-to-browse when it is unavailable.

---

## License

MIT Licence

Built with ❤️ for the design community.
