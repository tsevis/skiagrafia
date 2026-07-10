# Skiagrafia User Manual

*Version 0.3.1*

Skiagrafia turns photographs into layered, masked, vectorized artwork using
ML models that run entirely on your Mac. This manual walks through everything
from first launch to batch production.

---

## Contents

1. [Starting the App](#1-starting-the-app)
2. [First-Run Setup](#2-first-run-setup)
3. [Choosing a VLM Backend](#3-choosing-a-vlm-backend)
4. [Single Image Mode](#4-single-image-mode)
5. [Batch Mode](#5-batch-mode)
6. [Domain Guides](#6-domain-guides)
7. [Preferences Reference](#7-preferences-reference)
8. [Tips & Recipes](#8-tips--recipes)
9. [FAQ](#9-faq)

---

## 1. Starting the App

Always launch with the shell script — it sets the environment the ML stack
needs (offline mode, MPS fallback, cairo paths):

```bash
./run.sh
```

The window opens in **Single** mode. Switch between Single and Batch with
the segmented control in the title bar. A log of everything the app does is
written to `~/.config/skiagrafia/skiagrafia.log`.

---

## 2. First-Run Setup

On a machine that already has the model library, the app starts silently and
is ready to use — nothing to do.

On a fresh install, a **First-run setup** window opens automatically a moment
after launch. It shows a checklist:

| Column | Meaning |
|--------|---------|
| Component | Weight file, source checkout, backend server, or Ollama model |
| Approx. size | Download size estimate |
| Status | ✓ ready · ✕ missing |

- **Download missing** — fetches all missing weights (GroundingDINO, SAM 2.1,
  VitMatte, Grounded-SAM-2 source) and pulls the required Ollama models.
  A progress bar tracks each file.
- **Get Ollama…** — opens the Ollama download page if the server itself is
  not installed. Start it with `ollama serve`, then press **Recheck**.
- **Continue anyway** — closes the wizard; you can finish later from
  Preferences → Models → Download missing.

Items marked *(optional)* (the fallback VLMs) improve quality but are not
required to run.

---

## 3. Choosing a VLM Backend

The "scan" step (identifying what's in the image) runs on a local
vision-language model. Two backends are supported — pick one in
**Preferences → Models → VLM backend**:

### Ollama (default)

- Runs at `http://localhost:11434`.
- Primary model **qwen2.5vl:3b** — fast and accurate for object listing.
- Falls back to **gemma4:e4b**, then **minicpm-v** when the primary result
  is weak; **gemma4:e4b** also ranks results as the text reasoner.
- Nothing to configure if Ollama is already running.

### llama.cpp server

- Runs at `http://localhost:8080`, speaks the OpenAI chat-completions API.
- Start it yourself with a **multimodal** model, for example:

  ```bash
  llama-server -hf Qwen/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M --port 8080 -c 8192
  ```

- One server = one loaded model. The loaded model answers every vision and
  text request; there is no fallback chain.
- Keep `-c 8192` (or similar) — the server's default context size can run
  Metal out of GPU memory during image processing.
- Qwen3-VL-8B is noticeably stronger than the 3B Ollama default — prefer
  this backend when you want the best labels and have the RAM.

Use **Test connection** in Preferences → Models to verify either backend.
Avoid running big models on both backends at the same time — they compete
for GPU memory.

---

## 4. Single Image Mode

The window has three panels: controls (left), canvas (center), layers (right).

### Step by step

1. **Load an image.** Drag it onto the drop zone or click to browse.
   JPEG, PNG, TIFF, BMP, and WebP are supported.

2. *(Optional)* **Load a domain guide** with the "Load guide" button if you
   have a TOML knowledge pack for this kind of image (see
   [Domain Guides](#6-domain-guides)).

3. **Scan.** The VLM proposes labels for the main objects. Detected boxes
   appear on the canvas (semi-transparent, 40% opacity by default) with a
   confidence heatmap. Boxes covering the same object are deduplicated
   automatically.

4. **Review the labels.**
   - Remove wrong labels with their ✕ button.
   - Add labels the model missed — type the name; the pipeline will find it
     with GroundingDINO even if the scan didn't.
   - You can also draw a **manual box** on the canvas for stubborn objects.

5. **Adjust parameters** (all have sensible defaults):
   - *SAM box / text thresholds* — lower finds more objects, higher is stricter.
   - *VTracer corner / speckle / length* — vector smoothness vs. fidelity.
   - *Bilateral filter* — mask edge cleanup strength.

6. **Process.** The 10-step pipeline runs: detection → segmentation →
   child parts → alpha matting → vectorization. Progress appears in the
   status area.

7. **Inspect layers** in the right panel. Toggle mask and vector overlays on
   the canvas; each parent object and its parts are separate layers.

8. **Export** — SVG (layered vectors), TIFF (per-layer RGBA with alpha),
   PNG, or PDF.

---

## 5. Batch Mode

Batch mode is a six-step wizard for processing whole folders. The sidebar
tracks your position; the bottom bar shows status and progress.

1. **Import** — drop or select a folder of images. Recent batches can be
   resumed; state is stored per batch in `state.db`, so a stopped run
   continues where it left off.

2. **Configure** — output mode (vector, bitmap, or both), recursion depth for
   child parts, VTracer quality, interrogation profile, fallback mode,
   preferred VLM and text reasoner. You can create a quick domain guide here
   or attach an existing one.

3. **Interrogate** — the VLM scans every image and builds a live tag cloud
   of detected labels. Blue tags are parents, green are children.

4. **Triage** — *mandatory human gate.* Review the aggregated labels and
   confirm exactly which objects the batch should extract. Nothing is
   processed until you approve.

5. **Progress** — parallel processing across CPU workers with per-image
   thumbnails, throughput (images/min), and ETA.

6. **Output** — summary, failures with reasons, and retry.

Templates: any batch configuration can be saved as a reusable template
(managed in Preferences → Templates).

---

## 6. Domain Guides

A domain guide is a small TOML "knowledge pack" that teaches the scanner
what to expect in a specific image domain (e.g., vintage electronics, Greek
pottery, sneakers).

Edit guides in **Preferences → Domain Guides**:

- **Domain** — name and description injected into every scan prompt.
- **Objects** — per object: canonical name, aliases, detector phrases
  (what GroundingDINO is asked to find), and child parts.
- **Batch defaults** — preferred VLM, tiling on/off, fallback chain.
- The TOML preview updates live; guides are stored wherever you save them
  (batch folders get `skiagrafia_guide.toml`).

When a guide is loaded, scans are *guided*: exemplar object families are fed
to the VLM, unknown-term fallbacks map to your canonical names, and known
child parts skip redundant VLM queries.

---

## 7. Preferences Reference

`Preferences` (⌘,) — six tabs. Settings live in
`~/.config/skiagrafia/preferences.json`.

| Tab | Key settings |
|-----|--------------|
| **General** | Output directory, startup mode, session save, notifications |
| **Models** | VLM backend (ollama / llamacpp), server URLs, primary/fallback/reasoner models, connection test, model library directory, installed-model list, Download missing |
| **Pipeline** | SAM box/text thresholds, VTracer corner/speckle/length, bilateral filter, CPU workers, interrogation profile (`fast` / `balanced` / `deep`), fallback mode, tiled fallback |
| **Appearance** | Theme, scan preview boxes/labels/heatmap and opacities, canvas background, mask overlay opacity, vector overlay colour |
| **Templates** | Saved batch templates |
| **Domain Guides** | Guide editor with live TOML preview |

Interrogation profiles:

- **fast** — one pass, no reasoner, 1 child query. Best for large batches of
  similar images.
- **balanced** — escalates only when confidence is low. Default.
- **deep** — always runs every stage including the reasoner. Best for complex
  scenes and unfamiliar domains.

---

## 8. Tips & Recipes

- **Weak labels?** Switch the backend to llama.cpp with Qwen3-VL-8B, or set
  the profile to `deep`, or write a small domain guide — in that order of
  effort.
- **Missed small objects?** Keep *tiled fallback* enabled; the image is
  re-scanned in quadrants when the full-frame scan is weak.
- **Object detected but mask covers only part of it?** Draw a manual box —
  manual boxes use SAM's multi-mask output and pick the fullest mask.
- **Soft edges (hair, fur, glass)?** That's VitMatte's job — it runs
  automatically; export TIFF to get the soft alpha.
- **Batch throughput** — set CPU workers to physical-core count minus 2;
  each worker loads its own model copies, so RAM is the real limit.
- **GPU memory pressure** — `ollama ps` shows what Ollama holds resident;
  models auto-unload after a few minutes of idle.

---

## 9. FAQ

**Does anything leave my machine?**
No. Inference is 100% local; `run.sh` even blocks HuggingFace network access.
The only network use is downloading models on first setup.

**Where do outputs go?**
Single mode: the output directory set in Preferences → General
(default `~/Desktop/skiagrafia_out`). Batch mode: `<output>/<batch_id>/`.

**Can I use my own model names?**
Yes — the Models tab comboboxes are editable; type any Ollama model tag.
Custom choices are never overwritten by upgrades.

**What happened to Moondream?**
It was the old default and still works (`moondream` in the model list), but
`qwen2.5vl:3b` is substantially better and became the default in 0.3.x.
Old preferences are migrated automatically.

**The setup wizard opened but I already have the models.**
Point *Model library directory* (Preferences → Models) at your existing
library and press Recheck in the wizard.

**Can I script the pipeline without the GUI?**
Yes — see the API Reference section of the [README](../README.md).
