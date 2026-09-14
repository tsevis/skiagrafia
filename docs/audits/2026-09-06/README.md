# Skiagrafia health and model audit

Audit: 6 September 2026. Code: `35d7afc`. Hardware verified live: Apple M1 Ultra, 128 GiB unified memory. The two supplied model inventories were reference data, not operational instructions. Application code, saved preferences, dependencies and model stores were left unchanged. No model downloads or cloud inference were performed.

**Recommendation:** use **Qwen3-VL 8B Instruct Q4_K_M through llama.cpp** as the leading installed candidate for object discovery, and integrate **the installed MLX SAM 3** for text-to-instance masks. Retain GroundingDINO + SAM 2.1 as a fallback, VitMatte for selected alpha mattes, and VTracer. Fix the demonstrated hierarchy, instance-count and fallback failures before judging larger models on final output quality.

This is a measured shortlist, not a claim of a universal winner. Two convenience images cannot establish object recall, segmentation accuracy or production latency percentiles. Numerical observations and exact inference settings are in [measurements.json](measurements.json).

## Health

| Check | Result |
|---|---|
| Default test suite | 522 passed in 7.88 s |
| Explicit GUI suite | 230 passed in 31.43 s |
| Ruff | All checks passed |
| Current Ollama service | Reachable; all configured model names installed |
| Shared llama.cpp router on port 8080 | Not running; this does not block the configured Ollama backend |
| Registered required model files | All present |
| Live GroundingDINO, SAM 2.1 Large, VitMatte | Loaded and inferred successfully on MPS |
| Full pipeline, existing settings, one 500 × 492 image | 31.777 s, no reported exception; 7 layers, 1 SVG, 9 RGBA TIFFs |
| Export verification | SVG parsed and rendered; TIFFs decoded at the expected dimensions; **duplicate SVG IDs and incorrect child assignments found** |
| Reproducibility | Shared Conda environment; `pip check` fails; dependencies and source archive mostly unpinned |
| Disk | Approximately 201 GiB free, data volume 89% used |

The tests establish a healthy software foundation, not reliable semantic output. Most model unit tests mock inference. The real pipeline test found errors those tests do not catch.

The interpreter selected by the launcher is the existing Miniconda Python 3.12. Important packages observed: torch 2.13.0, torchvision 0.28.0, transformers 4.57.6, MLX 0.31.2, mlx-vlm 0.4.4, Ollama Python client 0.6.0, NumPy 2.3.5, Pillow 12.3.0 and VTracer 0.6.12. The shared environment has unrelated dependency conflicts as well as relevant incompatibilities (for example mlx-vlm expects a newer transformers). OpenCV itself **works**: `opencv-contrib-python` provides it, although packages demanding the separately named `opencv-python`/`opencv-python-headless` distributions complain. Do not “repair” the whole shared environment as part of a model switch; create a tested, isolated environment for this project.

## What actually runs

| Stage | Active implementation |
|---|---|
| Object discovery | Ollama `qwen2.5vl:3b`, Q4_K_M |
| Vision fallbacks | `gemma4:e4b`, then `minicpm-v` |
| Optional text ranking | `gemma4:e4b` |
| Boxes | GroundingDINO SwinT-OGC |
| Masks | Standard SAM 2.1 Hiera Large, **not a distinct HQ checkpoint** |
| Alpha matting | VitMatte Base Composition-1K |
| Vectorization | VTracer, an algorithm rather than an ML model |

Saved preferences and recent startup logs agree on Qwen2.5-VL 3B. `models/moondream_client.py` is a compatibility shim. The misleading “Scan with Moondream” button remains in `ui/single/left_panel_labels.py:41`, and the progress stage still says “Moondream interrogation” in `core/orchestrator.py:60`.

## Correctness and speed findings, in priority order

1. **Child parts can belong to the wrong object, and mask names collide.** Child detection runs against the full image (`core/orchestrator.py:419`) and only needs to overlap an expanded parent box (`:119`). In the real export, `buttons` belonging to the mouse used the keyboard's bounding box. A `mouse button` was also accepted under the keyboard. Masks are stored globally as `masks[child_label]` (`:458`), so identically named parts of different parents overwrite one another. The SVG contained two IDs named `buttons`. Crop/ground parts within their parent, validate actual parent containment, and give every parent/child instance its own stable ID.
2. **Automatic detection discards repeated instances.** `models/grounded_sam.py:295` keeps only `logits.argmax()`. Multiple bags, chairs or monitors with the same label collapse to one returned box. The `Detector` contract also returns a single result. Support a list of detections and instance IDs throughout labels, masks and exports; merely replacing weights cannot solve this.
3. **The Gemma fallback can return no usable answer.** The Ollama transport sends a 200-token limit but does not disable thinking, and reads only `message.content` (`models/vlm_client.py:352`). On both benchmark images, Gemma exhausted all 200 tokens on thinking and returned empty content. Setting `think=false` in an isolated request restored answers. Apply supported model-specific thinking settings and detect output truncation; do not parse hidden reasoning as object labels. The same transport is used for the optional 256-token text reasoner, which has the same structural risk, although that separate prompt was not benchmarked. [Ollama thinking API](https://docs.ollama.com/capabilities/thinking).
4. **“Confidence” does not measure whether objects were found.** Vision responses receive fixed values of 0.72 or 0.8 (`core/interrogation.py:257`). Escalation mainly checks whether at least two non-vague names exist (`:401`). A direct probe escalated one valid `keyboard` but accepted two unsupported labels, `unicorn` and `spaceship`. Detection happens later and does not trigger another visual pass. Use grounding success, image coverage, unmatched guide targets and validated ambiguity to drive fallbacks; allow a valid single-object scene to finish quickly.
5. **The prompt and size limits favor omission.** Composition asks for dominant/grouped objects and discourages tiny details. Normal balanced mode can stop after that pass; each response is capped at 8 parent labels, preview detection at 4 candidates, and automatic parent/child masks under 0.5% of the image are rejected. The Mac sample grouped the monitor and horizontal computer case as “computer”; the case was absent from the exported masks. Make the foreground/part policy explicit, and use cropped or tiled passes for unresolved small objects instead of a larger model on every request.
6. **Image encoding and child queries are repeated unnecessarily.** `segment()` calls `set_image(image)` for every object (`models/grounded_sam.py:335`), recomputing SAM features even though its predictor remains resident. Scan constructs a fresh detector and processing constructs another. Child queries happen before boxes are established and run again during processing after label confirmation. Keep image features per image, batch independent prompts, crop children after parent localization, and reuse validated scan results.
7. **Batch documentation and execution diverge.** The GUI processing path (`ui/batch/steps/step_progress.py:146`) loops sequentially with one resident capability set. The separate `BatchRunner` process-pool path rebuilds models for every submitted image and defaults to CPU-count workers; it is not the GUI's current processing path. Do not infer 20-way GPU throughput from the worker preference. GUI processing also rebuilds interrogation from global preferences rather than all per-batch model/profile overrides used in analysis. Unify configuration and use one resident GPU owner with bounded CPU preprocessing/export workers.

## Is the installed Moondream current?

**Current Ollama tag: yes. Current Moondream generation: no.**

- Local `moondream:latest` digest begins `55fc3abd3867`. That exactly matches Ollama's current `latest`/`v2` entry: the older 1.8B Q4_0 build with 2K context. Re-pulling that tag does not upgrade it to a new generation. [Ollama Moondream tags](https://ollama.com/library/moondream/tags).
- The separate installed GGUF is explicitly `moondream2-20250414`, another older release. It was inspected but not benchmarked.
- The latest upstream release verified is **Moondream 3.1**, announced **7 July 2026**: 9B total / 2B active parameters, native query/detect/point/caption. Its official local path is **Photon**, including Apple Silicon. It is absent from the supplied inventory and the relevant local caches inspected. [Release](https://moondream.ai/blog/moondream-3-1-beyond-benchmarks), [model and local API](https://huggingface.co/moondream/moondream3.1-9B-A2B).
- Moondream 3.1 is the strongest **new-download experiment** for this app: it could provide both names and grounded boxes with reusable image encodings. It needs a Photon adapter and a local benchmark. Its advertised 34.2 requests/s is measured on an H100 at batch 16 with the vendor's evaluation settings, not your M1 Ultra; it is not a desktop latency promise. [Benchmark methodology](https://moondream.ai/blog/moondream-3-1-beyond-benchmarks).

## Installed VLM comparison

Both images used the application's actual composition prompt, a 200-token cap, temperature 0 and seed 42. This compares deployable **model/runtime combinations**: preprocessing and visual token budgets differ. The stock app does not currently set this deterministic sampling policy. Ollama's first request includes model load; llama.cpp was already ready after separately measured startup. Repeated identical requests benefit from cache and should not be advertised as fresh-image latency.

| Configuration | First Mac request | First groceries request | Observed result |
|---|---:|---:|---|
| Qwen2.5-VL 3B / Ollama | 6.10 s, including 4.00 s load | 2.31 s, model resident | Mac: desktop computer, keyboard, mouse. Groceries: one long combined phrase rather than a clean list. |
| **Qwen3-VL 8B Q4_K_M / llama.cpp** | **0.93 s** + 2.06 s server startup | **1.30 s** | Clean short lists in both cases: computer/keyboard/mouse; car/paper bags/cargo area. Leading installed choice. |
| MiniCPM-V 2.6 / Ollama `minicpm-v:latest` | 5.74 s, including 4.15 s load | 1.60 s | Invented a printer in the Mac scene; usable but verbose groceries phrases. |
| MiniCPM-V 4.6 Q4_K_M / llama.cpp | 0.80 s + 2.52 s startup | 2.20 s | Named monitor and computer base separately, but added sentence fragments; groceries response repeated until the 200-token limit. |
| Gemma 4 E4B / current thinking policy | 11.21 s, including 7.59 s load | 3.88 s | Empty final answer on both images; token budget consumed by thinking. |
| Gemma 4 E4B / `think=false` | 4.76 s, including 4.04 s load | 1.10 s | Usable Mac list; omitted the bags in the groceries scene. Useful optional text ranker, not the leading vision fallback here. |
| Old Moondream / Ollama | 2.29 s, including 1.61 s load | 0.54 s | Empty with composition prompt. Shorter fallback prompt returned “urn, mouse” and “ids, bags”. Fast failure is not useful throughput. |

The MiniCPM names are especially misleading: installed Ollama `minicpm-v` is **2.6**, while the cached **4.6** GGUF is a separate, much smaller model built from SigLIP2-400M and Qwen3.5-0.8B. A higher version number does not imply a larger or uniformly more capable replacement. [Ollama build](https://ollama.com/library/minicpm-v:latest), [MiniCPM-V 4.6 model card](https://huggingface.co/openbmb/MiniCPM-V-4.6).

Qwen3-VL's improved visual recognition and grounding make it a sensible candidate beyond these two examples, but those publisher claims do not substitute for project-specific accuracy measurements. [Qwen3-VL 8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).

Other inventory choices:

| Model | Recommendation for this pipeline |
|---|---|
| Qwen3.8-27B Q4_K_M | Candidate for difficult specialist images, not every scan. Inventory reports only 10.9 generated tokens/s on this Mac; that is a previous text-generation measurement, not a vision latency result. Native VLM, but not benchmarked in this audit. Use its intact absolute snapshot if the cache ref is dangling. [Official model](https://huggingface.co/Qwen/Qwen3.8-27B). |
| Gemma 4 12B GGUF | Tested in the follow-up below: 1.09–1.21 s per fresh-image request after startup, usable lists. A stronger fallback candidate than E4B in these samples; no demonstrated overall win over Qwen3-VL. |
| Gemma 4 E4B MLX | Not benchmarked. Requires its appropriate serving path; the project has no MLX VLM adapter. |
| Qwen2.5-VL 7B MLX | Reasonable alternative requiring an MLX serving adapter; lower integration priority than the already supported Qwen3-VL GGUF path. |
| FastVLM 0.5B / 1.5B / 7B | Worth testing for rapid preview/triage; installed variants verified. Needs an appropriate Apple/MLX/CoreML integration. Not evaluated here. [Apple implementation](https://github.com/apple/ml-fastvlm). |
| SmolVLM 500M | Preview/triage candidate; insufficient evidence to entrust final object discovery to it. |
| LLaVA 1.6 7B and older Moondream2 GGUF | No compelling reason to prioritize these over Qwen3-VL for a new integration. |
| Qwen3.6-35B-A3B / Nemotron 3.5 Lightning | Optional text-only ranking through their presently configured routes. Previous inventory throughput numbers are not end-to-end vision timings. Prefer deterministic guide/alias matching, or reuse the resident VLM, before loading another large model. |
| OCR, embedding, depth, restoration, face-only/person-only detectors | Different jobs; they do not replace open-vocabulary instance discovery and matting. |

## Detection, segmentation, matting and vectors

**MLX SAM 3 is the leading installed upgrade for the combined detection-and-mask stage.** It accepts text prompts and returns multiple instance masks, avoiding the separate per-label GroundingDINO → SAM sequence. The exact local port and existing weights loaded successfully in the current Python 3.12 environment; no dependencies were changed. Its package metadata requests Python ≥3.13, so direct-import success is a smoke test, not a reason to ignore its supported environment when integrating it.

Measured on the same samples, threshold 0.5, image input resolution 1008:

- Initial model loading/evaluation: 0.67 s.
- Mac image encoding: 0.62 s; monitor prompt 0.66 s on the first decoder call; keyboard 0.12 s; mouse 0.11 s.
- Groceries encoding: 0.79 s; “paper bag” prompt: 0.13 s. **Four separate masks**, scores 0.893–0.906; approximately **0.92 s including encoding**.
- The mouse prompt produced one strong detection (0.941) and a second mouse-shaped detection at 0.516. Review ambiguous instances and calibrate thresholds on examples; do not globally raise a threshold based on one image.

For comparison, the existing GroundingDINO wrapper took **2.03–2.34 s per warm label**, then SAM Large took another **0.74 s per mask** including the redundant image encoding. These are isolated stage measurements, not a measured end-to-end speedup for an integrated SAM 3 pipeline. Integrating SAM 3 requires new capability wiring, multiple-instance contracts, stable IDs and preservation of manual box interactions. SAM 3's text-guided image capabilities are documented in [Meta's repository](https://github.com/facebookresearch/sam3). The newer **SAM 3.1** release focuses on shared-memory multi-object **video tracking**; that is not evidence that it is a necessary or faster still-image replacement for the installed SAM 3 port. [SAM 3.1 release](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md).

**For a smaller integration change, expose SAM 2.1 Tiny/Small/Base+ choices.** All four checkpoints already exist in the shared model directory, although the app hardcodes Large and its config. Warm encode + mask decode on the same monitor box:

| SAM 2.1 checkpoint | Time | Mask IoU against Large |
|---|---:|---:|
| Large | 0.737 s | Reference |
| Tiny | 0.273 s | 0.987 |
| Small | 0.396 s | 0.980 |
| Base+ | 0.477 s | 0.992 |

Agreement with Large is **not ground-truth accuracy**: the masks differed around the monitor stand. These are single-image warm observations, not stable throughput estimates. Tiny is the speed candidate, Base+ the conservative compromise, Large the quality fallback. Fix image-feature reuse before interpreting the benefit of shrinking the model. Meta's official speed numbers use an A100 and must not be substituted for these Mac measurements. [SAM 2 checkpoint comparison](https://github.com/facebookresearch/sam2).

**Keep VitMatte for alpha refinement.** It took 0.35 s warm at 500 × 492, 1.14 s initially. It refines a supplied instance mask; BiRefNet and U²-Net principally predict foreground/background. Neither automatically replaces the instance-specific contract. Apply VitMatte selectively to soft edges/hair/fur rather than every rigid layer. Existing vector-only mode already skips it; it currently affects TIFF alpha, not SVG geometry. Its 1536-pixel working-size cap also needs attention for fine edges in large exports. [ViTMatte](https://github.com/hustvl/ViTMatte), [BiRefNet](https://github.com/ZhengPeng7/BiRefNet).

For a separate “remove the background from the whole subject” feature, BiRefNet is a quality candidate and u2netp a fast preview candidate. The inventory's 96 ms u2netp and 1618 ms BiRefNet measurements used one different subject-concentration experiment; they do not establish equal edge/matting quality for Skiagrafia.

**Keep VTracer.** One binary-mask trace took 0.006 s. The visible failures originate upstream. A generative SVG model would introduce a different task, not repair missing objects or incorrect mask hierarchy.

## Proposed implementation sequence

1. Repair child containment/cropping, stable instance IDs, duplicate SVG IDs, thinking/truncation handling and misleading model labels. Make scan and process honor the same model/profile settings.
2. Add per-stage timing, detection counts and a small labeled regression corpus. Distinguish empty results, unsupported labels and genuine single-object scenes.
3. Use the already supported llama.cpp backend to evaluate Qwen3-VL 8B as the discovery default. Pin the local GGUF and Q8 projector, 8192 context, one slot, reasoning off. A service must be running; the current port-8080 route is down. The temporary audit servers were stopped after testing.
4. Integrate MLX SAM 3 behind explicit capabilities, returning every validated instance; retain GroundingDINO/SAM 2.1 for fallback and manual interaction. Cache image features.
5. Offer a fast SAM 2.1 Tiny profile and selective VitMatte. Keep another large text model out of the normal path unless measurements justify it.
6. Benchmark on 30–50 representative images, including multiple identical objects, small accessories, rare objects, occlusion and soft boundaries. Measure object precision/recall, per-instance mask/boundary quality, invalid output rate, corrections needed, end-to-end cold/warm latency and memory. Two samples are enough to uncover defects and shortlist models, not to certify an accuracy winner.
7. Compare Moondream 3.1/Photon as a separate new-download experiment once the baseline is correct. Adopt it only if it improves the actual latency-versus-correction tradeoff.

The benchmark artifacts record observed configurations and results only. No production model selection was changed by this audit.


## Follow-up: upgrading Gemma 4

The installed Ollama `gemma4:e4b` digest still matches the current registry tag (`c6eb396dbd59`), so re-pulling it is not a model-capability upgrade. Other variants exist, including 12B, 26B A4B and 31B. The user already has **Gemma 4 12B Q4_K_M plus its Q8 projector** in the Hugging Face cache. [Current Ollama tags](https://ollama.com/library/gemma4/tags).

The 12B checkpoint was tested through a temporary llama.cpp server with thinking disabled and exactly the original comparison's prompt, token limit, temperature, seed and two images. Server startup took **5.07 seconds**, separately from request timings:

| Sample | First request after model ready | Identical cached repeat | Answer |
|---|---:|---:|---|
| Mac | 1.091 s | 0.385 s | All-in-one computer, keyboard, mouse |
| Groceries | 1.214 s | 0.382 s | open trunk, car, paper bags, groceries |

It found the bags that E4B missed and produced concise answers. It still called the separate monitor/base setup an all-in-one computer, so the larger model is not immune to mistakes. Qwen3-VL's corresponding fresh-image requests were 0.930 and 1.304 seconds; this sample is too small to establish a meaningful overall speed or quality winner. Google's published MMMU Pro comparison favors 12B over E4B (69.1 versus 52.6), but that benchmark is not object-recall accuracy for this application. [Google model card](https://huggingface.co/google/gemma-4-12B-it).

**Revised recommendation:** include the already installed Gemma 4 12B as a serious alternative and the leading Gemma upgrade candidate, especially for a stronger fallback. Keep reasoning disabled for short object lists. Qwen3-VL 8B remains the provisional default candidate; compare both on the project regression corpus. E4B's misconfigured thinking budget was a reason to repair its integration, not sufficient evidence to dismiss the whole Gemma family. The existing factory uses one transport for a run and disables fallbacks in llama.cpp mode, so mixing an Ollama primary with this cached llama.cpp Gemma fallback requires backend wiring. No new weights are needed for the 12B trial, and no production settings were changed.
