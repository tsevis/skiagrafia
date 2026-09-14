# Quality pipeline update — 7 September 2026

The initial health and model audit is in [the audit report](audits/2026-09-06/README.md). The two supplied model inventories were used as reference material. This update implements the subsequent request to improve recognition, masking, transparency and paths.

## Models and runtime

| Stage | Selected model | Behavior |
|---|---|---|
| Recognition and prompt interpretation | Qwen3-VL 8B Instruct, Q4_K_M | Managed local llama.cpp process; installed Q8 vision projector |
| Fallback and deeper review | Gemma4 12B Instruct, Q4_K_M | Separate model route with its installed Q8 projector; automatically selected when needed |
| Text-guided instance masks | MLX SAM 3 | Returns every detected instance and reuses image features between text prompts |
| Whole-object fallback and manual boxes | GroundingDINO + SAM 2.1 Hiera Large | Retained for cases SAM 3 misses; SAM 2 image features are reused across boxes |
| Alpha | VitMatte Base | Object-region inference, constrained known foreground/background, native-resolution boundary tiles in Detailed mode |
| Paths | VTracer | Fine-detail option preserves small components and limits tracing length to 1 pixel |

The managed VLM mode finds existing, complete GGUF snapshots in the Hugging Face cache. It does not require the old server on port 8080. It starts a loopback-only child process, serializes requests/model switches, and releases that process after three idle minutes or app exit. No weights were downloaded or moved. Ollama and externally managed llama.cpp remain selectable in Preferences.

Gemma's Ollama transport now disables thinking for the short object-list requests. The prior E4B configuration could consume its output budget without returning a final answer. The llama.cpp transport rejects truncated output instead of treating an incomplete list as valid.

## Working with images

1. Enter a request such as **All paper bags, exclude the car** or **Only the leftmost red bag**, then scan. A blank request finds the foreground objects. Structured requests preserve descriptive attributes and support all, leftmost, rightmost, largest and smallest selections. These selectors are applied to detected instances.
2. Choose **Fast**, **Balanced**, or **Detailed**. Use **Find visible components** when individual parts are useful.
3. Process, then inspect **Masks**, **Alpha**, and **Vectors**. Alpha shows the exported transparency over a checkerboard. Selecting a layer isolates it in Alpha view. Preview opacity is a viewing aid.
4. Use manual boxes for missed objects. **Re-segment** operates on the selected layer's box. **Apply name and edge edits** updates the selected mask, paths and alpha exports. Deleting a parent also removes its parts from the active result and export list.

| Quality | Recognition | Components | Alpha |
|---|---|---|---|
| Fast | Shorter recognition profile | Limited parent inspection when enabled | Binary alpha, preserving existing source transparency |
| Balanced | Adaptive primary recognition | SAM 3 proposals; stricter acceptance for suggested parts | VitMatte on the object region, bounded working resolution |
| Detailed | Deeper recognition including the fallback model | Additional detector fallback for suggested parts | Native-resolution boundary tiles for large objects |

In Batch mode, Draft/Balanced/Maximum map to Fast/Balanced/Detailed output quality; recognition depth is selected separately. The processing step now receives the batch's selected recognition settings. Apple GPU batch processing uses one resident inference worker instead of loading competing model replicas for every CPU core.

## Correctness changes

- Repeated objects and repeated part names have independent IDs, masks, paths and TIFF files.
- The recognizer names separate, touching objects instead of encouraging grouped descriptions. This recovered the separate computer case in the Mac sample.
- Components are recognized and detected inside the parent crop, remapped to the full image and required to overlap the parent silhouette by at least 90%. Suggested SAM 3 components require a score of at least 0.65. Scores are model outputs, not calibrated probabilities of correctness.
- Parent deduplication requires very high mask agreement. A cup inside a tray is no longer discarded just because it is contained in another object.
- A missing confirmed object produces a review message. It no longer silently becomes a full-frame mask.
- Mask cleanup preserves holes and thin structures. It no longer applies blanket closing and contour-area deletion to every mask.
- Alpha refinement enforces known foreground/background after inference and resizing. Existing source transparency is retained. Body/part alpha is computed to recompose without transparent seams.
- EXIF orientation is applied consistently before detection and display. Applicable RGB source profiles are retained in TIFF output.
- SVG groups have unique IDs and readable titles. Fine-detail tracing preserves holes and small features; the synthetic raster/vector round-trip test exceeds 98% mask IoU.
- PNG export is functional. TIFF/PNG export uses the current result's file list rather than copying every TIFF in a shared folder.
- Scan and processing cannot consume each other's background messages. Completed results for an earlier image are not attached to a newly opened image.
- Model and pipeline preference panels scroll, and show the actual local primary/fallback choices.

## Verification

Runtime: Apple M1 Ultra, 128 GiB, existing Python 3.12 Conda installation. All inference was local and offline except loopback communication with the app-owned model server.

- **544 non-GUI tests passed.**
- **231 GUI tests passed**, including the actual alpha-preview path. The affected Preferences and Batch GUI tests were rerun after their final changes.
- Ruff and `git diff --check` passed.
- Real Qwen3-VL, Gemma4 12B, MLX SAM 3, GroundingDINO, SAM 2.1, VitMatte and VTracer inference/export succeeded.

Final Balanced sample run:

| Sample | Request / settings | Result | Wall time |
|---|---|---|---|
| Grocery bags, 800 × 534 | All paper bags; exclude car; components off | Four separate bag masks and RGBA TIFFs | 7.3 s |
| Macintosh, 500 × 492 | Automatic objects; components on | Monitor, keyboard, mouse, computer case; screen and five physical controls | 16.8 s |

The earlier pipeline took approximately 32 seconds on the Mac image, missed the computer case and assigned neighboring objects as components. During development, runs with the additional detector fallback for tentative parts took 22–36 seconds. Restricting that extra component pass to Detailed mode retained the final sample's accepted parts and reduced the final Balanced run to 16.8 seconds.

These are individual local runs, not averaged benchmarks or a ground-truth accuracy evaluation. Startup, residency, other machine activity, image size and selected components affect timing. A larger labeled set of the user's artwork is still needed to tune thresholds and quantify recall/boundary accuracy. The samples also show that monitor-stand ownership and fine cables need visual review: preserving detail cannot recover pixels a model never selected.

Alpha export remains 8-bit. This update does not claim 16-bit matting, perfect semantic separation, or color-managed screen previews. The shared Conda environment still has the dependency metadata conflicts documented in the original health audit; the selected runtime passed these tests and inference checks.

Local verification artifacts:

- `/tmp/skiagrafia-quality-20260907/quality-preview.png`
- `/tmp/skiagrafia-quality-20260907/results.json`
- `/tmp/skiagrafia-quality-20260907/balanced-smoke.log`
- `/tmp/skiagrafia-quality-20260907/gemma-route.json`
- Per-layer SVG/TIFF outputs in that directory's `bags` and `mac` folders.

The verified selections were saved to `/Users/tsevis/.config/skiagrafia/preferences.json` and the local setup check passed. Previous preferences are backed up at `/Users/tsevis/.config/skiagrafia/preferences.before-quality-20260907.json`.

A persistent preview is saved at `/Users/tsevis/.codex/visualizations/2026/09/06/01a07873-2284-7e63-b969-d68a702b09cc/skiagrafia-quality-preview.png`.

Restart Skiagrafia to load the changed Python modules and saved model preferences.
