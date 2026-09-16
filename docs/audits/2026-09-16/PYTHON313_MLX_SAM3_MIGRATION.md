# Python 3.13 and MLX SAM 3 migration — 2026-09-16

## Outcome

Skiagrafia now has a locked Python `>=3.13,<3.14` runtime and makes the
locally installed MLX SAM 3 backend available through the normal `auto`
segmentation setting.  The previous Python 3.12 `.venv` was retained while a
parallel Python 3.13 environment was built and verified.

## Runtime contract

- `uv.lock` resolves for Python 3.13.14 on macOS ARM64.
- Runtime dependencies now explicitly include `mlx==0.31.2` and `ftfy`.
- MLX is pinned to `0.31.2`: the bundled SAM 3 `grid_sample_mlx.py` uses the
  MLX 0.31 custom-Metal-kernel calling contract.  MLX 0.32.2 loads the model
  but fails at its first text prompt, so it is not an acceptable replacement
  until that source is independently adapted and tested.
- `auto` chooses MLX SAM 3 only when
  `mlx_sam3/sam3-mod-weights/model.safetensors` is present.  Otherwise the
  existing GroundingDINO + SAM 2.1 path remains the active fallback.
- The former shipped `sam2` default is migrated to that checkpoint-aware
  `auto` mode; an existing explicit `mlx-sam3` selection is preserved.
- The former `sam3_confidence` default of `0.5` is migrated to `0.2` only when
  it is the exact old shipped default.  This admits valid thin typographic
  instances while preserving any different user-selected value.

## Real local verification

Input: `/Users/tsevis/01CLIENTI/01 EXPERIMENTS/TYPO/MATERIALE/PETE.jpg`

- Python 3.13.14 / MLX 0.31.2 loaded the local SAM 3 checkpoint successfully.
- With `letter` at confidence `0.2`, SAM 3 returned four native MLX masks:
  `P`, `E`, `T`, `E`; no GroundingDINO fallback was used.
- The local Qwen3-VL semantic observation independently read `P,E,T,E`.
- The full pipeline exported four `letter P/E/T/E` layers, a parseable SVG,
  and an RGBA `1159 × 602` all-objects TIFF whose alpha was pixel-exactly the
  union of its layer alphas.

This is evidence for this model, image and runtime combination; it is not a
claim that semantic recognition is universally accurate.

## Residual limitations

- MLX SAM 3 is an optional local model bundle. If its checkpoint is absent or
  its runtime fails, Skiagrafia reports the condition and uses the established
  SAM 2.1 fallback rather than fabricating a successful MLX result.
- Ambiguous images remain subject to semantic model uncertainty and Triage.
- `sqlitedict 2.1.0` still has unresolved advisory `PYSEC-2026-1939`, with no
  fixed version available.
