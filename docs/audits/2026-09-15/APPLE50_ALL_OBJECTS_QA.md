# Apple50 all-objects visual and semantic QA — 2026-09-15

## Scope and method

This was a read-only review of the 25 Apple50 source images and the two
existing Batch output folders in
`/Users/tsevis/01CLIENTI/01 EXPERIMENTS/APPLE50 2/for test/`
(`batch_without_guide` and `batch_with_guide`).  No input image or existing
verification output was modified.

The review used side-by-side source/foreground checkerboard contact sheets,
then inspected the ambiguous full-resolution cases.  A fresh Pillow/NumPy
check also recomputed each composite alpha from every matching layer TIFF.

## Artifact result

| Case | All-objects TIFFs | RGBA/source dimensions | Pixel-exact layer-alpha union |
| --- | ---: | ---: | ---: |
| `batch_without_guide` | 25 / 25 | 25 / 25 | 25 / 25 |
| `batch_with_guide` | 25 / 25 | 25 / 25 | 25 / 25 |

The deliberately transparent files were not counted as failures.  In the
unguided case they are pages 204, 208, 216, 221, 223, 227 and 244; in the
guided case they are pages 204, 208, 216, 221, 227, 244 and 250.  They are the
inputs for which no accepted semantic candidate remained, rather than a
fabricated foreground or an export error.

## Semantic findings

- Page 212 is a positive control: the guided composite keeps both visible
  Apple computers while excluding the people and hands holding them.
- Page 242 is an intended operating-system interface extraction.  Its text is
  part of the selected interface image, not an independently selected body
  text layer.
- Page 250 demonstrates a genuine unguided selection false positive: the
  `OPENMAC` vehicle licence plate was proposed as `california license plate`
  and isolated despite the request being limited to Apple hardware,
  peripherals, packaging and OS interfaces.  The guided run correctly leaves
  this input transparent.
- Page 223 is an ambiguity requiring curator control.  The unguided run
  correctly leaves the debris scene transparent, while the guide/fallback VLM
  proposes broad `Apple computer` and `Apple peripheral` labels and produces
  a 20.5% foreground composite of debris.  There is not enough visible
  identity evidence to automatically promote the scene to accepted Apple
  hardware.

The last two findings should not be solved by fabricating alpha or by a
hard-coded image-domain blacklist.  The reliable product control is human
Triage.  Batch Triage now supports a per-input skip for an otherwise globally
approved label, persists that decision in `triage.json`, and applies it before
the pipeline runs.  This lets a curator skip page 223 without suppressing
`Apple computer` on the legitimate Apple-product pages.

## Residual limitation

The local VLM is still a proposal generator, not a semantic guarantee.
Selection Request and Domain Guide prompts reduce false positives but cannot
replace image-level curator review for ambiguous editorial material.
