# Skiagrafia hardening, repair and verification audit — 2026-09-14

**Baseline commit:** `427a04d`  
**Verdict:** **PASS WITH LIMITATIONS**

This is an evidence-based verification result, not a claim that the
application is "100% secure".  The isolated supported runtime and all
functional/static checks pass.  The dependency audit still reports one
unfixed upstream advisory; the operational and supply-chain limitations below
remain relevant.

## Scope and baseline

The README, manual, and the prior audit (`docs/audits/2026-09-06/README.md`)
were reviewed before implementation.  The working tree was clean at
`427a04d`.  At baseline, `run.sh` selected the shared Miniconda interpreter
instead of a project environment.  The initial test baseline was 552 default
tests and 236 GUI tests passing; `ruff` reported one unused import; the shared
environment's `pip check` had unrelated conflicts and was not treated as the
application runtime.

## Implemented repairs

### Reproducible runtime

- Added `uv.lock`, a Python `>=3.12,<3.13` project specification, and a locked
  development dependency group.
- `run.sh` and `scripts/verify.sh` now both require `.venv/bin/python`; neither
  silently falls back to a global interpreter.  An explicit
  `SKIAGRAFIA_PYTHON` remains available for an intentionally managed override.
- Added the current patched `transformers` 5 line plus a small
  GroundingDINO/BERT compatibility shim, verified with a focused regression
  test.
- The bundled MLX SAM 3 source requires Python 3.13+, so the supported Python
  3.12 environment selects SAM 2.1 by default.  An explicit MLX selection on
  Python 3.12 now fails with an actionable error instead of partially loading
  an incompatible backend.

### I/O, model service and export hardening

- Added local-only VLM endpoint validation: only loopback HTTP(S) roots are
  accepted; remote hosts, credentials, paths, queries and invalid ports are
  rejected.
- Model registry downloads require approved HTTPS source hosts; archive
  extraction is staged and rejects traversal, absolute paths, symlinks,
  unexpected roots, excessive members and excessive expanded size.
- Output and template writes use safe flat child paths, refuse symlink targets,
  and use same-directory atomic replacement.
- SVG output is structurally validated and rejects DTD/entity constructs,
  active/external/resource-bearing elements, event attributes and unsafe URI
  schemes.  TIFF/PNG dimensions, alpha shape, output format/mode and
  readability are verified before success is reported.
- VTracer uses a private temporary directory and its SVG is parsed and reduced
  to the allowed path subset before assembly.  SVG layer identifiers are
  unique and labels are escaped.
- Empty/malformed VLM responses, invalid images, missing/failed masks, alpha
  failure and export failure now result in explicit error states rather than
  successful-looking empty output.

### Pipeline and batch correctness

- Output names contain a path-derived hash to prevent same-label/name
  collisions.
- Batch IDs are constrained to safe filename tokens; batch snapshots are
  written atomically.
- Repeated instances retain stable distinct IDs.  Child interrogation uses the
  parent crop and containment checks reject children outside their parent.
- The existing single/batch configuration and persistence coverage was kept;
  tests cover selection request, domain guide, triage labels, templates and
  batch resume.  Saved user preferences are not silently migrated or deleted.

### Exception boundaries

Core model, bootstrap, VLM, export, archive, state and template paths now
catch their expected operational exception classes and preserve a concrete
failure state.  The remaining broad catches are deliberately confined to
asynchronous UI callback/worker boundaries and `BatchRunner` future completion:
they log and surface the failed item to the user instead of returning a
successful empty result.  They are retained because third-party ML/Tk callback
errors cross those boundaries and must not terminate the UI event loop.

## Verification evidence

All commands below used the isolated project runtime:

| Check | Result |
| --- | --- |
| Python/runtime | Python 3.12.11, `sys.prefix=.venv` |
| Required import/dependency inventory | present: vtracer 0.6.15, sqlitedict 2.1.0, Pillow 12.3.0, NumPy 2.5.3, SciPy 1.18.1, Torch 2.14.0, Torchvision 0.29.0, Pydantic 2.13.5, OpenCV headless 5.0.0.93, CairoSVG 2.9.1 |
| `pytest -q` | **568 passed, 236 deselected** |
| `pytest -m gui -q` | **236 passed, 568 deselected** |
| Focused hardening/regression selection | **173 passed, 1 deselected** |
| `ruff check .` | **All checks passed** |
| `python -m compileall -q .` | **passed** |
| `python -m pip check` | **No broken requirements found** |
| `uv lock --check` and shell syntax checks | **passed** |
| `git diff --check` | **passed** |
| unsafe-code scan | no `shell=True`, `os.system`, Python `eval`/`exec`, pickle, unsafe YAML or `extractall`; `mx.eval` is the MLX tensor-evaluation API, not Python `eval` |

New deterministic tests exercise loopback URL rejection, malformed VLM JSON
and empty responses, traversal and symlinked outputs, archive traversal and
symlink members, active SVG markup, invalid TIFF alpha, unsafe batch IDs,
explicit pipeline failures, and the Python-version MLX policy.

## End-to-end run

The real local pipeline ran on `docs/Skiagrafia.png` with the installed local
VLM, GroundingDINO/SAM 2.1, VitMatte and VTracer.  It produced one parseable
SVG and 16 decoded RGBA TIFFs.  Every TIFF decoded at `3549 × 2327`; XML
parsing confirmed unique SVG IDs.  The result included several repeated
instances (`left-hand`) and parent/child layers, including a `laptop` and its
child parts.  No cloud model download was used during this run.

The disposable output directories created solely for this verification live
outside the repository in `/tmp`.  They are left intact to preserve the run
evidence; no user data or project outputs were deleted.

## Security audit result and residual risks

`pip-audit` reports `PYSEC-2026-1939` for `sqlitedict 2.1.0` twice, with **no
fixed version available**.  Skiagrafia itself is a local project and is not on
PyPI, so `pip-audit` also marks it as unauditable by that registry-based tool.
This prevents a fully clean dependency-security result.

Other residual risks and limitations:

1. Model weights and the Grounded-SAM source checkout are external artifacts.
   Host validation and safe extraction reduce local file risks, but the
   registry's moving upstream archive/download URLs are not content-hash
   pinned.  A release process should add vetted SHA-256 values and immutable
   revisions before a high-assurance deployment.
2. A loopback-only VLM service is not mutually authenticated.  It protects
   against remote endpoints but not another local process running as the same
   user.
3. Semantic detection and alpha quality are model outputs, not security
   guarantees.  The end-to-end image verifies behavior and artifact integrity,
   not accuracy across every domain image.
4. The installed GroundingDINO environment reported unavailable custom C++
   operators and ran in CPU mode.  This is a performance limitation, not an
   artifact-integrity failure.
5. MLX SAM 3 is intentionally outside this locked Python 3.12 runtime.  Use a
   separately verified Python 3.13 environment if that optional backend is
   required.
6. UI worker boundaries still catch unexpected third-party callback failures
   so the GUI remains alive.  They log the traceback and mark the item failed;
   monitoring those logs remains appropriate in production use.

## Reproduction

```bash
uv sync --locked --group dev
./scripts/verify.sh default -q
./scripts/verify.sh gui -q
.venv/bin/python -m ruff check .
.venv/bin/python -m compileall -q .
.venv/bin/python -m pip check
.venv/bin/python -m pip_audit  # currently reports the known sqlitedict advisory
./run.sh
```

`./scripts/verify.sh all` runs the combined matrix and exits non-zero until the
upstream `sqlitedict` advisory has a fix or an explicitly reviewed replacement
is adopted.
