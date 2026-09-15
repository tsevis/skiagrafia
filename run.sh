#!/usr/bin/env bash
# Skiagrafia launcher
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 100% local inference — block ALL network downloads from HuggingFace/transformers
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1
# Prevent stale .pyc caches from masking source changes
export PYTHONDONTWRITEBYTECODE=1
# Prevent transformers from importing TensorFlow/Flax (not needed, causes crashes)
export USE_TF=0
export USE_FLAX=0

# Ensure cairocffi can find Homebrew's libcairo for PDF export
if [ -d "/opt/homebrew/lib" ]; then
    export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
fi

# Resolve the interpreter explicitly. A bare `python3` runs whichever comes
# first on PATH, which can be a system Python with unrelated packages.  The
# normal launcher and verification runner both use .venv; SKIAGRAFIA_PYTHON is
# retained only for an intentionally managed, explicitly selected runtime.
if [ -n "${SKIAGRAFIA_PYTHON:-}" ]; then
    PYTHON="$SKIAGRAFIA_PYTHON"
else
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
fi

if [ ! -x "$PYTHON" ]; then
    echo "skiagrafia: project runtime is missing: $PYTHON" >&2
    echo "  fix: uv sync --locked --group dev" >&2
    echo "  override deliberately with: export SKIAGRAFIA_PYTHON=/path/to/python" >&2
    exit 1
fi

exec "$PYTHON" main.py "$@"
