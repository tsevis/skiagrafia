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
# first on PATH, which can be a system Python that has none of the app's
# dependencies -- it then dies at its first import, before logging exists.
# Set SKIAGRAFIA_PYTHON to force a particular interpreter or virtualenv.
if [ -n "${SKIAGRAFIA_PYTHON:-}" ]; then
    # An explicit choice is used as given, not second-guessed.
    PYTHON="$SKIAGRAFIA_PYTHON"
else
    PYTHON=""
    for candidate in \
        "$SCRIPT_DIR/.venv/bin/python3" \
        "$HOME/miniconda3/bin/python3" \
        "$(command -v python3 2>/dev/null || true)"
    do
        [ -n "$candidate" ] && [ -x "$candidate" ] || continue
        # find_spec reports availability without paying torch's import cost.
        if "$candidate" -c 'import importlib.util as u, sys; sys.exit(0 if u.find_spec("torch") and u.find_spec("rich") else 1)' 2>/dev/null; then
            PYTHON="$candidate"
            break
        fi
    done
fi

if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
    echo "skiagrafia: no Python with the required dependencies was found." >&2
    echo "  tried: ./.venv/bin/python3, ~/miniconda3/bin/python3, python3 on PATH" >&2
    echo "  fix:   export SKIAGRAFIA_PYTHON=/path/to/python3" >&2
    exit 1
fi

exec "$PYTHON" main.py "$@"
