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
# Prevent transformers from importing TensorFlow/Flax (not needed, causes crashes)
export USE_TF=0
export USE_FLAX=0

# Ensure cairocffi can find Homebrew's libcairo for PDF export
if [ -d "/opt/homebrew/lib" ]; then
    export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
fi

python3 main.py "$@"
