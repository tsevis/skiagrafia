#!/usr/bin/env bash
# Run Skiagrafia checks with the exact same project interpreter as run.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "skiagrafia: project runtime is missing; run: uv sync --locked --group dev" >&2
    exit 1
fi

cd "$SCRIPT_DIR"
mode="${1:-default}"
if [ "$#" -gt 0 ]; then
    shift
fi
case "$mode" in
    default) exec "$PYTHON" -m pytest "$@" ;;
    gui) exec "$PYTHON" -m pytest -m gui "$@" ;;
    all) "$PYTHON" -m pytest "$@"; "$PYTHON" -m pytest -m gui "$@"; "$PYTHON" -m ruff check .; "$PYTHON" -m compileall -q .; "$PYTHON" -m pip check; "$PYTHON" -m pip_audit ;;
    *) echo "usage: $0 [default|gui|all] [pytest arguments...]" >&2; exit 2 ;;
esac
