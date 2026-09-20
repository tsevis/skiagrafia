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
    gui-full) exec "$PYTHON" -m pytest -m "gui or gui_integration" "$@" ;;
    all) "$PYTHON" -m pytest "$@"; "$PYTHON" -m pytest -m "gui or gui_integration" "$@"; "$PYTHON" -m ruff check .; "$PYTHON" -m pyright; "$PYTHON" -m compileall -q .; "$PYTHON" -m pip check; "$PYTHON" -m pip_audit ;;
    # Everything in `all` except the windowed tests, which open real windows
    # on whoever's desktop is running this. `all` is for a release; this is
    # the one to run while working.
    check) "$PYTHON" -m pytest "$@"; "$PYTHON" -m ruff check .; "$PYTHON" -m pyright; "$PYTHON" -m compileall -q .; "$PYTHON" -m pip check; "$PYTHON" -m pip_audit ;;
    types) exec "$PYTHON" -m pyright "$@" ;;
    *) echo "usage: $0 [default|gui|gui-full|check|types|all] [pytest arguments...]" >&2; exit 2 ;;
esac
