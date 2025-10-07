#!/usr/bin/env bash
set -euo pipefail

PORT="${BACKEND_PORT:-5959}"

# aktywuj venv
source .venv/bin/activate

# ustaw ścieżkę projektu
export PYTHONPATH=/workspace/mrd69

PERSIST_DIR="${RUNPOD_PERSIST_DIR:-}"
if [ -n "$PERSIST_DIR" ]; then
  mkdir -p "$PERSIST_DIR/data" 2>/dev/null || true
fi

# odpal serwer FastAPI z server.py
uvicorn server:app --host 0.0.0.0 --port $PORT
