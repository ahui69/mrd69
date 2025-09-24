#!/usr/bin/env bash
set -euo pipefail
PORT="${BACKEND_PORT:-5959}"
curl -sf "http://127.0.0.1:${PORT}/health" || true
echo
echo "Docs: http://127.0.0.1:${PORT}/docs"
