#!/usr/bin/env bash
set -euo pipefail

cd /workspace/mrd69

echo "==> Zabijam stare uvicorny…"
pkill -f "uvicorn server:app" 2>/dev/null || true
pkill -f "uvicorn mrd69.server:app" 2>/dev/null || true

echo "==> Tworzę katalogi danych…"
mkdir -p /workspace/mrd69/data /workspace/mrd69/data/sq3 /workspace/mrd69/uploads

echo "==> Startuję serwer na porcie 8000…"
nohup uvicorn server:app --host 0.0.0.0 --port 8000 --log-level info > /workspace/mrd69/server.log 2>&1 &
echo $! > /workspace/mrd69/start.pid

sleep 1
echo "==> PID: $(cat /workspace/mrd69/start.pid)"
echo "==> Log: tail -f /workspace/mrd69/server.log"

echo "==> Szybkie sanity:"
curl -s http://127.0.0.1:8000/api/health || true
echo
curl -s http://127.0.0.1:8000/api/memory/health || true
echo
