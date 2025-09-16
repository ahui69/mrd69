#!/bin/bash
set -e

echo "=== [0] Python 3.11 jako domyślny ==="
python3 --version

# Sprawdź czy jesteśmy na RunPod
if [ -n "${RUNPOD_PERSIST_DIR}" ] && [ -d "${RUNPOD_PERSIST_DIR}" ]; then
    echo "=== RunPod Environment Detected ==="
    echo "RUNPOD_PERSIST_DIR: ${RUNPOD_PERSIST_DIR}"
    mkdir -p "${RUNPOD_PERSIST_DIR}/data"
    export USE_RUNPOD=True
    echo "USE_RUNPOD ustawione na: ${USE_RUNPOD}"
else
    echo "=== Local Environment ==="
    export USE_RUNPOD=False
fi

echo "=== [1] Instalacja braków (pip + node) ==="
pip install --upgrade pip setuptools wheel
pip install black isort
pip install -r /workspace/a/requirements.txt || true
cd /workspace/a/frontend && npm install && cd ..

echo "=== [2] Auto-fix składni w .py ==="
find /workspace/a -name "*.py" -exec sed -i 's/„/"/g; s/”/"/g; s/–/-/g; s/…/.../g' {} \;

echo "=== [3] Formatowanie kodu ==="
isort /workspace/a || true
black /workspace/a || true

echo "=== [4] Build frontu ==="
cd /workspace/a/frontend
npm run build || true
cd /workspace/a

# Upewnij się, że katalogi istnieją
echo "=== [5] Inicjalizacja katalogów danych ==="
mkdir -p data/mem
mkdir -p data/conversations
mkdir -p data/inbox

# Stosuj zmienne środowiskowe z env_patch.sh jeśli istnieje
if [ -f "env_patch.sh" ]; then
    echo "=== [6] Stosowanie zmiennych środowiskowych z env_patch.sh ==="
    source env_patch.sh
fi

echo "=== [7] Restart backend na :5959 ==="
pkill -f "uvicorn main:app" || true
nohup uvicorn main:app --host 0.0.0.0 --port 5959 --reload > backend.log 2>&1 &

sleep 2
echo "Health:" 
curl -s http://127.0.0.1:5959/health || echo "Backend nie odpowiada"

echo "=== DONE ==="
