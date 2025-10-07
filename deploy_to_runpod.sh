#!/bin/bash
# ═══════════════════════════════════════════════════════════
# AUTOMATYCZNY DEPLOY NA RUNPOD
# Uruchom: bash deploy_to_runpod.sh
# ═══════════════════════════════════════════════════════════

set -e  # Zatrzymaj przy błędzie

RUNPOD_HOST="213.192.2.99"
RUNPOD_PORT="41724"
RUNPOD_USER="root"
SSH_KEY="C:/Users/48501/.ssh/runpod_ed25519"
REMOTE_DIR="/workspace/mrd69"

echo "════════════════════════════════════════════════════════"
echo "  🚀 DEPLOY NA RUNPOD"
echo "════════════════════════════════════════════════════════"
echo ""

# ─────────────────────────────────────
# KROK 1: Sprawdź połączenie SSH
# ─────────────────────────────────────
echo "📡 Sprawdzam połączenie z RunPod..."
ssh -p $RUNPOD_PORT -i "$SSH_KEY" -o IdentitiesOnly=yes -o ConnectTimeout=10 \
    $RUNPOD_USER@$RUNPOD_HOST "echo '✅ Połączenie działa!'" || {
    echo "❌ Nie mogę połączyć się z RunPod!"
    echo "Sprawdź czy Pod działa w RunPod Console"
    exit 1
}

# ─────────────────────────────────────
# KROK 2: Przygotuj katalog na podzie
# ─────────────────────────────────────
echo "📁 Tworzę katalog na podzie..."
ssh -p $RUNPOD_PORT -i "$SSH_KEY" -o IdentitiesOnly=yes \
    $RUNPOD_USER@$RUNPOD_HOST "mkdir -p $REMOTE_DIR && rm -rf $REMOTE_DIR/*"

# ─────────────────────────────────────
# KROK 3: Kopiuj pliki przez SCP
# ─────────────────────────────────────
echo "📦 Kopiuję kod na RunPod..."

# Lista plików/folderów do skopiowania
FILES_TO_COPY=(
    "server.py"
    "requirements.txt"
    ".env"
    "routers"
    "src"
    "frontend"
    "webapp"
    "run.sh"
    "stop.sh"
)

for item in "${FILES_TO_COPY[@]}"; do
    if [ -e "$item" ]; then
        echo "  → Kopiuję $item..."
        scp -P $RUNPOD_PORT -i "$SSH_KEY" -o IdentitiesOnly=yes -r \
            "$item" $RUNPOD_USER@$RUNPOD_HOST:$REMOTE_DIR/
    fi
done

echo "✅ Pliki skopiowane!"

# ─────────────────────────────────────
# KROK 4: Zainstaluj zależności
# ─────────────────────────────────────
echo "📦 Instaluję zależności na podzie..."
ssh -p $RUNPOD_PORT -i "$SSH_KEY" -o IdentitiesOnly=yes \
    $RUNPOD_USER@$RUNPOD_HOST << 'ENDSSH'
cd /workspace/mrd69
pip install --no-cache-dir -r requirements.txt
echo "✅ Zależności zainstalowane!"
ENDSSH

# ─────────────────────────────────────
# KROK 5: Zatrzymaj stary serwer (jeśli był)
# ─────────────────────────────────────
echo "🛑 Zatrzymuję stary serwer..."
ssh -p $RUNPOD_PORT -i "$SSH_KEY" -o IdentitiesOnly=yes \
    $RUNPOD_USER@$RUNPOD_HOST "pkill -f 'uvicorn server:app' || true"

sleep 2

# ─────────────────────────────────────
# KROK 6: Uruchom serwer w tle
# ─────────────────────────────────────
echo "🚀 Uruchamiam serwer..."
ssh -p $RUNPOD_PORT -i "$SSH_KEY" -o IdentitiesOnly=yes \
    $RUNPOD_USER@$RUNPOD_HOST << 'ENDSSH'
cd /workspace/mrd69
nohup python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
sleep 3
curl -s http://localhost:8000/api/health
ENDSSH

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ✅ DEPLOY ZAKOŃCZONY!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "📡 Twój serwer działa na RunPod!"
echo ""
echo "🔗 Sprawdź URL w RunPod Console:"
echo "   https://www.runpod.io/console/pods"
echo "   Kliknij swój Pod → skopiuj URL (będzie coś jak xyz-8000.proxy.runpod.net)"
echo ""
echo "🧪 Test:"
echo "   curl https://TWOJ_URL/api/health"
echo ""
echo "📋 Zobacz logi:"
echo "   ssh -p $RUNPOD_PORT -i \"$SSH_KEY\" root@$RUNPOD_HOST"
echo "   cd /workspace/mrd69 && tail -f server.log"
echo ""
echo "════════════════════════════════════════════════════════"
