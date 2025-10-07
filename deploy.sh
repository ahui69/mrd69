#!/bin/bash
# ═══════════════════════════════════════════════════════════
# AUTOMATYCZNY DEPLOY NA RUNPOD
# Uruchomienie: bash deploy.sh
# ═══════════════════════════════════════════════════════════

set -e

# Konfiguracja
RUNPOD_HOST="213.192.2.99"
RUNPOD_PORT="41724"
RUNPOD_USER="root"
SSH_KEY="$HOME/.ssh/runpod_ed25519"  # Zmień jeśli klucz jest gdzie indziej
REMOTE_DIR="/workspace/mrd69"

echo "════════════════════════════════════════════════════════"
echo "  🚀 DEPLOY NA RUNPOD"
echo "════════════════════════════════════════════════════════"
echo ""

# ─────────────────────────────────────
# Sprawdź czy klucz SSH istnieje
# ─────────────────────────────────────
if [ ! -f "$SSH_KEY" ]; then
    echo "❌ Nie znaleziono klucza SSH: $SSH_KEY"
    echo "Sprawdź ścieżkę do klucza i edytuj skrypt (linia 13)"
    exit 1
fi

# ─────────────────────────────────────
# Test połączenia
# ─────────────────────────────────────
echo "📡 Testuję połączenie..."
ssh -p $RUNPOD_PORT -i "$SSH_KEY" -o IdentitiesOnly=yes -o ConnectTimeout=10 \
    $RUNPOD_USER@$RUNPOD_HOST "echo '✅ SSH działa!'" || {
    echo "❌ Nie mogę połączyć się z RunPod!"
    exit 1
}

# ─────────────────────────────────────
# Przygotuj katalog
# ─────────────────────────────────────
echo "📁 Przygotowuję katalog na podzie..."
ssh -p $RUNPOD_PORT -i "$SSH_KEY" -o IdentitiesOnly=yes \
    $RUNPOD_USER@$RUNPOD_HOST "mkdir -p $REMOTE_DIR"

# ─────────────────────────────────────
# Kopiuj kod przez rsync (szybsze niż scp)
# ─────────────────────────────────────
echo "📦 Kopiuję kod (to może chwilę potrwać)..."

rsync -avz --progress \
    -e "ssh -p $RUNPOD_PORT -i $SSH_KEY -o IdentitiesOnly=yes" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='data/' \
    --exclude='logs/' \
    --exclude='*.log' \
    --exclude='venv/' \
    --exclude='node_modules/' \
    --exclude='.pytest_cache/' \
    --exclude='htmlcov/' \
    --exclude='docs/' \
    --exclude='history/' \
    --exclude='reports/' \
    . $RUNPOD_USER@$RUNPOD_HOST:$REMOTE_DIR/

echo "✅ Kod skopiowany!"

# ─────────────────────────────────────
# Zainstaluj i uruchom
# ─────────────────────────────────────
echo "🔧 Instaluję zależności i uruchamiam..."

ssh -p $RUNPOD_PORT -i "$SSH_KEY" -o IdentitiesOnly=yes \
    $RUNPOD_USER@$RUNPOD_HOST << 'ENDSSH'

cd /workspace/mrd69

echo "📦 Instaluję requirements..."
pip install --no-cache-dir -r requirements.txt

echo "🛑 Zatrzymuję stary serwer..."
pkill -f "uvicorn server:app" 2>/dev/null || true
sleep 2

echo "🚀 Uruchamiam serwer w tle..."
nohup python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &

echo "⏳ Czekam 5 sekund..."
sleep 5

echo "🧪 Test serwera..."
curl -s http://localhost:8000/api/health || echo "❌ Serwer nie odpowiada jeszcze"

ENDSSH

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ✅ DEPLOY ZAKOŃCZONY!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "🔗 TWÓJ SERWER DZIAŁA NA RUNPOD!"
echo ""
echo "📋 Dalsze kroki:"
echo "   1. Idź na: https://www.runpod.io/console/pods"
echo "   2. Kliknij swój Pod"
echo "   3. Skopiuj 'Connect' → HTTP URL"
echo "   4. Test: curl https://TWOJ_URL/api/health"
echo ""
echo "📊 Zobacz logi:"
echo "   ssh -p $RUNPOD_PORT -i \"$SSH_KEY\" $RUNPOD_USER@$RUNPOD_HOST"
echo "   cd $REMOTE_DIR && tail -f server.log"
echo ""
echo "════════════════════════════════════════════════════════"
