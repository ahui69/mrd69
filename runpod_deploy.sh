#!/bin/bash
# Improved RunPod deploy with logging, venv, and retries

set -euo pipefail
IFS=$'\n\t'

# Configuration (edit if needed)
RUNPOD_HOST="${RUNPOD_HOST:-213.192.2.99}"
RUNPOD_PORT="${RUNPOD_PORT:-41882}"
RUNPOD_USER="${RUNPOD_USER:-root}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/runpod_g70}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/mordzix}"
LOCAL_DIR="${LOCAL_DIR:-.}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements.txt}"
LOG_DIR="${LOG_DIR:-$REMOTE_DIR/logs}"
HEALTH_URL="http://localhost:8080/health"
HEALTH_RETRIES=6
HEALTH_SLEEP=5

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log(){ echo -e "${GREEN}[$(date '+%H:%M:%S')] $1${NC}"; }
warn(){ echo -e "${YELLOW}[$(date '+%H:%M:%S')] $1${NC}"; }
error(){ echo -e "${RED}[$(date '+%H:%M:%S')] ERROR: $1${NC}"; exit 1; }

if [ ! -f "$SSH_KEY" ]; then
  error "SSH key not found: $SSH_KEY"
fi

log "Starting deploy -> $RUNPOD_USER@$RUNPOD_HOST:$REMOTE_DIR"

# Test SSH
if ! ssh -i "$SSH_KEY" -p "$RUNPOD_PORT" -o ConnectTimeout=10 "$RUNPOD_USER@$RUNPOD_HOST" "echo SSH_OK" >/dev/null 2>&1; then
  error "Cannot connect to RunPod instance via SSH"
fi
log "SSH OK"

# Ensure remote directories
ssh -i "$SSH_KEY" -p "$RUNPOD_PORT" "$RUNPOD_USER@$RUNPOD_HOST" "mkdir -p '$REMOTE_DIR' '$LOG_DIR' '$REMOTE_DIR/static' || true"

# Rsync files
log "Syncing files..."
rsync -avz --delete \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude 'node_modules' \
  --exclude '*.log' \
  --exclude 'data/*.db' \
  -e "ssh -i $SSH_KEY -p $RUNPOD_PORT" \
  "$LOCAL_DIR/" "$RUNPOD_USER@$RUNPOD_HOST:$REMOTE_DIR/" || error "rsync failed"

log "Files synced"

# Run remote setup and start server (capturing logs)
log "Running remote install and start"
ssh -i "$SSH_KEY" -p "$RUNPOD_PORT" "$RUNPOD_USER@$RUNPOD_HOST" bash -s <<'REMOTE_EOF'
set -euo pipefail
cd /workspace/mordzix

mkdir -p logs
# Use system python (or adjust) and venv for isolation
PY="$PYTHON_BIN"
if ! command -v "$PY" >/dev/null 2>&1; then
  PY="python3"
fi

# create venv if not exists
if [ ! -d ".venv" ]; then
  "$PY" -m venv .venv || { echo "venv creation failed"; exit 1; }
fi
source .venv/bin/activate

# Upgrade pip & install deps (log to file)
pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || true
echo "=== PIP INSTALL START ===" > logs/pip_install.log
if ! pip install -r "${REQUIREMENTS_FILE}" 2>&1 | tee -a logs/pip_install.log; then
  echo "PIP INSTALL FAILED (see logs/pip_install.log)" >&2
  exit 2
fi
echo "=== PIP INSTALL OK ===" >> logs/pip_install.log

# Kill previous processes cleanly
pkill -f "uvicorn mordzix_server:app" || true
sleep 1

# Start server and redirect logs
nohup .venv/bin/python -m uvicorn mordzix_server:app --host 0.0.0.0 --port 8080 --workers 1 > logs/mordzix.log 2>&1 &

# Wait a little for process spawn
sleep 3
REMOTE_EOF

log "Remote start command executed. Waiting for health check..."

# Healthcheck with retries
i=0
while [ $i -lt $HEALTH_RETRIES ]; do
  if ssh -i "$SSH_KEY" -p "$RUNPOD_PORT" "$RUNPOD_USER@$RUNPOD_HOST" "curl -s -f $HEALTH_URL >/dev/null 2>&1"; then
    log "Healthcheck OK"
    break
  else
    warn "Healthcheck failed (attempt $((i+1))/$HEALTH_RETRIES), waiting ${HEALTH_SLEEP}s..."
    sleep "$HEALTH_SLEEP"
  fi
  i=$((i+1))
done

if [ $i -ge $HEALTH_RETRIES ]; then
  warn "Healthcheck failed after retries. Fetching last logs..."
  ssh -i "$SSH_KEY" -p "$RUNPOD_PORT" "$RUNPOD_USER@$RUNPOD_HOST" "tail -n 200 $LOG_DIR/mordzix.log || true; echo '--- pip log ---'; tail -n 200 $LOG_DIR/pip_install.log || true"
  error "Deployment finished but service not healthy. Check logs above."
fi

log "Deployment completed. Access at http://$RUNPOD_HOST:8080 (if forwarded)"
log "To see logs: ssh -i $SSH_KEY -p $RUNPOD_PORT $RUNPOD_USER@$RUNPOD_HOST 'tail -f $LOG_DIR/mordzix.log'"