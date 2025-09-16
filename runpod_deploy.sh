#!/bin/bash
# runpod_deploy.sh - Automated deployment to RunPod

set -e

# Configuration
RUNPOD_HOST="213.192.2.99"
RUNPOD_PORT="41882"
RUNPOD_USER="root"
SSH_KEY="$HOME/.ssh/runpod_g70"
REMOTE_DIR="/workspace/mordzix"
LOCAL_DIR="."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check SSH key
if [ ! -f "$SSH_KEY" ]; then
    error "SSH key not found: $SSH_KEY"
fi

log "🚀 Starting Mordzix deployment to RunPod..."

# Test connection
log "Testing SSH connection..."
if ! ssh -i "$SSH_KEY" -p "$RUNPOD_PORT" -o ConnectTimeout=10 "$RUNPOD_USER@$RUNPOD_HOST" "echo 'SSH OK'" >/dev/null 2>&1; then
    error "Cannot connect to RunPod instance"
fi

log "✅ SSH connection successful"

# Create remote directory
log "Setting up remote directory..."
ssh -i "$SSH_KEY" -p "$RUNPOD_PORT" "$RUNPOD_USER@$RUNPOD_HOST" "mkdir -p $REMOTE_DIR/data $REMOTE_DIR/logs $REMOTE_DIR/static"

# Sync files (exclude unnecessary files)
log "Syncing files to RunPod..."
rsync -avz \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude 'node_modules' \
    --exclude '*.log' \
    --exclude 'data/*.db' \
    --exclude 'sqlitebrowser' \
    -e "ssh -i $SSH_KEY -p $RUNPOD_PORT" \
    "$LOCAL_DIR/" "$RUNPOD_USER@$RUNPOD_HOST:$REMOTE_DIR/"

# Install dependencies and start
log "Installing dependencies on RunPod..."
ssh -i "$SSH_KEY" -p "$RUNPOD_PORT" "$RUNPOD_USER@$RUNPOD_HOST" << 'EOF'
cd /workspace/mordzix

# Install Python deps
pip install -r requirements.txt

# Kill existing processes
pkill -f "uvicorn" || true
pkill -f "mordzix" || true

# Wait a bit
sleep 2

# Start Mordzix server
nohup python -m uvicorn mordzix_server:app --host 0.0.0.0 --port 8080 --workers 1 > logs/mordzix.log 2>&1 &

# Wait for startup
sleep 5

# Health check
curl -f http://localhost:8080/health || echo "Health check failed"

echo "🎉 Mordzix deployed and running!"
EOF

log "🌟 Deployment completed!"
log ""
log "Access your Mordzix instance at:"
log "  http://$RUNPOD_HOST:8080 (if port forwarded)"
log "  or through RunPod's web interface"
log ""
log "To check status:"
log "  ssh -i $SSH_KEY -p $RUNPOD_PORT $RUNPOD_USER@$RUNPOD_HOST"
log "  curl http://localhost:8080/health"
log ""
log "To view logs:"
log "  tail -f $REMOTE_DIR/logs/mordzix.log"