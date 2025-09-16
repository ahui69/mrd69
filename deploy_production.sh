#!/bin/bash
# deploy_production.sh - Production deployment script for Mordzix

set -e

# Configuration
APP_NAME="mordzix"
APP_USER="mordzix"
APP_DIR="/opt/mordzix"
SERVICE_PORT="6969"
DOMAIN="your-domain.com"  # Change this to your domain

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    error "Please run as root (use sudo)"
fi

log "Starting Mordzix production deployment..."

# 1. System updates and dependencies
log "Installing system dependencies..."
apt update
apt install -y python3 python3-pip python3-venv sqlite3 curl nginx ufw fail2ban

# 2. Create application user
if ! id "$APP_USER" &>/dev/null; then
    log "Creating application user: $APP_USER"
    useradd -r -s /bin/bash -d "$APP_DIR" "$APP_USER"
else
    log "User $APP_USER already exists"
fi

# 3. Create application directory
log "Setting up application directory: $APP_DIR"
mkdir -p "$APP_DIR"/{data,logs,backups}
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

# 4. Install Caddy
log "Installing Caddy web server..."
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/setup.deb.sh' | bash
apt install -y caddy

# 5. Deploy application
log "Deploying application files..."
# Copy your application files to $APP_DIR
# This assumes you're running from the app directory
cp -r . "$APP_DIR/app/"
chown -R "$APP_USER:$APP_USER" "$APP_DIR/app"

# 6. Setup Python environment
log "Setting up Python virtual environment..."
sudo -u "$APP_USER" python3 -m venv "$APP_DIR/.venv"
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/app/requirements.txt"

# 7. Copy and install systemd service
log "Installing systemd service..."
cp "$APP_DIR/app/mordzix.service" "/etc/systemd/system/"
# Update paths in service file
sed -i "s|/opt/mordzix|$APP_DIR|g" "/etc/systemd/system/mordzix.service"
sed -i "s|User=www-data|User=$APP_USER|g" "/etc/systemd/system/mordzix.service"
sed -i "s|Group=www-data|Group=$APP_USER|g" "/etc/systemd/system/mordzix.service"
systemctl daemon-reload
systemctl enable mordzix

# 8. Setup Caddy
log "Configuring Caddy reverse proxy..."
cp "$APP_DIR/app/Caddyfile" "/etc/caddy/"
sed -i "s|your-domain.com|$DOMAIN|g" "/etc/caddy/Caddyfile"
systemctl enable caddy

# 9. Setup health watchdog
log "Installing health watchdog..."
cp "$APP_DIR/app/health_watchdog.sh" "/usr/local/bin/"
chmod +x "/usr/local/bin/health_watchdog.sh"

# Create watchdog service
cat > "/etc/systemd/system/mordzix-watchdog.service" << EOF
[Unit]
Description=Mordzix Health Watchdog
After=mordzix.service
Wants=mordzix.service

[Service]
Type=simple
User=root
ExecStart=/usr/local/bin/health_watchdog.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable mordzix-watchdog

# 10. Setup firewall
log "Configuring firewall..."
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

# 11. Setup log rotation
log "Setting up log rotation..."
cat > "/etc/logrotate.d/mordzix" << EOF
/var/log/mordzix-watchdog.log {
    daily
    missingok
    rotate 30
    compress
    notifempty
    create 644 root root
}
EOF

# 12. Database integrity check
log "Checking database integrity..."
sudo -u "$APP_USER" sqlite3 "$APP_DIR/data/memory.db" "PRAGMA integrity_check;" || warn "Database integrity check failed"
sudo -u "$APP_USER" sqlite3 "$APP_DIR/data/memory.db" "PRAGMA wal_checkpoint(TRUNCATE);" || warn "WAL checkpoint failed"

# 13. Start services
log "Starting services..."
systemctl start mordzix
systemctl start mordzix-watchdog
systemctl start caddy

# 14. Health check
log "Waiting for services to start..."
sleep 10

# Check local health
if curl -f "http://localhost:$SERVICE_PORT/health" >/dev/null 2>&1; then
    log "✅ Local health check passed"
else
    warn "❌ Local health check failed"
fi

# Check through proxy
if curl -f "http://localhost/health" >/dev/null 2>&1; then
    log "✅ Proxy health check passed"
else
    warn "❌ Proxy health check failed"
fi

log "🚀 Deployment completed!"
log ""
log "Next steps:"
log "1. Update your DNS to point $DOMAIN to this server's IP"
log "2. Test the deployment: curl -I https://$DOMAIN/health"
log "3. Monitor logs: journalctl -fu mordzix"
log "4. Check status: systemctl status mordzix mordzix-watchdog caddy"
log ""
log "Service endpoints:"
log "- Health: https://$DOMAIN/health"
log "- WebSocket: wss://$DOMAIN/ws"
log "- API: https://$DOMAIN/run"