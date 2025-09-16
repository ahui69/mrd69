#!/bin/bash
# health_watchdog.sh - Monitor Mordzix server health and restart on failures

HEALTH_URL="http://localhost:6969/health"
SERVICE_NAME="mordzix"
CHECK_INTERVAL=30
MAX_FAILURES=3
LOG_FILE="/var/log/mordzix-watchdog.log"

failure_count=0

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG_FILE"
}

check_health() {
    local response
    local exit_code
    
    response=$(curl -s -w "%{http_code}" -o /dev/null --max-time 10 "$HEALTH_URL" 2>/dev/null)
    exit_code=$?
    
    if [ $exit_code -eq 0 ] && [ "$response" = "200" ]; then
        return 0
    else
        return 1
    fi
}

restart_service() {
    log_message "CRITICAL: Restarting $SERVICE_NAME after $MAX_FAILURES consecutive failures"
    systemctl restart "$SERVICE_NAME"
    
    if [ $? -eq 0 ]; then
        log_message "INFO: Service $SERVICE_NAME restarted successfully"
        failure_count=0
        # Wait a bit for service to start
        sleep 15
    else
        log_message "ERROR: Failed to restart service $SERVICE_NAME"
    fi
}

log_message "INFO: Starting health watchdog for $SERVICE_NAME"
log_message "INFO: Monitoring $HEALTH_URL every ${CHECK_INTERVAL}s"

while true; do
    if check_health; then
        if [ $failure_count -gt 0 ]; then
            log_message "INFO: Health check passed - resetting failure count (was $failure_count)"
            failure_count=0
        fi
    else
        failure_count=$((failure_count + 1))
        log_message "WARNING: Health check failed ($failure_count/$MAX_FAILURES)"
        
        if [ $failure_count -ge $MAX_FAILURES ]; then
            restart_service
        fi
    fi
    
    sleep $CHECK_INTERVAL
done