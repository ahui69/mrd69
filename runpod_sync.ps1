# runpod_sync.ps1 - PowerShell deployment script for Windows
param(
    [string]$Action = "deploy",
    [switch]$SkipBuild = $false,
    [switch]$TailLogs = $false
)

# Configuration
$RUNPOD_HOST = "213.192.2.99"
$RUNPOD_PORT = "41882" 
$RUNPOD_USER = "root"
$SSH_KEY = "$env:USERPROFILE\.ssh\runpod_g70"
$REMOTE_DIR = "/workspace/mordzix"

Write-Host "🚀 Mordzix RunPod Deployment (PowerShell)" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# Test SSH connection
Write-Host "Testing SSH connection..." -ForegroundColor Yellow
$testConnection = ssh -i $SSH_KEY -p $RUNPOD_PORT -o ConnectTimeout=10 "$RUNPOD_USER@$RUNPOD_HOST" "echo 'SSH OK'" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Cannot connect to RunPod instance" -ForegroundColor Red
    exit 1
}
Write-Host "✅ SSH connection successful" -ForegroundColor Green

switch ($Action) {
    "deploy" {
        Write-Host "📦 Deploying Mordzix to RunPod..." -ForegroundColor Cyan
        
        # Create remote directories
        ssh -i $SSH_KEY -p $RUNPOD_PORT "$RUNPOD_USER@$RUNPOD_HOST" "mkdir -p $REMOTE_DIR/data $REMOTE_DIR/logs $REMOTE_DIR/static"
        
        # Sync files using scp (since rsync might not be available on Windows)
        Write-Host "📤 Uploading files..." -ForegroundColor Yellow
        
        # Core Python files
        scp -i $SSH_KEY -P $RUNPOD_PORT *.py "$RUNPOD_USER@${RUNPOD_HOST}:${REMOTE_DIR}/"
        scp -i $SSH_KEY -P $RUNPOD_PORT requirements.txt "$RUNPOD_USER@${RUNPOD_HOST}:${REMOTE_DIR}/"
        scp -i $SSH_KEY -P $RUNPOD_PORT Dockerfile "$RUNPOD_USER@${RUNPOD_HOST}:${REMOTE_DIR}/"
        
        # Static files
        scp -i $SSH_KEY -P $RUNPOD_PORT -r static "$RUNPOD_USER@${RUNPOD_HOST}:${REMOTE_DIR}/"
        
        # Install and start
        Write-Host "🔧 Installing dependencies and starting service..." -ForegroundColor Yellow
        ssh -i $SSH_KEY -p $RUNPOD_PORT "$RUNPOD_USER@$RUNPOD_HOST" "
cd $REMOTE_DIR
pip install -r requirements.txt
pkill -f uvicorn ; pkill -f mordzix
sleep 2
nohup python -m uvicorn mordzix_server:app --host 0.0.0.0 --port 8080 --workers 1 > logs/mordzix.log 2>&1 &
sleep 5
curl -f http://localhost:8080/health && echo 'Mordzix is running!' || echo 'Health check failed'
"
    }
    
    "status" {
        Write-Host "📊 Checking Mordzix status..." -ForegroundColor Cyan
        ssh -i $SSH_KEY -p $RUNPOD_PORT "$RUNPOD_USER@$RUNPOD_HOST" "
cd $REMOTE_DIR
ps aux | grep -E 'uvicorn|mordzix' | grep -v grep
echo '=== Health Check ==='
curl -s http://localhost:8080/health | jq . || curl -s http://localhost:8080/health
echo '=== Disk Usage ==='
du -sh data/ logs/ 2>/dev/null || echo 'No data/logs yet'
"
    }
    
    "logs" {
        if ($TailLogs) {
            Write-Host "📜 Tailing Mordzix logs (Ctrl+C to exit)..." -ForegroundColor Cyan
            ssh -i $SSH_KEY -p $RUNPOD_PORT "$RUNPOD_USER@$RUNPOD_HOST" "tail -f $REMOTE_DIR/logs/mordzix.log"
        } else {
            Write-Host "📜 Recent Mordzix logs..." -ForegroundColor Cyan
            ssh -i $SSH_KEY -p $RUNPOD_PORT "$RUNPOD_USER@$RUNPOD_HOST" "tail -50 $REMOTE_DIR/logs/mordzix.log 2>/dev/null || echo 'No logs yet'"
        }
    }
    
    "restart" {
        Write-Host "🔄 Restarting Mordzix..." -ForegroundColor Cyan
        ssh -i $SSH_KEY -p $RUNPOD_PORT "$RUNPOD_USER@$RUNPOD_HOST" "
cd $REMOTE_DIR
pkill -f uvicorn ; pkill -f mordzix
sleep 3
nohup python -m uvicorn mordzix_server:app --host 0.0.0.0 --port 8080 --workers 1 > logs/mordzix.log 2>&1 &
sleep 5
curl -f http://localhost:8080/health && echo 'Restart successful!' || echo 'Restart failed'
"
    }
    
    "test" {
        Write-Host "🧪 Testing Mordzix API..." -ForegroundColor Cyan
        ssh -i $SSH_KEY -p $RUNPOD_PORT "$RUNPOD_USER@$RUNPOD_HOST" "
cd $REMOTE_DIR
echo '=== Health Check ==='
curl -s http://localhost:8080/health
echo -e '\n=== Chat Test ==='
curl -s -X POST http://localhost:8080/mordzix/chat -H 'Content-Type: application/json' -d '{\"user_id\":\"test\",\"content\":\"Elo Mordzix!\"}'
echo -e '\n=== Crypto Test ==='  
curl -s -X POST 'http://localhost:8080/mordzix/crypto/score?token_id=bitcoin'
"
    }
    
    default {
        Write-Host "❓ Unknown action: $Action" -ForegroundColor Red
        Write-Host "Available actions: deploy, status, logs, restart, test" -ForegroundColor Yellow
        Write-Host "Example: .\runpod_sync.ps1 -Action deploy" -ForegroundColor Yellow
        Write-Host "Example: .\runpod_sync.ps1 -Action logs -TailLogs" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "🌟 Operation completed! Access Mordzix at: http://$RUNPOD_HOST:8080" -ForegroundColor Green