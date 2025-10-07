#!/bin/bash
# Prosty skrypt do uruchamiania serwera

echo "🚀 Uruchamiam serwer..."

# Zatrzymaj stary proces jeśli działa
pkill -f "uvicorn server:app" 2>/dev/null
sleep 1

# Uruchom serwer
cd /workspace
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
