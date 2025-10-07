#!/bin/bash
# Skrypt do zatrzymania serwera

echo "🛑 Zatrzymuję serwer..."
pkill -f "uvicorn server:app"
sleep 1
echo "✅ Serwer zatrzymany"
