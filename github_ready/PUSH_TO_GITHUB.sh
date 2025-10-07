#!/bin/bash
# ═══════════════════════════════════════════════════════════
# WRZUĆ KOD NA GITHUB
# Uruchom: bash PUSH_TO_GITHUB.sh
# ═══════════════════════════════════════════════════════════

echo "🔄 Pushuję kod na GitHub..."

# Sprawdź czy jesteś w github_ready
if [ ! -f "server.py" ]; then
    echo "❌ Uruchom to z katalogu github_ready!"
    exit 1
fi

# Git init (jeśli nie ma)
if [ ! -d ".git" ]; then
    git init
    git remote add origin https://github.com/ahui69/mrd69.git
fi

# Dodaj wszystko
git add .

# Commit
git commit -m "Deploy to RunPod - cleaned code" || echo "Brak zmian do commit"

# Push
echo "📤 Pushuję na GitHub..."
git push -u origin master || git push -u origin main

echo "✅ Kod na GitHubie!"
echo ""
echo "🔗 Repo: https://github.com/ahui69/mrd69"
