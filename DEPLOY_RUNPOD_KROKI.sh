#!/bin/bash
# ═══════════════════════════════════════════════════════════
# DEPLOY NA RUNPOD - KOMENDY DO WKLEJENIA
# ═══════════════════════════════════════════════════════════

# ──────────────────────────────────────
# POŁĄCZ SIĘ (na swoim komputerze):
# ──────────────────────────────────────
ssh h36o457520pycq-64411a08@ssh.runpod.io -i ~/.ssh/id_ed25519

# ──────────────────────────────────────
# TERAZ JESTEŚ NA RUNPOD!
# Kopiuj i wklejaj poniższe komendy:
# ──────────────────────────────────────

# 1. Sprawdź gdzie jesteś
pwd
ls -la

# 2. Idź do workspace (jeśli jest) lub stwórz
cd /workspace || mkdir -p /workspace && cd /workspace

# 3. Usuń stary kod (jeśli był)
rm -rf mrd69

# 4. Ściągnij kod z GitHub
git clone https://github.com/ahui69/mrd69.git
cd mrd69

# 5. Sprawdź co się ściągnęło
ls -la

# 6. Zainstaluj zależności
pip install -r requirements.txt

# 7. Sprawdź czy .env jest (powinien być w repo)
ls -la .env

# 8. URUCHOM SERWER!
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000

# ──────────────────────────────────────
# SERWER DZIAŁA!
# ──────────────────────────────────────

# Sprawdź w RunPod Console jaki masz URL
# Będzie coś jak: https://abc123-8000.proxy.runpod.net

# Test (z twojego komputera):
# curl https://TWOJ_URL/api/health

# ──────────────────────────────────────
# ŻEBY DZIAŁAŁO W TLE (po rozłączeniu SSH):
# ──────────────────────────────────────

# Zatrzymaj serwer (Ctrl+C)
# Potem uruchom w tle:

nohup python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &

# Sprawdź czy działa:
curl http://localhost:8000/api/health

# Zobacz logi:
tail -f server.log

# Rozłącz SSH (serwer dalej działa!):
exit
