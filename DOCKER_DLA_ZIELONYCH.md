#!/bin/bash
# ════════════════════════════════════════════════════════
# DOCKER - ŚCIĄGA KOMEND (kopiuj i używaj)
# ════════════════════════════════════════════════════════

# ─────────────────────────────────────
# PODSTAWOWE (nauczysz się w 5 minut)
# ─────────────────────────────────────

# Uruchom appkę (zbuduje jeśli trzeba)
docker-compose up -d

# Zatrzymaj
docker-compose down

# Zobacz logi
docker-compose logs -f

# Restart
docker-compose restart

# ─────────────────────────────────────
# PRZYDATNE
# ─────────────────────────────────────

# Zbuduj od nowa (po zmianach w kodzie)
docker-compose up -d --build

# Sprawdź status
docker-compose ps

# Wejdź do środka kontenera (jak SSH)
docker-compose exec app bash
# Teraz jesteś WEWNĄTRZ kontenera!
# Możesz: ls, cat server.py, python3, itp.
# Wyjście: exit

# Zobacz co się dzieje (top)
docker-compose top

# Usuń WSZYSTKO (kontener + volumes)
docker-compose down -v

# ─────────────────────────────────────
# DEBUGGING
# ─────────────────────────────────────

# Zobacz logi tylko z ostatnich 50 linii
docker-compose logs --tail=50

# Zobacz logi z ostatniej godziny
docker-compose logs --since 1h

# Sprawdź ile zajmuje
docker system df

# Wyczyść stare obrazy (oszczędź miejsce)
docker system prune -a

# ─────────────────────────────────────
# ZAAWANSOWANE (opcjonalnie)
# ─────────────────────────────────────

# Zbuduj bez cache (od zera)
docker-compose build --no-cache

# Uruchom tylko jeden serwis
docker-compose up -d app

# Skalowanie (2 instancje)
docker-compose up -d --scale app=2

# Export obrazu do pliku
docker save moja-appka > moja-appka.tar

# Import obrazu z pliku
docker load < moja-appka.tar

# Push do Docker Hub (jak git push)
docker tag moja-appka twojanazwa/moja-appka
docker push twojanazwa/moja-appka

# ─────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────

# Usuń wszystkie zatrzymane kontenery
docker container prune

# Usuń nieużywane obrazy
docker image prune

# Usuń WSZYSTKO (UWAGA!)
docker system prune -a --volumes

# ════════════════════════════════════════════════════════
# WORKFLOW CODZIENNY:
# ════════════════════════════════════════════════════════

# Rano:
docker-compose up -d          # Uruchom
docker-compose logs -f        # Zobacz co się dzieje

# Zmieniasz kod:
nano server.py                # Edytuj
docker-compose restart        # Restart (szybki)
# LUB
docker-compose up -d --build  # Rebuild (po większych zmianach)

# Wieczorem:
docker-compose down           # Zatrzymaj

# ════════════════════════════════════════════════════════
# TROUBLESHOOTING:
# ════════════════════════════════════════════════════════

# "Cannot connect to Docker daemon"
# → Uruchom Docker Desktop (GUI) lub service:
sudo systemctl start docker

# "Port 8000 already in use"  
# → Zatrzymaj stary kontener:
docker-compose down
# LUB zmień port w docker-compose.yml:
# ports: - "8001:8000"

# "No space left on device"
# → Wyczyść stare obrazy:
docker system prune -a

# Kontener się wywala co chwilę
# → Zobacz logi:
docker-compose logs --tail=100

# ════════════════════════════════════════════════════════
