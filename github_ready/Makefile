# ════════════════════════════════════════════════════════
# MAKEFILE - SUPER PROSTE KOMENDY
# 
# Zamiast pisać: docker-compose up -d --build
# Piszesz:       make start
# ════════════════════════════════════════════════════════

.PHONY: help install start stop restart logs test clean docker-start docker-stop

# Domyślna komenda (gdy wpiszesz tylko "make")
help:
	@echo "════════════════════════════════════════"
	@echo "  TWOJA APPKA - DOSTĘPNE KOMENDY"
	@echo "════════════════════════════════════════"
	@echo ""
	@echo "📦 SETUP:"
	@echo "  make install        - Zainstaluj zależności"
	@echo ""
	@echo "🚀 URUCHOMIENIE (normalnie):"
	@echo "  make start          - Uruchom serwer"
	@echo "  make stop           - Zatrzymaj serwer"
	@echo "  make restart        - Restart serwera"
	@echo "  make logs           - Zobacz logi"
	@echo ""
	@echo "🐳 DOCKER:"
	@echo "  make docker-build   - Zbuduj Docker image"
	@echo "  make docker-start   - Uruchom w Dockerze"
	@echo "  make docker-stop    - Zatrzymaj Docker"
	@echo "  make docker-logs    - Logi z Dockera"
	@echo ""
	@echo "🧪 TESTY:"
	@echo "  make test           - Uruchom wszystkie testy"
	@echo "  make test-fast      - Tylko szybkie testy"
	@echo "  make coverage       - Testy + coverage"
	@echo ""
	@echo "🧹 CLEANUP:"
	@echo "  make clean          - Wyczyść cache/logi"
	@echo "  make clean-all      - Wyczyść WSZYSTKO"
	@echo ""
	@echo "════════════════════════════════════════"

# ═══════════════════════════════════════════════════════
# INSTALACJA
# ═══════════════════════════════════════════════════════

install:
	@echo "📦 Instaluję zależności..."
	pip install -r requirements.txt
	pip install -r requirements-test.txt
	@echo "✅ Gotowe!"

# ═══════════════════════════════════════════════════════
# URUCHOMIENIE (normalnie, bez Dockera)
# ═══════════════════════════════════════════════════════

start:
	@echo "🚀 Uruchamiam serwer..."
	@pkill -f "uvicorn server:app" 2>/dev/null || true
	@sleep 1
	@python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload

stop:
	@echo "🛑 Zatrzymuję serwer..."
	@pkill -f "uvicorn server:app" || true
	@echo "✅ Serwer zatrzymany"

restart: stop
	@sleep 1
	@make start

logs:
	@tail -f server_run.log 2>/dev/null || tail -f logs/server.log 2>/dev/null || echo "Brak logów"

# ═══════════════════════════════════════════════════════
# DOCKER
# ═══════════════════════════════════════════════════════

docker-build:
	@echo "🐳 Buduję Docker image..."
	docker-compose build
	@echo "✅ Image zbudowany!"

docker-start:
	@echo "🐳 Uruchamiam w Dockerze..."
	docker-compose up -d
	@echo "✅ Kontener działa!"
	@echo "📖 Docs: http://localhost:8000/docs"
	@echo "🎨 App:  http://localhost:8000/app"

docker-stop:
	@echo "🛑 Zatrzymuję Docker..."
	docker-compose down
	@echo "✅ Zatrzymany"

docker-logs:
	docker-compose logs -f

docker-restart:
	docker-compose restart

docker-shell:
	@echo "🐚 Wchodzę do kontenera..."
	docker-compose exec app bash

# ═══════════════════════════════════════════════════════
# TESTY
# ═══════════════════════════════════════════════════════

test:
	@echo "🧪 Uruchamiam testy..."
	pytest -v

test-fast:
	@echo "⚡ Szybkie testy..."
	pytest -v -m "not slow"

coverage:
	@echo "📊 Testy + coverage..."
	pytest --cov=server --cov=routers --cov-report=term-missing

coverage-html:
	@echo "📊 Coverage HTML report..."
	pytest --cov=server --cov=routers --cov-report=html
	@echo "✅ Otwórz: htmlcov/index.html"

# ═══════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════

clean:
	@echo "🧹 Czyszczę cache..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage
	@echo "✅ Cache wyczyszczony"

clean-all: clean
	@echo "🧹 Czyszczę WSZYSTKO..."
	rm -rf logs/*.log data/*.db-wal data/*.db-shm
	docker-compose down -v 2>/dev/null || true
	@echo "✅ Wszystko wyczyszczone"

# ═══════════════════════════════════════════════════════
# DEV HELPERS
# ═══════════════════════════════════════════════════════

check:
	@echo "🔍 Sprawdzam środowisko..."
	@python3 --version
	@pip --version
	@docker --version 2>/dev/null || echo "⚠️  Docker nie zainstalowany"
	@pytest --version 2>/dev/null || echo "⚠️  pytest nie zainstalowany"

health:
	@echo "💚 Health check..."
	@curl -s http://localhost:8000/api/health | python3 -m json.tool || echo "❌ Serwer nie działa"

endpoints:
	@echo "📡 Dostępne endpointy:"
	@curl -s http://localhost:8000/openapi.json | python3 -c "import json,sys;d=json.load(sys.stdin);[print(f'{m.upper():7} {p}') for p in sorted(d['paths'].keys()) for m in d['paths'][p].keys()]" 2>/dev/null || echo "❌ Serwer nie działa"

# ═══════════════════════════════════════════════════════
# QUICK ACTIONS
# ═══════════════════════════════════════════════════════

dev: install start
	@echo "✅ Development environment ready!"

prod: docker-build docker-start
	@echo "✅ Production environment ready!"

# ════════════════════════════════════════════════════════
# PRZYKŁADY UŻYCIA:
# 
# make               → pokaż help
# make install       → zainstaluj biblioteki
# make start         → uruchom normalnie
# make docker-start  → uruchom w Dockerze
# make test          → uruchom testy
# make logs          → zobacz logi
# make clean         → wyczyść cache
# ════════════════════════════════════════════════════════
