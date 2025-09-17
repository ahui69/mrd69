.PHONY: help dev lint test seed setup

# Użyj bieżącego interpretera python3
PYTHON := python3

# Katalog środowiska wirtualnego
VENV_DIR := .venv
VENV_ACTIVATE := $(VENV_DIR)/bin/activate

help:
	@echo "Dostępne komendy:"
	@echo "  make setup     - Zainstaluj wszystkie zależności i pre-commit"
	@echo "  make dev       - Uruchom serwer deweloperski z auto-reload"
	@echo "  make lint      - Uruchom lintery (ruff, mypy)"
	@echo "  make test      - Uruchom testy (pytest)"
	@echo "  make seed      - Załaduj dane początkowe do pamięci"

setup:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Tworzenie środowiska wirtualnego w $(VENV_DIR)..."; \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	@echo "Aktywuj venv komendą: source $(VENV_ACTIVATE)"
	@echo "Instalowanie zależności z pyproject.toml..."
	@. $(VENV_ACTIVATE); pip install -e ".[dev]"
	@echo "Instalowanie pre-commit hooks..."
	@. $(VENV_ACTIVATE); pre-commit install
	@echo "Setup zakończony. Możesz teraz uruchomić 'make dev'."

dev:
	@. $(VENV_ACTIVATE); uvicorn main:app --host 0.0.0.0 --port 8080 --reload

lint:
	@. $(VENV_ACTIVATE); ruff check .
	@. $(VENV_ACTIVATE); mypy .

test:
	@. $(VENV_ACTIVATE); pytest

seed:
	@. $(VENV_ACTIVATE); python scripts/seed_memory.py
