# Struktura projektu MRD69

## 📋 Przegląd plików

### 🎯 Główne pliki
- **server.py** - Główny serwer FastAPI, obsługuje routing, czat, historie
- **requirements.txt** - Lista wszystkich zależności Pythona
- **.env.example** - Przykładowa konfiguracja zmiennych środowiskowych
- **.gitignore** - Pliki ignorowane przez Git
- **README.md** - Główna dokumentacja projektu

### 📁 Katalogi

#### `/routers` - Routery API
Specjalistyczne endpointy API podzielone tematycznie:

- **crypto.py** - Endpoints do doradztwa kryptowalutowego
- **travel.py** - Endpoints do przewodnika podróży
- **writing.py** - Endpoints do asystenta pisania

#### `/src` - Kod źródłowy
Główna logika aplikacji:

- **llm_client.py** - Klient do komunikacji z LLM (Large Language Models)
- **memory.py** - System pamięci aplikacji
- **memory_api.py** - Router API dla systemu pamięci
- **api_bridge.py** - Most między różnymi API
- **config.py** - Konfiguracja aplikacji
- **io_pipeline.py** - Pipeline wejścia/wyjścia
- **reliability_core.py** - Rdzeń niezawodności systemu
- **runpod_sync.py** - Synchronizacja z RunPod

**Asystenci AI:**
- **autonauka.py** - System automatycznej nauki
- **programista.py** - Asystent programisty
- **psychika.py** - Asystent psychologiczny
- **travelguide.py** - Przewodnik turystyczny
- **crypto_advisor_full.py** - Pełny doradca kryptowalutowy
- **writing_all_pro.py** - Profesjonalny asystent pisania

**Klienty:**
- **file_client.py** - Klient do operacji na plikach
- **images_client.py** - Klient do obsługi obrazów

**Pomocnicze:**
- **prompt.py** - Zarządzanie promptami
- **print_routes.py** - Wyświetlanie dostępnych routów
- **env_guard.py** - Ochrona zmiennych środowiskowych
- **fix_writing_all_pro.py** - Naprawa asystenta pisania

#### `/frontend` - Interfejs użytkownika
Prosty frontend webowy:

- **index.html** - Główna strona HTML
- **app.js** - Logika JavaScript
- **style.css** - Style CSS

#### `/webapp` - Alternatywna aplikacja webowa
Dodatkowa wersja interfejsu webowego w katalogu `static/`

#### `/scripts` - Skrypty pomocnicze
Skrypty bash do zarządzania aplikacją:

- **start.sh** - Uruchomienie serwera
- **run_all.sh** - Uruchomienie wszystkich serwisów
- **setup_all.sh** - Instalacja i konfiguracja
- **test_api.sh** - Testowanie API
- **test_dialog.sh** - Testowanie dialogu
- **test_env.sh** - Testowanie zmiennych środowiskowych

#### `/data` - Dane aplikacji (nie w repo)
Bazy danych i pliki danych:

- **chat_history.db** - SQLite baza historii czatów
- **memory.db** - SQLite baza pamięci systemowej
- **memory.jsonl** - Pamięć w formacie JSONL
- **export.json** - Eksportowane dane
- **start.seed.jsonl** - Dane startowe
- **sq3/** - Dodatkowe pliki SQLite

#### `/logs` - Logi aplikacji (nie w repo)
Pliki logów:

- **server.log** - Logi serwera
- **backend.log** - Logi backendu
- **start.log** - Logi uruchomienia
- **start.pid** - PID procesu

#### `/history` - Historia operacji
Pliki JSONL z historią operacji (format: `YYYYMMDD_HHMMSS_hash.jsonl`)

#### `/uploads` - Przesłane pliki (nie w repo)
Pliki przesłane przez użytkowników

#### `/reports` - Raporty
Raporty techniczne:

- **routes_summary.txt** - Podsumowanie routów
- **routes_preview.txt** - Podgląd routów
- **all_endpoints_and_hits.txt** - Lista wszystkich endpointów
- **files_list.txt** - Lista plików

#### `/docs` - Dokumentacja
Dokumentacja projektu:

- **README.md** - Główna dokumentacja (kopia)
- **USAGE.md** - Instrukcja użytkowania
- **PROJECT_STRUCTURE.md** - Ten plik
- **report_*.md** - Różne raporty

## 🔄 Przepływ działania aplikacji

```
1. Użytkownik → Frontend (index.html, app.js)
2. Frontend → Server.py (FastAPI)
3. Server.py → Router (crypto/travel/writing)
4. Router → LLM Client → Model AI
5. Model AI → Response
6. Response → Database (SQLite)
7. Response → Frontend → Użytkownik
```

## 🗄️ Schemat bazy danych

### Tabela: chat_threads
- `id` (TEXT) - ID czatu
- `title` (TEXT) - Tytuł rozmowy
- `created_ts` (INTEGER) - Timestamp utworzenia
- `updated_ts` (INTEGER) - Timestamp aktualizacji

### Tabela: chat_messages
- `id` (INTEGER) - ID wiadomości
- `chat_id` (TEXT) - ID czatu
- `role` (TEXT) - Rola (user/assistant)
- `content` (TEXT) - Treść wiadomości
- `ts` (INTEGER) - Timestamp

## 🔌 Integracje

- **FastAPI** - Framework webowy
- **SQLite** - Baza danych
- **LLM Providers** - DeepInfra, OpenAI (opcjonalnie)
- **RunPod** - Synchronizacja w chmurze (opcjonalnie)

## 📦 Zależności główne

- `fastapi` - Framework webowy
- `uvicorn` - Serwer ASGI
- `pydantic` - Walidacja danych
- `requests` - HTTP client
- `psutil` - Informacje systemowe
- `numpy`, `pandas` - Analiza danych
- `reportlab` - Generowanie PDF
- `aiohttp` - Async HTTP
- `pytest` - Testy

## 🎯 Dalszy rozwój

Możliwe kierunki rozwoju:
1. Dodanie testów jednostkowych w `/tests`
2. Dockeryzacja aplikacji
3. Dodanie CI/CD
4. Rozbudowa frontiendu (React/Vue?)
5. Więcej asystentów AI
6. System kolejkowania zadań
7. WebSockets dla real-time
8. Autentykacja i autoryzacja
