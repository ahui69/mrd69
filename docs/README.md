# MRD69 - Multi-Purpose AI Chat API

Aplikacja FastAPI z interfejsem czatu, integracją LLM i wieloma specjalistycznymi routerami.

## 🚀 Funkcje

- **Chat API** - Czat z historią rozmów zapisaną w SQLite
- **Memory System** - System pamięci z API
- **Specialized Routers:**
  - 🔐 Crypto Advisor - Doradztwo kryptowalutowe
  - ✈️ Travel Guide - Przewodnik podróży
  - ✍️ Writing Assistant - Asystent pisania
- **Frontend** - Prosty interfejs webowy
- **File Upload** - Obsługa przesyłania plików
- **RunPod Integration** - Opcjonalna synchronizacja z RunPod

## 📋 Wymagania

- Python 3.8+
- FastAPI
- SQLite

## 🛠️ Instalacja

1. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

2. Skopiuj plik konfiguracyjny:
```bash
cp .env.example .env
```

3. Edytuj `.env` i uzupełnij swoje klucze API (opcjonalne)

## 🎯 Uruchomienie

### Szybki start:
```bash
./start.sh
```

### Lub ręcznie:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### Uruchomienie wszystkich serwisów:
```bash
./run_all.sh
```

## 📁 Struktura projektu

```
.
├── server.py              # Główny serwer FastAPI
├── routers/               # Routery API
│   ├── crypto.py         # Crypto advisor endpoints
│   ├── travel.py         # Travel guide endpoints
│   └── writing.py        # Writing assistant endpoints
├── frontend/             # Interfejs użytkownika
│   ├── index.html
│   ├── app.js
│   └── style.css
├── src/                  # Kod źródłowy aplikacji
│   ├── llm_client.py    # Klient LLM
│   ├── memory.py        # System pamięci
│   ├── memory_api.py    # API pamięci
│   └── ...
├── scripts/             # Skrypty pomocnicze
├── data/                # Bazy danych (nie w repo)
├── logs/                # Logi aplikacji (nie w repo)
└── requirements.txt     # Zależności Python
```

## 🔌 API Endpoints

### Główne endpointy:
- `GET /api/health` - Status aplikacji
- `GET /api/bootstrap` - Inicjalizacja
- `GET /api/history` - Lista czatów
- `GET /api/history/{chat_id}` - Historia konkretnego czatu
- `POST /api/chat` - Wyślij wiadomość
- `POST /api/upload` - Prześlij plik

### Routery specjalistyczne:
- `/api/memory/*` - Zarządzanie pamięcią
- `/api/listings/*` - Asystent pisania
- `/api/travel/*` - Przewodnik podróży
- `/api/crypto/*` - Doradca krypto

## 🔧 Konfiguracja

Zmienne środowiskowe (w pliku `.env`):
- `LLM_MODEL` - Model LLM (domyślnie: meta-llama/Meta-Llama-3.1-70B-Instruct)
- `LLM_PROVIDER` - Dostawca LLM (domyślnie: deepinfra)
- `LLM_TIMEOUT` - Timeout dla LLM (domyślnie: 30s)

## 🧪 Testowanie

```bash
# Test API
./test_api.sh

# Test dialogu
./test_dialog.sh

# Test zmiennych środowiskowych
./test_env.sh
```

## 📝 Rozwój

Projekt jest w aktywnym rozwoju. Główne komponenty:
- `autonauka.py` - Automatyczna nauka
- `programista.py` - Asystent programisty
- `psychika.py` - Asystent psychologiczny

## 🤝 Współpraca

To Twój pierwszy projekt - gratulacje! Możesz go rozwijać w dowolnym kierunku.

## 📄 Licencja

Projekt osobisty - do własnego użytku.
