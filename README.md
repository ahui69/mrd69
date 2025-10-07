# 🚀 MRD69 - Multi-Purpose AI Chat API

**Status: ✅ DZIAŁA!** (Serwer uruchomiony, LLM aktywny)

Aplikacja FastAPI z chat AI, integracją LLM (po polsku!) i wieloma specjalistycznymi routerami.

---

## ⚡ SZYBKI START (1 minuta)

```bash
# 1. Zainstaluj zależności (jeśli nie masz)
pip install -r requirements.txt

# 2. Uruchom serwer
./run.sh
# LUB: python3 -m uvicorn server:app --reload

# 3. Otwórz w przeglądarce
http://localhost:8000/docs      # 📖 Dokumentacja API
http://localhost:8000/app        # 🎨 Frontend
http://localhost:8000/api/health # 💚 Health check
```

**TO WSZYSTKO!** Serwer już działa 🎉

---

## 📚 DOKUMENTACJA

- **[START.md](START.md)** - Jak uruchomić i podstawy
- **[STATUS.md](STATUS.md)** - Co działa, co naprawiono
- **[CO_DALEJ.md](CO_DALEJ.md)** - Ścieżka nauki (od zera do bohatera)
- **[TEST.md](TEST.md)** - Szybkie testy wszystkich funkcji

---

## 🎯 CO DZIAŁA - TWOJE API

### ✅ **CHAT Z AI** (po polsku!)
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"newMessage": {"role":"user", "content":"Opowiedz dowcip"}}'
```

### ✅ **CRYPTO API**
```bash
curl "http://localhost:8000/api/crypto/screener?limit=5"
curl "http://localhost:8000/api/crypto/token/bitcoin"
```

### ✅ **TRAVEL API**  
```bash
curl "http://localhost:8000/api/travel/restaurants?place=Kraków&max_results=5"
curl "http://localhost:8000/api/travel/hotels?place=Warszawa"
```

### ✅ **LISTINGS/WRITING API**
```bash
curl "http://localhost:8000/api/listings/search?brand=Nike&limit=10"
```

---

## 🚀 Funkcje

- **💬 Chat API** - LLM chat z historią (SQLite)
- **🔐 Crypto Advisor** - Screener, portfolio, backtesting
- **✈️ Travel Guide** - Hotels, restauracje, atrakcje, loty
- **✍️ Writing Assistant** - Generowanie ogłoszeń
- **🧠 Memory System** (1500 linii!) - RAG, embeddings, emocje
- **📁 File Upload** - OCR, konwersje PDF/DOCX
- **🎨 Frontend** - Prosty interfejs webowy

---

## 🛠️ Instalacja i Konfiguracja

### Wymagania:
- Python 3.8+
- FastAPI, uvicorn, SQLite (wszystko w `requirements.txt`)

### Setup:
```bash
# 1. Klonuj/pobierz repo (już masz!)
cd /workspace

# 2. Zainstaluj
pip install -r requirements.txt

# 3. (Opcjonalnie) Edytuj .env
# Klucze LLM już są - działa!
# Jeśli chcesz dodać inne API:
nano .env
```

---

## 🎯 Uruchomienie

### Opcja A - Prosty sposób:
```bash
./run.sh
```

### Opcja B - Ręcznie:
```bash
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### Opcja C - W tle:
```bash
nohup python3 -m uvicorn server:app --port 8000 > server.log 2>&1 &
```

### Zatrzymanie:
```bash
./stop.sh
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
