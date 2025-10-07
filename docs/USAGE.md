# Instrukcja użytkowania

## 🚀 Szybki start

### 1. Instalacja zależności
```bash
pip install -r requirements.txt
```

### 2. Konfiguracja
```bash
# Skopiuj przykładowy plik konfiguracji
cp .env.example .env

# Edytuj plik .env i dodaj swoje klucze API
nano .env
```

### 3. Uruchomienie serwera
```bash
# Opcja 1: Przez skrypt (zalecane)
./scripts/start.sh

# Opcja 2: Ręcznie
uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# Opcja 3: Wszystkie serwisy
./scripts/run_all.sh
```

## 📡 Dostęp do aplikacji

Po uruchomieniu:
- **API**: http://localhost:8000
- **Frontend**: http://localhost:8000/app
- **Dokumentacja API**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🧪 Testowanie

```bash
# Test API
./scripts/test_api.sh

# Test dialogu z czatem
./scripts/test_dialog.sh

# Test zmiennych środowiskowych
./scripts/test_env.sh
```

## 📝 Przykłady użycia API

### Chat - Wysyłanie wiadomości
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "newMessage": {
      "role": "user",
      "content": "Cześć, jak się masz?"
    }
  }'
```

### Historia czatów
```bash
# Lista wszystkich czatów
curl http://localhost:8000/api/history

# Konkretny czat
curl http://localhost:8000/api/history/{chat_id}
```

### Upload pliku
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@/ścieżka/do/pliku.jpg"
```

### Health check
```bash
curl http://localhost:8000/api/health
```

## 🔧 Komponenty projektu

### Routers (Routery)
- **crypto.py** - Doradca kryptowalutowy
- **travel.py** - Przewodnik turystyczny
- **writing.py** - Asystent pisarza

### Src (Kod źródłowy)
- **llm_client.py** - Klient do modeli językowych
- **memory.py** - System pamięci
- **memory_api.py** - API pamięci
- **autonauka.py** - Automatyczna nauka
- **programista.py** - Asystent programisty
- **psychika.py** - Asystent psychologiczny

## 🗄️ Baza danych

Aplikacja używa SQLite do przechowywania:
- Historii czatów
- Wiadomości
- Pamięci systemowej

Pliki bazy danych znajdują się w katalogu `data/`:
- `chat_history.db` - Historia czatów
- `memory.db` - Pamięć systemowa

## 📂 Struktura katalogów

```
.
├── server.py           # Główny serwer
├── requirements.txt    # Zależności
├── .env               # Konfiguracja (nie w repo)
├── .env.example       # Przykładowa konfiguracja
├── README.md          # Dokumentacja główna
│
├── routers/           # Endpointy API
├── src/               # Kod źródłowy
├── frontend/          # Interfejs użytkownika
├── scripts/           # Skrypty pomocnicze
├── docs/              # Dokumentacja
├── data/              # Bazy danych (nie w repo)
├── logs/              # Logi (nie w repo)
└── uploads/           # Przesłane pliki (nie w repo)
```

## 💡 Wskazówki

1. **Pierwszy raz z Pythonem?**
   - Zainstaluj Python 3.8+ z python.org
   - Użyj wirtualnego środowiska: `python -m venv venv`
   - Aktywuj: `source venv/bin/activate` (Linux/Mac) lub `venv\Scripts\activate` (Windows)

2. **Problemy z uruchomieniem?**
   - Sprawdź czy port 8000 jest wolny: `lsof -i :8000`
   - Sprawdź logi w katalogu `logs/`
   - Upewnij się, że masz wszystkie zależności: `pip install -r requirements.txt`

3. **Rozwój projektu**
   - Dodawaj nowe routery w katalogu `routers/`
   - Kod źródłowy trzymaj w `src/`
   - Testy umieszczaj w katalogu `tests/` (do utworzenia)

## 🆘 Pomoc

Jeśli masz problemy:
1. Sprawdź logi w `logs/`
2. Sprawdź czy wszystkie zależności są zainstalowane
3. Upewnij się, że plik `.env` jest poprawnie skonfigurowany
4. Sprawdź dokumentację FastAPI: https://fastapi.tiangolo.com/
