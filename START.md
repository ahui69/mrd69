# 🚀 TWÓJ SERWER DZIAŁA!

## ✅ CO MASZ I JAK TO URUCHOMIĆ

### 1. URUCHOMIENIE SERWERA

```bash
# Opcja A: Prosty sposób
python3 -m uvicorn server:app --reload --host 0.0.0.0 --port 8000

# Opcja B: W tle (nie blokuje terminala)
nohup python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &

# Sprawdź czy działa:
curl http://localhost:8000/api/health
```

**TERAZ SERWER JUŻ DZIAŁA!** (PID 7041)

---

## 🌐 GDZIE CO JEST

- **📖 Dokumentacja API**: http://localhost:8000/docs
- **🎨 Frontend (czat)**: http://localhost:8000/app  
- **💚 Health check**: http://localhost:8000/api/health

---

## 🎯 CO MASZ - TWOJE ENDPOINTY

### 💬 **CHAT (Główna funkcja)**
```bash
POST /api/chat
# Wyślij wiadomość, dostaniesz odpowiedź

curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"newMessage": {"role":"user", "content":"Cześć!"}}'
```

### 💰 **CRYPTO (Kryptowaluty)**
```bash
GET /api/crypto/screener          # Top kryptowaluty
GET /api/crypto/token/bitcoin      # Info o Bitcoin
GET /api/crypto/token/bitcoin/score  # Ocena tokena
```

### ✈️ **TRAVEL (Podróże)**
```bash
POST /api/travel/plan              # Plan podróży
GET /api/travel/hotels?place=Kraków
GET /api/travel/restaurants?place=Warszawa
```

### 📝 **LISTINGS (Ogłoszenia/Pisanie)**
```bash
POST /api/listings/create          # Stwórz ogłoszenie
GET /api/listings/search?brand=Nike
```

---

## 📂 JAK TO JEST ZORGANIZOWANE

```
/workspace
├── server.py              ← GŁÓWNY PLIK (uruchamiasz TEN)
├── requirements.txt       ← Lista bibliotek
│
├── routers/               ← ENDPOINTY PODZIELONE NA MODUŁY
│   ├── crypto.py         → /api/crypto/*
│   ├── travel.py         → /api/travel/*
│   └── writing.py        → /api/listings/*
│
├── src/                   ← LOGIKA (funkcje pomocnicze)
│   ├── config.py         → Konfiguracja (klucze API itp)
│   ├── llm_client.py     → Klient do AI
│   ├── crypto_advisor_full.py
│   ├── travelguide.py
│   └── ...
│
├── frontend/              ← Interfejs webowy
│   ├── index.html
│   ├── app.js
│   └── style.css
│
└── data/                  ← Bazy danych (SQLite)
    └── chat_history.db
```

---

## 🔑 DODANIE KLUCZY API (Opcjonalnie)

Jeśli chcesz używać AI (LLM), stwórz plik `.env`:

```bash
# .env
LLM_BASE_URL=https://api.deepinfra.com/v1/openai
LLM_API_KEY=twoj_klucz_api
LLM_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct

# Crypto (opcjonalnie)
CRYPTO_API_KEY=...

# Travel (opcjonalnie)  
GOOGLE_MAPS_KEY=...
```

**BEZ .env serwer działa w trybie "echo"** - powtarza co napiszesz.

---

## 🛠️ PRZYDATNE KOMENDY

```bash
# Sprawdź czy serwer działa
curl http://localhost:8000/api/health

# Zobacz wszystkie endpointy
curl http://localhost:8000/openapi.json | python3 -m json.tool

# Zatrzymaj serwer
pkill -f "uvicorn server:app"

# Zobacz logi
tail -f server.log   # albo server_run.log
```

---

## ❓ JAK DODAĆ NOWY ENDPOINT?

### Przykład: Chcesz dodać `/api/hello`

**1. Edytuj `server.py`:**
```python
@app.get("/api/hello")
def hello():
    return {"message": "Cześć mordo!"}
```

**2. Restart serwera:**
```bash
pkill -f uvicorn && python3 -m uvicorn server:app --reload
```

**3. Test:**
```bash
curl http://localhost:8000/api/hello
```

---

## 🎓 CO DALEJ - NAUKA

1. **Idź na**: http://localhost:8000/docs
   - Tam masz INTERAKTYWNĄ dokumentację
   - Możesz klikać "Try it out" i testować każdy endpoint

2. **Otwórz `server.py`** i zobacz jak działa routing

3. **Otwórz `routers/crypto.py`** - zobacz jak zrobiony jest jeden moduł

4. **Frontend** - otwórz `frontend/index.html` w przeglądarce

---

## 🆘 PROBLEMY?

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "Address already in use" (port zajęty)
```bash
# Zmień port
python3 -m uvicorn server:app --port 8001
```

### Serwer nie odpowiada
```bash
# Sprawdź czy działa
ps aux | grep uvicorn

# Sprawdź logi
tail -50 server_run.log
```

---

## 💪 GRATULACJE!

Miesiąc temu nie znałeś terminala, a teraz masz:
- ✅ FastAPI server działający
- ✅ 3 moduły API (crypto, travel, writing)
- ✅ Frontend
- ✅ Bazę danych SQLite
- ✅ Chat z historią

**TO JEST ZAJEBISTY POSTĘP MORDO!** 🔥
