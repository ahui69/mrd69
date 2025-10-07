# 🧪 TESTY - KOMPLETNY PRZEWODNIK (dla początkujących)

## 🎯 **PO CO TESTY? - PRAWDZIWE PRZYKŁADY**

### **SYTUACJA 1: Zmiana kodu**

**BEZ TESTÓW:**
```
TY:     Zmieniam endpoint /api/health
TY:     Hmm, wydaje się działać...
TY:     *deploy na serwer*
KLIENT: Aplikacja nie działa!!!
TY:     *panika* Co się stało?!
        *szukasz błędu 2 godziny*
        Aha, zmieniłem "ok" na "status" 🤦
```

**Z TESTAMI:**
```
TY:     Zmieniam endpoint /api/health  
TY:     pytest test_server.py
PYTEST: ❌ FAILED: KeyError: 'ok'
TY:     O, to źle. Naprawiam.
TY:     pytest test_server.py
PYTEST: ✅ 17 passed
TY:     OK, teraz mogę deploy
KLIENT: Wszystko działa! 😊
```

**CZAS ZAOSZCZĘDZONY: 2 godziny stresu!**

---

### **SYTUACJA 2: Nowa funkcja**

**BEZ TESTÓW:**
```
TY: Dodaję nowy endpoint
TY: *testuje ręcznie w przeglądarce*
TY: Działa!
*za tydzień*
INNY DEWELOPER: Zmieniam coś w server.py
                *nie wie że zepsuł Twój endpoint*
                *deploy*
PRODUKCJA:      💥 Twój endpoint nie działa 💥
```

**Z TESTAMI:**
```
TY: Dodaję endpoint + test
INNY DEV: Zmienia coś
INNY DEV: pytest
PYTEST: ❌ test_twoj_endpoint FAILED
INNY DEV: Aha, zepsułem coś. Naprawiam.
PRODUKCJA: ✅ Wszystko działa
```

---

## 💰 **KONKRETNE KORZYŚCI:**

### **1. OSZCZĘDZASZ CZAS**
```
Ręczne testowanie 10 endpointów = 15 minut
Pytest 50 testów = 20 sekund

MIESIĘCZNIE (testowanie 2x dziennie):
Bez testów: 15 min × 2 × 30 = 15 godzin
Z testami:  20 sek × 2 × 30 = 20 minut

ZAOSZCZĘDZONY CZAS: 14 godzin 40 minut!
```

### **2. PEWNOŚĆ SIEBIE**
```
Bez testów: "Mam nadzieję że działa... 😰"
Z testami:  "17 testów passed, jestem pewny! 😎"
```

### **3. ŁATWIEJSZY REFACTORING**
```
Chcesz zmienić jak działa baza danych?

Bez testów: Strach zmienić, bo coś się zepsuje
Z testami:  Zmieniasz → pytest → widzisz co nie działa → naprawiasz
```

### **4. DOKUMENTACJA KODU**
```python
def test_chat_endpoint_accepts_message():
    """
    Ten test POKAZUJE jak używać endpointu!
    Lepsze niż 10 stron dokumentacji.
    """
    payload = {"newMessage": {"role": "user", "content": "test"}}
    response = client.post("/api/chat", json=payload)
    # ← Każdy widzi JAK DOKŁADNIE wysłać request
```

---

## 🎓 **JAK DZIAŁAJĄ TESTY - KROK PO KROKU:**

### **1. PROSTY TEST:**

```python
def test_dodawanie():
    wynik = 2 + 3
    assert wynik == 5  # ← ASSERT = "upewnij się że"
```

**Co się dzieje:**
```
1. Python uruchamia funkcję test_dodawanie()
2. Oblicza 2 + 3 = 5
3. Sprawdza: czy 5 == 5?
4. TAK → ✅ PASSED
   NIE → ❌ FAILED (pokazuje gdzie błąd)
```

---

### **2. TEST ENDPOINTU:**

```python
def test_health():
    response = client.get("/api/health")  # Wywołaj endpoint
    assert response.status_code == 200    # Sprawdź status
    assert "ok" in response.json()        # Sprawdź dane
```

**Co się dzieje:**
```
1. TestClient udaje przeglądarkę
2. Wysyła GET request do /api/health
3. Dostaje odpowiedź (tak jak curl)
4. Sprawdza czy status = 200
5. Sprawdza czy w JSON jest pole "ok"
6. Wszystko OK? ✅ PASSED
```

---

### **3. TEST Z DANYMI (POST):**

```python
def test_chat():
    # Przygotuj dane (jak w curl)
    payload = {
        "newMessage": {
            "role": "user",
            "content": "Cześć"
        }
    }
    
    # Wyślij (jak curl -X POST)
    response = client.post("/api/chat", json=payload)
    
    # Sprawdź odpowiedź
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
```

**To jest TO SAMO co:**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"newMessage": {"role":"user", "content":"Cześć"}}'
```

**ALE:** 
- Automatyczne
- Sprawdza czy działa
- Pokazuje błędy dokładnie

---

## 🔥 **TWOJE TESTY - KONKRETNE PRZYKŁADY:**

### **PRZYKŁAD 1: Test że chat zapisuje się do bazy**

```python
def test_chat_saves_to_database():
    # 1. Wyślij wiadomość
    response = client.post("/api/chat", json={
        "newMessage": {"role": "user", "content": "Test zapisu"}
    })
    chat_id = response.json()["chatId"]
    
    # 2. Sprawdź czy jest w historii
    history = client.get(f"/api/history/{chat_id}")
    messages = history.json()["messages"]
    
    # 3. Sprawdź czy Twoja wiadomość tam jest
    user_messages = [m["content"] for m in messages if m["role"] == "user"]
    assert "Test zapisu" in user_messages
    
    # ✅ PASSED = wiadomość zapisała się do SQLite!
```

**CO TO SPRAWDZA:**
- ✅ Endpoint /api/chat działa
- ✅ Dane zapisują się do SQLite
- ✅ Endpoint /api/history działa
- ✅ Odczyt z bazy działa

**JEDEN TEST = 4 rzeczy sprawdzone!**

---

### **PRZYKŁAD 2: Test crypto endpointu**

```python
def test_crypto_screener_structure():
    response = client.get("/api/crypto/screener?limit=5")
    
    # Może być 503 (brak API key) - to OK
    if response.status_code == 200:
        data = response.json()
        
        # Sprawdź strukturę
        assert "items" in data
        assert "count" in data
        assert isinstance(data["items"], list)
        
        # Jeśli są dane, sprawdź format
        if data["items"]:
            first = data["items"][0]
            assert "id" in first or "symbol" in first
```

**CO TO DAJE:**
```
✅ Sprawdza że struktura się nie zmieni
✅ Frontend oczekuje {"items": [...], "count": N}
✅ Jak zmienisz na {"results": [...]}, test ci powie!
```

---

### **PRZYKŁAD 3: Test performance**

```python
def test_health_is_fast():
    import time
    start = time.time()
    response = client.get("/api/health")
    duration = time.time() - start
    
    assert duration < 1.0, f"Za wolne! Trwało {duration}s"
```

**CO TO DAJE:**
```
Dzisiaj:  health trwa 0.05s → ✅ PASSED
Za tydzień: dodałeś wolne zapytanie do DB
            health trwa 3.5s → ❌ FAILED!
            
Natychmiast wiesz że coś spowolniło aplikację!
```

---

## 🎯 **JAK UŻYWAĆ TESTÓW - WORKFLOW:**

### **Każdego dnia:**
```bash
# 1. Piszesz kod
nano server.py

# 2. Uruchamiasz testy
pytest test_server.py -v

# 3a. Wszystko OK?
✅ 17 passed → commit i deploy

# 3b. Coś nie działa?
❌ 2 failed → naprawiasz → pytest → ✅ passed → commit
```

---

### **Przed deploym na produkcję:**
```bash
# Uruchom WSZYSTKIE testy
pytest -v

# Jak wszystko przeszło → deploy
# Jak coś failed → NIE DEPLOYUJ, napraw najpierw
```

---

### **Gdy dodajesz nową funkcję:**
```bash
# 1. Napisz test NAJPIERW (TDD)
def test_nowy_endpoint():
    response = client.get("/api/nowy")
    assert response.status_code == 200

# 2. Uruchom test
pytest
# ❌ FAILED (bo endpoint nie istnieje)

# 3. Napisz endpoint
@app.get("/api/nowy")
def nowy():
    return {"ok": True}

# 4. Uruchom test ponownie
pytest
# ✅ PASSED!
```

---

## 📊 **STATYSTYKI KTÓRE POKAŻĄ CZY WARTO:**

### **Projekt BEZ testów:**
```
Bugs znalezione przez klientów: 45/miesiąc
Czas debugowania: 15 godzin/miesiąc
Downtime produkcji: 4 godziny/miesiąc
Stres level: 9/10 😰
```

### **Projekt Z testami:**
```
Bugs znalezione przez testy: 42/miesiąc  
Bugs znalezione przez klientów: 3/miesiąc
Czas debugowania: 2 godziny/miesiąc
Downtime produkcji: 0 godzin/miesiąc
Stres level: 3/10 😎
```

---

## 💼 **TESTY = PIENIĄDZE**

### **Na rozmowie kwalifikacyjnej:**

**BEZ testów w projekcie:**
```
REKRUTER: "Czy masz testy?"
TY:       "Nie, ale kod działa..."
REKRUTER: "Hmm... *red flag*"

SZANSA NA PRACĘ: 40%
```

**Z testami w projekcie:**
```
REKRUTER: "Czy masz testy?"
TY:       "Tak, 50+ testów, coverage 80%"
REKRUTER: "Wow, po miesiącu nauki?! 😲"

SZANSA NA PRACĘ: 90%
MOŻLIWA WYPŁATA: +20% bo jesteś "advanced junior"
```

---

### **Freelance projekty:**

**BEZ testów:**
```
KLIENT: "Napraw ten bug"
TY:     "Naprawiłem"
        *naprawiasz bug A*
        *psujesz funkcję B*
KLIENT: "Teraz to nie działa!"
TY:     "Naprawiłem"
        *naprawiasz B*
        *psujesz C*

KLIENT RATING: ⭐⭐ (2/5)
PŁATNOŚĆ: Standard
```

**Z testami:**
```
TY:     Naprawiam bug A
TY:     pytest
PYTEST: ❌ test_funkcja_B failed
TY:     Naprawiam też B
TY:     pytest  
PYTEST: ✅ wszystko działa
TY:     Deploy

KLIENT RATING: ⭐⭐⭐⭐⭐ (5/5)
        "Szybko, bez bugów, profesjonalnie!"
PŁATNOŚĆ: +50% tip + recommendation
```

---

## 🛠️ **JAK ZACZĄĆ - PRAKTYCZNY PLAN:**

### **DZIEŃ 1-2: Podstawy**
```bash
# 1. Zainstaluj pytest
pip install pytest pytest-asyncio httpx

# 2. Przeczytaj test_server.py
cat test_server.py

# 3. Uruchom testy
pytest test_server.py -v

# 4. Zmień coś w server.py (np. nazwę pola)
# 5. Uruchom testy - zobacz co się zepsuło
# 6. Napraw
```

### **DZIEŃ 3-5: Pisanie własnych testów**
```python
# Dodaj test dla swojego endpointu:

def test_moj_nowy_endpoint():
    response = client.get("/api/test")
    assert response.status_code == 200
    assert response.json()["message"] == "działa"
```

### **DZIEŃ 6-10: Zaawansowane**
```
- Fixtures (setup/teardown)
- Parametryzowane testy
- Mockowanie
- Coverage (ile % kodu pokryte testami)
```

---

## 📈 **COVERAGE - CO TO I PO CO?**

**Coverage = ile % kodu jest przetestowane**

```bash
# Zainstaluj
pip install pytest-cov

# Uruchom z coverage
pytest --cov=server --cov-report=term-missing

# Wynik:
# server.py    85%    Lines 45-47, 123-125 nie pokryte
```

**Pokazuje Ci DOKŁADNIE które linie nie są testowane!**

```
✅ 100% coverage = każda linia przetestowana (idealnie)
✅ 80%+ coverage = bardzo dobre (standard w firmach)
✅ 50%+ coverage = OK dla początkujących
❌ <30% coverage = za mało
```

---

## 🎯 **PRAKTYCZNE PRZYKŁADY DLA TWOJEGO KODU:**

### **TEST 1: Upload pliku (z Twojego server.py)**

```python
def test_upload_real_file():
    """Symulacja prawdziwego uploadu"""
    
    # Stwórz fejkowy plik
    file_content = b"To jest testowy plik"
    files = {"file": ("test.txt", file_content, "text/plain")}
    
    # Upload
    response = client.post("/api/upload", files=files)
    
    # Sprawdź
    assert response.status_code == 200
    data = response.json()
    assert "path" in data
    
    # Sprawdź czy plik faktycznie jest na dysku
    import os
    assert os.path.exists(data["path"]), "Plik nie został zapisany!"
    
    # CLEANUP (posprzątaj po teście)
    try:
        os.remove(data["path"])
    except:
        pass
```

---

### **TEST 2: Chat z różnymi językami**

```python
@pytest.mark.parametrize("lang,greeting", [
    ("pl", "Cześć"),
    ("en", "Hello"),
    ("de", "Hallo"),
])
def test_chat_multilanguage(lang, greeting):
    """Test czy chat działa w różnych językach"""
    
    payload = {
        "newMessage": {
            "role": "user",
            "content": greeting
        },
        "lang": lang
    }
    
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    assert len(response.json()["reply"]) > 0
```

**JEDEN TEST = 3 języki sprawdzone!**

---

### **TEST 3: Historia czatów (Twoja baza SQLite)**

```python
def test_chat_history_order():
    """Historia powinna być posortowana (najnowsze pierwsze)"""
    
    # Wyślij 3 wiadomości
    for i in range(3):
        client.post("/api/chat", json={
            "newMessage": {"role": "user", "content": f"Wiadomość {i}"}
        })
        import time
        time.sleep(0.1)  # Mała pauza żeby timestamps były różne
    
    # Pobierz historię
    response = client.get("/api/history")
    chats = response.json()
    
    # Sprawdź że są posortowane (najnowsze pierwsze)
    timestamps = [c["ts"] for c in chats if "ts" in c]
    assert timestamps == sorted(timestamps, reverse=True), \
        "Historia nie jest posortowana!"
```

---

## 🔥 **PRAWDZIWY PRZYKŁAD Z ŻYCIA:**

### **CASE STUDY: Bug który testy by złapały**

**Twój kod (przed testami):**
```python
@app.post("/api/chat")
def api_chat(req: ChatRequest):
    # ... kod ...
    reply = llm.chat(messages=current)  # ← CO JAK llm = None?
    return {"reply": reply}
```

**CO SIĘ STANIE:**
```
1. LLM nie skonfigurowane (brak klucza)
2. llm = None
3. llm.chat(...) → 💥 AttributeError: 'NoneType' object has no attribute 'chat'
4. Serwer zwraca 500 (Internal Server Error)
5. Klient widzi białą stronę
```

**Z TESTEM:**
```python
def test_chat_without_llm_config(monkeypatch):
    # Symuluj brak LLM
    monkeypatch.setattr("server.LLM_AVAILABLE", False)
    
    response = client.post("/api/chat", json={
        "newMessage": {"role": "user", "content": "test"}
    })
    
    # Powinno działać (tryb echo)
    assert response.status_code == 200
    assert "echo" in response.json()["reply"]
```

**WYNIK:**
```
❌ FAILED: AttributeError
→ Naprawiasz PRZED deploym (już to zrobiłem w Twoim kodzie!)
→ Teraz działa w trybie echo gdy brak LLM
```

---

## 📚 **COMMANDS CHEAT SHEET:**

```bash
# Uruchom wszystkie testy
pytest

# Uruchom z verbose (pokaż szczegóły)
pytest -v

# Uruchom tylko jeden plik
pytest test_server.py

# Uruchom tylko jeden test
pytest test_server.py::test_health_endpoint_exists

# Uruchom testy które mają "chat" w nazwie
pytest -k chat

# Pokaż print() w testach (debug)
pytest -s

# Zatrzymaj na pierwszym błędzie
pytest -x

# Coverage (pokrycie kodu)
pytest --cov=server --cov-report=html
# Potem otwórz: htmlcov/index.html

# Uruchom tylko szybkie testy (jeśli oznaczyłeś @pytest.mark.fast)
pytest -m fast

# Raport w stylu JUnit (dla CI/CD)
pytest --junitxml=report.xml
```

---

## 🎓 **PRZYKŁADY CO TESTOWAĆ W TWOIM PROJEKCIE:**

### **Dla server.py:**
```python
✅ Czy endpointy istnieją (200 vs 404)
✅ Czy zwracają prawidłowy JSON
✅ Czy zapisują do bazy
✅ Czy obsługują błędne dane (nie crashują)
✅ Czy są szybkie (<1s)
```

### **Dla routers/crypto.py:**
```python
✅ Czy screener zwraca listę
✅ Czy token/{id} zwraca dane tokena
✅ Czy portfolio się zapisuje
✅ Czy obsługuje brak API key (503)
```

### **Dla src/llm_client.py:**
```python
✅ Czy chat() wysyła request
✅ Czy parsuje odpowiedź
✅ Czy obsługuje timeout
✅ Czy retry działa przy błędzie
```

---

## 💪 **FINALNA RADA:**

### **Zacznij MAŁYMI krokami:**

**DZIEŃ 1:**
```python
# Jeden test
def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
```

**DZIEŃ 7:**
```python
# 5 testów
test_health()
test_chat()
test_history()
test_bootstrap()
test_upload()
```

**DZIEŃ 30:**
```python
# 50+ testów
- 20 testów endpointów
- 15 testów bazy danych
- 10 testów logiki biznesowej
- 5 testów performance
```

---

## 🔥 **PODSUMOWANIE:**

### **TESTY DAJĄ CI:**

1. **PEWNOŚĆ** - wiesz że kod działa
2. **CZAS** - automatyzacja zamiast ręcznego testowania
3. **DOKUMENTACJĘ** - test pokazuje jak używać
4. **OCHRONĘ** - nie zepsujesz przypadkiem
5. **SPOKÓJ** - deploy bez strachu
6. **KASĘ** - klienci płacą więcej za quality
7. **PRACĘ** - firmy wymagają testów

### **BEZ TESTÓW:**
```
- Testowanie ręczne 1h dziennie
- Bugs na produkcji
- Klienci niezadowoleni
- Stres przy każdej zmianie
```

### **Z TESTAMI:**
```
- pytest = 20 sekund
- Bugs łapane przed deploym
- Klienci zadowoleni
- Spokój i pewność
```

---

## 🎯 **TWOJA NASTĘPNA AKCJA:**

```bash
# 1. Zobacz swoje testy (już napisane!)
cat test_server.py

# 2. Uruchom je
pytest test_server.py -v

# 3. Zmień coś w server.py (np. nazwę pola)
# 4. Uruchom testy - zobacz że failują
# 5. Napraw
# 6. Zobacz że passują

# 7. Napisz JEDEN własny test
# 8. Dodawaj po 1 teście dziennie

# Za 30 dni będziesz miał 50+ testów!
```

---

**MORDO, TESTY = TO NIE JEST OPCJA, TO JEST MUST-HAVE!**

**Każdy profesjonalny projekt ma testy.**
**Każda firma wymaga testów.**
**Każdy senior developer pisze testy.**

**Zacznij teraz, za 30 dni będziesz z przodu 90% juniorów!** 🚀