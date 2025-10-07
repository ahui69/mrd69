# 📝 ŚCIĄGA - TESTY (kopiuj-wklej gotowe wzorce)

## 🎯 BASIC TEMPLATE - kopiuj i zmień

```python
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)

def test_NAZWA():
    """Opisz co testuje"""
    response = client.get("/api/TWOJ_ENDPOINT")
    
    assert response.status_code == 200
    data = response.json()
    assert "POLE" in data
```

---

## 📦 **GOTOWE WZORCE - KOPIUJ:**

### **1. Test GET endpointu**
```python
def test_get_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["ok"] == True
```

### **2. Test POST endpointu**
```python
def test_post_endpoint():
    payload = {"key": "value"}
    response = client.post("/api/endpoint", json=payload)
    assert response.status_code == 200
```

### **3. Test z parametrem URL**
```python
def test_endpoint_with_param():
    response = client.get("/api/user/123")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "123"
```

### **4. Test z query params**
```python
def test_endpoint_with_query():
    response = client.get("/api/search?q=test&limit=10")
    assert response.status_code == 200
    assert len(response.json()["items"]) <= 10
```

### **5. Test błędnych danych**
```python
def test_rejects_bad_data():
    response = client.post("/api/chat", json={})
    assert response.status_code in [400, 422]  # Bad Request lub Validation Error
```

### **6. Test że zwraca listę**
```python
def test_returns_list():
    response = client.get("/api/history")
    data = response.json()
    assert isinstance(data, list)
```

### **7. Test że zwraca dict**
```python
def test_returns_dict():
    response = client.get("/api/user/1")
    data = response.json()
    assert isinstance(data, dict)
    assert "id" in data
```

### **8. Test uploadu pliku**
```python
def test_file_upload():
    files = {"file": ("test.txt", b"content", "text/plain")}
    response = client.post("/api/upload", files=files)
    assert response.status_code == 200
```

### **9. Test z wieloma wartościami**
```python
@pytest.mark.parametrize("value,expected", [
    (1, 2),
    (5, 10),
    (10, 20),
])
def test_double(value, expected):
    assert value * 2 == expected
```

### **10. Test wyjątków**
```python
def test_raises_error():
    with pytest.raises(ValueError):
        # kod który powinien rzucić ValueError
        raise ValueError("test")
```

---

## 🔧 **FIXTURE TEMPLATES:**

### **Fixture: Testowy chat**
```python
@pytest.fixture
def test_chat():
    """Tworzy czat do testów"""
    response = client.post("/api/chat", json={
        "newMessage": {"role": "user", "content": "test"}
    })
    return response.json()

def test_using_chat(test_chat):
    chat_id = test_chat["chatId"]
    # Użyj chat_id w teście
    assert chat_id is not None
```

### **Fixture: Testowy użytkownik**
```python
@pytest.fixture
def test_user():
    """Tworzy użytkownika, po teście usuwa"""
    # Setup
    user_id = "test_user_123"
    # ... stwórz użytkownika ...
    
    yield user_id  # Zwróć do testu
    
    # Teardown (cleanup)
    # ... usuń użytkownika ...
```

### **Fixture: Testowa baza**
```python
@pytest.fixture
def clean_db():
    """Czysta baza przed każdym testem"""
    # Backup obecnej bazy
    import shutil
    shutil.copy("data/chat_history.db", "data/backup.db")
    
    yield  # Uruchom test
    
    # Przywróć backup
    shutil.copy("data/backup.db", "data/chat_history.db")
```

---

## 🎯 **ASSERT PATTERNS (co sprawdzać):**

### **Sprawdzanie typów:**
```python
assert isinstance(data, dict)
assert isinstance(items, list)
assert isinstance(count, int)
assert isinstance(price, float)
assert isinstance(name, str)
```

### **Sprawdzanie wartości:**
```python
assert value == expected
assert value > 0
assert value >= 10
assert len(items) > 0
assert "key" in data
```

### **Sprawdzanie stringów:**
```python
assert "error" not in response.text
assert message.startswith("Success")
assert "test" in message.lower()
```

### **Sprawdzanie list:**
```python
assert len(items) == 5
assert "bitcoin" in [i["id"] for i in items]
assert all(i["price"] > 0 for i in items)
```

---

## 🚀 **WORKFLOW NA CODZIEŃ:**

### **Rano:**
```bash
git pull                    # Pobierz zmiany
pytest                      # Sprawdź czy wszystko działa
```

### **Podczas kodowania:**
```bash
# Zmień kod
nano server.py

# Uruchom testy
pytest test_server.py

# Jak failed → napraw → pytest
```

### **Przed commitem:**
```bash
pytest -v                   # Wszystkie testy
git add .
git commit -m "dodałem feature X, testy passed"
```

### **Przed deploym:**
```bash
pytest --cov=server         # Testy + coverage
# Jak passed + coverage >70% → deploy
```

---

## 📊 **METRYKI - ŚLEDŹ POSTĘP:**

### **Twoje cele:**
```
DZIEŃ 1:   3 testy   | coverage 20%  | ✅ START
TYDZIEŃ 1: 10 testów | coverage 40%  | ✅ DOBRZE
TYDZIEŃ 2: 20 testów | coverage 60%  | ✅ ŚWIETNIE
MIESIĄC:   50 testów | coverage 80%  | ✅ PROFESJONALNIE
```

### **Śledź w projekcie:**
```bash
# Policz testy
grep -r "^def test_" . | wc -l

# Coverage
pytest --cov=server --cov=routers

# Coverage raport
pytest --cov=server --cov-report=term-missing
```

---

## 🎓 **NAUKA - KOLEJNOŚĆ:**

### **TYDZIEŃ 1: Podstawy**
- [ ] Uruchom test_server.py
- [ ] Przeczytaj każdy test
- [ ] Napisz 1 własny test

### **TYDZIEŃ 2: Praktyka**
- [ ] Test dla każdego endpointu (10+)
- [ ] Nauczysz się assert
- [ ] Nauczysz się parametrize

### **TYDZIEŃ 3: Zaawansowane**
- [ ] Fixtures
- [ ] Coverage >50%
- [ ] Testy integracyjne

### **TYDZIEŃ 4: Pro**
- [ ] Mocking
- [ ] Coverage >80%
- [ ] CI/CD (GitHub Actions)

---

## 💰 **TESTY = PIENIĄDZE (konkretnie):**

### **Freelance:**
```
Projekt BEZ testów: $1000
Projekt Z testami:  $1500  (+50%)

Dlaczego więcej?
- Mniej bugów = mniej poprawek = oszczędność czasu klienta
- Profesjonalny wygląd
- Łatwiejsze maintenance
```

### **Praca:**
```
Junior BEZ testów: $30k/rok
Junior Z testami:  $40k/rok  (+33%)

Mid Z testami:     $70k/rok
Senior Z testami:  $100k+/rok
```

---

## 🔥 **OSTATNIA RZECZ:**

**Uruchom teraz:**
```bash
pytest test_server.py -v
```

**Zobacz:**
```
✅ 17 passed w 12 sekund

Te 17 testów sprawdziło:
- 7 endpointów
- Bazę danych
- Upload plików
- Performance
- Security (SQL injection)
- Integration flow

Ręcznie byś testował to 20 minut.
```

**TO JEST POWÓD!** 🎯

Masz więcej pytań o testy? Chcesz zobaczyć jakiś konkretny przykład?
