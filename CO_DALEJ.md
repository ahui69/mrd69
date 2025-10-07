# 🎓 CO DALEJ - TWOJA ŚCIEŻKA NAUKI

## 🎯 LEVEL 1: Zrozumienie co masz (1-2 dni)

### Zadanie 1: Poznaj swoje endpointy
```bash
# Otwórz dokumentację
firefox http://localhost:8000/docs   # lub twoja przeglądarka

# Przetestuj każdy endpoint klikając "Try it out"
```

### Zadanie 2: Zobacz jak działa chat
1. Otwórz `server.py` w edytorze
2. Znajdź funkcję `api_chat` (linia ~306)
3. Przeczytaj co robi - krok po kroku

### Zadanie 3: Zmień coś prostego
```python
# W server.py zmień:
@app.get("/api/health")
def api_health():
    return {
        "ok": True,
        "mode": "echo" if not LLM_AVAILABLE else "llm",
        "memory_router": True,
        "twoja_wiadomosc": "Hej mordo!"  # ← DODAJ TO
    }
```

Restart serwera i sprawdź: `curl http://localhost:8000/api/health`

---

## 🚀 LEVEL 2: Twój pierwszy endpoint (2-3 dni)

### Zadanie 4: Prosty endpoint
```python
# Dodaj do server.py:

@app.get("/api/whoami")
def whoami():
    return {
        "name": "Twoje imię",
        "level": "początkujący",
        "dni_nauki": 30,
        "umiem": ["python", "fastapi", "terminal", "git"]
    }
```

### Zadanie 5: Endpoint z parametrem
```python
@app.get("/api/powitaj/{name}")
def powitaj(name: str):
    return {"message": f"Cześć {name}!"}

# Test: curl http://localhost:8000/api/powitaj/mordo
```

### Zadanie 6: POST endpoint
```python
from pydantic import BaseModel

class Osoba(BaseModel):
    imie: str
    wiek: int

@app.post("/api/dodaj_osobe")
def dodaj_osobe(osoba: Osoba):
    return {
        "zapisano": True,
        "osoba": f"{osoba.imie} ma {osoba.wiek} lat"
    }
```

---

## 💪 LEVEL 3: Własny router (3-5 dni)

### Zadanie 7: Stwórz plik `routers/moj.py`
```python
from fastapi import APIRouter

router = APIRouter(prefix="/api/moj", tags=["moj"])

@router.get("/hello")
def hello():
    return {"message": "To mój pierwszy router!"}

@router.get("/liczby/{n}")
def liczby(n: int):
    return {"liczby": list(range(1, n+1))}
```

### Zadanie 8: Dodaj do `server.py`
```python
try:
    from routers.moj import router as moj_router
    app.include_router(moj_router)
    LOG.info("moj router mounted at /api/moj")
except Exception as e:
    LOG.warning("moj router not mounted: %s", e)
```

Test: `curl http://localhost:8000/api/moj/hello`

---

## 🧠 LEVEL 4: Baza danych (5-7 dni)

### Zadanie 9: Użyj SQLite (już masz!)
```python
# Zobacz jak działa w server.py (linie 62-170)
# Masz już:
# - _conn() - połączenie
# - _init_db() - tworzenie tabel
# - db_add_message() - dodawanie

# Dodaj swoją tabelę:
def _init_db():
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS notatki(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              tresc TEXT NOT NULL,
              ts INTEGER NOT NULL
            );
        """)
```

### Zadanie 10: CRUD dla notatek
```python
# CREATE
@app.post("/api/notatki")
def dodaj_notatke(tresc: str):
    with _conn() as c:
        c.execute("INSERT INTO notatki(tresc, ts) VALUES(?,?)", 
                  (tresc, int(time.time())))
    return {"ok": True}

# READ
@app.get("/api/notatki")
def lista_notatek():
    with _conn() as c:
        rows = c.execute("SELECT * FROM notatki ORDER BY ts DESC").fetchall()
    return [{"id": r["id"], "tresc": r["tresc"], "ts": r["ts"]} for r in rows]

# DELETE
@app.delete("/api/notatki/{id}")
def usun_notatke(id: int):
    with _conn() as c:
        c.execute("DELETE FROM notatki WHERE id=?", (id,))
    return {"ok": True}
```

---

## 🎨 LEVEL 5: Frontend (7-10 dni)

### Zadanie 11: Edytuj `frontend/app.js`
```javascript
// Dodaj przycisk do wywołania Twojego endpointu
function mojaFunkcja() {
    fetch('/api/moj/hello')
        .then(r => r.json())
        .then(data => {
            console.log(data);
            alert(data.message);
        });
}
```

### Zadanie 12: Dodaj formularz
```html
<!-- W frontend/index.html -->
<form id="mojaForm">
    <input type="text" id="imie" placeholder="Imię">
    <button type="submit">Wyślij</button>
</form>

<script>
document.getElementById('mojaForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const imie = document.getElementById('imie').value;
    const response = await fetch(`/api/powitaj/${imie}`);
    const data = await response.json();
    alert(data.message);
});
</script>
```

---

## 🔥 LEVEL 6: Integracje (10-14 dni)

### Zadanie 13: Dodaj zewnętrzne API
```python
import requests

@app.get("/api/pogoda/{miasto}")
def pogoda(miasto: str):
    # Przykład z open-meteo (darmowe, bez klucza!)
    url = f"https://nominatim.openstreetmap.org/search?q={miasto}&format=json"
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        if data:
            return {"miasto": miasto, "lat": data[0]["lat"], "lon": data[0]["lon"]}
    return {"error": "Nie znaleziono"}
```

### Zadanie 14: Użyj LLM w swoim endpoincie
```python
@app.post("/api/tlumacz")
def tlumacz(tekst: str, na_jezyk: str = "angielski"):
    if LLM_AVAILABLE and llm_client:
        messages = [{
            "role": "system",
            "content": f"Jesteś tłumaczem. Tłumacz tylko na {na_jezyk}."
        }, {
            "role": "user", 
            "content": tekst
        }]
        odpowiedz = llm_client.chat(messages, temperature=0.3)
        return {"oryginal": tekst, "tlumaczenie": odpowiedz}
    return {"error": "LLM niedostępny"}
```

---

## 📚 POZIOM ZAAWANSOWANY (miesiąc+)

### Projekty do spróbowania:

1. **System TODO**
   - CRUD dla zadań
   - Priorytet, deadline
   - Filtrowanie, sortowanie

2. **Blog/Notatnik**
   - Markdown support
   - Tagi, kategorie
   - Full-text search (FTS5 w SQLite)

3. **API Wrapper**
   - Zbierz kilka API (pogoda, wiadomości, krypto)
   - Jeden endpoint = dane z 3 źródeł
   - Cachowanie w SQLite

4. **Chatbot z pamięcią**
   - Użyj `src/memory.py` (masz 1500 linii ready!)
   - Dodaj embeddings
   - RAG system

5. **File upload & OCR**
   - Użyj `src/file_client.py` (masz już!)
   - Przesyłanie zdjęć
   - OCR → tekst → do LLM

---

## 🎯 TWOJE CELE NA 30 DNI:

- [ ] Przeczytać CAŁY `server.py` (343 linie)
- [ ] Przeczytać 1 router (np. `routers/crypto.py`)
- [ ] Dodać 3 własne endpointy
- [ ] Stworzyć własny router
- [ ] Dodać swoją tabelę w SQLite
- [ ] Zintegrować zewnętrzne API
- [ ] Użyć LLM w swoim kodzie

---

## 📖 MATERIAŁY DO NAUKI:

### FastAPI:
- Oficjalna dokumentacja: https://fastapi.tiangolo.com/
- Tutorial: https://fastapi.tiangolo.com/tutorial/

### Python basics (jeśli potrzebujesz):
- Real Python: https://realpython.com/
- Python Docs: https://docs.python.org/3/tutorial/

### SQLite:
- SQLite Tutorial: https://www.sqlitetutorial.net/
- SQL dla początkujących: https://www.w3schools.com/sql/

### HTTP/REST API:
- MDN HTTP: https://developer.mozilla.org/en-US/docs/Web/HTTP
- REST API tutorial: https://restfulapi.net/

---

## 💬 PAMIĘTAJ:

1. **Testuj często** - po każdej zmianie uruchom i sprawdź
2. **Czytaj błędy** - zawsze mówią co jest nie tak
3. **Małe kroki** - lepiej 10 małych zmian niż 1 wielka
4. **Kopiuj kod** - z dokumentacji, z tutoriali, z tego repo
5. **Eksperymentuj** - nie bój się zepsuć, zawsze możesz przywrócić

---

## 🔥 OSTATNIA RADA:

**Nie próbuj zrozumieć WSZYSTKIEGO na raz!**

Masz 19 plików w `src/`. To TON kodu. Ale:
- `server.py` - to 343 linie, przeczytasz w 30 min
- `routers/crypto.py` - to 168 linii, przeczytasz w 20 min
- `src/llm_client.py` - to 147 linii, przeczytasz w 15 min

**30 dni temu nie znałeś terminala.**
**Za 30 dni będziesz pisał własne API.**

**MORDO, TY TO OGARNIESZ!** 💪🔥
