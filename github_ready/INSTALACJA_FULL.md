# 🚀 INSTALACJA PEŁNEGO SYSTEMU

## ✅ CO DOSTAJESZ:

1. **AUTONAUKA FULL**
   - ✅ SERPAPI (masz wykupione!)
   - ✅ Firecrawl (masz wykupione!)
   - ✅ Wikipedia (FREE)
   - ✅ DuckDuckGo (FREE)
   - ✅ Zapis do LTM (nauka w locie!)
   - ✅ Streszczenia przez LLM

2. **NEWSY**
   - ✅ Google News przez SERPAPI
   - ✅ Najświeższe wiadomości

3. **SPORT - JUVENTUS TRACKER!** ⚽
   - ✅ Wyniki meczów Serie A
   - ✅ Ostatni mecz Juventusu
   - ✅ Następny mecz
   - ✅ **AUTO-CHECK CO GODZINĘ!**
   - ✅ Powiadomienia zapisywane do LTM

4. **FRONTEND**
   - ✅ Zegarek + data PL
   - ✅ Mikrofon (polski!)
   - ✅ Upload plików
   - ✅ Sidebar z historią
   - ✅ Safari iOS ready

---

## 📋 INSTALACJA NA RUNPOD (10 minut):

### **KROK 1: Pobierz pliki**

```bash
# SSH do RunPod
ssh root@213.192.2.99 -p 41724 -i ~/.ssh/runpod_ed25519

# Przejdź do folderu
cd /workspace/mrd69

# Pobierz nowe pliki
wget -O autonauka_full.py https://raw.githubusercontent.com/ahui69/mrd69/main/autonauka_full.py
wget -O frontend.html https://raw.githubusercontent.com/ahui69/mrd69/main/frontend_full.html

# Skopiuj frontend do /workspace/
cp frontend.html /workspace/frontend.html

# Sprawdź czy są
ls -lh autonauka_full.py frontend.html
```

---

### **KROK 2: Edytuj monolit.py**

```bash
# Otwórz monolit.py
nano monolit.py

# ZNAJDŹ linię (około 50-70):
# import os, re, sys, time, json, uuid...

# DODAJ IMPORT (po innych importach):
try:
    import autonauka_full
    AUTONAUKA_AVAILABLE = True
except Exception:
    AUTONAUKA_AVAILABLE = False

# ZAPISZ (Ctrl+O, Enter, Ctrl+X)
```

---

### **KROK 3: Dodaj endpointy**

```bash
# Otwórz ponownie
nano monolit.py

# ZNAJDŹ koniec pliku (przed if __name__ == "__main__":)
# Użyj Ctrl+W i wyszukaj: if __name__

# PRZED tą linią WKLEJ (skopiuj z endpoints_dodaj_do_monolit.py):
```

**Pobierz gotowe endpointy:**
```bash
wget -O /tmp/endpoints.txt https://raw.githubusercontent.com/ahui69/mrd69/main/endpoints_dodaj_do_monolit.py

# Zobacz co dodać
cat /tmp/endpoints.txt

# Skopiuj endpointy i wklej do monolit.py przed if __name__
```

**LUB ręcznie dodaj te 4 endpointy:**
```python
@app.post("/api/auto/learn")
async def auto_learn_endpoint(request: Request):
    body = await request.json()
    query = body.get("q", "")
    if not query:
        raise HTTPException(status_code=400, detail="Brak 'q'")
    result = autonauka_full.autonauka(query, topk=10)
    return result

@app.get("/api/news")
async def news_endpoint(q: str = "świat", limit: int = 10):
    return autonauka_full.get_news_sync(query=q, limit=limit)

@app.get("/api/sport/juventus")
async def juventus_endpoint():
    return autonauka_full.track_juventus_sync()

@app.get("/api/sport/football")
async def football_endpoint(league: str = "ita.1"):
    return autonauka_full.get_football_scores_sync(league=league)
```

---

### **KROK 4: Zainstaluj zależności**

```bash
# Upewnij się że masz wszystko
pip install --no-cache-dir httpx beautifulsoup4 lxml duckduckgo-search requests
```

---

### **KROK 5: RESTART**

```bash
# Zatrzymaj stary
pkill -9 -f monolit
sleep 3

# Uruchom NOWY
cd /workspace/mrd69
nohup python3 monolit.py --port 8080 > monolit.log 2>&1 &

sleep 7

# Sprawdź logi
tail -50 monolit.log | grep -E "(ERROR|OK|WARN|Started)"
```

---

### **KROK 6: TEST!**

```bash
# Test autonauka
curl -X POST http://localhost:8080/api/auto/learn \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ssjjMijaja6969" \
  -d '{"q":"Python"}' | head -50

# Test newsy
curl "http://localhost:8080/api/news?q=technologia&limit=5" \
  -H "Authorization: Bearer ssjjMijaja6969" | head -50

# Test Juventus!
curl http://localhost:8080/api/sport/juventus \
  -H "Authorization: Bearer ssjjMijaja6969"
```

---

## 🎯 **W PRZEGLĄDARCE:**

```
https://h36o457520pycq-8080.proxy.runpod.net/
```

**Masz:**
- ✅ Zegarek + data
- ✅ Mikrofon (mów po polsku!)
- ✅ Checkbox "Web" = autonauka z SERPAPI/Firecrawl!
- ✅ Upload plików
- ✅ Sidebar (☰)

**I W TLE co godzinę sprawdza Juventus!** ⚽

---

## ⚡ QUICK START (jeśli nie chcesz edytować monolit.py):

Możesz uruchomić jako osobny serwis:

```bash
# Port 9000 dla autonauka+sport
nohup python3 -c "
from autonauka_full import *
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.post('/api/auto/learn')
async def learn(request):
    body = await request.json()
    return autonauka(body.get('q',''), topk=10)

@app.get('/api/sport/juventus')  
async def juve():
    return track_juventus_sync()

@app.get('/api/news')
async def news(q: str = 'świat'):
    return get_news_sync(q, 10)

uvicorn.run(app, host='0.0.0.0', port=9000)
" > auto_server.log 2>&1 &
```

**Potem frontend łączy się z portem 9000 dla web search!**

---

## 📊 **TESTY KOŃCOWE:**

```bash
# Autonauka z wszystkimi źródłami
curl -X POST http://localhost:8080/api/auto/learn \
  -H "Authorization: Bearer ssjjMijaja6969" \
  -H "Content-Type: application/json" \
  -d '{"q":"Sztuczna inteligencja 2024","deep_research":true}' \
  | python3 -m json.tool | head -100

# JUVENTUS!
curl http://localhost:8080/api/sport/juventus \
  -H "Authorization: Bearer ssjjMijaja6969" \
  | python3 -m json.tool

# Newsy tech
curl "http://localhost:8080/api/news?q=technologia+AI&limit=5" \
  -H "Authorization: Bearer ssjjMijaja6969" \
  | python3 -m json.tool
```

---

## 🔥 BONUS - Juventus notyfikacje:

W monolit.py jest już background thread który:
- ✅ Sprawdza Juventus CO GODZINĘ
- ✅ Zapisuje do LTM automatycznie
- ✅ Dodaje jako psyche event

**Możesz sprawdzić historię:**
```bash
curl "http://localhost:8080/api/ltm/search?q=juventus&limit=20" \
  -H "Authorization: Bearer ssjjMijaja6969"
```

**BĘDĄ WSZYSTKIE AUTO-UPDATY O MECZACH!** ⚽🔥

---

**Gotowy? WKLEJAJ KOMENDY!** 💪
