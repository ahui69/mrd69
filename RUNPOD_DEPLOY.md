# 🚀 DEPLOY NA RUNPOD - INSTRUKCJA

## 💰 TWOJA SYTUACJA:
- ✅ Masz RunPod account
- ✅ Model leci z DeepInfra (nie potrzebujesz GPU!)
- ✅ RunPod CPU wystarczy = **TANIEJ**

---

## 📋 KROK PO KROKU (15 minut):

### **KROK 1: Przygotuj kod (ZROBIONE!)**

✅ Masz już:
- `Dockerfile.runpod` - gotowy
- `requirements.txt` - z wersjami
- `.env` - z kluczami
- Kod działa lokalnie

---

### **KROK 2: Wejdź na RunPod**

1. Idź na: https://www.runpod.io/console
2. Zaloguj się
3. Kliknij "Templates" (lewy menu)

---

### **KROK 3: Stwórz Custom Template**

```
Name: mrd69-api
Container Image: python:3.13-slim

Container Start Command:
bash -c "pip install -r requirements.txt && uvicorn server:app --host 0.0.0.0 --port 8000"

Container Disk: 10 GB (wystarczy)
Expose HTTP Ports: 8000
Expose TCP Ports: -
```

**Environment Variables (dodaj w RunPod):**
```
LLM_BASE_URL=https://api.deepinfra.com/v1/openai
LLM_API_KEY=twoj_klucz_deepinfra
LLM_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct
USE_RUNPOD=1
RUNPOD_PERSIST_DIR=/runpod/persist
```

---

### **KROK 4: Deploy z GitHub (NAJŁATWIEJ)**

**Opcja A - przez GitHub (polecam!):**

1. Wrzuć kod na GitHub:
```bash
cd /workspace
git init
git add .
git commit -m "initial commit"
gh repo create mrd69-api --private --source=. --remote=origin --push
# LUB ręcznie: stwórz repo na github.com i:
git remote add origin https://github.com/TWOJA_NAZWA/mrd69-api.git
git push -u origin master
```

2. W RunPod Template:
```
Container Image: wybierz "From GitHub"
GitHub Repo: TWOJA_NAZWA/mrd69-api
Branch: master
Dockerfile Path: Dockerfile.runpod
```

**Opcja B - przez Docker Hub:**
```bash
# Zbuduj lokalnie
docker build -t twoja_nazwa/mrd69-api -f Dockerfile.runpod .

# Push do Docker Hub
docker login
docker push twoja_nazwa/mrd69-api

# W RunPod:
# Container Image: twoja_nazwa/mrd69-api
```

---

### **KROK 5: Uruchom Pod**

1. Kliknij "Pods" → "Deploy"
2. Wybierz swój template
3. **WAŻNE:** Wybierz **CPU** (nie GPU!)
   - Recommended: 2 vCPU, 4GB RAM
4. **Attach Volume:** TAK (10GB) - tu będzie baza danych
5. Kliknij "Deploy"

---

### **KROK 6: Sprawdź czy działa**

Po ~2 minutach:

1. Kliknij na swój Pod
2. Zobacz "Connect" → skopiuj URL (będzie coś jak: `https://abc123-8000.proxy.runpod.net`)
3. Test:
```bash
curl https://abc123-8000.proxy.runpod.net/api/health
```

**Powinno zwrócić:**
```json
{"ok": true, "mode": "llm", "memory_router": true}
```

**DZIAŁA! 🎉**

---

## 💰 KOSZTY (konkretnie):

### **RunPod CPU Pricing:**
```
SECURE CLOUD (zawsze dostępne):
- 2 vCPU, 4GB RAM: ~$0.10/h = $72/mies (24/7)
- 4 vCPU, 8GB RAM: ~$0.20/h = $144/mies (24/7)

COMMUNITY CLOUD (taniej, ale może być niedostępne):
- 2 vCPU, 4GB RAM: ~$0.04/h = $30/mies (24/7)

TWÓJ BUDŻET: $199
= 2 miesiące Secure Cloud
= 6 miesięcy Community Cloud (jeśli wyłączasz nocami)
```

### **Optymalizacja kosztów:**
```bash
# Włącz tylko gdy używasz:
9:00-22:00 = 13h/dzień
13h × $0.10 = $1.30/dzień = $40/miesiąc

$199 = ~5 miesięcy działania!
```

---

## 🔧 KONFIGURACJA .env NA RUNPOD:

W RunPod Console → Twój Pod → Environment Variables:

```bash
# LLM (DeepInfra)
LLM_BASE_URL=https://api.deepinfra.com/v1/openai
LLM_API_KEY=<twój_klucz_deepinfra>
LLM_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct
LLM_TIMEOUT=30

# RunPod specifics
USE_RUNPOD=1
RUNPOD_PERSIST_DIR=/runpod/persist
MEM_ROOT=/workspace

# Opcjonalnie (jeśli masz):
SERPAPI_KEY=<twój_klucz>
FIRECRAWL_KEY=<twój_klucz>
GOOGLE_MAPS_KEY=<twój_klucz>
```

---

## 📡 DOSTĘP DO TWOJEJ APPKI:

Po deployu dostaniesz URL typu:
```
https://abc123xyz-8000.proxy.runpod.net
```

**To jest TWÓJ publiczny adres!**

Możesz:
```
✅ Dać link znajomym
✅ Używać w innych aplikacjach
✅ Połączyć z frontendem
✅ Dodać własną domenę
```

---

## 🛠️ ZARZĄDZANIE PODEM:

### **Start/Stop (oszczędzaj kasę!):**
```
W RunPod Console:
- "Stop Pod" - zatrzymaj (nie płacisz!)
- "Start Pod" - uruchom (płacisz od teraz)
```

### **Logs (debugging):**
```
W RunPod Console → Twój Pod → Logs
Albo przez API
```

### **SSH do poda:**
```
W RunPod Console → Twój Pod → Connect → "Start Web Terminal"
Albo przez SSH (dają Ci klucz)
```

---

## 📊 MONITORING:

### **W RunPod Dashboard zobaczysz:**
```
- CPU usage (powinno być <50%)
- RAM usage (powinno być <2GB)
- Network (ile requestów)
- Koszty (ile wydałeś)
```

---

## 🔥 ALTERNATYWA: SERVERLESS (jeszcze taniej!)

RunPod ma też **Serverless**:
```
Płacisz TYLKO gdy używasz:
- $0.00010/s active time

Przykład:
100 requestów/dzień × 2s każdy = 200s
200s × $0.0001 = $0.02/dzień = $0.60/miesiąc!

$199 = 330 miesięcy! (27 LAT!) 😱
```

Ale wymaga więcej setupu - na razie zostań przy Pods.

---

## 🎯 QUICK DEPLOY (TL;DR):

```bash
# 1. Wrzuć kod na GitHub
cd /workspace
git init
git add .
git commit -m "deploy to runpod"
# Push do GitHub (załóż repo na github.com)

# 2. W RunPod:
# - New Template
# - GitHub integration
# - Deploy CPU Pod (2 vCPU, 4GB RAM)
# - Dodaj zmienne env
# - Deploy!

# 3. Po 2 minutach:
# - Skopiuj URL
# - Test: curl https://twoj-url.runpod.net/api/health

GOTOWE!
```

---

## 💪 FINAL:

**Za $199 możesz mieć:**
- ~2 miesiące 24/7 (Secure Cloud)
- ~5 miesięcy jeśli wyłączasz nocami
- ~6 miesięcy (Community Cloud)

**Chcesz żebym pomógł w deployu?** Albo masz już działające i chcesz tylko zoptymalizować?
