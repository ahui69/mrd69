# 🚀 DEPLOY - SZYBKA INSTRUKCJA

## 📦 **CO JEST W TYM FOLDERZE:**

To jest **CZYSTY KOD** gotowy do wrzucenia na GitHub i RunPod!

```
github_ready/
├── server.py          # Główny serwer
├── requirements.txt   # Zależności
├── .env.example       # Przykładowa konfiguracja
├── routers/           # API endpoints
├── src/               # Logika
├── frontend/          # UI
├── deploy.sh          # Skrypt do deploy na RunPod
└── data/, logs/       # Puste katalogi (na dane)
```

**BRAK śmieci:** logs, cache, test files, docs

---

## 🎯 **JAK WRZUCIĆ NA RUNPOD - 3 KROKI:**

### **KROK 1: Edytuj deploy.sh (jeśli potrzeba)**

Otwórz `deploy.sh` i sprawdź linię 13:
```bash
SSH_KEY="$HOME/.ssh/runpod_ed25519"
```

Jeśli klucz jest gdzie indziej, zmień ścieżkę.

---

### **KROK 2: Uruchom deploy**

```bash
cd github_ready
./deploy.sh
```

Skrypt:
- Połączy się z RunPod
- Skopiuje kod
- Zainstaluje zależności
- Uruchomi serwer

**CZAS: 2-5 minut**

---

### **KROK 3: Sprawdź czy działa**

1. Idź na: https://www.runpod.io/console/pods
2. Kliknij swój Pod
3. Skopiuj URL (Connect → HTTP Services → port 8000)
4. Test:
```bash
curl https://TWOJ_URL/api/health
```

**Jak zobaczysz `{"ok": true}` = DZIAŁA ONLINE!** 🎉

---

## 🔧 **TROUBLESHOOTING:**

### Problem: "Connection refused"
```bash
# Sprawdź czy Pod działa w RunPod Console
# Jeśli "Stopped" - kliknij "Start"
```

### Problem: "Permission denied (publickey)"
```bash
# Zła ścieżka do klucza SSH
# Edytuj deploy.sh, linia 13
```

### Problem: Serwer nie startuje
```bash
# Połącz się SSH i zobacz logi:
ssh root@213.192.2.99 -p 41724 -i ~/.ssh/runpod_ed25519
cd /workspace/mrd69
tail -50 server.log
```

---

## 📋 **RĘCZNE WGRANIE (przez Git):**

Jeśli wolisz bez skryptu:

```bash
# 1. Push kod na GitHub (z github_ready/)
cd github_ready
git init
git add .
git commit -m "clean code for RunPod"
git push

# 2. SSH do RunPod
ssh root@213.192.2.99 -p 41724 -i ~/.ssh/runpod_ed25519

# 3. Na podzie:
cd /workspace
git clone https://github.com/ahui69/mrd69.git
cd mrd69
pip install -r requirements.txt
nohup python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
```

---

## ✅ **GOTOWE!**

Uruchom:
```bash
cd /workspace/github_ready
./deploy.sh
```

**I TYLE!** 🚀
