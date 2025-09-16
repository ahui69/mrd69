# 🧠 MORDZIX - ZAAWANSOWANY SYSTEM AI Z PAMIĘCIĄ EPISODYCZNĄ

## 📖 OPIS

Mordzix to zaawansowany system AI z pełną pamięcią episodyczną, kontekstową i emocjonalną. System posiada hybrydowy RAG z zaawansowanymi funkcjami pamięci i integracją z RunPod.

### 🎯 KLUCZOWE FUNKCJE:

- **📅 Pamięć episodyczna** - Timeline z automatycznymi podsumowaniami dnia
- **🧠 Pamięć kontekstowa** - 7 różnych "mózgów" (coding, creative, business, etc.)
- **🤔 Samorefleksja AI** - Automatyczne notatki coacha i reguły
- **📁 Pamięć sensoryczna** - Storage plików z metadanymi
- **😊 Pamięć emocjonalna** - Wykrywanie nastroju i adaptacja odpowiedzi
- **🔮 Pamięć predykcyjna** - Przewidywanie następnych akcji
- **💾 System wersjonowania** - Git-like backupy pamięci
- **👥 Mapping relacji** - Graf osób i ich relacji
- **⚡ Hybrydowy RAG** - Embeddings + TF-IDF + BM25/FTS5 + świeżość
- **🔒 RunPod Integration** - Persystencja danych w chmurze

## 🚀 QUICK START

### Lokalne uruchomienie:
```bash
git clone https://github.com/YOUR_USERNAME/mordzix-advanced.git
cd mordzix-advanced
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
python main.py
```

### RunPod deployment:
Sprawdź [RUNPOD_DEPLOYMENT_GUIDE.md](RUNPOD_DEPLOYMENT_GUIDE.md) dla pełnych instrukcji.

## 🏗️ ARCHITEKTURA

- **main.py** - FastAPI server z endpointami dla advanced memory
- **memory.py** - Zaawansowany system pamięci (3000+ linii)
- **mordzix_core.py** - Główny silnik AI z integracją pamięci
- **config.py** - Konfiguracja systemu

## 📊 API ENDPOINTS

### Chat & WebSocket:
- `/ws` - główny endpoint do komunikacji czatowej
- `/chat` - REST endpoint dla czatu

### Memory Timeline:
- `GET /memory/timeline` - Pobierz timeline interakcji
- `GET /memory/timeline/search` - Wyszukaj w timeline
- `POST /memory/timeline/daily-summary` - Stwórz podsumowanie dnia

### Context Management:
- `GET /memory/context/{type}` - Pobierz pamięć kontekstową
- `POST /memory/context/switch` - Przełącz kontekst

### Mood & Predictions:
- `GET /memory/mood` - Wykryj nastrój z tekstu
- `GET /memory/predictions` - Przewiduj następne akcje

### Memory Versioning:
- `GET /memory/versions` - Lista wersji pamięci
- `POST /memory/backup` - Stwórz backup
- `POST /memory/restore/{id}` - Przywróć wersję

### Relationships:
- `GET /memory/relationships` - Graf relacji
- `GET /memory/person/{name}` - Profil osoby

### Legacy API:
- `/health` - sprawdzenie stanu serwera
- `/episodes` - pobiera historię rozmów
- `/reset` - czyści pamięć krótkotrwałą
- `/search` - wyszukiwanie wiedzy
- `/runpod/status` - status integracji RunPod

## 🔧 KONFIGURACJA

### Environment Variables:
```bash
export OPENAI_API_KEY="your-api-key"
export APP_TITLE="Mordzix Advanced"

# RunPod Integration
export USE_RUNPOD=true
export RUNPOD_PERSIST_DIR="/workspace/data"
export RUNPOD_API_KEY="your-runpod-key"
export RUNPOD_ENDPOINT_ID="your-endpoint-id"

# Memory Settings
export MEM_NS="default"
export PSY_ENCRYPT_KEY="your-encryption-key"
export LTM_MIN_CONF=0.25
export MAX_LTM_FACTS=2000000
```

### Database:
System używa SQLite z zaawansowaną pamięcią:
- **Główna baza**: `data/memory.db` - RAG i podstawowa pamięć
- **Advanced Memory**: 7 nowych tabel dla episodycznej pamięci
- **RunPod Sync**: Automatyczna synchronizacja co 15 minut

## 🧪 TESTOWANIE

```bash
# Uruchom testy
python -m pytest tests/

# Test systemu pamięci
curl "http://localhost:5959/memory/timeline"
curl "http://localhost:5959/memory/mood?text=test"
curl "http://localhost:5959/health"

# Test chat integration
curl -X POST "http://localhost:5959/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Cześć!", "mode": "mordzix"}'
```

## 📈 MONITORING

System loguje do:
- `backend.log` - Główne logi systemu
- `data/advanced_memory.log` - Logi pamięci
- `server.log` - Logi serwera

### RunPod Management:
```bash
# Ręczna synchronizacja
python runpod_sync.py force

# Status RunPod
python runpod_sync.py status

# Restore z RunPod
python runpod_sync.py restore
```

## 🚀 DEPLOYMENT

### Lokalne:
```bash
uvicorn main:app --host 0.0.0.0 --port 5959 --reload
```

### Production (RunPod):
```bash
# Patrz RUNPOD_DEPLOYMENT_GUIDE.md
uvicorn main:app --host 0.0.0.0 --port 5959
```

### Docker:
```bash
docker build -t mordzix-advanced .
docker run -d --gpus all -p 5959:5959 mordzix-advanced
```

## 📄 LICENCJA

MIT License - sprawdź plik LICENSE

## 🆘 WSPARCIE

W przypadku problemów:
1. Sprawdź [RUNPOD_DEPLOYMENT_GUIDE.md](RUNPOD_DEPLOYMENT_GUIDE.md)
2. Zgłoś issue na GitHub
3. Sprawdź logi w folderze `data/`

---

**🧠 Mordzix - AI z prawdziwą pamięcią episodyczną!**