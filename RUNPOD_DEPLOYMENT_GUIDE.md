# 🚀 PRZEWODNIK URUCHOMIENIA NA RUNPOD - ZAAWANSOWANY SYSTEM PAMIĘCI

## 📋 WYMAGANIA

### Hardware Requirements:
- **GPU**: RTX 4090/A100 (24GB VRAM minimum)
- **RAM**: 32GB minimum  
- **Storage**: 100GB SSD minimum
- **Template**: PyTorch 2.1.0, Python 3.10+

### Software Dependencies:
```bash
# Główne dependencies już w requirements.txt
pip install -r requirements.txt

# Dodatkowe dla zaawansowanej pamięci
pip install sentence-transformers
pip install scikit-learn>=1.3.0
pip install networkx>=3.0
```

## 🔧 KONFIGURACJA ŚRODOWISKA

### 1. Przygotowanie RunPod Container

```bash
# === KLONOWANIE REPO ===
cd /workspace
git clone YOUR_REPO_URL mordzix-advanced
cd mordzix-advanced

# === SETUP ŚRODOWISKA ===
python -m venv .venv
source .venv/bin/activate

# === INSTALACJA DEPENDENCIES ===
pip install --upgrade pip
pip install -r requirements.txt

# === DODATKOWE PAKIETY DLA ADVANCED MEMORY ===
pip install sentence-transformers scikit-learn networkx

# === SPRAWDZENIE GPU ===
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}')"
```

### 2. Konfiguracja Bazy Danych

```bash
# === INICJALIZACJA SQLITE Z ADVANCED MEMORY ===
python -c "
from memory import get_advanced_memory
advanced_mem = get_advanced_memory()
print('✅ Advanced Memory System Initialized')
print(f'Database path: {advanced_mem.db_path}')
"

# === TESTOWANIE MEMORY SYSTEM ===
python -c "
from memory import get_advanced_memory, ContextType
advanced_mem = get_advanced_memory()

# Test timeline
entry = advanced_mem.create_timeline_entry(
    'System startup test',
    'deployment',
    user_input='RunPod deployment test',
    ai_response='System initialized successfully'
)
entry_id = advanced_mem.add_timeline_entry(entry)
print(f'✅ Timeline test - Entry ID: {entry_id}')

# Test context switching
result = advanced_mem.switch_context(ContextType.CODING)
print(f'✅ Context switch test: {result}')

print('🎉 Advanced Memory System Ready!')
"
```

## 🚀 URUCHOMIENIE SERWERA

### Metoda 1: Standardowe uruchomienie
```bash
cd /workspace/mordzix-advanced
source .venv/bin/activate

# === URUCHOMIENIE Z ADVANCED MEMORY ===
python main.py

# Serwer dostępny na:
# http://localhost:5959/mordzix - główny chat
# http://localhost:5959/memory/timeline - API timeline
# http://localhost:5959/memory/context/CODING - API kontekstu
```

### Metoda 2: Uvicorn z reloadem
```bash
cd /workspace/mordzix-advanced
source .venv/bin/activate

# === DEVELOPMENT MODE ===
uvicorn main:app --reload --host 0.0.0.0 --port 5959

# Production mode (background)
nohup uvicorn main:app --host 0.0.0.0 --port 5959 > server.log 2>&1 &
```

### Metoda 3: Docker deployment
```bash
# === BUILD CONTAINER Z ADVANCED MEMORY ===
docker build -t mordzix-advanced .

# === RUN CONTAINER ===
docker run -d \
  --name mordzix-advanced \
  --gpus all \
  -p 5959:5959 \
  -v /workspace/data:/app/data \
  mordzix-advanced
```

## 🧠 TESTOWANIE ADVANCED MEMORY

### 1. Test API Endpoints

```bash
# === TIMELINE API ===
curl "http://localhost:5959/memory/timeline?limit=10"

# === CONTEXT SWITCHING ===
curl -X POST "http://localhost:5959/memory/context/switch" \
  -H "Content-Type: application/json" \
  -d '{"context_type": "CODING"}'

# === MOOD DETECTION ===
curl "http://localhost:5959/memory/mood?text=Mordo%20znowu%20się%20drze%20o%20ucinki"

# === PREDICTIONS ===
curl "http://localhost:5959/memory/predictions?user_input=chcę%20napisać%20kod"

# === RELATIONSHIPS ===
curl "http://localhost:5959/memory/relationships"

# === PERSON PROFILE ===
curl "http://localhost:5959/memory/person/Mordo"
```

### 2. Test Chat Integration

```bash
# === TEST THROUGH CHAT ===
curl -X POST "http://localhost:5959/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Cześć, jestem w trybie coding i potrzebuję pomocy z Python",
    "mode": "mordzix"
  }'

# System automatycznie:
# 1. Wykryje kontekst CODING
# 2. Zastosuje mood detection
# 3. Dostosuje odpowiedź do nastroju
# 4. Zapisze w timeline
# 5. Zaktualizuje prediction patterns
```

## 🔍 MONITORING I DIAGNOSTYKA

### 1. Health Check
```bash
# === STATUS SYSTEMU ===
curl "http://localhost:5959/health"

# === MEMORY STATISTICS ===
curl "http://localhost:5959/memory/versions"

# === REFLECTIONS ===
curl "http://localhost:5959/memory/reflections?limit=5"
```

### 2. Log Analysis
```bash
# === MEMORY LOGS ===
tail -f data/advanced_memory.log

# === SERVER LOGS ===
tail -f server.log

# === RELIABILITY LOGS ===
tail -f backend.log
```

### 3. Database Inspection
```bash
# === SQLITE BROWSER ===
python -c "
import sqlite3
conn = sqlite3.connect('data/memory.db')
cursor = conn.cursor()

# Check tables
cursor.execute(\"SELECT name FROM sqlite_master WHERE type='table'\")
tables = cursor.fetchall()
print('📊 Database Tables:')
for table in tables:
    print(f'  - {table[0]}')

# Timeline entries count
cursor.execute('SELECT COUNT(*) FROM timeline_entries')
count = cursor.fetchone()[0]
print(f'📝 Timeline Entries: {count}')

# Person profiles
cursor.execute('SELECT name, role FROM person_profiles')
people = cursor.fetchall()
print('👥 People in Memory:')
for person in people:
    print(f'  - {person[0]}: {person[1]}')

conn.close()
"
```

## 🛠️ KONFIGURACJA ADVANCED FEATURES

### 1. Context Types Customization
```python
# W pliku memory.py można dodać nowe konteksty:
class ContextType(Enum):
    CODING = "coding"
    CREATIVE_WRITING = "creative_writing"
    BUSINESS = "business"
    RESEARCH = "research"
    DEBUGGING = "debugging"
    ARCHITECTURE = "architecture"  # NOWY
    DEVOPS = "devops"             # NOWY
```

### 2. Mood Types Extension
```python
# Rozszerzenie typów nastrojów:
class MoodType(Enum):
    FRIENDLY = "friendly"
    TECHNICAL = "technical"
    FRUSTRATED = "frustrated"
    URGENT = "urgent"
    CURIOUS = "curious"
    AGGRESSIVE = "aggressive"     # NOWY
    CONFUSED = "confused"         # NOWY
```

### 3. Memory Versioning Schedule
```bash
# === AUTOMATIC BACKUPS ===
# Dodaj do crontab:
0 */6 * * * curl -X POST "http://localhost:5959/memory/backup" -d '{"description": "Scheduled backup"}'

# Daily summary generation:
0 0 * * * curl -X POST "http://localhost:5959/memory/timeline/daily-summary"
```

## 🔒 BEZPIECZEŃSTWO I BACKUP

### 1. Memory Backup Strategy
```bash
# === MANUAL BACKUP ===
curl -X POST "http://localhost:5959/memory/backup" \
  -H "Content-Type: application/json" \
  -d '{"description": "Pre-deployment backup"}'

# === RESTORE SPECIFIC VERSION ===
# Najpierw sprawdź dostępne wersje:
curl "http://localhost:5959/memory/versions"

# Przywróć konkretną wersję:
curl -X POST "http://localhost:5959/memory/restore/VERSION_ID"
```

### 2. Data Protection
```bash
# === COPY CRITICAL DATA ===
cp data/memory.db data/memory_backup_$(date +%Y%m%d_%H%M%S).db
cp -r data/ /workspace/backup/data_$(date +%Y%m%d)/

# === SYNC TO EXTERNAL STORAGE ===
# Skonfiguruj w runpod_sync.py dla automatic backup
```

## 📊 PERFORMANCE OPTIMIZATION

### 1. Memory Settings
```python
# W config.py:
ADVANCED_MEMORY_CONFIG = {
    "max_timeline_entries": 10000,
    "max_context_facts": 500,
    "embedding_batch_size": 32,
    "prediction_lookback_days": 30,
    "reflection_frequency": 24,  # hours
}
```

### 2. Database Optimization
```sql
-- Wykonaj w SQLite dla lepszej wydajności:
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = 10000;
PRAGMA temp_store = memory;

-- Indeksy dla szybszego wyszukiwania:
CREATE INDEX IF NOT EXISTS idx_timeline_timestamp ON timeline_entries(timestamp);
CREATE INDEX IF NOT EXISTS idx_timeline_context ON timeline_entries(context_type);
CREATE INDEX IF NOT EXISTS idx_person_name ON person_profiles(name);
```

## 🎯 PRODUKTYWNE UŻYTKOWANIE

### 1. Typical Workflow
```bash
# 1. Uruchom serwer
uvicorn main:app --host 0.0.0.0 --port 5959

# 2. Przełącz kontekst na coding
curl -X POST "http://localhost:5959/memory/context/switch" \
  -d '{"context_type": "CODING"}'

# 3. Rozpocznij pracę - system automatycznie:
#    - Wykrywa nastrój użytkownika
#    - Dostosowuje szczegółowość odpowiedzi
#    - Zapisuje interakcje w timeline
#    - Uczy się patterns dla predictions
#    - Buduje profile osób (Mordo, Papik, etc.)
```

### 2. Advanced Features Usage
```javascript
// Frontend JavaScript do integracji:
async function switchContext(contextType) {
    const response = await fetch('/memory/context/switch', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({context_type: contextType})
    });
    return await response.json();
}

async function getTimeline(limit = 20) {
    const response = await fetch(`/memory/timeline?limit=${limit}`);
    return await response.json();
}

async function getMoodDetection(text) {
    const response = await fetch(`/memory/mood?text=${encodeURIComponent(text)}`);
    return await response.json();
}
```

## 🚨 TROUBLESHOOTING

### Common Issues:

1. **Memory nie zapisuje timeline:**
   ```bash
   # Sprawdź permissions
   chmod 777 data/
   # Sprawdź space
   df -h
   ```

2. **Context switching nie działa:**
   ```python
   # Test manual:
   from memory import ContextType
   print(list(ContextType))
   ```

3. **Mood detection nieprecyzyjny:**
   ```python
   # Upgrade sentence-transformers:
   pip install --upgrade sentence-transformers
   ```

4. **Predictions nie generują się:**
   ```bash
   # Sprawdź czy jest wystarczająco danych:
   curl "http://localhost:5959/memory/timeline?limit=100"
   ```

## ✅ DEPLOYMENT CHECKLIST

- [ ] RunPod container z GPU access
- [ ] Python 3.10+ installed
- [ ] All requirements.txt packages installed
- [ ] Advanced memory dependencies installed
- [ ] Database initialized with all tables
- [ ] Server runs on 0.0.0.0:5959
- [ ] All API endpoints respond correctly
- [ ] Timeline creation works
- [ ] Context switching works
- [ ] Mood detection active
- [ ] Relationship graph builds
- [ ] Memory versioning operational
- [ ] Backup strategy configured
- [ ] Monitoring scripts ready

## 🎉 SUCCESS CONFIRMATION

Po udanym deployment powinieneś móc:

1. ✅ Otworzyć http://YOUR_RUNPOD_IP:5959/mordzix
2. ✅ Prowadzić rozmowy z automatycznym zapisem timeline
3. ✅ Przełączać konteksty (CODING/CREATIVE_WRITING/etc.)
4. ✅ Widzieć adaptację do nastroju użytkownika
5. ✅ Otrzymywać predictions następnych akcji
6. ✅ Przeglądać grafy relacji interpersonalnych
7. ✅ Tworzyć i przywracać backupy pamięci

**🚀 ZAAWANSOWANY SYSTEM PAMIĘCI DZIAŁA!**

---

*Ten przewodnik pokrywa pełną implementację zaawansowanego systemu pamięci na RunPod. W przypadku problemów sprawdź logi, API endpoints i database integrity.*