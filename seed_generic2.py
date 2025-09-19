import json, sqlite3, sys, os, time

src = sys.argv[1] if len(sys.argv) > 1 else r"data\sq3\seed.jsonl"
db  = sys.argv[2] if len(sys.argv) > 2 else r"data\memory.db"

def iter_records(path):
    # Czytamy cały plik, wykrywamy format
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    # usuń BOM jeśli jest
    if raw.startswith("\ufeff"):
        raw = raw.lstrip("\ufeff")
    # znajdź pierwszy nie-biały znak
    head = next((ch for ch in raw if not ch.isspace()), "")
    if head == "[":  # zwykły JSON array
        try:
            arr = json.loads(raw)
            for obj in arr:
                if isinstance(obj, dict):
                    yield obj
        except json.JSONDecodeError as e:
            raise SystemExit(f"[seed_generic2] Zły JSON (array): {e}")
    else:  # JSONL
        for ln, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line or line in ("[", "]", ","):
                continue
            if line.startswith("//") or line.startswith("#"):
                continue
            if line.endswith(","):  # JSONL z przecinkami
                line = line[:-1].rstrip()
            if line.startswith("\ufeff"):
                line = line.lstrip("\ufeff")
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except json.JSONDecodeError as e:
                raise SystemExit(f"[seed_generic2] Błąd w linii {ln}: {e}\nLinia: {line[:200]}")

os.makedirs(os.path.dirname(db), exist_ok=True)
con = sqlite3.connect(db)
cur = con.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS facts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  topic   TEXT,
  fact    TEXT NOT NULL,
  source  TEXT,
  created_at REAL DEFAULT (strftime('%s','now'))
)
""")

cnt = 0
for obj in iter_records(src):
    topic  = obj.get("topic") or obj.get("category")
    fact   = obj.get("fact")  or obj.get("text") or obj.get("content")
    source = obj.get("source") or obj.get("url")
    if not fact:
        continue
    cur.execute(
        "INSERT INTO facts(topic,fact,source,created_at) VALUES(?,?,?,?)",
        (topic, fact, source, time.time())
    )
    cnt += 1

con.commit(); con.close()
print(f"[seed_generic2] OK: wczytano {cnt} rekordów do {db}")
