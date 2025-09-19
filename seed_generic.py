import json, sqlite3, sys, os, time

jsonl = sys.argv[1] if len(sys.argv) > 1 else r"data\sq3\seed.jsonl"
db    = sys.argv[2] if len(sys.argv) > 2 else r"data\memory.db"

os.makedirs(os.path.dirname(db), exist_ok=True)
con = sqlite3.connect(db)
cur = con.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS facts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  topic TEXT,
  fact  TEXT NOT NULL,
  source TEXT,
  created_at REAL DEFAULT (strftime('%s','now'))
)
""")

cnt = 0
with open(jsonl, "r", encoding="utf-8") as f:
    for line in f:
        line=line.strip()
        if not line: continue
        obj = json.loads(line)
        topic = obj.get("topic") or obj.get("category")
        fact  = obj.get("fact") or obj.get("text") or obj.get("content")
        source= obj.get("source") or obj.get("url")
        if not fact: 
            continue
        cur.execute(
            "INSERT INTO facts(topic,fact,source,created_at) VALUES(?,?,?,?)",
            (topic, fact, source, time.time())
        )
        cnt += 1

con.commit()
con.close()
print(f"[seed_generic] OK: wczytano {cnt} rekordów do {db}")
