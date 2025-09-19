import sqlite3, os

DB = r"data\memory.db"

if not os.path.exists(DB):
    raise SystemExit(f"Brak pliku bazy: {DB}")

con = sqlite3.connect(DB)
cur = con.cursor()

tables = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("tabele:", tables)

if any(t[0] == "facts" for t in tables):
    cnt = cur.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    print("facts:", cnt)
    sample = cur.execute("SELECT topic, fact, source FROM facts LIMIT 3").fetchall()
    print("próbka:", sample)
else:
    print("Brak tabeli 'facts' (nie zasiane albo inna nazwa).")

con.close()
