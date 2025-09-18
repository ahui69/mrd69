"""
load_seed_sq3.py — Jednorazowy importer wiedzy z pliku seed.jsonl do pamięci LTM (SQLite)

Uruchom ten plik raz po przygotowaniu pliku seed.jsonl w data/sq3/.
"""
import json
from pathlib import Path
from memory import Memory

SEED_PATH = Path("data/sq3/seed.jsonl")


def load_seed_to_memory():
    mem = Memory()
    if not SEED_PATH.exists():
        print(f"Brak pliku: {SEED_PATH}")
        return
    count = 0
    with SEED_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                data = json.loads(line)
                text = data.get("text") or data.get("fact")
                meta = data.get("meta", {})
                tags = data.get("tags", ["seed", "sq3"])
                score = float(data.get("score", 0.95))
                if text:
                    mem.add_fact(text, meta_data={"tags": tags, **meta}, score=score)
                    count += 1
            except Exception as e:
                print(f"Błąd w linii: {line}\n{e}")
    print(f"Zaimportowano {count} rekordów z {SEED_PATH}")

if __name__ == "__main__":
    load_seed_to_memory()
