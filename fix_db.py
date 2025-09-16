#!/usr/bin/env python
"""
Script to fix the database schema for memory.db
- Updates the ltm table to add missing columns
- Migrates data if needed
"""
import json
import shutil
import sqlite3
from pathlib import Path

# Ścieżka do bazy danych
DB_PATH = Path("data/memory.db")
BACKUP_PATH = Path("data/memory.db.bak")


def fix_ltm_table():
    """Naprawia tabelę ltm, dodając brakujące kolumny i migrując dane."""
    print(f"Naprawa bazy danych: {DB_PATH}")

    # Tworzenie kopii zapasowej
    if DB_PATH.exists():
        print(f"Tworzenie kopii zapasowej: {BACKUP_PATH}")
        shutil.copy2(DB_PATH, BACKUP_PATH)

    # Połączenie z bazą danych
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Sprawdzenie struktury tabeli ltm
    cursor.execute("PRAGMA table_info(ltm)")
    columns = {col[1]: col for col in cursor.fetchall()}
    print(f"Znaleziono kolumny: {', '.join(columns.keys())}")

    # Sprawdzenie czy potrzebna jest migracja
    if "text_enc" in columns and "conf" in columns and "tags" in columns:
        print("Potrzebna migracja ze starego formatu do nowego.")

        # Tworzenie tymczasowej tabeli
        print("Tworzenie tymczasowej tabeli...")
        cursor.execute(
            """
        CREATE TABLE ltm_new (
            id TEXT PRIMARY KEY,
            ts REAL NOT NULL,
            kind TEXT NOT NULL,
            text TEXT NOT NULL,
            meta TEXT,
            score REAL
        )
        """
        )

        # Migracja danych
        print("Migracja danych...")
        cursor.execute("SELECT id, ts, text_enc, conf, tags FROM ltm")
        rows = cursor.fetchall()

        for row in rows:
            # Konwersja danych
            id = row["id"]
            ts = row["ts"]

            # Dekodowanie tekstu jeśli jest zaszyfrowany
            text = row["text_enc"]
            try:
                # Jeśli text_enc jest w formacie JSON, próbujemy go zdekodować
                if text.startswith("{") or text.startswith("["):
                    decoded = json.loads(text)
                    if isinstance(decoded, dict) and "enc" in decoded:
                        # Używamy dekodowanego pola enc
                        text = decoded.get("enc", text)
                    else:
                        text = str(decoded)
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                # W przypadku błędu, używamy oryginalnego tekstu (log tylko informacyjny)
                print("[fix_db] Uwaga: błąd dekodowania text_enc jako JSON:", e)

            # Migracja conf -> score
            score = row["conf"]

            # Migracja tagów do meta
            meta_data = {}
            try:
                tags = json.loads(row["tags"]) if row["tags"] else []
                if tags:
                    meta_data["tags"] = tags
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                print("[fix_db] Uwaga: błąd parsowania tags jako JSON:", e)

            meta_json = json.dumps(meta_data) if meta_data else None

            # Wstawienie danych do nowej tabeli
            cursor.execute(
                (
                    "INSERT INTO ltm_new (id, ts, kind, text, meta, score) "
                    "VALUES (?, ?, ?, ?, ?, ?)"
                ),
                (id, ts, "fact", text, meta_json, score),
            )

        # Zamiana tabel
        print("Zamiana tabel...")
        cursor.execute("DROP TABLE ltm")
        cursor.execute("ALTER TABLE ltm_new RENAME TO ltm")

        # Tworzenie indeksów
        print("Tworzenie indeksów...")
        cursor.execute("CREATE INDEX idx_ltm_ts ON ltm(ts)")
        cursor.execute("CREATE INDEX idx_ltm_kind ON ltm(kind)")
        cursor.execute("CREATE INDEX idx_ltm_score ON ltm(score)")

    elif not all(col in columns for col in ["text", "meta", "score"]):
        print("Brakuje wymaganych kolumn. Tworzenie nowej tabeli...")

        # Tworzenie nowej tabeli jeśli brakuje kolumn
        cursor.execute("DROP TABLE IF EXISTS ltm")
        cursor.execute(
            """
        CREATE TABLE ltm (
            id TEXT PRIMARY KEY,
            ts REAL NOT NULL,
            kind TEXT NOT NULL,
            text TEXT NOT NULL,
            meta TEXT,
            score REAL
        )
        """
        )

        # Tworzenie indeksów
        cursor.execute("CREATE INDEX idx_ltm_ts ON ltm(ts)")
        cursor.execute("CREATE INDEX idx_ltm_kind ON ltm(kind)")
        cursor.execute("CREATE INDEX idx_ltm_score ON ltm(score)")
    else:
        print("Struktura tabeli ltm jest już poprawna.")

    # Aktualizacja indeksu FTS jeśli istnieje
    try:
        print("Aktualizacja indeksu FTS...")
        cursor.execute("DROP TABLE IF EXISTS ltm_fts")
        cursor.execute(
            """
        CREATE VIRTUAL TABLE ltm_fts USING fts5(
            id, text, tokenize='unicode61'
        )
        """
        )

        # Wypełnianie indeksu FTS
        print("Wypełnianie indeksu FTS...")
        cursor.execute("INSERT INTO ltm_fts(id, text) SELECT id, text FROM ltm")

    except sqlite3.OperationalError as e:
        print(f"Błąd operacyjny SQLite podczas aktualizacji indeksu FTS: {e}")
    except Exception as e:
        print(f"Nieoczekiwany błąd podczas aktualizacji indeksu FTS: {e}")

    # Zatwierdzenie zmian
    conn.commit()
    conn.close()

    print("Naprawa bazy danych zakończona pomyślnie.")


if __name__ == "__main__":
    fix_ltm_table()
