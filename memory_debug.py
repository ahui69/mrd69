#!/usr/bin/env python
"""
Script to debug the database access in memory.py
"""
import sqlite3
import sys
import traceback

# Załaduj moduł memory
try:
    sys.path.insert(0, ".")
    from memory import Memory

    print("Moduł memory załadowany poprawnie.")
except Exception as e:
    print(f"Błąd podczas ładowania modułu memory: {e}")
    traceback.print_exc()
    sys.exit(1)


def check_db_connection():
    """Sprawdza połączenie z bazą danych i wyświetla jej schemat."""
    db_path = "data/memory.db"
    print(f"Sprawdzam bazę danych: {db_path}")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Wyświetl tabele
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print("Tabele w bazie danych:")
        for table in tables:
            print(f" - {table[0]}")
            # Pokaż schemat tabeli
            cursor.execute(f"PRAGMA table_info({table[0]})")
            columns = cursor.fetchall()
            for col in columns:
                print(f"    {col[1]}: {col[2]}")

    except Exception as e:
        print(f"Błąd podczas sprawdzania bazy danych: {e}")
        traceback.print_exc()


def test_memory_operations():
    """Testuje podstawowe operacje Memory."""
    try:
        # Inicjalizacja pamięci
        mem = Memory()
        print("Memory zainicjalizowany pomyślnie.")

        # Test dodania faktu
        try:
            fact_id = mem.add_fact(
                "To jest testowy fakt dodany przez debugger.",
                meta_data={"tags": ["test", "debug"]},
                score=0.8,
            )
            print(f"Fakt dodany pomyślnie, ID: {fact_id}")
        except Exception as e:
            print(f"Błąd podczas dodawania faktu: {e}")
            traceback.print_exc()

        # Test ładowania wiedzy
        try:
            result = mem.load_knowledge("To jest testowa wiedza dodana przez debugger.")
            print(f"Ładowanie wiedzy: {result}")
        except Exception as e:
            print(f"Błąd podczas ładowania wiedzy: {e}")
            traceback.print_exc()

        # Znajdź miejsce błędu z conf
        try:
            print("\nWyszukiwanie miejsc użycia kolumn 'conf' i 'tags':")
            conn = sqlite3.connect("data/memory.db")
            cursor = conn.cursor()

            # Znajdź instrukcje SQL używające conf lub tags
            import inspect
            import re

            memory_source = inspect.getsource(Memory)
            sql_statements = re.findall(
                r"(SELECT|INSERT|UPDATE|DELETE).*?;", memory_source, re.DOTALL
            )

            problem_columns = ["conf", "tags", "text_enc"]

            for stmt in sql_statements:
                for col in problem_columns:
                    if col in stmt:
                        print(f"Znaleziono SQL używające '{col}': {stmt.strip()}")

                        # Sprawdź strukturę tabeli
                        cursor.execute("PRAGMA table_info(ltm)")
                        columns = {col[1] for col in cursor.fetchall()}
                        print(f"Dostępne kolumny w tabeli ltm: {columns}")
                        break  # Unikaj duplikatów

        except Exception as e:
            print(f"Błąd podczas szukania problematycznych kolumn: {e}")
            traceback.print_exc()

    except Exception as e:
        print(f"Główny błąd podczas testowania Memory: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    check_db_connection()
    print("\n" + "=" * 50 + "\n")
    test_memory_operations()
