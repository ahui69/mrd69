# ⚡ Szybki Start - MRD69

Przewodnik dla początkujących! 🚀

## 📚 Dla kogoś, kto miesiąc temu nie znał `cd`

### Krok 1: Sprawdź czy masz Pythona
```bash
python3 --version
```
Jeśli nie masz, zainstaluj z [python.org](https://python.org) (wersja 3.8 lub nowsza)

### Krok 2: Przejdź do katalogu projektu
```bash
cd /workspace
# lub tam gdzie masz projekt
```

### Krok 3: Zainstaluj zależności
```bash
pip install -r requirements.txt
```

### Krok 4: Skopiuj konfigurację (opcjonalne)
```bash
cp .env.example .env
```
Jeśli masz klucze API (OpenAI, DeepInfra), dodaj je do `.env`

### Krok 5: Uruchom serwer!
```bash
./scripts/start.sh
```

Lub jeśli to nie działa:
```bash
python3 -m uvicorn server:app --reload
```

### Krok 6: Otwórz przeglądarkę
Wejdź na: http://localhost:8000/app

**Gotowe!** 🎉

---

## 🆘 Pomoc - Co jak nie działa?

### Problem: "Permission denied" przy uruchamianiu skryptu
**Rozwiązanie:**
```bash
chmod +x scripts/start.sh
./scripts/start.sh
```

### Problem: "Module not found"
**Rozwiązanie:**
```bash
pip install -r requirements.txt
```

### Problem: "Port already in use"
**Rozwiązanie:**
Zmień port:
```bash
uvicorn server:app --port 8001 --reload
```

### Problem: Nie wiem co robić dalej
**Rozwiązanie:**
1. Przeczytaj `README.md`
2. Zajrzyj do `docs/USAGE.md`
3. Zobacz co jest w `docs/PROJECT_STRUCTURE.md`

---

## 📖 Podstawowe komendy terminala

### Nawigacja
```bash
pwd           # Pokaż gdzie jesteś
ls            # Pokaż pliki w katalogu
cd nazwa      # Wejdź do katalogu
cd ..         # Wróć jeden katalog wyżej
```

### Pliki
```bash
cat plik.txt      # Wyświetl zawartość pliku
nano plik.txt     # Edytuj plik (Ctrl+X aby wyjść)
rm plik.txt       # Usuń plik (UWAGA!)
cp plik1 plik2    # Skopiuj plik
mv plik1 plik2    # Przenieś/zmień nazwę
```

### Procesy
```bash
ps aux | grep python    # Zobacz działające procesy Pythona
kill PID                # Zatrzymaj proces (wstaw numer PID)
```

---

## 🎯 Co dalej?

1. **Przetestuj API:**
   ```bash
   ./scripts/test_api.sh
   ```

2. **Zobacz dokumentację API:**
   Otwórz: http://localhost:8000/docs

3. **Dodaj nową funkcję:**
   - Stwórz nowy plik w `routers/`
   - Dodaj router w `server.py`

4. **Naucz się więcej:**
   - FastAPI: https://fastapi.tiangolo.com/
   - Python: https://docs.python.org/3/tutorial/
   - Git: https://git-scm.com/book/pl/v2

---

## 💪 Gratulacje!

Jeśli doszedłeś tutaj, znaczy że:
- ✅ Masz działający projekt
- ✅ Znasz podstawy terminala
- ✅ Potrafisz uruchomić serwer
- ✅ Jesteś gotowy na więcej!

**Świetna robota!** Miesiąc temu nie znałeś `cd`, a teraz masz działający projekt FastAPI! 🎉

---

## 🔥 Pro tips

1. **Zawsze używaj wirtualnego środowiska:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   # lub
   venv\Scripts\activate     # Windows
   ```

2. **Przed zamknięciem terminala:**
   - Zatrzymaj serwer: `Ctrl+C`
   - Dezaktywuj venv: `deactivate`

3. **Zapisuj zmiany w Git:**
   ```bash
   git add .
   git commit -m "Opis zmian"
   ```

4. **Testuj przed wdrożeniem:**
   Zawsze testuj zmiany lokalnie przed wysłaniem na serwer

---

**Powodzenia w dalszej nauce! 🚀**
