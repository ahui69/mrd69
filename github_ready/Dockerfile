# ════════════════════════════════════════════════════════
# DOCKERFILE - przepis jak "zapakować" Twoją appkę
# ════════════════════════════════════════════════════════

# KROK 1: Wybierz bazę (system operacyjny + Python)
# To jest jak "czysta instalka Windows/Linux"
FROM python:3.13-slim

# KROK 2: Ustaw katalog roboczy w kontenerze
# Wszystko będzie w /app
WORKDIR /app

# KROK 3: Skopiuj requirements i zainstaluj
# (robimy to osobno żeby Docker cachował to)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Dodatkowe pakiety które są potrzebne
RUN pip install --no-cache-dir python-multipart duckduckgo-search

# KROK 4: Skopiuj CAŁY kod do kontenera
COPY . .

# KROK 5: Stwórz katalogi na dane
RUN mkdir -p data logs uploads

# KROK 6: Port na którym działa aplikacja
EXPOSE 8000

# KROK 7: Komenda która uruchomi serwer
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

# ════════════════════════════════════════════════════════
# WYJAŚNIENIE:
# 
# FROM python:3.13-slim
#   → Weź gotowy obraz z Pythonem (jak "czysta instalka")
#
# WORKDIR /app
#   → cd /app (wszystko będzie tutaj)
#
# COPY requirements.txt .
#   → Skopiuj plik z bibliotek
#
# RUN pip install...
#   → Zainstaluj biblioteki (jak "pip install" u Ciebie)
#
# COPY . .
#   → Skopiuj CAŁY Twój kod
#
# EXPOSE 8000
#   → Aplikacja słucha na porcie 8000
#
# CMD [...]
#   → To uruchom gdy kontener startuje
# ════════════════════════════════════════════════════════
