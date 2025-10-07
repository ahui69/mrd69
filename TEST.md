# 🧪 TESTY - Sprawdź czy wszystko działa

## Szybki test wszystkich funkcji

```bash
# 1. Health check
curl http://localhost:8000/api/health

# 2. Bootstrap
curl http://localhost:8000/api/bootstrap

# 3. Chat (echo mode - bez LLM)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"newMessage": {"role":"user", "content":"test"}}'

# 4. Historia czatów
curl http://localhost:8000/api/history

# 5. Crypto - screener
curl "http://localhost:8000/api/crypto/screener?limit=5"

# 6. Travel - restauracje w Warszawie
curl "http://localhost:8000/api/travel/restaurants?place=Warszawa&max_results=3"

# 7. Listings - szukaj
curl "http://localhost:8000/api/listings/search?limit=5"
```

## Jeśli coś nie działa:

```bash
# Zobacz logi
tail -100 server_run.log

# Sprawdź czy serwer żyje
ps aux | grep uvicorn

# Restart
./stop.sh
./run.sh
```
