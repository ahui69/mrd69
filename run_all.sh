#!/usr/bin/env bash
set -euo pipefail
cd /workspace/mrd69

# 1) venv + deps
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
python -m pip -q install --upgrade pip
if [ -f requirements.txt ]; then
  python -m pip -q install -r requirements.txt
else
  python -m pip -q install fastapi "uvicorn[standard]" requests psutil numpy pandas pydantic python-dotenv websockets PyYAML reportlab aiohttp
fi

# 2) port i ścieżka
export PYTHONPATH=/workspace/mrd69
PORT=5959

# 3) kill stare
fuser -k ${PORT}/tcp 2>/dev/null || true
pkill -f "uvicorn .*:${PORT}" 2>/dev/null || true

# 4) wybór app
APP="server:app"
python - <<PY >/dev/null 2>&1 || APP="_fallback_app:app"
import importlib
m=importlib.import_module("server"); getattr(m,"app")
PY

if [ "$APP" = "_fallback_app:app" ]; then
cat > _fallback_app.py <<'PY'
from fastapi import FastAPI
app = FastAPI(title="fallback")
@app.get("/api/health")
def health(): return {"status":"ok","app":"fallback"}
PY
fi

# 5) start backend
nohup uvicorn "$APP" --host 0.0.0.0 --port $PORT --reload > start.log 2>&1 &
echo $! > start.pid

# 6) czekaj na zdrowie
for i in {1..30}; do
  curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null && break || sleep 1
done

echo "== /api/health ==";        curl -s "http://127.0.0.1:$PORT/api/health" || true; echo
echo "== /docs ==";              echo "http://127.0.0.1:$PORT/docs"; echo

# 7) testy API (memory, chat)
echo "== /api/memory/health =="; curl -s "http://127.0.0.1:$PORT/api/memory/health" || true; echo
echo "== /api/chat (new) ==";    curl -s -X POST "http://127.0.0.1:$PORT/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"newMessage":{"role":"user","content":"Siema"}}' | tee /tmp/chat1.json || true; echo
CHAT=$(jq -r '.chatId' /tmp/chat1.json 2>/dev/null || echo "")
if [ -n "$CHAT" ] && [ "$CHAT" != "null" ]; then
  echo "== /api/chat (cont) =="; curl -s -X POST "http://127.0.0.1:$PORT/api/chat" \
    -H "Content-Type: application/json" \
    -d '{"chatId":"'"$CHAT"'","newMessage":{"role":"user","content":"Dalej"}}' || true; echo
fi

# 8) SELFTESTY MODUŁÓW
python - <<'PY' || true
import importlib, sys

def try_funcs(mod, *names, text="siema mordo popraw to zdanie"):
    for n in names:
        f = getattr(mod, n, None)
        if callable(f):
            try:
                out = f(text)
                print(f"[OK] {mod.__name__}.{n} ->", str(out)[:120])
                return True
            except Exception as e:
                print(f"[ERR] {mod.__name__}.{n} ->", e)
    return False

def try_classes(mod, *names, text="siema mordo popraw to zdanie"):
    for n in names:
        C = getattr(mod, n, None)
        if C:
            try:
                inst = C()
                for m in ("run","process","fix","improve"):
                    if hasattr(inst, m):
                        out = getattr(inst, m)(text)
                        print(f"[OK] {mod.__name__}.{n}.{m} ->", str(out)[:120])
                        return True
            except Exception as e:
                print(f"[ERR] {mod.__name__}.{n} ->", e)
    return False

targets = [
    ("fix_writing_all_pro", ("fix_text","fix","improve","rewrite","process"), ("WriterFixer","Fixer","ProWriter","Pipeline")),
    ("writing_all_pro",     ("fix_text","fix","improve","rewrite","process"), ("Writer","ProWriter","Pipeline")),
    ("psychika",            ("respond","think","process","run"),             ("Psychika","Brain","Core")),
    ("memory",              ("health","stats","remember","recall"),          ("Memory","Store","DB")),
    ("llm_client",          ("embed","chat","complete"),                     ("Client","LLMClient")),
    ("travelguide",         ("plan","suggest","build_itinerary"),            ("TravelGuide","Planner")),
    ("crypto_advisor_full", ("advice","analyze","price"),                    ("CryptoAdvisor","Advisor")),
    ("images_client",       ("gen","generate","text2img"),                   ("Images","Generator")),
]

for mod_name, funcs, classes in targets:
    try:
        mod = importlib.import_module(mod_name)
        ok = try_funcs(mod, *funcs) or try_classes(mod, *classes)
        if not ok:
            print(f"[NO-ENTRY] {mod_name} — nie znaleziono funkcji/klasy startowej")
    except Exception as e:
        print(f"[IMPORT-ERR] {mod_name} -> {e}")
PY

echo "== DONE =="
