from __future__ import annotations
import os, json, time, uuid, re
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, APIRouter, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import requests

# ─── stałe ścieżki ─────────────────────────────────────────────────────────
ROOT = Path("/workspace/mrd69")
DATA = ROOT / "data"
SEED = ROOT / "data" / "sq3" / "seed.jsonl"
THREADS_DIR = DATA / "threads"
UPLOADS = DATA / "uploads"
FRONT = ROOT / "frontend"
for d in (THREADS_DIR, UPLOADS, FRONT):
    d.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")

# ─── LLM (DeepInfra/OpenAI-compatible) ─────────────────────────────────────
LLM_BASE = (os.getenv("LLM_BASE_URL") or "https://api.deepinfra.com/v1/openai").rstrip(
    "/"
)
LLM_KEY = os.getenv("LLM_API_KEY") or ""
LLM_MODEL = (os.getenv("LLM_MODEL") or "zai-org/GLM-4.5-Air").strip()
HTTP_TIMEOUT = int(os.getenv("TIMEOUT_HTTP", "60"))


def llm_reply(
    messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 800
) -> str:
    if not LLM_KEY:
        return "⚠️ Brak LLM_API_KEY — ustaw w .env albo w środowisku."
    try:
        url = f"{LLM_BASE}/chat/completions"
        body = {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        r = requests.post(
            url,
            json=body,
            timeout=HTTP_TIMEOUT,
            headers={
                "Authorization": f"Bearer {LLM_KEY}",
                "Content-Type": "application/json",
            },
        )
        r.raise_for_status()
        j = r.json()
        return ((j.get("choices") or [{}])[0].get("message") or {}).get(
            "content", ""
        ).strip() or "(pusto)"
    except Exception as e:
        return f"⚠️ API error: {e}"


# ─── threads storage ───────────────────────────────────────────────────────
def _tid() -> str:
    return time.strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4().int % 10**6)


def thread_path(tid: str) -> Path:
    return THREADS_DIR / f"{tid}.jsonl"


def create_thread() -> str:
    tid = _tid()
    p = thread_path(tid)
    p.touch()
    return tid


def append_msg(tid: str, role: str, content: str) -> None:
    p = thread_path(tid)
    with open(p, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"ts": int(time.time() * 1000), "role": role, "content": content},
                ensure_ascii=False,
            )
            + "\n"
        )


def read_thread(tid: str) -> List[Dict[str, Any]]:
    p = thread_path(tid)
    if not p.exists():
        return []
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for ln in f:
            try:
                out.append(json.loads(ln))
            except:
                pass
    return out


def list_threads() -> List[Dict[str, Any]]:
    items = []
    for p in sorted(
        THREADS_DIR.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True
    ):
        try:
            with open(p, "r", encoding="utf-8") as f:
                first = None
                last_ts = int(p.stat().st_mtime * 1000)
                for ln in f:
                    try:
                        obj = json.loads(ln)
                        last_ts = obj.get("ts", last_ts)
                        if not first and obj.get("role") == "user":
                            first = obj.get("content", "")
                    except:
                        pass
            name = (first or p.stem)[:60]
            items.append({"id": p.stem, "title": name, "updated_at": last_ts})
        except:
            pass
    return items


# ─── pamięć z SEED (proste wczytanie do RAM) ───────────────────────────────
MEM = []


def load_seed(seed_path: str | None = None) -> int:
    path = Path(seed_path or SEED)
    if not path.exists():
        return 0
    MEM.clear()
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                txt = obj.get("text") or obj.get("content") or obj.get("fact") or ""
                if txt:
                    MEM.append(txt)
            except:
                pass
    return len(MEM)


def memory_context(k: int = 12) -> str:
    # b. lekki kontekst: ostatnie k faktów
    if not MEM:
        return ""
    sl = MEM[-k:]
    return "Kontekst (wiedza):\n- " + "\n- ".join(sl)


# ─── psyche ────────────────────────────────────────────────────────────────
PSYCHE = {"mood": "spokój", "energy": 70, "creativity": 50}

# ─── FastAPI ───────────────────────────────────────────────────────────────
app = FastAPI(title="Mordzix API Bridge", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frontend (mój) pod /app
if FRONT.exists():
    app.mount("/app", StaticFiles(directory=str(FRONT), html=True), name="app")

api = APIRouter(prefix="/api")


@api.post("/system/boot")
async def boot(body: Dict[str, Any]):
    cnt = load_seed(body.get("seed_path"))
    return {"ok": True, "facts": cnt}


@api.get("/psyche")
async def psyche():
    return PSYCHE


@api.get("/threads")
async def get_threads():
    return list_threads()


@api.post("/threads")
async def post_thread():
    tid = create_thread()
    # startowy komunikat systemowy (niewidoczny na liście)
    append_msg(tid, "system", "Jesteś Mordzix PRO ULTRA. Odpowiadaj po polsku.")
    return {"id": tid}


@api.get("/threads/{tid}")
async def get_thread(tid: str):
    return read_thread(tid)


@api.post("/files")
async def upload_files(files: List[UploadFile] = File(...)):
    ids = []
    for f in files:
        name = f"{int(time.time()*1000)}_{re.sub(r'[^a-zA-Z0-9._-]+','_', f.filename)}"
        p = UPLOADS / name
        with open(p, "wb") as out:
            out.write(await f.read())
        ids.append(str(p))
    return {"ids": ids}


@api.post("/chat")
async def chat(body: Dict[str, Any]):
    tid = body.get("threadId")
    text = (body.get("text") or "").strip()
    files = body.get("files") or []
    if not tid:
        tid = create_thread()
    if not text and not files:
        return {"reply": "Nic nie dostałem 🤷"}

    # zapis mojej wypowiedzi
    user_txt = text
    if files:
        user_txt += "\n\n[Załączniki]:\n" + "\n".join(
            f"- {Path(p).name}" for p in files
        )
    append_msg(tid, "user", user_txt)

    # konstruuj kontekst LLM: system + pamięć + ostatnie 20 wypowiedzi
    hist = read_thread(tid)
    msgs = [
        {
            "role": "system",
            "content": "Jesteś Mordzix PRO ULTRA. Odpowiadaj po polsku, konkretnie.",
        }
    ]
    ctx = memory_context(12)
    if ctx:
        msgs.append({"role": "system", "content": ctx})
    for m in hist[-20:]:
        if m["role"] in ("user", "assistant"):
            msgs.append({"role": m["role"], "content": m["content"]})

    # wołanie modelu
    reply = llm_reply(msgs, temperature=0.65, max_tokens=900)
    append_msg(tid, "assistant", reply)

    return {"reply": reply, "threadId": tid}


app.include_router(api)


@app.get("/api/health")
def health():
    return {"ok": True, "model": LLM_MODEL, "seed_facts": len(MEM)}


# ── uruchamianie: uvicorn api_bridge:app --host 0.0.0.0 --port 8000
