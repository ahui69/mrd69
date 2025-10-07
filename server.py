#!/usr/bin/env python3
# server.py – FastAPI + prosta historia czatu + router pamięci (jeśli jest)

from __future__ import annotations

import json
import logging
import os
import pathlib
import sqlite3
import time
import traceback
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from src.runpod_sync import RunPodSync  # type: ignore
except Exception:
    RunPodSync = None  # type: ignore

LOG = logging.getLogger("mrd69")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# ──────────────────────────────────────────────────────────────────────────────
# Opcjonalny klient LLM
# ──────────────────────────────────────────────────────────────────────────────
try:
    from src.llm_client import LLMClient  # type: ignore
except Exception:
    LLMClient = None  # type: ignore

llm = None
if LLMClient:
    try:
        llm = LLMClient(
            model=os.getenv(
                "LLM_MODEL",
                "meta-llama/Meta-Llama-3.1-70B-Instruct"
            ),
            provider=os.getenv("LLM_PROVIDER", "deepinfra"),
            timeout=int(os.getenv("LLM_TIMEOUT", "30")),
        )
    except Exception as e:
        LOG.warning("LLMClient init failed: %s", e)
        llm = None

runpod_sync = None
if RunPodSync:
    try:
        runpod_sync = RunPodSync()
    except Exception as e:
        LOG.warning("RunPodSync init failed: %s", e)
        runpod_sync = None

# ──────────────────────────────────────────────────────────────────────────────
# SQLite – prosta historia czatów
# ──────────────────────────────────────────────────────────────────────────────
  
  
def _db_path() -> str:
    cand = os.path.join(os.getcwd(), "mrd69", "data", "memory.db")
    if os.path.exists(cand):
        return cand
    pathlib.Path("data").mkdir(parents=True, exist_ok=True)
    return os.path.join("data", "chat_history.db")

  
CHAT_DB = _db_path()

  
def _conn():
    c = sqlite3.connect(CHAT_DB, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c

  
def _init_db() -> None:
    with _conn() as c:
        c.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS chat_threads(
              id TEXT PRIMARY KEY,
              title TEXT,
              created_ts INTEGER,
              updated_ts INTEGER
            );
            CREATE TABLE IF NOT EXISTS chat_messages(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              chat_id TEXT NOT NULL,
              role TEXT NOT NULL,
              content TEXT NOT NULL,
              ts INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chat_messages_chat
              ON chat_messages(chat_id, ts);
            """
        )

  
_init_db()

  
def db_upsert_thread(chat_id: str, title: Optional[str]) -> None:
    now = int(time.time())
    with _conn() as c:
        row = c.execute(
            "SELECT id FROM chat_threads WHERE id=?",
            (chat_id,)
        ).fetchone()
        if row is None:
            c.execute(
                "INSERT INTO chat_threads(id, title, created_ts, updated_ts) "
                "VALUES(?,?,?,?)",
                (chat_id, title or "Rozmowa", now, now),
            )
        else:
            if title:
                c.execute(
                    "UPDATE chat_threads SET title=?, updated_ts=? WHERE id=?",
                    (title, now, chat_id)
                )
            else:
                c.execute(
                    "UPDATE chat_threads SET updated_ts=? WHERE id=?",
                    (now, chat_id)
                )

  
def db_add_message(chat_id: str, role: str, content: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO chat_messages(chat_id, role, content, ts) "
            "VALUES(?,?,?,?)",
            (chat_id, role, content, int(time.time())),
        )

  
def db_get_chat(chat_id: str) -> Dict[str, Any]:
    with _conn() as c:
        msgs = c.execute(
            "SELECT role, content, ts FROM chat_messages "
            "WHERE chat_id=? ORDER BY ts ASC",
            (chat_id,)
        ).fetchall()
    return {
        "messages": [
            {"role": m["role"], "content": m["content"]} for m in msgs
        ]
    }

  
def db_list_chats(limit: int = 200) -> List[Dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, title, updated_ts FROM chat_threads "
            "ORDER BY updated_ts DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [
        {"id": r["id"], "title": r["title"], "ts": r["updated_ts"]}
        for r in rows
    ]

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI – app
# ──────────────────────────────────────────────────────────────────────────────
  
  
app = FastAPI(title="MRD69 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

  
  
# Frontend – jeśli istnieje ./frontend,
# montujemy pod /app (żeby /api/* było wolne)
FRONT_DIR = os.path.join(os.getcwd(), "frontend")
if os.path.isdir(FRONT_DIR):
    app.mount(
        "/app",
        StaticFiles(directory=FRONT_DIR, html=True),
        name="frontend"
    )

# Router pamięci (Twój memory_api.py). Jeśli jest – dodaj.
try:
    from src.memory_api import router as memory_router  # type: ignore
    app.include_router(memory_router)
    LOG.info("memory_api router mounted at /api/memory")
except Exception as e:
    LOG.warning("memory_api not mounted: %s", e)

try:
    from routers.writing import router as writing_router  # type: ignore
    app.include_router(writing_router)
    LOG.info("writing router mounted at /api/listings")
except Exception as e:
    LOG.warning("writing router not mounted: %s", e)

try:
    from routers.travel import router as travel_router  # type: ignore
    app.include_router(travel_router)
    LOG.info("travel router mounted at /api/travel")
except Exception as e:
    LOG.warning("travel router not mounted: %s", e)

try:
    from routers.crypto import router as crypto_router  # type: ignore
    app.include_router(crypto_router)
    LOG.info("crypto router mounted at /api/crypto")
except Exception as e:
    LOG.warning("crypto router not mounted: %s", e)



  
@app.on_event("startup")
async def _runpod_startup():
    if runpod_sync:
        try:
            runpod_sync.start_sync()
        except Exception as exc:
            LOG.warning("RunPodSync start failed: %s", exc)


@app.on_event("shutdown")
async def _runpod_shutdown():
    if runpod_sync:
        try:
            runpod_sync.stop_sync()
        except Exception as exc:
            LOG.warning("RunPodSync stop failed: %s", exc)

# ──────────────────────────────────────────────────────────────────────────────
# Modele żądań
# ──────────────────────────────────────────────────────────────────────────────
  
class ChatRequest(BaseModel):
    chatId: Optional[str] = None
    messages: List[Dict[str, Any]] = []
    newMessage: Dict[str, Any]
    attachments: Optional[List[Dict[str, Any]]] = []
    lang: Optional[str] = "pl-PL"

# ──────────────────────────────────────────────────────────────────────────────
# Endpoints API
# ──────────────────────────────────────────────────────────────────────────────
  
@app.get("/api/health")
def api_health():
    # proste info + czy router pamięci działa
    mem_ok = False
    try:
    # jeżeli router pamięci zamontowany,
    # powinien odpowiadać na /api/memory/health
        mem_ok = True
    except Exception:
        mem_ok = False
    return {
        "ok": True,
        "mode": "echo" if llm is None else "llm",
        "memory_router": mem_ok
    }

  
@app.get("/api/bootstrap")
def api_bootstrap():
    return {"prompts": [], "memory": [], "version": "local"}

  
@app.get("/api/history")
def api_history():
    return db_list_chats(200)

  
@app.get("/api/history/{chat_id}")
def api_history_chat(chat_id: str):
    return db_get_chat(chat_id)

  
@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    out = os.path.join(
        "data",
        "uploads",
        f"{int(time.time()*1000)}_{file.filename}"
    )
    pathlib.Path(os.path.dirname(out)).mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(await file.read())
    return {"name": file.filename, "path": out, "temp": False}

  
@app.post("/api/chat")
def api_chat(req: ChatRequest):
    try:
        chat_id = req.chatId or f"local-{int(time.time()*1000)}"
        current = db_get_chat(chat_id)["messages"]

        # dopisz nową wiadomość usera
        if req.newMessage:
            current.append(req.newMessage)
            if req.newMessage.get("role") == "user":
                title = (
                    req.newMessage.get("content") or "Rozmowa"
                ).strip()[:60]
                db_upsert_thread(chat_id, title or "Rozmowa")
            db_add_message(
                chat_id,
                req.newMessage.get("role", "user"),
                req.newMessage.get("content", "")
            )

        # odpowiedź modelu lub echo
        if llm:
            reply = llm.chat(
                messages=current,
                temperature=0.4,
                max_tokens=800,
                stream=False
            )
        else:
            reply = f"(echo) {current[-1]['content'] if current else ''}"

        db_add_message(chat_id, "assistant", reply)
        db_upsert_thread(chat_id, None)
        return {"reply": reply, "chatId": chat_id}
    except Exception as e:
        LOG.error("chat error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
