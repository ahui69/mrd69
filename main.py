from __future__ import annotations

# main.py
import atexit
import json
import os
import subprocess
import time
import logging
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

# --- Konfiguracja i Logowanie ---
import config

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(logging, str(config.LOG_LEVEL).upper(), logging.INFO)
    ),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)
log = structlog.get_logger()

# --- Import modułów projektu ---
import autonauka
import crypto_advisor_full
import file_client
import kimi_client
import memory
import programista
import psychika
import runpod_sync
import search_client
import travelguide
import writing_all_pro

# --- Dodatkowe importy wymagane przez użytkownika ---
import prompt
import io_pipeline
import images_client
import pathlib

# --- Sprawdzenie folderu data ---
DATA_DIR = pathlib.Path("data")
if not DATA_DIR.exists():
    log.warning("Brak folderu data! Tworzę...", path=str(DATA_DIR))
    DATA_DIR.mkdir(parents=True, exist_ok=True)

from memory import ContextType, get_advanced_memory
from mordzix_core import crypto_integration, mordzix_engine
from reliability_core import get_reliable_system, reliable_operation, require_reliability_check

# --- Globalne obiekty ---
mem: memory.Memory | None = None
advanced_memory: Any | None = None
reliable_system: Any | None = None


# === Cykl życia aplikacji (inicjalizacja przy starcie) ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    global mem, advanced_memory, reliable_system
    log.info("--- Aplikacja startuje: Inicjalizacja systemów ---")

    try:
        reliable_system = get_reliable_system()
        health = reliable_system.get_system_health()
        log.info("System niezawodności zainicjalizowany", health_score=health.get("health_score", "N/A"))
    except Exception as e:
        log.critical("Nie można zainicjalizować systemu niezawodności!", error=str(e), exc_info=True)
        raise SystemExit(1)

    mem = memory.get_memory()
    advanced_memory = get_advanced_memory()

    if not mem.get_profile():
        log.info("Inicjalizacja nowej pamięci...")
        mem.set_profile_many({"name": "Użytkownik", "version": "1.0.0", "created_at": time.time()})
        mem.add_fact("Nowa instalacja asystenta z pamięcią.", tags=["system", "init"])
        log.info("Inicjalizacja pamięci zakończona.")

    # Ładowanie wiedzy z plików (opcjonalne – jeśli plików brak, logujemy ostrzeżenie i lecimy dalej)
    load_knowledge_from_jsonl("data/memory.jsonl")
    load_knowledge_from_jsonl("data/start.seed.jsonl")

    # Inicjalizacja RunPod Sync
    if getattr(config, "USE_RUNPOD", False):
        log.info("Inicjalizacja synchronizacji z RunPod...")
        runpod_sync.start_runpod_sync()
        atexit.register(runpod_cleanup)

    log.info("--- Aplikacja gotowa do pracy ---")
    yield
    log.info("--- Aplikacja się zamyka ---")


def runpod_cleanup():
    if getattr(config, "USE_RUNPOD", False):
        log.info("Zatrzymywanie synchronizacji RunPod...")
        runpod_sync.force_runpod_sync()
        runpod_sync.stop_runpod_sync()


# === Konfiguracja FastAPI ===
app = FastAPI(title=config.APP_TITLE, lifespan=lifespan)

# CORS – dbaj o listę (nawet gdy w .env jest pojedynczy string)
_allowed = getattr(config, "CORS_ALLOWED_ORIGINS", ["*"])
if isinstance(_allowed, str):
    _allowed = [_allowed]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


# === Funkcje pomocnicze ===
def get_git_commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"


def load_knowledge_from_jsonl(file_path: str):
    log.info("Ładowanie wiedzy z pliku", file_path=file_path)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("text")
                    if text:
                        tags = ["wiedza_jsonl"]
                        if isinstance(data.get("tags"), list):
                            tags += data["tags"]
                        mem.add_fact(text, conf=0.95, tags=tags)
                except json.JSONDecodeError as je:
                    log.warning("Pominięto nieprawidłową linię JSON", file_path=file_path, error=str(je))
    except FileNotFoundError:
        log.warning("Plik wiedzy nie został znaleziony, pomijam.", file_path=file_path)
    except Exception as e:
        log.error("Błąd podczas ładowania wiedzy", file_path=file_path, error=str(e))


# === Główne Endpoints ===
@app.get("/")
async def root_index():
    return FileResponse("static/index.html")


@app.get("/mordzix")
async def mordzix_interface():
    return FileResponse("static/mordzix.html")


# === Monitoring i Zdrowie ===
@app.get("/health")
async def health():
    checks = {
        "memory": "ok" if mem is not None else "error",
        "reliability_system": "ok" if reliable_system is not None else "error",
        "llm_keys": "ok" if os.getenv("LLM_API_KEY") else "missing",
    }
    status_ok = all(v == "ok" for v in checks.values())
    status_code = 200 if status_ok else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if status_ok else "degraded",
            "timestamp": time.time(),
            "checks": checks,
        },
    )


@app.get("/version")
async def version():
    return {
        "version": "3.0.0",
        "git_sha": get_git_commit_sha(),
        "build_date": os.getenv("BUILD_DATE", "unknown"),
    }


# === API DLA ZAAWANSOWANEJ PAMIĘCI ===

# --- Timeline API ---
@app.get("/memory/timeline", tags=["Advanced Memory"])
async def get_timeline_entries(limit: int = 50, offset: int = 0):
    return advanced_memory.get_timeline_entries(limit, offset)


@app.get("/memory/timeline/search", tags=["Advanced Memory"])
async def search_timeline_entries(q: str, limit: int = 20):
    return advanced_memory.search_timeline(q, limit)


# --- Context API ---
class ContextMemoryData(BaseModel):
    context_type: str
    priority_facts: list[str]
    active_goals: list[str]
    notes: str


@app.get("/memory/context/{context_type}", tags=["Advanced Memory"])
async def get_context(context_type: str):
    return advanced_memory.get_context_memory(context_type)


@app.post("/memory/context", tags=["Advanced Memory"])
async def update_context(data: ContextMemoryData):
    advanced_memory.update_context_memory(**data.dict())
    return {"status": "success"}


# --- Reflection & Emotion API ---
class ReflectionData(BaseModel):
    summary: str
    lessons_learned: list[str]
    rules_to_remember: list[str]


@app.get("/memory/reflections", tags=["Advanced Memory"])
async def get_reflections(limit: int = 10):
    return advanced_memory.get_recent_reflections(limit)


@app.post("/memory/reflections", tags=["Advanced Memory"])
async def add_reflection(data: ReflectionData):
    reflection_id = advanced_memory.add_self_reflection(**data.dict())
    return {"status": "success", "reflection_id": reflection_id}


@app.get("/memory/mood", tags=["Advanced Memory"])
async def get_mood(text: str):
    mood = advanced_memory.detect_user_mood(text)
    return {"text": text, "detected_mood": mood}


# --- File & Prediction API ---
class FileMemoryData(BaseModel):
    filename: str
    file_type: str
    file_path: str
    description: str
    tags: list[str]


@app.post("/memory/files", tags=["Advanced Memory"])
async def save_file(data: FileMemoryData):
    file_id = advanced_memory.save_file_memory(**data.dict())
    return {"status": "success", "file_id": file_id}


@app.get("/memory/files/{filename}", tags=["Advanced Memory"])
async def find_file(filename: str):
    return advanced_memory.find_file_by_name(filename)


@app.get("/memory/predict", tags=["Advanced Memory"])
async def get_prediction(trigger: str):
    return advanced_memory.get_prediction(trigger)


# --- Versioning & Relations API ---
@app.post("/memory/backup", tags=["Advanced Memory"])
async def backup_memory(description: str = "Manual backup"):
    version_hash = advanced_memory.create_memory_backup(description)
    return {"status": "success", "version_hash": version_hash}


@app.get("/memory/versions", tags=["Advanced Memory"])
async def list_versions(limit: int = 20):
    return advanced_memory.list_memory_versions(limit)


@app.post("/memory/restore/{version_hash}", tags=["Advanced Memory"])
async def restore_version(version_hash: str):
    success = advanced_memory.restore_memory_version(version_hash)
    return {"status": "success" if success else "error"}


class PersonProfileData(BaseModel):
    name: str
    role: str
    notes: str
    aliases: list[str] | None = None


@app.post("/memory/persons", tags=["Advanced Memory"])
async def add_person(data: PersonProfileData):
    person_id = advanced_memory.add_person_profile(**data.dict())
    return {"status": "success", "person_id": person_id}


@app.get("/memory/persons/{name}", tags=["Advanced Memory"])
async def get_person(name: str):
    return advanced_memory.get_person_profile(name)


# === WebSocket ===
@app.websocket("/ws")
@require_reliability_check
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    log.info("WebSocket client connected", client_id=client_id)
    try:
        while True:
            msg = await websocket.receive_text()
            log.info("WebSocket message received", client_id=client_id, message=msg)
            response_text = _chat_llm(msg, "")  # ewentualny kontekst później
            await websocket.send_text(response_text)
            log.info("WebSocket response sent", client_id=client_id, response=response_text)
    except WebSocketDisconnect:
        log.info("WebSocket client disconnected", client_id=client_id)
    except Exception as e:
        log.error("WebSocket critical error", client_id=client_id, error=str(e), exc_info=True)


# === Logika LLM ===
@reliable_operation("chat_llm")
@require_reliability_check
def _chat_llm(user_text: str, ctx_text: str) -> str:
    # komenda wsadowa ładowania wiedzy
    if user_text.lower().startswith(("wiedze laduj", "wiedzę ładuj")):
        return _handle_knowledge_loading_command(user_text)

    msgs = [{"role": "system", "content": config.SYS_PROMPT}]
    if ctx_text:
        msgs.append({"role": "system", "content": ctx_text})
    msgs.append({"role": "user", "content": user_text})

    try:
        return kimi_client.kimi_chat(
            msgs,
            max_tokens=getattr(config, "CHAT_MAX_TOKENS", 512),
            temperature=getattr(config, "CHAT_TEMPERATURE", 0.7),
        )
    except Exception as e:
        log.error("Kimi API error, fallback to Qwen", error=str(e))
        return (
            psychika._llm_chat(
                msgs,
                maxtok=getattr(config, "CHAT_MAX_TOKENS", 512),
                temp=getattr(config, "CHAT_TEMPERATURE", 0.7),
            )
            or "Nie mam pewnej odpowiedzi."
        )


@reliable_operation("memory_operation")
@require_reliability_check
def _handle_knowledge_loading_command(user_text: str) -> str:
    parts = user_text.split(" ", 2)
    if len(parts) < 3:
        return "ℹ️ Nie podano treści. Użyj: wiedze laduj [tekst/url:/plik:]"

    content = parts[2].strip()
    if content.startswith(("url:", "adres:")):
        result = mem.load_knowledge_from_url(content.split(":", 1)[1].strip())
    elif content.startswith(("plik:", "file:")):
        result = mem.load_knowledge_from_file(content.split(":", 1)[1].strip())
    else:
        result = mem.load_knowledge(content)

    return (
        f"✅ Załadowano: {result['added']} | Połączono: {result['merged']}"
        if result.get("ok")
        else f"❌ Błąd: {result.get('reason', 'unknown')}"
    )


# === Pozostałe Endpoints ===
@app.get("/boot")
async def boot():
    try:
        st = mem.stats()
        prof = mem.get_profile()
        goals = mem.get_goals()[:3]
        intro = [
            "[Pakiet wiedzy załadowany]",
            f"Fakty: {st.get('facts', 0)}, cele: {len(goals)}, profil: {len(prof)} kluczy.",
            "Cele: " + "; ".join(g["title"] for g in goals) if goals else "",
        ]
        return {"msg": "\n".join(filter(None, intro))}
    except Exception as e:
        log.error("Boot endpoint failed", error=str(e))
        return {"msg": "[Pamięć gotowa, błąd statystyk]"}


@app.post("/uploadfile/")
async def create_upload_file(files: list[UploadFile]):
    out = []
    for f in files:
        try:
            file_bytes = await f.read()
            result = file_client.upload_file(file_bytes, f.filename)
            if not result.ok:
                raise Exception(result.err)

            rel_path = result.data["path"].replace(str(file_client.ROOT), "").lstrip("/\\")
            text_result = file_client.extract_text(rel_path)

            if text_result.ok and text_result.data.get("text"):
                mem.add_entry(
                    f"Zawartość pliku {f.filename}:\n{text_result.data['text']}",
                    metadata={"source": "file_upload", "filename": f.filename},
                )
                out.append({"msg": f"📂 Plik {f.filename} wczytany i przetworzony.", "filename": f.filename})
            else:
                out.append({"msg": f"📂 Plik {f.filename} zapisany, bez ekstrakcji tekstu.", "filename": f.filename})
        except Exception as e:
            log.error("Upload file failed", filename=f.filename, error=str(e))
            out.append({"error": f"Błąd przetwarzania pliku {f.filename}", "details": str(e)})
    return out[0] if len(out) == 1 else {"results": out}


@app.get("/episodes")
async def episodes(limit: int = 50):
    return mem.episodes_tail(limit)


@app.post("/reset")
async def reset():
    mem.force_flush_stm()
    return {"ok": True}


@app.post("/restore")
async def restore(payload: dict):
    if u := payload.get("u"):
        mem.stm_add(u, payload.get("a", ""))
    return {"ok": True}


@app.get("/modules")
async def modules():
    try:
        import images_client as _images_client

        has_images = hasattr(_images_client, "generate_image")
    except Exception:
        has_images = False
    return {
        "autonauka": hasattr(autonauka, "web_learn"),
        "psychika": True,
        "memory": True,
        "writer": hasattr(writing_all_pro, "assist"),
        "travelguide": hasattr(travelguide, "plan"),
        "images": has_images,
        "files": hasattr(file_client, "process_file"),
        "crypto": hasattr(crypto_advisor_full, "analyze"),
        "programista": hasattr(programista, "write_code"),
        "runpod": getattr(config, "USE_RUNPOD", False),
        "knowledge_loader": hasattr(mem, "load_knowledge"),
    }


# --- RunPod Integration ---
@app.get("/runpod/status")
async def runpod_status():
    if not getattr(config, "USE_RUNPOD", False):
        return {"status": "disabled"}
    return {"status": "enabled", "pod_status": runpod_sync.get_runpod_sync().check_pod_status()}


@app.post("/runpod/sync")
async def force_runpod_backup():
    if not getattr(config, "USE_RUNPOD", False):
        return JSONResponse(status_code=400, content={"error": "disabled"})
    return {"status": "success"} if runpod_sync.force_runpod_sync() else JSONResponse(status_code=500, content={"status": "error"})


@app.post("/runpod/restore")
async def restore_from_runpod():
    if not getattr(config, "USE_RUNPOD", False):
        return JSONResponse(status_code=400, content={"error": "disabled"})
    return (
        {"status": "success"}
        if runpod_sync.get_runpod_sync()._restore_from_backup()
        else JSONResponse(status_code=500, content={"status": "error"})
    )


# --- Knowledge Loader API ---
class KnowledgeTextData(BaseModel):
    content: str


class KnowledgeURLData(BaseModel):
    url: str


class KnowledgeFileData(BaseModel):
    file_path: str


@app.post("/knowledge/load-text")
async def load_knowledge_from_text(data: KnowledgeTextData):
    result = mem.load_knowledge(data.content)
    return result if result.get("ok") else JSONResponse(status_code=400, content=result)


@app.post("/knowledge/load-url")
async def load_knowledge_from_url(data: KnowledgeURLData):
    result = mem.load_knowledge_from_url(data.url)
    return result if result.get("ok") else JSONResponse(status_code=400, content=result)


@app.post("/knowledge/load-file")
async def load_knowledge_from_file(data: KnowledgeFileData):
    result = mem.load_knowledge_from_file(data.file_path)
    return result if result.get("ok") else JSONResponse(status_code=400, content=result)


# --- MORDZIX API ---
class MordzixChatRequest(BaseModel):
    thread_id: str | None = None
    user_id: str
    content: str


@app.post("/mordzix/chat")
async def mordzix_chat(request: MordzixChatRequest):
    try:
        thread_id = request.thread_id or mordzix_engine.create_thread(request.user_id, "New Chat").id
        ai_response = await mordzix_engine.process_message(thread_id, request.user_id, request.content)
        return {"status": "success", "thread_id": thread_id, "content": ai_response.content}
    except Exception as e:
        log.error("Mordzix chat error", error=str(e), exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


# === Uruchamianie ===
if __name__ == "__main__":
    log.info("Uruchamianie serwera w trybie deweloperskim...")
    import uvicorn

    uvicorn.run("main:app", host=getattr(config, "HOST", "0.0.0.0"), port=getattr(config, "PORT", 6969), reload=True)

