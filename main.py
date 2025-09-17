from __future__ import annotations

import atexit
import json
import time
import traceback
from collections import defaultdict, deque
from typing import Any

from fastapi import FastAPI, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# === Import modułów ===
import autonauka
import config
import crypto_advisor_full
import file_client
import kimi_client
import memory

# === MORDZIX CORE - NAJLEPSZA PLATFORMA AI ===
import programista
import psychika
import runpod_sync
import search_client
import travelguide
import writing_all_pro

# === ZAAWANSOWANY SYSTEM PAMIĘCI ===
from memory import (
    ContextType,
    get_advanced_memory,
)
from mordzix_core import crypto_integration, mordzix_engine

# === SYSTEM NIEZAWODNOŚCI - ZAWSZE AKTYWNY ===
from reliability_core import get_reliable_system, reliable_operation, require_reliability_check


# === Rate Limiting ===
class RateLimiter:
    def __init__(self):
        # Store request timestamps per IP
        self.requests: dict[str, deque[float]] = defaultdict(lambda: deque())
        self.limits = {
            "default": (60, 60),  # 60 requests per 60 seconds
            "/run": (10, 60),  # 10 requests per 60 seconds for /run
            "/chat": (20, 60),  # 20 requests per 60 seconds for chat
        }

    def is_allowed(self, client_ip: str, path: str) -> bool:
        now = time.time()

        # Determine rate limit for this path
        limit_requests, limit_window = self.limits.get(path, self.limits["default"])

        # Clean old requests outside the window
        client_requests = self.requests[client_ip]
        while client_requests and client_requests[0] < now - limit_window:
            client_requests.popleft()

        # Check if within limit
        if len(client_requests) >= limit_requests:
            return False

        # Add current request
        client_requests.append(now)
        return True


rate_limiter = RateLimiter()

# === FastAPI setup ===
app = FastAPI(title=config.APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === INICJALIZACJA ZAAWANSOWANEGO SYSTEMU PAMIĘCI ===
advanced_memory = get_advanced_memory()


# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    path = request.url.path

    # Skip rate limiting for static files and health checks
    if path.startswith("/static/") or path == "/health":
        return await call_next(request)

    if not rate_limiter.is_allowed(client_ip, path):
        return JSONResponse(
            status_code=429, content={"error": "Rate limit exceeded. Please try again later."}
        )

    return await call_next(request)


# Serwowanie plików statycznych pod /static oraz root index.html
app.mount("/static", StaticFiles(directory="static"), name="static")


# Inicjalizacja pamięci i sprawdzenie czy wymaga załadowania danych początkowych
mem = memory.get_memory()
if not mem.get_profile():
    print("💡 Inicjalizacja nowej pamięci...")

    # Dodaj podstawowy profil użytkownika
    mem.set_profile_many({"name": "Użytkownik", "version": "1.0.0", "created_at": time.time()})

    # Dodaj kilka podstawowych faktów
    mem.add_fact(
        "To jest nowa instalacja asystenta z pamięcią.", conf=0.9, tags=["system", "init"]
    )
    mem.add_fact(
        "Pamięć długoterminowa została zainicjowana.", conf=0.9, tags=["system", "init"]
    )

    # Dodaj przykładowy cel
    mem.add_goal(
        "Pomaganie użytkownikowi najlepiej jak to możliwe.",
        priority=1.0,
        tags=["system", "goal"],
    )

    print("Inicjalizacja pamięci zakończona.")


def load_knowledge_from_jsonl(file_path: str):
    """Ładuje wiedzę z pliku JSONL do pamięci LTM."""
    print(f"🧠 Ładowanie wiedzy z pliku {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            count = 0
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get("text")
                    tags = data.get("tags", [])
                    if text:
                        mem.add_fact(text, conf=0.95, tags=["wiedza_jsonl"] + tags)
                        count += 1
                except json.JSONDecodeError:
                    print(f"⚠️ Ostrzeżenie: Pominięto nieprawidłową linię JSON w {file_path}")
            print(f"✅ Załadowano {count} faktów z {file_path}.")
    except FileNotFoundError:
        print(f"⚠️ Ostrzeżenie: Plik wiedzy {file_path} nie został znaleziony. Pomiń.")
    except Exception as e:
        print(f"❌ Błąd podczas ładowania wiedzy z {file_path}: {e}")


# Jednorazowe ładowanie wiedzy z memory.jsonl przy starcie aplikacji
load_knowledge_from_jsonl("data/memory.jsonl")


# === SYSTEM NIEZAWODNOŚCI - ZAWSZE AKTYWNY ===
from reliability_core import get_reliable_system, reliable_operation, require_reliability_check


# === Rate Limiting ===
class RateLimiter:
    def __init__(self):
        # Store request timestamps per IP
        self.requests: dict[str, deque[float]] = defaultdict(lambda: deque())
        self.limits = {
            "default": (60, 60),  # 60 requests per 60 seconds
            "/run": (10, 60),  # 10 requests per 60 seconds for /run
            "/chat": (20, 60),  # 20 requests per 60 seconds for chat
        }

    def is_allowed(self, client_ip: str, path: str) -> bool:
        now = time.time()

        # Determine rate limit for this path
        limit_requests, limit_window = self.limits.get(path, self.limits["default"])

        # Clean old requests outside the window
        client_requests = self.requests[client_ip]
        while client_requests and client_requests[0] < now - limit_window:
            client_requests.popleft()

        # Check if within limit
        if len(client_requests) >= limit_requests:
            return False

        # Add current request
        client_requests.append(now)
        return True


rate_limiter = RateLimiter()

# === FastAPI setup ===
app = FastAPI(title=config.APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === INICJALIZACJA ZAAWANSOWANEGO SYSTEMU PAMIĘCI ===
advanced_memory = get_advanced_memory()


# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    path = request.url.path

    # Skip rate limiting for static files and health checks
    if path.startswith("/static/") or path == "/health":
        return await call_next(request)

    if not rate_limiter.is_allowed(client_ip, path):
        return JSONResponse(
            status_code=429, content={"error": "Rate limit exceeded. Please try again later."}
        )

    return await call_next(request)


# Serwowanie plików statycznych pod /static oraz root index.html
app.mount("/static", StaticFiles(directory="static"), name="static")


# Inicjalizacja pamięci i sprawdzenie czy wymaga załadowania danych początkowych
mem = memory.get_memory()
if not mem.get_profile():
    print("💡 Inicjalizacja nowej pamięci...")

    # Dodaj podstawowy profil użytkownika
    mem.set_profile_many({"name": "Użytkownik", "version": "1.0.0", "created_at": time.time()})

    # Dodaj kilka podstawowych faktów
    mem.add_fact(
        "To jest nowa instalacja asystenta z pamięcią.", conf=0.9, tags=["system", "init"]
    )
    mem.add_fact(
        "Pamięć długoterminowa została zainicjowana.", conf=0.9, tags=["system", "init"]
    )

    # Dodaj przykładowy cel
    mem.add_goal(
        "Pomaganie użytkownikowi najlepiej jak to możliwe.",
        priority=1.0,
        tags=["system", "goal"],
    )

    print("Inicjalizacja pamięci zakończona.")


def load_knowledge_from_jsonl(file_path: str):
    """Ładuje wiedzę z pliku JSONL do pamięci LTM."""
    print(f"🧠 Ładowanie wiedzy z pliku {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            count = 0
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get("text")
                    tags = data.get("tags", [])
                    if text:
                        mem.add_fact(text, conf=0.95, tags=["wiedza_jsonl"] + tags)
                        count += 1
                except json.JSONDecodeError:
                    print(f"⚠️ Ostrzeżenie: Pominięto nieprawidłową linię JSON w {file_path}")
            print(f"✅ Załadowano {count} faktów z {file_path}.")
    except FileNotFoundError:
        print(f"⚠️ Ostrzeżenie: Plik wiedzy {file_path} nie został znaleziony. Pomiń.")
    except Exception as e:
        print(f"❌ Błąd podczas ładowania wiedzy z {file_path}: {e}")


# Jednorazowe ładowanie wiedzy z memory.jsonl przy starcie aplikacji
load_knowledge_from_jsonl("data/memory.jsonl")


# === SYSTEM NIEZAWODNOŚCI - ZAWSZE AKTYWNY ===
from reliability_core import get_reliable_system, reliable_operation, require_reliability_check


# === Rate Limiting ===
class RateLimiter:
    def __init__(self):
        # Store request timestamps per IP
        self.requests: dict[str, deque[float]] = defaultdict(lambda: deque())
        self.limits = {
            "default": (60, 60),  # 60 requests per 60 seconds
            "/run": (10, 60),  # 10 requests per 60 seconds for /run
            "/chat": (20, 60),  # 20 requests per 60 seconds for chat
        }

    def is_allowed(self, client_ip: str, path: str) -> bool:
        now = time.time()

        # Determine rate limit for this path
        limit_requests, limit_window = self.limits.get(path, self.limits["default"])

        # Clean old requests outside the window
        client_requests = self.requests[client_ip]
        while client_requests and client_requests[0] < now - limit_window:
            client_requests.popleft()

        # Check if within limit
        if len(client_requests) >= limit_requests:
            return False

        # Add current request
        client_requests.append(now)
        return True


rate_limiter = RateLimiter()

# === FastAPI setup ===
app = FastAPI(title=config.APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === INICJALIZACJA ZAAWANSOWANEGO SYSTEMU PAMIĘCI ===
advanced_memory = get_advanced_memory()


# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    path = request.url.path

    # Skip rate limiting for static files and health checks
    if path.startswith("/static/") or path == "/health":
        return await call_next(request)

    if not rate_limiter.is_allowed(client_ip, path):
        return JSONResponse(
            status_code=429, content={"error": "Rate limit exceeded. Please try again later."}
        )

    return await call_next(request)


# Serwowanie plików statycznych pod /static oraz root index.html
app.mount("/static", StaticFiles(directory="static"), name="static")


# Inicjalizacja pamięci i sprawdzenie czy wymaga załadowania danych początkowych
mem = memory.get_memory()
if not mem.get_profile():
    print("💡 Inicjalizacja nowej pamięci...")

    # Dodaj podstawowy profil użytkownika
    mem.set_profile_many({"name": "Użytkownik", "version": "1.0.0", "created_at": time.time()})

    # Dodaj kilka podstawowych faktów
    mem.add_fact(
        "To jest nowa instalacja asystenta z pamięcią.", conf=0.9, tags=["system", "init"]
    )
    mem.add_fact(
        "Pamięć długoterminowa została zainicjowana.", conf=0.9, tags=["system", "init"]
    )

    # Dodaj przykładowy cel
    mem.add_goal(
        "Pomaganie użytkownikowi najlepiej jak to możliwe.",
        priority=1.0,
        tags=["system", "goal"],
    )

    print("Inicjalizacja pamięci zakończona.")


def load_knowledge_from_jsonl(file_path: str):
    """Ładuje wiedzę z pliku JSONL do pamięci LTM."""
    print(f"🧠 Ładowanie wiedzy z pliku {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            count = 0
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get("text")
                    tags = data.get("tags", [])
                    if text:
                        mem.add_fact(text, conf=0.95, tags=["wiedza_jsonl"] + tags)
                        count += 1
                except json.JSONDecodeError:
                    print(f"⚠️ Ostrzeżenie: Pominięto nieprawidłową linię JSON w {file_path}")
            print(f"✅ Załadowano {count} faktów z {file_path}.")
    except FileNotFoundError:
        print(f"⚠️ Ostrzeżenie: Plik wiedzy {file_path} nie został znaleziony. Pomiń.")
    except Exception as e:
        print(f"❌ Błąd podczas ładowania wiedzy z {file_path}: {e}")


# Jednorazowe ładowanie wiedzy z memory.jsonl przy starcie aplikacji
load_knowledge_from_jsonl("data/memory.jsonl")


# === INICJALIZACJA SYSTEMU NIEZAWODNOŚCI ===
def _initialize_reliability_system():
    """Inicjalizuje i sprawdza system niezawodności."""
    try:
        reliable_system = get_reliable_system()
        health = reliable_system.get_system_health()

        print("🚀 System niezawodności zainicjalizowany:")
        print(f"   ✅ Health Score: {health['health_score']:.2f}/1.0")
        print("   🔄 Backpressure: Aktywny")
        print("   🔐 Idempotencja: Aktywna")
        print("   📊 Telemetria: Aktywna")

        # Test krytyczny
        test_result = reliable_system.process_critical_operation(
            "system_test", test_data="System niezawodności aktywny"
        )

        if test_result["success"]:
            print("   ✅ Test krytyczny: PASSED")
        else:
            raise RuntimeError(f"Test krytyczny FAILED: {test_result}")

        return reliable_system

    except Exception as e:
        print(f"❌ KRYTYCZNY BŁĄD: Nie można zainicjalizować systemu niezawodności: {e}")
        print("🚨 APLIKACJA NIE MOŻE DZIAŁAĆ BEZ SYSTEMU NIEZAWODNOŚCI")
        raise SystemExit(1)


# Inicjalizacja synchronizacji RunPod, jeśli skonfigurowana
if config.USE_RUNPOD:
    print("Inicjalizacja synchronizacji z RunPod...")
    runpod_sync.start_runpod_sync()

    # Rejestracja funkcji zamykającej przy wyłączaniu aplikacji
    def cleanup():
        print("Zatrzymywanie synchronizacji RunPod i wykonywanie końcowej kopii...")
        runpod_sync.force_runpod_sync()  # Wykonaj końcową kopię
        runpod_sync.stop_runpod_sync()  # Zatrzymaj synchronizację

    atexit.register(cleanup)

# Inicjalizacja pamięci
_initialize_memory()

# === KRYTYCZNA INICJALIZACJA SYSTEMU NIEZAWODNOŚCI ===
reliable_system = _initialize_reliability_system()


@app.get("/")
async def root_index():
    return FileResponse("static/index.html")


# WebSocket endpoint
@require_reliability_check
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # Sprawdź system niezawodności na początku sesji
        reliable_system = get_reliable_system()
        health = reliable_system.get_system_health()

        if health["health_score"] < 0.5:
            await websocket.send_text(
                "⚠️ UWAGA: System działa w trybie ograniczonym ze względu na problemy niezawodności."
            )

        while True:
            msg = await websocket.receive_text()

            # --- Logika czatu z pełnym monitoringiem ---
            mem = _mem()

            # 1) zapisz user do STM - przez system niezawodności
            reliable_result = reliable_system.process_critical_operation(
                "memory_operation",
                memory_obj=mem,
                operation="stm_add",
                user_message=msg,
                assistant_message="",
            )

            if not reliable_result["success"]:
                await websocket.send_text(f"❌ Błąd systemu pamięci: {reliable_result['reason']}")
                continue

            # 2) zbierz kontekst z pamięci (RAG) - przez system niezawodności
            ctx = ""
            try:
                ctx_result = reliable_system.process_critical_operation(
                    "memory_operation",
                    memory_obj=mem,
                    operation="compose_context",
                    query=msg,
                    limit_chars=config.CONTEXT_CHARS_LIMIT,
                    topk=config.CONTEXT_TOP_K,
                )
                if ctx_result["success"]:
                    ctx = ctx_result["result"]["memory_result"]
            except Exception:
                ctx = ""

            # 3) odpowiedź LLM - przez system niezawodności
            try:
                answer_result = reliable_system.process_critical_operation(
                    "chat_completion",
                    user_text=msg,
                    context=ctx,
                    result=_chat_llm(msg, ctx),  # To już ma dekoratory
                )

                if answer_result["success"]:
                    answer = answer_result["result"]["original_result"]["original_result"]
                else:
                    answer = f"❌ Błąd systemu LLM: {answer_result['reason']}"

            except Exception as e:
                answer = f"❌ Błąd krytyczny: {str(e)}"

            # 4) domknij turę (asystent -> STM/LTM) - przez system niezawodności
            reliable_system.process_critical_operation(
                "memory_operation",
                memory_obj=mem,
                operation="stm_add",
                user_message="",
                assistant_message=answer,
            )

            # 4.5) Aktywne przetwarzanie okna STM
            try:
                mem.process_stm_window()
            except Exception as e:
                print(f"Error during STM processing: {e}")

            # Wysłanie odpowiedzi do klienta
            await websocket.send_text(answer)

            # Sprawdź health systemu co kilka wiadomości
            if reliable_system.total_operations % 10 == 0:
                health = reliable_system.get_system_health()
                if health["health_score"] < 0.3:
                    await websocket.send_text(
                        "🚨 UWAGA: Wykryto problemy z systemem niezawodności. Zalecany restart."
                    )

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket critical error: {e}")
        try:
            await websocket.send_text(f"🚨 KRYTYCZNY BŁĄD: {str(e)}")
        except:
            pass


# ====== Utilities ======


@require_reliability_check
def _mem():
    try:
        return memory.get_memory()  # z naszego memory.py (singleton)
    except Exception:
        # awaryjnie zainicjuj
        from memory import get_memory

        return get_memory()


@reliable_operation("chat_llm")
@require_reliability_check
def _chat_llm(user_text: str, ctx_text: str) -> str:
    # Sprawdź czy to polecenie ładowania wiedzy
    if user_text.lower().startswith(
        ("wiedze laduj", "wiedzę ładuj", "wiedza laduj", "wiedza ładuj")
    ):
        return _handle_knowledge_loading_command(user_text)

    msgs = [{"role": "system", "content": config.SYS_PROMPT}]

    # Dodaj wyniki wyszukiwania zewnętrznej wiedzy, jeśli potrzeba
    web_search_results = ""
    if search_client.should_search_web(user_text):
        web_search_results = search_client.search_knowledge(user_text)
        if web_search_results and web_search_results != "Nie znaleziono wyników wyszukiwania.":
            # Dodaj wyniki wyszukiwania do kontekstu
            msgs.append(
                {
                    "role": "system",
                    "content": f"AKTUALNE INFORMACJE Z INTERNETU:\n\n{web_search_results}",
                }
            )

    # Dodaj istniejący kontekst z pamięci
    if ctx_text:
        msgs.append({"role": "system", "content": ctx_text})

    msgs.append({"role": "user", "content": user_text})

    # Używamy modelu Kimi-K2-Instruct na froncie
    try:
        # System niezawodności automatycznie monitoruje to wywołanie
        response = kimi_client.kimi_chat(
            msgs,
            max_tokens=config.CHAT_MAX_TOKENS,
            temperature=config.CHAT_TEMPERATURE,
        )
        return response
    except Exception as e:
        print(f"Kimi API error: {e}")
        # Fallback do Qwen jeśli Kimi zawiedzie
        out = psychika._llm_chat(
            msgs,
            maxtok=config.CHAT_MAX_TOKENS,
            temp=config.CHAT_TEMPERATURE,
        )
        return out or "Nie mam pewnej odpowiedzi. Doprecyzuj proszę."


@reliable_operation("memory_operation")
@require_reliability_check
def _handle_knowledge_loading_command(user_text: str) -> str:
    """Obsługuje komendy ładowania wiedzy w formacie 'wiedze laduj [treść/URL/ścieżka]'"""
    mem = _mem()
    command_parts = user_text.split(" ", 2)  # Podziel na "wiedze", "laduj", "reszta"

    if len(command_parts) < 3:
        return (
            "ℹ️ Nie podano treści do załadowania. Użyj jednego z formatów:\n"
            "- wiedze laduj [tekst do załadowania]\n"
            "- wiedze laduj url: [adres strony]\n"
            "- wiedze laduj plik: [ścieżka do pliku]"
        )

    content = command_parts[2].strip()

    # Sprawdzenie, czy to URL
    if content.startswith(("url:", "adres:", "link:", "strona:")):
        url = content.split(":", 1)[1].strip()
        if not url.startswith(("http://", "https://")):
            return "⚠️ Podany adres URL jest nieprawidłowy. Adres musi zaczynać się od http:// lub https://"

        result = mem.load_knowledge_from_url(url)
        if result["ok"]:
            return (
                f"✅ Załadowano wiedzę z URL: {url}\n"
                f"Dodano: {result['added']} faktów\n"
                f"Połączono: {result['merged']} faktów\n"
                f"Łącznie: {result['total']} faktów w bazie wiedzy\n"
                f"Źródło: {result['source']}"
            )
        else:
            return f"❌ Błąd podczas ładowania wiedzy z URL: {result['reason']}"

    # Sprawdzenie, czy to ścieżka do pliku
    elif content.startswith(("plik:", "file:", "path:")):
        file_path = content.split(":", 1)[1].strip()

        result = mem.load_knowledge_from_file(file_path)
        if result["ok"]:
            return (
                f"✅ Załadowano wiedzę z pliku: {file_path}\n"
                f"Dodano: {result['added']} faktów\n"
                f"Połączono: {result['merged']} faktów\n"
                f"Łącznie: {result['total']} faktów w bazie wiedzy\n"
                f"Źródło: {result['source']}"
            )
        else:
            return f"❌ Błąd podczas ładowania wiedzy z pliku: {result['reason']}"

    # Domyślnie traktuj jako tekst do załadowania
    else:
        result = mem.load_knowledge(content)
        if result["ok"]:
            return (
                f"✅ Załadowano wiedzę z podanego tekstu\n"
                f"Dodano: {result['added']} faktów\n"
                f"Połączono: {result['merged']} faktów\n"
                f"Łącznie: {result['total']} faktów w bazie wiedzy\n"
                f"Źródło: {result['source']}"
            )
        else:
            return f"❌ Błąd podczas ładowania wiedzy: {result['reason']}"


# === SYSTEM NIEZAWODNOŚCI - Endpointy monitorowania ===


@require_reliability_check
@app.get("/reliability/status")
async def reliability_status():
    """Szczegółowy status systemu niezawodności."""
    try:
        reliable_system = get_reliable_system()
        health = reliable_system.get_system_health()

        return {
            "status": "operational" if health["active"] else "disabled",
            "health_score": health["health_score"],
            "uptime_hours": health["uptime_seconds"] / 3600,
            "total_operations": health["total_operations"],
            "error_rate": health["error_rate"],
            "components": {
                "backpressure": {
                    "active_ticks": health["backpressure_status"]["active_ticks"],
                    "utilization": health["backpressure_status"]["utilization"],
                    "rejection_rate": health["backpressure_status"]["rejection_rate"],
                },
                "memory_contracts": {
                    "coverage": health["memory_contract_metrics"]["coverage"],
                    "fallback_rate": health["memory_contract_metrics"]["fallback_rate"],
                },
                "idempotency": {
                    "registered_actions": health["idempotency_stats"]["total_registered"],
                    "duplicate_rate": health["idempotency_stats"]["duplicate_rate"],
                },
                "telemetry": health["telemetry_metrics"],
            },
            "timestamp": health["timestamp"],
        }
    except Exception as e:
        return {
            "status": "critical_error",
            "error": str(e),
            "message": "System niezawodności niedostępny",
            "timestamp": time.time(),
        }


@require_reliability_check
@app.post("/reliability/cleanup")
async def force_reliability_cleanup():
    """Wymusza czyszczenie systemu niezawodności."""
    try:
        reliable_system = get_reliable_system()
        reliable_system.force_cleanup()

        return {
            "status": "success",
            "message": "Czyszczenie systemu zakończone",
            "timestamp": time.time(),
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "timestamp": time.time()}


@app.post("/reliability/emergency_disable")
async def emergency_disable_reliability():
    """UWAGA: Wyłącza system niezawodności - tylko w sytuacjach awaryjnych!"""
    try:
        reliable_system = get_reliable_system()
        reliable_system.disable_system("Emergency manual override")

        return {
            "status": "disabled",
            "warning": "SYSTEM NIEZAWODNOŚCI WYŁĄCZONY - Aplikacja działa bez mechanizmów bezpieczeństwa!",
            "timestamp": time.time(),
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "timestamp": time.time()}


@app.post("/reliability/enable")
async def enable_reliability():
    """Włącza system niezawodności z powrotem."""
    try:
        reliable_system = get_reliable_system()
        reliable_system.enable_system()

        return {
            "status": "enabled",
            "message": "System niezawodności włączony",
            "timestamp": time.time(),
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "timestamp": time.time()}


# === Health z monitoringiem niezawodności ===
@require_reliability_check
@app.get("/health")
async def health():
    try:
        reliable_system = get_reliable_system()
        system_health = reliable_system.get_system_health()

        return {
            "status": "ok" if system_health["health_score"] > 0.5 else "degraded",
            "ts": time.time(),
            "reliability": {
                "health_score": system_health["health_score"],
                "error_rate": system_health["error_rate"],
                "total_operations": system_health["total_operations"],
                "backpressure_utilization": system_health["backpressure_status"]["utilization"],
                "active": system_health["active"],
            },
        }
    except Exception as e:
        return {
            "status": "critical_error",
            "error": str(e),
            "ts": time.time(),
            "reliability": "system_unavailable",
        }


# === Wyszukiwanie wiedzy ===
@app.get("/search")
async def search(q: str, num: int = 5):
    """
    Endpoint do bezpośredniego wyszukiwania wiedzy.

    Args:
        q: Zapytanie do wyszukania
        num: Liczba wyników do zwrócenia (domyślnie 5)

    Returns:
        Wyniki wyszukiwania
    """
    try:
        # Filtrujemy liczbę wyników
        num = max(1, min(10, num))

        # Wyszukaj informacje
        results = search_client.search_web(q, num)

        return {
            "query": q,
            "results": results,
            "count": len(results),
        }
    except Exception as e:
        return {"error": str(e), "query": q}


# === Version ===


@app.post("/uploadfile/")
async def create_upload_file(
    file: list[UploadFile] | None = None,  # type: ignore[assignment]
):
    """Przyjmuje jeden lub wiele plików pod kluczem form-data 'file'."""
    out: list[dict] = []
    try:
        if file is None:
            return {"error": "No files provided"}
        for f in file:
            if not f.filename:
                out.append({"error": "No filename supplied"})
                continue
            file_bytes = await f.read()
            result = file_client.upload_file(file_bytes, f.filename)
            if not result.ok:
                out.append(
                    {
                        "error": "File upload failed",
                        "details": result.err,
                    }
                )
                continue
            rel_path = result.data["path"].replace(str(file_client.ROOT), "").lstrip("/\\")
            text_result = file_client.extract_text(rel_path)
            if text_result.ok and text_result.data and text_result.data.get("text"):
                extracted_text = text_result.data["text"]
                memory.get_memory().add_entry(
                    f"Zawartość pliku {f.filename}:\n{extracted_text}",
                    metadata={
                        "source": "file_upload",
                        "filename": f.filename,
                    },
                )
                out.append(
                    {
                        "msg": (f"📂 Plik {f.filename} wczytany i przetworzony."),
                        "filename": f.filename,
                    }
                )
            else:
                out.append(
                    {
                        "msg": (
                            f"📂 Plik {f.filename} zapisany, ale nie udało się "
                            "wyekstrahować tekstu."
                        ),
                        "filename": f.filename,
                    }
                )
        # Jeśli jeden plik – zwróć obiekt dla wygody frontu
        if len(out) == 1:
            return out[0]
        return {"results": out}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


@app.get("/boot")
async def boot():
    """Zwraca krótką wiadomość systemową z pamięci (stats + profil/goals)."""
    mem = _mem()
    try:
        st = mem.stats()
        prof = mem.get_profile()
        goals = mem.get_goals()[:3]
        intro = [
            "[Pakiet wiedzy załadowany]",
            (
                "Fakty: "
                f"{st.get('facts', 0)}, cele: {len(goals)}, profil: "
                f"{len(prof)} kluczy."
            ),
        ]
        if goals:
            intro.append("Cele: " + "; ".join(g["title"] for g in goals))
        return {"msg": "\n".join(intro)}
    except Exception:
        return {"msg": "[Pamięć gotowa]"}


@app.get("/episodes")
async def episodes(limit: int = 50):
    try:
        return memory.get_memory().episodes_tail(limit)
    except Exception:
        return []


@app.post("/reset")
async def reset():
    """Czyści STM (zachowując ogon) i zaczyna pusty czat."""
    try:
        memory.get_memory().force_flush_stm()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/restore")
async def restore(payload: dict | None = None):
    """Czyści STM i odtwarza wybrany epizod (u,a) do kontynuacji."""
    try:
        payload = payload or {}
        u = payload.get("u", "")
        a = payload.get("a", "")
        mem = memory.get_memory()
        mem.force_flush_stm()
        if u or a:
            mem.stm_add(u, a)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# === Extra endpoints ===


@app.get("/modules")
async def modules():
    # sprawdzaj opcjonalne moduły bez twardego importu
    try:
        import images_client as _images_client  # type: ignore

        has_images = hasattr(_images_client, "generate_image")
    except Exception:
        has_images = False

    # Sprawdź czy moduł ładowania wiedzy działa poprawnie
    knowledge_loader_ok = hasattr(memory.get_memory(), "load_knowledge")

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
        "runpod": config.USE_RUNPOD,
        "knowledge_loader": knowledge_loader_ok,
    }


# === RunPod Integration ===
@app.get("/runpod/status")
async def runpod_status():
    """Sprawdza status integracji RunPod."""
    if not config.USE_RUNPOD:
        return {"status": "disabled", "message": "RunPod integration is disabled"}

    sync = runpod_sync.get_runpod_sync()
    pod_status = sync.check_pod_status()

    return {
        "status": "enabled",
        "pod_status": pod_status,
        "local_db_exists": sync.local_db_path.exists(),
        "local_db_size": sync.local_db_path.stat().st_size if sync.local_db_path.exists() else 0,
        "remote_db_exists": sync.remote_db_path.exists(),
        "remote_db_size": sync.remote_db_path.stat().st_size if sync.remote_db_path.exists() else 0,
    }


@app.post("/runpod/sync")
async def force_runpod_backup():
    """Wymusza natychmiastową synchronizację z RunPod."""
    if not config.USE_RUNPOD:
        return JSONResponse(status_code=400, content={"error": "RunPod integration is disabled"})

    success = runpod_sync.force_runpod_sync()
    if success:
        return {"status": "success", "message": "Backup completed successfully"}
    else:
        return JSONResponse(
            status_code=500, content={"status": "error", "message": "Backup failed"}
        )


@app.post("/runpod/restore")
async def restore_from_runpod():
    """Odtwarza bazę danych z kopii zapasowej RunPod."""
    if not config.USE_RUNPOD:
        return JSONResponse(status_code=400, content={"error": "RunPod integration is disabled"})

    sync = runpod_sync.get_runpod_sync()
    success = sync._restore_from_backup()

    if success:
        return {"status": "success", "message": "Database restored successfully"}
    else:
        return JSONResponse(
            status_code=500, content={"status": "error", "message": "Restore failed"}
        )


# === Knowledge Loader API ===


class KnowledgeTextData(BaseModel):
    content: str
    source_name: str = "user_input"
    confidence: float = 0.85


class KnowledgeURLData(BaseModel):
    url: str
    source_name: str = "web_content"
    confidence: float = 0.85


class KnowledgeFileData(BaseModel):
    file_path: str
    source_name: str = "file_content"
    confidence: float = 0.85


@app.post("/knowledge/load-text")
async def load_knowledge_from_text(data: KnowledgeTextData):
    """Ładuje wiedzę z tekstu do pamięci długoterminowej."""
    mem = _mem()
    result = mem.load_knowledge(
        data.content, source_name=data.source_name, confidence=data.confidence
    )

    if result["ok"]:
        return {
            "status": "success",
            "added": result["added"],
            "merged": result["merged"],
            "total": result["total"],
            "source": result["source"],
        }
    else:
        return JSONResponse(
            status_code=400, content={"status": "error", "message": result["reason"]}
        )


@app.post("/knowledge/load-url")
async def load_knowledge_from_url(data: KnowledgeURLData):
    """Ładuje wiedzę ze strony internetowej do pamięci długoterminowej."""
    mem = _mem()
    result = mem.load_knowledge_from_url(data.url, source_name=data.source_name)

    if result["ok"]:
        return {
            "status": "success",
            "added": result["added"],
            "merged": result["merged"],
            "total": result["total"],
            "source": result["source"],
        }
    else:
        return JSONResponse(
            status_code=400, content={"status": "error", "message": result["reason"]}
        )


@app.post("/knowledge/load-file")
async def load_knowledge_from_file(data: KnowledgeFileData):
    """Ładuje wiedzę z pliku do pamięci długoterminowej."""
    mem = _mem()
    result = mem.load_knowledge_from_file(data.file_path, source_name=data.source_name)

    if result["ok"]:
        return {
            "status": "success",
            "added": result["added"],
            "merged": result["merged"],
            "total": result["total"],
            "source": result["source"],
        }
    else:
        return JSONResponse(
            status_code=400, content={"status": "error", "message": result["reason"]}
        )


# === MORDZIX API MODELS ===
class MordzixChatRequest(BaseModel):
    thread_id: str | None = None
    user_id: str
    content: str
    message_type: str = "text"
    attachments: list[dict[str, Any]] | None = None


class MordzixThreadRequest(BaseModel):
    user_id: str
    title: str = "New Chat"


class MordzixVoiceRequest(BaseModel):
    thread_id: str
    user_id: str
    audio_data: str  # base64 encoded audio


# === MORDZIX API ENDPOINTS ===


@app.post("/mordzix/chat")
async def mordzix_chat(request: MordzixChatRequest):
    """Główny endpoint czatu Mordzix - bez kagańca, bez filtrów!"""
    try:
        # Create thread if doesn't exist
        if not request.thread_id:
            thread = mordzix_engine.create_thread(request.user_id, "New Chat")
            thread_id = thread.id
        else:
            thread_id = request.thread_id

        # Process message through Mordzix engine
        ai_response = await mordzix_engine.process_message(
            thread_id, request.user_id, request.content, request.message_type
        )

        return {
            "status": "success",
            "thread_id": thread_id,
            "message_id": ai_response.id,
            "content": ai_response.content,
            "timestamp": ai_response.timestamp,
            "message_type": ai_response.message_type,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"status": "error", "message": f"Mordzix error: {str(e)}"}
        )


@app.post("/mordzix/thread/create")
async def create_mordzix_thread(request: MordzixThreadRequest):
    """Tworzy nowy wątek rozmowy z Mordzix."""
    try:
        thread = mordzix_engine.create_thread(request.user_id, request.title)
        return {
            "status": "success",
            "thread_id": thread.id,
            "title": thread.title,
            "created_at": thread.created_at,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/mordzix/thread/{thread_id}/history")
async def get_thread_history(thread_id: str, limit: int = 100):
    """Pobiera historię wiadomości z wątku."""
    try:
        if thread_id not in mordzix_engine.message_history:
            return JSONResponse(
                status_code=404, content={"status": "error", "message": "Thread not found"}
            )

        messages = mordzix_engine.message_history[thread_id][-limit:]
        return {
            "status": "success",
            "thread_id": thread_id,
            "messages": [
                {
                    "id": msg.id,
                    "user_id": msg.user_id,
                    "content": msg.content,
                    "message_type": msg.message_type,
                    "timestamp": msg.timestamp,
                    "status": msg.status,
                    "attachments": msg.attachments,
                }
                for msg in messages
            ],
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/mordzix/threads/{user_id}")
async def get_user_threads(user_id: str):
    """Pobiera wszystkie wątki użytkownika."""
    try:
        user_threads = [
            {
                "id": thread.id,
                "title": thread.title,
                "created_at": thread.created_at,
                "last_activity": thread.last_activity,
                "message_count": thread.message_count,
            }
            for thread in mordzix_engine.active_threads.values()
            if thread.user_id == user_id
        ]

        return {"status": "success", "user_id": user_id, "threads": user_threads}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.websocket("/mordzix/ws/{user_id}")
async def mordzix_websocket(websocket: WebSocket, user_id: str):
    """WebSocket endpoint dla real-time komunikacji z Mordzix."""
    await websocket.accept()
    mordzix_engine.websocket_connections[user_id] = websocket

    try:
        # Send welcome message
        await websocket.send_text(
            json.dumps(
                {
                    "type": "system",
                    "content": "Yo! Mordzix online - jestem gotowy do gadania! 🚀",
                    "timestamp": time.time(),
                }
            )
        )

        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message_data = json.loads(data)

                # Process through chat engine
                if message_data.get("type") == "chat":
                    ai_response = await mordzix_engine.process_message(
                        message_data.get("thread_id"),
                        user_id,
                        message_data.get("content", ""),
                        message_data.get("message_type", "text"),
                    )

                    # Send response back
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "chat_response",
                                "message_id": ai_response.id,
                                "content": ai_response.content,
                                "timestamp": ai_response.timestamp,
                            }
                        )
                    )

            except WebSocketDisconnect:
                break
            except Exception as e:
                await websocket.send_text(
                    json.dumps(
                        {"type": "error", "content": f"Error: {str(e)}", "timestamp": time.time()}
                    )
                )

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Clean up connection
        if user_id in mordzix_engine.websocket_connections:
            del mordzix_engine.websocket_connections[user_id]


@app.post("/mordzix/crypto/portfolio/monitor")
async def start_crypto_monitoring(user_id: str, portfolio_id: str):
    """Rozpoczyna monitoring crypto portfolio z alertami w czacie."""
    try:
        await crypto_integration.start_portfolio_monitoring(user_id, portfolio_id)
        return {"status": "success", "message": f"Portfolio monitoring started for {portfolio_id}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/mordzix/crypto/score")
async def get_crypto_score_mordzix(token_id: str):
    """Pobiera score tokenu przez Mordzix system."""
    try:
        score_data = crypto_advisor_full.calculate_token_score(token_id)

        # Format response in Mordzix style
        mordzix_response = f"""🚀 Crypto score dla {token_id.upper()}:
💰 Cena: ${score_data['price']:,.2f}
📊 Score: {score_data['scores']['composite']}/100
⚡ Liquidity: {score_data['scores']['liquidity']}/100
🛡️ Trust: {score_data['scores']['trust']}/100
👥 Community: {score_data['scores']['community']}/100
⚠️ Risk level: {score_data['risk_level']}

{mordzix_engine.personality.enhance_response('Analiza gotowa!')}"""

        return {
            "status": "success",
            "token_id": token_id,
            "raw_data": score_data,
            "mordzix_response": mordzix_response,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


# === ZAAWANSOWANE ENDPOINTY PAMIĘCI EPIZODYCZNEJ ===


@app.get("/memory/timeline")
async def get_timeline(date_from: str | None = None, date_to: str | None = None, limit: int = 50):
    """Pobiera timeline interakcji"""
    try:
        entries = advanced_memory.get_timeline_entries(
            date_from=date_from, date_to=date_to, limit=limit
        )
        return {
            "status": "success",
            "entries": [entry.to_dict() for entry in entries],
            "count": len(entries),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/memory/timeline/search")
async def search_timeline(q: str, limit: int = 10):
    """Wyszukuje w timeline"""
    try:
        entries = advanced_memory.search_timeline(q, limit=limit)
        return {
            "status": "success",
            "query": q,
            "entries": [entry.to_dict() for entry in entries],
            "count": len(entries),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/memory/timeline/daily-summary")
async def create_daily_summary(date: str | None = None):
    """Tworzy automatyczne podsumowanie dnia"""
    try:
        entry = advanced_memory.create_daily_summary(date)
        if entry:
            entry_id = advanced_memory.add_timeline_entry(entry)
            return {"status": "success", "entry_id": entry_id, "entry": entry.to_dict()}
        else:
            return {"status": "info", "message": "Brak aktywności do podsumowania"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/memory/context/{context_type}")
async def get_context_memory(context_type: str):
    """Pobiera pamięć kontekstową"""
    try:
        context_enum = ContextType(context_type)
        context_memory = advanced_memory.get_context_memory(context_enum)
        return {
            "status": "success",
            "context_type": context_type,
            "memory": {
                "priority_facts": context_memory.priority_facts,
                "preferred_tools": context_memory.preferred_tools,
                "common_patterns": context_memory.common_patterns,
                "success_strategies": context_memory.success_strategies,
                "usage_count": context_memory.usage_count,
                "last_used": context_memory.last_used,
            },
        }
    except ValueError:
        return JSONResponse(
            status_code=400, content={"status": "error", "message": "Invalid context type"}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/memory/context/switch")
async def switch_context(context_type: str):
    """Przełącza kontekst pamięci"""
    try:
        context_enum = ContextType(context_type)
        result = advanced_memory.switch_context(context_enum)
        return {"status": "success", "context_info": result}
    except ValueError:
        return JSONResponse(
            status_code=400, content={"status": "error", "message": "Invalid context type"}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/memory/predictions")
async def get_predictions(user_input: str):
    """Przewiduje następne akcje"""
    try:
        predictions = advanced_memory.predict_next_action(user_input)
        return {"status": "success", "user_input": user_input, "predictions": predictions}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/memory/mood")
async def detect_mood(text: str):
    """Wykrywa nastrój z tekstu"""
    try:
        mood_info = advanced_memory.detect_user_mood(text)
        return {"status": "success", "text": text, "mood_info": mood_info}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/memory/reflections")
async def get_reflections(limit: int = 10):
    """Pobiera ostatnie refleksje AI"""
    try:
        reflections = advanced_memory.get_recent_reflections(limit=limit)
        return {
            "status": "success",
            "reflections": [vars(reflection) for reflection in reflections],
            "count": len(reflections),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/memory/reflection")
async def create_reflection(session_summary: str):
    """Tworzy nową refleksję"""
    try:
        reflection = advanced_memory.create_session_reflection(session_summary)
        reflection_id = advanced_memory.add_self_reflection(reflection)
        return {"status": "success", "reflection_id": reflection_id, "reflection": vars(reflection)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/memory/files")
async def search_files(q: str = "", file_type: str | None = None):
    """Wyszukuje w pamięci plików"""
    try:
        files = advanced_memory.search_file_memory(q, file_type)
        return {
            "status": "success",
            "query": q,
            "file_type": file_type,
            "files": files,
            "count": len(files),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/memory/file")
async def save_file_memory(
    filename: str,
    file_type: str,
    description: str = "",
    file_path: str = "",
    tags: list[str] | None = None,
):
    """Zapisuje pamięć o pliku"""
    try:
        file_id = advanced_memory.save_file_memory(
            filename=filename,
            file_type=file_type,
            description=description,
            file_path=file_path,
            tags=tags or [],
        )
        return {"status": "success", "file_id": file_id, "filename": filename}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/memory/versions")
async def list_memory_versions(limit: int = 20):
    """Lista wersji pamięci"""
    try:
        versions = advanced_memory.list_memory_versions(limit=limit)
        return {"status": "success", "versions": versions, "count": len(versions)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/memory/backup")
async def create_memory_backup(description: str = "Manual backup"):
    """Tworzy backup pamięci"""
    try:
        version_id = advanced_memory.create_memory_backup(description)
        return {"status": "success", "version_id": version_id, "description": description}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/memory/restore/{version_id}")
async def restore_memory(version_id: str):
    """Przywraca wersję pamięci"""
    try:
        result = advanced_memory.restore_memory_version(version_id)
        return {"status": "success", "restore_result": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/memory/relationships")
async def get_relationship_graph():
    """Pobiera graf relacji"""
    try:
        graph = advanced_memory.get_relationship_graph()
        return {"status": "success", "graph": graph}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/memory/person/{name}")
async def get_person_profile(name: str):
    """Pobiera profil osoby"""
    try:
        profile = advanced_memory.get_person_profile(name)
        return {"status": "success", "name": name, "profile": vars(profile)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/mordzix")
async def mordzix_interface():
    """Główny interfejs Mordzix - production chat."""
    return FileResponse("static/mordzix.html")


# === Run ===
# Uwaga: Do uruchamiania serwera użyj komendy:
# uvicorn main:app --host 0.0.0.0 --port 8080
# Blok poniżej jest tylko dla bezpośredniego uruchamiania python main.py
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=False)
