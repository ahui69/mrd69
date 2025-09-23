from __future__ import annotations
import os

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=os.path.join("/workspace/mrd69", ".env"))
except Exception:
    pass

# LLM
LLM_BASE_URL = (os.getenv("LLM_BASE_URL") or "").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY") or ""
LLM_MODEL = os.getenv("LLM_MODEL") or ""

MINI_LLM_BASE_URL = (os.getenv("MINI_LLM_BASE_URL") or LLM_BASE_URL).rstrip("/")
MINI_LLM_API_KEY = os.getenv("MINI_LLM_API_KEY") or LLM_API_KEY
MINI_LLM_MODEL = os.getenv("MINI_LLM_MODEL") or "Qwen/Qwen2.5-4B-Instruct"

# NET
WEB_HTTP_TIMEOUT = int(os.getenv("WEB_HTTP_TIMEOUT", os.getenv("TIMEOUT_HTTP", "40")))
WEB_USER_AGENT = os.getenv("WEB_USER_AGENT", "Mrd69/1.0")

# MEMORY/DB
MEM_ROOT = os.getenv("MEM_ROOT", "/workspace/mrd69")
MEM_NS = (os.getenv("MEM_NS", "default") or "default").strip()
LTM_MIN_SCORE = float(os.getenv("LTM_MIN_SCORE", "0.25"))
MAX_LTM_FACTS = int(os.getenv("MAX_LTM_FACTS", "2000000"))
RECALL_TOPK_PER_SRC = int(os.getenv("RECALL_TOPK_PER_SRC", "40"))
STM_MAX_TURNS = int(os.getenv("STM_MAX_TURNS", "400"))
STM_KEEP_TAIL = int(os.getenv("STM_KEEP_TAIL", "100"))

# Embeddings (opcjonalnie)
LLM_EMBED_URL = (os.getenv("LLM_EMBED_URL") or "").rstrip("/")
LLM_EMBED_MODEL = os.getenv("LLM_EMBED_MODEL", "text-embedding-3-large")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("OPENAI_KEY", ""))

# RUNPOD
USE_RUNPOD = os.getenv("USE_RUNPOD", "0") in ("1", "true", "True")
RUNPOD_PERSIST_DIR = os.getenv("RUNPOD_PERSIST_DIR", "/runpod/persist")

# Travel / maps / weather (opcjonalnie)
GOOGLE_MAPS_KEY = os.getenv("GOOGLE_MAPS_KEY", "")
OPENROUTESERVICE_KEY = os.getenv("OPENROUTESERVICE_KEY", "")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")

# Files/IO
WRITER_OUT_DIR = os.getenv("WRITER_OUT_DIR", "/workspace/mrd69/out/writing")

# Crypto (opcjonalnie)
CRYPTO_API_BASE = os.getenv("CRYPTO_API_BASE", "")
CRYPTO_API_KEY = os.getenv("CRYPTO_API_KEY", "")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
