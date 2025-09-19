# config.py
import os
from dotenv import load_dotenv

# === Ładowanie .env ===
# Ważne: plik w katalogu głównym MUSI nazywać się ".env"
load_dotenv()

# ========== APLIKACJA ==========
APP_TITLE: str = os.getenv("APP_TITLE", "MORDZIX CORE v2.0")
APP_VERSION: str = os.getenv("APP_VERSION", "3.2.0")
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8080"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# ========== PAMIĘĆ ==========
CONTEXT_CHARS_LIMIT: int = int(os.getenv("CONTEXT_CHARS_LIMIT", "4000"))
CONTEXT_TOP_K: int = int(os.getenv("CONTEXT_TOP_K", "5"))
MEM_DIR: str = os.getenv("MEM_DIR", "data")
LTM_DB_PATH: str = os.getenv("LTM_DB_PATH", os.path.join(MEM_DIR, "memory.db"))

# ========== LLM ==========
SYS_PROMPT: str = os.getenv("SYS_PROMPT", "Jesteś pomocnym asystentem AI.")
CHAT_MAX_TOKENS: int = int(os.getenv("CHAT_MAX_TOKENS", "2048"))
CHAT_TEMPERATURE: float = float(os.getenv("CHAT_TEMPERATURE", "0.7"))

LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "60"))

# ========== RUNPOD ==========
USE_RUNPOD: bool = os.getenv("USE_RUNPOD", "False").lower() in ("true", "1", "t")
RUNPOD_API_KEY: str = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_REGION: str = os.getenv("RUNPOD_REGION", "eu-central")
RUNPOD_POD_ID: str = os.getenv("RUNPOD_POD_ID", "")
RUNPOD_PERSIST_DIR: str = os.getenv("RUNPOD_PERSIST_DIR", "runpod_data")

# ========== BEZPIECZEŃSTWO ==========
AUTH_TOKEN: str = os.getenv("AUTH_TOKEN", "")
CORS_ALLOWED_ORIGINS: list[str] = os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")

# ========== INNE ==========
DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "0") in ("1", "true", "True")
