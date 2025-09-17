# config.py
import os
from dotenv import load_dotenv

# Ładuj zmienne z pliku .env na samym początku
load_dotenv()

# === Konfiguracja Aplikacji ===
APP_TITLE: str = "MORDZIX CORE v2.0"
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8080"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# === Konfiguracja Pamięci ===
CONTEXT_CHARS_LIMIT: int = int(os.getenv("CONTEXT_CHARS_LIMIT", "4000"))
CONTEXT_TOP_K: int = int(os.getenv("CONTEXT_TOP_K", "5"))

# === Konfiguracja LLM ===
SYS_PROMPT: str = os.getenv("SYS_PROMPT", "Jesteś pomocnym asystentem AI.")
CHAT_MAX_TOKENS: int = int(os.getenv("CHAT_MAX_TOKENS", "2048"))
CHAT_TEMPERATURE: float = float(os.getenv("CHAT_TEMPERATURE", "0.7"))

# === Integracje ===
USE_RUNPOD: bool = os.getenv("USE_RUNPOD", "False").lower() in ("true", "1", "t")

# === Bezpieczeństwo ===
# W produkcji ustaw na jawną listę domen, np. "https://twoja-domena.com,http://localhost:3000"
CORS_ALLOWED_ORIGINS: list[str] = os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")
