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
WEB_HTTP_TIMEOUT = int(os.getenv("WEB_HTTP_TIMEOUT", "40"))
HTTP_TIMEOUT = WEB_HTTP_TIMEOUT
WEB_USER_AGENT = os.getenv("WEB_USER_AGENT", "Mrd69/1.0")
LLM_HTTP_TIMEOUT = int(os.getenv("LLM_HTTP_TIMEOUT_S", str(WEB_HTTP_TIMEOUT)))

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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# RUNPOD
USE_RUNPOD = os.getenv("USE_RUNPOD", "0") in ("1", "true", "True")
RUNPOD_PERSIST_DIR = os.getenv("RUNPOD_PERSIST_DIR", "/runpod/persist")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")
RUNPOD_SYNC_INTERVAL = int(os.getenv("RUNPOD_SYNC_INTERVAL", "1800"))

# Travel / maps / weather (opcjonalnie)
GOOGLE_MAPS_KEY = os.getenv("GOOGLE_MAPS_KEY", "")
TRIPADVISOR_KEY = os.getenv("TRIPADVISOR_KEY", "")
OPENTRIPMAP_KEY = os.getenv("OPENTRIPMAP_KEY", "")
TRANSITLAND_API_KEY = os.getenv("TRANSITLAND_API_KEY", "")
TRAVEL_PARTNER_API = os.getenv("TRAVEL_PARTNER_API", "")
MAPS_STATIC_API = os.getenv("MAPS_STATIC_API", "")
MAPTILER_KEY = os.getenv("MAPTILER_KEY", "")
XWEATHER_APP_ID = os.getenv("XWEATHER_APP_ID", "")
XWEATHER_SECRET = os.getenv("XWEATHER_SECRET", "")
TRAVEL_CACHE_TTL_MIN = int(os.getenv("TRAVEL_CACHE_TTL_MIN", "120"))
TRAVEL_FOOD_PREFS = os.getenv("TRAVEL_FOOD_PREFS", "")
TRAVEL_HOTEL_PREFS = os.getenv("TRAVEL_HOTEL_PREFS", "")
TRAVEL_GOOGLE_TTL = int(os.getenv("TRAVEL_GOOGLE_TTL", "3600"))
TRAVEL_TA_TTL = int(os.getenv("TRAVEL_TA_TTL", "7200"))
TRAVEL_OTM_TTL = int(os.getenv("TRAVEL_OTM_TTL", "86400"))
TRAVEL_WIKI_TTL = int(os.getenv("TRAVEL_WIKI_TTL", "604800"))
TRAVEL_OSM_TTL = int(os.getenv("TRAVEL_OSM_TTL", "86400"))
TRAVEL_WEATHER_TTL = int(os.getenv("TRAVEL_WEATHER_TTL", "3600"))
TRAVEL_TRANSIT_TTL = int(os.getenv("TRAVEL_TRANSIT_TTL", "3600"))
TRAVEL_FLIGHTS_TTL = int(os.getenv("TRAVEL_FLIGHTS_TTL", "1800"))

# Files/IO
WRITER_OUT_DIR = os.getenv("WRITER_OUT_DIR", "/workspace/mrd69/out/writing")
DEV_OUT_DIR = os.getenv("DEV_OUT_DIR", "/workspace/mrd69/out/dev")

# Crypto (opcjonalnie)
CRYPTO_API_BASE = os.getenv("CRYPTO_API_BASE", "")
CRYPTO_API_KEY = os.getenv("CRYPTO_API_KEY", "")
CRYPTO_BASE_FIAT = os.getenv("CRYPTO_BASE_FIAT", "USD")
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY", "")
COINGECKO_VS = os.getenv("COINGECKO_VS", "usd")

# Misc
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
SONAR_TOKEN = os.getenv("SONAR_TOKEN", "")
