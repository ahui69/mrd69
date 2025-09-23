import os, sys
from pathlib import Path

DOTENV_PATH = "/workspace/mrd69/.env"

# Ładuj .env jeśli jest, bez krzyczenia gdy brak
try:
    from dotenv import load_dotenv

    if Path(DOTENV_PATH).exists():
        load_dotenv(dotenv_path=DOTENV_PATH, override=False)
except Exception:
    pass

# Sensowne domyślne (nie nadpisują .env)
os.environ.setdefault("LLM_BASE_URL", "https://api.deepinfra.com/v1/openai")
os.environ.setdefault("LLM_MODEL", "zai-org/GLM-4.5-Air")

# Upewnij się, że root projektu jest na ścieżce importu
ROOT = "/workspace/mrd69"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
