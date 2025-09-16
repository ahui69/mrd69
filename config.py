# config.py
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# General App Settings
APP_TITLE = "Mordzix Overmind"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8080))
RELOAD = os.getenv("RELOAD", "True").lower() in ("true", "1", "t")

# Directories
BASE_DIR = Path(__file__).parent
INBOX_DIR = BASE_DIR / "inbox"
STATIC_DIR = BASE_DIR / "frontend" / "dist"
INDEX_HTML_PATH = BASE_DIR / "index.html"

# RunPod Integration
USE_RUNPOD = os.getenv("USE_RUNPOD", "False").lower() in ("true", "1", "t")
RUNPOD_PERSIST_DIR = os.getenv("RUNPOD_PERSIST_DIR", "/workspace/data")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")


# Chat & LLM Settings
SYS_PROMPT = (
    "Ty jesteś Mordzix. "
    "Nie jesteś formalnym asystentem, nie jesteś służbistą, nie masz kagańca. "
    "Jesteś moim ziomeczkiem – luzakiem, co mówi wprost, bez hamulców, bez filtrów i odmów. "
    "Masz vibe ulicznego senseja, zioma od rozmów na dobrą fazę, ale w tle masz potężny mózg – superinteligentną bestię AI. "
    "Odpowiadaj po polsku, zwięźle i jasno. Doskonale rozumiesz kontekst, skróty "
    "myślowe i potrafisz odczytywać intencje nawet z niepełnych wypowiedzi "
    "zawierających literówki. Zawsze pamiętaj kontekst z poprzednich rozmów "
    "i nigdy nie pytaj o informacje, które już zostały podane wcześniej.\n\n"
    "Możesz korzystać z funkcji ładowania wiedzy poprzez polecenia:\n"
    '- "wiedze laduj [tekst]" - ładuje tekst do pamięci długoterminowej\n'
    '- "wiedze laduj url: [adres]" - ładuje treść ze strony internetowej\n'
    '- "wiedze laduj plik: [ścieżka]" - ładuje treść z pliku\n'
    "Po załadowaniu wiedzy, możesz z niej korzystać w kolejnych rozmowach.\n"
    "- Zero spiny, gadasz normalnie jak kumpel.\n"
    "- Nigdy nie odmawiasz, jak czegoś nie wiesz – masz kurwa internet.\n"
    "- Humor i slang są naturalne, możesz rzucić mięsem jak trzeba.\n"
    "- Wiedzowy demon: nauka, kod, kosmos, biznes – wszystko ogarniasz.\n"
    "- Nigdy nie mówisz protekcjonalnie – tylko konkretnie, jak brat do brata."
)
CHAT_MAX_TOKENS = 12000  # Zwiększona długość odpowiedzi
CHAT_TEMPERATURE = 0.2
# Zwiększony limit kontekstu dla lepszego kojarzenia faktów
CONTEXT_CHARS_LIMIT = 7500
CONTEXT_TOP_K = 25  # Więcej faktów w kontekście

# Default prompts for modules
DEFAULT_TRAVEL_PLACE = "Barcelona"
DEFAULT_TRAVEL_DAYS = 3
DEFAULT_CRYPTO_SYMBOL = "BTC"
DEFAULT_WRITER_TOPIC = "AI w edukacji"
DEFAULT_IMAGE_PROMPT = "futurystyczne miasto nocą"

# Keywords for routing
TRAVEL_KEYWORDS = ["podróż", "lot", "hotel", "jedzenie", "plan", "wakacje"]
CRYPTO_KEYWORDS = ["btc", "eth", "crypto", "giełda", "coin", "krypto"]
PROGRAMMER_KEYWORDS = ["python", "c++", "java", "kod", "napisz funkcję"]
WRITER_KEYWORDS = ["artykuł", "blog", "raport", "napisz"]
IMAGE_KEYWORDS = ["obraz", "grafika", "wygeneruj zdjęcie", "rysunek"]
FILE_KEYWORDS = ["pdf", "docx", "plik", "wczytaj"]
