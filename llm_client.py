import httpx
import os
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger as log
from circuitbreaker import circuit
from openai import OpenAI, APIError, APITimeoutError

# --- Konfiguracja ---
# Używamy zmiennych, które już masz, z domyślnymi wartościami dla Kimi
KIMI_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepinfra.com/v1/openai")
KIMI_API_KEY = os.getenv("LLM_API_KEY")
KIMI_MODEL = os.getenv("LLM_MODEL", "moonshotai/Kimi-K2-Instruct-0905")

# W przyszłości można dodać fallback do innego modelu Kimi lub innego dostawcy
# FALLBACK_BASE_URL = ...
# FALLBACK_API_KEY = ...
# FALLBACK_MODEL = ...

HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", 45))

# --- Circuit Breaker ---
# Jeśli 3 zapytania pod rząd się nie udadzą, wyłącz klienta na 60 sekund
@circuit(failure_threshold=3, recovery_timeout=60)
@retry(wait=wait_exponential(multiplier=1, min=1, max=5), stop=stop_after_attempt(3))
def _make_request(client: OpenAI, payload: dict) -> str:
    """
    Wykonuje zapytanie do API Kimi z ponawianiem i circuit breakerem.
    """
    try:
        chat_completion = client.chat.completions.create(**payload)
        return chat_completion.choices[0].message.content or ""
    except APITimeoutError as e:
        log.error(f"Timeout podczas połączenia z Kimi: {e}", model=payload.get("model"))
        raise
    except APIError as e:
        log.error(f"Błąd API Kimi: {e.status_code} - {e.message}", model=payload.get("model"))
        raise
    except Exception as e:
        log.error(f"Nieoczekiwany błąd podczas połączenia z Kimi: {e}", model=payload.get("model"))
        raise

class LLMClient:
    def __init__(self):
        self.kimi_client = OpenAI(
            api_key=KIMI_API_KEY,
            base_url=KIMI_BASE_URL,
            timeout=HTTP_TIMEOUT,
        )

    def chat(self, messages: list[dict], max_tokens: int = 2048, temperature: float = 0.7) -> str:
        """
        Główna funkcja do komunikacji z Kimi, opakowana w mechanizmy niezawodności.
        """
        if not KIMI_API_KEY:
            log.error("Brak klucza LLM_API_KEY. Nie można połączyć z Kimi.")
            return "Błąd konfiguracji: Brak klucza API."

        payload = {
            "model": KIMI_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            return _make_request(self.kimi_client, payload)
        except Exception as e:
            log.error(f"Nie udało się uzyskać odpowiedzi od Kimi po kilku próbach. Błąd: {e}")
            # W przyszłości tutaj można dodać logikę fallback
            return "Przepraszam, wystąpił błąd. Spróbuj ponownie później."

# Singleton
llm_client = LLMClient()
