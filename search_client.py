"""
search_client.py - Moduł do wyszukiwania zewnętrznej wiedzy

Integracja z SerpApi do wyszukiwania informacji oraz formater wyników
dla lepszej integracji z kontekstem LLM.
"""

import logging
import os
import re
import time

import requests

# SerpAPI key
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

# Konfiguracja
MAX_RETRIES = 3
RETRY_DELAY = 2  # w sekundach
REQUEST_TIMEOUT = 10  # w sekundach

# Utworzenie loggera
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Czyści tekst z nadmiarowych białych znaków i HTML."""
    if not text:
        return ""
    # Usunięcie tagów HTML
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalizacja białych znaków
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def search_web(query: str, num_results: int = 5) -> list[dict[str, str]]:
    """
    Wyszukuje informacje w internecie za pomocą SerpApi.

    Args:
        query: Zapytanie do wyszukania
        num_results: Liczba wyników do zwrócenia

    Returns:
        Lista słowników z wynikami wyszukiwania, każdy zawierający:
        - title: Tytuł strony
        - link: URL do strony
        - snippet: Fragment tekstu ze strony
    """
    if not SERPAPI_KEY:
        logger.warning("Brak klucza SerpAPI. Wyszukiwanie nie jest możliwe.")
        return []

    params = {
        "engine": "google",
        "q": query,
        "num": str(max(1, min(10, num_results))),
        "api_key": SERPAPI_KEY,
        "hl": "pl",  # język polski
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                "https://serpapi.com/search.json", params=params, timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 200:
                data = response.json()
                results = []

                # Przetwarzanie wyników organicznych
                for result in (data.get("organic_results") or [])[:num_results]:
                    results.append(
                        {
                            "title": clean_text(result.get("title", "")),
                            "link": result.get("link", ""),
                            "snippet": clean_text(result.get("snippet", "")),
                        }
                    )

                # Dodanie featured snippet, jeśli istnieje
                if data.get("answer_box") and len(results) < num_results:
                    answer = data.get("answer_box", {})
                    if "answer" in answer:
                        results.insert(
                            0,
                            {
                                "title": "Szybka odpowiedź",
                                "link": answer.get("link", ""),
                                "snippet": clean_text(answer.get("answer", "")),
                            },
                        )
                    elif "snippet" in answer:
                        results.insert(
                            0,
                            {
                                "title": clean_text(answer.get("title", "Szybka odpowiedź")),
                                "link": answer.get("link", ""),
                                "snippet": clean_text(answer.get("snippet", "")),
                            },
                        )

                return results

            else:
                logger.error(f"Błąd SerpAPI: {response.status_code}, {response.text}")

        except requests.RequestException as e:
            logger.error(f"Błąd podczas wyszukiwania: {e}")

        # Retry po błędzie
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)

    return []


def format_search_results(results: list[dict[str, str]], query: str) -> str:
    """
    Formatuje wyniki wyszukiwania w czytelny format dla LLM.

    Args:
        results: Lista wyników wyszukiwania
        query: Oryginalne zapytanie

    Returns:
        Sformatowany tekst z wynikami wyszukiwania
    """
    if not results:
        return "Nie znaleziono wyników wyszukiwania."

    formatted_text = f'WYNIKI WYSZUKIWANIA DLA: "{query}"\n\n'

    for i, result in enumerate(results, 1):
        formatted_text += f"{i}. {result['title']}\n"
        formatted_text += f"   {result['link']}\n"
        formatted_text += f"   {result['snippet']}\n\n"

    return formatted_text


def search_knowledge(query: str, num_results: int = 5) -> str:
    """
    Główna funkcja do wyszukiwania wiedzy, gotowa do użycia w kontekście LLM.

    Args:
        query: Zapytanie do wyszukania
        num_results: Liczba wyników do zwrócenia

    Returns:
        Sformatowany tekst z wynikami wyszukiwania
    """
    results = search_web(query, num_results)
    return format_search_results(results, query)


def should_search_web(query: str) -> bool:
    """
    Określa czy zapytanie powinno być wyszukane w internecie.

    Args:
        query: Zapytanie od użytkownika

    Returns:
        True jeśli zapytanie powinno być wyszukane, False w przeciwnym przypadku
    """
    # Lista słów kluczowych sugerujących potrzebę wyszukiwania
    search_keywords = [
        "kto",
        "co",
        "gdzie",
        "kiedy",
        "jak",
        "dlaczego",
        "wyjaśnij",
        "definicja",
        "znaczenie",
        "aktualny",
        "ostatni",
        "najnowszy",
        "wydarzenie",
        "wiadomości",
        "informacja",
        "dane",
        "szukaj",
        "znajdź",
        "wyszukaj",
        "sprawdź",
        "dowiedz się",
        "informacje o",
        "co to jest",
        "kim jest",
        "jaki jest",
        "ile kosztuje",
        "cena",
        "historia",
        "kiedy powstał",
        "kiedy urodził się",
        "kiedy zmarł",
        "jak działa",
    ]

    # Czy zawiera którekolwiek ze słów kluczowych (case insensitive)
    query_lower = query.lower()
    for keyword in search_keywords:
        if re.search(r"\b" + re.escape(keyword) + r"\b", query_lower):
            return True

    return False
