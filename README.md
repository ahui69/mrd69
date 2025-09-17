# MORDZIX CORE v3.0 - Zaawansowany System Pamięci

**MORDZIX CORE** to platforma AI nowej generacji, wyposażona w rewolucyjny, 8-modułowy system pamięci, który naśladuje ludzkie procesy poznawcze. Koniec z amnezją. Czas na AI, która rozumie, kojarzy i uczy się jak człowiek.

---

## 🧠 Kluczowe Funkcje Zaawansowanej Pamięci

1.  **Pamięć Epizodyczna (Timeline):** Każda interakcja jest zapisywana jako wpis w dzienniku, tworząc bogatą historię Waszej współpracy.
2.  **Pamięć Kontekstowa:** Różne "mózgi" dla różnych zadań (kodowanie, pisanie, biznes), każdy z własnymi priorytetami i wiedzą.
3.  **Samorefleksja:** AI analizuje swoje działania, uczy się z błędów i tworzy notatki "co robić lepiej", aby unikać powtarzania tych samych pomyłek.
4.  **Pamięć Emocjonalna:** Wykrywa nastrój użytkownika i dostosowuje styl komunikacji – od ultra-konkretnych komend po luźną rozmowę.
5.  **Pamięć Plikowa:** Zapamiętuje pliki, grafiki i diagramy, które razem przerabialiście, i potrafi do nich wracać.
6.  **Pamięć Predykcyjna:** Uczy się Twoich wzorców zachowań, aby przewidywać, czego zaraz będziesz potrzebować.
7.  **Wersjonowanie Pamięci:** "Git dla pamięci" – twórz snapshoty i przywracaj stan pamięci do dowolnego momentu w czasie.
8.  **Mapowanie Relacji:** Tworzy graf powiązań między osobami i projektami, rozumiejąc, kto jest kim w Twoim świecie.

## 🚀 Uruchomienie

Projekt wykorzystuje `Makefile` do zarządzania.

1.  **Instalacja:**
    ```bash
    make setup
    ```
    *Ta komenda stworzy środowisko wirtualne, zainstaluje wszystkie zależności z `pyproject.toml` i skonfiguruje pre-commit hooks.*

2.  **Uruchomienie serwera deweloperskiego:**
    ```bash
    make dev
    ```
    *Serwer FastAPI uruchomi się z auto-reload na porcie 8080.*

## 🛠️ Dostępne Komendy

-   `make dev`: Uruchom serwer deweloperski.
-   `make lint`: Sprawdź jakość kodu (ruff, mypy).
-   `make test`: Uruchom testy (pytest).
-   `make seed`: Załaduj dane początkowe do pamięci.
-   `make setup`: Zainstaluj projekt.

##  API Endpoints

Aplikacja udostępnia bogate API do zarządzania zaawansowaną pamięcią. Pełna dokumentacja jest dostępna pod `/docs` po uruchomieniu serwera.

### Przykładowe Endpoints:

-   `GET /memory/timeline`: Pobierz ostatnie wpisy z timeline'u.
-   `POST /memory/context`: Zaktualizuj pamięć kontekstową.
-   `POST /memory/backup`: Stwórz backup całej pamięci.
-   `GET /memory/persons/{name}`: Pobierz profil osoby.
-   `GET /health`: Sprawdź stan aplikacji.
-   `GET /version`: Sprawdź wersję i hash commita.

---

*Ten projekt to krok w stronę prawdziwej, świadomej AI. Rozwijajmy go dalej.*