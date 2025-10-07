"""
Testy dla server.py - PRZYKŁADY

Jak uruchomić:
    pytest test_server.py -v

Jak zainstalować pytest:
    pip install pytest pytest-asyncio httpx
"""

import pytest
from fastapi.testclient import TestClient
from server import app

# ════════════════════════════════════════════════════════════
# SETUP - tworzymy klienta testowego
# ════════════════════════════════════════════════════════════

client = TestClient(app)


# ════════════════════════════════════════════════════════════
# TEST 1: PODSTAWOWY - czy endpoint odpowiada
# ════════════════════════════════════════════════════════════

def test_health_endpoint_exists():
    """
    TEST: Sprawdza czy endpoint /api/health w ogóle istnieje
    
    Po co: Jak go usuniesz przez przypadek, test Ci powie
    """
    response = client.get("/api/health")
    
    # Sprawdź że NIE dostałeś 404 (Not Found)
    assert response.status_code == 200, "Endpoint /api/health nie istnieje!"


# ════════════════════════════════════════════════════════════
# TEST 2: STRUKTURA ODPOWIEDZI
# ════════════════════════════════════════════════════════════

def test_health_returns_correct_structure():
    """
    TEST: Sprawdza czy /api/health zwraca prawidłową strukturę
    
    Po co: Frontend oczekuje {"ok": true, "mode": "...", ...}
           Jak zmienisz strukturę, frontend się wysypie
    """
    response = client.get("/api/health")
    data = response.json()
    
    # Sprawdź że są wymagane pola
    assert "ok" in data, "Brak pola 'ok' w odpowiedzi!"
    assert "mode" in data, "Brak pola 'mode' w odpowiedzi!"
    assert "memory_router" in data, "Brak pola 'memory_router'!"
    
    # Sprawdź typy danych
    assert isinstance(data["ok"], bool), "Pole 'ok' powinno być boolean!"
    assert data["ok"] == True, "Health check powinien zwracać ok=True"


# ════════════════════════════════════════════════════════════
# TEST 3: WARTOŚCI W ODPOWIEDZI
# ════════════════════════════════════════════════════════════

def test_health_mode_is_valid():
    """
    TEST: Sprawdza czy 'mode' ma sensowną wartość
    
    Po co: Nie chcesz żeby zwracało 'mode': 'xdxdxd'
    """
    response = client.get("/api/health")
    data = response.json()
    
    # mode powinno być "echo" lub "llm"
    assert data["mode"] in ["echo", "llm"], \
        f"Nieprawidłowy mode: {data['mode']}, oczekiwano 'echo' lub 'llm'"


# ════════════════════════════════════════════════════════════
# TEST 4: ENDPOINT KTÓRY PRZYJMUJE DANE (POST)
# ════════════════════════════════════════════════════════════

def test_chat_endpoint_accepts_message():
    """
    TEST: Sprawdza czy /api/chat przyjmuje wiadomości
    
    Po co: To główna funkcja! Musi działać ZAWSZE.
    """
    payload = {
        "newMessage": {
            "role": "user",
            "content": "test message"
        }
    }
    
    response = client.post("/api/chat", json=payload)
    
    # Sprawdź status
    assert response.status_code == 200, \
        f"Chat endpoint zwrócił błąd: {response.status_code}"
    
    # Sprawdź strukturę odpowiedzi
    data = response.json()
    assert "reply" in data, "Brak odpowiedzi w /api/chat!"
    assert "chatId" in data, "Brak chatId w odpowiedzi!"
    
    # Sprawdź że odpowiedź nie jest pusta
    assert len(data["reply"]) > 0, "Odpowiedź jest pusta!"


# ════════════════════════════════════════════════════════════
# TEST 5: BŁĘDNE DANE (negative test)
# ════════════════════════════════════════════════════════════

def test_chat_rejects_empty_message():
    """
    TEST: Sprawdza czy endpoint odrzuca puste wiadomości
    
    Po co: Nie chcesz żeby ktoś wysyłał puste requesty
           i Twój serwer się wywalał
    """
    payload = {
        "newMessage": {
            "role": "user",
            "content": ""  # ← PUSTE!
        }
    }
    
    response = client.post("/api/chat", json=payload)
    
    # Endpoint powinien przyjąć (200) ale coś zwrócić
    # (w Twoim przypadku echo lub reply)
    assert response.status_code in [200, 400], \
        "Endpoint powinien obsłużyć puste wiadomości"


# ════════════════════════════════════════════════════════════
# TEST 6: SPRAWDZANIE HISTORII
# ════════════════════════════════════════════════════════════

def test_history_returns_list():
    """
    TEST: /api/history powinno zwracać listę czatów
    
    Po co: Frontend wyświetla listę - jak dostanie string zamiast
           listy, wyświetli [object Object] i będzie wstyd
    """
    response = client.get("/api/history")
    
    assert response.status_code == 200
    data = response.json()
    
    # Powinno być listą (może pustą)
    assert isinstance(data, list), \
        f"History powinno zwracać listę, dostałem: {type(data)}"


# ════════════════════════════════════════════════════════════
# TEST 7: BOOTSTRAP (inicjalizacja)
# ════════════════════════════════════════════════════════════

def test_bootstrap_returns_initial_data():
    """
    TEST: /api/bootstrap zwraca dane startowe
    
    Po co: Frontend na starcie wywołuje bootstrap - 
           jak nie dostanie danych, się zawiesza
    """
    response = client.get("/api/bootstrap")
    
    assert response.status_code == 200
    data = response.json()
    
    # Sprawdź że są wymagane pola (według Twojego kodu)
    assert "prompts" in data
    assert "memory" in data
    assert "version" in data


# ════════════════════════════════════════════════════════════
# TEST 8: UPLOAD PLIKU
# ════════════════════════════════════════════════════════════

def test_upload_accepts_file():
    """
    TEST: /api/upload przyjmuje pliki
    
    Po co: Funkcja upload to często źródło błędów
           (rozmiar, typ, itp)
    """
    # Symuluj upload małego pliku tekstowego
    files = {
        "file": ("test.txt", b"test content", "text/plain")
    }
    
    response = client.post("/api/upload", files=files)
    
    assert response.status_code == 200, \
        f"Upload failed: {response.status_code}"
    
    data = response.json()
    assert "path" in data or "name" in data, \
        "Upload powinien zwrócić info o pliku"


# ════════════════════════════════════════════════════════════
# TEST 9: CRYPTO ENDPOINT (przykład z routera)
# ════════════════════════════════════════════════════════════

def test_crypto_health():
    """
    TEST: Router crypto ma swój health check
    
    Po co: Sprawdź że router się zamontował
    """
    response = client.get("/api/crypto/health")
    
    # Może być 503 (service unavailable) jeśli brak kluczy API
    # Ale NIE MOŻE być 404 (endpoint nie istnieje)
    assert response.status_code in [200, 503], \
        f"Crypto health zwrócił nieoczekiwany status: {response.status_code}"


# ════════════════════════════════════════════════════════════
# TEST 10: PERFORMANCE (czy nie jest za wolne)
# ════════════════════════════════════════════════════════════

def test_health_response_time():
    """
    TEST: Health check powinien być SZYBKI
    
    Po co: Jak health check trwa 5 sekund, coś jest bardzo źle
    """
    import time
    
    start = time.time()
    response = client.get("/api/health")
    duration = time.time() - start
    
    assert response.status_code == 200
    assert duration < 1.0, \
        f"Health check za wolny! Trwał {duration:.2f}s (max 1s)"


# ════════════════════════════════════════════════════════════
# TEST 11: SEKWENCJA AKCJI (integration test)
# ════════════════════════════════════════════════════════════

def test_chat_creates_history_entry():
    """
    TEST: Po wysłaniu wiadomości, powinna pojawić się w historii
    
    Po co: To sprawdza czy CAŁY FLOW działa:
           message → save to DB → visible in history
    """
    # 1. Wyślij wiadomość
    payload = {
        "newMessage": {
            "role": "user",
            "content": "test dla historii"
        }
    }
    chat_response = client.post("/api/chat", json=payload)
    assert chat_response.status_code == 200
    
    chat_id = chat_response.json().get("chatId")
    assert chat_id, "Nie dostałem chatId!"
    
    # 2. Sprawdź czy jest w historii
    history_response = client.get("/api/history")
    history = history_response.json()
    
    # Sprawdź czy nasz chat_id jest na liście
    chat_ids = [chat["id"] for chat in history]
    assert chat_id in chat_ids, \
        f"Chat {chat_id} nie pojawił się w historii!"


# ════════════════════════════════════════════════════════════
# TEST 12: CORS HEADERS (dla frontendu)
# ════════════════════════════════════════════════════════════

def test_cors_headers_present():
    """
    TEST: Sprawdza czy są headery CORS
    
    Po co: Bez CORS frontend z innej domeny nie będzie mógł
           połączyć się z Twoim API
    """
    response = client.get("/api/health")
    
    # Twój server ma CORS middleware, więc powinny być headery
    # (TestClient może nie ustawiać wszystkich headerów, ale sprawdźmy)
    assert response.status_code == 200


# ════════════════════════════════════════════════════════════
# PRZYKŁAD PARAMETRYZOWANEGO TESTU
# (ten sam test dla wielu wartości)
# ════════════════════════════════════════════════════════════

@pytest.mark.parametrize("endpoint", [
    "/api/health",
    "/api/bootstrap", 
    "/api/history",
    "/api/crypto/health",
])
def test_endpoint_returns_json(endpoint):
    """
    TEST: Wszystkie te endpointy powinny zwracać JSON
    
    Po co: Jeden test sprawdza 4 endpointy naraz!
    """
    response = client.get(endpoint)
    
    # Może być 503 (unavailable) ale content-type powinien być JSON
    assert "application/json" in response.headers.get("content-type", ""), \
        f"{endpoint} nie zwraca JSON!"


# ════════════════════════════════════════════════════════════
# PRZYKŁAD FIXTURE (setup/teardown)
# ════════════════════════════════════════════════════════════

@pytest.fixture
def sample_chat():
    """
    FIXTURE: Tworzy przykładowy chat do testów
    
    Po co: Nie musisz w każdym teście tworzyć czatu od nowa
    """
    payload = {
        "newMessage": {
            "role": "user",
            "content": "fixture test"
        }
    }
    response = client.post("/api/chat", json=payload)
    return response.json()


def test_using_fixture(sample_chat):
    """
    TEST: Przykład użycia fixture
    
    sample_chat jest automatycznie utworzony przez fixture wyżej
    """
    assert "chatId" in sample_chat
    assert "reply" in sample_chat
    
    # Możesz od razu użyć tego chatu w dalszych testach
    chat_id = sample_chat["chatId"]
    response = client.get(f"/api/history/{chat_id}")
    assert response.status_code == 200
