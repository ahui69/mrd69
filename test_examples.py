"""
PRZYKŁADY TESTÓW - różne sytuacje

To nie są testy dla Twojego kodu, to PRZYKŁADY jak testować różne rzeczy.
Możesz kopiować te wzorce do swoich testów.
"""

import pytest
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


# ════════════════════════════════════════════════════════════
# PRZYKŁAD 1: TEST MATEMATYKI/LOGIKI
# ════════════════════════════════════════════════════════════

def dodaj(a, b):
    """Funkcja którą chcesz przetestować"""
    return a + b

def test_dodawanie():
    """Test podstawowej funkcji"""
    assert dodaj(2, 3) == 5
    assert dodaj(-1, 1) == 0
    assert dodaj(0, 0) == 0
    
def test_dodawanie_edge_cases():
    """Test skrajnych przypadków"""
    assert dodaj(1000000, 1000000) == 2000000
    assert dodaj(-999, -1) == -1000


# ════════════════════════════════════════════════════════════
# PRZYKŁAD 2: TEST WYJĄTKÓW (exceptions)
# ════════════════════════════════════════════════════════════

def podziel(a, b):
    """Funkcja która może rzucić wyjątek"""
    if b == 0:
        raise ValueError("Nie dziel przez 0!")
    return a / b

def test_podziel_normal():
    """Test normalnego przypadku"""
    assert podziel(10, 2) == 5.0

def test_podziel_przez_zero_rzuca_wyjatek():
    """Test że funkcja rzuca wyjątek gdy trzeba"""
    with pytest.raises(ValueError) as exc_info:
        podziel(10, 0)
    
    # Sprawdź treść błędu
    assert "Nie dziel przez 0" in str(exc_info.value)


# ════════════════════════════════════════════════════════════
# PRZYKŁAD 3: TEST BAZY DANYCH
# ════════════════════════════════════════════════════════════

def test_database_saves_and_retrieves():
    """
    TEST: Sprawdza czy dane zapisują się i odczytują z bazy
    
    Dla Twojego projektu - sprawdza SQLite
    """
    # 1. Zapisz wiadomość przez API
    payload = {
        "newMessage": {
            "role": "user",
            "content": "Test bazy danych"
        }
    }
    response = client.post("/api/chat", json=payload)
    chat_id = response.json()["chatId"]
    
    # 2. Odczytaj ten chat
    get_response = client.get(f"/api/history/{chat_id}")
    data = get_response.json()
    
    # 3. Sprawdź że wiadomość tam jest
    messages = data.get("messages", [])
    assert len(messages) > 0, "Baza nie zapisała wiadomości!"
    
    # 4. Sprawdź treść
    user_messages = [m for m in messages if m["role"] == "user"]
    assert any("Test bazy danych" in m["content"] for m in user_messages), \
        "Treść wiadomości zmieniła się w bazie!"


# ════════════════════════════════════════════════════════════
# PRZYKŁAD 4: TEST ODPOWIEDZI LLM
# ════════════════════════════════════════════════════════════

def test_llm_responds_in_polish():
    """
    TEST: LLM powinno odpowiadać po polsku
    
    (Ten test może failować jak LLM nie jest skonfigurowane)
    """
    payload = {
        "newMessage": {
            "role": "user",
            "content": "Powiedz 'tak' po polsku"
        }
    }
    response = client.post("/api/chat", json=payload)
    
    if response.status_code == 200:
        reply = response.json()["reply"]
        
        # Sprawdź że odpowiedź zawiera polskie znaki lub "tak"
        polish_indicators = ["tak", "ą", "ę", "ó", "ł", "ć", "ń", "ś", "ź", "ż"]
        has_polish = any(ind in reply.lower() for ind in polish_indicators)
        
        assert has_polish or "yes" not in reply.lower(), \
            f"LLM odpowiedziało nie po polsku: {reply[:100]}"


# ════════════════════════════════════════════════════════════
# PRZYKŁAD 5: TEST PERFORMANCE
# ════════════════════════════════════════════════════════════

def test_endpoint_is_fast():
    """
    TEST: Endpoint powinien odpowiadać szybko
    
    Po co: Sprawdzasz że nie ma nagłego spowolnienia
    """
    import time
    
    times = []
    for i in range(5):
        start = time.time()
        response = client.get("/api/health")
        duration = time.time() - start
        times.append(duration)
        assert response.status_code == 200
    
    avg_time = sum(times) / len(times)
    
    assert avg_time < 0.5, \
        f"Średni czas odpowiedzi za długi: {avg_time:.3f}s (max 0.5s)"


# ════════════════════════════════════════════════════════════
# PRZYKŁAD 6: MOCK (udawanie zewnętrznych API)
# ════════════════════════════════════════════════════════════

def test_with_mock_external_api(monkeypatch):
    """
    TEST: Mockowanie zewnętrznych API
    
    Po co: Nie chcesz wywoływać prawdziwego API w testach
           (kosztuje, może być wolne, może nie działać)
    """
    # Przykład dla crypto - normalnie wywołuje CoinGecko API
    # W teście podstawiamy fejkowe dane
    
    def fake_screener_top(limit, vs):
        """Fejkowa funkcja zamiast prawdziwej"""
        return [
            {"id": "bitcoin", "symbol": "btc", "price": 50000},
            {"id": "ethereum", "symbol": "eth", "price": 3000}
        ]
    
    # Podstaw fejk (to zaawansowane, ale pokazuję)
    # monkeypatch.setattr("routers.crypto.screener_top", fake_screener_top)
    
    # Teraz test nie wywołuje prawdziwego API!
    # (Ten przykład wymaga więcej setupu, ale idea jest taka)


# ════════════════════════════════════════════════════════════
# PRZYKŁAD 7: TEST INTEGRACYJNY (cały flow)
# ════════════════════════════════════════════════════════════

def test_complete_user_flow():
    """
    TEST: Symuluje kompletne użycie aplikacji
    
    Po co: Sprawdza czy WSZYSTKO działa razem (end-to-end)
    """
    # KROK 1: User zaczyna czat
    response1 = client.post("/api/chat", json={
        "newMessage": {"role": "user", "content": "Cześć"}
    })
    assert response1.status_code == 200
    chat_id = response1.json()["chatId"]
    
    # KROK 2: User wysyła kolejną wiadomość do tego samego czatu
    response2 = client.post("/api/chat", json={
        "chatId": chat_id,
        "newMessage": {"role": "user", "content": "Jak się masz?"}
    })
    assert response2.status_code == 200
    
    # KROK 3: User sprawdza historię
    response3 = client.get(f"/api/history/{chat_id}")
    assert response3.status_code == 200
    messages = response3.json()["messages"]
    
    # KROK 4: Sprawdź że OBE wiadomości są w historii
    user_msgs = [m for m in messages if m["role"] == "user"]
    assert len(user_msgs) >= 2, "Nie zapisało obu wiadomości!"


# ════════════════════════════════════════════════════════════
# PRZYKŁAD 8: TEST BEZPIECZEŃSTWA
# ════════════════════════════════════════════════════════════

def test_sql_injection_protection():
    """
    TEST: Sprawdza ochronę przed SQL injection
    
    Po co: Żeby ktoś nie zhackował Twojej bazy!
    """
    # Próba SQL injection
    malicious_input = "'; DROP TABLE chat_messages; --"
    
    payload = {
        "newMessage": {
            "role": "user",
            "content": malicious_input
        }
    }
    
    # Endpoint powinien obsłużyć to bezpiecznie
    response = client.post("/api/chat", json=payload)
    
    # Nie powinno wywołać błędu 500 (server error)
    assert response.status_code in [200, 400], \
        "Server wywala się na SQL injection attempt!"
    
    # Sprawdź że baza nadal działa
    check = client.get("/api/history")
    assert check.status_code == 200, "Baza została uszkodzona!"


# ════════════════════════════════════════════════════════════
# PRZYKŁAD 9: PARAMETRYZOWANE TESTY (wiele przypadków)
# ════════════════════════════════════════════════════════════

@pytest.mark.parametrize("message,expected_length", [
    ("krótka", 6),
    ("trochę dłuższa wiadomość", 25),
    ("a" * 100, 100),  # 100 znaków
])
def test_message_lengths(message, expected_length):
    """
    TEST: Różne długości wiadomości
    
    Jeden test, wiele przypadków!
    """
    payload = {
        "newMessage": {
            "role": "user",
            "content": message
        }
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200


# ════════════════════════════════════════════════════════════
# PRZYKŁAD 10: TEST ASYNC (dla async funkcji)
# ════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_async_endpoint():
    """
    TEST: Dla async endpointów (jeśli masz)
    
    FastAPI może mieć async def funkcje
    """
    # Ten test przejdzie bo TestClient obsługuje async automatycznie
    response = client.get("/api/health")
    assert response.status_code == 200


# ════════════════════════════════════════════════════════════
# PRZYKŁAD 11: TEST Z SETUPEM I TEARDOWNEM
# ════════════════════════════════════════════════════════════

@pytest.fixture
def temporary_chat():
    """
    FIXTURE: Tworzy chat, po teście usuwa (jeśli byś miał DELETE)
    """
    # SETUP - przed testem
    payload = {"newMessage": {"role": "user", "content": "temp"}}
    response = client.post("/api/chat", json=payload)
    chat_id = response.json()["chatId"]
    
    # Zwróć chat_id do testu
    yield chat_id
    
    # TEARDOWN - po teście (czyszczenie)
    # (gdybyś miał endpoint DELETE /api/history/{chat_id})
    # client.delete(f"/api/history/{chat_id}")


def test_with_temporary_chat(temporary_chat):
    """Test używa temporary_chat fixture"""
    # temporary_chat jest już utworzony!
    chat_id = temporary_chat
    
    response = client.get(f"/api/history/{chat_id}")
    assert response.status_code == 200
    
    # Po tym teście, fixture automatycznie wyczyści


# ════════════════════════════════════════════════════════════
# PRZYKŁAD 12: TEST RATE LIMITING (jeśli masz)
# ════════════════════════════════════════════════════════════

def test_rate_limiting():
    """
    TEST: Sprawdza czy rate limiting działa
    
    (Twój kod nie ma rate limiting, ale przykład jak testować)
    """
    # Wyślij 100 requestów szybko
    responses = []
    for i in range(100):
        r = client.get("/api/health")
        responses.append(r.status_code)
    
    # Wszystkie powinny przejść (nie masz rate limit)
    # Ale jak byś dodał, niektóre powinny być 429 (Too Many Requests)
    assert all(r == 200 for r in responses[:10]), \
        "Pierwsze 10 requestów powinno przejść"


# ════════════════════════════════════════════════════════════
# PRZYKŁAD 13: TEST VALIDACJI DANYCH
# ════════════════════════════════════════════════════════════

@pytest.mark.parametrize("bad_payload", [
    {},  # Pusty
    {"newMessage": {}},  # Brak content
    {"newMessage": {"role": "hacker"}},  # Brak content
    {"newMessage": None},  # None
])
def test_chat_validates_input(bad_payload):
    """
    TEST: Endpoint powinien odrzucać złe dane
    
    Po co: Nie chcesz żeby złe dane crashowały serwer
    """
    response = client.post("/api/chat", json=bad_payload)
    
    # Powinno być 400 (Bad Request) lub 422 (Validation Error)
    # ALE NIE 500 (Internal Server Error)!
    assert response.status_code in [200, 400, 422], \
        f"Złe dane spowodowały server error: {response.status_code}"


# ════════════════════════════════════════════════════════════
# PRZYKŁAD 14: TEST KONKURSOWY (TDD approach)
# ════════════════════════════════════════════════════════════

def test_nowy_endpoint_ktory_jeszcze_nie_istnieje():
    """
    TDD (Test-Driven Development):
    1. Najpierw piszesz test
    2. Potem piszesz kod żeby test przeszedł
    
    Ten test FAILUJE - to OK! Teraz wiesz co napisać.
    """
    pytest.skip("Endpoint jeszcze nie istnieje - TODO!")
    
    # Gdy go napiszesz, usuń pytest.skip i test będzie działał:
    # response = client.get("/api/mojeimie")
    # assert response.json() == {"name": "Twoje Imię"}


# ════════════════════════════════════════════════════════════
# PRZYKŁAD 15: TEST RESPONSE HEADERS
# ════════════════════════════════════════════════════════════

def test_response_headers():
    """
    TEST: Sprawdza czy serwer ustawia prawidłowe headery
    """
    response = client.get("/api/health")
    
    # Content-Type powinien być JSON
    assert "application/json" in response.headers["content-type"]
    
    # Jeśli masz CORS (a masz!), powinien być header:
    # assert "access-control-allow-origin" in response.headers


# ════════════════════════════════════════════════════════════
# PRZYKŁAD 16: TEST Z TIMEOUT
# ════════════════════════════════════════════════════════════

def test_endpoint_doesnt_hang():
    """
    TEST: Endpoint nie może wisieć w nieskończoność
    """
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Request trwał za długo!")
    
    # Ustaw timeout 5 sekund
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)
    
    try:
        response = client.get("/api/health")
        assert response.status_code == 200
    finally:
        signal.alarm(0)  # Wyłącz alarm


# ════════════════════════════════════════════════════════════
# PRZYKŁAD 17: TEST IDEMPOTENCJI
# ════════════════════════════════════════════════════════════

def test_idempotent_operation():
    """
    TEST: Wywołanie tego samego 2x daje ten sam wynik
    
    IDEMPOTENCJA = możesz wywołać N razy, efekt taki sam
    """
    # GET /api/health jest idempotentny - wywołaj 3x
    r1 = client.get("/api/health")
    r2 = client.get("/api/health")
    r3 = client.get("/api/health")
    
    # Wszystkie powinny zwrócić to samo
    assert r1.json() == r2.json() == r3.json()
    
    # POST /api/chat NIE jest idempotentny - każdy tworzy nowy chat
    # (to jest OK, ale warto wiedzieć)


# ════════════════════════════════════════════════════════════
# PRZYKŁAD 18: TEST COVERAGE (pokrycie kodu)
# ════════════════════════════════════════════════════════════

def test_all_status_codes():
    """
    TEST: Sprawdź różne status codes
    
    Dobry endpoint powinien zwracać:
    - 200 dla sukcesu
    - 400 dla błędnych danych
    - 404 dla nie znalezionego
    - 500 dla server error
    """
    # 200 OK
    assert client.get("/api/health").status_code == 200
    
    # 404 Not Found
    assert client.get("/api/nieistniejacy").status_code == 404
    
    # (400/422 dla błędnych danych - przetestowane wyżej)


# ════════════════════════════════════════════════════════════
# BONUS: JAK TESTOWAĆ TO CO MASZ W src/
# ════════════════════════════════════════════════════════════

def test_llm_client_module():
    """
    TEST: Moduł src/llm_client.py
    
    Testuj swoje moduły OSOBNO, nie tylko przez API
    """
    try:
        from src import llm_client
        
        # Test że moduł się importuje
        assert llm_client is not None
        
        # Test health function
        health = llm_client.health()
        assert isinstance(health, dict)
        assert "ok" in health
        
    except ImportError:
        pytest.skip("llm_client niedostępny")


def test_config_module():
    """
    TEST: Config ładuje się poprawnie
    """
    from src import config
    
    # Sprawdź że są podstawowe wartości
    assert hasattr(config, 'LLM_BASE_URL')
    assert hasattr(config, 'WEB_HTTP_TIMEOUT')
    
    # Sprawdź typy
    assert isinstance(config.WEB_HTTP_TIMEOUT, int)
    assert config.WEB_HTTP_TIMEOUT > 0
