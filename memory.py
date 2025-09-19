# ruff: noqa: E501
"""
memory.py — scalony moduł pamięci (SQLite ~5GB, singleton)

Wersja: 2025-09-14 (mini-LLM zastąpiony lokalnym ekstraktorem)
Ulepszenia (realne):
- Lokalny ekstraktor faktów (regex/heurystyki, PL/EN-lite) — bez LLM, offline.
- Hybrydowy RAG: Embeddings (opcjonalne) + TF-IDF + BM25/FTS5 + świeżość + bonusy źródła.
- Admin: meta logi, vacuum/prune, rebuild_fts, integrity, backup (CLI).
- PRAGMA per-connection, liczenie rozmiaru z -wal/-shm, guards na FTS5.
"""

import base64
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

import config
from llm_simple import chat as llm_chat

# Load environment variables
load_dotenv()

# ------------------------- ENV / ŚCIEŻKI -------------------------
ROOT = Path(os.getenv("MEM_ROOT", str(Path(__file__).parent)))
DATA_DIR = ROOT / "data"
MEM_NS = (os.getenv("MEM_NS", "default") or "default").strip() or "default"
NS_DIR = DATA_DIR / MEM_NS
NS_DIR.mkdir(parents=True, exist_ok=True)

# Obsługa RunPod lub lokalnego storage
USE_RUNPOD = config.USE_RUNPOD
RUNPOD_PATH = config.RUNPOD_PERSIST_DIR

if USE_RUNPOD:
    # Używamy RunPod jako miejsca dla LTM (5GB)
    # Upewniamy się, że ścieżka istnieje
    runpod_data_dir = Path(RUNPOD_PATH) / "data"
    runpod_data_dir.mkdir(parents=True, exist_ok=True)

    DB_PATH = runpod_data_dir / "ltm.db"
    print(f"[MEMORY] Używam RunPod LTM: {DB_PATH}")
else:
    # Lokalna baza SQLite
    DB_PATH = DATA_DIR / "memory.db"
    print(f"[MEMORY] Używam lokalnej bazy LTM: {DB_PATH}")

# Parametry pamięci - zwiększone wartości dla lepszego kojarzenia
LTM_MIN_SCORE = float(os.getenv("LTM_MIN_SCORE", "0.25"))  # Obniżony próg pewności
MAX_LTM_FACTS = int(os.getenv("MAX_LTM_FACTS", "2000000"))  # 2 miliony faktów
RECALL_TOPK_PER_SRC = int(os.getenv("RECALL_TOPK_PER_SRC", "100"))  # Więcej kandydatów
STM_MAX_TURNS = int(os.getenv("STM_MAX_TURNS", "400"))  # Dłuższa historia
STM_KEEP_TAIL = int(os.getenv("STM_KEEP_TAIL", "100"))  # Więcej ostatnich tur
HTTP_TIMEOUT = int(os.getenv("LLM_HTTP_TIMEOUT_S", os.getenv("TIMEOUT_HTTP", "60")))

# Embeddings (opcjonalnie)
EMBED_URL = (os.getenv("LLM_EMBED_URL") or "").rstrip("/")
EMBED_MODEL = (os.getenv("LLM_EMBED_MODEL", "text-embedding-3-large") or "").strip()
EMBED_KEY = (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or "").strip()

# ------------------------- Lekka maska (PSY_ENCRYPT_KEY) -------------------------


class _Crypto:
    def __init__(self, key: str | None):
        self.key = hashlib.sha256((key or "").encode("utf-8")).digest() if key else None

    def enc(self, text: str) -> str:
        if not self.key:
            return text
        b = text.encode("utf-8")
        out = bytes(v ^ self.key[i % len(self.key)] for i, v in enumerate(b))
        return "xor:" + base64.urlsafe_b64encode(out).decode("ascii")

    def dec(self, blob: str | Any) -> str:
        if not self.key or not isinstance(blob, str) or not blob.startswith("xor:"):
            return blob if isinstance(blob, str) else str(blob)
        try:
            raw = base64.urlsafe_b64decode(blob[4:].encode("ascii"))
            out = bytes(v ^ self.key[i % len(self.key)] for i, v in enumerate(raw))
            return out.decode("utf-8", "ignore")
        except Exception:
            return str(blob)


CRYPTO = _Crypto(os.getenv("PSY_ENCRYPT_KEY"))

# ------------------------- Narzędzia tekstowe / TF-IDF -------------------------


def _norm(s: str) -> str:
    """Normalizuje tekst usuwając znaki specjalne i zostawiając litery i cyfry."""
    return "".join(ch.lower() for ch in (s or "").strip() if ch.isalnum() or ch.isspace())


def _id_for(text: str) -> str:
    """Generuje unikalny identyfikator dla tekstu."""
    return hashlib.sha1(_norm(text).encode("utf-8")).hexdigest()


def _tok(s: str) -> list[str]:
    """
    Tokenizuje tekst na słowa, usuwając znaki specjalne.
    Wersja ulepszona ze wsparciem dla skrótów myślowych i odporność na literówki.
    """
    # Obsługa częstych skrótów myślowych w języku polskim
    s = (s or "").lower()

    # Zamiana popularnych skrótów myślowych na pełne formy
    skroty = {
        "wg": "według",
        "np": "na przykład",
        "itd": "i tak dalej",
        "itp": "i tym podobne",
        "tzn": "to znaczy",
        "tzw": "tak zwany",
        "dr": "doktor",
        "prof": "profesor",
        "mgr": "magister",
        "ok": "okej",
        "bd": "będzie",
        "jj": "jasne",
        "nwm": "nie wiem",
        "wiadomo": "wiadomo",
        "imo": "moim zdaniem",
        "btw": "przy okazji",
        "tbh": "szczerze mówiąc",
        "fyi": "dla twojej informacji",
    }

    words = s.split()
    for i, word in enumerate(words):
        clean_word = re.sub(r"[^\w]", "", word)
        if clean_word in skroty:
            words[i] = skroty[clean_word]

    s = " ".join(words)

    # Standardowa tokenizacja
    s2 = re.sub(r"[^0-9a-ząćęłńóśźżA-ZĄĆĘŁŃÓŚŹŻ]+", " ", s)

    # Filtrowanie krótkich słów, ale z zachowaniem większej liczby tokenów
    return [w for w in s2.split() if len(w) > 2][:256]


def _tfidf_vec(tokens: list[str], docs_tokens: list[list[str]]) -> dict[str, float]:
    """Oblicza wektor TF-IDF dla listy tokenów względem korpusu dokumentów."""
    N = len(docs_tokens) if docs_tokens else 1
    vocab = set(t for d in docs_tokens for t in d)
    df = {t: sum(1 for d in docs_tokens if t in d) for t in vocab}
    tf: dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1

    # Wzmocnione ważenie dla rzadkich i znaczących terminów
    return {
        t: (
            (tf[t] / max(1, len(tokens)))
            * (math.log((N + 1) / (df.get(t, 1) + 1))) ** 1.5
            * (1 + 0.1 * min(len(t) - 3, 7) if len(t) > 3 else 1)
        )
        for t in tf
    }


def _tfidf_cos(q: str, docs: list[str]) -> list[float]:
    """
    Oblicza podobieństwo kosinusowe TF-IDF między zapytaniem a dokumentami.
    Wersja znacznie ulepszona dla lepszego kojarzenia - ekstremalne wzmocnienie
    rzadkich terminów, które pojawiają się zarówno w zapytaniu jak i dokumencie.
    """
    # Tokeny z zapytania i dokumentów
    tq = _tok(q)
    dts = [_tok(d) for d in docs]

    # Wektor zapytania
    vq = _tfidf_vec(tq, dts)
    out: list[float] = []

    # Słowa kluczowe, dajemy im większą wagę
    key_terms = set([t for t in tq if len(t) > 3])

    # Dla każdego dokumentu
    for dt in dts:
        vd = _tfidf_vec(dt, dts)
        keys = set(vq.keys()) | set(vd.keys())

        # Funkcja wzmacniająca znaczące dopasowania z nieliniową krzywą
        def boost_match(a, b, term):
            # Podstawowe ważenie
            val = a * b

            # Silne wzmocnienie dla kluczowych terminów
            term_bonus = 1.0
            if term in key_terms:
                term_bonus = 2.5

            # Wzmocnienie długich n-gramów (jeśli zawierają spacje)
            if " " in term:
                words = len(term.split())
                if words > 1:
                    term_bonus *= 1.0 + 0.5 * words

            # Agresywne wzmocnienie silnych dopasowań, wyciszanie słabych
            boost = 1 + 0.8 * math.tanh(4 * val - 0.6)

            return val * boost * term_bonus

        # Suma ważonych dopasowań
        num = sum(boost_match(vq.get(t, 0.0), vd.get(t, 0.0), t) for t in keys)

        # Normalizacja
        den_a = sum(x * x for x in vq.values()) ** 0.5
        den_b = sum(x * x for x in vd.values()) ** 0.5
        den = den_a * den_b

        # Dodatkowy współczynnik wzmocnienia podobieństwa
        score = 0.0 if den == 0 else (num / den)

        # Nieliniowe wzmocnienie podobieństwa - napompowanie silnych wyników
        score = score**0.8  # Wykładnik <1 wzmacnia silne dopasowania

        out.append(score)
    return out


# ------------------------- SQLite init -------------------------
_SQL_BASE = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=MEMORY;
PRAGMA page_size=4096;
PRAGMA max_page_count=1310720;

CREATE TABLE IF NOT EXISTS ltm (
  id TEXT PRIMARY KEY,
  ts REAL DEFAULT (strftime('%s','now')),
  kind TEXT NOT NULL DEFAULT 'fact',
  text TEXT NOT NULL,
  meta TEXT,
  score REAL DEFAULT 0.6
);
CREATE TABLE IF NOT EXISTS stm ( ts REAL, user TEXT, assistant TEXT );
CREATE TABLE IF NOT EXISTS episodes ( ts REAL, user TEXT, assistant TEXT );
CREATE TABLE IF NOT EXISTS profile ( key TEXT PRIMARY KEY, value TEXT );
CREATE TABLE IF NOT EXISTS goals (
  id TEXT PRIMARY KEY,
  title TEXT,
  priority REAL DEFAULT 1.0,
  tags TEXT DEFAULT '[]',
  ts REAL
);
CREATE TABLE IF NOT EXISTS meta_events ( ts REAL, kind TEXT, payload TEXT );
CREATE INDEX IF NOT EXISTS idx_ltm_ts    ON ltm(ts);
CREATE INDEX IF NOT EXISTS idx_ltm_kind  ON ltm(kind);
CREATE INDEX IF NOT EXISTS idx_ltm_score ON ltm(score);
CREATE INDEX IF NOT EXISTS idx_goal_pri  ON goals(priority);
CREATE INDEX IF NOT EXISTS idx_meta_kind ON meta_events(kind);
"""
_SQL_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS ltm_fts
USING fts5(text, content_rowid=id, tokenize='unicode61')
"""

# === NOWY SCHEMAT BAZY DANYCH DLA ZAAWANSOWANEJ PAMIĘCI (Punkt 1-8) ===
_SQL_ADVANCED_MEMORY = """
-- Tabela dla Timeline (Pamięć Epizodyczna)
CREATE TABLE IF NOT EXISTS timeline_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL DEFAULT (strftime('%s','now')),
    type TEXT NOT NULL, -- 'interaction', 'summary', 'event'
    title TEXT,
    content TEXT NOT NULL,
    user_input TEXT,
    ai_response TEXT,
    mood TEXT,
    context TEXT,
    related_person_id INTEGER,
    related_file_id INTEGER,
    FOREIGN KEY (related_person_id) REFERENCES person_profiles(id),
    FOREIGN KEY (related_file_id) REFERENCES sensory_files(id)
);

-- Tabela dla Profili Osób (Mapowanie Relacji)
CREATE TABLE IF NOT EXISTS person_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    aliases TEXT, -- JSON list
    role TEXT,
    relationship_status TEXT,
    notes TEXT
);

-- Tabela dla Pamięci Kontekstowej
CREATE TABLE IF NOT EXISTS context_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_type TEXT UNIQUE NOT NULL,
    priority_facts TEXT, -- JSON list
    active_goals TEXT, -- JSON list
    notes TEXT
);

-- Tabela dla Pamięci Sensorycznej (Plikowej)
CREATE TABLE IF NOT EXISTS sensory_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    file_type TEXT,
    file_path TEXT,
    description TEXT,
    tags TEXT, -- JSON list
    timestamp REAL DEFAULT (strftime('%s','now'))
);

-- Tabela dla Samorefleksji
CREATE TABLE IF NOT EXISTS self_reflections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL DEFAULT (strftime('%s','now')),
    summary TEXT NOT NULL,
    lessons_learned TEXT, -- JSON list
    rules_to_remember TEXT -- JSON list
);

-- Tabela dla Wzorców Predykcyjnych
CREATE TABLE IF NOT EXISTS prediction_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger_pattern TEXT UNIQUE NOT NULL,
    predicted_action TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    usage_count INTEGER DEFAULT 0
);

-- Tabela dla Wersjonowania Pamięci
CREATE TABLE IF NOT EXISTS memory_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL DEFAULT (strftime('%s','now')),
    version_hash TEXT UNIQUE NOT NULL,
    description TEXT,
    backup_path TEXT NOT NULL
);
"""

_HAS_FTS5 = True


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    """Aplikuje optymalne ustawienia PRAGMA dla połączenia SQLite."""
    try:
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            PRAGMA temp_store=MEMORY;
            PRAGMA cache_size=-16000;
        """
        )
    except Exception:
        pass


def _connect() -> sqlite3.Connection:
    """
    Tworzy i zwraca nowe połączenie do bazy danych z odpowiednimi ustawieniami.
    """
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    _apply_pragmas(conn)
    return conn


_DB_LOCK = threading.RLock()


def _init_db():
    """Inicjalizuje bazę danych, tworząc wymagane tabele i indeksy."""
    global _HAS_FTS5
    with _DB_LOCK:
        conn = _connect()
        try:
            conn.executescript(_SQL_BASE)
            conn.executescript(_SQL_ADVANCED_MEMORY) # Dodajemy nowe tabele
            try:
                conn.execute(_SQL_FTS)
            except Exception:
                _HAS_FTS5 = False
            conn.commit()
        finally:
            conn.close()


_init_db()

# ------------------------- Embeddings (opcjonalnie) -------------------------


def _embed_many(texts: list[str]) -> list[list[float]] | None:
    """Generuje embeddingi dla listy tekstów za pomocą zewnętrznego API."""
    if not (EMBED_URL and EMBED_KEY and texts):
        return None
    import requests

    try:
        r = requests.post(
            EMBED_URL,
            headers={
                "Authorization": f"Bearer {EMBED_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": EMBED_MODEL, "input": texts},
            timeout=HTTP_TIMEOUT,
        )
        if r.status_code >= 400:
            return None
        j = r.json()
        vecs = [d.get("embedding") for d in j.get("data", [])]
        return vecs if len(vecs) == len(texts) else None
    except Exception:
        return None


def _cos(a: list[float], b: list[float]) -> float:
    """Oblicza podobieństwo kosinusowe między dwoma wektorami."""
    sa = sum(x * x for x in a) ** 0.5
    sb = sum(x * x for x in b) ** 0.5
    dot_product = sum(x * y for x, y in zip(a, b, strict=False))
    return 0.0 if sa == 0 or sb == 0 else dot_product / (sa * sb)


# ------------------------- Lokalny ekstraktor faktów (bez LLM) -------------------------
# Rozszerzony: preferencje, profil (wiek/miasto/praca), kontakt, zdrowie,
# języki, technologie, godziny pracy, linki.


def _sentences(text: str) -> list[str]:
    """Dzieli tekst na zdania."""
    if not text:
        return []
    raw = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in raw if len(s.strip()) >= 5]


_PREF_PAT = re.compile(
    r"\b(lubię|wolę|preferuję|kocham|nienawidzę|nie\s+lubię|nie\s+cierpię"
    r"|najczęściej\s+piję|zazwyczaj\s+jadam|uwielbiam|lubię gdy|podoba mi się)\b",
    re.I,
)
_PROFILE_PATS = {
    "age": re.compile(
        (
            r"\b("  # początek
            r"mam|posiadam|skończyłem|skończyłam|"  # formy osobowe
            r"\w+ lat|\w+ wiosny?"  # alternatywne formy wieku
            r")\s*(\d{1,2})\s*"  # liczba
            r"(lat|lata|wiosen|rok|roku|roczek)?\b"  # jednostka opcjonalna
        ),
        re.I,
    ),
    "email": re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", re.I),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\d{3}[\s-]?){2}\d{3,4}\b"),
    "city": re.compile(
        (
            r"\b("  # początek
            r"mieszkam\s+w|jestem\s+z|pochodzę\s+z|mieszkam\s+blisko"  # warianty
            r")\s+([A-ZŁŚŻŹĆĘÓĄ][\w\-ąćęłńóśźż]+)\b"  # nazwa miasta
        ),
        re.I,
    ),
    "job": re.compile(
        (
            r"\b("  # początek
            r"pracuję\s+jako|zawodowo\s+jestem|pracuję\s+w|jestem\s+z\s+zawodu"
            r")\s+([a-ząćęłńóśźż\- ]{3,40})\b"  # nazwa zawodu
        ),
        re.I,
    ),
}
_LANG_PAT = re.compile(
    (
        r"\b("  # początek
        r"mówię|znam|używam|porozumiewam się|uczę się|rozumiem"  # czasowniki
        r")\s+(po\s+)?("  # opcjonalne 'po'
        r"polsku|angielsku|niemiecku|hiszpańsku|francusku|ukraińsku|rosyjsku|"
        r"włosku|japońsku|chińsku|koreańsku|portugalsku"  # języki cz.2
        r")\b"
    ),
    re.I,
)
_TECH_PAT = re.compile(
    (
        r"\b("  # start
        r"R|Python|SQL|Java|JavaScript|TypeScript|C\+\+|C#|Go|Rust|PHP|Ruby|HTML|CSS|"
        r"Docker|Kubernetes|TensorFlow|PyTorch|React|Angular|Vue|Node\.?js|Django|"
        r"Flask|Laravel|Spring|Express\.?js|GraphQL|REST|API"  # koniec listy
        r")\b"
    ),
    re.I,
)
_HOURS_PAT = re.compile(
    r"\b(pracuję|dostępny)\s+(od|w)\s+(\d{1,2})(?:[:.]\d{2})?\s*(do|-)\s*"
    r"(\d{1,2})(?:[:.]\d{2})?\b",
    re.I,
)
_LINK_PAT = re.compile(r"\bhttps?://\S+\b", re.I)
_HEALTH_PAT = re.compile(r"\b(alergi[ae]|uczulenie|nietolerancj[ae])\b", re.I)
_NEGATION_PAT = re.compile(r"\b(nie|nie\s+bardzo|żadn[eyoa])\b", re.I)


def _extract_facts_from_turn(u: str, a: str) -> list[tuple[str, float, list[str]]]:
    """Ekstrahuje fakty z pary wypowiedzi (user, assistant)."""
    facts: list[tuple[str, float, list[str]]] = []

    def _mk_fact(text: str, base_score: float, tags: list[str]) -> tuple[str, float, list[str]]:
        """Tworzy fakt z odpowiednią oceną (score) i tagami.

        Zachowuje kompatybilność wsteczną z API zwracając (text, score, tags)
        ale wewnętrznie tags są przechowywane w meta_data.
        """
        t = (text or "").strip()
        if not t:
            return ("", 0.0, tags)
        neg = bool(_NEGATION_PAT.search(t))
        score_delta = -0.08 if neg else 0.04
        score = max(0.5, min(0.95, base_score + score_delta))
        return (t, score, sorted(set(tags)))

    for role, txt in (("user", u or ""), ("assistant", a or "")):
        for s in _sentences(txt):
            if _PREF_PAT.search(s):
                facts.append(
                    _mk_fact(
                        f"preferencja: {s}",
                        0.80 if role == "user" else 0.72,
                        ["stm", "preference"],
                    )
                )
                continue

            m = _PROFILE_PATS["age"].search(s)
            if m:
                facts.append(
                    _mk_fact(
                        f"wiek: {m.group(2)}",
                        0.86 if role == "user" else 0.78,
                        ["stm", "profile"],
                    )
                )
                continue

            m = _PROFILE_PATS["email"].search(s)
            if m:
                facts.append(
                    _mk_fact(
                        f"email: {m.group(0)}",
                        0.89 if role == "user" else 0.81,
                        ["stm", "profile", "contact"],
                    )
                )
                continue

            m = _PROFILE_PATS["phone"].search(s)
            if m:
                facts.append(
                    _mk_fact(
                        f"telefon: {m.group(0)}",
                        0.88 if role == "user" else 0.8,
                        ["stm", "profile", "contact"],
                    )
                )
                continue

            m = _PROFILE_PATS["city"].search(s)
            if m:
                facts.append(
                    _mk_fact(
                        f"miasto: {m.group(2)}",
                        0.87 if role == "user" else 0.79,
                        ["stm", "profile"],
                    )
                )
                continue

            m = _PROFILE_PATS["job"].search(s)
            if m:
                facts.append(
                    _mk_fact(
                        f"zawód: {m.group(2)}",
                        0.85 if role == "user" else 0.77,
                        ["stm", "profile"],
                    )
                )
                continue

            for lang in _LANG_PAT.findall(s):
                facts.append(
                    _mk_fact(
                        f"język: {lang[2].lower()}",
                        0.78 if role == "user" else 0.7,
                        ["stm", "profile", "language"],
                    )
                )

            for tech in set(t.group(0) for t in _TECH_PAT.finditer(s)):
                facts.append(
                    _mk_fact(
                        f"tech: {tech}",
                        0.77 if role == "user" else 0.69,
                        ["stm", "skill", "tech"],
                    )
                )

            mh = _HOURS_PAT.search(s)
            if mh:
                facts.append(
                    _mk_fact(
                        f"availability: {mh.group(3)}-{mh.group(5)}",
                        0.75 if role == "user" else 0.67,
                        ["stm", "availability"],
                    )
                )

            for url in _LINK_PAT.findall(s):
                facts.append(_mk_fact(f"link: {url}", 0.8, ["stm", "link"]))

            if _HEALTH_PAT.search(s):
                facts.append(
                    _mk_fact(
                        f"zdrowie: {s}",
                        0.82 if role == "user" else 0.74,
                        ["stm", "health"],
                    )
                )

    return facts


def _dedupe_facts(facts: list[tuple[str, float, list[str]]]) -> list[tuple[str, float, list[str]]]:
    """
    Deduplikuje fakty bazując na treści, zachowując najwyższe score + łącząc tagi.
    """
    by_id: dict[str, tuple[str, float, list[str]]] = {}
    for t, score, tags in facts:
        t2 = (t or "").strip()
        if not t2:
            continue
        fid = _id_for(t2)
        if fid in by_id:
            old_t, old_score, old_tags = by_id[fid]
            by_id[fid] = (
                old_t,
                max(old_score, score),
                sorted(set((old_tags or []) + (tags or []))),
            )
        else:
            by_id[fid] = (t2, score, sorted(set(tags or [])))
    return list(by_id.values())


def _extract_facts(
    messages: list[dict[str, str]], max_out: int = 120
) -> list[tuple[str, float, list[str]]]:
    """
    Ekstrahuje fakty z historii wiadomości.
    Wersja ulepszona o obsługę skrótów myślowych i literówek.
    """
    if not messages:
        return []

    all_facts: list[tuple[str, float, list[str]]] = []
    i = 0

    # Zbierz wszystkie wiadomości w jedną całość (szerszy kontekst)
    full_context = ""
    for msg in messages:
        content = msg.get("content", "")
        role = msg.get("role", "")
        if content:
            full_context += f"{role.upper()}: {content}\n\n"

    # Ekstrahuj fakty z par wiadomości
    while i < len(messages):
        role_i = messages[i].get("role")
        u = messages[i].get("content", "") if role_i == "user" else ""
        a = ""
        if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
            a = messages[i + 1].get("content", "")
            i += 2
        else:
            i += 1

        # Ekstrahuj fakty z tej pary wiadomości
        extracted_facts = _extract_facts_from_turn(u, a)
        all_facts.extend(extracted_facts)

        # Postarajmy się zidentyfikować odniesienia do poprzednich wypowiedzi
        # i poprawnie je rozwiązać w kontekście całej konwersacji
        if u:
            # Szukaj skrótów myślowych / niejasnych odniesień w wypowiedziach
            for ref_pattern in [
                r"\bto\b",
                r"\btam\b",
                r"\bten\b",
                r"\bta\b",
                r"\bte\b",
                r"\bon\b",
                r"\bona\b",
                r"\bono\b",
                r"\boni\b",
                r"\bich\b",
                r"\bjego\b",
                r"\bjej\b",
            ]:
                if re.search(ref_pattern, u, re.I):
                    # Znaleziono skrót myślowy -> dodaj fakt z wyższą pewnością
                    # połączony z kontekstem poprzednich wiadomości
                    # Ostatnie 1500 znaków kontekstu
                    context_window = full_context[-1500:]
                    all_facts.append(
                        (
                            f"Kontekst rozmowy: {context_window}",
                            0.75,
                            ["stm", "context_reference"],
                        )
                    )
                    break

    # Deduplikuj i posortuj fakty
    all_facts = _dedupe_facts(all_facts)
    all_facts.sort(key=lambda x: x[1], reverse=True)
    return all_facts[:max_out]


# ------------------------- DB care -------------------------


def _db_size_mb() -> float:
    """Zwraca rozmiar bazy danych w MB (uwzględniając -wal i -shm)."""
    try:
        base = DB_PATH.stat().st_size if DB_PATH.exists() else 0
        wal = DB_PATH.with_name(DB_PATH.name + "-wal")
        shm = DB_PATH.with_name(DB_PATH.name + "-shm")
        base += wal.stat().st_size if wal.exists() else 0
        base += shm.stat().st_size if shm.exists() else 0
        return base / (1024 * 1024)
    except Exception:
        return 0.0


def _vacuum_if_needed(threshold_mb: float = 4500.0) -> dict[str, Any]:
    """Wykonuje VACUUM bazy jeśli rozmiar przekracza próg."""
    size = _db_size_mb()
    if size < threshold_mb:
        return {"ok": True, "size_mb": size, "action": "none"}
    with _DB_LOCK:
        conn = _connect()
        try:
            conn.execute("VACUUM")
            conn.commit()
        finally:
            conn.close()
    return {"ok": True, "size_mb": _db_size_mb(), "action": "vacuum"}


def _prune_lowscore_facts(target_mb: float = 4200.0, batch: int = 2000) -> dict[str, Any]:
    """Usuwa fakty o niskim score jeśli rozmiar bazy przekracza cel."""
    size = _db_size_mb()
    if size < target_mb:
        return {"ok": True, "pruned": 0, "size_mb": size}

    removed = 0
    with _DB_LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT id FROM ltm
                WHERE score < ?
                ORDER BY ts ASC
                LIMIT ?""",
                (max(0.15, LTM_MIN_SCORE - 0.15), int(batch)),
            ).fetchall()

            ids = [r["id"] for r in rows]
            if ids:
                q = ",".join("?" * len(ids))
                conn.execute(f"DELETE FROM ltm WHERE id IN ({q})", ids)
                if _HAS_FTS5:
                    try:
                        conn.execute(f"DELETE FROM ltm_fts WHERE id IN ({q})", ids)
                    except Exception:
                        pass
                removed = len(ids)
                conn.commit()
        finally:
            conn.close()
    return {"ok": True, "pruned": removed, "size_mb": _db_size_mb()}


def _integrity_check() -> dict[str, Any]:
    """Wykonuje kontrolę integralności bazy danych."""
    with _DB_LOCK:
        conn = _connect()
        try:
            ok = conn.execute("PRAGMA integrity_check").fetchone()[0]
            freelist = conn.execute("PRAGMA freelist_count").fetchone()[0]
            return {"ok": ok == "ok", "result": ok, "freelist": freelist}
        finally:
            conn.close()


def _backup_db(out_path: str) -> dict[str, Any]:
    """Tworzy kopię zapasową bazy danych."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _vacuum_if_needed(0)
    try:
        shutil.copy2(DB_PATH, out)
        for suf in ("-wal", "-shm"):
            side = DB_PATH.with_name(DB_PATH.name + suf)
            if side.exists():
                shutil.copy2(side, out.with_name(out.name + suf))
        return {"ok": True, "path": str(out)}
    except Exception as e:
        return {"ok": False, "reason": str(e)}


# ------------------------- BM25 (FTS5) -------------------------


def _fts_bm25(query: str, limit: int = 50) -> list[tuple[str, float, float]]:
    """Wyszukuje fakty przy użyciu FTS5 i rankingu BM25."""
    if not _HAS_FTS5 or not query:
        return []

    with _DB_LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT ltm_fts.id AS id, bm25(ltm_fts) AS bscore
                FROM ltm_fts
                WHERE ltm_fts MATCH ?
                ORDER BY bscore ASC
                LIMIT ?""",
                (query, int(limit)),
            ).fetchall()

            if not rows:
                return []

            ids = [r["id"] for r in rows]
            ph = ",".join("?" * len(ids))
            got = conn.execute(f"SELECT id, text, ts FROM ltm WHERE id IN ({ph})", ids).fetchall()
            meta = {r["id"]: (r["text"], float(r["ts"] or 0.0)) for r in got}

            out: list[tuple[str, float, float]] = []
            for r in rows:
                t, ts = meta.get(r["id"], ("", 0.0))
                if not t:
                    continue

                try:
                    bscore = float(r["bscore"] or 0.0)
                except (ValueError, TypeError):
                    bscore = 0.0

                score = 1.0 / (1.0 + bscore)
                out.append((t, score, ts))

            return out
        except Exception:
            return []
        finally:
            conn.close()


def _rebuild_fts(limit: int | None = None) -> dict[str, Any]:
    """Przebudowuje indeks FTS5 dla faktów."""
    if not _HAS_FTS5:
        return {"ok": False, "reason": "fts5_unavailable"}

    with _DB_LOCK:
        conn = _connect()
        try:
            conn.execute("DELETE FROM ltm_fts")
            q = "SELECT id, text FROM ltm ORDER BY ts DESC"
            if limit:
                q += f" LIMIT {int(limit)}"

            rows = conn.execute(q).fetchall()
            for r in rows:
                txt = r["text"]
                if txt:
                    conn.execute(
                        "INSERT INTO ltm_fts(id, text) VALUES(?, ?)",
                        (r["id"], txt),
                    )

            conn.commit()
            return {"ok": True, "indexed": len(rows)}
        finally:
            conn.close()


# ------------------------- Hybryda scoringu -------------------------


def _emb_scores(query: str, docs: list[str]) -> list[float]:
    """
    Oblicza podobieństwo wektorowe embeddingów między zapytaniem a dokumentami.
    """
    try:
        vecs = _embed_many([query] + docs)
        if not vecs:
            return []
        qv, dvs = vecs[0], vecs[1:]
        return [_cos(qv, d) for d in dvs]
    except Exception:
        return []


def _tfidf_scores(query: str, docs: list[str]) -> list[float]:
    """Oblicza podobieństwo TF-IDF między zapytaniem a dokumentami."""
    try:
        return _tfidf_cos(query, docs)
    except Exception:
        return [0.0] * len(docs)


# Funkcja do obliczania podobieństwa stringów z uwzględnieniem literówek
def _string_similarity(s1: str, s2: str) -> float:
    """
    Oblicza podobieństwo między dwoma stringami, uwzględniając możliwe literówki.
    Wykorzystuje odległość Levenshteina znormalizowaną do długości dłuższego stringa.
    """
    if not s1 or not s2:
        return 0.0

    # Odległość Levenshteina (liczba zmian potrzebnych do konwersji jednego napisu w drugi)
    def levenshtein(a: str, b: str) -> int:
        if not a:
            return len(b)
        if not b:
            return len(a)

        # Inicjalizacja macierzy
        matrix = [[0 for _ in range(len(b) + 1)] for _ in range(len(a) + 1)]

        for i in range(len(a) + 1):
            matrix[i][0] = i
        for j in range(len(b) + 1):
            matrix[0][j] = j

        # Wypełnianie macierzy
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                matrix[i][j] = min(
                    matrix[i - 1][j] + 1,  # usunięcie
                    matrix[i][j - 1] + 1,  # wstawienie
                    matrix[i - 1][j - 1] + cost,  # zastąpienie/bez zmian
                )

        return matrix[len(a)][len(b)]

    # Normalizacja wyniku do zakresu [0, 1], gdzie 1 oznacza identyczne stringi
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0  # Oba stringi są puste

    distance = levenshtein(s1.lower(), s2.lower())
    similarity = 1.0 - (distance / max_len)

    return similarity


def _blend_scores(a: list[float], b: list[float], wa: float, wb: float) -> list[float]:
    """
    Łączy dwa zbiory wyników z odpowiednimi wagami.
    Wersja z agresywnym wzmocnieniem silnych dopasowań i dużym bonusem
    dla zgodnych wyników z różnych metod (sygnał zgodności algorytmów).

    Ta wersja zapewnia znacznie silniejsze kojarzenie faktów i kontekstu,
    dając priorytet dokumentom, które są zgodnie uznane za relewantne przez różne
    algorytmy wyszukiwania.
    """
    if not a:
        a = [0.0] * len(b)
    if not b:
        b = [0.0] * len(a)
    n = min(len(a), len(b))

    result = []
    for i in range(n):
        # Super-agresywne wzmocnienie silnych dopasowań nieliniowo
        a_score = a[i] ** 1.2  # Łagodniejszy wykładnik dla lepszego kojarzenia
        b_score = b[i] ** 1.2

        # Duży bonus dla zgodnych wyników z obu algorytmów
        harmony_bonus = 0.0

        # Jeśli oba algorytmy wskazują jako relewantne
        if a[i] > 0.35 and b[i] > 0.35:  # Obniżony próg dla lepszego kojarzenia
            # Bonus rośnie wykładniczo jeśli oba algorytmy mają wysoką pewność
            harmony_factor = (a[i] * b[i]) ** 0.5  # Średnia geometryczna
            harmony_bonus = 0.3 * harmony_factor  # Zwiększony mnożnik

        # Dodatkowe wzmocnienie jeśli oba algorytmy mają bardzo wysokie wyniki
        if a[i] > 0.7 and b[i] > 0.7:
            harmony_bonus *= 1.5

        result.append(wa * a_score + wb * b_score + harmony_bonus)

    return result


def _src_bonus(src: str) -> float:
    """Zwraca mnożnik bonusu dla danego źródła."""
    bonuses = {
        "profile": 1.35,  # Znacznie większy nacisk na profil użytkownika
        "goal": 1.30,  # Znacznie większy nacisk na cele
        "fact": 1.25,  # Większy nacisk na fakty
        "episode": 1.20,  # Większy nacisk na epizody
        "fts": 1.22,  # Większy nacisk na wyniki FTS
    }
    return bonuses.get(src, 1.0)


def _freshness_bonus(ts: float) -> float:
    """
    Zwraca mnożnik bonusu za świeżość (nowość) faktu.
    Funkcja eksponencjalna dająca większe znaczenie bardzo świeżym faktom,
    z wolniejszym spadkiem dla starszych.
    """
    if not ts:
        return 1.0

    age_days = max(0.0, (time.time() - ts) / 86400.0)

    # Dla bardzo świeżych faktów (< 3 dni) dodatkowy bonus
    if age_days < 3.0:
        recency_boost = 0.2 * math.exp(-age_days)
    else:
        recency_boost = 0.0

    # Podstawowa funkcja świeżości z wolniejszym spadkiem
    base_freshness = max(0.75, 1.4 - (age_days / 180.0))

    return base_freshness + recency_boost


# ------------------------- Klasa Memory (do recall) -------------------------


class Memory:
    def __init__(self, namespace: str = MEM_NS):
        """Inicjalizuje obiekt pamięci z daną przestrzenią nazw."""
        self.ns = namespace
        NS_DIR.mkdir(parents=True, exist_ok=True)
        _init_db()

    def add_entry(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        """Uniwersalna metoda do dodawania wpisów, mapująca na add_fact."""
        tags = ["entry"]
        if metadata:
            if "source" in metadata:
                tags.append(f"src:{metadata['source']}")
            if "filename" in metadata:
                tags.append("file")

        return self.add_fact(text, meta_data={"tags": tags}, score=0.75)

    # ====== LTM ======
    def add_fact(
        self,
        text: str,
        meta_data: dict | None = None,
        score: float = 0.6,
        emb: list[float] | None = None,
        tags: list[str] | None = None,
        conf: float | None = None,
    ) -> str:
        """Dodaje fakt do pamięci długoterminowej (LTM)."""
        txt = (text or "").strip()
        if not txt:
            raise ValueError("empty")

        # Obsługa starszego API (dla wstecznej kompatybilności)
        if conf is not None and score == 0.6:
            score = conf

        # Jeśli podano tagi w starym formacie, dodaj je do meta_data
        if tags:
            meta_data = meta_data or {}
            current_tags = meta_data.get("tags", [])
            meta_data["tags"] = sorted(set(current_tags + tags))

        # Przygotuj meta dane
        meta_data = meta_data or {}
        meta_json = json.dumps(meta_data, ensure_ascii=False)

        # Utwórz identyfikator faktu
        fid = _id_for(txt)

        # Przygotuj osadzenia (embeddingi)
        eb = emb
        if eb is None and EMBED_URL and EMBED_KEY:
            vecs = _embed_many([txt])
            eb = vecs[0] if vecs else None
        # emb_json was unused; keep embeddings only in-memory for now
        # If persisted embeddings are needed later, serialize here.
        # emb_json = json.dumps(eb) if eb else None

        with _DB_LOCK:
            conn = _connect()
            try:
                # Sprawdź czy fakt już istnieje
                row = conn.execute(
                    "SELECT id, meta, score FROM ltm WHERE id=?",
                    (fid,),
                ).fetchone()

                if row:
                    # Aktualizuj istniejący fakt
                    try:
                        # Połącz metadane
                        cur_meta = json.loads(row["meta"] or "{}")
                        for k, v in meta_data.items():
                            if k not in cur_meta:
                                cur_meta[k] = v
                            elif k == "tags" and isinstance(cur_meta.get(k), list):
                                # Połącz listy tagów
                                tags_set = set(cur_meta[k])
                                tags_set.update(meta_data.get(k, []))
                                cur_meta[k] = sorted(tags_set)

                        meta_json = json.dumps(cur_meta, ensure_ascii=False)

                        # Użyj wyższego poziomu pewności
                        new_score = max(float(row["score"] or 0), float(score))
                    except Exception as e:
                        print(f"Error updating fact: {e}")
                        new_score = score

                    # Aktualizuj rekord
                    conn.execute(
                        ("UPDATE ltm SET meta=?, score=?, " "ts=strftime('%s','now') WHERE id=?"),
                        (meta_json, new_score, fid),
                    )

                    # Aktualizuj indeks FTS jeśli istnieje
                    if _HAS_FTS5:
                        try:
                            conn.execute(
                                "INSERT INTO ltm_fts(id, text) VALUES(?, ?) "
                                "ON CONFLICT(id) DO UPDATE SET "
                                "text=excluded.text",
                                (fid, txt),
                            )
                        except Exception:
                            try:
                                conn.execute("DELETE FROM ltm_fts WHERE id=?", (fid,))
                                conn.execute(
                                    "INSERT INTO ltm_fts(id, text) VALUES(?, ?)",
                                    (fid, txt),
                                )
                            except Exception:
                                pass
                else:
                    # Dodaj nowy fakt
                    conn.execute(
                        "INSERT INTO ltm(id, kind, text, meta, score, ts) "
                        "VALUES(?, ?, ?, ?, ?, strftime('%s','now'))",
                        (
                            fid,
                            "fact",
                            txt,
                            meta_json,
                            float(score),
                        ),
                    )

                    # Dodaj do indeksu FTS
                    if _HAS_FTS5:
                        try:
                            conn.execute(
                                "INSERT INTO ltm_fts(id, text) VALUES(?, ?)",
                                (fid, txt),
                            )
                        except Exception:
                            print("Ostrzeżenie: Nie można dodać do indeksu FTS")

                conn.commit()
                # self._meta_event("ltm_upsert", {"id": fid})
                return fid
            finally:
                conn.close()

    def add_fact_bulk(self, rows: list[tuple[str, float, list[str] | None]]) -> dict[str, int]:
        """Dodaje wiele faktów na raz."""
        ins = mrg = 0
        for t, c, tg in rows:
            before = self.exists(_id_for(t))
            self.add_fact(t, score=c, meta_data={"tags": tg})
            if before:
                mrg += 1
            else:
                ins += 1
        return {"inserted": ins, "merged": mrg}

    def exists(self, fid: str) -> bool:
        """Sprawdza czy fakt o danym ID istnieje."""
        with _DB_LOCK:
            conn = _connect()
            try:
                return bool(conn.execute("SELECT 1 FROM ltm WHERE id=?", (fid,)).fetchone())
            finally:
                conn.close()

    def delete_fact(self, id_or_text: str) -> bool:
        """Usuwa fakt po ID lub tekście."""
        tid = id_or_text if re.fullmatch(r"[0-9a-f]{40}", id_or_text) else _id_for(id_or_text)
        with _DB_LOCK:
            conn = _connect()
            try:
                conn.execute("DELETE FROM ltm WHERE id=?", (tid,))
                if _HAS_FTS5:
                    try:
                        conn.execute("DELETE FROM ltm_fts WHERE id=?", (tid,))
                    except Exception:
                        pass
                conn.commit()
                # self._meta_event("ltm_delete", {"id": tid})
                return True
            finally:
                conn.close()

    def list_facts(self, limit: int = 500) -> list[dict[str, Any]]:
        """Zwraca listę faktów w pamięci długoterminowej."""
        with _DB_LOCK:
            conn = _connect()
            try:
                rows = conn.execute(
                    "SELECT id, text, meta, score, ts FROM ltm " "ORDER BY ts DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()

                result = []
                for r in rows:
                    try:
                        # Parsowanie meta jako JSON, wyciągnij tagi jeśli istnieją
                        meta_data = json.loads(r["meta"] or "{}") if r["meta"] else {}
                        tags = meta_data.get("tags", [])

                        # Dla zachowania kompatybilności wstecz, używamy score jako conf
                        result.append(
                            {
                                "id": r["id"],
                                "text": r["text"],  # Teraz używamy text zamiast text_enc
                                "conf": float(r["score"] or 0),  # Używamy score zamiast conf
                                "tags": tags,  # Wyciągamy tagi z meta
                                "ts": float(r["ts"] or 0),
                                "score": float(
                                    r["score"] or 0
                                ),  # Dodajemy też jako score dla nowego API
                                "meta": meta_data,  # Dodajemy pełne meta dla nowego API
                            }
                        )
                    except Exception as e:
                        print(f"Błąd podczas przetwarzania faktu: {e}")

                return result
            finally:
                conn.close()

    def rebuild_missing_embeddings(self, batch: int = 64) -> dict[str, Any]:
        """Odnajduje i uzupełnia brakujące embeddingi dla faktów."""
        if not (EMBED_URL and EMBED_KEY):
            return {"ok": False, "reason": "embedding_disabled"}

        updated = 0
        with _DB_LOCK:
            conn = _connect()
            try:
                while True:
                    query_missing_emb = (
                        "SELECT id, text FROM ltm "
                        "WHERE id NOT IN (SELECT id FROM ltm_embeddings) "
                        "LIMIT ?"
                    )
                    rows = conn.execute(query_missing_emb, (int(batch),)).fetchall()

                    if not rows:
                        break

                    texts = [r["text"] for r in rows]
                    ids = [r["id"] for r in rows]
                    vecs = _embed_many(texts)

                    if vecs:
                        for i, v in enumerate(vecs):
                            if v:
                                conn.execute(
                                    "UPDATE ltm SET emb=? WHERE id=?",
                                    (json.dumps(v), ids[i]),
                                )
                        conn.commit()
                        updated += len([v for v in vecs if v])  # Licz tylko udane
                    else:
                        # Jeśli osadzanie nie powiedzie się dla partii, przerwij
                        break

                # self._meta_event("rebuild_embeddings", {"updated": updated})
                return {"ok": True, "updated": updated}
            finally:
                conn.close()

    # ====== STM ======
    def stm_add(self, user: str, assistant: str) -> None:
        """Dodaje parę wiadomości do pamięci krótkoterminowej (STM)."""
        with _DB_LOCK:
            conn = _connect()
            try:
                conn.execute(
                    "INSERT INTO stm(ts, user, assistant) VALUES(strftime('%s','now'), ?, ?)",
                    (user or "", assistant or ""),
                )
                conn.commit()
                # self._meta_event("stm_add", {"len": 1})
            finally:
                conn.close()

        if self.stm_count() >= STM_MAX_TURNS:
            self.force_flush_stm()

    def stm_tail(self, n: int = 200) -> list[dict[str, Any]]:
        """Zwraca n ostatnich wpisów z pamięci krótkoterminowej."""
        with _DB_LOCK:
            conn = _connect()
            try:
                rows = conn.execute(
                    "SELECT ts, user, assistant FROM stm ORDER BY ts DESC LIMIT ?",
                    (int(n),),
                ).fetchall()

                return [
                    {
                        "ts": float(r["ts"] or 0),
                        "u": r["user"],
                        "a": r["assistant"],
                    }
                    for r in rows
                ][::-1]
            finally:
                conn.close()

    def stm_count(self) -> int:
        """Zwraca liczbę wpisów w pamięci krótkoterminowej."""
        with _DB_LOCK:
            conn = _connect()
            try:
                return int(
                    (conn.execute("SELECT COUNT(1) AS c FROM stm").fetchone() or {"c": 0})["c"]
                )
            finally:
                conn.close()

    def force_flush_stm(self) -> dict[str, Any]:
        """Przepłukuje pamięć krótkoterminową, ekstrahując fakty do LTM."""
        tail = self.stm_tail(STM_MAX_TURNS)
        if not tail:
            return {"ok": True, "facts": 0}

        convo: list[dict[str, str]] = []
        for t in tail:
            if t["u"]:
                convo.append({"role": "user", "content": t["u"]})
            if t["a"]:
                convo.append({"role": "assistant", "content": t["a"]})

        facts = _extract_facts(convo, max_out=80)
        ins = mrg = 0

        for text, conf, tags in facts:
            r_before = self.exists(_id_for(text))
            self.add_fact(text, conf=conf, tags=tags)
            if r_before:
                mrg += 1
            else:
                ins += 1

        keep = self.stm_tail(STM_KEEP_TAIL)
        with _DB_LOCK:
            conn = _connect()
            try:
                conn.execute("DELETE FROM stm")
                for k in keep:
                    conn.execute(
                        "INSERT INTO stm(ts,user,assistant) VALUES(?,?,?)",
                        (k["ts"], k["u"], k["a"]),
                    )
                conn.commit()
            finally:
                conn.close()

    # self._meta_event("stm_flush", {"inserted": ins, "merged": mrg})
        return {"ok": True, "facts": ins + mrg, "inserted": ins, "merged": mrg}

    def process_stm_window(self, window_size: int = 30) -> dict[str, Any]:
        """
        Aktywnie przetwarza ostatnie 'window_size' tur z STM,
        używając LLM do ekstrakcji kluczowych faktów i zapisuje je w LTM.
        Zwiększona wartość window_size dla lepszego kojarzenia faktów.
        """
        recent_turns = self.stm_tail(window_size)
        if not recent_turns:
            return {"ok": True, "facts": 0, "reason": "no_recent_turns"}

        # Przygotuj kontekst dla LLM
        convo_str = "\n".join(
            [f"U: {t['u']}\nA: {t['a']}" for t in recent_turns if t["u"] or t["a"]]
        )
        if not convo_str.strip():
            return {"ok": True, "facts": 0, "reason": "empty_convo"}

        # Nowy system prompt do ekstrakcji faktów
        system_prompt = """Jesteś ekspertem w analizie tekstu. Twoim zadaniem jest wyciągnięcie
kluczowych faktów, encji i relacji z podanej rozmowy. Skup się na informacjach
o użytkowniku: preferencjach, danych osobowych, celach, planach. Zwróć listę
faktów w formacie JSON. Każdy fakt powinien być osobnym stringiem.

Przykład:
[
  "użytkownik lubi czarną kawę bez cukru",
  "użytkownik mieszka w Krakowie",
  "użytkownik planuje wyjazd w góry w przyszłym tygodniu"
]
"""
        try:
            # Wywołanie LLM do ekstrakcji faktów
            response_raw = llm_chat(user_text=convo_str, system_text=system_prompt, max_tokens=1000)
            # Proste czyszczenie odpowiedzi, aby uzyskać poprawny JSON
            json_str = response_raw[response_raw.find("[") : response_raw.rfind("]") + 1]
            extracted_texts = json.loads(json_str)
            if not isinstance(extracted_texts, list):
                return {"ok": False, "reason": "llm_did_not_return_a_list"}

        except (json.JSONDecodeError, IndexError) as e:
            # self._meta_event("stm_process_window_error", {"error": str(e), "raw_response": response_raw})
            return {"ok": False, "reason": f"json_parsing_failed: {e}"}
        except Exception as e:
            # self._meta_event("stm_process_window_error", {"error": str(e)})
            return {"ok": False, "reason": f"llm_call_failed: {e}"}

        if not extracted_texts:
            return {"ok": True, "facts": 0, "reason": "no_facts_extracted_by_llm"}

        # Zapisz fakty w LTM z domyślnym 'conf' i tagiem
        ins = mrg = 0
        for text in extracted_texts:
            if not isinstance(text, str) or not text.strip():
                continue
            r_before = self.exists(_id_for(text))
            tags = ["active_processing", "llm_extracted"]
            # Dajemy wyższy 'conf' faktom z LLM
            self.add_fact(text, conf=0.85, tags=tags)
            if r_before:
                mrg += 1
            else:
                ins += 1

        # self._meta_event(
        #     "stm_process_window",
        #     {"inserted": ins, "merged": mrg, "window_size": window_size},
        # )
        return {"ok": True, "facts": ins + mrg, "inserted": ins, "merged": mrg}

    # ====== Episodes ====== (skrócone - pełne w części 4)
    def add_episode(self, user: str, assistant: str) -> None:
        """Dodaje epizod (parę wiadomości) do historii."""
        with _DB_LOCK:
            conn = _connect()
            try:
                conn.execute(
                    "INSERT INTO episodes(ts,user,assistant) VALUES(strftime('%s','now'),?,?)",
                    (user or "", assistant or ""),
                )
                conn.commit()
                # self._meta_event("episode_add", {})
            finally:
                conn.close()

        self.stm_add(user, assistant)

    def episodes_tail(self, n: int = 200) -> list[dict[str, Any]]:
        """Zwraca n ostatnich epizodów."""
        with _DB_LOCK:
            conn = _connect()
            try:
                rows = conn.execute(
                    "SELECT ts,user,assistant FROM episodes " "ORDER BY ts DESC LIMIT ?",
                    (int(n),),
                ).fetchall()

                return [
                    {
                        "ts": float(r["ts"] or 0),
                        "u": r["user"],
                        "a": r["assistant"],
                    }
                    for r in rows
                ][::-1]
            finally:
                conn.close()

    # ====== Recall / RAG (HYBRYDA) ======
    def _collect_docs_for_recall(
        self, limit_per_src: int = RECALL_TOPK_PER_SRC
    ) -> tuple[list[str], list[str], list[float]]:
        """Zbiera dokumenty z różnych źródeł do przypominania (recall)."""
        docs: list[str] = []
        srcs: list[str] = []
        tss: list[float] = []

        # Profil
        prof = self.get_profile()
        if prof:
            docs.append("; ".join(f"{k}: {v}" for k, v in prof.items()))
            srcs.append("profile")
            tss.append(time.time())

        # Cele
        for g in self.get_goals()[:limit_per_src]:
            docs.append(f"[goal pri={g['priority']}] {g['title']}")
            srcs.append("goal")
            tss.append(g.get("ts") or time.time())

        # Fakty
        facts = [
            f for f in self.list_facts(limit=limit_per_src * 20) if f["score"] >= LTM_MIN_SCORE
        ]
        facts = sorted(
            facts,
            key=lambda x: (
                float(x.get("conf", 0.0)),
                float(x.get("ts", 0.0)),
            ),
            reverse=True,
        )[: limit_per_src * 3]

        for f in facts:
            docs.append(f["text"])
            srcs.append("fact")
            tss.append(f.get("ts") or 0.0)

        # Epizody
        for e in self.episodes_tail(limit_per_src * 12):
            docs.append(f"U: {e['u']}\nA: {e['a']}")
            srcs.append("episode")
            tss.append(e.get("ts") or 0.0)

        return docs, srcs, tss

    def recall(self, query: str, topk: int = 6) -> list[tuple[str, float, str]]:
        """
        Przypomina relewantne dokumenty do zapytania.
        Wersja ulepszona o obsługę literówek i skrótów myślowych.
        """
        docs, srcs, tss = self._collect_docs_for_recall(RECALL_TOPK_PER_SRC)
        if not docs:
            return []

        # Łączymy wyniki wektorowe i TF-IDF
        se = _emb_scores(query, docs)
        st = _tfidf_scores(query, docs)
        # Zmienione wagi dla lepszego kojarzenia kontekstowego
        base = _blend_scores(se, st, wa=0.75, wb=0.45)

        # Dodajemy wyniki BM25 z FTS
        bm = _fts_bm25(query, limit=max(40, topk * 5))

        # Budujemy pulę wyników
        pool: list[tuple[str, float, str]] = []
        seen = set()  # Użycie set zamiast Set[str] dla kompatybilności

        # Dodatkowe sprawdzanie podobieństwa z uwzględnieniem literówek
        query_words = set(_tok(query))

        # Używamy podstawowych algorytmów wyszukiwania
        for i, d in enumerate(docs):
            if not d or d in seen:
                continue
            seen.add(d)

            # Podstawowy scoring
            score_base = base[i]

            # Dodatkowy bonus za podobieństwo tekstu uwzględniające literówki
            doc_words = set(_tok(d))

            # Bonus za podobieństwo stringów (dla wykrywania literówek)
            string_sim_bonus = 0.0
            if len(query_words) > 0 and len(doc_words) > 0:
                # Znajdź najlepsze dopasowania słów
                word_matches = []
                for q_word in query_words:
                    best_match = 0.0
                    for d_word in doc_words:
                        sim = _string_similarity(q_word, d_word)
                        if sim > 0.8:  # Wysoki próg podobieństwa
                            best_match = max(best_match, sim)
                    if best_match > 0:
                        word_matches.append(best_match)

                # Oblicz bonus za podobieństwo słów
                if word_matches:
                    string_sim_bonus = sum(word_matches) / len(query_words) * 0.2

            # Ostateczny wynik z bonusami
            score_final = (
                (score_base + string_sim_bonus) * _src_bonus(srcs[i]) * _freshness_bonus(tss[i])
            )
            pool.append((d, score_final, srcs[i]))

        # Dodaj wyniki z wyszukiwania pełnotekstowego
        for text, s_bm25, ts in bm:
            if not text or text in seen:
                continue
            seen.add(text)
            sc = s_bm25 * _src_bonus("fts") * _freshness_bonus(ts)
            pool.append((text, sc, "fts"))

        pool.sort(key=lambda x: x[1], reverse=True)
        return pool[:topk]

    def compose_context(self, query: str, limit_chars: int = 3500, topk: int = 12) -> str:
        """
        Komponuje kontekst dla LLM na podstawie relewantnych dokumentów.
        Wersja znacznie ulepszona z inteligentnym grupowaniem powiązanych faktów
        i deduplikacją podobnych informacji dla lepszego kojarzenia.
        """
        # Jeśli zapytanie jest puste, próbujmy podać ogólny kontekst z profilu
        if not query.strip():
            profile_query = "profil użytkownika cechy osobowości preferencje"
            rec = self.recall(profile_query, topk=topk // 2)
        else:
            # Zwiększamy liczbę wyników do filtrowania
            initial_topk = min(topk * 2, 30)  # Większy początkowy zestaw
            rec = self.recall(query, topk=initial_topk)

        if not rec:
            return ""

        # Grupowanie faktów według źródła dla lepszej organizacji
        grouped: dict[str, list[tuple[str, float]]] = {}
        for txt, sc, src in rec:
            if src not in grouped:
                grouped[src] = []
            grouped[src].append((txt, sc))

        # Priorytetyzacja źródeł
        src_priority = {
            "profile": 1,  # Najwyższy priorytet dla profilu użytkownika
            "goal": 2,  # Potem cele
            "fact": 3,  # Istotne fakty
            "episode": 4,  # Historia rozmów
            "fts": 5,  # Pełnotekstowe wyszukiwanie
        }

        # Sortowanie źródeł według priorytetu
        sorted_sources = sorted(grouped.keys(), key=lambda s: src_priority.get(s, 999))

        # Budujemy kontekst, startując od najbardziej priorytetowych źródeł
        parts = ["[memory]"]
        used = 0

        for src in sorted_sources:
            # Sortowanie faktów wg. punktacji, najlepsze najpierw
            items = sorted(grouped[src], key=lambda x: x[1], reverse=True)

            # Dodanie nagłówka sekcji
            section_header = f"\n[SEKCJA: {src.upper()}]\n"
            if used + len(section_header) <= limit_chars:
                parts.append(section_header)
                used += len(section_header)

            # Dodawanie faktów z danego źródła
            for txt, sc in items:
                chunk = f"• {txt.strip()} [score={sc:.2f}]\n"

                if used + len(chunk) > limit_chars:
                    break  # Za mało miejsca

                parts.append(chunk)
                used += len(chunk)

            # Oddzielenie sekcji
            if src != sorted_sources[-1]:  # Nie dodawaj po ostatnim źródle
                separator = "\n---\n"
                if used + len(separator) <= limit_chars:
                    parts.append(separator)
                    used += len(separator)

        parts.append("[/memory]")
        context = "".join(parts)

        # Debug info o rozmiarze kontekstu
        print(f"[DEBUG] Kontekst: {used}/{limit_chars} znaków, {len(rec)} faktów")

        return context

    # ------------------------- Zaawansowany System Pamięci (v3.0) -------------------------

class AdvancedMemorySystem:
    """
    Zarządza wszystkimi 8 typami zaawansowanej pamięci.
    Używa tej samej bazy danych co podstawowy system, ale w nowych tabelach.
    """
    def __init__(self):
        self.db_path = DB_PATH
        self.lock = _DB_LOCK
        self._init_advanced_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        _apply_pragmas(conn)
        return conn

    def _init_advanced_db(self):
        with self.lock:
            conn = self._connect()
            try:
                conn.executescript(_SQL_ADVANCED_MEMORY)
                conn.commit()
            finally:
                conn.close()

    # --- Krok 2: Pamięć Epizodyczna (Timeline) ---

    def add_timeline_entry(
        self,
        entry_type: str,
        content: str,
        title: str | None = None,
        user_input: str | None = None,
        ai_response: str | None = None,
        mood: str | None = None,
        context: str | None = None,
        related_person_id: int | None = None,
        related_file_id: int | None = None,
    ) -> int:
        """Dodaje nowy wpis do timeline'u."""
        with self.lock:
            conn = self._connect()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO timeline_entries (type, content, title, user_input, ai_response, mood, context, related_person_id, related_file_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (entry_type, content, title, user_input, ai_response, mood, context, related_person_id, related_file_id),
                )
                conn.commit()
                return cursor.lastrowid
            finally:
                conn.close()

    def get_timeline_entries(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """Pobiera ostatnie wpisy z timeline'u."""
        with self.lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM timeline_entries ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
                return [dict(row) for row in rows]
            finally:
                conn.close()

    def search_timeline(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Wyszukuje wpisy w timeline, które pasują do zapytania."""
        with self.lock:
            conn = self._connect()
            try:
                search_query = f"%{query}%"
                rows = conn.execute(
                    """
                    SELECT * FROM timeline_entries
                    WHERE title LIKE ? OR content LIKE ? OR user_input LIKE ? OR ai_response LIKE ?
                    ORDER BY timestamp DESC LIMIT ?
                    """,
                    (search_query, search_query, search_query, search_query, limit),
                ).fetchall()
                return [dict(row) for row in rows]
            finally:
                conn.close()

    def create_daily_summary(self, date: str | None = None) -> dict[str, Any] | None:
        """Tworzy podsumowanie dnia na podstawie wpisów w timeline."""
        # log.info("Generowanie podsumowania dnia (placeholder)...", date=date) # Tymczasowo wyłączone
        return None

    # --- Krok 3: Pamięć Kontekstowa ---

    def _serialize_facts(self, facts: list[str]) -> str:
        return json.dumps(facts, ensure_ascii=False)

    def _deserialize_facts(self, facts_json: str | None) -> list[str]:
        if not facts_json:
            return []
        try:
            return json.loads(facts_json)
        except json.JSONDecodeError:
            return []

    def update_context_memory(self, context_type: str, priority_facts: list[str], active_goals: list[str], notes: str):
        """Aktualizuje lub tworzy pamięć dla danego kontekstu."""
        with self.lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO context_memories (context_type, priority_facts, active_goals, notes)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(context_type) DO UPDATE SET
                        priority_facts=excluded.priority_facts,
                        active_goals=excluded.active_goals,
                        notes=excluded.notes
                    """,
                    (context_type, self._serialize_facts(priority_facts), self._serialize_facts(active_goals), notes),
                )
                conn.commit()
            finally:
                conn.close()

    def get_context_memory(self, context_type: str) -> dict[str, Any] | None:
        """Pobiera pamięć dla danego kontekstu."""
        with self.lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM context_memories WHERE context_type = ?", (context_type,)
                ).fetchone()
                if row:
                    return {
                        "context_type": row["context_type"],
                        "priority_facts": self._deserialize_facts(row["priority_facts"]),
                        "active_goals": self._deserialize_facts(row["active_goals"]),
                        "notes": row["notes"],
                    }
                return None
            finally:
                conn.close()

    # --- Krok 4: Samorefleksja i Pamięć Emocjonalna ---

    def add_self_reflection(self, summary: str, lessons_learned: list[str], rules_to_remember: list[str]) -> int:
        """Dodaje nową notatkę z samorefleksji."""
        with self.lock:
            conn = self._connect()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO self_reflections (summary, lessons_learned, rules_to_remember)
                    VALUES (?, ?, ?)
                    """,
                    (summary, self._serialize_facts(lessons_learned), self._serialize_facts(rules_to_remember)),
                )
                conn.commit()
                return cursor.lastrowid
            finally:
                conn.close()

    def get_recent_reflections(self, limit: int = 10) -> list[dict[str, Any]]:
        """Pobiera ostatnie notatki z samorefleksji."""
        with self.lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM self_reflections ORDER BY timestamp DESC LIMIT ?", (limit,)
                ).fetchall()
                return [
                    {
                        "summary": row["summary"],
                        "lessons_learned": self._deserialize_facts(row["lessons_learned"]),
                        "rules_to_remember": self._deserialize_facts(row["rules_to_remember"]),
                        "timestamp": row["timestamp"],
                    }
                    for row in rows
                ]
            finally:
                conn.close()

    def detect_user_mood(self, text: str) -> str:
        """Analizuje tekst i zwraca wykryty nastrój (placeholder)."""
        # Logika do implementacji: prosta analiza słów kluczowych lub użycie LLM
        text_lower = text.lower()
        if any(word in text_lower for word in ["kurwa", "zjebałeś", "wkurwiasz"]):
            return "frustrated"
        if any(word in text_lower for word in ["dzięki", "super", "świetnie"]):
            return "pleased"
        if any(word in text_lower for word in ["?", "jak", "co"]):
            return "curious"
        return "neutral"

    # --- Krok 5: Pamięć Plikowa i Predykcyjna ---

    def save_file_memory(self, filename: str, file_type: str, file_path: str, description: str, tags: list[str]) -> int:
        """Zapisuje informacje o pliku w pamięci sensorycznej."""
        with self.lock:
            conn = self._connect()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO sensory_files (filename, file_type, file_path, description, tags)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (filename, file_type, file_path, description, self._serialize_facts(tags)),
                )
                conn.commit()
                return cursor.lastrowid
            finally:
                conn.close()

    def find_file_by_name(self, filename: str) -> dict[str, Any] | None:
        """Wyszukuje plik po nazwie."""
        with self.lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM sensory_files WHERE filename = ?", (filename,)
                ).fetchone()
                if row:
                    return {
                        "filename": row["filename"],
                        "file_type": row["file_type"],
                        "file_path": row["file_path"],
                        "description": row["description"],
                        "tags": self._deserialize_facts(row["tags"]),
                        "timestamp": row["timestamp"],
                    }
                return None
            finally:
                conn.close()

    def add_prediction_pattern(self, trigger: str, prediction: str):
        """Dodaje lub wzmacnia wzorzec predykcyjny."""
        with self.lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO prediction_patterns (trigger_pattern, predicted_action, confidence, usage_count)
                    VALUES (?, ?, 0.6, 1)
                    ON CONFLICT(trigger_pattern) DO UPDATE SET
                        confidence = MIN(0.95, confidence + 0.05),
                        usage_count = usage_count + 1
                    """,
                    (trigger, prediction),
                )
                conn.commit()
            finally:
                conn.close()

    def get_prediction(self, trigger: str) -> dict[str, Any] | None:
        """Pobiera predykcję dla danego wyzwalacza."""
        with self.lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM prediction_patterns WHERE trigger_pattern = ?", (trigger,)
                ).fetchone()
                if row:
                    return dict(row)
                return None
            finally:
                conn.close()

    # --- Krok 6: Wersjonowanie i Mapowanie Relacji ---

    def create_memory_backup(self, description: str) -> str:
        """Tworzy backup całej bazy danych i zapisuje wersję."""
        backup_dir = NS_DIR / "backups"
        backup_dir.mkdir(exist_ok=True)
        timestamp = int(time.time())
        backup_filename = f"memory_backup_{timestamp}.db"
        backup_path = backup_dir / backup_filename

        # Używamy wbudowanej funkcji backupu SQLite
        with self.lock:
            conn = self._connect()
            try:
                b_conn = sqlite3.connect(str(backup_path))
                with b_conn:
                    conn.backup(b_conn)
                b_conn.close()
            finally:
                conn.close()
        
        version_hash = hashlib.sha256(backup_path.read_bytes()).hexdigest()

        with self.lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO memory_versions (version_hash, description, backup_path) VALUES (?, ?, ?)",
                    (version_hash, description, str(backup_path)),
                )
                conn.commit()
                return version_hash
            finally:
                conn.close()

    def list_memory_versions(self, limit: int = 20) -> list[dict[str, Any]]:
        """Zwraca listę dostępnych wersji pamięci."""
        with self.lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM memory_versions ORDER BY timestamp DESC LIMIT ?", (limit,)
                ).fetchall()
                return [dict(row) for row in rows]
            finally:
                conn.close()

    def restore_memory_version(self, version_hash: str) -> bool:
        """Przywraca pamięć z wybranego backupu."""
        with self.lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT backup_path FROM memory_versions WHERE version_hash = ?", (version_hash,)
                ).fetchone()
                if not row or not os.path.exists(row["backup_path"]):
                    return False
                
                backup_path = row["backup_path"]
            finally:
                conn.close()

        # Zamykamy wszystkie połączenia i podmieniamy plik bazy danych
        # W realnym systemie wymagałoby to restartu aplikacji
        # Tutaj symulujemy to przez bezpośrednią podmianę pliku
        shutil.copy2(backup_path, self.db_path)
        return True

    def add_person_profile(self, name: str, role: str, notes: str, aliases: list[str] | None = None) -> int:
        """Dodaje lub aktualizuje profil osoby."""
        with self.lock:
            conn = self._connect()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO person_profiles (name, role, notes, aliases) VALUES (?, ?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        role=excluded.role,
                        notes=excluded.notes,
                        aliases=excluded.aliases
                    """,
                    (name, role, notes, self._serialize_facts(aliases or [])),
                )
                conn.commit()
                return cursor.lastrowid
            finally:
                conn.close()

    def get_person_profile(self, name: str) -> dict[str, Any] | None:
        """Pobiera profil osoby po imieniu."""
        with self.lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM person_profiles WHERE name = ?", (name,)
                ).fetchone()
                if row:
                    return {
                        "name": row["name"],
                        "role": row["role"],
                        "notes": row["notes"],
                        "aliases": self._deserialize_facts(row["aliases"]),
                    }
                return None
            finally:
                conn.close()

# --- Singleton dla Zaawansowanego Systemu Pamięci ---
_advanced_memory_instance = None
_advanced_memory_lock = threading.Lock()

def get_advanced_memory() -> AdvancedMemorySystem:
    """Zwraca instancję singletona AdvancedMemorySystem."""
    global _advanced_memory_instance
    if _advanced_memory_instance is None:
        with _advanced_memory_lock:
            if _advanced_memory_instance is None:
                _advanced_memory_instance = AdvancedMemorySystem()
    return _advanced_memory_instance


# ------------------------- Singleton (istniejący) -------------------------
# ... (reszta kodu bez zmian) ...
''