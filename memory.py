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
                self._meta_event("ltm_upsert", {"id": fid})
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
                self._meta_event("ltm_delete", {"id": tid})
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

                self._meta_event("rebuild_embeddings", {"updated": updated})
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
                self._meta_event("stm_add", {"len": 1})
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

        self._meta_event("stm_flush", {"inserted": ins, "merged": mrg})
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
            self._meta_event(
                "stm_process_window_error",
                {"error": str(e), "raw_response": response_raw},
            )
            return {"ok": False, "reason": f"json_parsing_failed: {e}"}
        except Exception as e:
            self._meta_event("stm_process_window_error", {"error": str(e)})
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

        self._meta_event(
            "stm_process_window",
            {"inserted": ins, "merged": mrg, "window_size": window_size},
        )
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
                self._meta_event("episode_add", {})
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

    # ====== Episodes (ciąg dalszy) ======
    def rotate_episodes(self, keep_tail: int = 5000) -> dict[str, Any]:
        """Usuwa starsze epizody, zachowując keep_tail najnowszych."""
        with _DB_LOCK:
            conn = _connect()
            try:
                n = int(
                    (conn.execute("SELECT COUNT(1) c FROM episodes").fetchone() or {"c": 0})["c"]
                )
                if n <= keep_tail:
                    return {"ok": True, "rotated": 0, "kept": n}

                conn.execute(
                    """
                    DELETE FROM episodes
                    WHERE ts NOT IN (
                        SELECT ts FROM episodes ORDER BY ts DESC LIMIT ?
                    )""",
                    (keep_tail,),
                )

                conn.commit()
                self._meta_event("episodes_rotate", {"left": keep_tail})
                return {"ok": True, "rotated": n - keep_tail, "kept": keep_tail}
            finally:
                conn.close()

    def purge_old_episodes(self, older_than_days: int = 90) -> dict[str, Any]:
        """Usuwa epizody starsze niż podana liczba dni."""
        cutoff = time.time() - older_than_days * 86400
        with _DB_LOCK:
            conn = _connect()
            try:
                conn.execute("DELETE FROM episodes WHERE ts < ?", (cutoff,))
                conn.commit()
                self._meta_event("episodes_purge", {"cutoff": cutoff})
                return {"ok": True}
            finally:
                conn.close()

    # ====== Profile ======
    def get_profile(self) -> dict[str, Any]:
        """Pobiera profil użytkownika."""
        with _DB_LOCK:
            conn = _connect()
            try:
                rows = conn.execute("SELECT key, value FROM profile").fetchall()
                out: dict[str, Any] = {}

                for r in rows:
                    try:
                        out[r["key"]] = json.loads(r["value"])
                    except Exception:
                        out[r["key"]] = r["value"]

                return out
            finally:
                conn.close()

    def set_profile_many(self, updates: dict[str, Any]) -> None:
        """Aktualizuje wiele pól profilu jednocześnie."""
        if not updates:
            return

        with _DB_LOCK:
            conn = _connect()
            try:
                for k, v in updates.items():
                    conn.execute(
                        "INSERT INTO profile(key, value) VALUES(?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (k, json.dumps(v, ensure_ascii=False)),
                    )

                conn.commit()
                self._meta_event("profile_set_many", {"n": len(updates)})
            finally:
                conn.close()

    # ====== Goals ======
    def get_goals(self) -> list[dict[str, Any]]:
        """Pobiera listę celów."""
        with _DB_LOCK:
            conn = _connect()
            try:
                rows = conn.execute(
                    "SELECT id, title, priority, tags, ts FROM goals "
                    "ORDER BY priority DESC, ts DESC"
                ).fetchall()

                out = []
                for r in rows:
                    out.append(
                        {
                            "id": r["id"],
                            "title": r["title"],
                            "priority": float(r["priority"] or 0),
                            "tags": json.loads(r["tags"] or "[]"),
                            "ts": float(r["ts"] or 0),
                        }
                    )

                return out
            finally:
                conn.close()

    def add_goal(
        self, title: str, priority: float = 1.0, tags: list[str] | None = None
    ) -> dict[str, Any]:
        """Dodaje nowy cel."""
        t = (title or "").strip()
        if not t:
            return {"ok": False, "reason": "empty"}

        gid = _id_for(t)[:16]
        with _DB_LOCK:
            conn = _connect()
            try:
                row = conn.execute(
                    "SELECT id, priority, tags FROM goals WHERE id=?", (gid,)
                ).fetchone()
                if row:
                    newp = max(float(row["priority"] or 0), float(priority))
                    try:
                        cur_tags = json.loads(row["tags"] or "[]")
                    except Exception:
                        cur_tags = []

                    st = set(cur_tags)
                    st.update(tags or [])

                    conn.execute(
                        "UPDATE goals SET priority=?, tags=?, "
                        "ts=strftime('%s','now') WHERE id=?",
                        (
                            newp,
                            json.dumps(sorted(st), ensure_ascii=False),
                            gid,
                        ),
                    )

                    msg = {"ok": True, "updated": True, "id": gid}
                else:
                    conn.execute(
                        "INSERT INTO goals(id, title, priority, tags, ts) "
                        "VALUES(?, ?, ?, ?, strftime('%s','now'))",
                        (
                            gid,
                            t,
                            float(priority),
                            json.dumps(sorted(set(tags or [])), ensure_ascii=False),
                        ),
                    )

                    msg = {"ok": True, "inserted": True, "id": gid}

                conn.commit()
                self._meta_event("goal_upsert", {"id": gid})
                return msg
            finally:
                conn.close()

    def update_goal(self, gid: str, **fields: Any) -> dict[str, Any]:
        """Aktualizuje istniejący cel."""
        allow = ("title", "priority", "tags")
        sets = []
        vals = []

        for k, v in fields.items():
            if k in allow:
                sets.append(f"{k}=?")
                vals.append(json.dumps(v, ensure_ascii=False) if k == "tags" else v)

        if not sets:
            return {"ok": False, "reason": "no_fields"}

        vals.append(gid)
        with _DB_LOCK:
            conn = _connect()
            try:
                conn.execute(
                    f"UPDATE goals SET {', '.join(sets)}, " f"ts=strftime('%s','now') WHERE id=?",
                    tuple(vals),
                )

                conn.commit()
                self._meta_event("goal_update", {"id": gid})
                return {"ok": True}
            finally:
                conn.close()

    def delete_goal(self, gid: str) -> dict[str, Any]:
        """Usuwa cel o podanym ID."""
        with _DB_LOCK:
            conn = _connect()
            try:
                conn.execute("DELETE FROM goals WHERE id=?", (gid,))
                conn.commit()
                self._meta_event("goal_delete", {"id": gid})
                return {"ok": True}
            finally:
                conn.close()

    # ====== Meta / IO / Admin ======
    def _meta_event(self, kind: str, payload: dict[str, Any]) -> None:
        """Zapisuje zdarzenie metadanych."""
        with _DB_LOCK:
            conn = _connect()
            try:
                conn.execute(
                    "INSERT INTO meta_events(ts, kind, payload) VALUES(strftime('%s','now'), ?, ?)",
                    (kind, json.dumps(payload, ensure_ascii=False)),
                )
                conn.commit()
            finally:
                conn.close()

    def get_meta_events(self, limit: int = 200) -> list[dict[str, Any]]:
        """Pobiera ostatnie zdarzenia metadanych."""
        with _DB_LOCK:
            conn = _connect()
            try:
                rows = conn.execute(
                    "SELECT ts, kind, payload FROM meta_events " "ORDER BY ts DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()

                out: list[dict[str, Any]] = []
                for r in rows:
                    try:
                        payload = json.loads(r["payload"] or "{}")
                    except Exception:
                        payload = {"_raw": r["payload"]}

                    out.append(
                        {
                            "ts": float(r["ts"] or 0),
                            "kind": r["kind"],
                            "payload": payload,
                        }
                    )

                return out
            finally:
                conn.close()

    def stats(self) -> dict[str, Any]:
        """Zwraca statystyki pamięci."""
        with _DB_LOCK:
            conn = _connect()
            try:
                f = conn.execute("SELECT COUNT(1) c FROM ltm").fetchone()["c"]
                s = conn.execute("SELECT COUNT(1) c FROM stm").fetchone()["c"]
                e = conn.execute("SELECT COUNT(1) c FROM episodes").fetchone()["c"]
                p = conn.execute("SELECT COUNT(1) c FROM profile").fetchone()["c"]
                g = conn.execute("SELECT COUNT(1) c FROM goals").fetchone()["c"]

                return {
                    "facts": f,
                    "stm": s,
                    "episodes": e,
                    "profile_keys": p,
                    "goals": g,
                    "namespace": self.ns,
                    "db": str(DB_PATH),
                    "size_mb": round(_db_size_mb(), 2),
                }
            finally:
                conn.close()

    def export_json(self, out_path: str) -> dict[str, Any]:
        """Eksportuje dane pamięci do pliku JSON."""
        from pathlib import Path as _P

        pkg = {
            "ns": self.ns,
            "ts": time.time(),
            "facts": self.list_facts(limit=MAX_LTM_FACTS),
            "episodes": self.episodes_tail(n=100000),
            "profile": self.get_profile(),
            "goals": self.get_goals(),
            "stm": self.stm_tail(n=1000),
        }

        _P(out_path).parent.mkdir(parents=True, exist_ok=True)
        _P(out_path).write_text(json.dumps(pkg, ensure_ascii=False, indent=2), encoding="utf-8")
        self._meta_event("export_json", {"path": out_path})
        return {"ok": True, "path": out_path}

    def import_json(self, in_path: str, merge: bool = True) -> dict[str, Any]:
        """Importuje dane pamięci z pliku JSON."""
        from pathlib import Path as _P

        try:
            pkg = json.loads(_P(in_path).read_text(encoding="utf-8"))
        except Exception as e:
            return {"ok": False, "reason": str(e)}

        if merge:
            # Dodawanie faktów
            for f in pkg.get("facts", []):
                self.add_fact(
                    f.get("text", ""),
                    conf=float(f.get("conf", 0.6)),
                    tags=f.get("tags", []),
                )

            # Aktualizacja profilu
            profile_dict = self.get_profile()
            profile_dict.update(pkg.get("profile") or {})
            self.set_profile_many(profile_dict)

            # Dodawanie celów
            for g in pkg.get("goals", []):
                self.add_goal(
                    g.get("title", ""),
                    priority=float(g.get("priority", 1.0)),
                    tags=g.get("tags", []),
                )

            # Dodawanie epizodów
            for ep in pkg.get("episodes", []):
                self.add_episode(ep.get("u", ""), ep.get("a", ""))

            # Dodawanie STM
            for st in pkg.get("stm", []):
                self.stm_add(st.get("u", ""), st.get("a", ""))
        else:
            # Czyszczenie wszystkich tabel
            with _DB_LOCK:
                conn = _connect()
                try:
                    conn.executescript(
                        "DELETE FROM ltm; DELETE FROM ltm_fts; "
                        "DELETE FROM stm; DELETE FROM episodes; "
                        "DELETE FROM profile; DELETE FROM goals;"
                    )
                    conn.commit()
                finally:
                    conn.close()

            # Dodawanie faktów
            for f in pkg.get("facts", []):
                self.add_fact(
                    f.get("text", ""),
                    conf=float(f.get("conf", 0.6)),
                    tags=f.get("tags", []),
                )

        _prune_lowscore_facts()
        _vacuum_if_needed()
        self._meta_event("import_json", {"path": in_path, "merge": merge})
        return {"ok": True}

    def load_knowledge(
        self, knowledge_data: str | dict, source_name: str = "external", confidence: float = 0.85
    ) -> dict[str, Any]:
        """
        Ładuje wiedzę z tekstu lub słownika do pamięci długoterminowej.

        Args:
            knowledge_data: Tekst lub słownik zawierający wiedzę do załadowania
            source_name: Nazwa źródła wiedzy (używana jako tag)
            confidence: Poziom pewności dla dodawanych faktów (0.0-1.0)

        Returns:
            Słownik z informacjami o wyniku ładowania wiedzy
        """
        tags = ["knowledge", f"src:{source_name}"]
        added_facts = 0
        merged_facts = 0

        # Obsługa tekstu
        if isinstance(knowledge_data, str):
            # Podziel tekst na akapity
            paragraphs = [p.strip() for p in knowledge_data.split("\n\n") if p.strip()]

            # Jeśli tekst jest bardzo długi, podziel go na mniejsze fragmenty
            if len(paragraphs) == 1 and len(paragraphs[0]) > 1000:
                # Podziel na zdania
                sentences = _sentences(paragraphs[0])
                # Grupuj zdania w sensowne fragmenty (maksymalnie ~500 znaków)
                chunks = []
                current_chunk = ""

                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < 500:
                        current_chunk += " " + sentence if current_chunk else sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence

                if current_chunk:
                    chunks.append(current_chunk)

                paragraphs = chunks

            # Dodaj każdy paragraf jako fakt
            for paragraph in paragraphs:
                if paragraph and len(paragraph) > 10:  # Ignoruj zbyt krótkie akapity
                    fact_id = _id_for(paragraph)
                    fact_exists = self.exists(fact_id)

                    # Używamy score zamiast conf i dodajemy tagi w meta
                    meta_data = {"tags": tags} if tags else {}
                    self.add_fact(paragraph, meta_data=meta_data, score=confidence)

                    if fact_exists:
                        merged_facts += 1
                    else:
                        added_facts += 1

        # Obsługa słownika
        elif isinstance(knowledge_data, dict):
            # Przeszukaj słownik rekurencyjnie i dodaj wszystkie wartości tekstowe
            def process_dict(d, path=""):
                nonlocal added_facts, merged_facts

                for key, value in d.items():
                    current_path = f"{path}.{key}" if path else key

                    if isinstance(value, str) and len(value.strip()) > 10:
                        # Dodaj wartość jako fakt
                        fact_text = f"{current_path}: {value.strip()}"
                        fact_id = _id_for(fact_text)
                        fact_exists = self.exists(fact_id)

                        # Używamy score zamiast conf i dodajemy tagi w meta
                        meta_data = {"tags": tags} if tags else {}
                        self.add_fact(fact_text, meta_data=meta_data, score=confidence)

                        if fact_exists:
                            merged_facts += 1
                        else:
                            added_facts += 1

                    elif isinstance(value, dict):
                        # Przeszukaj zagnieżdżony słownik
                        process_dict(value, current_path)

                    elif isinstance(value, (list, tuple)):
                        # Przeszukaj listę i dodaj wszystkie wartości tekstowe
                        for i, item in enumerate(value):
                            if isinstance(item, str) and len(item.strip()) > 10:
                                fact_text = f"{current_path}[{i}]: {item.strip()}"
                                fact_id = _id_for(fact_text)
                                fact_exists = self.exists(fact_id)

                                # Używamy score zamiast conf i dodajemy tagi w meta
                                meta_data = {"tags": tags} if tags else {}
                                self.add_fact(fact_text, meta_data=meta_data, score=confidence)

                                if fact_exists:
                                    merged_facts += 1
                                else:
                                    added_facts += 1
                            elif isinstance(item, dict):
                                process_dict(item, f"{current_path}[{i}]")

            # Rozpocznij przetwarzanie słownika
            process_dict(knowledge_data)

        else:
            return {"ok": False, "reason": "Unsupported data type"}

        # Zapisz zdarzenie w meta
        self._meta_event(
            "load_knowledge", {"source": source_name, "added": added_facts, "merged": merged_facts}
        )

        return {
            "ok": True,
            "added": added_facts,
            "merged": merged_facts,
            "total": added_facts + merged_facts,
            "source": source_name,
        }

    def load_knowledge_from_file(
        self, file_path: str, source_name: str | None = None
    ) -> dict[str, Any]:
        """
        Ładuje wiedzę z pliku do pamięci długoterminowej.
        Wspierane formaty: .txt, .json, .md, .csv

        Args:
            file_path: Ścieżka do pliku z wiedzą
            source_name: Opcjonalna nazwa źródła (jeśli nie podano, zostanie użyta nazwa pliku)

        Returns:
            Słownik z informacjami o wyniku ładowania wiedzy
        """
        from pathlib import Path as _P

        path_obj = _P(file_path)

        if not path_obj.exists():
            return {"ok": False, "reason": f"File not found: {file_path}"}

        # Jeśli nie podano nazwy źródła, użyj nazwy pliku
        if source_name is None:
            source_name = path_obj.stem

        try:
            # Wczytaj zawartość pliku
            if path_obj.suffix.lower() == ".json":
                # Obsługa plików JSON
                content = json.loads(path_obj.read_text(encoding="utf-8"))
                return self.load_knowledge(content, source_name or "external")

            elif path_obj.suffix.lower() == ".csv":
                # Obsługa plików CSV - konwersja do słownika
                import csv

                # Konwertuj listę wierszy do pojedynczego słownika
                # gdzie kluczem jest numer wiersza
                csv_dict = {}
                with open(path_obj, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for i, row in enumerate(reader):
                        csv_dict[f"row_{i}"] = row

                return self.load_knowledge(csv_dict, source_name or "external")

            else:
                # Obsługa plików tekstowych (TXT, MD, itp.)
                content = path_obj.read_text(encoding="utf-8")
                return self.load_knowledge(content, source_name or "external")

        except Exception as e:
            return {"ok": False, "reason": f"Error loading file: {str(e)}"}

    def load_knowledge_from_url(self, url: str, source_name: str | None = None) -> dict[str, Any]:
        """
        Ładuje wiedzę ze strony internetowej do pamięci długoterminowej.

        Args:
            url: Adres URL strony z wiedzą
            source_name: Opcjonalna nazwa źródła (jeśli nie podano, zostanie użyta domena)

        Returns:
            Słownik z informacjami o wyniku ładowania wiedzy
        """
        import re
        from urllib.parse import urlparse

        import requests

        # Jeśli nie podano nazwy źródła, użyj domeny
        if source_name is None:
            parsed_url = urlparse(url)
            source_name = parsed_url.netloc

        try:
            # Pobierz zawartość strony
            response = requests.get(url, timeout=HTTP_TIMEOUT)
            response.raise_for_status()

            content = response.text

            # Spróbuj usunąć tagi HTML i niepotrzebne białe znaki
            content = re.sub(r"<[^>]*>", " ", content)
            content = re.sub(r"\s+", " ", content)

            # Załaduj wiedzę
            return self.load_knowledge(content, source_name)

        except Exception as e:
            return {"ok": False, "reason": f"Error loading URL: {str(e)}"}


# ------------------------- ZAAWANSOWANY SYSTEM PAMIĘCI EPIZODYCZNEJ -------------------------

import datetime
from dataclasses import asdict, dataclass
from enum import Enum


class MoodType(Enum):
    """Typy nastrojów/tonów rozmowy"""

    FRIENDLY = "friendly"
    TECHNICAL = "technical"
    FRUSTRATED = "frustrated"
    URGENT = "urgent"
    RELAXED = "relaxed"
    FOCUSED = "focused"
    CREATIVE = "creative"
    BUSINESS = "business"


class ContextType(Enum):
    """Typy kontekstów/trybów pracy"""

    CODING = "coding"
    CREATIVE_WRITING = "creative_writing"
    BUSINESS = "business"
    LEARNING = "learning"
    DEBUGGING = "debugging"
    PLANNING = "planning"
    CHATTING = "chatting"


@dataclass
class TimelineEntry:
    """Wpis w timeline - reprezentuje jeden dzień/sesję"""

    date: str  # Format: "2025-09-16"
    title: str  # np. "Mordo wrzucał mi ZIP-a, darł się o ucinki"
    summary: str  # Szczegółowy opis co się działo
    context_type: ContextType
    mood: MoodType
    participants: list[str]  # ["Mordo", "Papik", "AI"]
    projects: list[str]  # ["winda-6kw", "mordzix-core"]
    achievements: list[str]  # Co się udało
    problems: list[str]  # Co nie wyszło
    lessons_learned: list[str]  # Wnioski na przyszłość
    files_worked_on: list[str]  # Pliki z którymi pracowaliśmy
    code_snippets: list[str]  # Ważne kawałki kodu
    decisions_made: list[str]  # Podjęte decyzje
    next_actions: list[str]  # Co robić dalej
    emotional_notes: list[str]  # Notatki o nastroju/tonie
    prediction_patterns: list[str]  # Wzorce zachowań do przewidywania
    related_timeline_ids: list[str]  # Powiązane dni/sesje
    confidence_score: float = 0.8

    def to_dict(self) -> dict:
        """Konwersja do słownika z obsługą enumów"""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "TimelineEntry":
        """Tworzenie z słownika z obsługą enumów"""
        if "context_type" in data and isinstance(data["context_type"], str):
            data["context_type"] = ContextType(data["context_type"])
        if "mood" in data and isinstance(data["mood"], str):
            data["mood"] = MoodType(data["mood"])
        return cls(**data)


@dataclass
class PersonProfile:
    """Profil osoby w systemie relacji"""

    name: str
    aliases: list[str]  # ["Mordo", "Boss", "Szef"]
    role: str  # "user", "collaborator", "client"
    expertise: list[str]  # ["python", "devops", "crypto"]
    communication_style: str  # "direct", "detailed", "casual"
    typical_requests: list[str]  # Typowe prośby tej osoby
    mood_patterns: list[str]  # Wzorce nastrojów
    preferred_tone: str  # Preferowany ton odpowiedzi
    projects_involved: list[str]  # Projekty w których uczestniczy
    last_interaction: str  # Data ostatniej interakcji
    relationship_strength: float = 0.5  # 0-1 siła relacji


@dataclass
class ContextMemory:
    """Pamięć kontekstowa dla różnych trybów"""

    context_type: ContextType
    priority_facts: list[str]  # Fakty istotne w tym kontekście
    preferred_tools: list[str]  # Preferowane narzędzia
    common_patterns: list[str]  # Typowe wzorce w tym kontekście
    success_strategies: list[str]  # Sprawdzone strategie
    avoid_patterns: list[str]  # Czego unikać
    typical_workflows: list[str]  # Typowe przepływy pracy
    last_used: str  # Kiedy ostatnio używany
    usage_count: int = 0


@dataclass
class PredictionPattern:
    """Wzorzec do przewidywania kolejnych akcji"""

    trigger_phrase: str  # Co prowadzi do akcji
    predicted_action: str  # Przewidywana akcja
    confidence: float  # Pewność przewidywania
    success_rate: float  # Jak często to się sprawdza
    context_conditions: list[str]  # W jakich warunkach działa
    example_sequence: list[str]  # Przykładowa sekwencja


@dataclass
class SelfReflectionNote:
    """Notatka z samorefleksji AI"""

    date: str
    session_summary: str
    what_worked: list[str]
    what_failed: list[str]
    user_feedback_signals: list[str]  # Sygnały od użytkownika
    improvements_needed: list[str]
    rules_to_remember: list[str]
    strategy_adjustments: list[str]
    next_session_goals: list[str]
    confidence_level: float = 0.7


class AdvancedMemorySystem:
    """Zaawansowany system pamięci epizodycznej z timeline'em"""

    def __init__(self, base_memory: Memory):
        self.base_memory = base_memory
        self._init_advanced_tables()

    def _init_advanced_tables(self):
        """Inicjalizuje dodatkowe tabele dla zaawansowanego systemu"""
        with _DB_LOCK:
            conn = _connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS timeline_entries (
                        id TEXT PRIMARY KEY,
                        date TEXT NOT NULL,
                        data TEXT NOT NULL,
                        ts REAL DEFAULT (strftime('%s','now'))
                    );
                    
                    CREATE TABLE IF NOT EXISTS person_profiles (
                        name TEXT PRIMARY KEY,
                        data TEXT NOT NULL,
                        ts REAL DEFAULT (strftime('%s','now'))
                    );
                    
                    CREATE TABLE IF NOT EXISTS context_memories (
                        context_type TEXT PRIMARY KEY,
                        data TEXT NOT NULL,
                        ts REAL DEFAULT (strftime('%s','now'))
                    );
                    
                    CREATE TABLE IF NOT EXISTS prediction_patterns (
                        id TEXT PRIMARY KEY,
                        trigger_phrase TEXT NOT NULL,
                        data TEXT NOT NULL,
                        success_count INTEGER DEFAULT 0,
                        total_count INTEGER DEFAULT 0,
                        ts REAL DEFAULT (strftime('%s','now'))
                    );
                    
                    CREATE TABLE IF NOT EXISTS self_reflections (
                        id TEXT PRIMARY KEY,
                        date TEXT NOT NULL,
                        data TEXT NOT NULL,
                        ts REAL DEFAULT (strftime('%s','now'))
                    );
                    
                    CREATE TABLE IF NOT EXISTS sensory_files (
                        id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        description TEXT,
                        metadata TEXT,
                        file_path TEXT,
                        tags TEXT DEFAULT '[]',
                        ts REAL DEFAULT (strftime('%s','now'))
                    );
                    
                    CREATE TABLE IF NOT EXISTS memory_versions (
                        version_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        description TEXT,
                        backup_path TEXT,
                        changes_summary TEXT,
                        ts REAL DEFAULT (strftime('%s','now'))
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_timeline_date ON timeline_entries(date);
                    CREATE INDEX IF NOT EXISTS idx_patterns_trigger ON prediction_patterns(trigger_phrase);
                    CREATE INDEX IF NOT EXISTS idx_reflections_date ON self_reflections(date);
                    CREATE INDEX IF NOT EXISTS idx_files_type ON sensory_files(file_type);
                """
                )
                conn.commit()
            finally:
                conn.close()

    # ====== TIMELINE / PAMIĘĆ EPIZODYCZNA ======

    def add_timeline_entry(self, entry: TimelineEntry) -> str:
        """Dodaje wpis do timeline"""
        entry_id = _id_for(f"{entry.date}_{entry.title}")

        with _DB_LOCK:
            conn = _connect()
            try:
                # Sprawdź czy wpis już istnieje
                existing = conn.execute(
                    "SELECT data FROM timeline_entries WHERE id = ?", (entry_id,)
                ).fetchone()

                if existing:
                    # Połącz z istniejącym wpisem
                    existing_data = json.loads(existing["data"])
                    existing_entry = TimelineEntry.from_dict(existing_data)

                    # Połącz listy
                    existing_entry.achievements.extend(entry.achievements)
                    existing_entry.problems.extend(entry.problems)
                    existing_entry.lessons_learned.extend(entry.lessons_learned)
                    existing_entry.files_worked_on.extend(entry.files_worked_on)
                    existing_entry.decisions_made.extend(entry.decisions_made)

                    # Aktualizuj inne pola
                    if entry.summary:
                        existing_entry.summary += f"\n\n{entry.summary}"

                    entry = existing_entry

                conn.execute(
                    "INSERT OR REPLACE INTO timeline_entries (id, date, data) VALUES (?, ?, ?)",
                    (entry_id, entry.date, json.dumps(entry.to_dict(), ensure_ascii=False)),
                )
                conn.commit()

                # Dodaj również jako fakt do LTM
                self.base_memory.add_fact(
                    f"Timeline {entry.date}: {entry.title} - {entry.summary}",
                    meta_data={"tags": ["timeline", "episodic", entry.context_type.value]},
                    score=entry.confidence_score,
                )

                return entry_id
            finally:
                conn.close()

    def get_timeline_entries(
        self,
        date_from: str = None,
        date_to: str = None,
        context_type: ContextType = None,
        limit: int = 50,
    ) -> list[TimelineEntry]:
        """Pobiera wpisy z timeline z filtrami"""
        with _DB_LOCK:
            conn = _connect()
            try:
                query = "SELECT data FROM timeline_entries WHERE 1=1"
                params = []

                if date_from:
                    query += " AND date >= ?"
                    params.append(date_from)
                if date_to:
                    query += " AND date <= ?"
                    params.append(date_to)

                query += " ORDER BY date DESC LIMIT ?"
                params.append(limit)

                rows = conn.execute(query, params).fetchall()

                entries = []
                for row in rows:
                    try:
                        data = json.loads(row["data"])
                        entry = TimelineEntry.from_dict(data)

                        # Filtruj po context_type jeśli podano
                        if context_type is None or entry.context_type == context_type:
                            entries.append(entry)
                    except Exception as e:
                        print(f"Błąd parsowania timeline entry: {e}")

                return entries
            finally:
                conn.close()

    def search_timeline(self, query: str, limit: int = 10) -> list[TimelineEntry]:
        """Wyszukuje w timeline po słowach kluczowych"""
        all_entries = self.get_timeline_entries(limit=1000)

        # Proste wyszukiwanie tekstowe
        query_lower = query.lower()
        matches = []

        for entry in all_entries:
            score = 0.0

            # Sprawdź różne pola
            if query_lower in entry.title.lower():
                score += 2.0
            if query_lower in entry.summary.lower():
                score += 1.5
            if any(query_lower in proj.lower() for proj in entry.projects):
                score += 1.0
            if any(query_lower in file.lower() for file in entry.files_worked_on):
                score += 1.0

            if score > 0:
                matches.append((entry, score))

        # Sortuj po score
        matches.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in matches[:limit]]

    def create_daily_summary(self, date: str = None) -> TimelineEntry:
        """Tworzy automatyczne podsumowanie dnia na podstawie STM i Episodes"""
        if date is None:
            date = datetime.datetime.now().strftime("%Y-%m-%d")

        # Pobierz aktywność z ostatnich 24h
        cutoff_time = time.time() - 86400  # 24h temu

        recent_episodes = []
        stm_entries = self.base_memory.stm_tail(200)

        # Filtruj po czasie
        for entry in stm_entries:
            if entry.get("ts", 0) >= cutoff_time:
                recent_episodes.append(entry)

        if not recent_episodes:
            return None

        # Analizuj zawartość dla automatycznego podsumowania
        all_text = ""
        for episode in recent_episodes:
            all_text += f"U: {episode.get('u', '')}\nA: {episode.get('a', '')}\n"

        # Wyodrębnij informacje
        projects = self._extract_projects(all_text)
        files = self._extract_files(all_text)
        achievements = self._extract_achievements(all_text)
        problems = self._extract_problems(all_text)

        # Określ nastrój i kontekst
        mood = self._detect_mood(all_text)
        context = self._detect_context(all_text)

        # Utwórz wpis
        entry = TimelineEntry(
            date=date,
            title=f"Sesja robocza {date}",
            summary=f"Praca nad {', '.join(projects[:3])} i innymi zadaniami",
            context_type=context,
            mood=mood,
            participants=["User", "AI"],
            projects=projects,
            achievements=achievements,
            problems=problems,
            lessons_learned=[],
            files_worked_on=files,
            code_snippets=[],
            decisions_made=[],
            next_actions=[],
            emotional_notes=[],
            prediction_patterns=[],
            related_timeline_ids=[],
        )

        return entry

    def _extract_projects(self, text: str) -> list[str]:
        """Wyodrębnia nazwy projektów z tekstu"""
        # Proste heurystyki dla typowych nazw projektów
        patterns = [
            r"\b([a-z]+[-_]?[a-z0-9]+(?:[-_][a-z0-9]+)*)\.(py|js|ts|html|css|md)\b",
            r"\bprojekt\s+([a-zA-Z0-9_-]+)\b",
            r"\b([A-Z][a-zA-Z0-9]+(?:[A-Z][a-zA-Z0-9]*)*)\b",  # CamelCase
        ]

        projects = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    projects.add(match[0])
                else:
                    projects.add(match)

        # Dodaj znane projekty
        known_projects = ["mordzix", "runpod", "apkl", "winda", "crypto"]
        for proj in known_projects:
            if proj in text.lower():
                projects.add(proj)

        return list(projects)[:10]

    def _extract_files(self, text: str) -> list[str]:
        """Wyodrębnia nazwy plików z tekstu"""
        pattern = r"\b([a-zA-Z0-9_-]+\.[a-z]{2,4})\b"
        files = set(re.findall(pattern, text))
        return list(files)[:20]

    def _extract_achievements(self, text: str) -> list[str]:
        """Wyodrębnia osiągnięcia z tekstu"""
        achievement_keywords = [
            "naprawione",
            "naprawiłem",
            "działa",
            "ukończone",
            "gotowe",
            "zaimplementowane",
            "dodane",
            "utworzone",
            "poprawione",
        ]

        achievements = []
        sentences = _sentences(text)

        for sentence in sentences:
            for keyword in achievement_keywords:
                if keyword in sentence.lower():
                    achievements.append(sentence.strip())
                    break

        return achievements[:10]

    def _extract_problems(self, text: str) -> list[str]:
        """Wyodrębnia problemy z tekstu"""
        problem_keywords = [
            "błąd",
            "error",
            "problem",
            "nie działa",
            "broken",
            "failed",
            "crash",
            "exception",
            "bug",
            "issue",
        ]

        problems = []
        sentences = _sentences(text)

        for sentence in sentences:
            for keyword in problem_keywords:
                if keyword in sentence.lower():
                    problems.append(sentence.strip())
                    break

        return problems[:10]

    def _detect_mood(self, text: str) -> MoodType:
        """Wykrywa nastrój z tekstu"""
        mood_indicators = {
            MoodType.FRUSTRATED: ["cholera", "kurwa", "zjebane", "nie działa", "error", "shit"],
            MoodType.TECHNICAL: ["function", "class", "import", "def", "return", "algorithm"],
            MoodType.URGENT: ["szybko", "natychmiast", "pilne", "urgent", "asap"],
            MoodType.RELAXED: ["ok", "spoko", "git", "nice", "dzięki", "thanks"],
            MoodType.FOCUSED: ["sprawdź", "przeanalizuj", "zoptymalizuj", "refactor"],
            MoodType.CREATIVE: ["pomysł", "idea", "kreatywne", "design", "concept"],
            MoodType.BUSINESS: ["klient", "sprzedaż", "zysk", "biznes", "marketing"],
        }

        text_lower = text.lower()
        mood_scores = {}

        for mood, keywords in mood_indicators.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                mood_scores[mood] = score

        if mood_scores:
            return max(mood_scores.items(), key=lambda x: x[1])[0]

        return MoodType.FRIENDLY  # domyślny

    def _detect_context(self, text: str) -> ContextType:
        """Wykrywa kontekst z tekstu"""
        context_indicators = {
            ContextType.CODING: ["def", "class", "import", "function", "python", "javascript"],
            ContextType.DEBUGGING: ["error", "bug", "debug", "trace", "exception", "fix"],
            ContextType.CREATIVE_WRITING: ["write", "story", "article", "content", "blog"],
            ContextType.BUSINESS: ["client", "sale", "profit", "marketing", "strategy"],
            ContextType.LEARNING: ["learn", "study", "understand", "explain", "tutorial"],
            ContextType.PLANNING: ["plan", "schedule", "organize", "roadmap", "timeline"],
        }

        text_lower = text.lower()
        context_scores = {}

        for context, keywords in context_indicators.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                context_scores[context] = score

        if context_scores:
            return max(context_scores.items(), key=lambda x: x[1])[0]

        return ContextType.CHATTING  # domyślny

    # ====== PAMIĘĆ KONTEKSTOWA ======

    def save_context_memory(self, context_memory: ContextMemory):
        """Zapisuje pamięć kontekstową"""
        with _DB_LOCK:
            conn = _connect()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO context_memories (context_type, data) VALUES (?, ?)",
                    (
                        context_memory.context_type.value,
                        json.dumps(asdict(context_memory), ensure_ascii=False),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def get_context_memory(self, context_type: ContextType) -> ContextMemory:
        """Pobiera pamięć kontekstową"""
        with _DB_LOCK:
            conn = _connect()
            try:
                row = conn.execute(
                    "SELECT data FROM context_memories WHERE context_type = ?",
                    (context_type.value,),
                ).fetchone()

                if row:
                    data = json.loads(row["data"])
                    data["context_type"] = ContextType(data["context_type"])
                    return ContextMemory(**data)
                else:
                    # Utwórz domyślną pamięć kontekstową
                    return ContextMemory(
                        context_type=context_type,
                        priority_facts=[],
                        preferred_tools=[],
                        common_patterns=[],
                        success_strategies=[],
                        avoid_patterns=[],
                        typical_workflows=[],
                        last_used=datetime.datetime.now().isoformat(),
                    )
            finally:
                conn.close()

    def switch_context(self, new_context: ContextType) -> dict[str, Any]:
        """Przełącza kontekst i przygotowuje odpowiednie fakty"""
        context_memory = self.get_context_memory(new_context)

        # Aktualizuj czas użycia
        context_memory.last_used = datetime.datetime.now().isoformat()
        context_memory.usage_count += 1
        self.save_context_memory(context_memory)

        # Pobierz fakty priorytetowe dla tego kontekstu
        priority_facts = []
        for fact_text in context_memory.priority_facts:
            priority_facts.append(fact_text)

        return {
            "context": new_context.value,
            "priority_facts": priority_facts,
            "preferred_tools": context_memory.preferred_tools,
            "common_patterns": context_memory.common_patterns,
            "success_strategies": context_memory.success_strategies,
        }

    # ====== SELF REFLECTION / AI COACHING ======

    def add_self_reflection(self, reflection: SelfReflectionNote) -> str:
        """Dodaje notatkę z samorefleksji"""
        reflection_id = _id_for(f"reflection_{reflection.date}")

        with _DB_LOCK:
            conn = _connect()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO self_reflections (id, date, data) VALUES (?, ?, ?)",
                    (
                        reflection_id,
                        reflection.date,
                        json.dumps(asdict(reflection), ensure_ascii=False),
                    ),
                )
                conn.commit()

                # Dodaj również jako fakt
                self.base_memory.add_fact(
                    f"Self-reflection {reflection.date}: {reflection.session_summary}",
                    meta_data={"tags": ["self_reflection", "coaching", "ai_learning"]},
                    score=reflection.confidence_level,
                )

                return reflection_id
            finally:
                conn.close()

    def get_recent_reflections(self, limit: int = 10) -> list[SelfReflectionNote]:
        """Pobiera ostatnie refleksje"""
        with _DB_LOCK:
            conn = _connect()
            try:
                rows = conn.execute(
                    "SELECT data FROM self_reflections ORDER BY date DESC LIMIT ?", (limit,)
                ).fetchall()

                reflections = []
                for row in rows:
                    try:
                        data = json.loads(row["data"])
                        reflections.append(SelfReflectionNote(**data))
                    except Exception as e:
                        print(f"Błąd parsowania refleksji: {e}")

                return reflections
            finally:
                conn.close()

    def create_session_reflection(self, session_summary: str) -> SelfReflectionNote:
        """Tworzy automatyczną refleksję po sesji"""
        date = datetime.datetime.now().strftime("%Y-%m-%d")

        # Analizuj ostatnią aktywność
        recent_episodes = self.base_memory.stm_tail(50)

        what_worked = []
        what_failed = []
        user_signals = []

        for episode in recent_episodes:
            user_text = episode.get("u", "").lower()
            ai_text = episode.get("a", "").lower()

            # Pozytywne sygnały
            if any(
                word in user_text for word in ["dzięki", "super", "git", "ok", "działa", "good"]
            ):
                what_worked.append(f"User pozytywnie zareagował na: {ai_text[:100]}...")

            # Negatywne sygnały
            if any(word in user_text for word in ["nie", "źle", "error", "błąd", "wrong"]):
                what_failed.append(f"User negatywnie zareagował na: {ai_text[:100]}...")

            # Sygnały frustracji
            if any(word in user_text for word in ["kurwa", "cholera", "zjebane", "shit"]):
                user_signals.append("Użytkownik wykazuje frustrację")

        reflection = SelfReflectionNote(
            date=date,
            session_summary=session_summary,
            what_worked=what_worked,
            what_failed=what_failed,
            user_feedback_signals=user_signals,
            improvements_needed=[
                "Lepsze rozpoznawanie frustracji użytkownika",
                "Szybsze dostarczanie rozwiązań",
                "Unikanie zbyt długich odpowiedzi gdy user jest zdenerwowany",
            ],
            rules_to_remember=[
                "Jeśli user klnie - przejdź do ultra-konkretnych komend",
                "Nie dawaj szkieletów kodu - zawsze pełne pliki",
                "Sprawdzaj czy komendy nie są puste przed podaniem",
            ],
            strategy_adjustments=[],
            next_session_goals=[],
        )

        return reflection

    # ====== PAMIĘĆ SENSORYCZNA/PLIKOWA ======

    def save_file_memory(
        self,
        filename: str,
        file_type: str,
        description: str = "",
        metadata: dict = None,
        file_path: str = "",
        tags: list[str] = None,
    ) -> str:
        """Zapisuje pamięć o pliku"""
        file_id = _id_for(f"{filename}_{time.time()}")

        if tags is None:
            tags = []

        with _DB_LOCK:
            conn = _connect()
            try:
                conn.execute(
                    """INSERT INTO sensory_files 
                       (id, filename, file_type, description, metadata, file_path, tags) 
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        file_id,
                        filename,
                        file_type,
                        description,
                        json.dumps(metadata or {}, ensure_ascii=False),
                        file_path,
                        json.dumps(tags, ensure_ascii=False),
                    ),
                )
                conn.commit()

                # Dodaj również jako fakt
                self.base_memory.add_fact(
                    f"File memory: {filename} ({file_type}) - {description}",
                    meta_data={"tags": ["file_memory", "sensory", file_type] + tags},
                    score=0.8,
                )

                return file_id
            finally:
                conn.close()

    def search_file_memory(self, query: str, file_type: str = None) -> list[dict]:
        """Wyszukuje w pamięci plików"""
        with _DB_LOCK:
            conn = _connect()
            try:
                base_query = """
                    SELECT filename, file_type, description, metadata, file_path, tags 
                    FROM sensory_files WHERE 1=1
                """
                params = []

                if file_type:
                    base_query += " AND file_type = ?"
                    params.append(file_type)

                if query:
                    base_query += " AND (filename LIKE ? OR description LIKE ?)"
                    params.extend([f"%{query}%", f"%{query}%"])

                base_query += " ORDER BY ts DESC LIMIT 50"

                rows = conn.execute(base_query, params).fetchall()

                results = []
                for row in rows:
                    try:
                        results.append(
                            {
                                "filename": row["filename"],
                                "file_type": row["file_type"],
                                "description": row["description"],
                                "metadata": json.loads(row["metadata"] or "{}"),
                                "file_path": row["file_path"],
                                "tags": json.loads(row["tags"] or "[]"),
                            }
                        )
                    except Exception as e:
                        print(f"Błąd parsowania file memory: {e}")

                return results
            finally:
                conn.close()

    # ====== PAMIĘĆ PREDYKCYJNA ======

    def add_prediction_pattern(self, pattern: PredictionPattern) -> str:
        """Dodaje wzorzec predykcyjny"""
        pattern_id = _id_for(pattern.trigger_phrase)

        with _DB_LOCK:
            conn = _connect()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO prediction_patterns 
                       (id, trigger_phrase, data, success_count, total_count) 
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        pattern_id,
                        pattern.trigger_phrase,
                        json.dumps(asdict(pattern), ensure_ascii=False),
                        int(pattern.success_rate * 100),
                        100,
                    ),
                )
                conn.commit()

                return pattern_id
            finally:
                conn.close()

    def predict_next_action(self, current_input: str) -> list[dict]:
        """Przewiduje następną akcję na podstawie wzorców"""
        with _DB_LOCK:
            conn = _connect()
            try:
                # Znajdź pasujące wzorce
                rows = conn.execute(
                    """SELECT trigger_phrase, data, success_count, total_count 
                       FROM prediction_patterns 
                       WHERE ? LIKE '%' || trigger_phrase || '%'
                       ORDER BY success_count DESC""",
                    (current_input.lower(),),
                ).fetchall()

                predictions = []
                for row in rows:
                    try:
                        data = json.loads(row["data"])
                        pattern = PredictionPattern(**data)

                        success_rate = row["success_count"] / max(1, row["total_count"])

                        predictions.append(
                            {
                                "predicted_action": pattern.predicted_action,
                                "confidence": pattern.confidence,
                                "success_rate": success_rate,
                                "trigger": pattern.trigger_phrase,
                                "context_conditions": pattern.context_conditions,
                            }
                        )
                    except Exception as e:
                        print(f"Błąd parsowania prediction pattern: {e}")

                return predictions[:5]  # Top 5 predictions
            finally:
                conn.close()

    def update_prediction_success(self, trigger_phrase: str, was_successful: bool):
        """Aktualizuje sukces przewidywania"""
        pattern_id = _id_for(trigger_phrase)

        with _DB_LOCK:
            conn = _connect()
            try:
                if was_successful:
                    conn.execute(
                        "UPDATE prediction_patterns SET success_count = success_count + 1, total_count = total_count + 1 WHERE id = ?",
                        (pattern_id,),
                    )
                else:
                    conn.execute(
                        "UPDATE prediction_patterns SET total_count = total_count + 1 WHERE id = ?",
                        (pattern_id,),
                    )
                conn.commit()
            finally:
                conn.close()

    # ====== SYSTEM WERSJONOWANIA PAMIĘCI ======

    def create_memory_backup(self, description: str = "") -> str:
        """Tworzy backup pamięci jak Git commit"""
        timestamp = datetime.datetime.now().isoformat()
        version_id = _id_for(f"backup_{timestamp}")[:12]

        # Ścieżka backupu
        backup_dir = Path(self.base_memory.ns).parent / "memory_backups"
        backup_dir.mkdir(exist_ok=True)
        backup_path = backup_dir / f"memory_backup_{version_id}.json"

        # Eksportuj aktualny stan
        export_result = self.base_memory.export_json(str(backup_path))

        if export_result.get("ok"):
            with _DB_LOCK:
                conn = _connect()
                try:
                    conn.execute(
                        """INSERT INTO memory_versions 
                           (version_id, timestamp, description, backup_path, changes_summary) 
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            version_id,
                            timestamp,
                            description,
                            str(backup_path),
                            "Full memory backup",
                        ),
                    )
                    conn.commit()
                finally:
                    conn.close()

        return version_id

    def restore_memory_version(self, version_id: str) -> dict:
        """Przywraca wersję pamięci"""
        with _DB_LOCK:
            conn = _connect()
            try:
                row = conn.execute(
                    "SELECT backup_path FROM memory_versions WHERE version_id = ?", (version_id,)
                ).fetchone()

                if row and Path(row["backup_path"]).exists():
                    return self.base_memory.import_json(row["backup_path"], merge=False)
                else:
                    return {"ok": False, "reason": "Backup not found"}
            finally:
                conn.close()

    def list_memory_versions(self, limit: int = 20) -> list[dict]:
        """Lista wersji pamięci"""
        with _DB_LOCK:
            conn = _connect()
            try:
                rows = conn.execute(
                    """SELECT version_id, timestamp, description, changes_summary 
                       FROM memory_versions 
                       ORDER BY timestamp DESC LIMIT ?""",
                    (limit,),
                ).fetchall()

                return [dict(row) for row in rows]
            finally:
                conn.close()

    # ====== GRAF RELACJI OSÓB ======

    def save_person_profile(self, profile: PersonProfile):
        """Zapisuje profil osoby"""
        with _DB_LOCK:
            conn = _connect()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO person_profiles (name, data) VALUES (?, ?)",
                    (profile.name, json.dumps(asdict(profile), ensure_ascii=False)),
                )
                conn.commit()

                # Dodaj również jako fakt
                self.base_memory.add_fact(
                    f"Person profile: {profile.name} ({profile.role}) - {profile.communication_style}",
                    meta_data={"tags": ["person_profile", "relationship", profile.role]},
                    score=profile.relationship_strength,
                )
            finally:
                conn.close()

    def get_person_profile(self, name: str) -> PersonProfile:
        """Pobiera profil osoby"""
        with _DB_LOCK:
            conn = _connect()
            try:
                row = conn.execute(
                    "SELECT data FROM person_profiles WHERE name = ?", (name,)
                ).fetchone()

                if row:
                    data = json.loads(row["data"])
                    return PersonProfile(**data)
                else:
                    # Utwórz domyślny profil
                    return PersonProfile(
                        name=name,
                        aliases=[],
                        role="unknown",
                        expertise=[],
                        communication_style="unknown",
                        typical_requests=[],
                        mood_patterns=[],
                        preferred_tone="neutral",
                        projects_involved=[],
                        last_interaction=datetime.datetime.now().isoformat(),
                    )
            finally:
                conn.close()

    def update_person_interaction(self, name: str, interaction_summary: str, mood: str = "neutral"):
        """Aktualizuje profil osoby po interakcji"""
        profile = self.get_person_profile(name)

        profile.last_interaction = datetime.datetime.now().isoformat()
        profile.mood_patterns.append(f"{datetime.datetime.now().strftime('%Y-%m-%d')}: {mood}")

        # Ograniczenie historii nastrojów
        if len(profile.mood_patterns) > 50:
            profile.mood_patterns = profile.mood_patterns[-30:]

        # Zwiększ siłę relacji przy pozytywnych interakcjach
        if any(word in interaction_summary.lower() for word in ["dzięki", "super", "git", "good"]):
            profile.relationship_strength = min(1.0, profile.relationship_strength + 0.05)

        self.save_person_profile(profile)

    def get_relationship_graph(self) -> dict:
        """Zwraca graf wszystkich relacji"""
        with _DB_LOCK:
            conn = _connect()
            try:
                rows = conn.execute("SELECT name, data FROM person_profiles").fetchall()

                graph = {"nodes": [], "edges": []}

                for row in rows:
                    try:
                        data = json.loads(row["data"])
                        profile = PersonProfile(**data)

                        graph["nodes"].append(
                            {
                                "id": profile.name,
                                "role": profile.role,
                                "expertise": profile.expertise,
                                "relationship_strength": profile.relationship_strength,
                                "projects": profile.projects_involved,
                            }
                        )

                        # Dodaj połączenia przez wspólne projekty
                        for project in profile.projects_involved:
                            graph["edges"].append(
                                {
                                    "from": profile.name,
                                    "to": f"project_{project}",
                                    "type": "works_on",
                                }
                            )
                    except Exception as e:
                        print(f"Błąd parsowania person profile: {e}")

                return graph
            finally:
                conn.close()

    # ====== PAMIĘĆ EMOCJONALNA/TONALNA ======

    def detect_user_mood(self, user_input: str) -> dict:
        """Wykrywa nastrój użytkownika"""
        mood_indicators = {
            "frustrated": ["kurwa", "cholera", "zjebane", "shit", "fuck", "damn", "błąd", "error"],
            "happy": ["super", "git", "działa", "dzięki", "good", "great", "excellent"],
            "urgent": ["szybko", "natychmiast", "pilne", "urgent", "asap", "immediately"],
            "confused": ["nie rozumiem", "co to", "jak to", "what", "how", "confused"],
            "focused": ["sprawdź", "przeanalizuj", "zoptymalizuj", "check", "analyze"],
        }

        user_lower = user_input.lower()
        detected_moods = {}

        for mood, keywords in mood_indicators.items():
            count = sum(1 for keyword in keywords if keyword in user_lower)
            if count > 0:
                detected_moods[mood] = count

        primary_mood = "neutral"
        if detected_moods:
            primary_mood = max(detected_moods.items(), key=lambda x: x[1])[0]

        return {
            "primary_mood": primary_mood,
            "mood_scores": detected_moods,
            "recommended_tone": self._get_recommended_tone(primary_mood),
        }

    def _get_recommended_tone(self, mood: str) -> str:
        """Zwraca rekomendowany ton odpowiedzi"""
        tone_mapping = {
            "frustrated": "ultra_concise",  # Bardzo zwięzłe, konkretne komendy
            "happy": "friendly",  # Można być bardziej rozlewny
            "urgent": "direct",  # Bez zbędnych słów
            "confused": "explanatory",  # Szczegółowe wyjaśnienia
            "focused": "technical",  # Techniczne detale
            "neutral": "balanced",  # Zrównoważony ton
        }

        return tone_mapping.get(mood, "balanced")

    def adapt_response_to_mood(self, response: str, user_mood: str) -> str:
        """Adaptuje odpowiedź do nastroju użytkownika"""
        if user_mood == "frustrated":
            # Usuń zbędne słowa, zostaw tylko komendy
            lines = response.split("\n")
            essential_lines = []
            for line in lines:
                if any(
                    indicator in line for indicator in ["```", "$", ">", "cd ", "python", "run"]
                ):
                    essential_lines.append(line)
                elif len(line.strip()) < 100 and any(
                    word in line.lower() for word in ["błąd", "fix", "napraw"]
                ):
                    essential_lines.append(line)

            if essential_lines:
                return "\n".join(essential_lines)

        elif user_mood == "urgent":
            # Znajdź najważniejsze akcje
            lines = response.split("\n")
            priority_lines = []
            for line in lines:
                if line.strip().startswith("1.") or line.strip().startswith("->") or "```" in line:
                    priority_lines.append(line)

            if priority_lines:
                return "SZYBKA AKCJA:\n" + "\n".join(priority_lines[:5])

        return response  # Bez zmian dla innych nastrojów

    # ====== GŁÓWNE API ======

    def process_interaction(
        self, user_input: str, ai_response: str, context_type: ContextType = None
    ) -> dict:
        """Główna funkcja przetwarzająca interakcję"""
        timestamp = datetime.datetime.now()

        # Wykryj nastrój i kontekst
        mood_info = self.detect_user_mood(user_input)
        if context_type is None:
            context_type = self._detect_context(user_input + " " + ai_response)

        # Utwórz/aktualizuj wpis timeline
        date_str = timestamp.strftime("%Y-%m-%d")
        existing_entries = self.get_timeline_entries(date_from=date_str, date_to=date_str, limit=1)

        if existing_entries:
            entry = existing_entries[0]
            entry.summary += f"\n{timestamp.strftime('%H:%M')}: {user_input[:100]}..."
        else:
            entry = TimelineEntry(
                date=date_str,
                title=f"Sesja {date_str}",
                summary=f"{timestamp.strftime('%H:%M')}: {user_input[:100]}...",
                context_type=context_type,
                mood=(
                    MoodType(mood_info["primary_mood"])
                    if mood_info["primary_mood"] in [m.value for m in MoodType]
                    else MoodType.FRIENDLY
                ),
                participants=["User", "AI"],
                projects=self._extract_projects(user_input + " " + ai_response),
                achievements=[],
                problems=[],
                lessons_learned=[],
                files_worked_on=self._extract_files(user_input + " " + ai_response),
                code_snippets=[],
                decisions_made=[],
                next_actions=[],
                emotional_notes=[f"Mood: {mood_info['primary_mood']}"],
                prediction_patterns=[],
                related_timeline_ids=[],
            )

        entry_id = self.add_timeline_entry(entry)

        # Przewidywania dla następnej akcji
        predictions = self.predict_next_action(user_input)

        # Aktualizuj profil użytkownika
        self.update_person_interaction("User", user_input, mood_info["primary_mood"])

        return {
            "timeline_entry_id": entry_id,
            "detected_mood": mood_info,
            "context_type": context_type.value,
            "predictions": predictions,
            "recommended_tone": mood_info["recommended_tone"],
        }


# ====== SINGLETON FACTORY ======

_ADVANCED_MEM_SINGLETON: AdvancedMemorySystem | None = None


def get_advanced_memory(namespace: str = None) -> AdvancedMemorySystem:
    """Zwraca zaawansowany system pamięci"""
    global _ADVANCED_MEM_SINGLETON
    if _ADVANCED_MEM_SINGLETON is None or (
        namespace and _ADVANCED_MEM_SINGLETON.base_memory.ns != namespace
    ):
        base_memory = get_memory(namespace)
        _ADVANCED_MEM_SINGLETON = AdvancedMemorySystem(base_memory)
    return _ADVANCED_MEM_SINGLETON


# ------------------------- Singleton -------------------------
_MEM_SINGLETON: Optional["Memory"] = None


def get_memory(namespace: str | None = None) -> "Memory":
    """Zwraca singleton pamięci dla danej przestrzeni nazw."""
    global _MEM_SINGLETON
    if _MEM_SINGLETON is None or (namespace and _MEM_SINGLETON.ns != namespace):
        _MEM_SINGLETON = Memory(namespace=namespace or MEM_NS)
    return _MEM_SINGLETON


# ------------------------- CLI -------------------------
if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="memory.py — singleton pamięci (RAG HYBRYDA)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stats")
    a = sub.add_parser("add")
    a.add_argument("--text", required=True)
    a.add_argument("--conf", type=float, default=0.6)
    a.add_argument("--tags", default="")

    list_parser = sub.add_parser("list")
    list_parser.add_argument("--limit", type=int, default=20)

    d = sub.add_parser("del")
    d.add_argument("--id", required=True)

    r = sub.add_parser("recall")
    r.add_argument("--q", required=True)
    r.add_argument("--topk", type=int, default=6)

    c = sub.add_parser("ctx")
    c.add_argument("--q", required=True)
    c.add_argument("--topk", type=int, default=8)
    c.add_argument("--limit", type=int, default=2200)

    e = sub.add_parser("export")
    e.add_argument("--out", required=True)

    i = sub.add_parser("import")
    i.add_argument("--in", dest="inp", required=True)
    i.add_argument("--merge", action="store_true")

    # Admin/Meta rozszerzenia
    m = sub.add_parser("meta")
    m.add_argument("--limit", type=int, default=50)

    v = sub.add_parser("vacuum")

    p = sub.add_parser("prune")
    p.add_argument("--target_mb", type=float, default=4200.0)
    p.add_argument("--batch", type=int, default=2000)

    f = sub.add_parser("rebuild_fts")
    f.add_argument("--limit", type=int, default=0)

    g = sub.add_parser("integrity")

    b = sub.add_parser("backup")
    b.add_argument("--out", required=True)

    sub.add_parser("rebuild_embeddings").add_argument("--batch", type=int, default=64)

    args = ap.parse_args()
    mem = get_memory()

    if args.cmd == "meta":
        print(json.dumps(mem.get_meta_events(limit=args.limit), ensure_ascii=False, indent=2))
        sys.exit(0)
    if args.cmd == "vacuum":
        print(json.dumps(_vacuum_if_needed(0), ensure_ascii=False, indent=2))
        sys.exit(0)
    if args.cmd == "prune":
        print(
            json.dumps(
                _prune_lowscore_facts(target_mb=args.target_mb, batch=args.batch),
                ensure_ascii=False,
                indent=2,
            )
        )
        sys.exit(0)
    if args.cmd == "rebuild_fts":
        lim = None if not args.limit or args.limit <= 0 else int(args.limit)
        print(json.dumps(_rebuild_fts(limit=lim), ensure_ascii=False, indent=2))
        sys.exit(0)
    if args.cmd == "integrity":
        print(json.dumps(_integrity_check(), ensure_ascii=False, indent=2))
        sys.exit(0)

    if args.cmd == "backup":
        print(json.dumps(_backup_db(args.out), ensure_ascii=False, indent=2))
        sys.exit(0)

    if args.cmd == "rebuild_embeddings":
        print(
            json.dumps(
                mem.rebuild_missing_embeddings(batch=args.batch),
                ensure_ascii=False,
                indent=2,
            )
        )
        sys.exit(0)

    args = ap.parse_args()
    mem = get_memory()

    if args.cmd == "meta":
        print(
            json.dumps(
                mem.get_meta_events(limit=args.limit),
                ensure_ascii=False,
                indent=2,
            )
        )
        sys.exit(0)
    if args.cmd == "vacuum":
        print(json.dumps(_vacuum_if_needed(0), ensure_ascii=False, indent=2))
        sys.exit(0)
    if args.cmd == "prune":
        print(
            json.dumps(
                _prune_lowscore_facts(target_mb=args.target_mb, batch=args.batch),
                ensure_ascii=False,
                indent=2,
            )
        )
        sys.exit(0)
    if args.cmd == "rebuild_fts":
        lim = None if not args.limit or args.limit <= 0 else int(args.limit)
        print(json.dumps(_rebuild_fts(limit=lim), ensure_ascii=False, indent=2))
        sys.exit(0)
    if args.cmd == "integrity":
        print(json.dumps(_integrity_check(), ensure_ascii=False, indent=2))
        sys.exit(0)
    # Domyślne polecenia
    if args.cmd == "stats":
        print(json.dumps(mem.stats(), ensure_ascii=False, indent=2))
    elif args.cmd == "add":
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        mem.add_fact(args.text, conf=args.conf, tags=tags)
        print(json.dumps({"ok": True, "id": _id_for(args.text)}))
    elif args.cmd == "list":
        print(json.dumps(mem.list_facts(limit=args.limit), ensure_ascii=False, indent=2))
    elif args.cmd == "del":
        mem.delete_fact(args.id)
        print(json.dumps({"ok": True}))
    elif args.cmd == "recall":
        print(json.dumps(mem.recall(args.q, topk=args.topk), ensure_ascii=False, indent=2))
    elif args.cmd == "ctx":
        print(mem.compose_context(args.q, topk=args.topk, limit_chars=args.limit))
    elif args.cmd == "export":
        print(json.dumps(mem.export_json(args.out), ensure_ascii=False, indent=2))
    elif args.cmd == "import":
        print(
            json.dumps(
                mem.import_json(args.inp, merge=args.merge),
                ensure_ascii=False,
                indent=2,
            )
        )
