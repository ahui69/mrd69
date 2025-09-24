#!/usr/bin/env python3
# coding: utf-8
"""
memory.py — hardened ULTRA+ version (full, advanced)

Funkcje kluczowe:
- LTM/STM/Episodes/Profile/Goals/Meta (DB: SQLite + FTS5) — z hybrydowym recall (embeddings+tfidf+bm25+string-sim)
- Sensory memory: pliki (hash/mime/size), opis (LLM lub heurystyka), embedding opisu, wpis do LTM
- Emotional memory: analiza tonu/emocji i zapis do tabeli + tagi
- Samorefleksja: notatki + reguły (LLM jeśli dostępne; fallback heurystyczny) – przy flush STM i /reflect
- Predykcja: łańcuch Markowa na intencjach + reguły (learned_rules) – /predict i automatyczne uczenie
- NER + WSD: regex/heurystyki (działa bez modeli) + opcjonalnie spaCy, WSD pseudo-Lesk bez zewn. danych
- REST API: FastAPI – wszystkie funkcje + upload plików, emocje, ner/wsd, predykcja, self-reflection itd.
- Bezpieczeństwo: SSRF guard, FTS safe query, PII maska XOR, clamp tekstów, circuit-breaker fix, logger

WYMAGANIA (opcjonalne):
- embeddings: ustaw LLM_EMBED_URL + OPENAI_API_KEY
- LLM: OPENAI_API_KEY (używane przez llm_simple.chat jeśli masz swój wrapper)
- REST: pip install fastapi uvicorn
- spaCy (opcjonalnie): pip install spacy pl-core-news-sm  (jeśli chcesz lepszy NER)
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import mimetypes
import os
import config
import re
import shutil
import sqlite3
import threading
import time
import traceback
import random
from collections import deque, defaultdict
from pathlib import Path
from typing import Any, Optional, List, Dict, Tuple, Union, Iterable, Callable

from dotenv import load_dotenv

# Opcjonalna integracja LLM (Twój wrapper)
try:
    from llm_simple import chat as llm_chat  # type: ignore
except Exception:
    def llm_chat(*args, **kwargs):
        return ""

# Opcjonalnie spaCy (NER)
try:
    import spacy
    _SPACY_OK = True
except Exception:
    spacy = None
    _SPACY_OK = False

# ------------------------- Konfiguracja/ENV -------------------------
load_dotenv()

DEBUG = os.getenv("MEM_DEBUG", "0") in ("1", "true", "True")
def log(*a):
    if DEBUG: print(*a)

ROOT = Path(config.MEM_ROOT or str(Path(__file__).parent))
DATA_DIR = ROOT / "data"
MEM_NS = (config.MEM_NS or "default").strip() or "default"
NS_DIR = DATA_DIR / MEM_NS
NS_DIR.mkdir(parents=True, exist_ok=True)

# Storage plików
FILES_DIR = NS_DIR / "files"
FILES_DIR.mkdir(parents=True, exist_ok=True)

USE_RUNPOD = config.USE_RUNPOD
RUNPOD_PATH = config.RUNPOD_PERSIST_DIR or "/runpod/persist"
DB_PATH = (Path(RUNPOD_PATH) / "data" / "ltm.db") if USE_RUNPOD else (DATA_DIR / "memory.db")

LTM_MIN_SCORE = config.LTM_MIN_SCORE
MAX_LTM_FACTS = config.MAX_LTM_FACTS
RECALL_TOPK_PER_SRC = config.RECALL_TOPK_PER_SRC
STM_MAX_TURNS = config.STM_MAX_TURNS
STM_KEEP_TAIL = config.STM_KEEP_TAIL
HTTP_TIMEOUT = config.LLM_HTTP_TIMEOUT
_MAX_TEXT = int(os.getenv("MEM_MAX_TEXT", "4000"))

def _clamp_env():
    global LTM_MIN_SCORE, STM_KEEP_TAIL
    LTM_MIN_SCORE = max(0.0, min(1.0, LTM_MIN_SCORE))
    STM_KEEP_TAIL = min(STM_KEEP_TAIL, STM_MAX_TURNS)
_clamp_env()

# Embeddings
EMBED_URL = config.LLM_EMBED_URL
EMBED_MODEL = (config.LLM_EMBED_MODEL or "text-embedding-3-large").strip()
EMBED_KEY = config.OPENAI_API_KEY.strip()

_RETRY_MAX = int(os.getenv("MEM_RETRY_MAX", "5"))
_RETRY_INITIAL = float(os.getenv("MEM_RETRY_INITIAL", "0.5"))
_CIRCUIT_FAILS_THRESHOLD = int(os.getenv("MEM_CIRCUIT_FAILS", "5"))
_CIRCUIT_OPEN_SECONDS = int(os.getenv("MEM_CIRCUIT_OPEN_S", "300"))

def _clamp_text(s: str) -> str:
    s = (s or "").strip()
    return s if len(s) <= _MAX_TEXT else s[:_MAX_TEXT] + "…"

# ------------------------- Crypto -------------------------
class _Crypto:
    def __init__(self, key: Optional[str]):
        self.key = hashlib.sha256((key or "").encode("utf-8")).digest() if key else None
    def enc(self, text: str) -> str:
        if not self.key: return text
        b = text.encode("utf-8")
        out = bytes(v ^ self.key[i % len(self.key)] for i, v in enumerate(b))
        return "xor:" + base64.urlsafe_b64encode(out).decode("ascii")
    def dec(self, blob: Union[str, Any]) -> str:
        if not self.key or not isinstance(blob, str) or not blob.startswith("xor:"):
            return blob if isinstance(blob, str) else str(blob)
        try:
            raw = base64.urlsafe_b64decode(blob[4:].encode("ascii"))
            out = bytes(v ^ self.key[i % len(self.key)] for i, v in enumerate(raw))
            return out.decode("utf-8", "ignore")
        except Exception:
            return str(blob)

CRYPTO = _Crypto(os.getenv("PSY_ENCRYPT_KEY"))

# ------------------------- Feature Flags -------------------------
def _env_check() -> Dict[str, bool]:
    flags = {
        "EMBED": bool(EMBED_URL and EMBED_KEY),
        "LLM": bool(config.OPENAI_API_KEY),
        "RERANK": bool(os.getenv("LLM_RERANK_URL") or os.getenv("RERANK_URL")),
    }
    log("[MEM]", flags)
    return flags
_FEATURES = _env_check()

# ------------------------- Circuit-breaker / Retry -------------------------
_CB_STATE: Dict[str, Dict[str, Union[int, float]]] = {
    "emb": {"fails": 0, "open_until": 0.0},
    "llm": {"fails": 0, "open_until": 0.0},
    "rerank": {"fails": 0, "open_until": 0.0},
}
def _is_cb_open(kind: str) -> bool:
    s = _CB_STATE.get(kind, {})
    return float(s.get("open_until", 0) or 0) > time.time()
def _record_cb_failure(kind: str) -> None:
    s = _CB_STATE.setdefault(kind, {"fails": 0, "open_until": 0.0})
    s["fails"] = int(s.get("fails", 0)) + 1
    if s["fails"] >= _CIRCUIT_FAILS_THRESHOLD:
        s["open_until"] = time.time() + _CIRCUIT_OPEN_SECONDS
        s["fails"] = 0
        log(f"[CB] circuit OPEN for {kind} until {time.ctime(s['open_until'])}")
def _record_cb_success(kind: str) -> None:
    s = _CB_STATE.setdefault(kind, {"fails": 0, "open_until": 0.0})
    s["fails"] = 0
    s["open_until"] = 0.0
def retry_with_backoff(kind: str):
    def deco(fn: Callable):
        def wrapped(*args, **kwargs):
            if _is_cb_open(kind): raise RuntimeError(f"circuit_open:{kind}")
            delay = _RETRY_INITIAL
            for attempt in range(1, _RETRY_MAX + 1):
                try:
                    res = fn(*args, **kwargs)
                    _record_cb_success(kind)
                    return res
                except Exception:
                    _record_cb_failure(kind)
                    if attempt == _RETRY_MAX: raise
                    sleep = delay * (2 ** (attempt - 1)) * (0.8 + 0.4 * random.random())
                    time.sleep(sleep)
        return wrapped
    return deco

# ------------------------- Tekst/Tokenizacja/TF-IDF -------------------------
def _norm(s: str) -> str:
    return "".join(ch.lower() for ch in (s or "").strip() if ch.isalnum() or ch.isspace())
def _id_for(text: str) -> str:
    return hashlib.sha1(_norm(text).encode("utf-8")).hexdigest()
def _tok(s: str) -> List[str]:
    s = (s or "").lower()
    skroty = {"wg":"według","np":"na przykład","itd":"i tak dalej","itp":"i tym podobne","tzn":"to znaczy",
              "tzw":"tak zwany","dr":"doktor","prof":"profesor","mgr":"magister","ok":"okej","bd":"będzie",
              "jj":"jasne","nwm":"nie wiem","wiadomo":"wiadomo","imo":"moim zdaniem","btw":"przy okazji",
              "tbh":"szczerze mówiąc","fyi":"dla twojej informacji"}
    words = s.split()
    for i,w in enumerate(words):
        cw = re.sub(r"[^\wąćęłńóśźż]", "", w)
        if cw in skroty: words[i] = skroty[cw]
    s2 = re.sub(r"[^0-9a-ząćęłńóśźżA-ZĄĆĘŁŃÓŚŹŻ]+", " ", " ".join(words))
    return [w for w in s2.split() if len(w) > 2][:256]
def _tfidf_vec(tokens: List[str], docs_tokens: List[List[str]]) -> Dict[str, float]:
    N = len(docs_tokens) if docs_tokens else 1
    vocab = set(t for d in docs_tokens for t in d)
    df = {t: sum(1 for d in docs_tokens if t in d) for t in vocab}
    tf: Dict[str, int] = {}
    for t in tokens: tf[t] = tf.get(t, 0) + 1
    out: Dict[str, float] = {}
    for t in tf:
        out[t] = (tf[t]/max(1,len(tokens))) * (math.log((N+1)/(df.get(t,1)+1)))**1.5 * (1 + 0.1*min(len(t)-3,7) if len(t)>3 else 1)
    return out
def _tfidf_cos(q: str, docs: List[str]) -> List[float]:
    tq = _tok(q); dts = [_tok(d) for d in docs]; vq = _tfidf_vec(tq, dts)
    out: List[float] = []; key_terms = set([t for t in tq if len(t)>3])
    for dt in dts:
        vd = _tfidf_vec(dt, dts); keys = set(vq.keys())|set(vd.keys())
        def boost_match(a,b,term):
            val=a*b; term_bonus=2.5 if term in key_terms else 1.0
            if " " in term:
                words=len(term.split()); 
                if words>1: term_bonus*=1.0+0.5*words
            boost=1+0.8*math.tanh(4*val-0.6); return val*boost*term_bonus
        num=sum(boost_match(vq.get(t,0.0),vd.get(t,0.0),t) for t in keys)
        den=(sum(x*x for x in vq.values())**0.5)*(sum(x*x for x in vd.values())**0.5)
        score=0.0 if den==0 else (num/den); score=score**0.8; out.append(score)
    return out

# ------------------------- SQLite schema -------------------------
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
  score REAL DEFAULT 0.6,
  emb TEXT
);
CREATE TABLE IF NOT EXISTS stm (
  id INTEGER PRIMARY KEY,
  ts REAL, user TEXT, assistant TEXT
);
CREATE TABLE IF NOT EXISTS episodes (
  id INTEGER PRIMARY KEY,
  ts REAL, user TEXT, assistant TEXT
);
CREATE TABLE IF NOT EXISTS profile (
  key TEXT PRIMARY KEY, value TEXT
);
CREATE TABLE IF NOT EXISTS goals (
  id TEXT PRIMARY KEY,
  title TEXT,
  priority REAL DEFAULT 1.0,
  tags TEXT DEFAULT '[]',
  ts REAL
);
CREATE TABLE IF NOT EXISTS meta_events ( ts REAL, kind TEXT, payload TEXT );
CREATE INDEX IF NOT EXISTS idx_ltm_ts ON ltm(ts);
CREATE INDEX IF NOT EXISTS idx_ltm_score ON ltm(score);
CREATE INDEX IF NOT EXISTS idx_ltm_ts_score ON ltm(ts, score);
CREATE INDEX IF NOT EXISTS idx_stm_ts ON stm(ts);
CREATE INDEX IF NOT EXISTS idx_eps_ts ON episodes(ts);
CREATE INDEX IF NOT EXISTS idx_goals_pri_ts ON goals(priority, ts);
CREATE INDEX IF NOT EXISTS idx_meta_kind_ts ON meta_events(kind, ts);

-- Sensory memory: files
CREATE TABLE IF NOT EXISTS files (
  id TEXT PRIMARY KEY,
  ts REAL DEFAULT (strftime('%s','now')),
  path TEXT NOT NULL,
  mime TEXT,
  size INTEGER,
  sha1 TEXT,
  desc TEXT,
  emb TEXT,
  meta TEXT
);
CREATE INDEX IF NOT EXISTS idx_files_ts ON files(ts);

-- Emotional memory
CREATE TABLE IF NOT EXISTS emotions (
  id INTEGER PRIMARY KEY,
  ts REAL,
  source TEXT,       -- 'stm','episodes','api','file_desc'
  ref_id TEXT,       -- np. hash STM lub id pliku
  emotion TEXT,      -- joy/sad/anger/fear/surprise/disgust/neutral
  polarity REAL,     -- [-1,1]
  intensity REAL,    -- [0,1]
  meta TEXT
);
CREATE INDEX IF NOT EXISTS idx_emotions_ts ON emotions(ts);

-- Reflections (notes + rules)
CREATE TABLE IF NOT EXISTS reflections (
  id INTEGER PRIMARY KEY,
  ts REAL,
  note TEXT,         -- notatka podsumowująca
  rules TEXT,        -- JSON: lista reguł
  meta TEXT
);

-- Prediction: Markov on intents
CREATE TABLE IF NOT EXISTS intents_markov (
  prev TEXT, next TEXT, cnt INTEGER DEFAULT 0,
  PRIMARY KEY (prev, next)
);
CREATE TABLE IF NOT EXISTS learned_rules (
  id INTEGER PRIMARY KEY,
  ts REAL,
  rule TEXT,         -- prosty warunek->akcja w JSON
  meta TEXT
);
"""

_SQL_FTS = """CREATE VIRTUAL TABLE IF NOT EXISTS ltm_fts USING fts5(id UNINDEXED, text, tokenize='unicode61');"""

_SQL_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS ltm_ai AFTER INSERT ON ltm BEGIN
  INSERT INTO ltm_fts(id, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS ltm_ad AFTER DELETE ON ltm BEGIN
  DELETE FROM ltm_fts WHERE id = old.id;
END;
CREATE TRIGGER IF NOT EXISTS ltm_au AFTER UPDATE OF text ON ltm BEGIN
  INSERT INTO ltm_fts(id, text) VALUES (new.id, new.text)
    ON CONFLICT(id) DO UPDATE SET text=excluded.text;
END;
"""

_HAS_FTS5 = True
_DB_LOCK = threading.RLock()

def _apply_pragmas(conn: sqlite3.Connection) -> None:
    try:
        conn.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            PRAGMA temp_store=MEMORY;
            PRAGMA cache_size=-16000;
            PRAGMA busy_timeout=30000;
            PRAGMA foreign_keys=ON;
        """)
    except Exception:
        pass

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    _apply_pragmas(conn)
    return conn

def _init_db():
    global _HAS_FTS5
    with _DB_LOCK:
        conn = _connect()
        try:
            conn.executescript(_SQL_BASE)
            try: conn.executescript(_SQL_FTS)
            except Exception: _HAS_FTS5 = False
            try: conn.executescript(_SQL_TRIGGERS)
            except Exception: pass
            conn.commit()
        finally:
            conn.close()
def _ensure_schema_upgrades():
    with _DB_LOCK:
        conn = _connect()
        try:
            try: conn.execute("ALTER TABLE ltm ADD COLUMN emb TEXT")
            except Exception: pass
            try: conn.execute("SELECT id FROM ltm_fts LIMIT 1").fetchall()
            except Exception:
                if _HAS_FTS5:
                    try: conn.execute("DROP TABLE IF EXISTS ltm_fts")
                    except Exception: pass
                    conn.executescript(_SQL_FTS)
                    rows = conn.execute("SELECT id,text FROM ltm").fetchall()
                    for r in rows:
                        if r["text"]:
                            try: conn.execute("INSERT INTO ltm_fts(id,text) VALUES(?,?)",(r["id"], r["text"]))
                            except Exception: pass
            conn.commit()
        finally:
            conn.close()
_init_db(); _ensure_schema_upgrades()

# ------------------------- Embeddings helpers -------------------------
def _embed_many(texts: List[str]) -> Optional[List[List[float]]]:
    if not (_FEATURES.get("EMBED") and texts): return None
    if _is_cb_open("emb"): raise RuntimeError("embeddings_circuit_open")
    import requests
    try:
        @retry_with_backoff("emb")
        def _call(texts_inner: List[str]):
            r = requests.post(
                EMBED_URL,
                headers={"Authorization": f"Bearer {EMBED_KEY}", "Content-Type":"application/json"},
                json={"model": EMBED_MODEL, "input": texts_inner},
                timeout=HTTP_TIMEOUT,
            )
            r.raise_for_status()
            j = r.json()
            vecs = [d.get("embedding") for d in j.get("data", [])]
            return vecs if len(vecs)==len(texts_inner) else None
        CHUNK=64; allv=[]
        for i in range(0, len(texts), CHUNK):
            part=texts[i:i+CHUNK]; vecs=_call(part)
            if not vecs: return None
            allv.extend(vecs); time.sleep(0.05+0.02*random.random())
        return allv
    except Exception as e:
        log("[EMBED] error:", e); return None
def _cos(a: List[float], b: List[float]) -> float:
    sa=sum(x*x for x in a)**0.5; sb=sum(x*x for x in b)**0.5
    if sa==0 or sb==0: return 0.0
    return sum(x*y for x,y in zip(a,b))/(sa*sb)

# ------------------------- NLU heurystyki -------------------------
def _sentences(text: str) -> List[str]:
    if not text: return []
    raw = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in raw if len(s.strip()) >= 5]

_PROFILE_PATS = {
    "age": re.compile(r"\b(?:mam|posiadam|skończyłem|skończyłam)\s*(\d{1,2})\s*(?:lat|lata|wiosen|rok|roku)?\b", re.I),
    "email": re.compile(r"\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b", re.I),
    "phone": re.compile(r"\b(?:\+?48[-\s]?)?(?:\d{3}[-\s]?\d{3}[-\s]?\d{3})\b"),
    "city": re.compile(r"\b(?:mieszkam w|jestem z|pochodzę z|mieszkam blisko)\s+([A-ZŁŚŻŹĆĘÓĄ][\w\-ąćęłńóśźż]+)\b", re.I),
    "job": re.compile(r"\b(?:pracuję jako|zawodowo jestem|pracuję w|jestem z zawodu)\s+([a-ząćęłńóśźż\- ]{3,40})\b", re.I),
}
_LANG_PAT = re.compile(r"\b(?:mówię|znam|używam|porozumiewam się|uczę się|rozumiem)\s+(po\s+)?(polsku|angielsku|niemiecku|hiszpańsku|francusku|ukraińsku|rosyjsku|włosku|japońsku|chińsku|koreańsku|portugalsku)\b", re.I)
_TECH_PAT = re.compile(r"\b(R|Python|SQL|Java|JavaScript|TypeScript|C\+\+|C#|Go|Rust|PHP|Ruby|HTML|CSS|Docker|Kubernetes|TensorFlow|PyTorch|React|Angular|Vue|Node\.?js|Django|Flask|Laravel|Spring|Express\.?js|GraphQL|REST|API)\b", re.I)
_HOURS_PAT = re.compile(r"\b(?:pracuję|dostępny)\s+(?:od|w)\s+(\d{1,2})(?:[:.]\d{2})?\s*(?:do|-)\s*(\d{1,2})(?:[:.]\d{2})?\b", re.I)
_LINK_PAT = re.compile(r"\bhttps?://\S+\b", re.I)
_HEALTH_PAT = re.compile(r"\b(alergi[ae]|uczulenie|nietolerancj[ae])\b", re.I)
_NEGATION_PAT = re.compile(r"\b(nie|nie\s+bardzo|żadn[eyoa])\b", re.I)

def _tag_pii_in_text(s: str) -> Tuple[str, List[str]]:
    tags=[]
    if _PROFILE_PATS["email"].search(s): tags.append("pii:email")
    if _PROFILE_PATS["phone"].search(s): tags.append("pii:phone")
    if _LINK_PAT.search(s): tags.append("pii:link")
    return s, sorted(set(tags))

def _mk_fact(text: str, base_score: float, tags: List[str]) -> Tuple[str, float, List[str]]:
    t=(text or "").strip()
    if not t: return ("",0.0,tags)
    score_delta = -0.08 if _NEGATION_PAT.search(t) else 0.04
    score = max(0.5, min(0.95, base_score + score_delta))
    return (t, score, sorted(set(tags)))

def _extract_facts_from_turn(u: str, a: str) -> List[Tuple[str, float, List[str]]]:
    facts=[]
    for role, txt in (("user", u or ""), ("assistant", a or "")):
        for s in _sentences(txt):
            s_clean, pii_tags = _tag_pii_in_text(s)
            if re.search(r"\b(lubię|wolę|preferuję|kocham|nienawidzę|nie\s+lubię|nie\s+cierpię|najczęściej\s+piję|zazwyczaj\s+jadam|uwielbiam|lubię gdy|podoba mi się)\b", s, re.I):
                facts.append(_mk_fact(f"preferencja: {s_clean}", 0.80 if role=="user" else 0.72, ["stm","preference"]+pii_tags)); continue
            m=_PROFILE_PATS["age"].search(s)
            if m: facts.append(_mk_fact(f"wiek: {m.group(1)}", 0.86 if role=="user" else 0.78, ["stm","profile"])); continue
            m=_PROFILE_PATS["email"].search(s)
            if m: facts.append(_mk_fact(f"email: {m.group(0)}", 0.89 if role=="user" else 0.81, ["stm","profile","contact","pii:email"])); continue
            m=_PROFILE_PATS["phone"].search(s)
            if m: facts.append(_mk_fact(f"telefon: {m.group(0)}", 0.88 if role=="user" else 0.8, ["stm","profile","contact","pii:phone"])); continue
            m=_PROFILE_PATS["city"].search(s)
            if m: facts.append(_mk_fact(f"miasto: {m.group(1)}", 0.87 if role=="user" else 0.79, ["stm","profile"])); continue
            m=_PROFILE_PATS["job"].search(s)
            if m: facts.append(_mk_fact(f"zawód: {m.group(1)}", 0.85 if role=="user" else 0.77, ["stm","profile"])); continue
            for lang in _LANG_PAT.findall(s): facts.append(_mk_fact(f"język: {lang[1].lower()}", 0.78 if role=="user" else 0.7, ["stm","profile","language"]))
            for tech in set(t.group(0) for t in _TECH_PAT.finditer(s)): facts.append(_mk_fact(f"tech: {tech}", 0.77 if role=="user" else 0.69, ["stm","skill","tech"]))
            mh=_HOURS_PAT.search(s)
            if mh: facts.append(_mk_fact(f"availability: {mh.group(1)}-{mh.group(2)}", 0.75 if role=="user" else 0.67, ["stm","availability"]))
            for url in _LINK_PAT.findall(s): facts.append(_mk_fact(f"link: {url}", 0.8, ["stm","link","pii:link"]))
            if _HEALTH_PAT.search(s): facts.append(_mk_fact(f"zdrowie: {s_clean}", 0.82 if role=="user" else 0.74, ["stm","health"]))
    return facts

def _dedupe_facts(facts: List[Tuple[str,float,List[str]]]) -> List[Tuple[str,float,List[str]]]:
    by: Dict[str,Tuple[str,float,List[str]]]={}
    for t,sc,tg in facts:
        t2=(t or "").strip()
        if not t2:
            continue
        fid=_id_for(t2)
        if fid in by:
            ot, os, otg = by[fid]
            by[fid]=(ot, max(os,sc), sorted(set((otg or [])+(tg or []))))
        else:
            by[fid]=(t2,sc,sorted(set(tg or [])))
    return list(by.values())

def _extract_facts(messages: List[dict], max_out: int = 120) -> List[Tuple[str,float,List[str]]]:
    if not messages: return []
    all_facts=[]; i=0; full_context=""
    for msg in messages:
        content=msg.get("content",""); role=msg.get("role","")
        if content: full_context+=f"{role.upper()}: {content}\n\n"
    while i<len(messages):
        role_i=messages[i].get("role")
        u=messages[i].get("content","") if role_i=="user" else ""; a=""
        if i+1<len(messages) and messages[i+1].get("role")=="assistant":
            a=messages[i+1].get("content",""); i+=2
        else: i+=1
        all_facts.extend(_extract_facts_from_turn(u,a))
    all_facts=_dedupe_facts(all_facts)
    all_facts.sort(key=lambda x:x[1], reverse=True)
    return all_facts[:max_out]

# ------------------------- DB utils -------------------------
def _db_size_mb()->float:
    try:
        base=DB_PATH.stat().st_size if DB_PATH.exists() else 0
        for suf in ("-wal","-shm"):
            side=DB_PATH.with_name(DB_PATH.name+suf)
            base+=side.stat().st_size if side.exists() else 0
        return base/(1024*1024)
    except Exception:
        return 0.0

def _vacuum_if_needed(threshold_mb: float = 4500.0)->Dict[str,Any]:
    size=_db_size_mb()
    if size<threshold_mb: return {"ok":True,"size_mb":round(size,2),"action":"none"}
    with _DB_LOCK:
        conn=_connect()
        try: conn.execute("VACUUM"); conn.commit()
        finally: conn.close()
    return {"ok":True,"size_mb":round(_db_size_mb(),2),"action":"vacuum"}

def _prune_lowscore_facts(target_mb: float = 4200.0, batch: int = 2000)->Dict[str,Any]:
    size=_db_size_mb()
    if size<target_mb: return {"ok":True,"pruned":0,"size_mb":round(size,2)}
    removed=0
    with _DB_LOCK:
        conn=_connect()
        try:
            rows=conn.execute("""SELECT id FROM ltm WHERE score < ? ORDER BY ts ASC LIMIT ?""",
                              (max(0.15, LTM_MIN_SCORE-0.15), int(batch))).fetchall()
            ids=[r["id"] for r in rows]
            if ids:
                q=",".join("?"*len(ids))
                conn.execute(f"DELETE FROM ltm WHERE id IN ({q})", ids)
                if _HAS_FTS5:
                    try: conn.execute(f"DELETE FROM ltm_fts WHERE id IN ({q})", ids)
                    except Exception: pass
                conn.commit(); removed=len(ids)
        finally: conn.close()
    return {"ok":True,"pruned":removed,"size_mb":round(_db_size_mb(),2)}

def _integrity_check()->Dict[str,Any]:
    with _DB_LOCK:
        conn=_connect()
        try:
            ok=conn.execute("PRAGMA integrity_check").fetchone()[0]
            freelist=conn.execute("PRAGMA freelist_count").fetchone()[0]
            return {"ok": ok=="ok","result":ok,"freelist":int(freelist)}
        finally: conn.close()

def _backup_db(out_path: str)->Dict[str,Any]:
    out=Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    _vacuum_if_needed(0)
    try:
        shutil.copy2(DB_PATH, out)
        for suf in ("-wal","-shm"):
            side=DB_PATH.with_name(DB_PATH.name+suf)
            if side.exists(): shutil.copy2(side, out.with_name(out.name+suf))
        return {"ok":True,"path":str(out)}
    except Exception as e:
        return {"ok":False,"reason":str(e)}

# ------------------------- FTS BM25 -------------------------
def _fts_safe_query(q: str)->str:
    toks=[t for t in _tok(q) if t]
    if not toks: return '""'
    return " AND ".join(f'"{t}"' for t in toks[:8])
def _fts_bm25(query: str, limit: int = 50)->List[Tuple[str,float,float]]:
    if not _HAS_FTS5 or not query: return []
    safe=_fts_safe_query(query)
    with _DB_LOCK:
        conn=_connect()
        try:
            rows=conn.execute("""
              SELECT ltm_fts.id AS id, bm25(ltm_fts) AS bscore
              FROM ltm_fts WHERE ltm_fts MATCH ?
              ORDER BY bscore ASC LIMIT ?""",(safe,int(limit))).fetchall()
            if not rows: return []
            ids=[r["id"] for r in rows]; ph=",".join("?"*len(ids))
            got=conn.execute(f"SELECT id,text,ts FROM ltm WHERE id IN ({ph})", ids).fetchall()
            meta={r["id"]:(r["text"], float(r["ts"] or 0.0)) for r in got}
            out=[]
            for r in rows:
                t,ts=meta.get(r["id"],("",0.0)); 
                if not t: 
                    continue
                try: bscore=float(r["bscore"] or 0.0)
                except: bscore=0.0
                score=1.0/(1.0+bscore); out.append((t,score,ts))
            return out
        except Exception as e:
            log("[FTS] error:", e); return []
        finally: conn.close()

def _rebuild_fts(limit: Optional[int]=None)->Dict[str,Any]:
    if not _HAS_FTS5: return {"ok":False,"reason":"fts5_unavailable"}
    with _DB_LOCK:
        conn=_connect()
        try:
            try: conn.execute("DELETE FROM ltm_fts")
            except Exception: pass
            q="SELECT id,text FROM ltm ORDER BY ts DESC"
            if limit: q+=f" LIMIT {int(limit)}"
            rows=conn.execute(q).fetchall()
            for r in rows:
                if r["text"]:
                    try: conn.execute("INSERT INTO ltm_fts(id,text) VALUES(?,?)",(r["id"],r["text"]))
                    except Exception: pass
            conn.commit(); return {"ok":True,"indexed":len(rows)}
        finally: conn.close()

# ------------------------- Scoring hybrydowy -------------------------
def _emb_scores(query: str, docs: List[str]) -> List[float]:
    try:
        vecs=_embed_many([query]+docs)
        if not vecs: return []
        qv,dvs=vecs[0],vecs[1:]; return [_cos(qv,d) for d in dvs]
    except Exception: return []
def _tfidf_scores(query: str, docs: List[str]) -> List[float]:
    try: return _tfidf_cos(query, docs)
    except Exception: return [0.0]*len(docs)
def _string_similarity(s1: str, s2: str)->float:
    if not s1 or not s2: return 0.0
    def lev(a,b):
        if not a: return len(b)
        if not b: return len(a)
        la,lb=len(a),len(b); prev=list(range(lb+1))
        for i,ca in enumerate(a,1):
            cur=[i]+[0]*lb
            for j,cb in enumerate(b,1):
                cost=0 if ca==cb else 1
                cur[j]=min(prev[j]+1, cur[j-1]+1, prev[j-1]+cost)
            prev=cur
        return prev[lb]
    max_len=max(len(s1),len(s2)); dist=lev(s1.lower(), s2.lower())
    return max(0.0, min(1.0, 1.0 - dist/max_len))
def _blend_scores(a: List[float], b: List[float], wa: float, wb: float) -> List[float]:
    if not a: a=[0.0]*len(b)
    if not b: b=[0.0]*len(a)
    n=min(len(a),len(b)); out=[]
    for i in range(n):
        a_score=a[i]**1.2; b_score=b[i]**1.2
        harm=0.0
        if a[i]>0.35 and b[i]>0.35:
            harm=0.3*((a[i]*b[i])**0.5)
            if a[i]>0.7 and b[i]>0.7: harm*=1.5
        out.append(wa*a_score + wb*b_score + harm)
    return out
def _src_bonus(src:str)->float:
    return {"profile":1.35,"goal":1.30,"fact":1.25,"episode":1.20,"fts":1.22}.get(src,1.0)
def _freshness_bonus(ts: float)->float:
    if not ts: return 1.0
    age=(time.time()-ts)/86400.0
    rec=0.2*math.exp(-age) if age<3.0 else 0.0
    base=max(0.75, 1.4 - (age/180.0))
    return base+rec

# ------------------------- Emocje (heurystyki) -------------------------
_EMO_LEX = {
    "joy": {"super","kocham","uwielbiam","świetnie","dobra","fajnie","zadowolony","szczęśliwy"},
    "sad": {"smutny","przykro","żałuję","zdołowany","płakać","depresyjny"},
    "anger": {"wkurzony","wściekły","nienawidzę","zły","wnerwia","wkurza"},
    "fear": {"boję","lęk","strach","obawiam","przeraża"},
    "surprise": {"zaskoczony","wow","what","niesamowite","szok"},
    "disgust": {"obrzydliwe","odraza","ble","fe","ohyda"},
}
def analyze_emotion(text: str)->Tuple[str,float,float,Dict[str,float]]:
    t=(text or "").lower()
    scores={k:0 for k in _EMO_LEX}
    words=set(_tok(t))
    for emo,keys in _EMO_LEX.items():
        scores[emo]=sum(1 for w in words if w in keys)
    emo=max(scores, key=lambda k: scores[k]) if any(scores.values()) else "neutral"
    total=sum(scores.values()); intensity=min(1.0, total/5.0) if total>0 else 0.0
    # prosta polaryzacja
    polarity={"joy":1,"surprise":0.3,"neutral":0,"sad":-0.5,"fear":-0.6,"anger":-0.7,"disgust":-0.7}.get(emo,0.0)
    return emo, polarity, intensity, {k:float(v) for k,v in scores.items()}

# ------------------------- NER + WSD -------------------------
def _ner_basic(text: str)->List[Dict[str,Any]]:
    ents=[]
    for m in _PROFILE_PATS["email"].finditer(text): ents.append({"text":m.group(0),"label":"EMAIL"})
    for m in _PROFILE_PATS["phone"].finditer(text): ents.append({"text":m.group(0),"label":"PHONE"})
    for m in _LINK_PAT.finditer(text): ents.append({"text":m.group(0),"label":"URL"})
    for m in re.finditer(r"\b\d{1,2}\s?(?:stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|października|listopada|grudnia)\b", text, re.I):
        ents.append({"text":m.group(0),"label":"DATE"})
    for m in re.finditer(r"\b[A-ZŁŚŻŹĆĘÓĄ][a-ząćęłńóśźż]{2,}(?:\s+[A-ZŁŚŻŹĆĘÓĄ][a-ząćęłńóśźż]{2,})*\b", text):
        if not re.search(r"^(?:http|www\.)", m.group(0), re.I):
            ents.append({"text":m.group(0),"label":"PROPN"})
    return ents

_SPACY_NLP = None
def _ensure_spacy():
    global _SPACY_NLP
    if _SPACY_OK and _SPACY_NLP is None:
        try:
            _SPACY_NLP = spacy.load("pl_core_news_sm")
        except Exception:
            _SPACY_NLP = None

def ner(text: str)->List[Dict[str,Any]]:
    _ensure_spacy()
    if _SPACY_NLP:
        doc=_SPACY_NLP(text)
        return [{"text":ent.text,"label":ent.label_} for ent in doc.ents]
    return _ner_basic(text)

# WSD – uproszczony Lesk: definicje sensów w mini-słowniku + overlap z kontekstem
_WSD_DICT = {
    "zamek":[
        {"sense":"castle","def":"budowla obronna mieszkalna dawna rezydencja","ex":["średniowieczny zamek","mury zamku"]},
        {"sense":"lock","def":"mechanizm do zamykania drzwi kłódka","ex":["zamek w drzwiach","kłódka zamek"]},
        {"sense":"zipper","def":"zapięcie suwak w ubraniu","ex":["zamek w kurtce","zapiąć suwak"]},
    ],
}
def wsd(word: str, context: str)->Dict[str,Any]:
    w=word.lower(); ctx=set(_tok(context))
    senses=_WSD_DICT.get(w,[])
    if not senses:
        return {"word":word,"sense":"unknown","confidence":0.0}
    best=None; best_score=-1
    for s in senses:
        tokens=set(_tok(s["def"]+" "+" ".join(s.get("ex",[]))))
        score=len(tokens & ctx)
        if score>best_score:
            best_score=score; best=s
    conf=min(1.0, best_score/3.0) if best_score>0 else 0.2
    return {"word":word,"sense":best["sense"],"confidence":conf}

# ------------------------- Sensory memory (pliki) -------------------------
def _sha1_file(p: Path)->str:
    h=hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _describe_file_heuristic(filename: str, mime: str, size: int)->str:
    base=Path(filename).name
    hint=re.sub(r"[_\-]+"," ", Path(base).stem)
    return f"Plik '{base}' ({mime or 'application/octet-stream'}, {size} bajtów). Słowa kluczowe: {hint}."

def _save_uploaded_file(bytez: bytes, filename: str)->Path:
    safe=re.sub(r"[^0-9A-Za-z._-]+","_", filename)
    p=FILES_DIR / f"{int(time.time())}_{safe}"
    with p.open("wb") as f: f.write(bytez)
    return p

# ------------------------- Memory class -------------------------
class Memory:
    def __init__(self, namespace: str = MEM_NS):
        self.ns = namespace
        NS_DIR.mkdir(parents=True, exist_ok=True)
        self.context_keywords: deque[str] = deque(maxlen=5)
        self.user_message_count = 0
        self._lock = threading.RLock()

    # ---------------------- DB helpers ----------------------
    def _conn(self): return _connect()

    # ---------------------- Facts ----------------------
    def add_entry(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        tags=["entry"]
        if metadata:
            if "source" in metadata: tags.append(f"src:{metadata['source']}")
            if "filename" in metadata: tags.append("file")
        return self.add_fact(text, meta_data={"tags":tags}, score=0.75)

    def add_fact(self, text: str, meta_data: Optional[Dict]=None, score: float=0.6, emb: Optional[List[float]]=None, tags: Optional[List[str]]=None, conf: Optional[float]=None) -> str:
        txt=_clamp_text(text or "")
        if not txt: raise ValueError("empty")
        if conf is not None and score==0.6: score=conf
        if tags:
            meta_data=meta_data or {}; cur=meta_data.get("tags", [])
            meta_data["tags"]=sorted(set(cur + tags))
        meta_data=meta_data or {}
        _, pii_tags=_tag_pii_in_text(txt)
        if pii_tags:
            meta_data.setdefault("tags", [])
            meta_data["tags"]=sorted(set(meta_data["tags"]+pii_tags))
        txt_stored = CRYPTO.enc(txt) if pii_tags else txt
        meta_json=json.dumps(meta_data, ensure_ascii=False)
        fid=_id_for(txt)
        with _DB_LOCK:
            conn=self._conn()
            try:
                row=conn.execute("SELECT id,meta,score FROM ltm WHERE id=?", (fid,)).fetchone()
                if row:
                    try:
                        cur_meta=json.loads(row["meta"] or "{}")
                        for k,v in meta_data.items():
                            if k not in cur_meta: cur_meta[k]=v
                            elif k=="tags" and isinstance(cur_meta.get(k),list):
                                st=set(cur_meta[k]); st.update(meta_data.get(k,[])); cur_meta[k]=sorted(st)
                        meta_json=json.dumps(cur_meta, ensure_ascii=False)
                        new_score=max(float(row["score"] or 0), float(score))
                    except Exception:
                        new_score=score
                    conn.execute("UPDATE ltm SET meta=?, score=?, ts=strftime('%s','now'), text=? WHERE id=?", (meta_json, new_score, txt_stored, fid))
                    if _HAS_FTS5:
                        try:
                            conn.execute("INSERT INTO ltm_fts(id,text) VALUES(?,?) ON CONFLICT(id) DO UPDATE SET text=excluded.text", (fid, txt_stored))
                        except Exception:
                            try:
                                conn.execute("DELETE FROM ltm_fts WHERE id=?", (fid,))
                                conn.execute("INSERT INTO ltm_fts(id,text) VALUES(?,?)",(fid, txt_stored))
                            except Exception: pass
                else:
                    conn.execute("INSERT INTO ltm(id,kind,text,meta,score,emb,ts) VALUES(?,?,?,?,?,?,strftime('%s','now'))",
                                 (fid,"fact",txt_stored,meta_json,float(score), json.dumps(emb) if emb else None))
                    if _HAS_FTS5:
                        try: conn.execute("INSERT INTO ltm_fts(id,text) VALUES(?,?)",(fid, txt_stored))
                        except Exception: log("Warning: cannot add to FTS")
                conn.commit(); self._meta_event("ltm_upsert", {"id":fid}); return fid
            finally: conn.close()

    def add_fact_bulk(self, rows: List[Tuple[str,float,Optional[List[str]]]])->Dict[str,int]:
        ins=mrg=0
        for t,c,tg in rows:
            before=self.exists(_id_for(t)); self.add_fact(t, score=c, meta_data={"tags":tg})
            if before: mrg+=1
            else: ins+=1
        return {"inserted":ins,"merged":mrg}

    def exists(self, fid: str)->bool:
        with _DB_LOCK:
            conn=self._conn()
            try: return bool(conn.execute("SELECT 1 FROM ltm WHERE id=?", (fid,)).fetchone())
            finally: conn.close()

    def delete_fact(self, id_or_text: str)->bool:
        tid=id_or_text if re.fullmatch(r"[0-9a-f]{40}", id_or_text) else _id_for(id_or_text)
        with _DB_LOCK:
            conn=self._conn()
            try:
                conn.execute("DELETE FROM ltm WHERE id=?", (tid,))
                if _HAS_FTS5:
                    try: conn.execute("DELETE FROM ltm_fts WHERE id=?", (tid,))
                    except Exception: pass
                conn.commit(); self._meta_event("ltm_delete", {"id":tid}); return True
            finally: conn.close()

    def list_facts(self, limit: int=500)->List[Dict[str,Any]]:
        with _DB_LOCK:
            conn=self._conn()
            try:
                rows=conn.execute("SELECT id,text,meta,score,ts FROM ltm ORDER BY ts DESC LIMIT ?", (int(limit),)).fetchall()
                out=[]
                for r in rows:
                    meta=json.loads(r["meta"] or "{}") if r["meta"] else {}
                    out.append({
                        "id": r["id"],
                        "text": CRYPTO.dec(r["text"]),
                        "conf": float(r["score"] or 0),
                        "tags": meta.get("tags",[]),
                        "ts": float(r["ts"] or 0),
                        "score": float(r["score"] or 0),
                        "meta": meta,
                    })
                return out
            finally: conn.close()

    # ---------------------- STM/Episodes ----------------------
    def stm_add(self, user: str, assistant: str)->None:
        if user: self.user_message_count += 1
        with _DB_LOCK:
            conn=self._conn()
            try:
                conn.execute("INSERT INTO stm(ts,user,assistant) VALUES(strftime('%s','now'),?,?)",(user or "", assistant or ""))
                conn.commit(); self._meta_event("stm_add", {"len":1})
            finally: conn.close()
        if self.stm_count() >= STM_MAX_TURNS: self.force_flush_stm()

    def stm_tail(self, n: int=200)->List[Dict[str,Any]]:
        with _DB_LOCK:
            conn=self._conn()
            try:
                rows=conn.execute("SELECT ts,user,assistant FROM stm ORDER BY ts DESC LIMIT ?", (int(n),)).fetchall()
                return [{"ts":float(r["ts"] or 0),"u":r["user"],"a":r["assistant"]} for r in rows][::-1]
            finally: conn.close()
    def stm_count(self)->int:
        with _DB_LOCK:
            conn=self._conn()
            try: return int((conn.execute("SELECT COUNT(1) c FROM stm").fetchone() or {"c":0})["c"])
            finally: conn.close()

    def add_episode(self, user: str, assistant: str)->None:
        with _DB_LOCK:
            conn=self._conn()
            try:
                conn.execute("INSERT INTO episodes(ts,user,assistant) VALUES(strftime('%s','now'),?,?)",(user or "",assistant or ""))
                conn.commit(); self._meta_event("episode_add", {})
            finally: conn.close()
        self.stm_add(user, assistant)

    def episodes_tail(self, n: int=200)->List[Dict[str,Any]]:
        with _DB_LOCK:
            conn=self._conn()
            try:
                rows=conn.execute("SELECT ts,user,assistant FROM episodes ORDER BY ts DESC LIMIT ?", (int(n),)).fetchall()
                return [{"ts":float(r["ts"] or 0),"u":r["user"],"a":r["assistant"]} for r in rows][::-1]
            finally: conn.close()

    def rotate_episodes(self, keep_tail: int=5000)->Dict[str,Any]:
        with _DB_LOCK:
            conn=self._conn()
            try:
                n=int((conn.execute("SELECT COUNT(1) c FROM episodes").fetchone() or {"c":0})["c"])
                if n<=keep_tail: return {"ok":True,"rotated":0,"kept":n}
                conn.execute("DELETE FROM episodes WHERE ts NOT IN (SELECT ts FROM episodes ORDER BY ts DESC LIMIT ?)", (keep_tail,))
                conn.commit(); self._meta_event("episodes_rotate", {"left":keep_tail})
                return {"ok":True,"rotated":n-keep_tail,"kept":keep_tail}
            finally: conn.close()

    def purge_old_episodes(self, older_than_days: int=90)->Dict[str,Any]:
        cutoff=time.time()-older_than_days*86400
        with _DB_LOCK:
            conn=self._conn()
            try: conn.execute("DELETE FROM episodes WHERE ts < ?", (cutoff,)); conn.commit(); self._meta_event("episodes_purge", {"cutoff":cutoff}); return {"ok":True}
            finally: conn.close()

    # ---------------------- Emotional memory zapis ----------------------
    def _save_emotion(self, source: str, ref_id: str, text: str)->Dict[str,Any]:
        emo, pol, inten, dist = analyze_emotion(text)
        with _DB_LOCK:
            conn=self._conn()
            try:
                conn.execute("INSERT INTO emotions(ts,source,ref_id,emotion,polarity,intensity,meta) VALUES(strftime('%s','now'),?,?,?,?,?,?)",
                             (source, ref_id, emo, float(pol), float(inten), json.dumps({"dist":dist}, ensure_ascii=False)))
                conn.commit()
            finally: conn.close()
        return {"emotion": emo, "polarity": pol, "intensity": inten, "dist": dist}

    # ---------------------- Force flush STM + self reflection ----------------------
    def force_flush_stm(self)->Dict[str,Any]:
        tail=self.stm_tail(STM_MAX_TURNS)
        if not tail: return {"ok":True,"facts":0}
        convo=[]
        for t in tail:
            if t["u"]: convo.append({"role":"user","content":t["u"]})
            if t["a"]: convo.append({"role":"assistant","content":t["a"]})
        # zapisz emocje ostatnich wejść
        last = tail[-min(5,len(tail)):]
        for t in last:
            if t["u"]: self._save_emotion("stm","u",t["u"])
            if t["a"]: self._save_emotion("stm","a",t["a"])
        facts=_extract_facts(convo, max_out=80)
        ins=mrg=0
        for text,conf,tags in facts:
            r_before=self.exists(_id_for(text))
            self.add_fact(text, conf=conf, tags=tags)
            mrg += 1 if r_before else 0
            ins += 0 if r_before else 1
        keep=self.stm_tail(STM_KEEP_TAIL)
        with _DB_LOCK:
            conn=self._conn()
            try:
                conn.execute("DELETE FROM stm")
                for k in keep: conn.execute("INSERT INTO stm(ts,user,assistant) VALUES(?,?,?)",(k["ts"],k["u"],k["a"]))
                conn.commit()
            finally: conn.close()
        self._meta_event("stm_flush", {"inserted":ins,"merged":mrg})
        # Samorefleksja po flushu
        self._auto_reflect(convo)
        return {"ok":True,"facts":ins+mrg,"inserted":ins,"merged":mrg}

    # ---------------------- Profile/Goals ----------------------
    def get_profile(self)->Dict[str,Any]:
        with _DB_LOCK:
            conn=self._conn()
            try:
                rows=conn.execute("SELECT key,value FROM profile").fetchall()
                out={}
                for r in rows:
                    try: out[r["key"]]=json.loads(r["value"])
                    except Exception: out[r["key"]]=r["value"]
                return out
            finally: conn.close()
    def set_profile_many(self, updates: Dict[str,Any])->None:
        if not updates: return
        with _DB_LOCK:
            conn=self._conn()
            try:
                for k,v in updates.items():
                    conn.execute("INSERT INTO profile(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                                 (k, json.dumps(v, ensure_ascii=False)))
                conn.commit(); self._meta_event("profile_set_many", {"n":len(updates)})
            finally: conn.close()

    def get_goals(self)->List[Dict[str,Any]]:
        with _DB_LOCK:
            conn=self._conn()
            try:
                rows=conn.execute("SELECT id,title,priority,tags,ts FROM goals ORDER BY priority DESC, ts DESC").fetchall()
                out=[]
                for r in rows:
                    out.append({"id":r["id"],"title":r["title"],"priority":float(r["priority"] or 0),"tags":json.loads(r["tags"] or "[]"),"ts":float(r["ts"] or 0)})
                return out
            finally: conn.close()

    def add_goal(self, title: str, priority: float=1.0, tags: Optional[List[str]]=None)->Dict[str,Any]:
        t=_clamp_text(title or "")
        if not t: return {"ok":False,"reason":"empty"}
        gid=_id_for(t)[:16]
        with _DB_LOCK:
            conn=self._conn()
            try:
                row=conn.execute("SELECT id,priority,tags FROM goals WHERE id=?", (gid,)).fetchone()
                if row:
                    newp=max(float(row["priority"] or 0), float(priority))
                    try: cur=json.loads(row["tags"] or "[]")
                    except Exception: cur=[]
                    st=set(cur); st.update(tags or [])
                    conn.execute("UPDATE goals SET priority=?, tags=?, ts=strftime('%s','now') WHERE id=?",
                                 (newp, json.dumps(sorted(st), ensure_ascii=False), gid))
                    msg={"ok":True,"updated":True,"id":gid}
                else:
                    conn.execute("INSERT INTO goals(id,title,priority,tags,ts) VALUES(?,?,?,?,strftime('%s','now'))",
                                 (gid,t,float(priority), json.dumps(sorted(set(tags or [])), ensure_ascii=False)))
                    msg={"ok":True,"inserted":True,"id":gid}
                conn.commit(); self._meta_event("goal_upsert", {"id":gid}); return msg
            finally: conn.close()

    def update_goal(self, gid: str, **fields: Any)->Dict[str,Any]:
        allow=("title","priority","tags"); sets=[]; vals=[]
        for k,v in fields.items():
            if k in allow:
                sets.append(f"{k}=?"); vals.append(json.dumps(v, ensure_ascii=False) if k=="tags" else v)
        if not sets: return {"ok":False,"reason":"no_fields"}
        vals.append(gid)
        with _DB_LOCK:
            conn=self._conn()
            try:
                conn.execute(f"UPDATE goals SET {', '.join(sets)}, ts=strftime('%s','now') WHERE id=?", tuple(vals))
                conn.commit(); self._meta_event("goal_update", {"id":gid}); return {"ok":True}
            finally: conn.close()
    def delete_goal(self, gid: str)->Dict[str,Any]:
        with _DB_LOCK:
            conn=self._conn()
            try: conn.execute("DELETE FROM goals WHERE id=?", (gid,)); conn.commit(); self._meta_event("goal_delete", {"id":gid}); return {"ok":True}
            finally: conn.close()

    # ---------------------- Meta/Stats/Import/Export ----------------------
    def _meta_event(self, kind: str, payload: Dict[str,Any])->None:
        with _DB_LOCK:
            conn=self._conn()
            try:
                conn.execute("INSERT INTO meta_events(ts,kind,payload) VALUES(strftime('%s','now'),?,?)",(kind, json.dumps(payload, ensure_ascii=False)))
                conn.commit()
            finally: conn.close()

    def get_meta_events(self, limit: int=200)->List[Dict[str,Any]]:
        with _DB_LOCK:
            conn=self._conn()
            try:
                rows=conn.execute("SELECT ts,kind,payload FROM meta_events ORDER BY ts DESC LIMIT ?", (int(limit),)).fetchall()
                out=[]; 
                for r in rows:
                    try: payload=json.loads(r["payload"] or "{}")
                    except Exception: payload={"_raw":r["payload"]}
                    out.append({"ts":float(r["ts"] or 0),"kind":r["kind"],"payload":payload})
                return out
            finally: conn.close()

    def stats(self)->Dict[str,Any]:
        with _DB_LOCK:
            conn=self._conn()
            try:
                f=conn.execute("SELECT COUNT(1) c FROM ltm").fetchone()["c"]
                s=conn.execute("SELECT COUNT(1) c FROM stm").fetchone()["c"]
                e=conn.execute("SELECT COUNT(1) c FROM episodes").fetchone()["c"]
                p=conn.execute("SELECT COUNT(1) c FROM profile").fetchone()["c"]
                g=conn.execute("SELECT COUNT(1) c FROM goals").fetchone()["c"]
                fl=conn.execute("SELECT COUNT(1) c FROM files").fetchone()["c"]
                em=conn.execute("SELECT COUNT(1) c FROM emotions").fetchone()["c"]
                return {"facts":int(f),"stm":int(s),"episodes":int(e),"profile_keys":int(p),"goals":int(g),
                        "files":int(fl),"emotions":int(em),"namespace":self.ns,"db":str(DB_PATH),"size_mb":round(_db_size_mb(),2)}
            finally: conn.close()

    def export_json(self, out_path: str)->Dict[str,Any]:
        from pathlib import Path as _P
        pkg={"ns":self.ns,"ts":time.time(),"facts":self.list_facts(limit=MAX_LTM_FACTS),
             "episodes":self.episodes_tail(n=100000),"profile":self.get_profile(),"goals":self.get_goals(),
             "stm":self.stm_tail(n=1000)}
        _P(out_path).parent.mkdir(parents=True, exist_ok=True)
        _P(out_path).write_text(json.dumps(pkg, ensure_ascii=False, indent=2), encoding="utf-8")
        self._meta_event("export_json", {"path":out_path}); return {"ok":True,"path":out_path}

    def import_json(self, in_path: str, merge: bool=True)->Dict[str,Any]:
        from pathlib import Path as _P
        try: pkg=json.loads(_P(in_path).read_text(encoding="utf-8"))
        except Exception as e: return {"ok":False,"reason":str(e)}
        if merge:
            for f in pkg.get("facts",[]): self.add_fact(f.get("text",""), conf=float(f.get("conf",0.6)), tags=f.get("tags",[]))
            prof=self.get_profile(); prof.update(pkg.get("profile") or {}); self.set_profile_many(prof)
            for g in pkg.get("goals",[]): self.add_goal(g.get("title",""), priority=float(g.get("priority",1.0)), tags=g.get("tags",[]))
            for ep in pkg.get("episodes",[]): self.add_episode(ep.get("u",""), ep.get("a",""))
            for st in pkg.get("stm",[]): self.stm_add(st.get("u",""), st.get("a",""))
        else:
            with _DB_LOCK:
                conn=self._conn()
                try:
                    conn.executescript("DELETE FROM ltm; DELETE FROM ltm_fts; DELETE FROM stm; DELETE FROM episodes; DELETE FROM profile; DELETE FROM goals;")
                    conn.commit()
                finally: conn.close()
            for f in pkg.get("facts",[]): self.add_fact(f.get("text",""), conf=float(f.get("conf",0.6)), tags=f.get("tags",[]))
        _prune_lowscore_facts(); _vacuum_if_needed(); self._meta_event("import_json", {"path":in_path,"merge":merge})
        return {"ok":True}

    # ---------------------- Recall / Context ----------------------
    def _collect_docs_for_recall(self, limit_per_src: int=RECALL_TOPK_PER_SRC)->Tuple[List[str],List[str],List[float]]:
        docs=[]; srcs=[]; tss=[]
        prof=self.get_profile()
        if prof:
            docs.append("; ".join(f"{k}: {v}" for k,v in prof.items())); srcs.append("profile"); tss.append(time.time())
        for g in self.get_goals()[:limit_per_src]:
            docs.append(f"[goal pri={g['priority']}] {g['title']}"); srcs.append("goal"); tss.append(g.get("ts") or time.time())
        facts=[f for f in self.list_facts(limit=limit_per_src*20) if f["score"]>=LTM_MIN_SCORE]
        facts=sorted(facts, key=lambda x:(float(x.get("conf",0.0)), float(x.get("ts",0.0))), reverse=True)[:limit_per_src*3]
        for f in facts:
            docs.append(f["text"]); srcs.append("fact"); tss.append(f.get("ts") or 0.0)
        for e in self.episodes_tail(limit_per_src*12):
            docs.append(f"U: {e['u']}\nA: {e['a']}"); srcs.append("episode"); tss.append(e.get("ts") or 0.0)
        return docs,srcs,tss

    def recall(self, query: str, topk: int=6)->List[Tuple[str,float,str]]:
        docs,srcs,tss=self._collect_docs_for_recall(RECALL_TOPK_PER_SRC)
        if not docs: return []
        se=_emb_scores(query, docs); st=_tfidf_scores(query, docs); base=_blend_scores(se,st,0.75,0.45)
        bm=_fts_bm25(query, limit=max(40, topk*5))
        pool=[]; seen=set(); qwords=set(_tok(query))
        for i,d in enumerate(docs):
            if not d or d in seen: continue
            seen.add(d); score_base=base[i] if i<len(base) else 0.0
            doc_words=set(_tok(d)); sim_bonus=0.0
            if qwords and doc_words:
                matches=[]
                for qw in qwords:
                    best=0.0
                    for dw in doc_words:
                        sim=_string_similarity(qw,dw)
                        if sim>0.8: best=max(best,sim)
                    if best>0: matches.append(best)
                if matches: sim_bonus = sum(matches)/len(qwords)*0.2
            score_final=(score_base + sim_bonus)*_src_bonus(srcs[i])*_freshness_bonus(tss[i])
            pool.append((d,score_final,srcs[i]))
        for text,sbm,ts in bm:
            if not text or text in seen: continue
            seen.add(text); sc=sbm*_src_bonus("fts")*_freshness_bonus(ts); pool.append((text,sc,"fts"))
        pool.sort(key=lambda x:x[1], reverse=True)
        return pool[:topk]

    def compose_context(self, query: str, limit_chars: int=3500, topk: int=12)->str:
        rec=self.recall("profil użytkownika cechy osobowości preferencje", topk=topk//2) if not query.strip() else self.recall(query, topk=min(topk*2,30))
        if not rec: return ""
        grouped: Dict[str,List[Tuple[str,float]]]={}
        for txt,sc,src in rec: grouped.setdefault(src,[]).append((txt,sc))
        src_priority={"profile":1,"goal":2,"fact":3,"episode":4,"fts":5}
        parts=["[memory]"]; used=0
        for src in sorted(grouped.keys(), key=lambda s:src_priority.get(s,999)):
            items=sorted(grouped[src], key=lambda x:x[1], reverse=True)
            hdr=f"\n[SEKCJA: {src.upper()}]\n"
            if used+len(hdr)<=limit_chars: parts.append(hdr); used+=len(hdr)
            for txt,sc in items:
                chunk=f"• {txt.strip()} [score={sc:.2f}]\n"
                if used+len(chunk)>limit_chars: break
                parts.append(chunk); used+=len(chunk)
            if src!="fts":
                sep="\n---\n"; 
                if used+len(sep)<=limit_chars: parts.append(sep); used+=len(sep)
        parts.append("[/memory]"); ctx="".join(parts); log("[CTX]", used, "/", limit_chars)
        return ctx

    # ---------------------- Flowing context keywords ----------------------
    def _extract_new_keyword(self, conversation_slice: List[Dict[str,Any]])->Optional[str]:
        if not conversation_slice: return None
        try:
            transcript="\n".join([f"{'U' if 'u' in t else 'A'}: {t.get('u') or t.get('a')}" for t in conversation_slice])
            system_prompt=("Przeanalizuj NAJNOWSZY fragment rozmowy. Zwróć JEDEN nowy temat/keyword (2-3 słowa max).")
            if not _FEATURES.get("LLM"): return None
            resp=(llm_chat(user_text=transcript, system_text=system_prompt, max_tokens=20) or "").strip().replace('"','').replace('.','')
            return resp or None
        except Exception as e:
            log("[KKEY] error:", e); return None

    def get_flowing_context_prompt(self, update_interval: int=3, history_slice_size: int=4)->str:
        if self.user_message_count>0 and self.user_message_count%update_interval==0:
            slice=self.stm_tail(history_slice_size)
            newk=self._extract_new_keyword(slice)
            if newk and newk.lower() not in [k.lower() for k in self.context_keywords]:
                self.context_keywords.append(newk)
        if not self.context_keywords: return ""
        return f"Płynący kontekst rozmowy (ostatnie tematy): {', '.join(self.context_keywords)}. Weź go pod uwagę, odpowiadając."

    # ---------------------- Reflections (auto + on demand) ----------------------
    def _reflect_heuristic(self, convo: List[Dict[str,str]])->Tuple[str,List[str]]:
        # proste podsumowanie i parę reguł na podstawie preferencji/goals
        texts=[m["content"] for m in convo if m.get("content")]
        joined=" ".join(texts)[-1000:]
        # znajdź preferencje i tech
        prefs=[t for t in texts if re.search(r"\b(lubię|wolę|preferuję|uwielbiam)\b", t, re.I)]
        techs=set()
        for t in texts:
            for m in _TECH_PAT.finditer(t): techs.add(m.group(0))
        note=f"Podsumowanie: {joined[:300]}..." if joined else "Brak kontekstu."
        rules=[]
        if prefs: rules.append("jeśli pojawia się rekomendacja -> bierz pod uwagę ostatnie preferencje użytkownika")
        if techs: rules.append("jeśli dyskusja jest techniczna -> priorytetowo podawaj przykłady kodu w: "+", ".join(sorted(techs)))
        return note, rules

    def _auto_reflect(self, convo: List[Dict[str,str]])->Dict[str,Any]:
        try:
            if _FEATURES.get("LLM"):
                prompt="Zrób krótką notatkę i 3 konkretne reguły działania na przyszłość w JSON: {note:str, rules:[str,str,str]}."
                resp=llm_chat(user_text="\n".join(m['content'] for m in convo if m.get('content')), system_text=prompt, max_tokens=300)
                jstart=resp.find("{"); jend=resp.rfind("}")
                if jstart!=-1 and jend!=-1 and jend>jstart:
                    js=json.loads(resp[jstart:jend+1])
                    note=js.get("note","")
                    rules=js.get("rules",[]) or []
                else:
                    note,rules=self._reflect_heuristic(convo)
            else:
                note,rules=self._reflect_heuristic(convo)
            with _DB_LOCK:
                conn=self._conn()
                try:
                    conn.execute("INSERT INTO reflections(ts,note,rules,meta) VALUES(strftime('%s','now'),?,?,?)",
                                 (note, json.dumps(rules, ensure_ascii=False), json.dumps({"auto":True}, ensure_ascii=False)))
                    for r in rules:
                        conn.execute("INSERT INTO learned_rules(ts,rule,meta) VALUES(strftime('%s','now'),?,?)", (json.dumps({"rule":r}, ensure_ascii=False), "{}"))
                    conn.commit()
                finally: conn.close()
            return {"ok":True,"rules":rules}
        except Exception as e:
            log("[REFLECT] error:", e); return {"ok":False,"reason":str(e)}

    def reflect_now(self)->Dict[str,Any]:
        tail=self.stm_tail(50)
        convo=[]
        for t in tail:
            if t["u"]: convo.append({"role":"user","content":t["u"]})
            if t["a"]: convo.append({"role":"assistant","content":t["a"]})
        return self._auto_reflect(convo)

    # ---------------------- Prediction (intents + Markov) ----------------------
    _INTENT_PATTERNS = [
        ("ask_info", re.compile(r"\b(kto|co|gdzie|kiedy|jak|dlaczego|po co)\b", re.I)),
        ("task_create", re.compile(r"\b(zrób|utwórz|stwórz|dodaj|napisz)\b", re.I)),
        ("preference", re.compile(r"\b(lubię|wolę|preferuję|uwielbiam|nienawidzę)\b", re.I)),
        ("share_link", _LINK_PAT),
        ("greet", re.compile(r"\b(cześć|hej|siema|witam)\b", re.I)),
        ("goodbye", re.compile(r"\b(pa|na razie|do zobaczenia|żegnaj)\b", re.I)),
        ("plan", re.compile(r"\b(plan|harmonogram|kroki|roadmap)\b", re.I)),
        ("code", re.compile(r"\b(def |class |SELECT |INSERT |function|const |let |var )", re.I)),
    ]
    def classify_intent(self, text: str)->str:
        for name,pat in self._INTENT_PATTERNS:
            if pat.search(text or ""): return name
        return "other"

    def _markov_inc(self, prev: str, nxt: str)->None:
        with _DB_LOCK:
            conn=self._conn()
            try:
                row=conn.execute("SELECT cnt FROM intents_markov WHERE prev=? AND next=?", (prev,nxt)).fetchone()
                if row:
                    conn.execute("UPDATE intents_markov SET cnt=cnt+1 WHERE prev=? AND next=?", (prev,nxt))
                else:
                    conn.execute("INSERT INTO intents_markov(prev,next,cnt) VALUES(?,?,1)", (prev,nxt))
                conn.commit()
            finally: conn.close()

    def predict_next_intent(self)->Dict[str,Any]:
        with _DB_LOCK:
            conn=self._conn()
            try:
                # weź ostatnie 2-3 wejścia usera ze STM
                rows=conn.execute("SELECT user FROM stm WHERE user!='' ORDER BY ts DESC LIMIT 3").fetchall()
                if not rows: return {"ok":True,"intent":"other","confidence":0.2}
                cur=self.classify_intent(rows[0]["user"])
                trans=conn.execute("SELECT next,cnt FROM intents_markov WHERE prev=? ORDER BY cnt DESC LIMIT 5", (cur,)).fetchall()
                if not trans:
                    return {"ok":True,"intent":"other","confidence":0.3}
                total=sum(int(r["cnt"]) for r in trans)
                best=trans[0]["next"]; conf=float(trans[0]["cnt"])/float(total or 1)
                return {"ok":True,"intent":best,"confidence":round(conf,3),"from":cur}
            finally: conn.close()

    def learn_from_turn(self, user_text: str, assistant_text: str="")->None:
        prev_intent = self.classify_intent(user_text)
        next_intent = self.classify_intent(assistant_text) if assistant_text else "other"
        self._markov_inc(prev_intent, next_intent)

    # ---------------------- Sensory API ----------------------
    def ingest_file(self, content: bytes, filename: str, meta: Optional[Dict[str,Any]]=None)->Dict[str,Any]:
        p=_save_uploaded_file(content, filename)
        sha=_sha1_file(p); size=p.stat().st_size
        mime=mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        desc=_describe_file_heuristic(filename, mime, size)
        if _FEATURES.get("LLM"):
            try:
                llm_note=llm_chat(user_text=f"Opisz krótko plik: {filename}, typ {mime}, rozmiar {size}. Podaj słowa kluczowe.", system_text="Polski, 1-2 zdania, keywords.", max_tokens=120)
                if isinstance(llm_note,str) and len(llm_note.strip())>5:
                    desc = (llm_note.strip())[:500]
            except Exception: pass
        emb=None
        if _FEATURES.get("EMBED"):
            vs=_embed_many([desc]); emb = vs[0] if vs and len(vs)==1 else None
        fid=_id_for(sha+filename)
        with _DB_LOCK:
            conn=self._conn()
            try:
                conn.execute("INSERT OR REPLACE INTO files(id,path,mime,size,sha1,desc,emb,meta,ts) VALUES(?,?,?,?,?,?,?,?,strftime('%s','now'))",
                             (fid, str(p), mime, int(size), sha, desc, json.dumps(emb) if emb else None, json.dumps(meta or {}, ensure_ascii=False)))
                conn.commit()
            finally: conn.close()
        # dodaj do LTM skrót pliku
        ltm_text=f"plik:{filename} ({mime}, {size}B) -> {desc}"
        self.add_fact(ltm_text, meta_data={"tags":["file","sensory","src:files","pii:link"]}, score=0.8)
        # emocja z opisu
        self._save_emotion("file_desc", fid, desc)
        return {"ok":True,"id":fid,"path":str(p),"mime":mime,"size":size,"sha1":sha,"desc":desc}

    # ---------------------- Rebuild embeddings ----------------------
    def rebuild_missing_embeddings(self, batch: int=64, throttle_sec: float=0.2)->Dict[str,Any]:
        if not _FEATURES.get("EMBED"): return {"ok":False,"reason":"embedding_disabled"}
        updated=0
        with _DB_LOCK:
            conn=self._conn()
            try:
                while True:
                    rows=conn.execute("SELECT id,text FROM ltm WHERE (emb IS NULL OR emb='') LIMIT ?", (int(batch),)).fetchall()
                    if not rows: break
                    texts=[CRYPTO.dec(r["text"]) for r in rows]; ids=[r["id"] for r in rows]
                    vecs=_embed_many(texts)
                    if vecs:
                        for i,v in enumerate(vecs):
                            if v:
                                try: conn.execute("UPDATE ltm SET emb=? WHERE id=?", (json.dumps(v), ids[i]))
                                except Exception: pass
                        conn.commit(); updated += len([v for v in vecs if v])
                    else:
                        time.sleep(5.0); break
                    time.sleep(throttle_sec + 0.05*random.random())
                self._meta_event("rebuild_embeddings", {"updated":updated}); return {"ok":True,"updated":updated}
            finally: conn.close()

# ------------------------- Singleton -------------------------
_MEM_SINGLETON: Optional["Memory"]=None
def get_memory(namespace: Optional[str]=None)->"Memory":
    global _MEM_SINGLETON
    if _MEM_SINGLETON is None or (namespace and _MEM_SINGLETON.ns!=namespace):
        _MEM_SINGLETON=Memory(namespace=namespace or MEM_NS)
    return _MEM_SINGLETON

# ------------------------- REST API (FastAPI) -------------------------
try:
    from fastapi import FastAPI, UploadFile, File, Form
    from fastapi.responses import JSONResponse, FileResponse
    from fastapi.middleware.cors import CORSMiddleware
    _HAS_FASTAPI=True
except Exception:
    _HAS_FASTAPI=False

app = FastAPI(title="Memory ULTRA+ API", version="1.0") if _HAS_FASTAPI else None
if app:
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    mem=get_memory()

    @app.get("/stats")
    def api_stats(): return mem.stats()

    @app.post("/facts/add")
    def api_add_fact(text: str = Form(...), conf: float = Form(0.6), tags: str = Form("")):
        tg=[t.strip() for t in (tags or "").split(",") if t.strip()]
        fid=mem.add_fact(text, conf=conf, tags=tg)
        return {"ok":True,"id":fid}

    @app.get("/facts/list")
    def api_list_facts(limit: int=50): return mem.list_facts(limit=limit)

    @app.delete("/facts/{id_or_text}")
    def api_del_fact(id_or_text: str):
        ok=mem.delete_fact(id_or_text); return {"ok":ok}

    @app.get("/recall")
    def api_recall(q: str, topk: int=6):
        return mem.recall(q, topk=topk)

    @app.get("/ctx")
    def api_ctx(q: str, topk: int=8, limit: int=2200):
        return JSONResponse(content=mem.compose_context(q, topk=topk, limit_chars=limit))

    @app.post("/stm/turn")
    def api_stm_turn(user: str = Form(""), assistant: str = Form("")):
        mem.stm_add(user, assistant)
        mem.learn_from_turn(user, assistant)
        # zapisz emocje dla usera
        if user: mem._save_emotion("stm","u",user)
        if assistant: mem._save_emotion("stm","a",assistant)
        return {"ok":True}

    @app.post("/stm/flush")
    def api_stm_flush(): return mem.force_flush_stm()

    @app.post("/reflect")
    def api_reflect(): return mem.reflect_now()

    @app.get("/predict")
    def api_predict(): return mem.predict_next_intent()

    @app.post("/files/upload")
    async def api_upload(file: UploadFile = File(...)):
        content = await file.read()
        return mem.ingest_file(content, file.filename, meta={"from":"api"})

    @app.get("/files/{file_id}")
    def api_get_file(file_id: str):
        with _DB_LOCK:
            conn=_connect()
            try:
                row=conn.execute("SELECT path,mime FROM files WHERE id=?", (file_id,)).fetchone()
                if not row: return JSONResponse({"ok":False,"reason":"not_found"}, status_code=404)
                return FileResponse(path=row["path"], media_type=row["mime"])
            finally: conn.close()

    @app.post("/analyze/emotion")
    def api_emotion(text: str = Form(...)):
        emo,pol,inten,dist = analyze_emotion(text); return {"emotion":emo,"polarity":pol,"intensity":inten,"dist":dist}

    @app.post("/analyze/ner")
    def api_ner(text: str = Form(...)): return {"entities": ner(text)}

    @app.post("/analyze/wsd")
    def api_wsd(word: str = Form(...), context: str = Form("")): return wsd(word, context)

    @app.post("/goals/add")
    def api_goal_add(title: str = Form(...), priority: float = Form(1.0), tags: str = Form("")):
        tg=[t.strip() for t in (tags or "").split(",") if t.strip()]
        return mem.add_goal(title, priority=priority, tags=tg)

    @app.get("/goals")
    def api_goals(): return mem.get_goals()

    @app.post("/profile/set")
    def api_profile_set(payload: str = Form(...)):
        try:
            data=json.loads(payload)
        except Exception:
            return {"ok":False,"reason":"invalid_json"}
        mem.set_profile_many(data); return {"ok":True}

# ------------------------- CLI -------------------------
if __name__=="__main__":
    import argparse, sys
    ap=argparse.ArgumentParser(description="memory.py — ULTRA+ memory (RAG, sensory, emotions, reflection, prediction, REST)")
    sub=ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    a=sub.add_parser("add"); a.add_argument("--text", required=True); a.add_argument("--conf", type=float, default=0.6); a.add_argument("--tags", default="")
    list_parser=sub.add_subparsers(dest="listcmd")
    l=sub.add_parser("list"); l.add_argument("--limit", type=int, default=20)
    d=sub.add_parser("del"); d.add_argument("--id", required=True)
    r=sub.add_parser("recall"); r.add_argument("--q", required=True); r.add_argument("--topk", type=int, default=6)
    c=sub.add_parser("ctx"); c.add_argument("--q", required=True); c.add_argument("--topk", type=int, default=8); c.add_argument("--limit", type=int, default=2200)
    e=sub.add_parser("export"); e.add_argument("--out", required=True)
    i=sub.add_parser("import"); i.add_argument("--in", dest="inp", required=True); i.add_argument("--merge", action="store_true")
    m=sub.add_parser("meta"); m.add_argument("--limit", type=int, default=50)
    sub.add_parser("vacuum")
    p=sub.add_parser("prune"); p.add_argument("--target_mb", type=float, default=4200.0); p.add_argument("--batch", type=int, default=2000)
    f=sub.add_parser("rebuild_fts"); f.add_argument("--limit", type=int, default=0)
    g=sub.add_parser("integrity")
    b=sub.add_parser("backup"); b.add_argument("--out", required=True)
    sub.add_parser("rebuild_embeddings").add_argument("--batch", type=int, default=64)
    sub.add_parser("api")  # uruchom FastAPI lokalnie (bez uvicorn importu)

    args=ap.parse_args()
    mem=get_memory()
    if args.cmd=="meta": print(json.dumps(mem.get_meta_events(limit=50), ensure_ascii=False, indent=2)); sys.exit(0)
    if args.cmd=="vacuum": print(json.dumps(_vacuum_if_needed(0), ensure_ascii=False, indent=2)); sys.exit(0)
    if args.cmd=="prune": print(json.dumps(_prune_lowscore_facts(target_mb=args.target_mb, batch=args.batch), ensure_ascii=False, indent=2)); sys.exit(0)
    if args.cmd=="rebuild_fts":
        lim=None if not args.limit or args.limit<=0 else int(args.limit)
        print(json.dumps(_rebuild_fts(limit=lim), ensure_ascii=False, indent=2)); sys.exit(0)
    if args.cmd=="integrity": print(json.dumps(_integrity_check(), ensure_ascii=False, indent=2)); sys.exit(0)
    if args.cmd=="backup": print(json.dumps(_backup_db(args.out), ensure_ascii=False, indent=2)); sys.exit(0)
    if args.cmd=="rebuild_embeddings": print(json.dumps(mem.rebuild_missing_embeddings(batch=args.batch), ensure_ascii=False, indent=2)); sys.exit(0)
    if args.cmd=="stats": print(json.dumps(mem.stats(), ensure_ascii=False, indent=2)); sys.exit(0)
    if args.cmd=="add":
        tags=[t.strip() for t in args.tags.split(",") if t.strip()]
        mem.add_fact(args.text, conf=args.conf, tags=tags)
        print(json.dumps({"ok":True,"id":_id_for(args.text)})); sys.exit(0)
    if args.cmd=="list":
        print(json.dumps(mem.list_facts(limit=20), ensure_ascii=False, indent=2)); sys.exit(0)
    if args.cmd=="del":
        mem.delete_fact(args.id); print(json.dumps({"ok":True})); sys.exit(0)
    if args.cmd=="recall":
        print(json.dumps(mem.recall(args.q, topk=args.topk), ensure_ascii=False, indent=2)); sys.exit(0)
    if args.cmd=="ctx":
        print(mem.compose_context(args.q, topk=args.topk, limit_chars=args.limit)); sys.exit(0)
    if args.cmd=="export":
        print(json.dumps(mem.export_json(args.out), ensure_ascii=False, indent=2)); sys.exit(0)
    if args.cmd=="import":
        print(json.dumps(mem.import_json(args.inp, merge=args.merge), ensure_ascii=False, indent=2)); sys.exit(0)
    if args.cmd=="api":
        if not _HAS_FASTAPI:
            print("Zainstaluj fastapi i uvicorn: pip install fastapi uvicorn"); sys.exit(1)
        import uvicorn
        uvicorn.run("memory:app", host="0.0.0.0", port=8000, reload=True)
