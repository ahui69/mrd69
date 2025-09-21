# -*- coding: utf-8 -*-
"""
memory.py — Silnik pamięci (SQLite LTM/STM) + NLU (online jeśli dostępne, offline fallback) + WSD + Predykcja + Samorefleksja
Wersja: 2025-09-21

Kluczowe cechy:
- NLU: jeśli wykryjemy dostępność API (llm_simple/chat lub inny endpoint) → używamy ONLINE. Gdy API brak/zwrot błędu → OFFLINE fallback (nigdy nie wyłączamy NLU).
- Brak Redisa (zero sieciowego cache).
- WSD (rozróżnianie znaczeń słów: zamek, mysz, pająk — łatwo rozszerzyć).
- Pamięć predykcyjna (prediction_patterns): uczy się wzorców i przewiduje następne kroki.
- Samorefleksja (self_reflections): notatki AI po długich rozmowach i na żądanie.
- Hybrydowy recall: TF-IDF + (opcjonalnie) embeddings + (opcjonalnie) FTS5 BM25. Bez embeddings/FTS5 nadal działa (TF-IDF).
- Bezpieczeństwo: SQL parametryzowane; XOR to maska, nie kryptografia.

ENV (opcjonalnie):
- EMBEDDINGS / LLM: LLM_EMBED_URL, LLM_EMBED_MODEL, OPENAI_API_KEY/OPENAI_KEY
- MEM_NS, STM_MAX_TURNS, LTM_MIN_SCORE, MEM_LOG_LEVEL
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
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import logging
from dotenv import load_dotenv

# Torch (tylko kosinus; brak -> fallback)
try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    _HAS_TORCH = False

# ------------------------- LOGGING -------------------------
logging.basicConfig(
    level=os.getenv("MEM_LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("memory.log")]
)
logger = logging.getLogger(__name__)

# ------------------------- ENV & ŚCIEŻKI -------------------------
load_dotenv()

ROOT = Path(os.getenv("MEM_ROOT", str(Path(__file__).parent))).resolve()
DATA_DIR = ROOT / "data"
MEM_NS = (os.getenv("MEM_NS", "default") or "default").strip() or "default"
NS_DIR = DATA_DIR / MEM_NS
NS_DIR.mkdir(parents=True, exist_ok=True)

def _lazy_import_config():
    try:
        import config  # type: ignore
        return config
    except Exception:
        return None

config = _lazy_import_config()
USE_RUNPOD = bool(getattr(config, "USE_RUNPOD", False)) if config else False
RUNPOD_PATH = Path(getattr(config, "RUNPOD_PERSIST_DIR", "/workspace")) if config else Path("/workspace")

if USE_RUNPOD:
    runpod_data_dir = RUNPOD_PATH / "data"
    runpod_data_dir.mkdir(parents=True, exist_ok=True)
    DB_PATH = runpod_data_dir / "ltm.db"
    print(f"[MEMORY] Używam RunPod LTM: {DB_PATH}")
else:
    DB_PATH = DATA_DIR / "memory.db"
    print(f"[MEMORY] Używam lokalnej bazy LTM: {DB_PATH}")

# Parametry
LTM_MIN_SCORE = float(os.getenv("LTM_MIN_SCORE", "0.25"))
RECALL_TOPK_PER_SRC = int(os.getenv("RECALL_TOPK_PER_SRC", "100"))
STM_MAX_TURNS = int(os.getenv("STM_MAX_TURNS", "400"))
STM_KEEP_TAIL = int(os.getenv("STM_KEEP_TAIL", "100"))
HTTP_TIMEOUT = int(os.getenv("LLM_HTTP_TIMEOUT_S", os.getenv("TIMEOUT_HTTP", "60")))

# Embeddings / LLM (opcjonalnie; NLU i tak działa — online jeśli API, inaczej offline)
EMBED_URL = (os.getenv("LLM_EMBED_URL") or "").rstrip("/")
EMBED_MODEL = (os.getenv("LLM_EMBED_MODEL", "text-embedding-3-large") or "").strip()
EMBED_KEY = (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or "").strip()

# Torch device
if _HAS_TORCH and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu") if _HAS_TORCH else None

# ------------------------- Maskowanie (PSY_ENCRYPT_KEY) -------------------------
class _Crypto:
    def __init__(self, key: Optional[str]):
        self.key = hashlib.sha256((key or "").encode("utf-8")).digest() if key else None
    def enc(self, text: str) -> str:
        if not self.key:
            return text
        b = text.encode("utf-8")
        out = bytes(v ^ self.key[i % len(self.key)] for i, v in enumerate(b))
        return "xor:" + base64.urlsafe_b64encode(out).decode("ascii")
    def dec(self, blob: Any) -> str:
        if not self.key or not isinstance(blob, str) or not blob.startswith("xor:"):
            return blob if isinstance(blob, str) else str(blob)
        try:
            raw = base64.urlsafe_b64decode(blob[4:].encode("ascii"))
            out = bytes(v ^ self.key[i % len(self.key)] for i, v in enumerate(raw))
            return out.decode("utf-8", "ignore")
        except Exception:
            return str(blob)

CRYPTO = _Crypto(os.getenv("PSY_ENCRYPT_KEY"))

# ------------------------- NLU: Online (jeśli jest) + offline fallback -------------------------
def _lazy_import_llm_simple():
    try:
        from llm_simple import chat as llm_chat  # type: ignore
        return llm_chat
    except Exception:
        return None

_LLM_CHAT = _lazy_import_llm_simple()

# WSD reguły
_WSD_RULES = {
    "zamek": {
        "castle": {"kontekst": {"średniowiecz", "warown", "fortec", "królew", "dziedziniec", "mur", "wieża", "muzeum", "zwiedzanie"}},
        "lock":   {"kontekst": {"drzwi", "klucz", "otworzyć", "zamknąć", "zaryglować", "wkładka", "zatrzasnął", "kłódka"}},
        "zipper": {"kontekst": {"suwak", "spodnie", "kurtka", "rozsuw", "zsunął", "błyskawiczny", "rozpiąć", "zapiąć"}},
    },
    "mysz": {
        "animal": {"kontekst": {"gryzoń", "klatka", "laboratorium", "ogonek", "karmienie"}},
        "device": {"kontekst": {"komputer", "klik", "dpi", "scroll", "usb", "bluetooth", "sensor"}},
    },
    "pająk": {
        "animal": {"kontekst": {"sieć", "odwłok", "ptasznik", "jad"}},
        "structure": {"kontekst": {"sufit", "przewody", "rozporowy", "pajączyna"}},
    },
}

def _tokenize(s: str) -> List[str]:
    return re.findall(r"[0-9a-ząćęłńóśźżA-ZĄĆĘŁŃÓŚŹŻ]+", (s or ""), re.UNICODE)

class OfflineNLU:
    _INTENT_PATTERNS = {
        "greeting": re.compile(r"\b(cześć|hej|siema|witam|dzie(n|ń) dobry)\b", re.I),
        "question": re.compile(r".*\?\s*$"),
        "command": re.compile(r"^\s*(zrób|dodaj|usuń|pokaż|wyszukaj|napisz|policz|przypomnij|rozpakuj)\b", re.I),
        "preference": re.compile(r"\b(lubi(ę|sz)|wol(ę|isz)|preferuj(ę|esz)|nienawidz(ę|isz)|nie\s+lubi(ę|sz))\b", re.I),
        "feedback": re.compile(r"\b(działa|nie działa|bug|błąd|popraw|napraw)\b", re.I),
    }
    _S_POS = set("super świetnie wspaniale kocham lubię uwielbiam spoko ekstra dobrze git stabilnie elegancko".split())
    _S_NEG = set("nienawidzę wkurza źle tragicznie fatalnie wolno problem błąd bug syf nie działa słabo".split())
    _RE_EMAIL = re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", re.I)
    _RE_PHONE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\d{3}[\s-]?){2}\d{3,4}\b")
    _RE_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")
    _RE_DATE = re.compile(r"\b(20\d{2}|19\d{2})[-/.](0?[1-9]|1[0-2])[-/.](0?[1-9]|[12]\d|3[01])\b")
    _RE_TIME = re.compile(r"\b([01]?\d|2[0-3])[:.][0-5]\d\b")

    def __init__(self):
        self.ctx = {'last_intent': None, 'intent_chain': [], 'emotional_arc': []}
        self._last_np = None

    def _wsd(self, tokens: List[str]) -> List[Tuple[str,str]]:
        out = []
        lt = [t.lower() for t in tokens]
        for i, tok in enumerate(lt):
            base = tok
            if base in _WSD_RULES:
                window = " ".join(tokens[max(0, i-4): i+5]).lower()
                best, hits = None, -1
                for sense, rule in _WSD_RULES[base].items():
                    c = sum(1 for kw in rule["kontekst"] if kw in window)
                    if c > hits:
                        hits, best = c, sense
                if hits > 0:
                    out.append((base, best))
        return out

    def analyze(self, text: str, role: str = "user") -> Dict[str, Any]:
        t = (text or "").strip()
        toks = _tokenize(t)
        ltoks = [x.lower() for x in toks]

        intent = "info"
        for name, rx in self._INTENT_PATTERNS.items():
            if rx.search(t):
                intent = name
                break

        pos = sum(1 for w in ltoks if w in self._S_POS)
        neg = sum(1 for w in ltoks if w in self._S_NEG)
        sentiment = 0.0 if (pos+neg)==0 else (pos-neg)/float(pos+neg)
        emotional = "positive" if sentiment > 0.2 else "negative" if sentiment < -0.2 else "neutral"

        entities: List[Tuple[str,str]] = []
        for m in self._RE_EMAIL.findall(t): entities.append((m,"email"))
        for m in self._RE_PHONE.findall(t): entities.append((m,"phone"))
        for m in self._RE_DATE.findall(t): entities.append(("".join(m),"date"))
        for m in self._RE_TIME.findall(t): entities.append((m,"time"))
        for m in self._RE_NUMBER.findall(t): entities.append((m,"number"))

        wsd_tags = self._wsd(toks)

        coref = None
        if re.search(r"\b(to|tamto|ten|ta|te|on|ona|ono|oni|one|jego|jej|ich)\b", t, re.I):
            coref = self._last_np
        nouns_like = [w for w in toks if not re.match(r"^\d", w)]
        if nouns_like:
            self._last_np = " ".join(nouns_like[-3:])

        self.ctx['last_intent'] = intent
        self.ctx['intent_chain'].append(intent)
        if len(self.ctx['intent_chain'])>20: self.ctx['intent_chain']=self.ctx['intent_chain'][-20:]
        self.ctx['emotional_arc'].append(emotional)
        if len(self.ctx['emotional_arc'])>50: self.ctx['emotional_arc']=self.ctx['emotional_arc'][-50:]

        return {'intent': intent, 'entities': entities, 'sentiment': sentiment,
                'emotional_state': emotional, 'wsd': wsd_tags, 'coref': coref,
                'context': self.ctx.copy()}

class OnlineNLU:
    """NLU online przez llm_simple.chat – zwraca czysty JSON; przy błędzie raise i caller zdecyduje o fallbacku."""
    def __init__(self):
        self.llm_chat = _LLM_CHAT
        if not self.llm_chat:
            raise RuntimeError("Online NLU unavailable: llm_simple.chat not importable")
        self.ctx = {'last_intent': None, 'intent_chain': [], 'emotional_arc': []}

    def analyze(self, text: str, role: str = "user") -> Dict[str, Any]:
        prompt = f"""
Analyze this text for NLU:
Text: {text}

Return only JSON with:
- intent: main intent (greeting, question, command, preference, info, feedback, unknown)
- entities: list of [entity_text, entity_type] where type in [person, location, date, time, number, email, phone]
- sentiment: float -1..1
- emotional_state: "positive"|"negative"|"neutral"
- disambiguation: list of objects {{ "token": "<lemma>", "sense": "<label>" }} for ambiguous Polish words (zamek, mysz, pająk etc.)
- coref_hint: short string for likely referent if pronouns like 'to, ten, ona' appear
"""
        raw = self.llm_chat(user_text=prompt, system_text="You are a precise NLU analyzer. Respond with pure JSON only.", max_tokens=700)
        data = json.loads(raw)
        # aktualizacja kontekstu
        intent = data.get('intent','info')
        emo = data.get('emotional_state','neutral')
        self.ctx['last_intent'] = intent
        self.ctx['intent_chain'].append(intent)
        if len(self.ctx['intent_chain'])>20: self.ctx['intent_chain']=self.ctx['intent_chain'][-20:]
        self.ctx['emotional_arc'].append(emo)
        if len(self.ctx['emotional_arc'])>50: self.ctx['emotional_arc']=self.ctx['emotional_arc'][-50:]
        # mapowanie do formatu kompatybilnego z offline
        wsd = []
        for obj in data.get('disambiguation', []):
            tok = (obj.get('token') or '').lower().strip()
            sn = (obj.get('sense') or '').lower().strip()
            if tok and sn:
                wsd.append((tok, sn))
        coref = data.get('coref_hint') or None
        out = {
            'intent': intent,
            'entities': data.get('entities', []),
            'sentiment': float(data.get('sentiment', 0.0)),
            'emotional_state': emo,
            'wsd': wsd,
            'coref': coref,
            'context': self.ctx.copy(),
        }
        return out

class NLUBridge:
    """Mostek: użyj ONLINE jeśli dostępny i działa; inaczej offline."""
    def __init__(self):
        self._offline = OfflineNLU()
        self._online = None
        try:
            if _LLM_CHAT:
                self._online = OnlineNLU()
        except Exception as e:
            logger.warning(f"Online NLU init failed: {e}; using offline fallback.")

    @property
    def has_online(self) -> bool:
        return self._online is not None

    def analyze(self, text: str, role: str="user") -> Dict[str, Any]:
        if self._online:
            try:
                return self._online.analyze(text, role)
            except Exception as e:
                logger.warning(f"Online NLU error: {e}; using offline fallback.")
        return self._offline.analyze(text, role)

NLU = NLUBridge()

# ------------------------- TF-IDF / Similarity -------------------------
def _norm(s: str) -> str:
    return "".join(ch.lower() for ch in (s or "").strip() if ch.isalnum() or ch.isspace())

def _id_for(text: str) -> str:
    return hashlib.sha1(_norm(text).encode("utf-8")).hexdigest()

def _tok(s: str) -> List[str]:
    s = (s or "").lower()
    s2 = re.sub(r"[^0-9a-ząćęłńóśźżA-ZĄĆĘŁŃÓŚŹŻ]+", " ", s)
    return [w for w in s2.split() if len(w) > 2][:256]

def _tfidf_vec(tokens: List[str], docs_tokens: List[List[str]]) -> Dict[str, float]:
    N = len(docs_tokens) if docs_tokens else 1
    vocab = set(t for d in docs_tokens for t in d)
    df = {t: sum(1 for d in docs_tokens if t in d) for t in vocab}
    tf: Dict[str, int] = defaultdict(int)
    for t in tokens:
        tf[t] += 1
    out: Dict[str, float] = {}
    for t in tf:
        tf_part = (tf[t] / max(1, len(tokens)))
        idf_part = (math.log((N + 1) / (df.get(t, 1) + 1))) ** 1.5
        len_bonus = (1 + 0.1 * min(len(t) - 3, 7)) if len(t) > 3 else 1.0
        out[t] = tf_part * idf_part * len_bonus
    return out

def _tfidf_cos(q: str, docs: List[str]) -> List[float]:
    tq = _tok(q); dts = [_tok(d) for d in docs]; vq = _tfidf_vec(tq, dts)
    out: List[float] = []
    for dt in dts:
        vd = _tfidf_vec(dt, dts)
        keys = set(vq.keys()) | set(vd.keys())
        num = sum(vq.get(t,0.0)*vd.get(t,0.0) for t in keys)
        den_a = (sum(x*x for x in vq.values()) ** 0.5) or 1e-9
        den_b = (sum(x*x for x in vd.values()) ** 0.5) or 1e-9
        out.append((num/(den_a*den_b))**0.9)
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
  score REAL DEFAULT 0.6,
  emb TEXT
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
USING fts5(id, text, tokenize='unicode61');
"""

_SQL_ADVANCED = """
CREATE TABLE IF NOT EXISTS timeline_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL DEFAULT (strftime('%s','now')),
    type TEXT NOT NULL,
    title TEXT,
    content TEXT NOT NULL,
    user_input TEXT,
    ai_response TEXT,
    mood TEXT,
    context TEXT,
    related_person_id INTEGER,
    related_file_id INTEGER
);

CREATE TABLE IF NOT EXISTS self_reflections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL DEFAULT (strftime('%s','now')),
    summary TEXT NOT NULL,
    lessons_learned TEXT,
    rules_to_remember TEXT
);

CREATE TABLE IF NOT EXISTS prediction_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger_pattern TEXT UNIQUE NOT NULL,
    predicted_action TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    usage_count INTEGER DEFAULT 0
);
"""

_HAS_FTS5 = True

def _apply_pragmas(conn: sqlite3.Connection) -> None:
    try:
        conn.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            PRAGMA temp_store=MEMORY;
            PRAGMA cache_size=-16000;
        """)
    except Exception:
        pass

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    _apply_pragmas(conn)
    return conn

_DB_LOCK = threading.RLock()

def _init_db():
    global _HAS_FTS5
    with _DB_LOCK:
        conn = _connect()
        try:
            conn.executescript(_SQL_BASE)
            conn.executescript(_SQL_ADVANCED)
            try:
                conn.executescript(_SQL_FTS)
                _HAS_FTS5 = True
            except Exception as e:
                logger.warning(f"FTS5 unavailable: {e}")
                _HAS_FTS5 = False
            conn.commit()
        finally:
            conn.close()

_init_db()

# ------------------------- Embeddings (opcjonalne) -------------------------
def _embed_many(texts: List[str]) -> Optional[List[List[float]]]:
    if not (EMBED_URL and EMBED_KEY and texts):
        return None
    try:
        import requests  # local
    except Exception:
        logger.warning("requests not available - embeddings disabled")
        return None
    try:
        resp = requests.post(
            EMBED_URL,
            headers={"Authorization": f"Bearer {EMBED_KEY}", "Content-Type": "application/json"},
            json={"model": EMBED_MODEL, "input": texts},
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code >= 400:
            logger.warning(f"embed_many HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        j = resp.json()
        vecs = [d.get("embedding") for d in j.get("data", [])]
        return vecs if len(vecs) == len(texts) else None
    except Exception as e:
        logger.warning(f"embed_many failed: {e}")
        return None

def _cos(a: List[float], b: List[float]) -> float:
    if not a or not b: return 0.0
    if _HAS_TORCH:
        va = torch.tensor(a, dtype=torch.float32, device=device)
        vb = torch.tensor(b, dtype=torch.float32, device=device)
        dot = float(torch.dot(va, vb).item())
        sa = float(torch.norm(va).item())
        sb = float(torch.norm(vb).item())
        return dot/(sa*sb) if sa>0 and sb>0 else 0.0
    num = sum(x*y for x,y in zip(a,b))
    sa = math.sqrt(sum(x*x for x in a))
    sb = math.sqrt(sum(y*y for y in b))
    return num/(sa*sb) if sa>0 and sb>0 else 0.0

def _get_stored_embeddings(conn: sqlite3.Connection, doc_ids: List[str]) -> Dict[str, List[float]]:
    if not doc_ids: return {}
    ph = ",".join("?" * len(doc_ids))
    rows = conn.execute(f"SELECT id, emb FROM ltm WHERE id IN ({ph})", doc_ids).fetchall()
    out: Dict[str, List[float]] = {}
    for r in rows:
        if r["emb"]:
            try:
                out[r["id"]] = json.loads(r["emb"])
            except Exception:
                pass
    return out

# ------------------------- Ekstrakcja faktów (NLU + regex + WSD) -------------------------
def _sentences(text: str) -> List[str]:
    if not text: return []
    raw = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in raw if len(s.strip()) >= 5]

_PREF_PAT = re.compile(r"\b(lubię|wolę|preferuję|kocham|nienawidzę|nie\s+lubię|nie\s+cierpię|uwielbiam|podoba mi się)\b", re.I)
_PROFILE_PATS = {
    "age": re.compile(r"\b(mam|skończył\w*|\w+ lat)\s*(\d{1,2})\s*(lat|lata|rok|roku|wiosen)?\b", re.I),
    "email": re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", re.I),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\d{3}[\s-]?){2}\d{3,4}\b"),
    "city": re.compile(r"\b(mieszkam\s+w|jestem\s+z|pochodzę\s+z)\s+([A-ZŁŚŻŹĆĘÓĄ][\w\-ąćęłńóśźż]+)\b", re.I),
    "job": re.compile(r"\b(pracuję\s+jako|zawodowo\s+jestem|jestem\s+z\s+zawodu)\s+([a-ząćęłńóśźż\- ]{3,40})\b", re.I),
}
_LANG_PAT = re.compile(r"\b(mówię|znam|używam|uczę się)\s+(po\s+)?(polsku|angielsku|niemiecku|hiszpańsku|francusku|ukraińsku|rosyjsku|włosku|japońsku|chińsku|koreańsku|portugalsku)\b", re.I)
_TECH_PAT = re.compile(r"\b(R|Python|SQL|Java|JavaScript|TypeScript|C\+\+|C#|Go|Rust|PHP|Ruby|HTML|CSS|Docker|Kubernetes|TensorFlow|PyTorch|React|Angular|Vue|Node\.?js|Django|Flask|Laravel|Spring|Express\.?js|GraphQL|REST|API)\b", re.I)
_HOURS_PAT = re.compile(r"\b(pracuję|dostępny)\s+(od|w)\s+(\d{1,2})(?:[:.]\d{2})?\s*(do|-)\s*(\d{1,2})(?:[:.]\d{2})?\b", re.I)
_LINK_PAT = re.compile(r"\bhttps?://\S+\b", re.I)
_HEALTH_PAT = re.compile(r"\b(alergi[ae]|uczulenie|nietolerancj[ae])\b", re.I)
_NEGATION_PAT = re.compile(r"\b(nie|nie\s+bardzo|żadn[eyoa])\b", re.I)

def _extract_facts_from_turn(u: str, a: str) -> List[Tuple[str, float, List[str]]]:
    facts: List[Tuple[str, float, List[str]]] = []
    def _mk(text: str, base: float, tags: List[str]) -> Tuple[str, float, List[str]]:
        t = (text or "").strip()
        if not t: return ("",0.0,tags)
        neg = bool(_NEGATION_PAT.search(t))
        score = max(0.5, min(0.95, base + (-0.08 if neg else 0.04)))
        return (t, score, sorted(set(tags)))

    # NLU — preferuj online; fallback offline przez NLUBridge
    nlu_user = NLU.analyze(u or "", 'user')
    nlu_assistant = NLU.analyze(a or "", 'assistant') if a else {'intent':'info','entities':[],'wsd':[],'sentiment':0,'emotional_state':'neutral'}

    if nlu_user.get('intent') and nlu_user['intent'] != 'info':
        facts.append(_mk(f"nlu_intent_user: {nlu_user['intent']}", 0.92, ["nlu","intent","user"]))
    for (ent_text, ent_type) in nlu_user.get('entities', []):
        facts.append(_mk(f"nlu_entity_{ent_type}: {ent_text}", 0.88, ["nlu","entity",ent_type]))
    if abs(float(nlu_user.get('sentiment', 0))) > 0.2:
        direction = "positive" if nlu_user['sentiment'] > 0 else "negative"
        facts.append(_mk(f"sentiment_user: {direction}", 0.85, ["nlu","sentiment"]))
    for (w, s) in nlu_user.get('wsd', []):
        facts.append(_mk(f"wsd:{w}={s}", 0.83, ["nlu","wsd"]))
    if nlu_user.get("coref"):
        facts.append(_mk(f"coref_refers_to: {nlu_user['coref']}", 0.76, ["nlu","coref"]))

    # regexy
    for role, txt in (("user", u or ""), ("assistant", a or "")):
        for s in _sentences(txt):
            if _PREF_PAT.search(s): facts.append(_mk(f"preferencja: {s}", 0.80 if role=="user" else 0.72, ["stm","preference"])); continue
            m = _PROFILE_PATS["age"].search(s)
            if m: facts.append(_mk(f"wiek: {m.group(2)}", 0.86 if role=="user" else 0.78, ["stm","profile"])); continue
            m = _PROFILE_PATS["email"].search(s)
            if m: facts.append(_mk(f"email: {m.group(0)}", 0.89 if role=="user" else 0.81, ["stm","profile","contact"])); continue
            m = _PROFILE_PATS["phone"].search(s)
            if m: facts.append(_mk(f"telefon: {m.group(0)}", 0.88 if role=="user" else 0.8, ["stm","profile","contact"])); continue
            m = _PROFILE_PATS["city"].search(s)
            if m: facts.append(_mk(f"miasto: {m.group(2)}", 0.87 if role=="user" else 0.79, ["stm","profile"])); continue
            m = _PROFILE_PATS["job"].search(s)
            if m: facts.append(_mk(f"zawód: {m.group(2)}", 0.85 if role=="user" else 0.77, ["stm","profile"])); continue
            for lang in _LANG_PAT.findall(s): facts.append(_mk(f"język: {lang[2].lower()}", 0.78 if role=="user" else 0.7, ["stm","profile","language"]))
            for tech in set(t.group(0) for t in _TECH_PAT.finditer(s)): facts.append(_mk(f"tech: {tech}", 0.77 if role=="user" else 0.69, ["stm","skill","tech"]))
            mh = _HOURS_PAT.search(s)
            if mh: facts.append(_mk(f"availability: {mh.group(3)}-{mh.group(5)}", 0.75 if role=="user" else 0.67, ["stm","availability"]))
            for url in _LINK_PAT.findall(s): facts.append(_mk(f"link: {url}", 0.8, ["stm","link"]))
            if _HEALTH_PAT.search(s): facts.append(_mk(f"zdrowie: {s}", 0.82 if role=="user" else 0.74, ["stm","health"]))

    return [f for f in facts if f[0]]

def _dedupe_facts(facts: List[Tuple[str, float, List[str]]]) -> List[Tuple[str, float, List[str]]]:
    by_id: Dict[str, Tuple[str, float, List[str]]] = {}
    for t, score, tags in facts:
        t2 = (t or "").strip()
        if not t2: continue
        fid = _id_for(t2)
        if fid in by_id:
            old_t, old_score, old_tags = by_id[fid]
            by_id[fid] = (old_t, max(old_score, score), sorted(set(old_tags + tags)))
        else:
            by_id[fid] = (t2, score, sorted(set(tags)))
    return list(by_id.values())

def _extract_facts(messages: List[Dict[str, str]], max_out: int = 120) -> List[Tuple[str, float, List[str]]]:
    if not messages: return []
    all_facts: List[Tuple[str, float, List[str]]] = []
    i = 0
    full_context = "\n\n".join(f"{m.get('role','').upper()}: {m.get('content','')}" for m in messages)
    while i < len(messages):
        role_i = messages[i].get("role")
        u = messages[i].get("content", "") if role_i == "user" else ""
        a = messages[i + 1].get("content", "") if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant" else ""
        i += 2 if a else 1
        extracted = _extract_facts_from_turn(u, a)
        all_facts.extend(extracted)
        if u and re.search(r"\b(to|tam|ten|ta|te|on|ona|ono|oni|one|jego|jej|ich)\b", u, re.I):
            ctxt = full_context[-1500:]
            all_facts.append((f"Kontekst rozmowy: {ctxt}", 0.74, ["stm","context_reference"]))
    all_facts = _dedupe_facts(all_facts)
    all_facts.sort(key=lambda x: x[1], reverse=True)
    return all_facts[:max_out]

# ------------------------- DB care -------------------------
def _db_size_mb() -> float:
    try:
        base = DB_PATH.stat().st_size if DB_PATH.exists() else 0
        for suf in ("-wal","-shm"):
            side = DB_PATH.with_name(DB_PATH.name + suf)
            base += side.stat().st_size if side.exists() else 0
        return base/(1024*1024)
    except Exception:
        return 0.0

def _vacuum_if_needed(threshold_mb: float = 4500.0) -> Dict[str, Any]:
    size = _db_size_mb()
    if size < threshold_mb:
        return {"ok": True, "size_mb": size, "action": "none"}
    with _DB_LOCK:
        conn = _connect()
        try:
            conn.execute("VACUUM"); conn.commit()
        finally:
            conn.close()
    return {"ok": True, "size_mb": _db_size_mb(), "action": "vacuum"}

def _prune_lowscore_facts(target_mb: float = 4200.0, batch: int = 2000) -> Dict[str, Any]:
    size = _db_size_mb()
    if size < target_mb:
        return {"ok": True, "pruned": 0, "size_mb": size}
    removed = 0
    with _DB_LOCK:
        conn = _connect()
        try:
            rows = conn.execute("SELECT id FROM ltm WHERE score < ? ORDER BY ts ASC LIMIT ?", (max(0.15, LTM_MIN_SCORE - 0.15), batch)).fetchall()
            ids = [r["id"] for r in rows]
            if ids:
                q = ",".join("?" * len(ids))
                conn.execute(f"DELETE FROM ltm WHERE id IN ({q})", ids)
                if _HAS_FTS5:
                    conn.execute(f"DELETE FROM ltm_fts WHERE id IN ({q})", ids)
                removed = len(ids); conn.commit()
            return {"ok": True, "pruned": removed, "size_mb": size}
        finally:
            conn.close()

def _integrity_check() -> Dict[str, Any]:
    with _DB_LOCK:
        conn = _connect()
        try:
            ok = conn.execute("PRAGMA integrity_check").fetchone()[0]
            freelist = conn.execute("PRAGMA freelist_count").fetchone()[0]
            return {"ok": ok == "ok", "result": ok, "freelist": freelist}
        finally:
            conn.close()

def _backup_db(out_path: str) -> Dict[str, Any]:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _vacuum_if_needed(0)
    try:
        shutil.copy2(DB_PATH, out)
        for suf in ("-wal","-shm"):
            side = DB_PATH.with_name(DB_PATH.name + suf)
            if side.exists():
                shutil.copy2(side, out.with_name(out.name + suf))
        return {"ok": True, "path": str(out)}
    except Exception as e:
        return {"ok": False, "reason": str(e)}

# ------------------------- BM25 (FTS5) -------------------------
def _fts_bm25(query: str, limit: int = 50) -> List[Tuple[str, float, float]]:
    if not _HAS_FTS5 or not query: return []
    with _DB_LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT ltm_fts.id AS id, bm25(ltm_fts) AS bscore FROM ltm_fts WHERE ltm_fts MATCH ? ORDER BY bscore ASC LIMIT ?",
                (query, limit),
            ).fetchall()
            if not rows: return []
            ids = [r["id"] for r in rows]
            ph = ",".join("?" * len(ids))
            got = conn.execute(f"SELECT id, text, ts FROM ltm WHERE id IN ({ph})", ids).fetchall()
            meta = {r["id"]: (r["text"], float(r["ts"] or 0.0)) for r in got}
            out: List[Tuple[str, float, float]] = []
            for r in rows:
                t, ts = meta.get(r["id"], ("", 0.0))
                if not t: continue
                bscore = float(r["bscore"] or 0.0)
                score = 1.0/(1.0+bscore)
                out.append((t, score, ts))
            return out
        finally:
            conn.close()

def _rebuild_fts(limit: Optional[int] = None) -> Dict[str, Any]:
    if not _HAS_FTS5:
        return {"ok": False, "reason": "fts5_unavailable"}
    with _DB_LOCK:
        conn = _connect()
        try:
            conn.execute("DELETE FROM ltm_fts")
            q = "SELECT id, text FROM ltm ORDER BY ts DESC"
            if limit: q += f" LIMIT {limit}"
            rows = conn.execute(q).fetchall()
            data = [(r["id"], r["text"]) for r in rows if r["text"]]
            if data:
                conn.executemany("INSERT INTO ltm_fts(id, text) VALUES(?, ?)", data)
            conn.commit()
            return {"ok": True, "indexed": len(data)}
        finally:
            conn.close()

# ------------------------- Scoring hybrydowy -------------------------
def _emb_scores(query: str, docs: List[str]) -> List[float]:
    try:
        vecs = _embed_many([query] + docs)
        if not vecs: return []
        qv, dvs = vecs[0], vecs[1:]
        return [_cos(qv, d) if d else 0.0 for d in dvs]
    except Exception:
        return []

def _tfidf_scores(query: str, docs: List[str]) -> List[float]:
    try:
        return _tfidf_cos(query, docs)
    except Exception:
        return [0.0]*len(docs)

def _src_bonus(src: str) -> float:
    bonuses = {"profile": 1.35, "goal": 1.30, "fact": 1.25, "episode": 1.20, "fts": 1.22, "nlu": 1.40}
    return bonuses.get(src, 1.0)

def _freshness_bonus(ts: float) -> float:
    if not ts: return 1.0
    age_days = max(0.0, (time.time() - ts)/86400.0)
    recency_boost = 0.2*math.exp(-age_days) if age_days < 3.0 else 0.0
    base_freshness = max(0.75, 1.4 - (age_days/180.0))
    return base_freshness + recency_boost

def _blend_scores(a: List[float], b: List[float], wa: float, wb: float) -> List[float]:
    if not a and not b: return []
    if not a: a = [0.0]*len(b)
    if not b: b = [0.0]*len(a)
    n = min(len(a), len(b))
    out = []
    for i in range(n):
        aa = a[i] ** 1.1
        bb = b[i] ** 1.1
        harmony = 0.25 * (a[i]*b[i])**0.5 if a[i]>0.35 and b[i]>0.35 else 0.0
        out.append(wa*aa + wb*bb + harmony)
    return out

# ------------------------- Predykcja: wzorce sekwencji -------------------------
def _normalize_trigger(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    # podbijamy czasowniki i rzeczowniki kluczowe
    return s

def _default_predicted_action_for(text: str, intent: str) -> str:
    # heurystyki domyślne (możesz rozszerzać)
    if "rozpakuj" in text.lower() or "unzip" in text.lower():
        return "listuj_plik_i_uruchom_skrypt"
    if intent == "command" and ("zip" in text.lower() or "archiwum" in text.lower()):
        return "ekstrakcja_i_walidacja_struktury"
    if intent == "preference":
        return "zaproponuj_personalizowane_kroki"
    if intent == "feedback":
        return "diagnoza_i_poprawka"
    if intent == "question":
        return "odpowiedz_i_podaj_przyklad"
    return "kontynuuj_dialog"

def _update_prediction_pattern(conn: sqlite3.Connection, trigger: str, predicted: str, delta_conf: float = 0.02) -> None:
    row = conn.execute("SELECT id, confidence, usage_count, predicted_action FROM prediction_patterns WHERE trigger_pattern=?", (trigger,)).fetchone()
    if row:
        conf = float(row["confidence"] or 0.5)
        cnt = int(row["usage_count"] or 0) + 1
        # jeśli nowa predykcja różna od starej — powoli dryfuj w stronę nowej
        if predicted != row["predicted_action"]:
            conf = max(0.1, conf - 0.05)
            conn.execute("UPDATE prediction_patterns SET predicted_action=?, confidence=?, usage_count=? WHERE id=?",
                         (predicted, conf, cnt, row["id"]))
        else:
            conf = min(0.99, conf + delta_conf)
            conn.execute("UPDATE prediction_patterns SET confidence=?, usage_count=? WHERE id=?",
                         (conf, cnt, row["id"]))
    else:
        conn.execute("INSERT INTO prediction_patterns(trigger_pattern, predicted_action, confidence, usage_count) VALUES(?,?,?,?)",
                     (trigger, predicted, 0.55, 1))

def _predict_next_actions(conn: sqlite3.Connection, user_text: str, nlu_intent: str, topk: int = 5) -> List[Tuple[str, float]]:
    trig = _normalize_trigger(user_text)
    rows = conn.execute("SELECT predicted_action, confidence, usage_count FROM prediction_patterns WHERE trigger_pattern LIKE ? ORDER BY confidence DESC, usage_count DESC LIMIT ?",
                        (f"%{trig[:32]}%", topk)).fetchall()
    out = [(r["predicted_action"], float(r["confidence"] or 0.0)) for r in rows]
    if not out:
        out = [(_default_predicted_action_for(user_text, nlu_intent), 0.5)]
    return out[:topk]

# ------------------------- Klasa Memory -------------------------
class Memory:
    def __init__(self, namespace: str = MEM_NS):
        self.ns = namespace
        NS_DIR.mkdir(parents=True, exist_ok=True)
        _init_db()
        self._start_background_prune()

    def _start_background_prune(self):
        def run_prune():
            try:
                _prune_lowscore_facts()
            except Exception as e:
                logger.error(f"background prune failed: {e}")
            finally:
                t = threading.Timer(3600, run_prune)
                t.daemon = True
                t.start()
        t0 = threading.Timer(3600, run_prune); t0.daemon = True; t0.start()

    # --- API zapisu ---
    def add_entry(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        tags = ["entry"]
        if metadata:
            if "source" in metadata: tags.append(f"src:{metadata['source']}")
            if "filename" in metadata: tags.append("file")
        return self.add_fact(text, meta_data={"tags": tags}, score=0.75)

    def add_fact(self, text: str, meta_data: Optional[Dict]=None, score: float=0.6, emb: Optional[List[float]]=None, tags: Optional[List[str]]=None, conf: Optional[float]=None) -> str:
        txt = (text or "").strip()
        if not txt: raise ValueError("empty")
        if conf is not None and score == 0.6: score = float(conf)
        if tags:
            meta_data = meta_data or {}
            meta_data["tags"] = sorted(set((meta_data.get("tags") or []) + tags))
        meta_data = meta_data or {}
        if 'tags' in meta_data and not isinstance(meta_data['tags'], list): raise TypeError("tags must be list")
        meta_json = json.dumps(meta_data, ensure_ascii=False)
        fid = _id_for(txt)
        emb_json = None
        if emb:
            emb_json = json.dumps(emb)
        elif EMBED_URL and EMBED_KEY:
            vecs = _embed_many([txt])
            if vecs and vecs[0]:
                emb_json = json.dumps(vecs[0])
        with _DB_LOCK:
            conn = _connect()
            try:
                row = conn.execute("SELECT id, meta, score FROM ltm WHERE id=?", (fid,)).fetchone()
                if row:
                    cur_meta = json.loads(row["meta"] or "{}")
                    for k, v in meta_data.items():
                        if k not in cur_meta: cur_meta[k]=v
                        elif k == "tags" and isinstance(cur_meta.get(k), list): cur_meta[k]=sorted(set(cur_meta[k]+meta_data.get(k, [])))
                    meta_json = json.dumps(cur_meta, ensure_ascii=False)
                    new_score = max(float(row["score"] or 0), float(score))
                    conn.execute("UPDATE ltm SET meta=?, score=?, ts=strftime('%s','now'), emb=? WHERE id=?", (meta_json, new_score, emb_json, fid))
                    if _HAS_FTS5: conn.execute("INSERT OR REPLACE INTO ltm_fts(id, text) VALUES(?, ?)", (fid, txt))
                else:
                    conn.execute("INSERT INTO ltm(id, kind, text, meta, score, emb, ts) VALUES(?,?,?,?,?,?,strftime('%s','now'))", (fid, "fact", txt, meta_json, float(score), emb_json))
                    if _HAS_FTS5: conn.execute("INSERT INTO ltm_fts(id, text) VALUES(?, ?)", (fid, txt))
                conn.commit()
                return fid
            finally:
                conn.close()

    def add_fact_bulk(self, rows: List[Tuple[str, float, Optional[List[str]]]]) -> Dict[str,int]:
        ins = mrg = 0
        batch_data: List[Tuple[str,str,str,str,float]] = []
        with _DB_LOCK:
            conn = _connect()
            try:
                conn.execute("BEGIN")
                for t, c, tg in rows:
                    txt = (t or "").strip()
                    if not txt: continue
                    fid = _id_for(txt)
                    meta_data = {"tags": tg or []}
                    if not isinstance(meta_data['tags'], list): raise TypeError("tags must be list")
                    meta_json = json.dumps(meta_data, ensure_ascii=False)
                    row = conn.execute("SELECT 1 FROM ltm WHERE id=?", (fid,)).fetchone()
                    if row:
                        conn.execute("UPDATE ltm SET score=MAX(score, ?), ts=strftime('%s','now') WHERE id=?", (c, fid))
                        if _HAS_FTS5: conn.execute("INSERT OR REPLACE INTO ltm_fts(id, text) VALUES(?, ?)", (fid, txt))
                        mrg += 1
                    else:
                        batch_data.append((fid, "fact", txt, meta_json, c)); ins += 1
                if batch_data:
                    conn.executemany("INSERT INTO ltm(id, kind, text, meta, score, emb, ts) VALUES(?,?,?,?,?,?,strftime('%s','now'))", [(d[0],d[1],d[2],d[3],d[4],None) for d in batch_data])
                    if _HAS_FTS5: conn.executemany("INSERT INTO ltm_fts(id, text) VALUES(?, ?)", [(d[0], d[2]) for d in batch_data])
                conn.execute("COMMIT")
            except Exception as e:
                conn.execute("ROLLBACK"); logger.error(f"Batch insert failed: {e}"); raise
            finally:
                conn.close()
        return {"inserted": ins, "merged": mrg}

    def exists(self, fid: str) -> bool:
        with _DB_LOCK:
            conn = _connect()
            try:
                return bool(conn.execute("SELECT 1 FROM ltm WHERE id=?", (fid,)).fetchone())
            finally:
                conn.close()

    def delete_fact(self, id_or_text: str) -> bool:
        tid = id_or_text if re.fullmatch(r"[0-9a-f]{40}", id_or_text) else _id_for(id_or_text)
        with _DB_LOCK:
            conn = _connect()
            try:
                conn.execute("DELETE FROM ltm WHERE id=?", (tid,))
                if _HAS_FTS5: conn.execute("DELETE FROM ltm_fts WHERE id=?", (tid,))
                conn.commit(); return True
            finally:
                conn.close()

    # --- STM/Episodes ---
    def stm_add(self, user: str, assistant: str) -> None:
        with _DB_LOCK:
            conn = _connect()
            try:
                conn.execute("INSERT INTO stm(ts, user, assistant) VALUES(strftime('%s','now'), ?, ?)", (user or "", assistant or ""))
                conn.commit()
                # PREDYKCYJNA aktualizacja (na bazie user)
                nlu = NLU.analyze(user or "", "user")
                trig = _normalize_trigger(user or "")
                predicted = _default_predicted_action_for(user or "", nlu.get("intent","info"))
                _update_prediction_pattern(conn, trig, predicted)
                conn.commit()
            finally:
                conn.close()
        if self.stm_count() >= STM_MAX_TURNS:
            self.force_flush_stm()

    def stm_tail(self, n: int = 200) -> List[Dict[str, Any]]:
        with _DB_LOCK:
            conn = _connect()
            try:
                rows = conn.execute("SELECT ts, user, assistant FROM stm ORDER BY ts DESC LIMIT ?", (n,)).fetchall()
                return [{"ts": float(r["ts"] or 0), "u": r["user"], "a": r["assistant"]} for r in rows][::-1]
            finally:
                conn.close()

    def stm_count(self) -> int:
        with _DB_LOCK:
            conn = _connect()
            try:
                return int(conn.execute("SELECT COUNT(1) AS c FROM stm").fetchone()["c"])
            finally:
                conn.close()

    def force_flush_stm(self) -> Dict[str, Any]:
        tail = self.stm_tail(STM_MAX_TURNS)
        if not tail: return {"ok": True, "facts": 0}
        convo: List[Dict[str, str]] = []
        for t in tail:
            if t["u"]: convo.append({"role":"user","content":t["u"]})
            if t["a"]: convo.append({"role":"assistant","content":t["a"]})
        facts = _extract_facts(convo, max_out=80)
        ins = mrg = 0
        for text, conf, tags in facts:
            before = self.exists(_id_for(text))
            self.add_fact(text, score=conf, tags=tags)
            if before: mrg += 1
            else: ins += 1
        # samorefleksja po flush
        try:
            self.reflect_and_save(convo)
        except Exception as e:
            logger.warning(f"Reflection failed: {e}")
        # skracamy STM
        keep = self.stm_tail(STM_KEEP_TAIL)
        with _DB_LOCK:
            conn = _connect()
            try:
                conn.execute("DELETE FROM stm")
                for k in keep:
                    conn.execute("INSERT INTO stm(ts,user,assistant) VALUES(?,?,?)", (k["ts"], k["u"], k["a"]))
                conn.commit()
            finally:
                conn.close()
        return {"ok": True, "facts": ins + mrg, "inserted": ins, "merged": mrg}

    def process_stm_window(self, window_size: int = 30) -> Dict[str, Any]:
        recent_turns = self.stm_tail(window_size)
        if not recent_turns:
            return {"ok": True, "facts": 0, "reason": "no_recent_turns"}
        convo_str = "\n".join(f"U: {t['u']}\nA: {t['a']}" for t in recent_turns if t["u"] or t["a"])
        if not convo_str.strip():
            return {"ok": True, "facts": 0, "reason": "empty_convo"}
        facts = _extract_facts([{"role": "user", "content": convo_str}], max_out=80)
        ins = mrg = 0
        for text, score, tags in facts:
            before = self.exists(_id_for(text))
            self.add_fact(text, score=score, tags=tags)
            if before: mrg += 1
            else: ins += 1
        return {"ok": True, "facts": ins + mrg, "inserted": ins, "merged": mrg, "note": "NLU bridge used"}

    def add_episode(self, user: str, assistant: str) -> None:
        with _DB_LOCK:
            conn = _connect()
            try:
                conn.execute("INSERT INTO episodes(ts,user,assistant) VALUES(strftime('%s','now'),?,?)", (user or "", assistant or ""))
                conn.commit()
            finally:
                conn.close()
        self.stm_add(user, assistant)

    def episodes_tail(self, n: int = 200) -> List[Dict[str, Any]]:
        with _DB_LOCK:
            conn = _connect()
            try:
                rows = conn.execute("SELECT ts,user,assistant FROM episodes ORDER BY ts DESC LIMIT ?", (n,)).fetchall()
                return [{"ts": float(r["ts"] or 0), "u": r["user"], "a": r["assistant"]} for r in rows][::-1]
            finally:
                conn.close()

    def get_profile(self) -> Dict[str, str]:
        with _DB_LOCK:
            conn = _connect()
            try:
                rows = conn.execute("SELECT key, value FROM profile").fetchall()
                return {r["key"]: r["value"] for r in rows}
            finally:
                conn.close()

    def get_goals(self) -> List[Dict[str, Any]]:
        with _DB_LOCK:
            conn = _connect()
            try:
                rows = conn.execute("SELECT * FROM goals ORDER BY priority DESC").fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def list_facts(self, limit: int = 500) -> List[Dict[str, Any]]:
        with _DB_LOCK:
            conn = _connect()
            try:
                rows = conn.execute("SELECT id, text, meta, score, ts FROM ltm ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
                result = []
                for r in rows:
                    meta_data = json.loads(r["meta"] or "{}") if r["meta"] else {}
                    tags = meta_data.get("tags", [])
                    result.append({
                        "id": r["id"], "text": r["text"], "conf": float(r["score"] or 0),
                        "tags": tags, "ts": float(r["ts"] or 0), "score": float(r["score"] or 0), "meta": meta_data,
                    })
                return result
            finally:
                conn.close()

    # --- Embeddings maintenance ---
    def rebuild_missing_embeddings(self, batch: int = 64) -> Dict[str, Any]:
        if not (EMBED_URL and EMBED_KEY):
            return {"ok": False, "reason": "embedding_disabled"}
        updated = 0
        with _DB_LOCK:
            conn = _connect()
            try:
                while True:
                    rows = conn.execute("SELECT id, text FROM ltm WHERE emb IS NULL OR emb = '' LIMIT ?", (batch,)).fetchall()
                    if not rows: break
                    texts = [r["text"] for r in rows]; ids = [r["id"] for r in rows]
                    vecs = _embed_many(texts)
                    if not vecs: break
                    for i, v in enumerate(vecs):
                        if v:
                            conn.execute("UPDATE ltm SET emb=? WHERE id=?", (json.dumps(v), ids[i]))
                            updated += 1
                    conn.commit()
            finally:
                conn.close()
        return {"ok": True, "updated": updated}

    # --- Recall / kontekst ---
    def _collect_docs_for_recall(self, limit_per_src: int = RECALL_TOPK_PER_SRC) -> Tuple[List[str], List[str], List[float]]:
        docs: List[str] = []; srcs: List[str] = []; tss: List[float] = []
        prof = self.get_profile()
        if prof:
            docs.append("; ".join(f"{k}: {v}" for k,v in prof.items())); srcs.append("profile"); tss.append(time.time())
        for g in self.get_goals()[:limit_per_src]:
            docs.append(f"[goal pri={g['priority']}] {g['title']}"); srcs.append("goal"); tss.append(float(g.get("ts") or time.time()))
        facts = [f for f in self.list_facts(limit=limit_per_src*20) if float(f.get("score",0.0)) >= LTM_MIN_SCORE]
        facts = sorted(facts, key=lambda x: (x.get("score",0.0), x.get("ts",0.0)), reverse=True)[:limit_per_src*3]
        for f in facts:
            docs.append(f["text"]); srcs.append("fact"); tss.append(float(f.get("ts") or 0.0))
        for e in self.episodes_tail(limit=limit_per_src*12):
            docs.append(f"U: {e['u']}\nA: {e['a']}"); srcs.append("episode"); tss.append(float(e.get("ts") or 0.0))
        return docs, srcs, tss

    def recall(self, query: str, topk: int = 6) -> List[Tuple[str, float, str]]:
        docs, srcs, tss = self._collect_docs_for_recall(RECALL_TOPK_PER_SRC)
        if not docs: return []
        with _DB_LOCK:
            conn = _connect()
            stored = _get_stored_embeddings(conn, [_id_for(d) for d in docs])
            conn.close()
        # embeddings gdy są
        query_emb: Optional[List[float]] = None
        doc_embs: List[Optional[List[float]]] = []
        missing_docs: List[str] = []; missing_idx: List[int] = []
        for i, d in enumerate(docs):
            did = _id_for(d)
            if did in stored and stored[did]:
                doc_embs.append(stored[did])
            else:
                doc_embs.append(None); missing_docs.append(d); missing_idx.append(i)
        if missing_docs:
            v = _embed_many([query] + missing_docs)
            if v:
                query_emb = v[0]
                for j, idx in enumerate(missing_idx):
                    doc_embs[idx] = v[j+1]
        else:
            v = _embed_many([query])
            if v: query_emb = v[0]
        se = []
        if query_emb:
            for de in doc_embs: se.append(_cos(query_emb, de) if de else 0.0)
        else:
            se = [0.0]*len(docs)
        st = _tfidf_scores(query, docs)
        base = _blend_scores(se, st, wa=0.75, wb=0.45)
        bm = _fts_bm25(query, limit=max(40, topk*5))
        pool: List[Tuple[str,float,str]] = []; seen=set()
        for i, d in enumerate(docs):
            if not d or d in seen: continue
            seen.add(d)
            score = (base[i] if i<len(base) else 0.0) * _src_bonus(srcs[i]) * _freshness_bonus(tss[i])
            pool.append((d, score, srcs[i]))
        for text, s_bm25, ts in bm:
            if not text or text in seen: continue
            seen.add(text)
            pool.append((text, s_bm25 * _src_bonus("fts") * _freshness_bonus(ts), "fts"))
        pool.sort(key=lambda x: x[1], reverse=True)
        return pool[:topk]

    def compose_context(self, query: str, limit_chars: int = 3500, topk: int = 12) -> str:
        rec = self.recall(query or "profil użytkownika preferencje", topk=min(max(topk,6), 30))
        if not rec: return ""
        grouped: Dict[str, List[Tuple[str,float]]] = defaultdict(list)
        for txt, sc, src in rec: grouped[src].append((txt, sc))
        src_priority = {"profile":1,"goal":2,"nlu":3,"fact":4,"episode":5,"fts":6}
        sorted_sources = sorted(grouped.keys(), key=lambda s: src_priority.get(s, 999))
        parts = ["[memory]"]; used = 0
        for src in sorted_sources:
            items = sorted(grouped[src], key=lambda x: x[1], reverse=True)
            sh = f"\n[SEKCJA: {src.upper()}]\n"
            if used + len(sh) <= limit_chars: parts.append(sh); used += len(sh)
            for txt, sc in items:
                chunk = f"• {txt.strip()} [score={sc:.2f}]\n"
                if used + len(chunk) > limit_chars: break
                parts.append(chunk); used += len(chunk)
            if src != sorted_sources[-1]:
                sep = "\n---\n"
                if used + len(sep) <= limit_chars: parts.append(sep); used += len(sep)
        parts.append("[/memory]")
        return "".join(parts)

    # --- Predykcja API ---
    def predict_next_actions(self, user_text: str, topk: int = 5) -> List[Tuple[str, float]]:
        with _DB_LOCK:
            conn = _connect()
            try:
                nlu = NLU.analyze(user_text or "", "user")
                preds = _predict_next_actions(conn, user_text or "", nlu.get("intent","info"), topk=topk)
                return preds
            finally:
                conn.close()

    # --- Samorefleksja ---
    def reflect_and_save(self, convo: List[Dict[str,str]]) -> Dict[str, Any]:
        """Tworzy notatkę: summary, lessons, rules. Preferuje NLU online (jeśli jest), inaczej heurystyka offline."""
        text = "\n".join(f"{m['role'][:1].upper()}: {m['content']}" for m in convo if m.get("content"))
        summary = ""
        lessons = []
        rules = []
        # online próba
        used_online = False
        if NLU.has_online:
            try:
                llm = _LLM_CHAT
                sys = "You are a terse coaching assistant for the assistant. Return pure JSON with keys: summary, lessons (list), rules (list)."
                prompt = f"""Summarize this dialog from the AI's perspective (Polish). Be concrete and pragmatic.
TEXT:
{text}
Return JSON like:
{{"summary":"...","lessons":["..."],"rules":["..."]}}"""
                raw = llm(user_text=prompt, system_text=sys, max_tokens=600)
                j = json.loads(raw)
                summary = (j.get("summary") or "").strip()
                lessons = [x for x in j.get("lessons", []) if isinstance(x,str)]
                rules = [x for x in j.get("rules", []) if isinstance(x,str)]
                used_online = True
            except Exception as e:
                logger.warning(f"Reflection online failed: {e}")
        if not used_online:
            # heurystyka offline
            turns = [m.get("content","") for m in convo if m.get("content")]
            summary = "Krótka sesja; główne tematy: " + ", ".join(list({w for w in _tok(" ".join(turns)) if len(w)>4})[:8])
            if any("pełn" in t.lower() and "plik" in t.lower() for t in turns):
                lessons.append("Zawsze dostarczaj pełny plik, bez ucinania.")
                rules.append("Nie wysyłaj szkieletów – generuj kompletne pliki.")
            if any("zip" in t.lower() or "rozpakuj" in t.lower() for t in turns):
                lessons.append("Po rozpakowaniu archiwum zaproponuj kolejny krok.")
                rules.append("Po komendzie 'rozpakuj' zaproponuj 'listuj i uruchom'.")
        with _DB_LOCK:
            conn = _connect()
            try:
                conn.execute("INSERT INTO self_reflections(summary, lessons_learned, rules_to_remember) VALUES(?,?,?)",
                             (summary, json.dumps(lessons, ensure_ascii=False), json.dumps(rules, ensure_ascii=False)))
                conn.commit()
            finally:
                conn.close()
        return {"ok": True, "used_online": used_online, "summary": summary, "lessons": lessons, "rules": rules}

    # --- Zdrowie rozmowy ---
    def get_conversation_health(self) -> Dict[str, Any]:
        # bierzemy kontekst z offline (bridge zawiera oba)
        ctx = (NLU._online.ctx if NLU.has_online else NLU._offline.ctx)
        intent_diversity = (len(set(ctx['intent_chain'])) / len(ctx['intent_chain'])) if ctx['intent_chain'] else 0.0
        dom = Counter(ctx['emotional_arc']).most_common(1)
        mood = dom[0][0] if dom else 'neutral'
        health = 0.0
        if intent_diversity > 0.3: health += 0.4
        if mood == 'positive': health += 0.3
        elif mood == 'negative': health -= 0.2
        return {
            'health_score': max(0.0, min(1.0, health)),
            'intent_diversity': intent_diversity,
            'dominant_mood': mood,
            'recent_intents': ctx['intent_chain'][-5:],
            'emotional_trend': ctx['emotional_arc'][-5:],
            'recommendation': self._get_health_recommendation(health, mood)
        }

    def _get_health_recommendation(self, health_score: float, mood: str) -> str:
        if health_score > 0.7: return "Rozmowa płynie naturalnie - kontynuuj!"
        if health_score > 0.4: return "Rozmowa OK, ale dorzuć bardziej otwarte pytania."
        if mood == 'negative': return "Wyczuwalna frustracja – zmień temat lub zaproponuj konkretną pomoc."
        return "Rozmowa wymaga ożywienia – zaproponuj angażujące działanie."

# Globalny singleton
memory = Memory()
