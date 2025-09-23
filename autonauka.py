"""
autonauka.py — MONOLIT zgodny z KONTRAKTEM (CAT EOF, bez ucięć)

FUNKCJA PUBLICZNA:
  web_learn(query, mode="full"|"fast"|"free"|"grounded")
Zwraca:
  {"query","materials","ltm_ids","trust_avg","count","backend","draft","citations"}

CECHY:
- Jedno LLM (ten sam endpoint/model do QE/edycji faktów i do krótkiego draftu).
- SERPAPI (opcjonalnie), Firecrawl (opcjonalnie), DuckDuckGo, Wikipedia, Semantic Scholar, arXiv.
- BEZ Google CSE / "asystenta Google" (wywalone).
- Dedup globalny (hash(fakt+źródło)) + write-behind WAL (JSONL) — teraz zwraca ID z LTM.
- Głosowanie faktów: zapis do LTM po min. liczbie źródeł lub wysokim "trust".
- Wagi domen (uczenie zwrotne), limit materiałów per domena, retry HTTP.

ENV (skrót; wszystko opcjonalne):
 SERPAPI_KEY
 FIRECRAWL_KEY
 WEB_HTTP_TIMEOUT(=45), AUTO_TOPK(=8), AUTO_FETCH(=4), AUTO_MIN_CHARS(=800), AUTO_MAX_CHARS(=8000)
 AUTON_WAL=/workspace/mrd69/data/mem/autonauka.wal
 AUTON_DEDUP_MAX=1000, AUTON_DOMAIN_MAX=2, VOTE_MIN_SOURCES=2
 AUTO_TAGS="autonauka,web,evidence"
 LLM_BASE_URL, LLM_API_KEY, LLM_MODEL (domyślnie Qwen/Qwen2.5-4B-Instruct)
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import random
import re
import time
from typing import Any
from urllib.parse import quote, urlparse

# Ładowanie zmiennych środowiskowych z pliku .env
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not found. Skipping .env file loading.")


import requests
from duckduckgo_search import DDGS
from requests.adapters import HTTPAdapter

# ====== ENV ======
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
FIRECRAWL_KEY = os.getenv("FIRECRAWL_KEY", "").strip()

WEB_HTTP_TIMEOUT = int(os.getenv("WEB_HTTP_TIMEOUT", os.getenv("TIMEOUT_HTTP", "45")))
AUTO_TOPK = int(os.getenv("AUTO_TOPK", "8"))
AUTO_FETCH = int(os.getenv("AUTO_FETCH", "4"))
AUTO_MIN_CHARS = int(os.getenv("AUTO_MIN_CHARS", "800"))
AUTO_MAX_CHARS = int(os.getenv("AUTO_MAX_CHARS", "8000"))
AUTO_TAGS = [
    t.strip()
    for t in os.getenv("AUTO_TAGS", "autonauka,web,evidence").split(",")
    if t.strip()
]

AUTON_WAL_PATH = os.getenv("AUTON_WAL", "/workspace/mrd69/data/mem/autonauka.wal")
AUTON_DEDUP_MAX = int(os.getenv("AUTON_DEDUP_MAX", "1000"))
AUTON_DOMAIN_MAX = int(os.getenv("AUTON_DOMAIN_MAX", "2"))
VOTE_MIN_SOURCES = int(os.getenv("VOTE_MIN_SOURCES", "2"))

# Jedno LLM (domyślnie Qwen 4B na DeepInfra)
# Uwaga: os.getenv może zwrócić None – zadbajmy, by zawsze operować na str
_LLM_BASE_RAW = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or ""
LLM_BASE = _LLM_BASE_RAW.rstrip("/")
LLM_KEY = (os.getenv("LLM_API_KEY") or "").strip()
LLM_MODEL = (os.getenv("LLM_MODEL") or "Qwen/Qwen2.5-4B-Instruct").strip()
LLM_TO = int(os.getenv("LLM_HTTP_TIMEOUT_S", "60"))

random.seed(69)

# ====== HTTP ======

try:
    from urllib3.util import Retry
except Exception:
    from requests.packages.urllib3.util.retry import Retry  # type: ignore


def _session() -> requests.Session:
    s = requests.Session()
    # Kompatybilny Retry: bez przekazywania listy metod w konstruktorze
    r = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
    )
    # Ustaw listę dozwolonych metod w zależności od wersji urllib3
    try:
        r.allowed_methods = frozenset(["GET", "POST"])
    except Exception:
        try:
            r.method_whitelist = frozenset(["GET", "POST"])  # type: ignore[attr-defined]
        except Exception:
            pass
    ad = HTTPAdapter(max_retries=r, pool_connections=16, pool_maxsize=32)
    s.mount("http://", ad)
    s.mount("https://", ad)
    s.headers.update({"User-Agent": "mrd69-autonauka/mono-4b"})
    return s


HTTP = _session()


# Pomocnicza normalizacja parametrów zapytań HTTP do formatu str->str
def _params(d: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            out[str(k)] = ",".join(str(x) for x in v)
        else:
            out[str(k)] = str(v)
    return out


# ====== optional readability / bs4 ======
try:
    from bs4 import BeautifulSoup as _BS
except Exception:
    _BS = None
try:
    from readability import Document as _Doc
except Exception:
    _Doc = None


# ====== MEMORY adapter ======
def _memory():
    try:
        from memory import get_memory

        return get_memory()
    except Exception:
        return None


# ====== utils ======
_SPACE = re.compile(r"\s+")
TAG_SCRUB = re.compile(
    r"<script[^>]*>.*?</script>|<style[^>]*>.*?</style>", re.I | re.S
)
TAG_ALL = re.compile(r"<[^>]+>")
YEAR_RX = re.compile(r"\b(20[0-9]{2})\b")
MONTH_RX = re.compile(
    (
        r"\b("
        r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
        r"sty|lut|mar|kwi|maj|cze|lip|sie|wrz|paź|lis|gru"
        r")\b"
    ),
    re.I,
)


def _norm(t: str) -> str:
    return _SPACE.sub(" ", (t or "").strip())


def _sha1(x: str) -> str:
    return hashlib.sha1((x or "").encode("utf-8")).hexdigest()


def _domain(u: str) -> str:
    try:
        h = (urlparse(u).hostname or "").lower()
        return h[4:] if h.startswith("www.") else h
    except Exception:
        return ""


def _tok(t: str) -> list[str]:
    return re.findall(r"[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9]{2,}", (t or "").lower())


def _chunks(text: str, target: int = 900, overlap: int = 120) -> list[str]:
    w = _tok(text)
    if not w:
        return []
    out = []
    step = max(50, target - overlap)
    for i in range(0, len(w), step):
        seg = w[i : i + target]
        if len(seg) < 30:
            break
        out.append(" ".join(seg))
    return out[:20]


def _rank_jaccard(query: str, chunks: list[str]) -> list[tuple[str, float]]:
    q = set(_tok(query))
    out = []
    for ch in chunks:
        toks = _tok(ch)
        st = set(toks)
        jac = (len(q & st) / len(q | st)) if (q and st) else 0.0
        dens = min(1.0, len(toks) / 1000.0)
        out.append((ch, 0.7 * jac + 0.3 * dens))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def _recency_score(text: str) -> float:
    years = [int(y) for y in YEAR_RX.findall(text or "")]
    if not years:
        return 0.2
    y = max(years)
    try:
        now_year = time.gmtime().tm_year
        diff = max(0, now_year - y)
        base = max(0.2, 1.0 - min(diff, 6) / 6.0)
    except Exception:
        base = 0.5
    if MONTH_RX.search(text or ""):
        base = min(1.0, base + 0.1)
    return round(base, 3)


def _trust(u: str) -> float:
    d = _domain(u)
    if not d:
        return 0.0
    base = 0.2
    if d.endswith(".gov") or d.endswith(".gov.pl"):
        base = 0.9
    elif d.endswith(".edu") or d.endswith(".edu.pl"):
        base = 0.85
    elif "wikipedia.org" in d:
        base = 0.75
    elif d.endswith(".org"):
        base = 0.55
    elif d.endswith(".pl"):
        base = 0.45
    else:
        base = 0.5
    if u.startswith("https://"):
        base += 0.05
    if any(k in d for k in ("blogspot.", "medium.com", "substack.com")):
        base -= 0.1
    return max(0.0, min(1.0, base))


# ====== LLM (jedno) ======
def _llm_chat(
    system: str, user: str, maxtok: int = 512, temp: float = 0.1
) -> str | None:
    base = LLM_BASE
    key = LLM_KEY
    model = LLM_MODEL
    if not (base and key and model):
        return None
    url = base.rstrip("/") + "/chat/completions"
    try:
        r = HTTP.post(
            url,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temp,
                "max_tokens": maxtok,
            },
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            timeout=LLM_TO,
        )
        if r.status_code < 400:
            j = r.json()
            return (
                j.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                or None
            )
    except Exception:
        return None
    return None


# ====== Search/fetch ======
def _serpapi(query: str, k: int) -> list[dict]:
    if not SERPAPI_KEY:
        return []
    try:
        r = HTTP.get(
            "https://serpapi.com/search.json",
            params=_params(
                {
                    "engine": "google",
                    "q": query,
                    "num": max(1, min(10, k)),
                    "api_key": SERPAPI_KEY,
                    "hl": "pl",
                }
            ),
            timeout=WEB_HTTP_TIMEOUT,
        )
        j = r.json()
        out = []
        for it in (j.get("organic_results") or [])[:k]:
            out.append(
                {
                    "title": _norm(it.get("title") or ""),
                    "link": it.get("link") or "",
                    "snippet": _norm(it.get("snippet") or ""),
                }
            )
        return out
    except Exception:
        return []


def _ddg(query: str, k: int = 10) -> list[dict]:
    """Używa biblioteki duckduckgo-search do uzyskania wyników."""
    try:
        with DDGS(timeout=WEB_HTTP_TIMEOUT) as ddgs:
            results = list(ddgs.text(query, region="pl-pl", max_results=k))
        # Konwersja na format oczekiwany przez resztę skryptu
        return [
            {
                "title": r.get("title", ""),
                "link": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in results
        ]
    except Exception:
        return []


def _wiki_search(query: str) -> list[dict]:
    try:
        r = HTTP.get(
            "https://pl.wikipedia.org/w/rest.php/v1/search/title",
            params=_params({"q": query, "limit": 5}),
            timeout=WEB_HTTP_TIMEOUT,
        )
        j = r.json()
        out = []
        for it in (j.get("pages") or [])[:5]:
            slug = it.get("key") or it.get("id")
            if slug:
                out.append({"title": it.get("title") or slug, "slug": slug})
        return out
    except Exception:
        return []


def _wiki_summary(slug: str) -> dict[str, str]:
    try:
        r = HTTP.get(
            f"https://pl.wikipedia.org/api/rest_v1/page/summary/{quote(slug)}",
            timeout=WEB_HTTP_TIMEOUT,
        )
        if r.status_code >= 400:
            return {}
        j = r.json()
        return {
            "title": j.get("title") or slug,
            "extract": _norm(j.get("extract") or ""),
            "url": j.get("content_urls", {}).get("desktop", {}).get("page") or "",
        }
    except Exception:
        return {}


def _firecrawl(url: str) -> str:
    if not FIRECRAWL_KEY:
        return ""
    try:
        r = HTTP.post(
            "https://api.firecrawl.dev/v1/scrape",
            headers={
                "Authorization": f"Bearer {FIRECRAWL_KEY}",
                "Content-Type": "application/json",
            },
            json={"url": url, "formats": ["markdown", "text"]},
            timeout=WEB_HTTP_TIMEOUT,
        )
        j = r.json()
        cand = []
        for k in ("text", "markdown", "content"):
            v = j.get(k)
            if isinstance(v, str) and v.strip():
                cand.append(v)
        data = j.get("data") or {}
        if isinstance(data.get("content"), str):
            cand.append(data["content"])
        body = max(cand, key=len) if cand else ""
        return _norm(html.unescape(body or ""))[:AUTO_MAX_CHARS]
    except Exception:
        return ""


def _http_text(url: str) -> str:
    try:
        r = HTTP.get(url, timeout=WEB_HTTP_TIMEOUT)
        if r.status_code >= 400:
            return ""
        t = r.text
        if _Doc:
            try:
                t = _Doc(t).summary()
            except Exception:
                pass
        if _BS:
            try:
                t = str(_BS(t, "html.parser"))
            except Exception:
                pass
        t = TAG_SCRUB.sub(" ", t)
        t = TAG_ALL.sub(" ", t)
        return _norm(html.unescape(t))[:AUTO_MAX_CHARS]
    except Exception:
        return ""


# Semantic Scholar / arXiv
def _s2_search(query: str, k: int = 5) -> list[dict]:
    try:
        r = HTTP.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=_params(
                {"query": query, "limit": k, "fields": "title,url,abstract"}
            ),
            timeout=WEB_HTTP_TIMEOUT,
        )
        j = r.json()
        return [
            {
                "title": _norm(it.get("title") or ""),
                "link": it.get("url") or "",
                "snippet": _norm(it.get("abstract") or ""),
            }
            for it in (j.get("data") or [])[:k]
        ]
    except Exception:
        return []


def _arxiv_search(query: str, k: int = 5) -> list[dict]:
    try:
        r = HTTP.get(
            "https://export.arxiv.org/api/query",
            params=_params({"search_query": query, "max_results": k}),
            timeout=WEB_HTTP_TIMEOUT,
        )
        txt = r.text
        items = re.findall(
            r"<entry>.*?<title>(.*?)</title>.*?<id>(.*?)</id>.*?<summary>(.*?)</summary>",
            txt,
            re.S,
        )
        out = []
        for t, u, s in items[:k]:
            out.append(
                {
                    "title": _norm(html.unescape(t)),
                    "link": u,
                    "snippet": _norm(html.unescape(s)),
                }
            )
        return out
    except Exception:
        return []


# ====== WAL + write-behind ======
def _ensure_parent(path: str) -> None:
    try:
        d = os.path.dirname(path)
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
    except Exception:
        pass


class _WriteBehind:
    def __init__(self, wal_path: str, max_batch: int = 64) -> None:
        self.wal_path = wal_path
        self.buf: list[tuple[str, str, float]] = []
        self.max_batch = max_batch
        _ensure_parent(wal_path)

    def replay_wal(self) -> None:
        try:
            if not os.path.isfile(self.wal_path):
                return
            with open(self.wal_path, encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        j = json.loads(ln)
                        self.buf.append((j["line"], j["source"], float(j["trust"])))
                    except Exception:
                        continue
            if self.buf:
                self._flush_now()
            open(self.wal_path, "w").close()
        except Exception:
            pass

    def add(self, line: str, source: str, trust: float) -> str | None:
        """Dodaje rekord do bufora/WAL i natychmiast zapisuje do LTM (zwraca id lub None)."""
        ltm_id: str | None = None
        try:
            # natychmiastowy zapis — żeby mieć ID
            mem = _memory()
            if mem and hasattr(mem, "add_fact"):
                ltm_id = mem.add_fact(line, conf=max(0.5, float(trust)), tags=AUTO_TAGS)
            # bufor + WAL (fallback)
            self.buf.append((line, source, float(trust)))
            with open(self.wal_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"line": line, "source": source, "trust": trust},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if len(self.buf) >= self.max_batch:
                self._flush_now()
        except Exception:
            pass
        return ltm_id

    def _flush_now(self) -> None:
        mem = _memory()
        if not self.buf:
            return
        batch = list(self.buf)
        self.buf.clear()
        for line, source, trust in batch:
            try:
                if mem and hasattr(mem, "add_fact"):
                    mem.add_fact(line, conf=max(0.5, trust), tags=AUTO_TAGS)
            except Exception:
                self.buf.append((line, source, trust))
        if not self.buf:
            try:
                open(self.wal_path, "w").close()
            except Exception:
                pass

    def flush(self) -> None:
        self._flush_now()


_WB = _WriteBehind(AUTON_WAL_PATH, max_batch=64)
_WB.replay_wal()


# ====== dedup + wagi domen ======
def _load_profile() -> dict[str, Any]:
    prof: dict[str, Any] = {}
    try:
        mem = _memory()
        if mem and hasattr(mem, "get_profile"):
            prof = mem.get_profile() or {}
    except Exception:
        prof = {}
    hashes = prof.get("web_fact_hashes") or []
    dw = prof.get("web_domain_weights") or {}
    return {
        "hashes": list(hashes)[-AUTON_DEDUP_MAX:],
        "domain_weights": dict(dw),
        "local_counts": {},
    }


def _save_profile(hashes: list[str], domain_weights: dict[str, float]) -> None:
    try:
        mem = _memory()
        if not (mem and hasattr(mem, "set_profile_many")):
            return
        prof: dict[str, Any] = dict(mem.get_profile() or {})
        prof["web_fact_hashes"] = hashes[-AUTON_DEDUP_MAX:]
        prof["web_domain_weights"] = {
            k: round(float(v), 3) for k, v in domain_weights.items()
        }
        mem.set_profile_many(prof)
    except Exception:
        pass


def _dedup_key(text: str, source: str) -> str:
    return _sha1((_norm(text) + "|" + (source or "")).lower())


def _domain_weight(dw: dict[str, float], domain: str) -> float:
    v = float(dw.get(domain, 1.0))
    return max(0.2, min(2.0, v))


def apply_feedback(success: float, materials: list[dict[str, Any]]) -> None:
    try:
        st = _load_profile()
        dw = st["domain_weights"]
        delta = (success - 0.5) * 0.4
        seen = set()
        for m in materials:
            d = (m.get("domain") or "").strip()
            if not d or d in seen:
                continue
            seen.add(d)
            dw[d] = _domain_weight(dw, d) + delta
        _save_profile(st["hashes"], dw)
    except Exception:
        pass


# ====== ekstrakcja faktów + redakcja (jedno LLM) ======
def _extract(text: str, max_n: int = 6) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    out, seen = [], set()
    for s in sents:
        s = s.strip()
        if len(s) < 8:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s[:300])
        if len(out) >= max_n:
            break
    return out


def _edit_facts(facts: list[str]) -> list[str]:
    if not facts:
        return facts
    sys = (
        "Przepisz fakty w krótkie, jasne zdania, usuń dublowanie. Zwróć listę punktów."
    )
    usr = "\n".join(f"- {f}" for f in facts)
    out = _llm_chat(sys, usr, maxtok=256, temp=0.0) or ""
    if not out:
        return facts
    lines = [
        re.sub(r"^[\\-\\*\\d\\.\\)\\s]+", "", line).strip() for line in out.splitlines()
    ]
    return [line for line in lines if len(line) >= 10][: len(facts)] or facts


def _extract_keywords(query: str, max_n: int = 5) -> list[str]:
    """Wyodrębnia słowa kluczowe z zapytania za pomocą LLM."""
    if not (LLM_BASE and LLM_KEY):
        return _tok(query)[:max_n]  # fallback na prosty tokenizer

    system_prompt = (
        "You are a keyword extraction expert. Your task is to extract the most "
        f"relevant keywords from the user's query for academic search engines like "
        "arXiv and Semantic Scholar. Return a comma-separated list of at most "
        f"{max_n} keywords. Use English for keywords if appropriate for the topic. "
        "Focus on nouns and technical terms."
    )
    user_prompt = f"Query: {query}"

    response = _llm_chat(system_prompt, user_prompt, maxtok=64, temp=0.0)

    if not response:
        return _tok(query)[:max_n]

    keywords = [kw.strip() for kw in response.split(",") if kw.strip()]
    return keywords[:max_n]


# ====== ingest jednego URL ======
def _ingest_url(
    query: str,
    url: str,
    title: str,
    snippet: str,
    ltm_ids: list[str],
    state: dict[str, Any],
    deep: bool = True,
) -> dict | None:
    text = (_firecrawl(url) if deep else "") or _http_text(url)
    if len(text) < (AUTO_MIN_CHARS if deep else 300):
        return None
    chs = _chunks(text)
    chs = [c for c, _ in _rank_jaccard(query, chs)]
    base = " ".join(chs[:2]) if chs else text
    facts = _extract(base, max_n=6)
    facts = _edit_facts(facts) if facts else []
    tr = round(_trust(url), 3)
    rec = _recency_score(text)
    mat = {
        "title": title or url,
        "url": url,
        "domain": _domain(url),
        "trust": tr,
        "recency": rec,
        "snippet": (snippet or text)[:240],
        "chunks": chs[:6],
        "facts": facts,
    }
    cite = f"[{mat['domain']}] {mat['title']} — {mat['url']}"
    # głosowanie faktów + dedup
    hashes = state["hashes"]
    local = state["local_counts"]
    for f in facts:
        # Mypy: ensure explicit str and float types for helper calls
        dom_val = mat.get("domain")
        dom: str = str(dom_val if isinstance(dom_val, str) else "web")
        key = _dedup_key(str(f), dom)
        local[key] = local.get(key, 0) + 1
        if local[key] >= VOTE_MIN_SOURCES or tr >= 0.85:
            if key not in hashes:
                fid = _WB.add(f"{str(f)} (źródło: {cite})", dom, float(tr))
                if fid:
                    ltm_ids.append(fid)
                hashes.append(key)
    return mat


# ====== scoring materiałów ======
def _score_material(m: dict[str, Any], dw: dict[str, float]) -> float:
    trust = float(m.get("trust", 0.0))
    rec = float(m.get("recency", 0.5))
    dom = str(m.get("domain") or "")
    w = _domain_weight(dw, dom)
    return 0.6 * trust + 0.2 * rec + 0.2 * (w / 2.0)


# ====== Pipelines ======
def _pipeline_serp_like(
    query: str,
    topk: int,
    max_fetch: int,
    deep: bool,
    ltm_ids: list[str],
    state: dict[str, Any],
) -> list[dict]:
    mats: list[dict[str, Any]] = []
    seen = set()
    per_domain: dict[str, int] = {}

    def consider(it: dict[str, Any]):
        nonlocal mats, seen, per_domain
        u = it.get("link") or ""
        if not u:
            return
        k = _sha1(u)
        if k in seen:
            return
        d = _domain(u) or ""
        if per_domain.get(d, 0) >= AUTON_DOMAIN_MAX:
            return
        seen.add(k)
        m = _ingest_url(
            query,
            u,
            it.get("title") or "",
            it.get("snippet") or "",
            ltm_ids,
            state,
            deep=deep,
        )
        if not m:
            return
        per_domain[d] = per_domain.get(d, 0) + 1
        mats.append(m)

    # Używamy DDG jako głównego, darmowego źródła
    ddg_results = _ddg(query, k=topk)
    for it in ddg_results:
        consider(it)
        if len(mats) >= max_fetch:
            break

    # Opcjonalnie, jeśli jest klucz, dodajemy wyniki z SerpAPI
    if SERPAPI_KEY and len(mats) < max_fetch:
        for it in _serpapi(query, k=max(1, topk)):
            consider(it)
            if len(mats) >= max_fetch:
                break

    # Ostatnia deska ratunku - Wikipedia
    if len(mats) < max_fetch:
        hits = _wiki_search(query)
        for h in hits:
            s = _wiki_summary(h["slug"])
            if not s or not s.get("extract"):
                continue
            u = s.get("url") or f"https://pl.wikipedia.org/wiki/{quote(h['slug'])}"
            consider(
                {
                    "link": u,
                    "title": s.get("title", ""),
                    "snippet": s.get("extract", ""),
                }
            )
            if len(mats) >= max_fetch:
                break

    dw = state["domain_weights"]
    mats.sort(key=lambda m: _score_material(m, dw), reverse=True)
    return mats[:max_fetch]


def _pipeline_free(
    query: str, max_fetch: int, ltm_ids: list[str], state: dict[str, Any]
) -> list[dict]:
    mats: list[dict[str, Any]] = []
    per_domain: dict[str, int] = {}
    seen = set()

    def consider(it: dict[str, Any], is_scientific: bool = False):
        nonlocal mats, seen, per_domain
        u = it.get("link") or ""
        if not u:
            return
        k = _sha1(u)
        if k in seen:
            return
        d = _domain(u) or ""
        if per_domain.get(d, 0) >= AUTON_DOMAIN_MAX:
            return

        seen.add(k)
        # Dla źródeł naukowych snippet to często cały abstrakt, więc deep scan nie jest potrzebny
        m = _ingest_url(
            query,
            u,
            it.get("title") or u,
            it.get("snippet", ""),
            ltm_ids,
            state,
            deep=False,
        )
        if m:
            # Podbijamy zaufanie dla zweryfikowanych źródeł naukowych
            if is_scientific:
                m["trust"] = min(1.0, m.get("trust", 0.5) + 0.2)
            per_domain[d] = per_domain.get(d, 0) + 1
            mats.append(m)

    # 1. Wyszukiwarki naukowe jako priorytet (z użyciem słów kluczowych)
    keywords = _extract_keywords(query)
    search_query_sci = " ".join(keywords) if keywords else query
    scientific_sources = _s2_search(search_query_sci, k=max_fetch) + _arxiv_search(
        search_query_sci, k=max_fetch
    )
    for it in scientific_sources:
        if len(mats) >= max_fetch:
            break
        consider(it, is_scientific=True)

    # 2. Ogólne wyszukiwanie, jeśli brakuje wyników (z oryginalnym zapytaniem)
    if len(mats) < max_fetch:
        ddg_results = _ddg(query, k=(max_fetch - len(mats)) * 2)
        for it in ddg_results:
            if len(mats) >= max_fetch:
                break
            consider(it)

    # 3. Wikipedia jako uzupełnienie (z oryginalnym zapytaniem)
    if len(mats) < max_fetch:
        hits = _wiki_search(query)
        for h in hits:
            if len(mats) >= max_fetch:
                break
            s = _wiki_summary(h["slug"])
            if not s or not s.get("extract"):
                continue
            u = s.get("url") or f"https://pl.wikipedia.org/wiki/{quote(h['slug'])}"
            # Ręczne przetworzenie, bo nie mamy `link` w standardowym formacie
            k = _sha1(u)
            if k in seen:
                continue
            d = _domain(u)
            if per_domain.get(d, 0) >= AUTON_DOMAIN_MAX:
                continue

            seen.add(k)
            txt = s["extract"]
            chs = [
                c
                for c, _ in _rank_jaccard(query, _chunks(txt, target=700, overlap=100))
            ]
            facts = _extract(" ".join(chs) or txt, max_n=6)
            facts = _edit_facts(facts) if facts else []
            tr = _trust(u)
            rec = _recency_score(txt)
            mat = {
                "title": s.get("title") or h["slug"],
                "url": u,
                "domain": d,
                "trust": round(tr, 3),
                "recency": rec,
                "snippet": txt[:240],
                "chunks": chs[:6],
                "facts": facts,
            }
            mats.append(mat)
            per_domain[d] = per_domain.get(d, 0) + 1

            cite = f"[{mat['domain']}] {mat['title']} — {mat['url']}"
            hashes = state["hashes"]
            local = state["local_counts"]
            for f in facts:
                dom_val = mat.get("domain")
                dom: str = str(dom_val if isinstance(dom_val, str) else "web")
                key = _dedup_key(str(f), dom)
                local[key] = local.get(key, 0) + 1
                if local[key] >= VOTE_MIN_SOURCES or tr >= 0.85:
                    if key not in hashes:
                        # trust is computed as float earlier; use it to avoid mixed types
                        fid = _WB.add(
                            f"{str(f)} (źródło: {cite})",
                            dom,
                            float(tr),
                        )
                        if fid:
                            ltm_ids.append(fid)
                        hashes.append(key)

    mats.sort(
        key=lambda m: 0.75 * float(m.get("trust", 0))
        + 0.25 * float(m.get("recency", 0)),
        reverse=True,
    )
    return mats[:max_fetch]


# ====== krótki draft (tym samym LLM) ======
def _llm_draft(materials: list[dict], query: str) -> str:
    """
    Tworzy podsumowanie analityczne zamiast prostego draftu.
    Porównuje źródła, wskazuje sprzeczności i ocenia zaufanie.
    """
    if not (LLM_BASE and LLM_KEY and materials):
        return ""

    # Przygotowanie danych wejściowych dla LLM
    sources_for_llm = []
    for i, m in enumerate(materials):
        source_info = {
            "id": i + 1,
            "title": m.get("title", "Brak tytułu"),
            "domain": m.get("domain", "Brak domeny"),
            "trust": m.get("trust", 0.0),
            "recency": m.get("recency", 0.0),
            "facts": m.get("facts", []),
        }
        sources_for_llm.append(source_info)

    system_prompt = """Jesteś analitykiem AI. 
Twoim zadaniem jest krytyczna analiza dostarczonych materiałów i
stworzenie zwięzłego raportu analitycznego.Twój raport powinien zawierać 3 sekcje:
1.  **Synteza Kluczowych Faktów:** Połącz najważniejsze, powtarzające się 
    informacje z różnych źródeł w spójną całość.
2.  **Identyfikacja Sprzeczności lub Różnic:** Wskaż, jeśli istnieją, 
    istotne różnice w przedstawianych faktach lub perspektywach między 
    źródłami. Odwołuj się do źródeł po ich `id`.
3.  **Ogólna Ocena Zaufania:** Na podstawie wskaźników `trust` i `recency` 
    oraz spójności informacji, wydaj krótką ocenę wiarygodności zebranych 
    danych (np. "Wysoka", "Średnia z zastrzeżeniami", "Niska").

Odpowiadaj w formacie Markdown. Bądź zwięzły i precyzyjny.
"""

    user_prompt = f"""**Zapytanie:** {query}

**Dostarczone materiały źródłowe:**
```json
{json.dumps(sources_for_llm, ensure_ascii=False, indent=2)}
```

Wygeneruj raport analityczny:
"""

    try:
        r = HTTP.post(
            LLM_BASE.rstrip("/") + "/chat/completions",
            headers={
                "Authorization": f"Bearer {LLM_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 1024,  # Zwiększony limit na potrzeby bardziej szczegółowego raportu
            },
            timeout=LLM_TO,
        )
        if r.status_code < 400:
            j = r.json()
            return (
                j.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                or ""
            )
    except Exception:
        # Logowanie błędu może być przydatne
        # print(f"LLM draft generation failed: {e}")
        return ""
    return ""


# ====== public API ======
def web_learn(
    query: str,
    mode: str = "full",
    topk: int = AUTO_TOPK,
    max_fetch: int = AUTO_FETCH,
    min_chars: int = AUTO_MIN_CHARS,
) -> dict[str, Any]:
    q = _norm(query)
    state = _load_profile()
    ltm_ids: list[str] = []

    if mode == "free":
        mats = _pipeline_free(q, max_fetch=max_fetch, ltm_ids=ltm_ids, state=state)
        backend = "free"
    elif mode == "fast":
        mats = _pipeline_serp_like(
            q,
            topk=max(1, topk // 2),
            max_fetch=min(2, max_fetch),
            deep=False,
            ltm_ids=ltm_ids,
            state=state,
        )
        backend = "fast"
    elif mode == "grounded":
        mats = _pipeline_serp_like(
            q,
            topk=max(1, topk // 2),
            max_fetch=max_fetch,
            deep=False,
            ltm_ids=ltm_ids,
            state=state,
        )
        backend = "grounded-lite"
    else:  # full
        mats = _pipeline_serp_like(
            q, topk=topk, max_fetch=max_fetch, deep=True, ltm_ids=ltm_ids, state=state
        )
        backend = "ddgs"

    tvals = [m["trust"] for m in mats] or [0.0]
    draft = _llm_draft(mats, q)
    cites = [m["url"] for m in mats]

    _WB.flush()
    _save_profile(state["hashes"], state["domain_weights"])

    return {
        "query": q,
        "materials": mats,
        "ltm_ids": ltm_ids,
        "trust_avg": round(sum(tvals) / len(tvals), 3),
        "count": len(mats),
        "backend": backend,
        "draft": draft,
        "citations": cites,
    }


# ====== CLI ======
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="autonauka — monolit (jedno LLM 4B, bez Google CSE)"
    )
    p.add_argument("query")
    p.add_argument(
        "--mode", choices=["full", "free", "fast", "grounded"], default="grounded"
    )
    p.add_argument("--topk", type=int, default=AUTO_TOPK)
    p.add_argument("--fetch", type=int, default=AUTO_FETCH)
    args = p.parse_args()
    out = web_learn(args.query, mode=args.mode, topk=args.topk, max_fetch=args.fetch)
    print(json.dumps(out, ensure_ascii=False, indent=2))
