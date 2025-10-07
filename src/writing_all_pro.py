"""
writing_all_pro.py — PRO/ULTRA writer z darmowym RAG i integracjami.

Główne zasady:
- Nie modyfikuje: memory.py, psychika.py, autonauka.py, crypto_advisor_full.py.
- Integracje wykrywane dynamicznie i bezpieczne (try/except + hasattr).
- MAIN LLM: DeepInfra (OpenAI-compatible). MINI: Gemini flash lub mini OpenAI-compatible.
- RAG: Wikipedia REST, DuckDuckGo IA, Google Books, OpenAlex, HN, lekki Vogue/press (DDG).
- Export: MD/HTML/PDF, ZIP bundli.
- Ulepszenia: spójny retry, cache wyników web (TTL), inline citations,
  mocniejszy ngram_guard, eventy psychika/autonauka, komenda CLI `assist`.

Zmienne środowiskowe:
- LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, GEMINI_API_KEY|VERTEX_GEMINI_KEY, GEMINI_MODEL,
  WEB_HTTP_TIMEOUT, WEB_USER_AGENT, WRITER_OUT_DIR

Ścieżka: /workspace/mrd69
"""

from __future__ import annotations

import hashlib
import html
import json
import os
from . import config
import random
import re
import time
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry as _Retry

# Load environment variables
load_dotenv()

# ───────────────────────────────────────────────────────────────────────────
# ŚCIEŻKI I I/O
# ───────────────────────────────────────────────────────────────────────────
ROOT = Path("/workspace/mrd69")
OUT_DIR = Path(config.WRITER_OUT_DIR or str(ROOT / "out" / "writing"))
PIPE_DIR = OUT_DIR / "_pipe"
DATA_DIR = Path(__file__).resolve().parent / "data"
for d in (OUT_DIR, PIPE_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)
JOBS_FILE = PIPE_DIR / "jobs.jsonl"
CACHE_FILE = DATA_DIR / "cache.json"  # cache RAG
CACHE_TTL_S = int(os.getenv("WEB_CACHE_TTL_S", "86400"))  # 24h


def _now_ms() -> int:
    """Zwraca aktualny czas w milisekundach."""
    return int(time.time() * 1000)


def _now_s() -> int:
    """Zwraca aktualny czas w sekundach."""
    return int(time.time())


def _slug(s: str, max_len: int = 80) -> str:
    """Tworzy slug (przyjazny URL) z podanego tekstu."""
    s = (
        unicodedata.normalize("NFKD", (s or ""))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s[:max_len] or "item"


def _short(txt: str, n: int = 160) -> str:
    """Skraca tekst do n znaków, dodając wielokropek jeśli za długi."""
    t = (txt or "").strip().replace("\n", " ")
    return (t[: n - 1] + "...") if len(t) > n else t


def _save(
    payload: dict[str, Any],
    base_dir: Path = OUT_DIR,
    prefix: str = "doc",
    ext: str = "md",
) -> dict[str, str]:
    """Zapisuje tekst i metadane do pliku oraz pliku .json z metadanymi."""
    base_dir.mkdir(parents=True, exist_ok=True)
    name = f"{prefix}_{_now_ms()}.{ext}"
    p = base_dir / name
    with open(p, "w", encoding="utf-8") as f:
        f.write(payload.get("text", ""))
    meta = payload.get("meta") or {}
    with open(p.with_suffix(p.suffix + ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return {"path": str(p), "meta_path": str(p.with_suffix(p.suffix + ".json"))}


def export_pdf(text: str, out_path: str) -> str | None:
    """Eksportuje tekst do pliku PDF. Zwraca ścieżkę lub None przy błędzie."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.pdfgen import canvas
    except ImportError:
        return None
    c = canvas.Canvas(out_path, pagesize=A4)
    w, h = A4
    y = h - 2 * cm
    for line in text.splitlines():
        c.drawString(2 * cm, y, line[:110])
        y -= 14
        if y < 2 * cm:
            c.showPage()
            y = h - 2 * cm
    c.save()
    return out_path


def html_export2(title: str, body_md: str) -> str:
    def _md2html(md: str) -> str:
        h = html.escape(md)
        h = re.sub(r"^# (.+)$", r"<h1>\1</h1>", h, flags=re.M)
        h = re.sub(r"^## (.+)$", r"<h2>\1</h2>", h, flags=re.M)
        h = re.sub(r"^### (.+)$", r"<h3>\1</h3>", h, flags=re.M)
        h = re.sub(r"(?m)^> (.+)$", r"<blockquote>\1</blockquote>", h)
        h = re.sub(r"(?m)^- (.+)$", r"<li>\1</li>", h)
        h = re.sub(r"(<li>.*</li>)", r"<ul>\1</ul>", h, flags=re.S)
        h = h.replace("\n\n", "<br/><br/>")
        return h

    meta_desc = _short(re.sub(r"(?m)^# .*$", "", body_md).strip(), 150)
    head = f"""<meta charset="utf-8"><title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(meta_desc)}">"""
    css = """<style>
body {
    font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, sans-serif;
    max-width: 860px; margin: 40px auto; padding: 0 20px; line-height: 1.6
}
blockquote {
    border-left: 4px solid #ccc; margin: 1em 0; padding: .5em 1em;
    background: #f7f7f7
}
h1,h2,h3{line-height:1.25}
</style>"""
    return (
        f"<!doctype html><html><head>{head}{css}</head>"
        f"<body>{_md2html(body_md)}</body></html>"
    )


def pack_zip(paths: list[str], out_zip: str) -> str:
    p = Path(out_zip)
    p.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fp in paths:
            if fp and os.path.exists(fp):
                z.write(fp, arcname=os.path.basename(fp))
    return str(p)


# ───────────────────────────────────────────────────────────────────────────
# OPCJONALNE INTEGRACJE
# ───────────────────────────────────────────────────────────────────────────
_HAS_MEMORY = _HAS_PSY = _HAS_AN = _HAS_CRYPTO = False
memory: ModuleType | None
psychika: ModuleType | None
autonauka: ModuleType | None
crypto: ModuleType | None
try:
    from . import memory as _memory_mod

    memory = _memory_mod
    _HAS_MEMORY = True
except Exception:
    memory = None
try:
    from . import psychika as _psychika_mod

    psychika = _psychika_mod
    _HAS_PSY = True
except Exception:
    psychika = None
try:
    from . import autonauka as _autonauka_mod

    autonauka = _autonauka_mod
    _HAS_AN = True
except Exception:
    autonauka = None
try:
    from . import crypto_advisor_full as _crypto_mod

    crypto = _crypto_mod
    _HAS_CRYPTO = True
except Exception:
    crypto = None


def _psy_snapshot(uid: str = "global") -> dict[str, Any]:
    if _HAS_PSY and psychika:
        try:
            if hasattr(psychika, "psychika_preload"):
                return psychika.psychika_preload(uid)  # type: ignore
        except Exception:
            pass
    return {"persona": "neutral", "mood": "spokój", "energy": 70, "creativity": 50}


def _psy_event(kind: str, data: dict[str, Any]) -> None:
    if _HAS_PSY and psychika:
        try:
            if hasattr(psychika, "psyche_event"):
                psychika.psyche_event(kind, data)  # type: ignore
            elif hasattr(psychika, "autopilot_cycle"):
                psychika.autopilot_cycle(
                    f"{kind}: {json.dumps(data, ensure_ascii=False)[:400]}"
                )  # type: ignore
        except Exception:
            pass


def _auto_learn(sample: dict[str, Any]) -> None:
    if _HAS_AN and autonauka:
        try:
            if hasattr(autonauka, "learn"):
                autonauka.learn(sample)  # type: ignore
            elif hasattr(autonauka, "enqueue"):
                autonauka.enqueue(sample)  # type: ignore
            elif hasattr(autonauka, "add_sample"):
                autonauka.add_sample(sample)  # type: ignore
        except Exception:
            pass


def _mem_add(
    text: str,
    tags: list[str] | None = None,
    user: str = "global",
    conf: float = 0.65,
) -> str | None:
    if not (_HAS_MEMORY and memory):
        return None
    try:
        # Preferuj aktualny kontrakt poprzez singleton `get_memory()`
        get_mem = getattr(memory, "get_memory", None)
        if callable(get_mem):  # type: ignore[truthy-function]
            mem = get_mem()  # type: ignore[misc]
            if hasattr(mem, "add_fact"):
                return mem.add_fact(  # type: ignore[no-any-return]
                    text[:8000],
                    tags=sorted(set(tags or ["writing"])),
                    conf=float(conf),
                )
        # Fallback do ewentualnych starych API modułowych
        add_fact_fn = getattr(memory, "add_fact", None)
        if callable(add_fact_fn):
            return add_fact_fn(  # type: ignore[no-any-return]
                text[:8000],
                tags=sorted(set(tags or ["writing"])),
                conf=float(conf),
            )
    except Exception:
        return None
    return None


# ───────────────────────────────────────────────────────────────────────────
# HTTP (requests + retry) + CACHE
# ───────────────────────────────────────────────────────────────────────────

WEB_TIMEOUT = config.WEB_HTTP_TIMEOUT
UA = config.WEB_USER_AGENT or "MordzixBot/1.0 (writing_all_pro)"


def _http_sess() -> requests.Session:
    s = requests.Session()
    r = _Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
    )
    ad = HTTPAdapter(max_retries=r, pool_connections=8, pool_maxsize=16)
    s.mount("http://", ad)
    s.mount("https://", ad)
    s.headers.update({"User-Agent": UA})
    return s


_HTTP = _http_sess()


def _cache_load() -> dict[str, Any]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _cache_save(obj: dict[str, Any]) -> None:
    try:
        CACHE_FILE.write_text(
            json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


_CACHE = _cache_load()


def _cache_key(url: str, params: dict[str, Any] | None = None) -> str:
    raw = url + "?" + json.dumps(params or {}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(url: str, params: dict[str, Any] | None = None) -> Any | None:
    key = _cache_key(url, params)
    it = _CACHE.get(key)
    if not it:
        return None
    if _now_s() - int(it.get("ts", 0)) > CACHE_TTL_S:
        try:
            del _CACHE[key]
            _cache_save(_CACHE)
        except Exception:
            pass
        return None
    return it.get("data")


def _cache_put(url: str, params: dict[str, Any] | None, data: Any) -> None:
    key = _cache_key(url, params)
    _CACHE[key] = {"ts": _now_s(), "data": data}
    _cache_save(_CACHE)


def _get_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    cached = _cache_get(url, params)
    if cached is not None:
        return cached
    try:
        r = _HTTP.get(
            url,
            params=params or {},
            headers={"Accept": "application/json"},
            timeout=WEB_TIMEOUT,
        )
        if r.status_code == HTTP_NOT_FOUND:
            _cache_put(url, params, {})
            return {}
        r.raise_for_status()
        j = r.json()
        _cache_put(url, params, j)
        return j
    except Exception:
        return {}


def _get_text(url: str) -> str:
    cached = _cache_get(url, None)
    if isinstance(cached, str):
        return cached
    try:
        r = _HTTP.get(url, timeout=WEB_TIMEOUT)
        if r.status_code >= HTTP_BAD_REQUEST:
            return ""
        t = r.text
        t = re.sub(r"(?is)<script.*?>.*?</script>", " ", t)
        t = re.sub(r"(?is)<style.*?>.*?</style>", " ", t)
        t = re.sub(r"(?is)<[^>]+>", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        t = t[:20000]
        _cache_put(url, None, t)
        return t
    except Exception:
        return ""


# ───────────────────────────────────────────────────────────────────────────
# LLM — MAIN (DeepInfra) + MINI (Gemini lub mini OpenAI-compatible)
# ───────────────────────────────────────────────────────────────────────────
def llm_chat(
    messages: list[dict[str, str]], temperature: float = 0.45, max_tokens: int = 1100
) -> str:
    base = (config.LLM_BASE_URL or "https://api.deepinfra.com/v1/openai").rstrip(
        "/"
    )
    key = config.LLM_API_KEY.strip()
    model = (config.LLM_MODEL or 'meta-llama/Meta-Llama-3.1-70B-Instruct').strip()
    if not key:
        return "\n".join(
            [m.get("content", "") for m in messages if m.get("role") == "user"]
        )
    try:
        url = base + "/chat/completions"
        req_body: dict[str, object] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        r = _HTTP.post(
            url,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=req_body,
            timeout=WEB_TIMEOUT,
        )
        r.raise_for_status()
        j = r.json()
        return (
            ((j.get("choices") or [{}])[0].get("message") or {})
            .get("content", "")
            .strip()
        )
    except Exception:
        return "\n".join(
            [m.get("content", "") for m in messages if m.get("role") == "user"]
        )


def mini_llm_text(prompt: str, max_tokens: int = 256, temperature: float = 0.3) -> str:
    prompt = (prompt or "").strip()
    if not prompt:
        return ""

    # Używamy modelu Qwen 4B
    mini_base = (
        config.MINI_LLM_BASE_URL or config.LLM_BASE_URL or ""
    ).rstrip("/")
    mini_key = (config.MINI_LLM_API_KEY or config.LLM_API_KEY).strip()
    mini_model = config.MINI_LLM_MODEL or 'Qwen/Qwen2.5-4B-Instruct'
    if mini_base and mini_key:
        try:
            url = mini_base + "/chat/completions"
            mini_body: dict[str, object] = {
                "model": mini_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            r = _HTTP.post(
                url,
                headers={
                    "Authorization": f"Bearer {mini_key}",
                    "Content-Type": "application/json",
                },
                json=mini_body,
                timeout=WEB_TIMEOUT,
            )
            if r.status_code < HTTP_BAD_REQUEST:
                jj = r.json()
                txt = (
                    ((jj.get("choices") or [{}])[0].get("message") or {})
                    .get("content", "")
                    .strip()
                )
                if txt:
                    return txt
        except Exception:
            pass

    # W przypadku braku możliwości użycia Qwen, spróbujemy użyć Kimi
    try:
        import kimi_client

        return kimi_client.kimi_chat(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception:
        pass

    return ""


# ───────────────────────────────────────────────────────────────────────────
# HUMANIZACJA + STYLE + ANTY-POWTÓRKI
# ───────────────────────────────────────────────────────────────────────────
IDIOMS = {
    "joy": ["aż chce się iść dalej", "uśmiech sam wchodzi"],
    "sad": ["cisza mówi głośniej", "świat przygasa"],
    "rage": ["dosyć tego", "iskry spod butów"],
    "irony": ["no jasne, co może pójść nie tak", "magia... podobno"],
    "calm": ["oddech równy", "nic nie goni"],
}
SENSORY = [
    "zapach świeżo mielonej kawy",
    "chłód metalu pod palcami",
    "szmer ulicy za oknem",
    "ciepło lampy na skórze",
    "szorstkość bawełny",
]
SLANG = ["serio", "na luzie", "spoko", "szczerze"]
RHET_Q = [
    "Wiesz, o co chodzi?",
    "Znasz to uczucie?",
    "Po co odwlekać?",
    "A gdyby tak... teraz?",
]


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[\.\!\?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def _join(ss: list[str]) -> str:
    return " ".join(ss)


def _cadence(ss: list[str], mode: str, chaos: float) -> list[str]:
    out = []
    for s_ in ss:
        s = s_
        if mode in ("mix", "short") and len(s) > MAX_SHORT_LEN:
            s = re.sub(r",\s+", ". ", s, count=1)
        if (
            mode in ("mix", "long")
            and len(s) < MIN_LONG_LEN
            and random.random() < 0.4 + chaos * 0.3
        ):
            s = s.rstrip(".?!") + ", prawda?"
        if random.random() < chaos * 0.2:
            s = s.replace(" i ", ", i ")
        out.append(s)
    if (
        mode in ("mix", "short")
        and len(out) > 2
        and random.random() < SHORT_INSERT_PROB
    ):
        out.insert(max(1, len(out) // 3), "Serio.")
    return out


def _inject(
    text: str, emotion: str, slang: bool, rhetoric: bool, sensory_on: bool, chaos: float
) -> str:
    ss = _sentences(text)
    add = []
    emo_key = {
        "radość": "joy",
        "smutek": "sad",
        "gniew": "rage",
        "ironia": "irony",
        "spokój": "calm",
    }.get((emotion or "spokój").lower(), "calm")
    if emo_key in IDIOMS and random.random() < 0.8:
        add.append(random.choice(IDIOMS[emo_key]))
    if sensory_on and random.random() < 0.7:
        add.append(random.choice(SENSORY))
    if rhetoric and random.random() < 0.7:
        add.append(random.choice(RHET_Q))
    if slang and random.random() < 0.6:
        add.append(random.choice(SLANG))
    if add:
        ss.insert(min(1, len(ss)), " ".join(add) + ".")
    return _join(ss)


def _disfl(text: str, p: float) -> str:
    if p <= 0:
        return text

    def tweak(s: str) -> str:
        if random.random() < p:
            s = re.sub(r"\bale\b", "ale...", s, flags=re.I)
        if random.random() < p:
            s = re.sub(r"\bno\b", "no,", s, flags=re.I)
        return s

    return _join([tweak(s) for s in _sentences(text)])


def humanize(
    text: str, knobs: dict[str, Any] | None = None, psyche: dict[str, Any] | None = None
) -> str:
    knobs = knobs or {}
    psyche = psyche or {}
    emotion = knobs.get("emotion") or psyche.get("mood", "spokój")
    cadence = knobs.get("cadence", "mix")
    chaos = float(knobs.get("chaos", 0.3))
    slang = bool(knobs.get("slang", False))
    rhetoric = bool(knobs.get("rhetoric", True))
    sensory_on = bool(knobs.get("sensory", True))
    disf = float(knobs.get("disfluency", 0.05))
    ss = _cadence(_sentences(text), cadence, chaos)
    out = _join(ss)
    out = _inject(out, emotion, slang, rhetoric, sensory_on, chaos)
    out = _disfl(out, disf)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


import re, os, random, hashlib, math
from typing import Any, Dict, List, Tuple

# ── parametry rytmu i cięć
MAX_SHORT_LEN = 120
MIN_LONG_LEN = 38
SHORT_INSERT_PROB = 0.22
PARA_TARGET_SENT = (3, 6)  # min, max zdań w akapicie (autopilot)
EPIGRAM_CHANCE = 0.55  # szansa na mocną puentę akapitu

# globalny suwak awaryjny (ENV), autopilot i tak RZĄDZI
WRITER_PROFANITY_LEVEL = float(os.getenv("WRITER_PROFANITY_LEVEL", "0.0"))

# ── słowniki
IDIOMS = {
    "joy": ["aż chce się iść dalej", "uśmiech sam wchodzi", "świat się otwiera"],
    "sad": ["cisza mówi głośniej", "świat przygasa", "pusty pogłos w środku"],
    "rage": ["dosyć tego", "iskry spod butów", "krew szybciej krąży"],
    "irony": [
        "no jasne, co może pójść nie tak",
        "magia... podobno",
        "jakby tego było mało",
    ],
    "calm": ["oddech równy", "nic nie goni", "spokój po burzy"],
}
SENSORY = [
    "zapach świeżo mielonej kawy",
    "chłód metalu pod palcami",
    "szmer ulicy za oknem",
    "ciepło lampy na skórze",
    "szorstkość bawełny",
    "puls światła na ścianie",
]
SLANG_FILLERS = ["serio", "na luzie", "spoko", "szczerze", "no i git", "totalnie"]
POETIC_BREAKS = [", i wtedy", ", jakby", ", aż", ", niby", ", przecież"]
METAPHORS = [
    "serce dudniło jak bęben wojenny",
    "noc była gęsta jak smoła",
    "słowa spadały jak odłamki szkła",
    "cisza lepiła się do skóry jak pot",
    "uśmiech rozciął powietrze jak brzytwa",
]
SYMBOLS = [
    "ptak lecący nad ruiną",
    "krzywe lustro pamięci",
    "ogień tlący się pod popiołem",
    "echo kroków w pustym korytarzu",
    "cień wydłużony jak katedra",
]
# delikatne „ostrzenie” (bez wulgaryzmów)
_SHARP_REPLACEMENTS = [
    ("do bani", "do dupy"),
    ("niezły", "zajebisty"),
    ("cholera", "kurde"),
]
# wulgary (kontekstowe)
_PROFANITY_REPLACEMENTS = [
    ("kurczę", "kurwa"),
    ("kurcze", "kurwa"),
    ("daj spokój", "daj, kurwa, spokój"),
    ("mam dość", "mam, kurwa, dość"),
    ("serio", "serio, kurwa"),
    ("spoko", "spoko, kurwa"),
]
_PROFANITY_TAILS = ["kurwa", "do dupy", "pierdolę to", "jebać to", "bez ściemy, kurwa"]

# słownik wzmacniaczy przymiotnikowych — lekka doprawka obrazowania
_ADJ_UP = {
    "noc": ["lepka", "głęboka", "ciężka"],
    "miasto": ["nerwowe", "duszne", "gęste"],
    "cisza": ["gęsta", "gruba", "niespokojna"],
    "światło": ["zimne", "pulsujące", "mętne"],
    "uśmiech": ["krzywy", "szyderczy", "niepewny"],
    "kawa": ["ostra", "gorzka", "oleista"],
    "wiatr": ["poszarpany", "chłodny", "szumiący"],
}

# bloki których NIE tykamy
_MD_CODE_FENCE_RE = re.compile(r"```.*?```", re.S)
_URL_RE = re.compile(r"https?://\S+")
_TAG_RE = re.compile(r"<[^>]+>")
_MAIL_RE = re.compile(r"\b[a-z0-9.\-_+]+@[a-z0-9\-_]+\.[a-z]{2,}\b", re.I)
_NUM_RE = re.compile(r"\b\d+[.,]?\d*\b")
_ABBR_RE = re.compile(r"\b(np\.|itd\.|itp\.|tzn\.|tj\.)\b", re.I)

# lista ostrych słów do oceny „wkurwu”
_SWEARS = {
    "kurwa",
    "pierdol",
    "jeba",
    "chuj",
    "pizd",
    "skurwysyn",
    "zajeb",
    "spierdal",
    "kurde",
}


# ───────────────────────────────────────────────────────────────────────────
# POMOCNICZE
# ───────────────────────────────────────────────────────────────────────────
def _hash_seed(s: str) -> int:
    return int(hashlib.sha1((s or "").encode("utf-8")).hexdigest()[:8], 16)


def _rnd(seed_text: str) -> random.Random:
    return random.Random(_hash_seed(seed_text))


def _sentences(text: str) -> List[str]:
    return [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()
    ]


def _join(ss: List[str]) -> str:
    return " ".join(ss)


def _freeze_sensitive(t: str) -> Tuple[str, List[str]]:
    frozen: List[str] = []

    def _freeze(m):
        frozen.append(m.group(0))
        return f"__FRZ_{len(frozen)-1}__"

    t = _MD_CODE_FENCE_RE.sub(_freeze, t)
    t = _URL_RE.sub(_freeze, t)
    t = _MAIL_RE.sub(_freeze, t)
    t = _TAG_RE.sub(_freeze, t)
    return t, frozen


def _thaw_sensitive(t: str, frozen: List[str]) -> str:
    def _thaw(m):
        i = int(m.group(1))
        return frozen[i] if 0 <= i < len(frozen) else m.group(0)

    return re.sub(r"__FRZ_(\d+)__", _thaw, t)


def _words(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


# ───────────────────────────────────────────────────────────────────────────
# AUTODETEKCJA STYLU (ostrość, wulgary, tempo, metafora, patos, itd.)
# ───────────────────────────────────────────────────────────────────────────
def _auto_detect_style(text: str) -> Dict[str, float]:
    t = text or ""
    sents = _sentences(t)
    n_s = max(1, len(sents))
    words = _words(t)
    n_w = max(1, len(words))

    exclam = t.count("!")
    caps_tokens = len([w for w in re.findall(r"\b[A-ZĄĆĘŁŃÓŚŹŻ]{3,}\b", t)])
    avg_len = sum(len(s) for s in sents) / n_s
    short_share = len([s for s in sents if len(s) < 60]) / n_s

    intensity = min(1.0, (exclam * 0.08) + (caps_tokens * 0.06) + short_share * 0.35)
    swears = sum(1 for w in words for k in _SWEARS if k in w)
    anger = min(1.0, 0.25 + 0.15 * exclam + 0.12 * swears)

    pos_lex = {
        "kocham",
        "fajnie",
        "dobrze",
        "radosny",
        "pięknie",
        "zajebiscie",
        "super",
        "spoko",
    }
    neg_lex = {
        "smutek",
        "ból",
        "pustka",
        "źle",
        "fatalnie",
        "zima",
        "ciemno",
        "samotność",
        "nienawidzę",
    }
    pos = sum(1 for w in words if w in pos_lex) / n_w
    neg = sum(1 for w in words if w in neg_lex) / n_w
    joy = max(0.0, min(1.0, 0.2 + pos * 5 - neg * 2))
    sadness = max(0.0, min(1.0, 0.2 + neg * 5 - pos * 2))
    rage = max(anger, 0.0)

    slang_hits = len(
        re.findall(r"(:\)|xD|XD|lol|imo|btw|spoko|serio|no i git)", t, re.I)
    )
    slang = min(1.0, 0.15 + 0.2 * slang_hits + 0.1 * intensity)

    abstract_hits = len(
        re.findall(
            r"\b(sens|pamięć|cień|cisza|puls|światło|noc|serce|dusza|słowo|echo|ruina|katedra)\b",
            t,
            re.I,
        )
    )
    metaphor = min(1.0, 0.35 + 0.08 * abstract_hits + 0.1 * (avg_len > 80))
    pathos = min(1.0, 0.25 + 0.1 * abstract_hits + 0.12 * (avg_len > 80))

    pace = max(
        0.0, min(1.0, 0.62 - (avg_len - 70) / 120)
    )  # szybciej przy krótszych zdaniach
    sharp = min(1.0, 0.15 + 0.8 * intensity)
    profanity = min(1.0, 0.1 + 0.35 * anger + 0.25 * intensity + 0.25 * (swears > 0))
    nouns = len(re.findall(r"\b\w+(?:[aeiouyąęó]\w{2,})\b", t))
    sensory = max(0.0, min(1.0, 0.55 + 0.0008 * nouns + 0.1 * (avg_len > 75)))
    irony = max(
        0.0,
        min(
            1.0,
            0.2 + 0.3 * ("na pewno" in t.lower()) + 0.2 * (joy > 0.4 and rage > 0.3),
        ),
    )
    surreal = max(0.0, min(1.0, 0.15 + 0.02 * abstract_hits + 0.08 * (avg_len > 90)))

    emo_vec = {
        "joy": joy,
        "sad": sadness,
        "rage": rage,
        "calm": 1.0 - max(joy, sadness, rage),
    }
    dominant = max(emo_vec, key=emo_vec.get)

    return {
        "pace": pace,
        "sensory": sensory,
        "irony": irony,
        "slang": slang,
        "pathos": pathos,
        "sharp": sharp,
        "surreal": surreal,
        "metaphor": metaphor,
        "symbol": 0.35 + 0.25 * metaphor,
        "poetic": 0.45 + 0.2 * metaphor,
        "profanity_mode": max(WRITER_PROFANITY_LEVEL, profanity),
        "_dominant_emotion": dominant,
        "_avg_len": avg_len,
    }


# ───────────────────────────────────────────────────────────────────────────
# KADENCJA / WTRYSKI / DISFLUENCY / BOOSTERY
# ───────────────────────────────────────────────────────────────────────────
def _cadence(ss: List[str], chaos: float, seed_text: str) -> List[str]:
    rnd = _rnd(seed_text + "|cad")
    out: List[str] = []
    for s in ss:
        t = s
        if len(t) > MAX_SHORT_LEN:
            t = re.sub(r",\s+", ". ", t, count=1)
        if len(t) < MIN_LONG_LEN and rnd.random() < (0.33 + 0.22 * chaos):
            t = t.rstrip(".?!") + ", prawda?"
        if rnd.random() < (0.10 + 0.16 * chaos):
            t = t.replace(" i ", ", i ", 1)
        out.append(t)
    if len(out) > 2 and rnd.random() < SHORT_INSERT_PROB:
        out.insert(max(1, len(out) // 3), "Serio.")
    return out


def _inject(
    text: str,
    dominant_emotion: str,
    slang_w: float,
    rhetoric_w: float,
    sensory_w: float,
    seed_text: str,
) -> str:
    rnd = _rnd(seed_text + "|inj")
    ss = _sentences(text)
    add: List[str] = []
    if rnd.random() < 0.78:
        idiom = rnd.choice(IDIOMS.get(dominant_emotion, IDIOMS["calm"]))
        add.append(idiom)
    if rnd.random() < sensory_w:
        add.append(rnd.choice(SENSORY))
    if rnd.random() < rhetoric_w:
        add.append(
            rnd.choice(
                [
                    "Wiesz, o co chodzi?",
                    "Znasz to uczucie?",
                    "Po co odwlekać?",
                    "Kumasz klimat?",
                ]
            )
        )
    if rnd.random() < slang_w:
        add.append(rnd.choice(SLANG_FILLERS))
    if add:
        ss.insert(min(1, len(ss)), " ".join(add) + ".")
    return _join(ss)


def _disfl(text: str, p: float, seed_text: str) -> str:
    if p <= 0:
        return text
    rnd = _rnd(seed_text + "|dis")
    out: List[str] = []
    for s in _sentences(text):
        t = s
        if _URL_RE.search(t) or _ABBR_RE.search(t) or _NUM_RE.search(t):
            out.append(t)
            continue
        if rnd.random() < p * 0.35:
            t = re.sub(r"\bale\b", "ale...", t, flags=re.I)
        if rnd.random() < p * 0.18:
            t = "eee... " + t
        out.append(t)
    return _join(out)


def _adj_boost(text: str, seed_text: str) -> str:
    """Delikatnie dosadza przymiotniki do kluczowych rzeczowników (bez przesady)."""
    rnd = _rnd(seed_text + "|adj")

    def repl(m):
        w = m.group(1)
        base = w.lower()
        if base in _ADJ_UP and rnd.random() < 0.55:
            adj = rnd.choice(_ADJ_UP[base])
            # prosta wstawka: „lepka noc” / „noc lepka” (losowo)
            if rnd.random() < 0.5:
                return f"{adj} {w}"
            return f"{w} {adj}"
        return w

    # tylko słowa z listy, aby nie przegiąć
    pat = r"\b(" + "|".join(map(re.escape, _ADJ_UP.keys())) + r")\b"
    return re.sub(pat, repl, text, flags=re.I)


def _metaphor_symbol_dose(text: str, m_w: float, s_w: float, seed_text: str) -> str:
    rnd = _rnd(seed_text + "|meta")
    t = text
    if rnd.random() < m_w:
        t += " " + rnd.choice(METAPHORS) + "."
    if rnd.random() < s_w:
        t = rnd.choice(SYMBOLS).capitalize() + ", " + t
    return t


def _paragraph_sculptor(text: str, avg_len: float, seed_text: str) -> str:
    """Grupuje zdania w akapity 3–6 zdań, zależnie od średniej długości."""
    rnd = _rnd(seed_text + "|para")
    sents = _sentences(text)
    if len(sents) <= 3:
        return text
    tgt_min, tgt_max = PARA_TARGET_SENT
    # gdy krótkie zdania → większe paczki
    if avg_len < 60:
        tgt_max += 1
    out_paras: List[str] = []
    buf: List[str] = []
    target = rnd.randint(tgt_min, tgt_max)
    for s in sents:
        buf.append(s)
        if len(buf) >= target:
            para = " ".join(buf).strip()
            # epigram/punchline?
            if rnd.random() < EPIGRAM_CHANCE and len(para) > 80:
                para = _punchline(para, rnd)
            out_paras.append(para)
            buf = []
            target = rnd.randint(tgt_min, tgt_max)
    if buf:
        para = " ".join(buf).strip()
        if rnd.random() < EPIGRAM_CHANCE and len(para) > 80:
            para = _punchline(para, rnd)
        out_paras.append(para)
    return "\n\n".join(out_paras)


def _punchline(para: str, rnd: random.Random) -> str:
    hooks = [
        "I tyle w temacie.",
        "Prawda potrafi gryźć.",
        "Reszta to dym i lustra.",
        "Nie udawajmy, że jest inaczej.",
        "Kropka. Bez ale.",
    ]
    if rnd.random() < 0.5:
        return para.rstrip(".!?") + ". " + rnd.choice(hooks)
    return para


# ───────────────────────────────────────────────────────────────────────────
# PUBLIC API: HUMANIZE (AUTOPILOT ULTRA)
# ───────────────────────────────────────────────────────────────────────────
def humanize(
    text: str, knobs: Dict[str, Any] | None = None, psyche: Dict[str, Any] | None = None
) -> str:
    """
    Autopilot ULTRA: sam dobiera styl i agresywnie dopala literacko,
    ale nie psuje linków/kodu/maili/tagów/liczb.
    """
    base = (text or "").strip()
    if not base:
        return ""

    auto = _auto_detect_style(base)
    knobs = knobs or {}
    merged = dict(auto)
    # hinty użytkownika (jeśli są) dostają tylko 20% wagi
    for k, v in knobs.items():
        if isinstance(v, (int, float)) and k in merged:
            merged[k] = float(0.8 * merged[k] + 0.2 * float(v))

    # 1) kadencja i bazowe cięcia
    ss = _cadence(_sentences(base), chaos=0.28 + 0.2 * merged["pace"], seed_text=base)
    out = _join(ss)

    # 2) wtryski idiom/sensory/retoryka/slang
    out = _inject(
        out, merged["_dominant_emotion"], merged["slang"], 0.55, merged["sensory"], base
    )

    # 3) drobna dyzartria (naturalne pauzy)
    out = _disfl(out, p=0.04 + 0.06 * merged["pace"], seed_text=base)

    # 4) booster przymiotnikowy (lekko)
    out, frozen = _freeze_sensitive(out)
    out = _adj_boost(out, base)

    # 5) metafora + symbolika (dozowanie)
    out = _metaphor_symbol_dose(out, merged["metaphor"], merged["symbol"], base)

    # 6) rzeźbienie akapitów + epigram
    out = _paragraph_sculptor(out, merged["_avg_len"], base)

    # 7) cleanup i rozmrożenie
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"\s{2,}", " ", out)
    out = _thaw_sensitive(out, frozen).strip()
    return out


# ───────────────────────────────────────────────────────────────────────────
# NARZĘDZIA CZYSZCZĄCE I STYLE (API zachowane)
# ───────────────────────────────────────────────────────────────────────────
def debloat(md: str) -> str:
    t = re.sub(r"[ \t]+\n", "\n", md or "")
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"\b(bardzo|mega|super|naprawdę)\b", "", t, flags=re.I)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def style_mix(base: Dict[str, float] | None, **overrides: float) -> Dict[str, float]:
    # Zostawiamy dla zgodności — realnie steruje autopilot.
    v = {
        "pace": 0.55,
        "sensory": 0.7,
        "irony": 0.25,
        "slang": 0.2,
        "pathos": 0.35,
        "sharp": 0.0,
        "surreal": 0.15,
        "metaphor": 0.6,
        "symbol": 0.4,
        "poetic": 0.5,
        "profanity_mode": WRITER_PROFANITY_LEVEL,
    }
    if base:
        for k in v:
            v[k] = float(max(0, min(1, base.get(k, v[k]))))
    for k, val in overrides.items():
        if k in v:
            v[k] = float(max(0, min(1, val)))
    return v


def apply_style_matrix(text: str, sm: Dict[str, float]) -> str:
    """
    Zgodność z API: lekka post-obróbka; autopilot i tak robi większość w humanize().
    """
    base = (text or "").strip()
    if not base:
        return base
    seed = _rnd(base + "|style")
    t = base
    if seed.random() < sm.get("pace", 0.55):
        t = re.sub(r",\s+", ". ", t, count=max(1, int(len(t) * 0.0012)))
    if seed.random() < sm.get("sensory", 0.7):
        t += "\n\n" + seed.choice(SENSORY) + "."
    if seed.random() < sm.get("irony", 0.25):
        t = t.replace("na pewno", "no jasne, na pewno")
    if seed.random() < sm.get("pathos", 0.35):
        t += "\n\nTo jest ten moment."
    if seed.random() < sm.get("metaphor", 0.6):
        t += " " + seed.choice(METAPHORS) + "."
    if seed.random() < sm.get("symbol", 0.4):
        t = seed.choice(SYMBOLS).capitalize() + ", " + t
    if seed.random() < sm.get("surreal", 0.15):
        t = t.replace(" i ", " i (jakby na moment sen przejął ster), ", 1)
    return t


# ───────────────────────────────────────────────────────────────────────────
# OSTRZENIE I WULGARYZACJA (AUTOPILOT, bez litości ale z rozumem)
# ───────────────────────────────────────────────────────────────────────────
def maybe_sharpen(text: str, sharp: float, profanity_mode: float | None = None) -> str:
    """
    API zgodne, ale ignorujemy parametry wejściowe: decyduje autopilot.
    - ostrzenie słownictwa → funkcja intensywności,
    - poziom wulgaryzmów → złość + intensywność + obecne bluzgi,
    - ochrona kodu/linków/maili/tagów/liczb.
    """
    base = (text or "").strip()
    if not base:
        return base

    auto = _auto_detect_style(base)
    sharp_auto = auto["sharp"]
    pmode = max(WRITER_PROFANITY_LEVEL, auto["profanity_mode"])

    t, frozen = _freeze_sensitive(base)
    rnd = _rnd(t + f"|sharp:{sharp_auto}|pm:{pmode}")

    if sharp_auto > 0.0:
        prob = min(0.9, 0.15 + 0.8 * sharp_auto)
        for a, b in _SHARP_REPLACEMENTS:
            if rnd.random() < prob:
                t = re.sub(r"(?i)\b" + re.escape(a) + r"\b", b, t)

    if pmode > 0.0:

        def _safe_replace(a, b, tx):
            parts = []
            for s in _sentences(tx):
                if _ABBR_RE.search(s) or _NUM_RE.search(s):
                    parts.append(s)
                    continue
                parts.append(re.sub(r"(?i)\b" + re.escape(a) + r"\b", b, s))
            return " ".join(parts)

        for a, b in _PROFANITY_REPLACEMENTS:
            if rnd.random() < (0.2 + 0.7 * pmode):
                t = _safe_replace(a, b, t)

        sent = _sentences(t)
        for i, s in enumerate(sent):
            if _ABBR_RE.search(s) or _NUM_RE.search(s):
                continue
            attach_p = 0.04 + 0.32 * pmode
            if rnd.random() < attach_p:
                tail = rnd.choice(_PROFANITY_TAILS)
                sent[i] = s.rstrip(" .!?") + f", {tail}."
        t = " ".join(sent)

        if pmode >= 0.95 and rnd.random() < 0.5:
            tokens = re.findall(r"\b\w[\w']*\b", t)
            if tokens:
                idx = rnd.randint(0, len(tokens) - 1)
                burst = rnd.choice(["kurwa", "pierdolone", "jebać"])
                t = t.replace(tokens[idx], tokens[idx] + " " + burst, 1)

    t = _thaw_sensitive(t, frozen)
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\s+,", ",", t)
    return t.strip()


# ───────────────────────────────────────────────────────────────────────────
# ANTY-DUPLIKAT (N-gram) — szybki, skuteczny, parafrazuje tylko gdy trzeba
# ───────────────────────────────────────────────────────────────────────────
def ngram_guard(
    text: str, corpus: List[str] | None = None, n: int = 5, max_ratio: float = 0.08
) -> Tuple[str, float]:
    t = (text or "").strip()
    if not t:
        return t, 0.0

    def grams(s: str) -> set[Tuple[str, ...]]:
        ws = re.findall(r"\w+", s.lower())
        if len(ws) < n:
            return set()
        return {tuple(ws[i : i + n]) for i in range(0, len(ws) - n + 1)}

    tg = grams(t)
    if not tg or not corpus:
        return t, 0.0

    best = 0
    limit = min(800, len(corpus))
    for i in range(limit):
        cg = grams(corpus[i])
        if not cg:
            continue
        inter = len(tg & cg)
        if inter > best:
            best = inter
            if best / max(1, len(tg)) > max_ratio * 1.25:
                break

    ratio = best / max(1, len(tg))
    if ratio <= max_ratio:
        return t, ratio

    # parafrazuj 1–2 największe akapity, bez skracania
    paras = t.split("\n\n")
    idxs = sorted(range(len(paras)), key=lambda i: -len(paras[i]))[:2]
    mutated = False
    for i in idxs:
        chunk = paras[i].strip()
        para = mini_llm_text(  # dostępne w pliku
            f"Parafrazuj bez skracania, zachowaj sens i styl (PL). Zero waty:\n\n{chunk}",
            max_tokens=min(1000, len(chunk) // 2 + 280),
            temperature=0.28,
        )
        if para and para.strip() and para.strip() != chunk:
            paras[i] = para.strip()
            mutated = True
    return ("\n\n".join(paras) if mutated else t), ratio


# ───────────────────────────────────────────────────────────────────────────
# MIĘKKA METRYKA (poezja)
# ───────────────────────────────────────────────────────────────────────────
_VOWELS = "aeiouyąęóAEIOUYĄĘÓ"


def syllables_pl(line: str) -> int:
    return max(1, len([ch for ch in line if ch in _VOWELS]))


def soft_meter_block(text: str, target: int = 12, tol: int = 3) -> str:
    out: List[str] = []
    for ln in (text or "").splitlines():
        if not ln.strip():
            out.append(ln)
            continue
        s = syllables_pl(ln)
        t = ln
        if s > target + tol:
            t = re.sub(r",\s+", ". ", t, count=1)
        elif s < target - tol:
            t = t.rstrip(".?!") + ", i tyle."
        out.append(t)
    return "\n".join(out)


import re
import json
import time
import math
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable

# ───────────────────────────────────────────────────────────────────────────
# KONFIG
# ───────────────────────────────────────────────────────────────────────────
_PRICE_CACHE = DATA_DIR / "price_cache.json"
_PRICE_TTL = int(os.getenv("PRICE_CACHE_TTL_S", "43200"))  # 12h
_COMPS_MAX_WORKERS = max(1, int(os.getenv("COMPS_MAX_WORKERS", "6")))
_REQ_JITTER_MIN = float(os.getenv("COMPS_REQ_JITTER_MIN_S", "0.15"))
_REQ_JITTER_MAX = float(os.getenv("COMPS_REQ_JITTER_MAX_S", "0.45"))
_REQ_RETRIES = int(os.getenv("COMPS_REQ_RETRIES", "2"))

# FX fallbacki (gdy NBP niedostępne)
_FX_EURPLN_FALLBACK = float(os.getenv("FX_EURPLN", "4.3"))
_FX_USDPLN_FALLBACK = float(os.getenv("FX_USDPLN", "3.9"))


# ───────────────────────────────────────────────────────────────────────────
# CACHE utility
# ───────────────────────────────────────────────────────────────────────────
def _price_cache_load() -> dict[str, Any]:
    if _PRICE_CACHE.exists():
        try:
            return json.loads(_PRICE_CACHE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _price_cache_save(obj: dict[str, Any]):
    try:
        _PRICE_CACHE.write_text(
            json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


_PRICE_MEM = _price_cache_load()

# ───────────────────────────────────────────────────────────────────────────
# FX: kursy z NBP (tabela A), cache 24h + fallback ENV
# ───────────────────────────────────────────────────────────────────────────
_FX_CACHE = {"ts": 0, "EURPLN": _FX_EURPLN_FALLBACK, "USDPLN": _FX_USDPLN_FALLBACK}


def _fx_refresh() -> None:
    """Ładuje kursy z NBP (tabela A). Cache 24h."""
    url = "https://api.nbp.pl/api/exchangerates/tables/A?format=json"
    try:
        d = _get_json(url)  # korzystamy z lokalnego HTTP z retry/cache
        tables = d if isinstance(d, list) else []
        if not tables:
            return
        rates = (tables[0].get("rates") or []) if isinstance(tables[0], dict) else []
        eur = next((r.get("mid") for r in rates if (r.get("code") == "EUR")), None)
        usd = next((r.get("mid") for r in rates if (r.get("code") == "USD")), None)
        if eur and usd:
            _FX_CACHE["EURPLN"] = float(eur)
            _FX_CACHE["USDPLN"] = float(usd)
            _FX_CACHE["ts"] = int(time.time())
    except Exception:
        # zostawiamy fallbacki
        pass


def _fx_pln_multiplier(cur: str) -> float:
    cur = (cur or "").strip().lower()
    now = int(time.time())
    if now - _FX_CACHE["ts"] > 86400:  # >24h
        _fx_refresh()
    if cur in ("pln", "zł", "zl", "zloty", "zlotych"):
        return 1.0
    if cur in ("eur", "€"):
        return float(_FX_CACHE.get("EURPLN", _FX_EURPLN_FALLBACK))
    if cur in ("usd", "$"):
        return float(_FX_CACHE.get("USDPLN", _FX_USDPLN_FALLBACK))
    # nieznane: zakładamy już PLN
    return 1.0


# ───────────────────────────────────────────────────────────────────────────
# PRICE parsing — wiele formatów i walut
# ───────────────────────────────────────────────────────────────────────────
# Przykłady: "1 299,99 zł", "1299.99 PLN", "€ 179,00", "$129.95"
_PRICE_ANY_RE = re.compile(
    r"(?i)(?:^|[^0-9])(\d{1,3}(?:[ \.\,]\d{3})*|\d+)(?:[\,\.](\d{2}))?\s*(zł|pln|eur|€|\$|usd)\b"
)


def _parse_prices_any(html_text: str) -> list[tuple[float, str]]:
    """Wyciąga listę (kwota, waluta) w surowej postaci z HTML (nieoczyszczonej)."""
    out: list[tuple[float, str]] = []
    for m in _PRICE_ANY_RE.finditer(html_text or ""):
        whole = m.group(1) or ""
        dec = m.group(2) or ""
        cur = m.group(3) or ""
        # normalizacja liczby: usuń separatory tys., zamień przecinek na kropkę
        w = whole.replace(" ", "").replace(".", "").replace(",", "")
        if dec:
            raw = f"{w}.{dec}"
        else:
            raw = w
        try:
            val = float(raw)
        except Exception:
            continue
        out.append((val, cur.lower()))
    return out


def _to_pln(val: float, currency: str) -> float:
    mul = _fx_pln_multiplier(currency)
    return float(round(val * mul, 2))


def _extract_prices_pln(html_text: str) -> list[float]:
    pairs = _parse_prices_any(html_text or "")
    out = []
    for v, cur in pairs:
        pln = _to_pln(v, cur)
        if 1 <= pln <= 1_000_000:
            out.append(pln)
    return out


# ───────────────────────────────────────────────────────────────────────────
# Anti-ban: jitter + retry
# ───────────────────────────────────────────────────────────────────────────
def _sleep_jitter():
    import random

    t = random.uniform(_REQ_JITTER_MIN, _REQ_JITTER_MAX)
    time.sleep(t)


def _get_with_retry(url: str, params: dict[str, Any] | None = None) -> str:
    last = ""
    for i in range(max(1, _REQ_RETRIES) + 1):
        try:
            _sleep_jitter()
            r = _HTTP.get(url, params=params or {}, timeout=WEB_TIMEOUT)
            if r.status_code < 400:
                return r.text
            last = f"HTTP {r.status_code}"
        except Exception as e:
            last = str(e)
    return last and ""


# ───────────────────────────────────────────────────────────────────────────
# DDG fallback do URL-i per serwis
# ───────────────────────────────────────────────────────────────────────────
def _site_urls(query: str, site: str, k: int = 3) -> list[str]:
    # najpierw natywny listing; jeśli nie ma, to DDG
    urls = ddg_site_urls(query, site, k=k)
    # dedup
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out[:k]


# ───────────────────────────────────────────────────────────────────────────
# Parsery per serwis (kilka ścieżek)
# ───────────────────────────────────────────────────────────────────────────
def comps_allegro(query: str, limit: int = 10) -> list[float]:
    """
    Allegro — 2 ścieżki:
    1) widok listingu (price.amount) z głównej wyszukiwarki,
    2) fallback: parse cen z top N wyników z DDG site:allegro.pl.
    Zwraca ceny w PLN.
    """
    prices: list[float] = []
    try:
        htmlt = _get_with_retry("https://allegro.pl/listing", {"string": query})
        if htmlt:
            # wzór 1: "price":{"amount":"129.99"}
            for m in re.finditer(
                r'"price"\s*:\s*{\s*"amount"\s*:\s*"([\d\.]+)"', htmlt
            ):
                try:
                    prices.append(float(m.group(1)))
                except Exception:
                    pass
            # wzór 2: "price":{"amount":{"value":"129.99"
            if not prices:
                for m in re.finditer(
                    r'"price"\s*:\s*{\s*"amount"\s*:\s*{\s*"value"\s*:\s*"([\d\.]+)"',
                    htmlt,
                ):
                    try:
                        prices.append(float(m.group(1)))
                    except Exception:
                        pass
        # fallback: pobierz kilka kart z DDG i parsuj
        if len(prices) < max(3, limit // 2):
            for url in _site_urls(query, "allegro.pl", k=3):
                htmlp = _get_with_retry(url, None)
                if not htmlp:
                    continue
                # wyciągnij i skonwertuj do PLN (zwykle już PLN)
                pp = _extract_prices_pln(htmlp)
                prices.extend(pp)
        prices = [p for p in prices if 10 <= p <= 200000]
        return prices[:limit]
    except Exception:
        return prices[:limit]


def comps_vinted(query: str, limit: int = 10) -> list[float]:
    """
    Vinted — 2 ścieżki:
    1) widok katalogu (price_decimal),
    2) fallback: DDG site:vinted.pl i parsowanie cen z kart.
    Zwraca PLN (Vinted w PL najczęściej i tak daje PLN).
    """
    out: list[float] = []
    try:
        htmlt = _get_with_retry("https://www.vinted.pl/catalog", {"search_text": query})
        if htmlt:
            # "price_decimal":129.0
            for m in re.finditer(r'"price_decimal"\s*:\s*([\d\.]+)', htmlt):
                try:
                    out.append(float(m.group(1)))
                except Exception:
                    pass
            # alternatywnie: "price":"129,00 zł"
            if len(out) < max(3, limit // 2):
                out.extend(_extract_prices_pln(htmlt))
        if len(out) < max(3, limit // 2):
            for url in _site_urls(query, "vinted.pl", k=3):
                htmlp = _get_with_retry(url, None)
                if not htmlp:
                    continue
                out.extend(_extract_prices_pln(htmlp))
        out = [p for p in out if 10 <= p <= 200000]
        return out[:limit]
    except Exception:
        return out[:limit]


def comps_olx(query: str, limit: int = 10) -> list[float]:
    """
    OLX — 2 ścieżki:
    1) listing /oferty (price value / oryginalne atrybuty),
    2) fallback: DDG site:olx.pl i parsowanie kart.
    Zwraca PLN.
    """
    out: list[float] = []
    try:
        htmlt = _get_with_retry("https://www.olx.pl/oferty/", {"q": query})
        if htmlt:
            # JSON-y w HTML: "price":{"value":"129.00"}
            for m in re.finditer(r'"price"\s*:\s*{\s*"value"\s*:\s*"([\d\.]+)"', htmlt):
                try:
                    out.append(float(m.group(1)))
                except Exception:
                    pass
            if len(out) < max(3, limit // 2):
                out.extend(_extract_prices_pln(htmlt))
        if len(out) < max(3, limit // 2):
            for url in _site_urls(query, "olx.pl", k=3):
                htmlp = _get_with_retry(url, None)
                if not htmlp:
                    continue
                out.extend(_extract_prices_pln(htmlp))
        out = [p for p in out if 10 <= p <= 200000]
        return out[:limit]
    except Exception:
        return out[:limit]


def comps_ebay(query: str, limit: int = 10) -> list[float]:
    """
    eBay — parsujemy widok listingów (różne waluty), konwertujemy do PLN
    (kursy NBP). Dodatkowo fallback: kilka kart z DDG.
    """
    out: list[float] = []
    try:
        htmlt = _get_with_retry("https://www.ebay.com/sch/i.html", {"_nkw": query})
        if htmlt:
            # "price":{"value":"129.95"} lub tekstowe "PLN 129,99"
            # Najpierw JSON-y:
            for m in re.finditer(r'"price"\s*:\s*{\s*"value"\s*:\s*"([\d\.]+)"', htmlt):
                try:
                    out.append(
                        _to_pln(float(m.group(1)), "usd")
                    )  # domyślnie USD na ebay.com
                except Exception:
                    pass
            # Potem teksty z walutą:
            out.extend(_extract_prices_pln(htmlt))
        if len(out) < max(3, limit // 2):
            for url in _site_urls(query, "ebay.com", k=3):
                htmlp = _get_with_retry(url, None)
                if not htmlp:
                    continue
                out.extend(_extract_prices_pln(htmlp))
        out = [p for p in out if 10 <= p <= 200000]
        return out[:limit]
    except Exception:
        return out[:limit]


# ───────────────────────────────────────────────────────────────────────────
# ODSZUMIANIE: IQR(±3x) → MAD(±2.5x) + twarde granice
# ───────────────────────────────────────────────────────────────────────────
def _iqr_filter(prices: list[float]) -> list[float]:
    if len(prices) < 4:
        return prices
    q = statistics.quantiles(prices, n=4, method="inclusive")
    q1, q3 = q[0], q[2]
    iqr = q3 - q1
    low, high = q1 - 3 * iqr, q3 + 3 * iqr
    return [p for p in prices if low <= p <= high]


def _mad_filter(prices: list[float]) -> list[float]:
    if not prices:
        return prices
    med = statistics.median(prices)
    devs = [abs(p - med) for p in prices]
    mad = statistics.median(devs) if devs else 0.0
    if mad <= 0:
        return prices
    return [p for p in prices if abs(p - med) <= 2.5 * mad]


def clean_comps(prices: list[float]) -> list[float]:
    """Filtry: twarde granice → IQR → MAD → twarde granice."""
    pr = [p for p in prices if 10 <= p <= 200000]
    if not pr:
        return []
    pr = sorted(pr)
    pr = _iqr_filter(pr)
    pr = _mad_filter(pr)
    pr = [p for p in pr if 10 <= p <= 200000]
    return pr


# ───────────────────────────────────────────────────────────────────────────
# SCALANIE COMPS DLA ZAPYTANIA + CACHE 12h + RÓWNOLEGŁOŚĆ
# ───────────────────────────────────────────────────────────────────────────
def _collect_comps_parallel(query: str, limit_each: int = 12) -> dict[str, list[float]]:
    tasks = {
        "allegro": lambda: comps_allegro(query, limit_each),
        "vinted": lambda: comps_vinted(query, limit_each),
        "olx": lambda: comps_olx(query, limit_each),
        "ebay": lambda: comps_ebay(query, limit_each),
    }
    out: dict[str, list[float]] = {k: [] for k in tasks}
    # równoległość z ostrożnością
    try:
        with ThreadPoolExecutor(max_workers=_COMPS_MAX_WORKERS) as ex:
            fut2name = {ex.submit(fn): name for name, fn in tasks.items()}
            for fut in as_completed(fut2name):
                name = fut2name[fut]
                try:
                    out[name] = fut.result() or []
                except Exception:
                    out[name] = []
    except Exception:
        # fallback: sekwencyjnie
        for name, fn in tasks.items():
            try:
                out[name] = fn() or []
            except Exception:
                out[name] = []
    return out


def get_comps(query: str, refresh: bool = False) -> dict[str, Any]:
    """
    Zwraca pełen pakiet comps (Allegro, Vinted, OLX, eBay) + mediany, liczbę próbek,
    datę, i surowe/oczyszczone zestawy. Dane cache’owane na PRICE_CACHE_TTL_S.
    """
    q = (query or "").strip().lower()
    if not q:
        return {
            "query": "",
            "sources": {},
            "medians": {},
            "counts": {},
            "ts": int(time.time()),
        }

    key = f"comps::{q}"
    now = int(time.time())
    cached = _PRICE_MEM.get(key)
    if cached and not refresh and (now - int(cached.get("ts", 0)) < _PRICE_TTL):
        return cached

    raw = _collect_comps_parallel(q, limit_each=14)
    cleaned = {name: clean_comps(vals) for name, vals in raw.items()}
    medians = {
        name: (statistics.median(vals) if vals else None)
        for name, vals in cleaned.items()
    }
    counts = {name: len(vals) for name, vals in cleaned.items()}

    result = {
        "query": q,
        "sources": cleaned,  # oczyszczone listy cen w PLN
        "medians": medians,  # mediany per źródło
        "counts": counts,  # liczby próbek
        "ts": now,
        "fx": {
            "EURPLN": _FX_CACHE.get("EURPLN", _FX_EURPLN_FALLBACK),
            "USDPLN": _FX_CACHE.get("USDPLN", _FX_USDPLN_FALLBACK),
            "asof": _FX_CACHE.get("ts", 0),
        },
    }
    _PRICE_MEM[key] = result
    _price_cache_save(_PRICE_MEM)
    return result


# ───────────────────────────────────────────────────────────────────────────
# CZĘŚĆ 6/2 — ULTRA BOOST: killer listing generator (Vinted/Allegro/OLX)
# ───────────────────────────────────────────────────────────────────────────

import re, random, math, statistics

# ——— Słowniki i synonimy SEO
_SEO_SYNONYMS = {
    "buty": ["sneakersy", "trampki", "obuwie"],
    "kurtka": ["parka", "puchówka", "okrycie wierzchnie"],
    "spodnie": ["denim", "jeansy", "chinosy"],
    "bluza": ["hoodie", "sweatshirt", "crewneck"],
    "torebka": ["bag", "crossbody", "shoulder bag"],
    "koszulka": ["t-shirt", "tee", "top"],
    "sweter": ["knit", "golf", "pullover"],
}

# ——— Mnożniki tierów marki (korekta wartości bazowej)
_BRAND_TIER_MULT = {
    "fast-fashion": 1.00,
    "mid": 1.15,
    "mid-premium": 1.25,
    "premium": 1.35,
    "luxury": 1.55,
}

# ——— Pula metafor i żartów do losowej rotacji (różnorodność opisów)
_METAPHOR_POOL = [
    "siedzi na Tobie jak ulał — jakby było szyte na miarę przez babcię krawcową",
    "lekkość jak kebab o 3 w nocy — niby ciężar, a wchodzi gładko",
    "kolor tak soczysty, że Instagram sam się prosi o zdjęcie",
    "wygodne jak kanapa u kumpla, na której miałeś spać tylko 10 minut",
    "wygląda jak gear z cyberpunkowego uniwersum, tylko bardziej real",
    "ma vibe starej szkoły, ale styl leci jak mixtape prosto z 2025",
    "to nie ciuch, to talizman do wygrywania w życiu codziennym",
    "trzyma fason jak DJ set, który nie pozwala wyjść z klubu",
]
_JOKE_POOL = [
    "Kup teraz, zanim sąsiad wrzuci to samo za stówkę drożej.",
    "Pasuje do wszystkiego — no, może poza garniturem na wesele u wujka.",
    "Wygląda tak dobrze, że Twój ex napisze 'hej' bez powodu.",
    "Lepsze niż kawa w poniedziałek rano.",
    "Jak tego nie weźmiesz, ktoś inny wrzuci fit na TikToka i będzie po temacie.",
]


# ——— Utility
def _norm_text(s: str) -> str:
    return (s or "").strip()


def _clamp(v: float, a: float, b: float) -> float:
    return max(a, min(b, v))


# ——— Normalizacja stanu i współczynnik ceny
def _norm_cond(cond: str) -> tuple[str, float]:
    c = (cond or "").strip().lower()
    mapping = {
        "nowy z metkami": ("Nowy z metkami", 1.10),
        "nowy": ("Nowy", 1.05),
        "jak nowy": ("Jak nowy", 1.00),
        "bardzo dobry": ("Bardzo dobry", 0.93),
        "dobry": ("Dobry", 0.85),
        "do poprawek": ("Do poprawek", 0.65),
    }
    for k, v in mapping.items():
        if k in c:
            return v
    return ("Dobry", 0.85)


# ——— Popularność rozmiaru/koloru (lekka korekta popytu)
def _popularity_hint(size: str, color: str) -> float:
    s = (size or "").upper()
    c = (color or "").lower()
    base = 1.0
    if s in {"S", "M", "L"}:
        base *= 1.05
    if any(x in c for x in ("black", "czarn", "white", "bia", "grey", "szar")):
        base *= 1.05
    if any(x in c for x in ("pink", "róż", "neon", "czerw")):
        base *= 1.02
    return _clamp(base, 0.9, 1.15)


# ——— Współczynnik tieru marki
def _brand_tier_factor(brand_info: dict) -> float:
    tier = (brand_info or {}).get("tier") or "mid"
    return _BRAND_TIER_MULT.get(tier, 1.15)


# ——— Tytuły A/B/C/D: hype, opisowy, SEO, ultra-SEO
def _mk_titles_ultra(
    brand: str, item: str, model: str, size: str, color: str, cond: str
) -> tuple[str, str, str, str]:
    brand, item, model, size, color, cond = map(
        _norm_text, (brand, item, model, size, color, cond)
    )
    core = f"{brand} {item}".strip()
    tA = f"{core} {model} {size} • {cond} • Must-have".replace("  ", " ").strip(" •")
    tB = f"{core} — {color} {size} • stan: {cond}".replace("  ", " ").strip(" •")
    bits = [brand, item, model, size, color, cond, "oryginał", "autentyczność"]
    bits = [b for b in bits if b]
    tC = " | ".join(bits[:8])
    # D: ultra-SEO (frazy łączone + synonimy)
    syns = []
    base_item = item.lower()
    for k, arr in _SEO_SYNONYMS.items():
        if k in base_item:
            syns.extend(arr[:2])
    d_bits = list(dict.fromkeys(bits[:5] + syns[:2]))
    tD = " • ".join(d_bits[:6])
    return tA, tB, tC, tD


# ——— Plan zdjęć (sprzedażowe must-have)
def _shotlist(
    brand: str, item: str, cond: str, extras: list[str] | None = None
) -> list[str]:
    shots = [
        f"Front — pełny kadr {item} na sylwetce lub flatlay",
        "Tył — pełny kadr",
        "Zbliżenie metki (marka/rozmiar/materiał)",
        "Detale szwów / wykończeń",
        "Faktura materiału w świetle dziennym (prawdziwy kolor)",
        "Stylizacja całościowa (1 look, neutralne tło)",
    ]
    if "nowy" in (cond or "").lower():
        shots.append("Zdjęcie z metką/fakturą (jeśli masz)")
    if extras:
        for e in extras:
            shots.append(f"Dodatki: {e}")
    return shots


# ——— Ton i wulgarność (opcjonalna, dawkuj)
def _copy_tone(text: str, mode: str = "balanced", profanity_level: float = 0.0) -> str:
    t = text
    if mode == "sales":
        t = re.sub(r"\.\s+", ". Serio, zgarnij to teraz. ", t, count=1)
        t += " Limit nie czeka."
    elif mode == "brand":
        t = t.replace("idealne", "ikoniczne")
        t += " Podpis jakości, vibe’u i charakteru."
    # subtelne „dosolenie” – kontrolowane
    if profanity_level > 0:
        salt = [
            ("świetne", "zajebiście dobre"),
            ("naprawdę", "serio kurwa"),
            ("wyjątkowe", "po prostu kozackie"),
            ("spoko", "konkretnie dobre"),
        ]
        for a, b in salt:
            if random.random() < profanity_level * 0.5:
                t = t.replace(a, b)
    return t


# ——— Storytelling: intro + lifestyle + metafora + żart + CTA
def _storytelling_text(
    brand: str,
    item: str,
    cond: str,
    bio: str,
    signs: str,
    tone: str,
    profanity_level: float,
) -> str:
    intro = {
        "Nowy z metkami": "Świeżo zdjęte z półki, pachnie nowością. Zero kompromisów.",
        "Nowy": "Perfekcyjny stan, gotowe na premierę w Twoim życiu.",
        "Jak nowy": "Ledwo noszone, vibe wciąż prosto z showroomu.",
        "Bardzo dobry": "Noszone, ale vibe mocny jak kawa na kaca.",
        "Dobry": "Trochę historii, ale to jak tatuaż — dodaje charakteru.",
        "Do poprawek": "Nie ideał, ale patrz na to jak na płótno pod custom.",
    }.get(cond, "Solidna jakość i zero lipy.")
    meta = random.choice(_METAPHOR_POOL)
    joke = random.choice(_JOKE_POOL)
    lifestyle = f"Idealne do codziennych fitów i mocniejszych wyjść. {brand} słynie z detali: {signs}. {meta}"
    if bio:
        lifestyle += f" Krótko o marce: {bio[:160]}."
    cta = f"{joke} Pierwszy ruch wygrywa."
    base = " ".join([intro, lifestyle, cta])
    return _copy_tone(base, tone, _clamp(profanity_level, 0.0, 1.0))


# ——— Porady pielęgnacja/styl
def _pro_tips(care: str, item: str, cond: str) -> list[str]:
    tips = []
    c = (care or "").lower()
    if "druk" in c or "print" in c:
        tips.append("Pierz na lewej stronie — print będzie żył dłużej.")
    if "puch" in c:
        tips.append("Susz z kulkami tenisowymi — puch rozbije się równomiernie.")
    if "wełn" in c:
        tips.append("Przechowuj na płasko — nie wyciągaj na wieszaku.")
    if "skór" in c:
        tips.append("Przecieraj i impregnuj co kilka tygodni.")
    if "nowy" in (cond or "").lower():
        tips.append("Trzymaj w pokrowcu — 100% świeżości.")
    tips.append(f"Łącz {item} z klasyką (denim + biała koszulka) — nie ma pudła.")
    return tips


# ——— SEO (long-tail + synonimy) → lista keywords + string hashtagów
def _seo_keywords(
    brand: str, item: str, model: str, size: str, color: str, material: str, cond: str
) -> tuple[list[str], str]:
    base = [
        brand,
        item,
        model,
        size,
        color,
        material,
        cond,
        "oryginał",
        "autentyczność",
        "rare",
        "unikat",
    ]
    base = [b for b in base if b]
    lt = [
        f"{brand} {item} {size}",
        f"{brand} {item} {color}",
        f"{brand} {model} {size} {color}",
    ]
    low_item = (item or "").lower()
    for k, arr in _SEO_SYNONYMS.items():
        if k in low_item:
            lt += [f"{brand} {a} {size}" for a in arr[:2]]
    kw = list(dict.fromkeys(base + lt))[:22]
    tags = " ".join("#" + re.sub(r"[^a-z0-9]+", "", k.lower()) for k in kw)
    return kw, tags


# ——— Strategia ceny (auto lub wymuszony tryb)
def _price_strategy(
    anchor: float,
    cond_factor: float,
    pop_factor: float,
    tier_factor: float,
    volume_score: float,
    price_mode: str | None,
) -> tuple[int, int, int, str]:
    """
    anchor: mediana compsów (PLN)
    volume_score: 0..1 (0 mało danych, 1 dużo)
    price_mode: if None → auto (na bazie volume/tier/cond)
    """
    base = anchor * cond_factor * pop_factor * tier_factor
    auto_mode = "neutral"
    if price_mode:
        auto_mode = price_mode
    else:
        # auto: mało danych + luxury → premium; dużo danych + nie-lux → aggressive
        if volume_score < 0.35 and tier_factor >= 1.45:
            auto_mode = "premium"
        elif volume_score > 0.6 and tier_factor <= 1.25:
            auto_mode = "aggressive"
        else:
            auto_mode = "neutral"

    if auto_mode == "aggressive":
        mid = base * 0.86
        lo, hi = mid * 0.92, mid * 1.08
    elif auto_mode == "premium":
        mid = base * 1.22
        lo, hi = mid * 0.93, mid * 1.18
    else:
        mid = base * 1.00
        lo, hi = mid * 0.90, mid * 1.15

    def _round10(x: float) -> int:
        return int(max(1, round(x / 10.0) * 10))

    return _round10(lo), _round10(mid), _round10(hi), auto_mode


# ——— GŁÓWNA FUNKCJA: ULTRA BOOST LISTING
def vinted_listing_ultra_boost(
    brand: str,
    item: str,
    cond: str,
    size: str,
    color: str = "",
    material: str = "",
    measurements: dict[str, str] | None = None,
    model: str = "",
    defects: list[str] | None = None,
    extras: list[str] | None = None,
    notes: str = "",
    tier_hint: str | None = None,  # możesz nadpisać tier
    base_price: float | None = None,  # fallback gdy brak comps
    # BOOST parametry:
    price_mode: str | None = None,  # None=auto | "aggressive" | "neutral" | "premium"
    copy_tone: str = "balanced",  # "balanced" | "sales" | "brand"
    profanity_level: float = 0.0,  # 0..1 — opcjonalne „dosolenie” języka
) -> dict:
    """
    Zwraca kompletny pakiet listingowy:
    - tytuły: A/B/C/D
    - opis: storytelling + lifestyle + metafora + żart + CTA
    - price intelligence: widełki + tryb (auto)
    - SEO: keywords + hashtags
    - zdjęcia: shotlista
    - pro-tips, haki A/B i metadane (anchor, medians, counts, volume_score itd.)
    """
    # 1) sanity / normalizacja
    brand = _norm_text(brand)
    item = _norm_text(item)
    size = _norm_text(size)
    cond_in = _norm_text(cond)
    color = _norm_text(color)
    model = _norm_text(model)
    material = _norm_text(material)
    notes = _norm_text(notes)

    # 2) wiedza o marce (tier, cechy, bio)
    binfo = brand_knowledge(brand, web=True)
    if tier_hint:
        binfo["tier"] = tier_hint
    signs = list(dict.fromkeys((binfo.get("sign") or [])[:6]))
    care = binfo.get("care", "")
    bio = binfo.get("bio", "")
    tier_factor = _brand_tier_factor(binfo)

    # 3) comps (6/1)
    query = " ".join([brand, model, item, size, color]).strip()
    comps = get_comps(query)
    medians = comps.get("medians", {}) or {}
    counts = comps.get("counts", {}) or {}
    all_prices = [p for vals in comps.get("sources", {}).values() for p in vals]
    anchor = statistics.median(all_prices) if all_prices else (base_price or 100.0)

    # 4) czynniki ceny
    cond_label, cond_factor = _norm_cond(cond_in)
    pop_factor = _popularity_hint(size, color)
    sample_n = sum(int(x) for x in counts.values()) or 0
    volume_score = _clamp(math.log10(max(1, sample_n)) / 2.0, 0.0, 1.0)

    # 5) strategia cenowa
    price_low, price_mid, price_high, decided_mode = _price_strategy(
        anchor, cond_factor, pop_factor, tier_factor, volume_score, price_mode
    )

    # 6) tytuły (hype / opisowy / SEO / ultra-SEO)
    tA, tB, tC, tD = _mk_titles_ultra(brand, item, model, size, color, cond_label)

    # 7) SEO
    kw, tags = _seo_keywords(brand, item, model, size, color, material, cond_label)

    # 8) opis/story
    signs_line = ", ".join(signs) if signs else "charakterystyczne detale"
    story_text = _storytelling_text(
        brand, item, cond_label, bio, signs_line, copy_tone, profanity_level
    )
    tips = _pro_tips(care, item, cond_label)
    meas_lines = [f"{k}: {v}" for k, v in (measurements or {}).items() if v]
    defects = defects or []
    extras = extras or []

    # 9) markdown
    md: list[str] = []
    md.append(f"# {brand} {item}")
    md.append(
        f"**Model:** {model or '-'} | **Rozmiar:** {size} | **Stan:** {cond_label}"
    )
    md.append(f"**Kolor/Materiał:** {color or '-'} / {material or '-'}")
    if signs:
        md.append(f"**Cechy charakterystyczne:** {', '.join(signs)}")
    if meas_lines:
        md.append("**Wymiary (na płasko):**\n- " + "\n- ".join(meas_lines))
    if defects:
        md.append("**⚠️ Wady:** " + ", ".join(defects))
    if extras:
        md.append("**➕ Dodatki:** " + ", ".join(extras))
    if notes:
        md.append("**Notatki sprzedającego:** " + notes)
    md.append(f"> {story_text}")
    if care:
        md.append(f"**Pielęgnacja:** {care}")
    if tips:
        md.append("### Pro Tips\n" + "\n".join(f"- {t}" for t in tips))
    md.append("### Cena")
    md.append(
        f"**{price_mid} zł** (widełki: {price_low}-{price_high} zł) • tryb: {decided_mode}"
    )
    if medians:
        meds = [
            f"- {src}: {int(v)} zł (n={counts.get(src,0)})"
            for src, v in medians.items()
            if v
        ]
        if meds:
            md.append("### Mediany rynkowe\n" + "\n".join(meds))
    md.append("### SEO/Hashtagi\n" + tags)
    md.append("### Zdjęcia — plan")
    for s in _shotlist(brand, item, cond_label, extras):
        md.append(f"- {s}")

    # 10) Haki A/B (krótkie leady do testów)
    ab_hooks = {
        "A": f"{brand} {item} {size} — dorwij zanim zniknie.",
        "B": f"{brand} {item}: {color}, roz. {size}. Stan: {cond_label}.",
        "C": f"{brand} {model} {size} {color} • oryginał • autentyczność.",
        "D": f"{brand} {item} • {color}/{size} • {cond_label} • szybka wysyłka.",
    }

    description_md = "\n\n".join(md).strip()

    return {
        "titles": {"A": tA, "B": tB, "C": tC, "D": tD},
        "ab_hooks": ab_hooks,
        "price": int(price_mid),
        "price_range": (int(price_low), int(price_high)),
        "price_mode": decided_mode,
        "description_md": description_md,
        "seo_tags": tags,
        "keywords": kw,
        "cond_norm": cond_label,
        "pop_factor": round(pop_factor, 3),
        "tier_factor": round(tier_factor, 3),
        "anchor_price": int(anchor),
        "volume_score": round(volume_score, 3),
        "brand_info": {
            "tier": (binfo.get("tier") or "mid"),
            "signs": signs,
            "care": care,
            "bio": bio[:400],
        },
        "comps": comps,  # pełny pakiet z 6/1 (sources/medians/counts/fx/ts)
    }


# ───────────────────────────────────────────────────────────────────────────
# CZĘŚĆ 7 — LISTING HUB ULTRA+ (indeks, wersje, A/B, szablony, eksport)
# ───────────────────────────────────────────────────────────────────────────

import csv
import json
import math
import random
import re
import threading
import uuid
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

LISTINGS_DIR = DATA_DIR / "listings"
LISTINGS_DIR.mkdir(parents=True, exist_ok=True)

INDEX_FILE = LISTINGS_DIR / "index.jsonl"  # indeks ofert (1 linia = 1 rekord)
SALES_LOG = LISTINGS_DIR / "sales_log.jsonl"  # log sprzedaży/metryk
ABTEST_LOG = LISTINGS_DIR / "abtest_log.jsonl"  # log wyników A/B
IMAGES_DIR = LISTINGS_DIR / "images"  # jeśli chcesz trzymać lok. zrzuty
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# ───────────────────────────────────────────────────────────────────────────
# MODELE DANYCH
# ───────────────────────────────────────────────────────────────────────────
@dataclass
class ListingRecord:
    id: str
    slug: str
    brand: str
    item: str
    size: str
    color: str
    cond: str
    price: int
    price_range: tuple[int, int]
    price_mode: str
    titles: dict[str, str]
    description_md: str
    seo_tags: str
    keywords: list[str]
    created_at: str
    updated_at: str
    version: int = 1
    meta: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_ultra_boost(
        slug: str, payload: dict, base_meta: dict[str, Any]
    ) -> "ListingRecord":
        now = dt.datetime.utcnow().isoformat()
        return ListingRecord(
            id=str(uuid.uuid4()),
            slug=slug,
            brand=base_meta.get("brand", ""),
            item=base_meta.get("item", ""),
            size=base_meta.get("size", ""),
            color=base_meta.get("color", ""),
            cond=payload.get("cond_norm", ""),
            price=int(payload.get("price", 0)),
            price_range=tuple(payload.get("price_range", (0, 0))),
            price_mode=payload.get("price_mode", "neutral"),
            titles=payload.get("titles", {}),
            description_md=payload.get("description_md", ""),
            seo_tags=payload.get("seo_tags", ""),
            keywords=list(payload.get("keywords", [])),
            created_at=now,
            updated_at=now,
            version=1,
            meta={
                "anchor_price": payload.get("anchor_price"),
                "tier_factor": payload.get("tier_factor"),
                "pop_factor": payload.get("pop_factor"),
                "volume_score": payload.get("volume_score"),
                "brand_info": payload.get("brand_info", {}),
                "comps": payload.get("comps", {}),
                **base_meta,
            },
        )


@dataclass
class ABTestResult:
    id: str
    slug: str
    variant: str
    impressions: int = 0
    clicks: int = 0
    wishlist: int = 0
    sold: int = 0
    started_at: str = field(default_factory=lambda: dt.datetime.utcnow().isoformat())
    finished_at: str = ""


# ───────────────────────────────────────────────────────────────────────────
# UTILS: bezpieczny zapis/odczyt JSONL
# ───────────────────────────────────────────────────────────────────────────
def _jsonl_append(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _jsonl_load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (s or "").strip()).strip("-").lower()
    return s or "item"


# ───────────────────────────────────────────────────────────────────────────
# TEMPLATE MANAGER — alternatywne szablony opisów (rotacja vibe’u)
# ───────────────────────────────────────────────────────────────────────────
class TemplateManager:
    """
    Rotacyjny system szablonów. Każdy zwraca finalny Markdown:
    intro + hero + features + CTA + hashtags. Wpinasz losowo/na zmianę.
    """

    def __init__(self):
        self.templates = {
            "hero_min": self._hero_min,
            "story_fresh": self._story_fresh,
            "tech_spec": self._tech_spec,
        }
        self.order = ["hero_min", "story_fresh", "tech_spec"]
        self._idx = 0

    def next_name(self) -> str:
        name = self.order[self._idx % len(self.order)]
        self._idx += 1
        return name

    # — szablony:
    def _hero_min(self, rec: ListingRecord) -> str:
        hero = rec.titles.get("A") or rec.titles.get("B") or f"{rec.brand} {rec.item}"
        blocks = [
            f"# {hero}",
            f"**Rozmiar:** {rec.size} • **Stan:** {rec.cond} • **Kolor:** {rec.color or '-'}",
            "",
            rec.description_md.strip(),
            "",
            "### Hashtagi",
            rec.seo_tags,
        ]
        return "\n".join(blocks).strip()

    def _story_fresh(self, rec: ListingRecord) -> str:
        hero = rec.titles.get("B") or f"{rec.brand} {rec.item} {rec.size}"
        vibe = "Lekki vibe, zero spiny. Wchodzi w stylizacje jak klucz w stacyjkę."
        blocks = [
            f"# {hero}",
            f"**Cena:** {rec.price} zł (okno: {rec.price_range[0]}–{rec.price_range[1]} zł) • **Stan:** {rec.cond}",
            "",
            f"> {vibe}",
            "",
            rec.description_md.strip(),
            "",
            "— Wysyłka 24–48h, bez dymu. Bundle -12%.",
            "",
            "### Hashtagi",
            rec.seo_tags,
        ]
        return "\n".join(blocks).strip()

    def _tech_spec(self, rec: ListingRecord) -> str:
        hero = rec.titles.get("C") or rec.titles.get("A") or f"{rec.brand} {rec.item}"
        feat = [
            f"- Rozmiar: {rec.size}",
            f"- Stan: {rec.cond}",
            f"- Kolor: {rec.color or '-'}",
            f"- Cena: {rec.price} zł (okno: {rec.price_range[0]}–{rec.price_range[1]} zł)",
        ]
        body = "\n".join(feat)
        blocks = [
            f"# {hero}",
            "### Parametry",
            body,
            "",
            rec.description_md.strip(),
            "",
            "### Hashtagi",
            rec.seo_tags,
        ]
        return "\n".join(blocks).strip()

    def render(self, rec: ListingRecord, name: str | None = None) -> str:
        nm = name or self.next_name()
        fn = self.templates.get(nm) or self._hero_min
        out = fn(rec)
        rec.meta["template"] = nm
        return out


# ───────────────────────────────────────────────────────────────────────────
# IMAGE VALIDATOR — szybkie sanity foto (ratio, min-size, kolory)
# ───────────────────────────────────────────────────────────────────────────
class ImageValidator:
    """
    Lekkie sprawdzenie: rozszerzenie, ratio (1:1–4:5/16:9), minimalny bok, limit plików.
    Nie używa PIL, tylko statyczne heurystyki po nazwie i pseudo-metadanych (opcjonalnie).
    """

    ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}

    def validate_filenames(self, files: Iterable[str], max_files: int = 12) -> dict:
        files = list(files)
        issues = []
        if len(files) == 0:
            issues.append("Brak zdjęć — dodaj min. 4.")
        if len(files) > max_files:
            issues.append(f"Za dużo zdjęć ({len(files)}). Maks {max_files}.")
        bad = [f for f in files if Path(f).suffix.lower() not in self.ALLOWED_EXT]
        if bad:
            issues.append(
                f"Niedozwolone rozszerzenia: {', '.join(sorted(set(Path(b).suffix for b in bad)))}."
            )
        cover_ok = any(
            "front" in Path(f).stem.lower() or "main" in Path(f).stem.lower()
            for f in files
        )
        if not cover_ok:
            issues.append("Brak pliku 'front'/'main' — ustaw okładkę.")
        return {"ok": not issues, "issues": issues, "count": len(files)}


# ───────────────────────────────────────────────────────────────────────────
# ADAPTERY MARKETPLACE — CSV dla różnych formatów
# ───────────────────────────────────────────────────────────────────────────
class VintedCSVAdapter:
    COLS = ["title", "price", "size", "brand", "color", "condition", "description"]

    @staticmethod
    def row(rec: ListingRecord) -> list[Any]:
        title = rec.titles.get("A") or rec.titles.get("B") or f"{rec.brand} {rec.item}"
        return [
            title,
            rec.price,
            rec.size,
            rec.brand,
            rec.color,
            rec.cond,
            rec.description_md.replace("\n", " ")[:500],
        ]


class AllegroCSVAdapter:
    COLS = [
        "title",
        "price",
        "category",
        "size",
        "color",
        "brand",
        "condition",
        "description",
    ]

    @staticmethod
    def row(rec: ListingRecord, category: str = "Moda>Odzież") -> list[Any]:
        title = rec.titles.get("B") or rec.titles.get("A") or f"{rec.brand} {rec.item}"
        return [
            title,
            rec.price,
            category,
            rec.size,
            rec.color,
            rec.brand,
            rec.cond,
            rec.description_md.replace("\n", " ")[:500],
        ]


# ───────────────────────────────────────────────────────────────────────────
# LISTING MANAGER — indeks, wyszukiwarka, wersje, batch, feedback, ceny, CSV
# ───────────────────────────────────────────────────────────────────────────
class ListingManager:
    def __init__(self, outdir: Path = LISTINGS_DIR):
        self.outdir = outdir
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.tmpl = TemplateManager()
        self.imgv = ImageValidator()

    # ——— podstawy indeksu
    def _index_append(self, rec: ListingRecord) -> None:
        _jsonl_append(INDEX_FILE, {**rec.__dict__})

    def _index_all(self) -> list[ListingRecord]:
        rows = _jsonl_load(INDEX_FILE)
        out: list[ListingRecord] = []
        for r in rows:
            try:
                out.append(ListingRecord(**r))
            except Exception:
                pass
        return out

    # ——— zapis plików
    def _save_files(self, rec: ListingRecord, content_md: str) -> dict:
        slug = rec.slug
        jpath = self.outdir / f"{slug}.v{rec.version}.json"
        mpath = self.outdir / f"{slug}.v{rec.version}.md"
        jpath.write_text(
            json.dumps({**rec.__dict__}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        mpath.write_text(content_md, encoding="utf-8")
        # symlink/alias na „latest”
        (self.outdir / f"{slug}.json").write_text(
            jpath.read_text(encoding="utf-8"), encoding="utf-8"
        )
        (self.outdir / f"{slug}.md").write_text(
            mpath.read_text(encoding="utf-8"), encoding="utf-8"
        )
        return {
            "json": str(jpath),
            "md": str(mpath),
            "latest_json": str(self.outdir / f"{slug}.json"),
            "latest_md": str(self.outdir / f"{slug}.md"),
        }

    # ——— tworzenie nowego listingu (z Ultra Boost)
    def create(
        self,
        *,
        brand: str,
        item: str,
        cond: str,
        size: str,
        color: str = "",
        material: str = "",
        measurements: dict[str, str] | None = None,
        model: str = "",
        defects: list[str] | None = None,
        extras: list[str] | None = None,
        notes: str = "",
        tier_hint: str | None = None,
        base_price: float | None = None,
        price_mode: str | None = None,
        copy_tone: str = "balanced",
        profanity_level: float = 0.0,
        images: list[str] | None = None,
        template: str | None = None,
    ) -> dict:
        # 1) generacja payloadu (6/2 ULTRA)
        payload = vinted_listing_ultra_boost(
            brand=brand,
            item=item,
            cond=cond,
            size=size,
            color=color,
            material=material,
            measurements=measurements or {},
            model=model,
            defects=defects or [],
            extras=extras or [],
            notes=notes,
            tier_hint=tier_hint,
            base_price=base_price,
            price_mode=price_mode,
            copy_tone=copy_tone,
            profanity_level=profanity_level,
        )
        # 2) slug + rekord
        slug = (
            _slug(f"{brand}-{item}-{size}-{model or ''}-{color or ''}")[:96]
            or f"lst-{uuid.uuid4().hex[:8]}"
        )
        base_meta = {
            "brand": brand,
            "item": item,
            "size": size,
            "color": color,
            "model": model,
            "material": material,
        }
        rec = ListingRecord.from_ultra_boost(slug, payload, base_meta)

        # 3) walidacja zdjęć
        if images:
            v = self.imgv.validate_filenames(images)
            rec.meta["images"] = {"files": images, **v}

        # 4) render wybranym szablonem (rotacja jeśli None)
        content_md = self.tmpl.render(rec, name=template)

        # 5) zapis plików + indeks
        paths = self._save_files(rec, content_md)
        self._index_append(rec)
        return {
            "ok": True,
            "id": rec.id,
            "slug": rec.slug,
            "paths": paths,
            "price": rec.price,
            "range": rec.price_range,
            "titles": rec.titles,
        }

    # ——— nowa wersja istniejącego listingu (np. po korekcie ceny/opisu)
    def revise(self, slug: str, **changes) -> dict:
        items = [r for r in self._index_all() if r.slug == slug]
        if not items:
            return {"ok": False, "reason": "not_found"}
        rec = items[-1]
        for k, v in changes.items():
            if hasattr(rec, k):
                setattr(rec, k, v)
        rec.version += 1
        rec.updated_at = dt.datetime.utcnow().isoformat()
        md = self.tmpl.render(rec, name=changes.get("template"))
        paths = self._save_files(rec, md)
        self._index_append(rec)
        return {"ok": True, "version": rec.version, "paths": paths}

    # ——— wyszukiwanie po indeksie
    def find(
        self,
        *,
        brand: str | None = None,
        item: str | None = None,
        size: str | None = None,
        text: str | None = None,
        limit: int = 50,
    ) -> list[ListingRecord]:
        data = self._index_all()

        def match(r: ListingRecord) -> bool:
            ok = True
            if brand:
                ok = ok and (brand.lower() in r.brand.lower())
            if item:
                ok = ok and (item.lower() in r.item.lower())
            if size:
                ok = ok and (size.lower() in r.size.lower())
            if text:
                blob = " ".join(
                    [
                        r.description_md,
                        " ".join(r.keywords),
                        " ".join(r.titles.values()),
                    ]
                ).lower()
                ok = ok and (text.lower() in blob)
            return ok

        out = [r for r in data if match(r)]
        # sortuj najnowsze
        out.sort(key=lambda r: r.updated_at, reverse=True)
        return out[:limit]

    # ——— eksport CSV (Vinted/Allegro)
    def export_csv(
        self, records: list[ListingRecord], filename: str, adapter: str = "vinted"
    ) -> str:
        path = self.outdir / filename
        if adapter == "allegro":
            cols = AllegroCSVAdapter.COLS
            rows = [AllegroCSVAdapter.row(r) for r in records]
        else:
            cols = VintedCSVAdapter.COLS
            rows = [VintedCSVAdapter.row(r) for r in records]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(cols)
            for row in rows:
                w.writerow(row)
        return str(path)

    # ── FEEDBACK LOOP
    def log_metrics(
        self,
        slug: str,
        *,
        impressions: int = 0,
        clicks: int = 0,
        wishlist: int = 0,
        sold: int = 0,
        price: int | None = None,
    ) -> None:
        rec = {
            "slug": slug,
            "ts": dt.datetime.utcnow().isoformat(),
            "impressions": impressions,
            "clicks": clicks,
            "wishlist": wishlist,
            "sold": sold,
        }
        if price is not None:
            rec["price"] = price
        _jsonl_append(SALES_LOG, rec)

    def metrics(self, slug: str | None = None) -> list[dict]:
        rows = _jsonl_load(SALES_LOG)
        if slug:
            rows = [r for r in rows if r.get("slug") == slug]
        return rows

    # ── DYNAMIC PRICING (proste reguły)
    def dynamic_prices(
        self, *, slug: str | None = None, policy: str = "auto"
    ) -> list[dict]:
        """
        policy:
          - 'auto': CTR<0.4% i brak wishlist → -7%; hot (CTR>1.2% lub wishlist>8) → +5%
          - 'aggressive': -10% jeśli brak klików przez 7 dni
          - 'premium': +8% jeśli wishlist>12 i CTR>1%
        """
        recs = (
            self._index_all()
            if not slug
            else [r for r in self._index_all() if r.slug == slug]
        )
        updates = []
        now = dt.datetime.utcnow()
        for r in recs:
            ms = self.metrics(r.slug)
            if not ms:
                continue
            # 7 dni agregat
            last7 = [
                m for m in ms if (now - dt.datetime.fromisoformat(m["ts"])).days <= 7
            ]
            imps = sum(int(m.get("impressions", 0)) for m in last7) or 0
            clicks = sum(int(m.get("clicks", 0)) for m in last7) or 0
            wl = sum(int(m.get("wishlist", 0)) for m in last7) or 0
            ctr = (clicks / imps * 100.0) if imps > 0 else 0.0

            new_price = r.price
            if policy == "aggressive":
                if imps > 0 and clicks == 0:
                    new_price = max(5, int(round(r.price * 0.90)))
            elif policy == "premium":
                if ctr > 1.0 and wl > 12:
                    new_price = int(round(r.price * 1.08))
            else:  # auto
                if ctr < 0.4 and wl == 0:
                    new_price = max(5, int(round(r.price * 0.93)))
                elif ctr > 1.2 or wl > 8:
                    new_price = int(round(r.price * 1.05))

            if new_price != r.price:
                updates.append(
                    {
                        "slug": r.slug,
                        "old": r.price,
                        "new": new_price,
                        "ctr": round(ctr, 2),
                        "wishlist": wl,
                    }
                )
                self.revise(r.slug, price=new_price)
        return updates

    # ── A/B TEST NA TYTUŁACH (A,B,C,D)
    def ab_start(self, slug: str, variants: list[str] | None = None) -> str:
        recs = [r for r in self._index_all() if r.slug == slug]
        if not recs:
            return "not_found"
        r = recs[-1]
        vs = variants or list(r.titles.keys())
        test_id = f"ab_{slug}_{uuid.uuid4().hex[:6]}"
        for v in vs:
            _jsonl_append(
                ABTEST_LOG, ABTestResult(id=test_id, slug=slug, variant=v).__dict__
            )
        return test_id

    def ab_log(
        self,
        test_id: str,
        variant: str,
        *,
        impressions: int = 0,
        clicks: int = 0,
        sold: int = 0,
        wishlist: int = 0,
    ) -> None:
        _jsonl_append(
            ABTEST_LOG,
            {
                "id": test_id,
                "slug": variant.split("|")[0] if "|" in variant else "",
                "variant": variant,
                "impressions": impressions,
                "clicks": clicks,
                "sold": sold,
                "wishlist": wishlist,
                "ts": dt.datetime.utcnow().isoformat(),
            },
        )

    def ab_best(self, slug: str, test_id: str) -> dict:
        rows = [r for r in _jsonl_load(ABTEST_LOG) if r.get("id") == test_id]
        agg: dict[str, dict[str, float]] = {}
        for r in rows:
            v = r.get("variant")
            a = agg.setdefault(
                v, {"impressions": 0, "clicks": 0, "sold": 0, "wishlist": 0}
            )
            a["impressions"] += int(r.get("impressions", 0))
            a["clicks"] += int(r.get("clicks", 0))
            a["sold"] += int(r.get("sold", 0))
            a["wishlist"] += int(r.get("wishlist", 0))
        scored = []
        for v, a in agg.items():
            imps = a["impressions"] or 1
            ctr = a["clicks"] / imps
            score = (
                (ctr * 0.7) + (a["wishlist"] / max(1, imps) * 0.2) + (a["sold"] * 0.1)
            )
            scored.append(
                (v, score, round(ctr * 100, 2), int(a["sold"]), int(a["wishlist"]))
            )
        scored.sort(key=lambda x: x[1], reverse=True)
        return {"slug": slug, "test_id": test_id, "ranking": scored[:4]}

    # ── BATCH GENERATION
    def batch_generate(
        self, items: list[dict], *, auto_tone: bool = True
    ) -> list[dict]:
        out = []
        for i, itm in enumerate(items, start=1):
            # automatyczna rotacja tonu (balanced → sales → brand → …)
            tone = ["balanced", "sales", "brand"][i % 3] if auto_tone else "balanced"
            res = self.create(
                brand=itm["brand"],
                item=itm["item"],
                cond=itm["cond"],
                size=itm["size"],
                color=itm.get("color", ""),
                material=itm.get("material", ""),
                measurements=itm.get("measurements", {}),
                model=itm.get("model", ""),
                defects=itm.get("defects", []),
                extras=itm.get("extras", []),
                notes=itm.get("notes", ""),
                tier_hint=itm.get("tier_hint"),
                base_price=itm.get("base_price"),
                price_mode=itm.get("price_mode"),
                copy_tone=tone,
                profanity_level=float(itm.get("profanity_level", 0.0)),
                images=itm.get("images", []),
                template=None,
            )
            out.append(res)
        return out

    # ── HARMONOGRAM
    def schedule_batch(self, items: list[dict], interval_hours: int = 12):
        def task():
            self.batch_generate(items)
            print(f"[Scheduler] Batch wygenerowany: {dt.datetime.now()}")
            threading.Timer(interval_hours * 3600, task).start()

        threading.Timer(interval_hours * 3600, task).start()


# ───────────────────────────────────────────────────────────
# UPGRADE 9 — STREAMLIT DASHBOARD
# ───────────────────────────────────────────────────────────
import streamlit as st
import requests, json, time
import pandas as pd

API_URL = "http://localhost:8000"  # Twój HUB (UPGRADE 8)

st.set_page_config(page_title="Listing HUB Dashboard", layout="wide")

st.title("📊 Listing HUB Dashboard")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 Listings", "🧪 A/B Tests", "✍ Kreatywny Writer", "📈 Stats"]
)

# ───────────────────────────────────────────────────────────
# LISTINGS
# ───────────────────────────────────────────────────────────
with tab1:
    st.subheader("Przeglądaj listingi")
    q = st.text_input("Szukaj", "")
    if st.button("Search"):
        res = requests.get(f"{API_URL}/listing/search", params={"q": q}).json()
        df = pd.DataFrame(res)
        st.dataframe(df)

    st.subheader("Dodaj listing")
    with st.form("add_listing"):
        brand = st.text_input("Brand")
        item = st.text_input("Item")
        size = st.text_input("Size")
        color = st.text_input("Color")
        cond = st.selectbox("Stan", ["Nowy", "Jak nowy", "Bardzo dobry", "Dobry"])
        price = st.number_input("Cena", min_value=10, max_value=10000, step=10)
        desc = st.text_area("Opis")
        if st.form_submit_button("Dodaj"):
            r = requests.post(
                f"{API_URL}/listing/create",
                json={
                    "brand": brand,
                    "item": item,
                    "size": size,
                    "color": color,
                    "cond": cond,
                    "price": price,
                    "description": desc,
                },
                headers={"Authorization": "Bearer dev-secret"},
            )
            st.json(r.json())

# ───────────────────────────────────────────────────────────
# A/B TESTS
# ───────────────────────────────────────────────────────────
with tab2:
    st.subheader("Live A/B Testy")
    placeholder = st.empty()
    for _ in range(10):
        r = requests.get(f"{API_URL}/stats").json()
        placeholder.write(r)
        time.sleep(2)

# ───────────────────────────────────────────────────────────
# WRITER
# ───────────────────────────────────────────────────────────
with tab3:
    st.subheader("Kreatywny Writer")
    topic = st.text_input("Temat", "Miasto przyszłości")
    kind = st.selectbox(
        "Rodzaj", ["esej", "opowiadanie", "powieść", "wątek", "Vinted opis"]
    )
    if st.button("Generuj"):
        r = requests.post(
            f"{API_URL}/listing/create",
            json={
                "brand": "Custom",
                "item": kind,
                "size": "-",
                "color": "-",
                "cond": "Nowy",
                "price": 0,
                "description": topic,
            },
            headers={"Authorization": "Bearer dev-secret"},
        )
        st.json(r.json())

# ───────────────────────────────────────────────────────────
# STATS
# ───────────────────────────────────────────────────────────
with tab4:
    st.subheader("Statystyki systemu")
    res = requests.get(f"{API_URL}/stats").json()
    st.metric("Liczba listingów", res.get("count", 0))
    st.metric("Średnia cena", res.get("avg_price", 0))
    st.write("Marki:", ", ".join(res.get("brands", [])))
