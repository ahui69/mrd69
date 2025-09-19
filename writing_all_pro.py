
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
OUT_DIR = Path(os.getenv("WRITER_OUT_DIR", str(ROOT / "out" / "writing")))
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
    s = unicodedata.normalize("NFKD", (s or "")).encode("ascii", "ignore").decode("ascii")
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
        f"<!doctype html><html><head>{head}{css}</head>" f"<body>{_md2html(body_md)}</body></html>"
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

WEB_TIMEOUT = int(os.getenv("WEB_HTTP_TIMEOUT", os.getenv("TIMEOUT_HTTP", "20")))
UA = os.getenv("WEB_USER_AGENT", "MordzixBot/1.0 (writing_all_pro)")


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
        CACHE_FILE.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
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
    base = (os.getenv("LLM_BASE_URL") or "https://api.deepinfra.com/v1/openai").rstrip("/")
    key = (os.getenv("LLM_API_KEY") or "").strip()
    model = (os.getenv("LLM_MODEL") or "meta-llama/Meta-Llama-3.1-70B-Instruct").strip()
    if not key:
        return "\n".join([m.get("content", "") for m in messages if m.get("role") == "user"])
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
        return ((j.get("choices") or [{}])[0].get("message") or {}).get("content", "").strip()
    except Exception:
        return "\n".join([m.get("content", "") for m in messages if m.get("role") == "user"])


def mini_llm_text(prompt: str, max_tokens: int = 256, temperature: float = 0.3) -> str:
    prompt = (prompt or "").strip()
    if not prompt:
        return ""

    # Używamy modelu Qwen 4B
    mini_base = (os.getenv("MINI_LLM_BASE_URL") or os.getenv("LLM_BASE_URL") or "").rstrip("/")
    mini_key = os.getenv("MINI_LLM_API_KEY") or os.getenv("LLM_API_KEY") or ""
    mini_model = os.getenv("MINI_LLM_MODEL", "Qwen/Qwen2.5-4B-Instruct")
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
                    ((jj.get("choices") or [{}])[0].get("message") or {}).get("content", "").strip()
                )
                if txt:
                    return txt
        except Exception:
            pass

    # W przypadku braku możliwości użycia Qwen, spróbujemy użyć Kimi
    try:
        import kimi_client

        return kimi_client.kimi_chat(
            [{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=temperature
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
        if mode in ("mix", "long") and len(s) < MIN_LONG_LEN and random.random() < 0.4 + chaos * 0.3:
            s = s.rstrip(".?!") + ", prawda?"
        if random.random() < chaos * 0.2:
            s = s.replace(" i ", ", i ")
        out.append(s)
    if mode in ("mix", "short") and len(out) > 2 and random.random() < SHORT_INSERT_PROB:
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


def debloat(md: str) -> str:
    t = re.sub(r"[ \t]+\n", "\n", md)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"\b(bardzo|mega|super|naprawdę)\b", "", t, flags=re.I)
    return t.strip()


def style_mix(base: dict[str, float] | None, **overrides: float) -> dict[str, float]:
    v = {
        "pace": 0.5,
        "sensory": 0.75,
        "irony": 0.25,
        "slang": 0.2,
        "pathos": 0.35,
        "sharp": 0.0,
        "surreal": 0.15,
    }
    if base:
        v.update({k: float(max(0, min(1, base.get(k, v[k])))) for k in v})
    for k, val in overrides.items():
        if k in v:
            v[k] = float(max(0, min(1, val)))
    return v


def apply_style_matrix(text: str, sm: dict[str, float]) -> str:
    t = text
    if sm["pace"] >= 0.65:
        t = re.sub(r",\s+", ". ", t, count=max(1, int(len(t) * 0.001)))
    elif sm["pace"] <= 0.35:
        t = re.sub(r"\.\s+", ", ", t, count=max(1, int(len(t) * 0.001)))
    if sm["sensory"] > 0.6:
        t += (
            "\n\n"
            + random.choice(["szmer neonów", "zapach mokrego asfaltu", "ciepło światła na skórze"])
            + "."
        )
    if sm["irony"] > 0.6:
        t = t.replace("na pewno", "no jasne, na pewno")
    if sm["slang"] > 0.6:
        t = re.sub(r"\.", ". Serio.", t, count=1)
    if sm["pathos"] > 0.7:
        t = t + "\n\nTo jest ten moment."
    if sm["surreal"] > 0.6:
        t = t.replace(" i ", " i (jakby na moment sen przejął ster), ", 1)
    return t


def maybe_sharpen(text: str, sharp: float) -> str:
    if sharp <= 0.6:
        return text
    rep = [("kurczę", "kurwa"), ("do bani", "do dupy"), ("niezły", "zajebisty")]
    t = text
    for a, b in rep:
        if random.random() < (sharp - 0.6):
            t = t.replace(a, b)
    return t


def ngram_guard(
    text: str, corpus: list[str] | None = None, n: int = 5, max_ratio: float = 0.08
) -> tuple[str, float]:
    if not text.strip():
        return text, 0.0

    def grams(s: str) -> set[tuple[str, ...]]:
        ws = re.findall(r"\w+", s.lower())
        return set(tuple(ws[i : i + n]) for i in range(0, max(0, len(ws) - n + 1)))

    tg = grams(text)
    best = 0
    if corpus:
        for c in corpus[:400]:
            best = max(best, len(tg & grams(c)))
    ratio = best / max(1, len(tg))
    if ratio <= max_ratio:
        return text, ratio
    paras = text.split("\n\n")
    worst_idx = sorted(range(len(paras)), key=lambda i: -len(paras[i]))[:2]
    mutated = False
    originals = {}
    for i in worst_idx:
        chunk = paras[i].strip()
        originals[i] = chunk
        para = (
            mini_llm_text(
                f"Parafrazuj bez skracania, utrzymaj sens i styl, bez powtórzeń:\n\n{chunk}",
                max_tokens=min(800, len(chunk) // 2 + 200),
                temperature=0.3,
            )
            or chunk
        )
        if para != chunk:
            mutated = True
            paras[i] = para
    out = "\n\n".join(paras)
    if mutated:
        _mem_add(
            json.dumps({"originals": originals, "paraphrased": out[:4000]}, ensure_ascii=False),
            tags=["writing", "guard", "paraphrase"],
            user="global",
        )
    return out, ratio


_VOWELS = "aeiouyąęóAEIOUYĄĘÓ"


def syllables_pl(line: str) -> int:
    return max(1, len([ch for ch in line if ch in _VOWELS]))


def soft_meter_block(text: str, target: int = 12, tol: int = 3) -> str:
    out = []
    for ln in text.splitlines():
        if not ln.strip():
            out.append(ln)
            continue
        s = syllables_pl(ln)
        if s > target + tol:
            ln = re.sub(r",\s+", ". ", ln, count=1)
        elif s < target - tol:
            ln = ln + ", i tyle."
        out.append(ln)
    return "\n".join(out)


# ───────────────────────────────────────────────────────────────────────────
# RESEARCH (Wiki, DDG, Books, OpenAlex, HN) + press via DDG
# ───────────────────────────────────────────────────────────────────────────
def wiki_summary(title: str, lang: str = "pl") -> dict:
    if not title:
        return {}
    try:
        url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ','_')}"
        d = _get_json(url)
        return {
            "title": d.get("title", ""),
            "extract": d.get("extract", ""),
            "url": ((d.get("content_urls") or {}).get("desktop") or {}).get("page", ""),
            "source": f"wikipedia:{lang}",
        }
    except Exception:
        return {}


def ddg_instant(q: str) -> dict:
    try:
        d = _get_json(
            "https://api.duckduckgo.com/",
            params={"q": q, "format": "json", "no_html": "1", "skip_disambig": "1"},
        )
        out = {
            "abstract": d.get("AbstractText") or "",
            "url": d.get("AbstractURL") or "",
            "source": "duckduckgo",
        }
        if not out["abstract"]:
            rt = d.get("RelatedTopics") or []
            if rt and isinstance(rt, list) and isinstance(rt[0], dict):
                out["abstract"] = rt[0].get("Text", "")
                out["url"] = rt[0].get("FirstURL", "")
        return out
    except Exception:
        return {}


_BOOKS_KEY = os.getenv("BOOKS_API_KEY") or os.getenv("GOOGLE_BOOKS_KEY") or ""


def google_books_search(q: str, max_results: int = 5) -> list[dict]:
    if not q:
        return []
    try:
        params = {"q": q, "maxResults": max(1, min(10, max_results))}
        if _BOOKS_KEY:
            params["key"] = _BOOKS_KEY
        d = _get_json("https://www.googleapis.com/books/v1/volumes", params=params)
        out = []
        for it in (d.get("items") or [])[:max_results]:
            info = it.get("volumeInfo") or {}
            out.append(
                {
                    "title": info.get("title", ""),
                    "authors": info.get("authors") or [],
                    "published": info.get("publishedDate", ""),
                    "snippet": (info.get("description") or "")[:320],
                    "link": info.get("infoLink", ""),
                    "source": "google_books",
                }
            )
        return out
    except Exception:
        return []


def hn_search(q: str, max_hits: int = 5) -> list[dict]:
    try:
        d = _get_json("https://hn.algolia.com/api/v1/search", params={"query": q, "tags": "story"})
        out = []
        for h in (d.get("hits") or [])[:max_hits]:
            out.append(
                {
                    "title": h.get("title") or h.get("story_title") or "",
                    "link": h.get("url") or "",
                    "snippet": (h.get("story_text") or "")[:280],
                    "source": "hackernews",
                }
            )
        return out
    except Exception:
        return []


def openalex_search(q: str, max_hits: int = 5) -> list[dict]:
    try:
        d = _get_json("https://api.openalex.org/works", params={"search": q, "per_page": max_hits})
        out = []
        for it in d.get("results") or []:
            title = it.get("title", "")
            url = (
                (it.get("primary_location") or {}).get("source", {}).get("homepage_url")
                or it.get("doi")
                or ""
            )
            ab = it.get("abstract_inverted_index") or {}
            words = []
            for w, idxs in ab.items():
                for _ in idxs:
                    words.append(w)
                    if len(words) >= 70:
                        break
                if len(words) >= 70:
                    break
            snippet = " ".join(words)[:320]
            out.append({"title": title, "link": url, "snippet": snippet, "source": "openalex"})
        return out[:max_hits]
    except Exception:
        return []


def ddg_site_urls(query: str, site: str, k: int = 2) -> list[str]:
    try:
        url = "https://duckduckgo.com/html/"
        r = _HTTP.get(url, params={"q": f"site:{site} {query}"}, timeout=WEB_TIMEOUT)
        if r.status_code >= 400:
            return []
        htmlt = r.text
        links = re.findall(r'href="(https?://[^"]+)"', htmlt)
        outs = []
        for u in links:
            if site in u and "duckduckgo.com" not in u and "ad_provider" not in u:
                outs.append(u.split("&")[0])
                if len(outs) >= k:
                    break
        return outs
    except Exception:
        return []


def fashion_press_enrich(brand: str, k: int = 2) -> dict:
    sources = ["vogue.com", "harpersbazaar.com", "gq.com", "hypebeast.com"]
    urls = []
    for s in sources:
        urls += ddg_site_urls(brand, s, k=1)
    texts = []
    for u in urls[:k]:
        txt = _get_text(u)
        if txt:
            texts.append(txt[:5000])
    bio = ""
    signs = []
    if texts:
        joined = " ".join(texts)
        bio = (
            mini_llm_text(
                (
                    f"Z poniższego tekstu wyciągnij 2-3 zdania bio o marce {brand} (PL):\n\n"
                    f"{joined[:6000]}"
                ),
                200,
                0.25,
            )
            or ""
        )
        signs_txt = (
            mini_llm_text(
                (
                    f"Z tekstu zidentyfikuj 5 charakterystycznych cech/kodów marki {brand} "
                    f"(po polsku, lista):\n\n{joined[:6000]}"
                ),
                180,
                0.25,
            )
            or ""
        )
        signs = [
            re.sub(r"^[\-\*\s]+", "", ln).strip() for ln in signs_txt.splitlines() if ln.strip()
        ][:6]
    return {"press_urls": urls[:k], "press_bio": bio, "press_signs": signs}


def research_evidence(topic: str, lang: str = "pl", k: int = 8) -> list[dict]:
    topic = (topic or "").strip()
    if not topic:
        return []
    ev = []
    wpl = wiki_summary(topic, lang="pl")
    if wpl and wpl.get("extract"):
        ev.append(
            {
                "title": wpl.get("title", ""),
                "link": wpl.get("url", ""),
                "snippet": wpl.get("extract", "")[:280],
                "source": wpl.get("source", "wikipedia"),
            }
        )
    else:
        wen = wiki_summary(topic, lang="en")
        if wen:
            ev.append(
                {
                    "title": wen.get("title", ""),
                    "link": wen.get("url", ""),
                    "snippet": wen.get("extract", "")[:280],
                    "source": wen.get("source", "wikipedia"),
                }
            )
    dd = ddg_instant(topic)
    if dd and (dd.get("abstract") or dd.get("url")):
        ev.append(
            {
                "title": topic,
                "link": dd.get("url", ""),
                "snippet": (dd.get("abstract") or "")[:280],
                "source": "duckduckgo",
            }
        )
    ev += google_books_search(topic, max_results=max(1, min(5, k)))
    ev += hn_search(topic, max_hits=max(2, min(5, k // 2)))
    ev += openalex_search(topic, max_hits=max(2, min(5, k // 2)))
    uniq = []
    seen = set()
    for e in ev:
        key = ((e.get("title") or "")[:80], (e.get("link") or "")[:140])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
        if len(uniq) >= k:
            break
    return uniq


def citations_block(evidences: list[dict]) -> str:
    if not evidences:
        return ""
    lines = ["### ŹRÓDŁA"]
    for e in evidences:
        t = (e.get("title") or "").strip() or "(bez tytułu)"
        u = e.get("link") or ""
        s = e.get("source") or "web"
        sn = (e.get("snippet") or "").strip()
        if u:
            lines.append(f"- {t} — {u}  \n  _{s}_: {sn}")
        else:
            lines.append(f"- {t}  \n  _{s}_: {sn}")
    return "\n".join(lines)


def enrich_with_research(
    md: str, topic: str, lang: str = "pl", k: int = 8, add_citations: bool = True
) -> tuple[str, list[dict]]:
    ev = research_evidence(topic, lang=lang, k=k)
    if not ev:
        return (md, [])
    block = citations_block(ev) if add_citations else ""
    out = md.rstrip() + ("\n\n" + block if block else "")
    if ev:
        _mem_add(
            f"[WRITING:RESEARCH] {topic}\n{out}",
            tags=["writing", "research", "citations"],
            user="global",
        )
    return (out, ev)


def attach_citations_per_section(text: str, topic: str, k: int = 3) -> str:
    blocks = text.split("\n\n")
    for i, base in enumerate(blocks):
        if len(base) < 100 or base.startswith("#"):
            continue
        _, ev = enrich_with_research(base, topic, lang="pl", k=k, add_citations=False)
        if ev:
            cite = ["> Źródła:"]
            for e in ev[:k]:
                t = (e.get("title") or "").strip() or "(bez tytułu)"
                u = e.get("link") or ""
                cite.append(f"> - {t} — {u}")
            blocks[i] = base.rstrip() + "\n\n" + "\n".join(cite)
    return "\n\n".join(blocks)


# ───────────────────────────────────────────────────────────────────────────
# FASHION/VINTED + MARKETING BUNDLE
# ───────────────────────────────────────────────────────────────────────────
_BRAND_CACHE = DATA_DIR / "brand_cache.json"


def _brand_cache_load() -> dict[str, Any]:
    if _BRAND_CACHE.exists():
        try:
            return json.loads(_BRAND_CACHE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _brand_cache_save(obj: dict):
    try:
        _BRAND_CACHE.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


_BRAND_MEM = _brand_cache_load()

_FASHION_BRANDS: dict[str, dict[str, Any]] = {
    "acne studios": {
        "year": 1996,
        "origin": "Szwecja",
        "sign": ["minimalizm", "denim", "oversize"],
        "tier": "premium",
        "size": "lekko zawyżona w okryciach",
        "care": "wełna/jedwab delikatnie; denim prać rzadko",
    },
    "gucci": {
        "year": 1921,
        "origin": "Włochy",
        "sign": ["GG monogram", "Web stripe", "horsebit"],
        "tier": "luxury",
        "size": "zwykle TTS; obuwie wąskie",
        "care": "skóra krem; jedwab chłodno",
    },
    "prada": {
        "year": 1913,
        "origin": "Włochy",
        "sign": ["Re-Nylon", "triangle logo"],
        "tier": "luxury",
        "size": "TTS; buty wąskie",
        "care": "nylon przecierać; wełna chemicznie",
    },
    "louis vuitton": {
        "year": 1854,
        "origin": "Francja",
        "sign": ["Monogram", "Damier"],
        "tier": "luxury",
        "size": "—",
        "care": "Vachetta patynuje",
    },
    "balenciaga": {
        "year": 1919,
        "origin": "Hiszpania/FR",
        "sign": ["oversize", "Triple S"],
        "tier": "luxury",
        "size": "oversize",
        "care": "dzianiny na płasko",
    },
    "dior": {
        "year": 1946,
        "origin": "Francja",
        "sign": ["New Look", "CD monogram"],
        "tier": "luxury",
        "size": "TTS",
        "care": "jedwab/tiul chemicznie",
    },
    "stone island": {
        "year": 1982,
        "origin": "Włochy",
        "sign": ["kompas", "garment dye"],
        "tier": "premium",
        "size": "TTS",
        "care": "uważać na farbowania",
    },
    "off-white": {
        "year": 2013,
        "origin": "IT/USA",
        "sign": ["diagonals", "zip-tie"],
        "tier": "luxury",
        "size": "luźne",
        "care": "print na lewą",
    },
    "supreme": {
        "year": 1994,
        "origin": "USA",
        "sign": ["box logo", "dropy"],
        "tier": "premium",
        "size": "luźno",
        "care": "dzianiny 30°C",
    },
    "cos": {
        "year": 2007,
        "origin": "Szwecja",
        "sign": ["clean lines"],
        "tier": "mid",
        "size": "lekko zawyżone",
        "care": "standard",
    },
    "celine": {
        "year": 1945,
        "origin": "Francja",
        "sign": ["Triomphe", "tailoring"],
        "tier": "luxury",
        "size": "TTS",
        "care": "wełny/jedwab chemicznie",
    },
    "burberry": {
        "year": 1856,
        "origin": "UK",
        "sign": ["trench", "nova check"],
        "tier": "luxury",
        "size": "TTS",
        "care": "gabardyna delikatnie",
    },
    "ralph lauren": {
        "year": 1967,
        "origin": "USA",
        "sign": ["polo pony", "preppy"],
        "tier": "mid-premium",
        "size": "Polo szersze; Slim węższe",
        "care": "bawełna 30°C",
    },
    "moncler": {
        "year": 1952,
        "origin": "FR/IT",
        "sign": ["puchówki"],
        "tier": "luxury",
        "size": "TTS",
        "care": "puch delikatnie; suszenie z kulkami",
    },
    "the north face": {
        "year": 1966,
        "origin": "USA",
        "sign": ["700 fill", "Nuptse"],
        "tier": "mid",
        "size": "luźne",
        "care": "membrany niska temp.",
    },
    "saint laurent": {
        "year": 1961,
        "origin": "Francja",
        "sign": ["Le Smoking", "YSL Cassandre"],
        "tier": "luxury",
        "size": "TTS, szczupłe ramiona",
        "care": "chemicznie",
    },
    "bottega veneta": {
        "year": 1966,
        "origin": "Włochy",
        "sign": ["intrecciato"],
        "tier": "luxury",
        "size": "TTS",
        "care": "skóry odżywiać",
    },
    "hermès": {
        "year": 1837,
        "origin": "Francja",
        "sign": ["Kelly", "Birkin"],
        "tier": "luxury",
        "size": "—",
        "care": "spa brandowe",
    },
    "chanel": {
        "year": 1910,
        "origin": "Francja",
        "sign": ["tweed", "2.55"],
        "tier": "luxury",
        "size": "TTS",
        "care": "tweed delikatnie",
    },
    "fendi": {
        "year": 1925,
        "origin": "Włochy",
        "sign": ["FF Zucca", "baguette"],
        "tier": "luxury",
        "size": "TTS",
        "care": "skóry krem",
    },
    "valentino": {
        "year": 1960,
        "origin": "Włochy",
        "sign": ["rockstud", "VLogo"],
        "tier": "luxury",
        "size": "TTS",
        "care": "delikatne tkaniny",
    },
    "givenchy": {
        "year": 1952,
        "origin": "Francja",
        "sign": ["4G", "couture black"],
        "tier": "luxury",
        "size": "TTS",
        "care": "chemicznie",
    },
    "loewe": {
        "year": 1846,
        "origin": "Hiszpania",
        "sign": ["puzzle", "anagram"],
        "tier": "luxury",
        "size": "—",
        "care": "skóry premium",
    },
    "maison margiela": {
        "year": 1988,
        "origin": "Francja",
        "sign": ["cztery szwy", "tabi"],
        "tier": "luxury",
        "size": "nietypowe",
        "care": "delikatnie",
    },
    "alexander mcqueen": {
        "year": 1992,
        "origin": "UK",
        "sign": ["oversole", "skull"],
        "tier": "luxury",
        "size": "buty zawyżone",
        "care": "—",
    },
    "kenzo": {
        "year": 1970,
        "origin": "Francja",
        "sign": ["tiger"],
        "tier": "premium",
        "size": "TTS",
        "care": "druk na lewą",
    },
    "comme des garçons": {
        "year": 1969,
        "origin": "Japonia",
        "sign": ["play heart", "deconstruction"],
        "tier": "premium",
        "size": "mniejsze",
        "care": "delikatne",
    },
    "issey miyake": {
        "year": 1970,
        "origin": "Japonia",
        "sign": ["pleats please"],
        "tier": "premium",
        "size": "elastyczne",
        "care": "wg linii",
    },
    "yohji yamamoto": {
        "year": 1972,
        "origin": "Japonia",
        "sign": ["czarny oversize"],
        "tier": "premium",
        "size": "oversize",
        "care": "chemicznie",
    },
    "a bathing ape": {
        "year": 1993,
        "origin": "Japonia",
        "sign": ["bape camo", "shark hoodie"],
        "tier": "premium",
        "size": "TTS/luźno",
        "care": "print na lewą",
    },
    "nike": {
        "year": 1964,
        "origin": "USA",
        "sign": ["swoosh", "air"],
        "tier": "mid",
        "size": "TTS",
        "care": "siatka 30°C",
    },
    "adidas": {
        "year": 1949,
        "origin": "Niemcy",
        "sign": ["3 stripes", "trefoil"],
        "tier": "mid",
        "size": "TTS",
        "care": "—",
    },
    "new balance": {
        "year": 1906,
        "origin": "USA",
        "sign": ["N logo", "grey"],
        "tier": "mid",
        "size": "TTS",
        "care": "zamsz szczotkować",
    },
    "levi's": {
        "year": 1853,
        "origin": "USA",
        "sign": ["501", "red tab"],
        "tier": "mid",
        "size": "TTS",
        "care": "denim rzadko prać",
    },
    "carhartt wip": {
        "year": 1889,
        "origin": "USA/DE",
        "sign": ["workwear", "square label"],
        "tier": "mid",
        "size": "luźne",
        "care": "—",
    },
    "patagonia": {
        "year": 1973,
        "origin": "USA",
        "sign": ["synchilla", "R1"],
        "tier": "mid",
        "size": "TTS",
        "care": "reimpregnacja DWR",
    },
    "arcteryx": {
        "year": 1989,
        "origin": "Kanada",
        "sign": ["dead bird", "Gore-Tex"],
        "tier": "premium",
        "size": "sport fit",
        "care": "membrany low",
    },
    "fear of god": {
        "year": 2013,
        "origin": "USA",
        "sign": ["essentials", "boxy"],
        "tier": "premium",
        "size": "oversize",
        "care": "dzianiny 30°C",
    },
    "amiri": {
        "year": 2014,
        "origin": "USA",
        "sign": ["distressed denim", "bones"],
        "tier": "luxury",
        "size": "slim",
        "care": "—",
    },
    "a.p.c.": {
        "year": 1987,
        "origin": "Francja",
        "sign": ["raw denim"],
        "tier": "premium",
        "size": "TTS",
        "care": "denim dry",
    },
    "stussy": {
        "year": 1980,
        "origin": "USA",
        "sign": ["script"],
        "tier": "premium",
        "size": "luźne",
        "care": "—",
    },
}
_TIER_BASE = {
    "fast-fashion": 1.0,
    "mid": 1.4,
    "mid-premium": 1.7,
    "premium": 2.3,
    "luxury": 3.5,
}
_COND_FACTOR = {
    "Nowy z metkami": 1.0,
    "Nowy": 0.93,
    "Jak nowy": 0.85,
    "Bardzo dobry": 0.78,
    "Dobry": 0.68,
    "Do poprawek": 0.45,
}


def _merge_brand(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in (extra or {}).items():
        if v in (None, "", [], {}):
            continue
        if k == "sign":
            out["sign"] = list(dict.fromkeys((out.get("sign") or []) + (v or [])))
        elif k == "links":
            out["links"] = list(dict.fromkeys((out.get("links") or []) + (v or [])))
        else:
            out[k] = v
    return out


def brand_info_web(name: str) -> dict:
    b = name.strip()
    wiki = wiki_summary(b, lang="en") or {}
    dd = ddg_instant(b + " fashion brand") or {}
    press = fashion_press_enrich(b)
    info = {
        "brand": b,
        "bio": (wiki.get("extract") or dd.get("abstract") or press.get("press_bio") or "").strip(),
        "links": [x for x in [wiki.get("url"), dd.get("url")] if x]
        + (press.get("press_urls") or []),
        "press_signs": press.get("press_signs") or [],
        "source_combo": [
            x
            for x in [
                "wikipedia" if wiki else "",
                "duckduckgo" if dd else "",
                "press" if press.get("press_urls") else "",
            ]
            if x
        ],
    }
    return info


def brand_knowledge(name: str, web: bool = True) -> dict:
    n = (name or "").strip().lower()
    base = _FASHION_BRANDS.get(n) or {}
    cache = _BRAND_MEM.get(n) or {}
    merged = _merge_brand(base, cache)
    need_web = web and (
        (not merged.get("bio")) or (time.time() - int(cache.get("web_ts", 0))) > 86400
    )
    if need_web:
        info = brand_info_web(name)
        extra = {}
        if info.get("bio"):
            extra["bio"] = info["bio"]
        if info.get("links"):
            extra["links"] = info["links"]
        if info.get("press_signs"):
            extra["sign"] = list(dict.fromkeys((merged.get("sign") or []) + info["press_signs"]))
        extra["web_ts"] = int(time.time())
        merged = _merge_brand(merged, extra)
        _BRAND_MEM[n] = merged
        _brand_cache_save(_BRAND_MEM)
    return merged


def price_suggest(
    brand: str, cond: str, base_price: float | None = None, tier_hint: str | None = None
) -> dict:
    binfo = brand_knowledge(brand, web=True)
    tier = tier_hint or binfo.get("tier") or "mid"
    base = base_price or 100.0 * _TIER_BASE.get(tier, 1.4)
    f = _COND_FACTOR.get(cond, 0.7)
    p = round(base * f, 0)
    band = (max(1, int(p * 0.9)), int(p * 1.15))
    return {"price": int(p), "range": band, "tier": tier}


# ───────────────────────────────────────────────────────────────────────────
# CREATIVE WRITER | SPEC I GENERATOR
# ───────────────────────────────────────────────────────────────────────────
@dataclass
class WriteSpec:
    kind: str
    topic: str
    user: str = "global"
    persona: str | None = None
    tone: str = "neutral"
    audience: str = "gen"
    keywords: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


def _sys_psy(uid: str, persona: str | None = None) -> str:
    p = _psy_snapshot(uid)
    per = persona or p.get("persona", "neutral")
    return (
        f"Persona:{per} | Mood:{p.get('mood','spokój')} | "
        f"Energy:{p.get('energy',70)} | Creativity:{p.get('creativity',50)}"
    )


def _ask_llm(system: str, prompt: str, temp: float, max_t: int) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    return llm_chat(msgs, temperature=temp, max_tokens=max_t).strip()


def poem_free(spec: WriteSpec, stanzas: int = 4, lines: tuple[int, int] = (3, 5)) -> str:
    sys = _sys_psy(spec.user, spec.persona)
    prm = f"""Napisz wiersz wolny o temacie: {spec.topic}.
Zwrotek: {stanzas}. Na zwrotkę {lines[0]}-{lines[1]} wersów. Zero banału, obrazowo, sensory."""
    return _ask_llm(sys, prm, temp=0.55, max_t=800)


def fraszka(spec: WriteSpec, rude: bool = True) -> str:
    sys = _sys_psy(spec.user, spec.persona)
    tag = "możesz przeklinać, sarkazm dozwolony" if rude else "ironia, bez wulgaryzmów"
    prm = f"Temat: {spec.topic}. Napisz fraszkę 2-3 wersy z ostrą puentą, {tag}."
    return _ask_llm(sys, prm, temp=0.6, max_t=160)


def essay(spec: WriteSpec, arguments: list[str] | None = None, paras: int = 5) -> str:
    sys = _sys_psy(spec.user, spec.persona)
    args = arguments or ["argument 1", "argument 2", "argument 3"]
    prm = f"""Teza: {spec.topic}
Argumenty: {', '.join(args)}.
Napisz esej na {paras} akapitów: teza → argumenty (przykłady) → kontrargumenty → wnioski."""
    return _ask_llm(sys, prm, temp=0.45, max_t=1400)


def story(spec: WriteSpec, paragraphs: int = 28, genre: str = "obserwacyjny") -> str:
    sys = _sys_psy(spec.user, spec.persona)
    prm = f"""Opowiadanie ({genre}). Temat: {spec.topic}.
~{paragraphs} akapitów, narracja spójna, sceny, zmysły, dialogi."""
    return _ask_llm(sys, prm, temp=0.6, max_t=3200)


def novel_outline(spec: WriteSpec, acts: int = 3) -> str:
    sys = _sys_psy(spec.user, spec.persona)
    prm = f"""Konspekt powieści (akty: {acts}) — tytuły H2/H3 scen i krótkie notki.
Temat: {spec.topic}. Ton: {spec.tone}. Uporządkuj klarownie."""
    return _ask_llm(sys, prm, temp=0.45, max_t=1600)


def screenplay(spec: WriteSpec, scenes: int = 10) -> str:
    sys = _sys_psy(spec.user, spec.persona)
    prm = f"Scenariusz filmowy. Temat: {spec.topic}. {scenes} scen. SCENE HEADING + opis + dialogi."
    return _ask_llm(sys, prm, temp=0.55, max_t=2400)


def manifesto(spec: WriteSpec, paras: int = 8) -> str:
    sys = _sys_psy(spec.user, spec.persona)
    prm = f"Napisz manifest ({paras} akapitów). Temat: {spec.topic}. Krótkie zdania, bez waty."
    return _ask_llm(sys, prm, temp=0.6, max_t=1600)


def dialogue(spec: WriteSpec, chars: list[str] | None = None, lines: int = 20) -> str:
    if chars is None:
        chars = ["A", "B"]
    sys = _sys_psy(spec.user, spec.persona)
    prm = f"""Dialog ({lines} wymian). Bohaterowie: {', '.join(chars)}.
Temat: {spec.topic}. Styl realistyczny, pauzy, język potoczny."""
    return _ask_llm(sys, prm, temp=0.6, max_t=1600)


def haiku(spec: WriteSpec, n: int = 3) -> str:
    sys = _sys_psy(spec.user, spec.persona)
    prm = f"Napisz {n} haiku o temacie {spec.topic}. Forma 5-7-5. Minimalizm, natura."
    return _ask_llm(sys, prm, temp=0.5, max_t=400)


def social_post(spec: WriteSpec, platform: str = "linkedin") -> str:
    sys = _sys_psy(spec.user, spec.persona)
    prm = f"""Post na {platform}. Temat: {spec.topic}.
Hook 1-2 zdania, 1 krótki akapit mięsa, CTA. Zero waty."""
    return _ask_llm(sys, prm, temp=0.45, max_t=360)


def thread(spec: WriteSpec, platform: str = "x", n: int = 8) -> str:
    sys = _sys_psy(spec.user, spec.persona)
    prm = f"""Wątek na {platform}. Temat: {spec.topic}. {n} punktów, każdy 1-2 zdania, numeracja."""
    return _ask_llm(sys, prm, temp=0.45, max_t=800)


def _corpus_cache(max_files: int = 1200) -> list[str]:
    pool: list[str] = []
    for dp, _, fns in os.walk(OUT_DIR):
        for fn in fns:
            if len(pool) >= max_files:
                return pool
            if fn.endswith((".md", ".txt")):
                try:
                    pool.append((Path(dp) / fn).read_text(encoding="utf-8")[:40000])
                except Exception:
                    pass
    return pool


# ───────────────────────────────────────────────────────────────────────────
# OPCJONALNE WSTRZYKNIĘCIE KONTEKSTU KRYPTOWALUT
# ───────────────────────────────────────────────────────────────────────────
def crypto_context(symbol: str, vs: str = "usd") -> str:
    if not (_HAS_CRYPTO and crypto and symbol):
        return ""
    try:
        lines = []
        if hasattr(crypto, "scan_symbol"):
            s = crypto.scan_symbol(symbol, vs=vs)  # type: ignore
            if isinstance(s, dict):
                price = s.get("price")
                ch24 = s.get("change_24h")
                mcap = s.get("market_cap")
                vol = s.get("volume_24h")
                lines.append(
                    f"**{symbol.upper()}**: cena {price} {vs}, 24h {ch24}%, "
                    f"MC {mcap}, wolumen 24h {vol}."
                )
        if hasattr(crypto, "make_report"):
            r = crypto.make_report(symbol, horizon="30d", risk_profile="medium")  # type: ignore
            if isinstance(r, dict):
                key = _short(str(r.get("summary") or "") or _short(json.dumps(r)[:600], 240), 240)
                lines.append(f"Raport 30d: {key}")
        if lines:
            _psy_event("writer_crypto_context", {"symbol": symbol})
            _mem_add("\n".join(lines), tags=["crypto", "context"], conf=0.7)
            _auto_learn({"kind": "writer_crypto_context", "symbol": symbol})
        return "\n".join(lines)
    except Exception:
        return ""


# ───────────────────────────────────────────────────────────────────────────
# PLUGIN INTEGRATION
# ───────────────────────────────────────────────────────────────────────────
def _try_plugin_generate(spec: WriteSpec) -> str | None:
    """Try to generate content using plugin system."""
    try:
        from plugins.writing.base import WritingContext
        from plugins.writing.manager import get_plugin_manager

        manager = get_plugin_manager()

        # Convert WriteSpec to WritingContext
        context = WritingContext(
            topic=spec.topic,
            user=spec.user or "global",
            persona=spec.persona,
            tone=spec.tone,
            audience=spec.audience,
            keywords=spec.keywords,
            style_params=spec.extra,
            research_enabled=spec.extra.get("research_web", True),
            mood=_psy_snapshot(spec.user or "global").get("mood", "spokój"),
            energy=_psy_snapshot(spec.user or "global").get("energy", 70),
            creativity=_psy_snapshot(spec.user or "global").get("creativity", 50),
        )

        # Auto-select plugin based on kind and topic
        plugin_name = None
        if spec.kind in ("vinted", "olx", "allegro", "marketplace", "aukcja"):
            plugin_name = "vinted"
        elif spec.kind in ("post", "social", "thread", "watek", "twitter", "linkedin"):
            plugin_name = "social"
        elif spec.kind in ("blog", "seo", "artykul", "article"):
            plugin_name = "seo_blog"

        # If no explicit plugin match, try auto-selection
        if not plugin_name:
            plugin_name = manager.auto_select_plugin(context)

        if plugin_name:
            result = manager.generate_content(plugin_name, context)
            if result:
                return result.content
    except Exception as e:
        print(f"Plugin system error: {e}")

    return None


# ───────────────────────────────────────────────────────────────────────────
# GENERATE (z RAG, humanizacją, guardami, opcjonalnym crypto)
# ───────────────────────────────────────────────────────────────────────────
def generate(spec: WriteSpec) -> dict[str, Any]:
    uid = spec.user or "global"
    sm = style_mix(
        None,
        pace=float(spec.extra.get("pace", 0.55)),
        sensory=float(spec.extra.get("sensory", 0.75)),
        irony=float(spec.extra.get("irony", 0.25)),
        slang=float(spec.extra.get("slang", 0.2)),
        pathos=float(spec.extra.get("pathos", 0.35)),
        sharp=float(spec.extra.get("profanity_mode", 0.0)),
        surreal=float(spec.extra.get("surreal", 0.15)),
    )

    # Try plugin system first
    raw = _try_plugin_generate(spec)

    # Fallback to legacy system if plugin fails
    if not raw:
        kind = spec.kind
        if kind in ("wiersz", "poem"):
            raw = poem_free(spec, stanzas=int(spec.extra.get("stanzas", 4)))
        elif kind in ("fraszka", "epigram"):
            raw = fraszka(spec, rude=True)
        elif kind in ("esej", "essay"):
            raw = essay(spec, paras=int(spec.extra.get("paras", 5)))
        elif kind in ("opowiadanie", "story"):
            raw = story(
                spec,
                paragraphs=int(spec.extra.get("paragraphs", 28)),
                genre=spec.extra.get("genre", "obserwacyjny"),
            )
        elif kind in ("konspekt", "powieść", "powiesc", "novel_outline"):
            raw = novel_outline(spec, acts=int(spec.extra.get("acts", 3)))
        elif kind in ("scenariusz",):
            raw = screenplay(spec, scenes=int(spec.extra.get("scenes", 10)))
        elif kind in ("manifest",):
            raw = manifesto(spec, paras=int(spec.extra.get("paras", 8)))
        elif kind in ("dialog",):
            raw = dialogue(
                spec,
                chars=spec.extra.get("chars", ["A", "B"]),
                lines=int(spec.extra.get("lines", 20)),
            )
        elif kind in ("haiku",):
            raw = haiku(spec, n=int(spec.extra.get("n", 3)))
        elif kind in ("post", "social"):
            raw = social_post(spec, platform=spec.extra.get("platform", "linkedin"))
        elif kind in ("watek", "thread"):
            raw = thread(
                spec,
                platform=spec.extra.get("platform", "x"),
                n=int(spec.extra.get("n", 8)),
            )
        else:
            sys = _sys_psy(uid, spec.persona)
            prm = f"{spec.topic}\nKrótki, mocny tekst (2-4 akapity). Zero waty."
            raw = _ask_llm(sys, prm, temp=0.45, max_t=900)

    txt = apply_style_matrix(raw, sm)
    txt = maybe_sharpen(txt, sm["sharp"])

    corpus = _corpus_cache()
    txt, dup = ngram_guard(
        txt,
        corpus=corpus,
        n=int(spec.extra.get("dup_n", 5)),
        max_ratio=float(spec.extra.get("dup_max", 0.08)),
    )

    txt = humanize(
        debloat(txt),
        {
            "emotion": spec.extra.get("emotion", "spokój"),
            "cadence": spec.extra.get("cadence", "mix"),
            "chaos": float(spec.extra.get("chaos", 0.32)),
            "slang": sm["slang"] > 0.5,
            "rhetoric": True,
            "sensory": sm["sensory"] > 0.5,
        },
        _psy_snapshot(uid),
    )

    # RAG/citations
    per_section = bool(spec.extra.get("per_section_citations", True))
    if bool(spec.extra.get("research_web", True)):
        if per_section:
            txt = attach_citations_per_section(txt, spec.topic, k=int(spec.extra.get("cit_k", 3)))
        else:
            txt, _ = enrich_with_research(
                txt,
                spec.topic,
                lang="pl",
                k=int(spec.extra.get("cit_k", 6)),
                add_citations=True,
            )

    # optional crypto injection
    if spec.extra.get("crypto_symbol"):
        note = crypto_context(
            str(spec.extra.get("crypto_symbol")),
            vs=str(spec.extra.get("crypto_vs", "usd")),
        )
        if note:
            txt += "\n\n---\n" + "**Kontekst rynkowy:**\n" + note

    prefix = f"{spec.kind}_{_slug(spec.topic)}"
    saved = _save(
        {
            "text": txt,
            "meta": {
                "kind": spec.kind,
                "topic": spec.topic,
                "tone": spec.tone,
                "audience": spec.audience,
                "persona": spec.persona,
                "keywords": spec.keywords,
                "dup_score": dup,
                "style": sm,
                "with_research": "### ŹRÓDŁA" in txt,
                "crypto": bool(spec.extra.get("crypto_symbol")),
            },
        },
        base_dir=OUT_DIR,
        prefix=prefix,
        ext=spec.extra.get("ext", "md"),
    )

    if spec.extra.get("html", False):
        Path(saved["path"]).with_suffix(".html").write_text(
            html_export2(spec.topic, txt), encoding="utf-8"
        )
    if spec.extra.get("pdf", False):
        export_pdf(txt, str(Path(saved["path"]).with_suffix(".pdf")))

    _mem_add(
        f"[WRITING:{spec.kind}] {spec.topic}\n{_short(txt, 400)}",
        tags=["writing", spec.kind],
        user=uid,
        conf=0.7,
    )
    _psy_event("writing_generate", {"kind": spec.kind, "len": len(txt), "topic": spec.topic})
    _psy_event(
        "writing_done", {"kind": spec.kind, "len": len(txt), "topic": spec.topic}
    )  # nowy event
    _auto_learn(
        {
            "kind": "writing_generate",
            "topic": spec.topic,
            "extra": {"crypto": bool(spec.extra.get("crypto_symbol"))},
        }
    )

    return {"text": txt, **saved}


# ───────────────────────────────────────────────────────────────────────────
# VINTED LISTING | MARKETING
# ───────────────────────────────────────────────────────────────────────────
def vinted_listing(
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
) -> dict:
    binfo = brand_knowledge(brand, web=True)
    signs = list(dict.fromkeys((binfo.get("sign") or [])[:6]))
    care = binfo.get("care", "")
    bio = binfo.get("bio", "")
    hints = []
    if "druk" in care.lower() or "print" in care.lower():
        hints.append("prać na lewą stronę")
    if "membr" in care.lower():
        hints.append("niskie temperatury, reimpregnacja DWR")
    if "puch" in care.lower() or "puchówki" in care.lower():
        hints.append("suszyć z kulkami, rozbić puch")

    pr = price_suggest(brand, cond, base_price, tier_hint)
    price, (pmin, pmax) = pr["price"], pr["range"]

    kw = [
        brand,
        item,
        model,
        size,
        color,
        material,
        "oryginał",
        "100% autentyczność",
        "fast shipping",
        "ideal na prezent",
    ]
    kw = [k for k in kw if k]
    kw = list(dict.fromkeys(kw))[:14]

    core = f"{brand} {item}".strip()
    tA = f"{core} {model} {size}".strip()
    tB = f"{core} — {color} {size} • stan: {cond}".strip()

    defects = defects or []
    extras = extras or []
    measurements = measurements or {}
    meas_lines = [f"{k}: {v}" for k, v in measurements.items() if v]
    defect_line = ("Braki/wady: " + ", ".join(defects)) if defects else ""
    extras_line = ("Dodatki: " + ", ".join(extras)) if extras else ""
    signs_line = (", ".join(signs)) if signs else ""
    md = []
    md.append(f"# {core}")
    md.append(f"**Model:** {model or '-'} | **Rozmiar:** {size} | **Stan:** {cond}")
    if signs_line:
        md.append(f"**Charakter:** {signs_line}")
    if color or material:
        md.append(f"**Kolor/Materiał:** {color or '-'} / {material or '-'}")
    if meas_lines:
        md.append("**Wymiary (na płasko):**\n- " + "\n- ".join(meas_lines))
    if defect_line:
        md.append(defect_line)
    if extras_line:
        md.append(extras_line)
    if bio:
        md.append("> " + _short(bio, 220))
    if care or hints:
        md.append(
            f"**Pielęgnacja:** {care or ''} {('• ' + '; '.join(hints)) if hints else ''}".strip()
        )
    md.append("\n**Wysyłka:** 24-48h • Bez dymu • Bundle -12%.")
    md.append("**Autentyczność:** legalny zakup; metki na zdjęciach.")
    md.append("\n### Cena\n" + f"Proponuję **{price} zł** (okno: {pmin}-{pmax} zł).")
    md.append(
        "\n### Hashtagi/SEO\n"
        + " ".join(["#" + re.sub(r"[^a-z0-9]+", "", x.lower()) for x in kw if x])
    )
    return {
        "title_A": tA,
        "title_B": tB,
        "price": price,
        "price_range": (pmin, pmax),
        "description_md": "\n\n".join(md).strip(),
        "hashtags": kw,
    }


# ───────────────────────────────────────────────────────────────────────────
# LONGFORM + sanitizers
# ───────────────────────────────────────────────────────────────────────────
def coherence_guard(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    ws = re.findall(r"\w+", t.lower())
    seen = set()
    bad = set()
    for n in (7, 6, 5, 4, 3):
        rng = max(0, len(ws) - n + 1)
        for i in range(0, rng):
            grp = tuple(ws[i : i + n])
            if grp in seen:
                bad.add(" ".join(grp))
            seen.add(grp)
    if bad:
        for g in sorted(bad, key=len, reverse=True):
            t = re.sub(r"(?i)\b" + re.escape(g) + r"\b", "", t)
        t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"(\n){3,}", "\n\n", t)
    return t.strip()


def deai_text(text: str) -> str:
    bad = [
        "Jako model językowy",
        "Jako sztuczna inteligencja",
        "As an AI",
        "I cannot",
        "I can't answer",
    ]
    t = text or ""
    for b in bad:
        t = re.sub(re.escape(b), "", t, flags=re.I)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def novel_chaptered(
    topic: str, chapters: int = 12, words: int = 1200, persona: str | None = None
) -> str:
    sys_msg = "Longform engine — spójnie, ludzki styl, detale i dialogi."
    outline = (
        llm_chat(
            [
                {"role": "system", "content": sys_msg},
                {
                    "role": "user",
                    "content": "Temat: "
                    + str(topic)
                    + ". Daj outline "
                    + str(chapters)
                    + " rozdziałów, każdy 2-3 zdania.",
                },
            ],
            temperature=0.45,
            max_tokens=900,
        )
        or ""
    )
    out = ["# " + str(topic), "## Spis treści", (outline or "").strip()]
    for i in range(1, int(chapters) + 1):
        ask = f"Napisz rozdział {i}/{chapters}. Temat: {topic}. 350-700 słów, dialogi i detale."
        ch = (
            llm_chat(
                [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": ask},
                ],
                temperature=0.6,
                max_tokens=1700,
            )
            or ""
        )
        ch = coherence_guard(ch)
        try:
            ch = humanize(ch, {"chaos": 0.3, "sensory": True, "rhetoric": True})
        except Exception:
            pass
        out.append(f"## Rozdział {i}")
        out.append(ch.strip())
    return "\n\n".join(out).strip()


# ───────────────────────────────────────────────────────────────────────────
# RESEARCH + EXPORT API
# ───────────────────────────────────────────────────────────────────────────
def research_and_export(
    topic: str,
    base_md: str,
    html_out: bool = True,
    pdf_out: bool = False,
    filename_prefix: str | None = None,
) -> dict:
    filename_prefix = filename_prefix or f"doc_{_slug(topic)}"
    md2, ev = enrich_with_research(base_md, topic, lang="pl", k=8, add_citations=True)
    saved = _save(
        {
            "text": md2,
            "meta": {"topic": topic, "with_research": True, "sources_count": len(ev)},
        },
        base_dir=OUT_DIR,
        prefix=filename_prefix,
        ext="md",
    )
    paths = {"md": saved["path"], "meta": saved["meta_path"]}
    if html_out:
        hx = html_export2(topic, md2)
        p = Path(saved["path"]).with_suffix(".html")
        p.write_text(hx, encoding="utf-8")
        paths["html"] = str(p)
    if pdf_out:
        p = Path(saved["path"]).with_suffix(".pdf")
        if export_pdf(md2, str(p)):
            paths["pdf"] = str(p)
    return {"paths": paths, "sources": ev}


# ───────────────────────────────────────────────────────────────────────────
# ASSIST — jeden punkt wejścia (bez ręcznego "trybu")
# ───────────────────────────────────────────────────────────────────────────
# Heurystyki: rodzaj treści + opcjonalne crypto wzbogacenie
_CRYPTO_SYMS = {
    "btc",
    "eth",
    "sol",
    "bnb",
    "xrp",
    "ada",
    "doge",
    "ton",
    "link",
    "arb",
    "op",
    "dot",
    "avax",
    "trx",
    "matic",
    "atom",
    "uni",
    "ltc",
    "aave",
    "fil",
    "near",
    "sui",
    "sei",
    "inj",
    "xlm",
    "apt",
    "ftm",
}
_SYM_RE = re.compile(r"(?i)(?:^|\b|\$)([a-z]{2,6})(?:\b)")


def _looks_like_crypto(prompt: str) -> str | None:
    p = prompt.lower()
    for m in _SYM_RE.finditer(p):
        s = (m.group(1) or "").lower()
        if s in _CRYPTO_SYMS:
            return s
    if any(
        k in p
        for k in (
            "crypto",
            "krypto",
            "coin",
            "token",
            "staking",
            "defi",
            "halving",
            "onchain",
            "mcap",
            "gas fee",
            "gas fees",
        )
    ):
        for s in _CRYPTO_SYMS:
            if f" {s} " in f" {p} ":
                return s
    return None


def _pick_kind(prompt: str) -> str:
    if re.search(r"(?i)\bthread|wątek\b", prompt):
        return "watek"
    if re.search(r"(?i)\bmanifest\b", prompt):
        return "manifest"
    wc = len(re.findall(r"\w+", prompt))
    if wc < 30:
        return "post"
    return "esej"


def assist(
    prompt: str,
    user: str = "global",
    html: bool = False,
    pdf: bool = False,
    inline_citations: bool = True,
) -> dict[str, Any]:
    symbol = _looks_like_crypto(prompt)
    kind = _pick_kind(prompt)
    spec = WriteSpec(
        kind=kind,
        topic=prompt,
        user=user,
        extra={
            "research_web": True,
            "per_section_citations": inline_citations,
            "html": html,
            "pdf": pdf,
            "crypto_symbol": symbol or "",
            "crypto_vs": "usd",
            "profanity_mode": 0.0,
        },
    )
    out = generate(spec)
    return {
        "path": out.get("path"),
        "meta": out.get("meta_path"),
        "kind": kind,
        "crypto": bool(symbol),
    }


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────
def main():
    import argparse

    p = argparse.ArgumentParser(
        description="writing_all_pro — PRO/ULTRA (DeepInfra MAIN, Gemini MINI, opcjonalne crypto)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("write", help="Creative/marketing text")
    g.add_argument(
        "--kind",
        required=True,
        help="wiersz|fraszka|esej|opowiadanie|konspekt|scenariusz|manifest|dialog|haiku|post|watek",
    )
    g.add_argument("--topic", required=True)
    g.add_argument("--persona", default=None)
    g.add_argument("--no-web", action="store_true")
    g.add_argument("--html", action="store_true")
    g.add_argument("--pdf", action="store_true")
    g.add_argument("--crypto", default="", help="opcjonalny symbol, np. btc, eth")
    g.add_argument("--crypto-vs", default="usd")
    g.add_argument("--inline-citations", action="store_true")

    v = sub.add_parser("vinted", help="Aukcja Vinted/fashion PRO")
    v.add_argument("--brand", required=True)
    v.add_argument("--item", required=True)
    v.add_argument("--cond", required=True)
    v.add_argument("--size", required=True)
    v.add_argument("--color", default="")
    v.add_argument("--material", default="")
    v.add_argument("--model", default="")
    v.add_argument("--tier", default=None)
    v.add_argument("--base", type=float, default=None)

    r = sub.add_parser("research", help="Wzbogacenie + eksport")
    r.add_argument("--topic", required=True)
    r.add_argument("--html", action="store_true")
    r.add_argument("--pdf", action="store_true")

    nf = sub.add_parser("novel", help="Rozdziałowa longform")
    nf.add_argument("--topic", required=True)
    nf.add_argument("--chapters", type=int, default=12)

    a = sub.add_parser("assist", help="Jeden prompt → gotowy tekst z RAG i opcjonalnym crypto")
    a.add_argument("--prompt", required=True)
    a.add_argument("--html", action="store_true")
    a.add_argument("--pdf", action="store_true")
    a.add_argument(
        "--no-inline",
        action="store_true",
        help="wyłącz przypisy sekcyjne, dawaj blok na końcu",
    )

    # Plugin system command
    pl = sub.add_parser("plugin", help="Użyj systemu pluginów pisarskich")
    pl.add_argument("--plugin", help="Nazwa pluginu (auto-select jeśli brak)")
    pl.add_argument("--topic", help="Temat do napisania")
    pl.add_argument("--variants", action="store_true", help="Generuj 3 warianty A/B")
    pl.add_argument("--list", action="store_true", help="Pokaż dostępne pluginy")
    pl.add_argument("--user", default="global", help="ID użytkownika")
    pl.add_argument("--tone", default="neutral", help="Ton wypowiedzi")
    pl.add_argument("--audience", default="gen", help="Grupa docelowa")
    pl.add_argument("--pace", type=float, default=0.55, help="Tempo (0-1)")
    pl.add_argument("--sensory", type=float, default=0.75, help="Obrazowość (0-1)")
    pl.add_argument("--irony", type=float, default=0.25, help="Ironia (0-1)")
    pl.add_argument("--slang", type=float, default=0.2, help="Slang (0-1)")
    pl.add_argument("--pathos", type=float, default=0.35, help="Emocje (0-1)")
    pl.add_argument("--creativity", type=float, default=50, help="Kreatywność (0-100)")

    args = p.parse_args()

    if args.cmd == "write":
        spec = WriteSpec(
            kind=args.kind,
            topic=args.topic,
            persona=args.persona,
            extra={
                "research_web": not args.no_web,
                "citations": True,
                "html": args.html,
                "pdf": args.pdf,
                "per_section_citations": args.inline_citations,
                "profanity_mode": 0.0,
                "crypto_symbol": args.crypto.strip(),
                "crypto_vs": args.crypto_vs,
            },
        )
        res = generate(spec)
        print(
            json.dumps(
                {"path": res.get("path"), "meta": res.get("meta_path")},
                ensure_ascii=False,
                indent=2,
            )
        )

    elif args.cmd == "vinted":
        res = vinted_listing(
            brand=args.brand,
            item=args.item,
            cond=args.cond,
            size=args.size,
            color=args.color,
            material=args.material,
            model=args.model,
            tier_hint=args.tier,
            base_price=args.base,
        )
        md = f"# {res['title_A']}\n\n" + res["description_md"]
        saved = _save(
            {
                "text": md,
                "meta": {
                    "kind": "vinted",
                    "brand": args.brand,
                    "item": args.item,
                    "price": res["price"],
                    "range": res["price_range"],
                },
            },
            base_dir=OUT_DIR,
            prefix=f"vinted_{_slug(args.brand+'_'+args.item)}",
            ext="md",
        )
        Path(saved["path"]).with_suffix(".ab.txt").write_text(
            res["title_A"] + "\n" + res["title_B"] + "\n", encoding="utf-8"
        )
        Path(saved["path"]).with_suffix(".tags.txt").write_text(
            "\n".join(res["hashtags"]) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "path": saved.get("path"),
                    "price": res["price"],
                    "range": res["price_range"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    elif args.cmd == "research":
        base = "# Draft\n\nTu wrzuć treść albo podmień przez CLI."
        out = research_and_export(args.topic, base_md=base, html_out=args.html, pdf_out=args.pdf)
        print(json.dumps(out, ensure_ascii=False, indent=2))

    elif args.cmd == "novel":
        text = novel_chaptered(args.topic, chapters=args.chapters)
        saved = _save(
            {"text": text, "meta": {"kind": "novel", "topic": args.topic}},
            base_dir=OUT_DIR,
            prefix=f"novel_{_slug(args.topic)}",
            ext="md",
        )
        print(json.dumps({"path": saved["path"]}, ensure_ascii=False, indent=2))

    elif args.cmd == "assist":
        res = assist(
            args.prompt,
            html=args.html,
            pdf=args.pdf,
            inline_citations=not args.no_inline,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))

    elif args.cmd == "plugin":
        try:
            from plugins.writing.base import WritingContext
            from plugins.writing.manager import get_plugin_manager

            manager = get_plugin_manager()

            # List plugins if requested
            if args.list:
                plugins = manager.list_plugins()
                print("\nDostępne pluginy:")
                for name in plugins:
                    plugin = manager.get_plugin(name)
                    if plugin:
                        print(f"  {name}: {plugin.description} [{plugin.category}]")
                exit(0)

            # Validate topic for generation
            if not args.topic:
                print("Błąd: --topic jest wymagany do generowania treści")
                exit(1)

            # Build context
            context = WritingContext(
                topic=args.topic,
                user=args.user,
                tone=args.tone,
                audience=args.audience,
                style_params={
                    "pace": args.pace,
                    "sensory": args.sensory,
                    "irony": args.irony,
                    "slang": args.slang,
                    "pathos": args.pathos,
                },
                creativity=args.creativity,
            )

            if args.variants:
                # Generate A/B variants
                plugin_name = args.plugin or manager.auto_select_plugin(context)
                if not plugin_name:
                    print("Błąd: Nie znaleziono odpowiedniego pluginu")
                    exit(1)

                results = manager.generate_variants(plugin_name, context)
                for i, result in enumerate(results, 1):
                    saved = _save(
                        {
                            "text": result.content,
                            "meta": {
                                "plugin": plugin_name,
                                "variant": i,
                                "score": result.metadata.get("score", 0),
                                "topic": args.topic,
                            },
                        },
                        base_dir=OUT_DIR,
                        prefix=f"plugin_{plugin_name}_v{i}_{_slug(args.topic)}",
                        ext="md",
                    )
                    print(
                        f"Wariant {i}: {saved['path']} (score: {result.metadata.get('score', 0)})"
                    )
            else:
                # Single generation
                plugin_name = args.plugin or manager.auto_select_plugin(context)
                if not plugin_name:
                    print("Błąd: Nie znaleziono odpowiedniego pluginu")
                    exit(1)

                result = manager.generate_content(plugin_name, context)
                if result:
                    saved = _save(
                        {
                            "text": result.content,
                            "meta": {
                                "plugin": plugin_name,
                                "score": result.metadata.get("score", 0),
                                "topic": args.topic,
                            },
                        },
                        base_dir=OUT_DIR,
                        prefix=f"plugin_{plugin_name}_{_slug(args.topic)}",
                        ext="md",
                    )
                    print(
                        json.dumps(
                            {
                                "path": saved["path"],
                                "plugin": plugin_name,
                                "score": result.metadata.get("score", 0),
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                    )
                else:
                    print("Błąd: Plugin nie wygenerował zawartości")

        except ImportError:
            print("Błąd: System pluginów nie jest dostępny")
        except Exception as e:
            print(f"Błąd pluginu: {e}")


if __name__ == "__main__":
    main()
