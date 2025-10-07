
import httpx, re, html as _html

def _ddg_search_html(q, limit=10, timeout=12):
    url = "https://duckduckgo.com/html/"
    params = {"q": q}
    out=[]
    try:
        r = httpx.post(url, data=params, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        # linki wyników (selektor pasujący do /html/ wersji DDG)
        for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', r.text):
            u = _html.unescape(m.group(1))
            t = re.sub("<.*?>","", _html.unescape(m.group(2))).strip()
            if not u or not t: continue
            # odfiltruj śmieciowe redirecty DDG
            if u.startswith("/"): continue
            out.append({"title": t[:200], "url": u})
            if len(out)>=limit: break
    except Exception:
        pass
    return out

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
autonauka_pro.py — v2 (sync, PRO)
- SERPAPI + DuckDuckGo + Firecrawl (search + scrape)
- twarde retry + timeouts, canonical URL, limit domen
- czyszczenie HTML/MD, ekstrakcja cytatów, chunkowanie
- SimHash dedup (między chunkami i źródłami)
- ranking per-źródło + globalny
- tagowanie; zapis do LTM (jeśli dostępny monolit)
- kontekst z oznaczeniami [n] -> sources[n-1]
Zachowuje sygnaturę: autonauka(query, topk=8, deep_research=False, use_external_module=True)
"""

from __future__ import annotations
import os, re, html, time, math, hashlib, random
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import httpx
from bs4 import BeautifulSoup

# spróbuj podpiąć monolit
try:
    import monolit as M
except Exception:
    M = None

# ── CONFIG ───────────────────────────────────────────────────────────────
SERPAPI_KEY   = os.getenv("SERPAPI_KEY", "")
FIRECRAWL_KEY = os.getenv("FIRECRAWL_API_KEY", "") or os.getenv("FIRECRAWL_KEY", "")
HTTP_TIMEOUT  = int(os.getenv("WEB_HTTP_TIMEOUT", "45") or "45")
UA            = os.getenv("AUTON_UA", "Autonauka/2.0")

AUTO_TOPK        = int(os.getenv("AUTO_TOPK", "8") or "8")
AUTO_FETCH       = int(os.getenv("AUTO_FETCH", "6") or "6")   # pobieramy trochę więcej, potem rank
AUTO_MIN_CHARS   = int(os.getenv("AUTO_MIN_CHARS", "600") or "600")
AUTO_MAX_CHARS   = int(os.getenv("AUTO_MAX_CHARS", "9000") or "9000")
AUTON_DOMAIN_MAX = int(os.getenv("AUTON_DOMAIN_MAX", "2") or "2")
AUTON_DEDUP_MAX  = int(os.getenv("AUTON_DEDUP_MAX", "1200") or "1200")
SAVE_TO_LTM      = os.getenv("AUTON_SAVE_LTM", "1") not in ("0","false","False","no")

# ── HTTP client z retry ──────────────────────────────────────────────────
def _client():
    limits = httpx.Limits(max_keepalive_connections=8, max_connections=16)
    return httpx.Client(timeout=HTTP_TIMEOUT, headers={"User-Agent": UA}, limits=limits, follow_redirects=True)

def _with_retry(fn, tries=3, backoff=0.7, *a, **kw):
    last = None
    for i in range(tries):
        try:
            return fn(*a, **kw)
        except Exception as e:
            last = e
            time.sleep(backoff*(1.0 + 0.2*random.random()) * (i+1))
    if last: raise last

# ── URL helpers ──────────────────────────────────────────────────────────
_CANON_SKIP_PARAMS = {"utm_source","utm_medium","utm_campaign","utm_term","utm_content","gclid","fbclid","igshid","ref","spm","mc_cid","mc_eid"}

def _domain(url: str) -> str:
    try: return urlparse(url).netloc.lower()
    except Exception: return ""

def _canonical_url(url: str) -> str:
    try:
        p = urlparse(url)
        scheme = "https" if p.scheme in ("http","https") else p.scheme
        query = urlencode([(k,v) for k,v in parse_qsl(p.query, keep_blank_values=True) if k not in _CANON_SKIP_PARAMS])
        path  = re.sub(r"/+$","", p.path or "")
        return urlunparse((scheme, p.netloc.lower(), path, "", query, ""))
    except Exception:
        return url

# ── Tekst helpers ────────────────────────────────────────────────────────
def _clean_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r"<(script|style).*?>.*?</\1>", "", text, flags=re.S|re.I)
    text = re.sub(r"\[([^\]]{1,120})\]\((https?://[^)]+)\)", r"\1", text)  # md link → plain
    text = BeautifulSoup(text, "lxml").get_text(" ")
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\u200b", "", text)
    return text.strip()

def _sentences(s: str) -> List[str]:
    s = s.replace("\r"," ")
    s = re.sub(r"\s+"," ", s)
    return [p.strip() for p in re.split(r"(?<=[\.\!\?])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ0-9])", s) if p.strip()]

def _extract_quotes(s: str) -> List[str]:
    out = []
    for line in s.splitlines():
        l = line.strip()
        if len(l) > 40 and (l.startswith(("„","“","\"","«","›","’","‘","“")) or l.endswith(("”","\"","»","‘","’"))):
            out.append(l.strip("„”“\"«»’‘"))
    return out[:4]

def _chunk_text(s: str, target_chars=900, max_chars=1400) -> List[str]:
    if not s: return []
    out, cur, cur_len = [], [], 0
    for z in _sentences(s):
        if cur_len + len(z) + 1 <= max_chars:
            cur.append(z); cur_len += len(z)+1
            if cur_len >= target_chars:
                out.append(" ".join(cur)); cur, cur_len = [], 0
        else:
            if cur: out.append(" ".join(cur))
            cur, cur_len = [z], len(z)
    if cur: out.append(" ".join(cur))
    return out

# ── SimHash dedup ────────────────────────────────────────────────────────
def _simhash(text: str, bits: int = 64) -> int:
    if not text: return 0
    tokens = re.findall(r"[a-zA-Ząćęłńóśźż0-9]{3,}", text.lower())
    if not tokens: return 0
    v = [0]*bits
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        for i in range(bits):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i, val in enumerate(v):
        if val > 0: out |= (1 << i)
    return out

def _hamm(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def _dedup_chunks(chunks: List[str], max_sim=10) -> List[str]:
    seen: List[int] = []
    out: List[str] = []
    for c in chunks:
        h = _simhash(c)
        if all(_hamm(h, s) > max_sim for s in seen):
            seen.append(h); out.append(c)
    return out

# ── Ranking ──────────────────────────────────────────────────────────────
def _kw_score(text: str, query: str) -> float:
    if not text or not query: return 0.0
    q = re.findall(r"[a-zA-Ząćęłńóśźż0-9]{3,}", query.lower())
    if not q: return 0.0
    t = text.lower()
    hits = sum(t.count(w) for w in q)
    uniq = len(set(q))
    length_pen = 1.0 / (1.0 + math.log10(1 + max(0, len(text) - 900) / 450.0))
    return (hits / max(1, uniq)) * length_pen

def _rank_chunks(chunks: List[str], query: str, topk: int) -> List[Tuple[str,float]]:
    if not chunks: return []
    if M and hasattr(M, "rank_hybrid"):
        ranked = M.rank_hybrid(chunks, query, topk=min(len(chunks), max(3, topk)))
        return [(c, float(s)) for (c,s) in ranked]
    scored = [(c, _kw_score(c, query)) for c in chunks]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:topk]

# ── Tagowanie ────────────────────────────────────────────────────────────
def _tag_guess(text: str, query: str, url: str) -> str:
    tags = []
    if query:
        tags.extend([w for w in re.findall(r"[a-zA-Ząćęłńóśźż0-9]{4,}", query.lower())][:3])
    d = _domain(url)
    if d: tags.append(d.split(":")[0])
    if re.search(r"\b(api|endpoint|http|json|fastapi|flask|django)\b", text, re.I): tags.append("dev")
    if re.search(r"\bmoda|trend|kolekcja|capsule|wardrobe\b", text, re.I): tags.append("moda")
    if re.search(r"\bpaper|arxiv|doi|citation|badanie|research\b", text, re.I): tags.append("research")
    tags = list(dict.fromkeys(tags))
    return ",".join(tags)

# ── SOURCES ──────────────────────────────────────────────────────────────
def _serpapi(q: str, n: int=10) -> List[Dict[str,str]]:
    if not SERPAPI_KEY: return []
    url = "https://serpapi.com/search.json"
    try:
        with _client() as c:
            r = _with_retry(c.get, 3, 0.6, url, params={"q": q, "hl":"pl", "num": n, "api_key": SERPAPI_KEY})
            j = r.json()
            out=[]
            for it in (j.get("organic_results") or []):
                u = it.get("link") or ""
                if u.startswith("http"):
                    out.append({"title": it.get("title",""), "url": u, "snippet": it.get("snippet","")})
            return out
    except Exception:
        return []

def _ddg(q: str, n: int=10) -> List[Dict[str,str]]:
    try:
        with _client() as c:
            r = _with_retry(c.get, 3, 0.6, "https://duckduckgo.com/html/", params={"q": q})
            soup = BeautifulSoup(r.text, "lxml")
            res=[]
            for a in soup.select(".result__a")[:n]:
                href=a.get("href") or ""
                if href.startswith("http"):
                    res.append({"title": a.get_text(strip=True), "url": href, "snippet": ""})
            return res
    except Exception:
        return []

def _firecrawl_search(q: str, n: int=8) -> List[Dict[str,str]]:
    if not FIRECRAWL_KEY: return []
    try:
        with _client() as c:
            r = _with_retry(c.post, 3, 0.7, "https://api.firecrawl.dev/v1/search",
                            json={"query": q, "num_results": n, "lang":"pl"},
                            headers={"Authorization": f"Bearer {FIRECRAWL_KEY}"})
            j = r.json()
            out=[]
            for it in (j.get("results") or []):
                u=it.get("url") or ""
                if u.startswith("http"):
                    out.append({"title": it.get("title",""), "url": u, "snippet": it.get("snippet","")})
            return out
    except Exception:
        return []

def _firecrawl_scrape(url: str) -> str:
    if not FIRECRAWL_KEY: return ""
    try:
        with _client() as c:
            r = _with_retry(c.post, 3, 0.7, "https://api.firecrawl.dev/v1/scrape",
                            json={"url": url, "formats": ["markdown","html"]},
                            headers={"Authorization": f"Bearer {FIRECRAWL_KEY}"})
            if r.status_code != 200: return ""
            j=r.json() or {}
            d=j.get("data") or {}
            txt = d.get("markdown") or d.get("content") or d.get("html") or ""
            if isinstance(txt, list): txt="\n\n".join([t for t in txt if isinstance(t,str)])
            return _clean_text(txt)[:AUTO_MAX_CHARS]
    except Exception:
        return ""

def _http_fetch(url: str) -> str:
    try:
        with _client() as c:
            r = _with_retry(c.get, 3, 0.6, url)
            if r.status_code == 200:
                return _clean_text(r.text)[:AUTO_MAX_CHARS]
    except Exception:
        return ""
    return ""

# ── MAIN ────────────────────────────────────────────────────────────────
def autonauka(query: str, topk: int = AUTO_TOPK, deep_research: bool = False, use_external_module: bool = True) -> Dict[str, Any]:
    t0 = time.time()
    q = (query or "").strip()
    if not q:
        return {"ok": True, "query": query, "context": "", "facts": [], "sources": [], "is_deep_research": deep_research, "source_count": 0, "powered_by":"autonauka-pro2"}

    # 1) agregacja kandydatów
    src: List[Dict[str,str]] = []
    src += _serpapi(q, 12)
    src += _ddg(q, 12)
    src += _firecrawl_search(q, 10)

    if deep_research:
        year = time.gmtime().tm_year
        variants = [
            f"{q} najlepsze praktyki",
            f"{q} przykłady",
            f"{q} zastosowania",
            f"{q} trendy {year}",
            f"{q} case study",
        ]
        for v in variants:
            src += _serpapi(v, 10)
            src += _ddg(v, 10)

    # 2) deduplikacja URL + limit domen
    seen, per_domain, uniq = set(), {}, []
    for it in src:
        u = _canonical_url(it.get("url",""))
        if not u or u in seen: continue
        d = _domain(u)
        if not d: continue
        per_domain[d] = per_domain.get(d,0) + 1
        if per_domain[d] > AUTON_DOMAIN_MAX:
            continue
        seen.add(u)
        it["url"] = u
        uniq.append(it)
        if len(uniq) >= AUTON_DEDUP_MAX: break

    # 3) pobranie treści
    items: List[Dict[str,Any]] = []
    for it in uniq[:max(AUTO_FETCH, topk*2)]:
        u = it["url"]
        body = _firecrawl_scrape(u) or _http_fetch(u)
        if not body or len(body) < AUTO_MIN_CHARS:
            continue
        items.append({"url": u, "title": it.get("title",""), "text": body, "snippet": it.get("snippet","")})

    if not items:
        return {"ok": True, "query": q, "context": "", "facts": [], "sources": [], "is_deep_research": deep_research, "source_count": 0, "powered_by":"autonauka-pro2"}

    # 4) ekstrakcja cytatów + chunkowanie, dedup chunków (SimHash)
    per_source_best: List[Tuple[str,float,str]] = []  # (chunk,score,url)
    for it in items:
        url = it["url"]
        quotes = _extract_quotes(it["text"])
        base_chunks = quotes + _chunk_text(it["text"], target_chars=900, max_chars=1400)
        base_chunks = _dedup_chunks(base_chunks, max_sim=8)
        ranked = _rank_chunks(base_chunks, q, topk=min(5, max(3, topk//2)))
        for c,s in ranked:
            per_source_best.append((c,s,url))

    # 5) globalny ranking + dedup między-źródłowy
    per_source_best.sort(key=lambda x: x[1], reverse=True)
    global_chunks, seen_sim = [], []
    for c,s,u in per_source_best:
        h = _simhash(c)
        if all(_hamm(h, hs) > 9 for hs in seen_sim):
            seen_sim.append(h)
            global_chunks.append((c,s,u))
        if len(global_chunks) >= topk:
            break

    if not global_chunks:  # fallback
        for c,s,u in per_source_best[:topk]:
            global_chunks.append((c,s,u))

    # 6) budowa źródeł i kontekstu z oznaczeniami [n]
    sources: List[Dict[str,str]] = []
    url_to_idx: Dict[str,int] = {}
    facts_texts: List[str] = []

    for c,_,u in global_chunks:
        if u not in url_to_idx:
            url_to_idx[u] = len(sources) + 1
            sources.append({"title":"", "url": u})
        idx = url_to_idx[u]
        facts_texts.append(f"{c} [{idx}]")

    context = "\n\n---\n\n".join(facts_texts)

    # 7) zapis do LTM
    if SAVE_TO_LTM and M and hasattr(M, "ltm_add"):
        for c,_,u in global_chunks:
            try:
                M.ltm_add(c, tags=_tag_guess(c, q, u), conf=0.7)
            except Exception:
                pass

    return {
        "ok": True,
        "query": q,
        "context": context,          # z [n]
        "facts": [c for (c,_,_) in global_chunks],
        "sources": sources,          # indeksowane od 1
        "is_deep_research": deep_research,
        "source_count": len(items),
        "powered_by": "autonauka-pro2",
        "elapsed_s": round(time.time()-t0, 2),
        "diagnostics": {
            "candidates": len(src),
            "uniq_urls": len(uniq),
            "fetched": len(items),
            "chunks_kept": len(global_chunks),
        }
    }
