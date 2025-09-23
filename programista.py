"""programista.py — PRO asystent kodowania
+ API + trening Języki:
C/C++/JS/Java/Python/curl/bash Nowe: -
LANG_SOURCES: 5 stałych źródeł na język
(samouczek z webu). - learn_from_web(lang,
topic): pobiera z wybranych domen, destyluje
notatki. - run_quality(lang, code):
lint/format/statyczna analiza/testy (jeśli
binarki są). - FastAPI: + /prog/shell
(biała lista komend) jako „terminal”.
Reszta: research
(SerpAPI/Firecrawl/Wiki/DDG), repo-ops,
runner, plugins, FACT-LOCK.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

ROOT = Path(os.getenv("PROG_ROOT", "/workspace/mrd69"))
OUT = ROOT / "out" / "prog"
OUT.mkdir(parents=True, exist_ok=True)
EXEC_DIR = OUT / "_exec"
EXEC_DIR.mkdir(parents=True, exist_ok=True)
_HAS_MEM = False
memory = None
try:
    from . import memory as _mem

    memory = _mem
    _HAS_MEM = True
except Exception:
    pass
_plugins: dict[str, Any] = {}


def _try_import() -> None:
    for m in ["images_client", "travelguide", "crypto_advisor_full", "file_client"]:
        try:
            _plugins[m] = __import__(f"{__package__}.{m}", fromlist=["*"])
        except Exception:
            try:
                _plugins[m] = __import__(m)
            except Exception:
                _plugins[m] = None


_try_import()
import requests  # noqa: E402

WEB_TIMEOUT = int(os.getenv("WEB_HTTP_TIMEOUT", os.getenv("TIMEOUT_HTTP", "25")))
UA = os.getenv("WEB_USER_AGENT", "Overmind/ProgramistaPRO/2.0")
S = requests.Session()
S.headers.update({"User-Agent": UA})
LLM_BASE = (os.getenv("LLM_BASE_URL") or "https://api.deepinfra.com/v1/openai").rstrip(
    "/"
)
LLM_KEY = (os.getenv("LLM_API_KEY") or "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct")
MINI_BASE = (os.getenv("MINI_LLM_BASE_URL") or LLM_BASE).rstrip("/")
MINI_KEY = (os.getenv("MINI_LLM_API_KEY") or os.getenv("LLM_API_KEY") or "").strip()
MINI_MODEL = os.getenv("MINI_LLM_MODEL", "Qwen/Qwen2.5-4B-Instruct")


def _chat(
    base: str,
    key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.25,
    max_tokens: int = 1600,
) -> str:
    if not key:
        return "\n".join(
            [m.get("content", "") for m in messages if m.get("role") == "user"]
        )
    try:
        r = S.post(
            base + "/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=max(10, WEB_TIMEOUT),
        )
        if r.status_code < 400:
            j = r.json()
            return (
                ((j.get("choices") or [{}])[0].get("message") or {})
                .get("content", "")
                .strip()
            )
    except Exception:
        pass
    return ""


def llm72(sysmsg: str, usermsg: str, temp: float = 0.28, max_t: int = 2600) -> str:
    return _chat(
        LLM_BASE,
        LLM_KEY,
        LLM_MODEL,
        [{"role": "system", "content": sysmsg}, {"role": "user", "content": usermsg}],
        temp,
        max_t,
    )


def llm4(prompt: str, temp: float = 0.18, max_t: int = 900) -> str:
    return _chat(
        MINI_BASE,
        MINI_KEY,
        MINI_MODEL,
        [{"role": "user", "content": prompt}],
        temp,
        max_t,
    )


SERPAPI_KEY = (os.getenv("SERPAPI_KEY") or "").strip()
FIRECRAWL_KEY = (os.getenv("FIRECRAWL_KEY") or "").strip()


def serpapi_search(q: str, num: int = 8) -> list[dict[str, str]]:
    if not SERPAPI_KEY:
        return []
    try:
        r = S.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google",
                "q": str(q),
                "num": str(int(max(1, min(10, num)))),
                "api_key": str(SERPAPI_KEY),
            },
            timeout=WEB_TIMEOUT,
        )
        if r.status_code >= 400:
            return []
        j = r.json()
        outs: list[dict[str, str]] = []
        for it in j.get("organic_results") or []:
            u = it.get("link") or ""
            if u:
                outs.append(
                    {
                        "url": u,
                        "title": it.get("title", ""),
                        "snippet": it.get("snippet", ""),
                    }
                )
        return outs[:num]
    except Exception:
        return []


def firecrawl_scrape(url: str, max_chars: int = 20000) -> str:
    if not FIRECRAWL_KEY:
        return ""
    try:
        r = S.post(
            "https://api.firecrawl.dev/v1/scrape",
            headers={
                "Authorization": f"Bearer {FIRECRAWL_KEY}",
                "Content-Type": "application/json",
            },
            json={"url": url, "formats": ["markdown", "html", "text"]},
            timeout=WEB_TIMEOUT,
        )
        if r.status_code >= 400:
            return ""
        j = (
            r.json()
            if r.headers.get("content-type", "").startswith("application/json")
            else {}
        )
        t = j.get("markdown") or j.get("text") or j.get("html") or ""
        return re.sub(r"\s+", " ", str(t)).strip()[:max_chars]
    except Exception:
        return ""


def _ddg_html(q: str) -> str:
    try:
        r = S.get("https://duckduckgo.com/html/", params={"q": q}, timeout=WEB_TIMEOUT)
        return r.text if r.status_code < 400 else ""
    except Exception:
        return ""


def ddg_links(q: str, k: int = 6) -> list[str]:
    html = _ddg_html(q)
    urls = re.findall(r'href="(https?://[^"]+)"', html)
    outs: list[str] = []
    seen: set[str] = set()
    for u in urls:
        if "duckduckgo.com" in u or "ad_provider" in u:
            continue
        u = u.split("&")[0]
        if u in seen:
            continue
        seen.add(u)
        outs.append(u)
        if len(outs) >= k:
            break
    return outs


def wiki_summary(title: str, lang: str = "pl") -> dict[str, str]:
    try:
        u = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
        r = S.get(u, timeout=WEB_TIMEOUT)
        if r.status_code == 404:
            return {}
        r.raise_for_status()
        d = r.json()
        return {
            "title": d.get("title", ""),
            "extract": d.get("extract", ""),
            "url": ((d.get("content_urls") or {}).get("desktop") or {}).get("page", ""),
        }
    except Exception:
        return {}


def collect_facts(query: str, k: int = 8) -> dict[str, Any]:
    hits = serpapi_search(query, num=k) if SERPAPI_KEY else []
    links = [h["url"] for h in hits if h.get("url")]
    if len(links) < max(3, k // 2):
        links += [u for u in ddg_links(query, k=k) if u not in links]
    wiki = wiki_summary(query, "pl") or wiki_summary(query, "en")
    excerpts: list[str] = []
    for u in links[:4]:
        txt = firecrawl_scrape(u, max_chars=12000) if FIRECRAWL_KEY else ""
        if not txt:
            try:
                r = S.get(u, timeout=WEB_TIMEOUT)
                if r.status_code < 400:
                    t = r.text
                    t = re.sub(r"(?is)<script.*?>.*?</script>", " ", t)
                    t = re.sub(r"(?is)<style.*?>.*?</style>", " ", t)
                    t = re.sub(r"(?is)<[^>]+>", " ", t)
                    t = re.sub(r"\s+", " ", t).strip()
                    txt = t[:12000]
            except Exception:
                txt = ""
        if txt:
            excerpts.append(f"[{u}]\n{txt[:3500]}")
    # WAŻNE: żadnych backslashy w
    # wyrażeniu f-stringa — przygotowujemy
    # tekst wcześniej.
    excerpts_block = "\n\n".join(excerpts) if excerpts else "(brak)"
    wiki_url = wiki.get("url", "") or "(brak)"
    wiki_extract = (wiki.get("extract", "") or "")[:420]
    prm = (
        "SUCHY BRIEF (PL). ZERO opinii.\n"
        "FACTS:\n"
        "- ...\n"
        "LINKS:\n"
        "- <URL>\n"
        "RRRRRRRRRÓDŁA:\n"
        f"{excerpts_block}\n"
        f"Wikipedia: {wiki_url} — {wiki_extract}"
    )
    facts = llm4(prm, temp=0.12, max_t=700)
    if "LINKS:" not in (facts or ""):
        facts = "FACTS:\n- (brak pewnych faktów)\n\nLINKS:\n" + "\n".join(
            f"- {u}" for u in links
        )
    return {"facts": facts, "links": links}


# ======= SAMOUK: stałe źródła per język
# =======
LANG_SOURCES: dict[str, list[str]] = {
    "python": [
        "docs.python.org",
        "peps.python.org",
        "realpython.com",
        "pydantic.dev",
        "fastapi.tiangolo.com",
    ],
    "javascript": [
        "developer.mozilla.org",
        "nodejs.org",
        "tc39.es",
        "eslint.org",
        "typescriptlang.org",
    ],
    "java": [
        "docs.oracle.com",
        "openjdk.org",
        "baeldung.com",
        "spring.io",
        "checkstyle.sourceforge.io",
    ],
    "c": [
        "en.cppreference.com",
        "open-std.org",
        "gcc.gnu.org",
        "clang.llvm.org",
        "c-faq.com",
    ],
    "cpp": [
        "en.cppreference.com",
        "isocpp.org",
        "gcc.gnu.org",
        "clang.llvm.org",
        "abseil.io",
    ],
    "bash": [
        "www.gnu.org/software/bash",
        "man7.org",
        "pubs.opengroup.org",
        "wiki.bash-hackers.org",
        "shellcheck.net",
    ],
}


def _match_lang(key: str) -> str:
    k = key.lower()
    if k in ("c++", "cpp", "cxx"):
        return "cpp"
    if k in ("js", "node", "javascript"):
        return "javascript"
    return {
        "py": "python",
        "python": "python",
        "java": "java",
        "c": "c",
        "cpp": "cpp",
        "bash": "bash",
    }.get(k, k)


def learn_from_web(lang: str, topic: str, k: int = 6) -> str:
    """Zbiera z preferowanych domen dla danego języka i destyluje notatki."""
    lang = _match_lang(lang)
    domains = LANG_SOURCES.get(lang, [])
    query = f"{topic} {lang}"
    links: list[str] = []
    # SERP
    if SERPAPI_KEY:
        for d in domains:
            hits = serpapi_search(f"site:{d} {query}", num=3)
            for h in hits:
                u = h.get("url")
                if u and u not in links:
                    links.append(u)
    # DDG fallback
    if len(links) < k:
        for d in domains:
            for u in ddg_links(f"site:{d} {query}", k=2):
                if u not in links:
                    links.append(u)
                if len(links) >= k:
                    break
            if len(links) >= k:
                break
    # scrapuj
    chunks: list[str] = []
    for u in links[:k]:
        tx = firecrawl_scrape(u, max_chars=10000) if FIRECRAWL_KEY else ""
        if tx:
            chunks.append(f"[{u}]\n{tx[:3000]}")
    corp = "\n\n".join(chunks) if chunks else "(brak treści)"
    prompt = (
        f"Język: {lang}. Temat: {topic}.\n"
        "Wyciągnij SKRÓT do nauki:\n"
        "- definicje\n"
        "- najważniejsze wzorce/pułapki\n"
        "- mini przykłady (po 4-10 linii)\n"
        "- checklista „zastosuj w kodzie”\n"
        "Dodaj listę linków.\n"
        f"RRRRRRRRRÓDŁA:\n{corp}"
    )
    notes = llm72(
        "Mentor kodu. Zwięźle, technicznie, po polsku.", prompt, temp=0.22, max_t=1800
    )
    return notes or "Brak notatek"


# ======= Repo narzędzia =======
IGNORE_DIRS = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
}
CODE_EXT = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".md",
    ".txt",
    ".sh",
    ".cfg",
    ".ini",
    ".c",
    ".cc",
    ".cpp",
    ".java",
    ".bash",
}


def walk_files(root: str | Path) -> list[Path]:
    root = Path(root)
    files: list[Path] = []
    for dp, _dn, fn in os.walk(root):
        if Path(dp).name in IGNORE_DIRS:
            continue
        for name in fn:
            p = Path(dp) / name
            if p.suffix.lower() in CODE_EXT:
                files.append(p)
    return files


def read_text(p: str | Path) -> str:
    try:
        return Path(p).read_text(encoding="utf-8")
    except Exception:
        return ""


def write_text(p: str | Path, txt: str) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt, encoding="utf-8")


def backup_file(p: Path) -> Path:
    bak = p.with_suffix(p.suffix + ".bak")
    shutil.copy2(p, bak)
    return bak


def unified_diff(a: str, b: str, name: str) -> str:
    return "".join(
        difflib.unified_diff(
            a.splitlines(True),
            b.splitlines(True),
            fromfile=name,
            tofile=name,
            lineterm="",
        )
    )


def repo_map(root: str | Path) -> dict[str, Any]:
    files = walk_files(root)
    stats = {"files": len(files), "by_ext": {}, "lines": 0}
    top: list[dict[str, Any]] = []
    for p in files:
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
            ln = t.count("\n") + 1
            stats["lines"] += ln
            stats["by_ext"][p.suffix] = stats["by_ext"].get(p.suffix, 0) + 1
            top.append({"path": str(p), "lines": ln})
        except Exception:
            pass
    return {"stats": stats, "top": sorted(top, key=lambda x: -x["lines"])[:80]}


def scan_todos(root: str | Path) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for p in walk_files(root):
        try:
            for i, ln in enumerate(
                p.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1
            ):
                if re.search(r"\b(TODO|FIXME|BUG)\b", ln):
                    out.append(
                        {"path": str(p), "line": str(i), "text": ln.strip()[:400]}
                    )
        except Exception:
            pass
    return out


def search_symbol(root: str | Path, sym: str) -> list[dict[str, Any]]:
    rx = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(sym)}(?![A-Za-z0-9_])")
    hits: list[dict[str, Any]] = []
    for p in walk_files(root):
        try:
            for i, ln in enumerate(
                p.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1
            ):
                if rx.search(ln):
                    hits.append(
                        {"path": str(p), "line": i, "preview": ln.strip()[:200]}
                    )
        except Exception:
            pass
    return hits


# ======= Code-ops (LLM) =======
def review_code(text: str, filename: str = "snippet.py") -> str:
    sysm = "Senior reviewer. BŁĘDY, RYZYKA, POPRAWKI, TESTY. Konkretne przykłady."
    prm = (
        f"PLIK: {filename}\n---\n{text}\n---\n"
        "Audyt, bez lania wody. Dodaj propozycje poprawek linia/fragment."
    )
    return llm72(sysm, prm, temp=0.26, max_t=1600)


def propose_fix(text: str, filename: str = "file.py") -> str:
    sysm = (
        "Refaktorysta. Zwróć komplet pliku po poprawkach. Idiomatic. "
        "Wyjątki opisane. Komentarze tylko krytyczne."
    )
    prm = (
        f"Plik: {filename}\n<<<CODE>>>\n{text}\n<<<END>>>\n"
        "Wyjście: pełny plik po poprawkach. Tylko kod."
    )
    return llm72(sysm, prm, temp=0.26, max_t=2400)


def quick_fix_file(path: str) -> dict[str, Any]:
    p = Path(path)
    src = read_text(p)
    if not src.strip():
        return {"ok": False, "error": "empty_or_missing"}
    fixed = propose_fix(src, filename=p.name).strip()
    if not fixed or len(fixed) < max(40, int(len(src) * 0.3)):
        return {"ok": False, "error": "weak_fix"}
    if fixed == src:
        return {"ok": True, "changed": False, "path": str(p)}
    bak = backup_file(p)
    write_text(p, fixed)
    diff = unified_diff(src, fixed, str(p))
    _log_mem(f"[FIX]{p.name}", {"diff": diff[:40000]})
    return {
        "ok": True,
        "changed": True,
        "path": str(p),
        "backup": str(bak),
        "diff": diff[:40000],
    }


def gen_tests(path: str, framework: str = "pytest") -> dict[str, Any]:
    src = read_text(path)
    if not src.strip():
        return {"ok": False, "error": "empty"}
    out = llm72(
        "Generator testów. Minimalne ale sensowne pokrycie. "
        "Edge cases. Stabilne asercje.",
        f"Framework: {framework}\nStwórz plik testów do kodu:\n---\n{src}\n"
        f"---\nZwróć pełny plik testów.",
        temp=0.26,
        max_t=2000,
    )
    name = test_filename(path, framework)
    write_text(name, out.strip())
    _log_mem(f"[TESTS]{Path(path).name}", {"test_file": name})
    return {"ok": True, "test_file": name}


def test_filename(path: str, framework: str) -> str:
    p = Path(path)
    if framework == "pytest":
        return str(p.parent / f"test_{p.stem}.py")
    return str(p.parent / f"{p.stem}_test.py")


def rename_symbol(path: str, old: str, new: str) -> dict[str, Any]:
    patt = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])")
    src = read_text(path)
    if not src.strip():
        return {"ok": False, "error": "empty"}
    changed = patt.sub(new, src)
    if changed == src:
        return {"ok": True, "changed": False, "path": path}
    bak = backup_file(Path(path))
    write_text(path, changed)
    diff = unified_diff(src, changed, path)
    _log_mem(f"[RENAME]{Path(path).name}", {"old": old, "new": new})
    return {
        "ok": True,
        "changed": True,
        "path": path,
        "backup": str(bak),
        "diff": diff[:40000],
    }


def inject_docstrings(path: str) -> dict[str, Any]:
    src = read_text(path)
    if not src.strip():
        return {"ok": False, "error": "empty"}
    out = llm72(
        (
            "Doc-wizard. Dodaj docstringi do modułu/klas/funkcji. "
            "Argumenty, zwroty, wyjątki. Nie zmieniaj logiki."
        ),
        (
            f"Plik:\n<<<CODE>>>\n{src}\n<<<END>>>\n"
            "Zwróć kompletny plik z docstringami. Tylko kod."
        ),
        temp=0.24,
        max_t=2400,
    )
    if not out.strip():
        return {"ok": False, "error": "llm_fail"}
    bak = backup_file(Path(path))
    write_text(path, out.strip())
    diff = unified_diff(src, out, path)
    _log_mem(f"[DOCS]{Path(path).name}", {"len": len(out)})
    return {"ok": True, "path": path, "backup": str(bak), "diff": diff[:40000]}


# ======= Runner =======
def detect_lang(code: str, filename: str | None = None) -> str:
    if filename:
        ext = Path(filename).suffix.lower()
        return {
            ".py": "python",
            ".js": "js",
            ".mjs": "js",
            ".c": "c",
            ".cc": "cpp",
            ".cpp": "cpp",
            ".java": "java",
            ".sh": "bash",
            ".bash": "bash",
            ".curl": "curl",
        }.get(ext, "auto")
    if re.search(r"^\s*#include\s+<", code, re.M):
        return "c" if "std::" not in code else "cpp"
    if "public class " in code or "static void main" in code:
        return "java"
    if re.search(r"^\s*def\s+\w+\(", code, re.M):
        return "python"
    if re.search(r"^\s*console\.log\(", code, re.M):
        return "js"
    if re.search(r"^\s*#!/bin/bash", code):
        return "bash"
    if code.strip().startswith("curl "):
        return "curl"
    return "python"


def _run(cmd: list[str], cwd: Path, timeout: int = 10) -> tuple[int, str, str]:
    try:
        p = subprocess.run(
            cmd,
            check=False,
            cwd=str(cwd),
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        return (p.returncode, p.stdout, p.stderr)
    except subprocess.TimeoutExpired:
        return (124, "", "timeout")
    except Exception as e:
        return (127, "", f"{e}")


def run_code(
    code: str,
    lang: str | None = None,
    filename: str | None = None,
    args: list[str] | None = None,
    timeout: int = 10,
) -> dict[str, Any]:
    if os.getenv("EXEC_ENABLE", "0") not in ("1", "true", "TRUE", "yes", "on"):
        return {"ok": False, "error": "exec_disabled"}
    lang = lang or detect_lang(code, filename)
    args = args or []
    work = Path(tempfile.mkdtemp(prefix="exec_", dir=str(EXEC_DIR)))
    if lang == "python":
        fn = work / "main.py"
        write_text(fn, code)
        rc, out, err = _run(["python3", str(fn), *args], work, timeout)
    elif lang == "js":
        fn = work / "main.js"
        write_text(fn, code)
        rc, out, err = _run(["node", str(fn), *args], work, timeout)
    elif lang == "java":
        fn = work / "Main.java"
        if "class Main" in code:
            write_text(fn, code)
        else:
            write_text(fn, code.replace("class ", "class Main ", 1))
        rc1, out1, err1 = _run(["javac", str(fn)], work, max(10, timeout))
        rc, out, err = (
            _run(["java", "Main", *args], work, timeout)
            if rc1 == 0
            else (rc1, out1, err1)
        )
    elif lang == "c":
        fn = work / "main.c"
        write_text(fn, code)
        rc1, out1, err1 = _run(
            [
                "bash",
                "-lc",
                (
                    f"gcc -std=c11 -O2 -Wall -Wextra {fn} -o main || "
                    f"gcc -std=c11 -O2 {fn} -o main"
                ),
            ],
            work,
            max(10, timeout),
        )
        rc, out, err = (
            _run(["./main", *args], work, timeout) if rc1 == 0 else (rc1, out1, err1)
        )
    elif lang == "cpp":
        fn = work / "main.cpp"
        write_text(fn, code)
        rc1, out1, err1 = _run(
            [
                "bash",
                "-lc",
                (
                    f"g++ -std=c++20 -O2 -Wall -Wextra {fn} -o main || "
                    f"g++ -std=c++20 -O2 {fn} -o main"
                ),
            ],
            work,
            max(10, timeout),
        )
        rc, out, err = (
            _run(["./main", *args], work, timeout) if rc1 == 0 else (rc1, out1, err1)
        )
    elif lang == "bash":
        fn = work / "run.sh"
        write_text(fn, code)
        os.chmod(fn, 0o755)
        rc, out, err = _run(["bash", str(fn), *args], work, timeout)
    elif lang == "curl":
        cmd = (
            ["bash", "-lc", code]
            if code.strip().startswith("curl ")
            else ["curl"] + args
        )
        rc, out, err = _run(cmd, work, timeout)
    else:
        return {"ok": False, "error": f"unsupported:{lang}"}
    return {
        "ok": True,
        "lang": lang,
        "rc": rc,
        "stdout": out,
        "stderr": err,
        "workdir": str(work),
    }


# ======= QUALITY: lint/format/analyze/test
# per język =======
def _bin(name: str) -> str | None:
    for p in os.getenv("PATH", "").split(os.pathsep):
        fp = Path(p) / name
        if fp.exists() and os.access(fp, os.X_OK):
            return str(fp)
    return None


def run_quality(lang: str, code: str) -> dict[str, Any]:
    """
    Tworzy sandbox, zapisuje plik i odpala dostępne narzędzia.
    Brakujące ignoruje.
    """
    work = Path(tempfile.mkdtemp(prefix="quality_", dir=str(EXEC_DIR)))
    res: dict[str, Any] = {"workdir": str(work), "steps": []}

    def step(name: str, cmd: list[str]) -> None:
        rc, out, err = _run(cmd, work, timeout=20)
        res["steps"].append(
            {"name": name, "rc": rc, "stdout": out[-8000:], "stderr": err[-8000:]}
        )

    if _match_lang(lang) == "python":
        fn = work / "app.py"
        write_text(fn, code)
        if _bin("black"):
            step("black-check", ["black", "--check", str(fn)])
        if _bin("ruff"):
            step("ruff", ["ruff", "check", str(fn)])
        if _bin("flake8"):
            step("flake8", ["flake8", str(fn)])
        if _bin("mypy"):
            step("mypy", ["mypy", str(fn)])
        if _bin("pytest"):
            write_text(work / "test_smoke.py", "def test_smoke():\n    assert True\n")
            step("pytest", ["pytest", "-q"])
    elif _match_lang(lang) == "javascript":
        fn = work / "app.js"
        write_text(fn, code)
        if _bin("node") and os.name != "nt":
            step("node-syntax", ["node", "-c", str(fn)])
        if _bin("eslint"):
            step("eslint", ["eslint", str(fn)])
        if _bin("prettier"):
            step("prettier-check", ["prettier", "--check", str(fn)])
        if _bin("tsc"):
            step("tsc", ["tsc", "--noEmit"])
    elif _match_lang(lang) == "java":
        fn = work / "Main.java"
        if "class Main" in code:
            write_text(fn, code)
        else:
            write_text(fn, code.replace("class ", "class Main ", 1))
        if _bin("javac"):
            step("javac-lint", ["javac", "-Xlint", str(fn)])
        if _bin("checkstyle"):
            step(
                "checkstyle",
                ["checkstyle", "-c", "/google_checks.xml", str(fn)],
            )
    elif _match_lang(lang) == "c":
        fn = work / "main.c"
        write_text(fn, code)
        if _bin("gcc"):
            step(
                "gcc-warnings",
                [
                    "bash",
                    "-lc",
                    f"gcc -std=c11 -Wall -Wextra -fanalyzer {fn} -o a.out " "|| true",
                ],
            )
        if _bin("cppcheck"):
            step(
                "cppcheck",
                [
                    "cppcheck",
                    "--enable=all",
                    "--suppress=missingIncludeSystem",
                    str(fn),
                ],
            )
        if _bin("clang-tidy"):
            step("clang-tidy", ["clang-tidy", str(fn), "--"])
    elif _match_lang(lang) == "cpp":
        fn = work / "main.cpp"
        write_text(fn, code)
        if _bin("g++"):
            step(
                "g++-warnings",
                [
                    "bash",
                    "-lc",
                    f"g++ -std=c++20 -Wall -Wextra {fn} -o a.out || true",
                ],
            )
        if _bin("cppcheck"):
            step(
                "cppcheck",
                ["cppcheck", "--enable=all", "--language=c++", str(fn)],
            )
        if _bin("clang-tidy"):
            step("clang-tidy", ["clang-tidy", str(fn), "--"])
    elif _match_lang(lang) == "bash":
        fn = work / "run.sh"
        write_text(fn, code)
        os.chmod(fn, 0o755)
        if _bin("shellcheck"):
            step("shellcheck", ["shellcheck", str(fn)])
        step("bash -n", ["bash", "-n", str(fn)])
    else:
        return {"ok": False, "error": "unsupported"}
    return {"ok": True, **res}


# ======= Plan + reply =======
@dataclass
class PlanStep:
    title: str
    detail: str


@dataclass
class ProgReply:
    text: str
    actions: dict[str, Any] = field(default_factory=dict)
    facts: str = ""
    links: list[str] = field(default_factory=list)


def make_plan(user_msg: str, repo_hint: dict[str, Any] | None = None) -> list[PlanStep]:
    hint = json.dumps(repo_hint or {})[:1200]
    raw = llm4(
        (
            f"Zaplanuj realizację zadania dev. Krótko. Kontekst:{hint}\n"
            f"ZADANIE:{user_msg}\n4–7 kroków: tytuł: jedno zdanie."
        ),
        temp=0.2,
        max_t=480,
    )
    steps: list[PlanStep] = []
    for ln in (raw or "").splitlines():
        m = re.match(r"[-*\d\.\)]?\s*(.+?)\s*[:\-–]\s*(.+)", ln)
        if m:
            steps.append(PlanStep(m.group(1).strip()[:80], m.group(2).strip()[:240]))
    return steps[:8] or [
        PlanStep("Analiza", "Zbadaj wymagania i kod."),
        PlanStep("Implementacja", "Wprowadź zmiany."),
        PlanStep("Testy", "Dodaj i uruchom testy."),
    ]


_FACT_LOCK = re.compile(
    r"\\b(wynik|wyniki|mecz|score|kurs|kursy|notowania|ile "
    r"kosztuje|cena|ceny|rating|ocena|gwiazdek|michelin|rezerwacj|terminarz|kiedy "
    r"gra)\\b",
    re.I,
)


def fact_lock(msg: str) -> bool:
    return bool(_FACT_LOCK.search(msg or ""))


_INTENT = {
    "image": re.compile(
        r"\\b(narysuj|wygeneruj|obraz|grafik[ae]|zdjęci[ea]|render)\\b", re.I
    ),
    "travel": re.compile(
        r"\\b(plan|itinerar|zwiedz|hotel|lot|restaurac|bilety|komunikac)\\b", re.I
    ),
    "crypto": re.compile(
        r"\\b(crypto|krypto|btc|eth|kurs|altcoin|token|wallet|defi|staking|price|notowani)\\b",
        re.I,
    ),
    "code": re.compile(
        r"\\b(kod|bug|błąd|refaktor|testy|moduł|funkcj|klas|import|traceback|exception|kompiluj|uruchom)\\b",
        re.I,
    ),
    "file": re.compile(
        r"\\b(pliki|wyślij plik|pobierz plik|zapisz plik|upload|download)\\b", re.I
    ),
    "learn": re.compile(r"\\b(ucz|samou[kc]|naucz|materiały|źródł[ao])\\b", re.I),
}


def detect_intent(msg: str) -> str:
    for k, rx in _INTENT.items():
        if rx.search(msg):
            return k
    return "general"


def _log_mem(title: str, payload: dict[str, Any]) -> None:
    if not _HAS_MEM or not memory:
        return
    try:
        # memory is guaranteed to be not None at this point
        if memory and hasattr(memory, "ltm_add_sync"):
            memory.ltm_add_sync(title, sources=[payload], user="global", tags=["prog"])
    except Exception:
        pass


def programista_reply(
    history: list[dict[str, str]], user_msg: str, repo_root: str | None = None
) -> ProgReply:
    intent = detect_intent(user_msg)
    f_lock = fact_lock(user_msg)
    if intent == "learn":
        m = re.search(
            r"(python|java|c\+\+|cpp|c|javascript|js|bash)\b.*?:\s*(.+)$",
            user_msg,
            re.I,
        )
        if m:
            lang = _match_lang(m.group(1))
            topic = m.group(2).strip()
            notes = learn_from_web(lang, topic, k=6)
            return ProgReply(text=notes, actions={"lang": lang, "topic": topic})
    if intent == "image" and _plugins.get("images_client"):
        try:
            p = _plugins["images_client"].generate_image(user_msg)
            return ProgReply(text=f"Obraz: {p}", actions={"image_path": p})
        except Exception:
            pass
    if intent == "travel" and _plugins.get("travelguide"):
        try:
            plan_txt = _plugins["travelguide"].generate_creative_plan(
                user_msg, days=None, prefs=None
            )
            facts_ctx = collect_facts(user_msg, k=6)
            final = llm72(
                "Asystent. Najpierw plan, potem fakty i linki.",
                f"PLAN:\n{plan_txt}\n\nFACTS:\n{facts_ctx.get('facts','')}\n\nPytanie:\n{user_msg}",
                0.3,
                2200,
            )
            return ProgReply(
                text=final.strip(),
                actions={"travel_plan": plan_txt},
                facts=facts_ctx.get("facts", ""),
                links=facts_ctx.get("links", []),
            )
        except Exception:
            pass
    if intent == "crypto" and _plugins.get("crypto_advisor_full"):
        try:
            ans = _plugins["crypto_advisor_full"].advise(user_msg)
            facts_ctx = collect_facts(user_msg, k=6)
            usr = (
                f"MATERIAŁ:\n{facts_ctx.get('facts','')}\n\n"
                f"DANE Z ADVISORA:\n{ans}\n\nPYTANIE:\n{user_msg}"
            )
            if f_lock:
                usr = (
                    f"FACTS (wyłączne źródło):\n{facts_ctx.get('facts','')}\n\n"
                    f"PYTANIE:\n{user_msg}"
                )
            final = llm72("Asystent krypto. FACT-LOCK dla cen.", usr, 0.28, 1800)
            if f_lock and not final.strip():
                final = "Brak jednoznacznych danych w FACTS."
            return ProgReply(
                text=final.strip(),
                actions={"advisor": bool(ans)},
                facts=facts_ctx.get("facts", ""),
                links=facts_ctx.get("links", []),
            )
        except Exception:
            pass
    if intent == "file" and _plugins.get("file_client"):
        try:
            m = re.search(r"zapisz plik\s+([^\s:]+)\s*:\s*(.+)$", user_msg, re.I)
            if m:
                name, body = m.group(1), m.group(2)
                if hasattr(_plugins["file_client"], "save_text"):
                    pth = _plugins["file_client"].save_text(name, body)
                else:
                    p = ROOT / "files" / name
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(body, encoding="utf-8")
                    pth = str(p)
                return ProgReply(text=f"Zapisano: {pth}", actions={"saved": pth})
        except Exception:
            pass
    # code run via message
    if intent == "code" and re.search(
        r"\b(uruchom|kompiluj|odpal|run)\b", user_msg, re.I
    ):
        m = re.search(
            r"uruchom\s+([a-z\+\#]+)\s*:\s*```(?:\w+)?\n(.+?)```", user_msg, re.I | re.S
        )
        if m:
            res = run_code(m.group(2), lang=m.group(1).lower(), timeout=12)
            return ProgReply(
                text=json.dumps(res, ensure_ascii=False, indent=2),
                actions={"run": True},
            )
        m = re.search(r"```(?:\w+)?\n(.+?)```", user_msg, re.S)
        if m:
            res = run_code(m.group(1), lang=None, timeout=12)
            return ProgReply(
                text=json.dumps(res, ensure_ascii=False, indent=2),
                actions={"run": True},
            )
    # general
    repo_info = repo_map(repo_root or str(ROOT))
    todos = scan_todos(repo_root or str(ROOT))
    plan = make_plan(
        user_msg,
        {
            "stats": repo_info.get("stats"),
            "top": repo_info.get("top")[:10],
            "todos": todos[:20],
        },
    )
    facts_ctx = collect_facts(user_msg, k=8)
    sysm = "Asystent programisty. Po polsku. Krótkie zdania. Najpierw FACTS. Potem plan i konkrety."
    if f_lock:
        sysm = "Asystent. FACT-LOCK. Odpowiadasz tylko na FACTS."
    usr = (
        f"HISTORIA:{json.dumps(history[-6:], ensure_ascii=False)[:1200]}\n\n"
        f"FACTS:\n{facts_ctx.get('facts','(brak)')}\n\n"
        "PLAN:\n"
        + "\n".join([f"- {s.title}: {s.detail}" for s in plan])
        + f"\n\nZADANIE:\n{user_msg}"
    )
    final = llm72(sysm, usr, 0.26 if f_lock else 0.3, 2400).strip()
    if _HAS_MEM and final and memory:
        try:
            # memory is guaranteed to be not None at this point
            if memory and hasattr(memory, "ltm_add_sync"):
                memory.ltm_add_sync(
                    f"[PROG]{'[FACT-LOCK]' if f_lock else ''} {user_msg[:72]}",
                    sources=[
                        {
                            "text": final,
                            "facts": facts_ctx.get("facts", ""),
                            "links": facts_ctx.get("links", []),
                        }
                    ],
                    user="global",
                    tags=["prog", "fact-lock" if f_lock else "web"],
                )
        except Exception:
            pass
    return ProgReply(
        text=final,
        actions={
            "repo_map": repo_info,
            "todos": todos[:50],
            "plan": [s.__dict__ for s in plan],
            "intent": intent,
        },
        facts=facts_ctx.get("facts", ""),
        links=facts_ctx.get("links", []),
    )


# ======= SHELL (API) =======
ALLOW_CMDS = {
    "python3",
    "node",
    "javac",
    "java",
    "gcc",
    "g++",
    "bash",
    "sh",
    "curl",
    "ls",
    "cat",
    "grep",
    "sed",
    "awk",
    "pytest",
    "flake8",
    "ruff",
    "black",
    "mypy",
    "eslint",
    "prettier",
    "tsc",
    "cppcheck",
    "clang-tidy",
    "shellcheck",
}


def run_shell(cmdline: str, timeout: int = 15) -> dict[str, Any]:
    if os.getenv("EXEC_ENABLE", "0") not in ("1", "true", "TRUE", "yes", "on"):
        return {"ok": False, "error": "exec_disabled"}
    toks = re.split(r"\\s+", cmdline.strip())
    if not toks:
        return {"ok": False, "error": "empty"}
    if Path(toks[0]).name not in ALLOW_CMDS:
        return {"ok": False, "error": "not_allowed"}
    work = Path(tempfile.mkdtemp(prefix="term_", dir=str(EXEC_DIR)))
    rc, out, err = _run(toks, work, timeout=timeout)
    return {
        "ok": True,
        "rc": rc,
        "stdout": out[-20000:],
        "stderr": err[-20000:],
        "workdir": str(work),
    }


# ======= CLI + FastAPI =======
def _stdin_or_file(path_or_dash: str) -> tuple[str, str]:
    if path_or_dash == "-":
        return ("stdin", sys.stdin.read())
    p = Path(path_or_dash)
    return (p.name, read_text(p))


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Programista PRO 2.0")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("reply")
    g.add_argument("msg")
    g.add_argument("--root", default=str(ROOT))
    g = sub.add_parser("run")
    g.add_argument("--lang", default=None)
    g.add_argument("--file", default=None)
    g.add_argument("--timeout", type=int, default=12)
    g.add_argument("--args", nargs="*", default=[])
    g.add_argument("code", nargs="?", default=None)
    g = sub.add_parser("review")
    g.add_argument("path")
    g = sub.add_parser("fix")
    g.add_argument("path")
    g = sub.add_parser("tests")
    g.add_argument("path")
    g.add_argument("--fw", default="pytest", choices=["pytest", "unittest"])
    g = sub.add_parser("rename")
    g.add_argument("path")
    g.add_argument("old")
    g.add_argument("new")
    g = sub.add_parser("docs")
    g.add_argument("path")
    g = sub.add_parser("quality")
    g.add_argument("lang")
    g.add_argument("path_or_dash")
    g = sub.add_parser("learn")
    g.add_argument("lang")
    g.add_argument("topic")
    g = sub.add_parser("api")
    args = ap.parse_args(argv)
    if args.cmd == "reply":
        res = programista_reply([], args.msg, repo_root=args.root)
        print(
            json.dumps(
                {
                    "text": res.text,
                    "links": res.links,
                    "plan": res.actions.get("plan"),
                    "facts": res.facts,
                    "intent": res.actions.get("intent"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.cmd == "run":
        code = (
            read_text(args.file)
            if (args.code is None and args.file)
            else (args.code or "")
        )
        res = run_code(
            code,
            lang=args.lang,
            filename=args.file,
            args=args.args,
            timeout=args.timeout,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "review":
        name, txt = _stdin_or_file(args.path)
        print(review_code(txt, name))
        return 0
    if args.cmd == "fix":
        print(json.dumps(quick_fix_file(args.path), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "tests":
        print(
            json.dumps(
                gen_tests(args.path, framework=args.fw),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.cmd == "rename":
        print(
            json.dumps(
                rename_symbol(args.path, args.old, args.new),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.cmd == "docs":
        print(json.dumps(inject_docstrings(args.path), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "quality":
        name, txt = _stdin_or_file(args.path_or_dash)
        print(json.dumps(run_quality(args.lang, txt), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "learn":
        print(learn_from_web(args.lang, args.topic))
        return 0
    if args.cmd == "api":
        import uvicorn
        from fastapi import FastAPI
        from pydantic import BaseModel

        app = FastAPI(title="Programista PRO API", version="2.0.0")

        class ReplyIn(BaseModel):
            history: list[dict[str, str]] = []
            msg: str
            root: str | None = None

        class RunIn(BaseModel):
            code: str
            lang: str | None = None
            filename: str | None = None
            args: list[str] | None = []
            timeout: int = 12

        class QualityIn(BaseModel):
            lang: str
            code: str

        class LearnIn(BaseModel):
            lang: str
            topic: str

        class ShellIn(BaseModel):
            cmdline: str
            timeout: int = 15

        @app.post("/prog/reply")
        def api_reply(d: ReplyIn):
            r = programista_reply(d.history, d.msg, repo_root=d.root or str(ROOT))
            return {
                "text": r.text,
                "links": r.links,
                "plan": r.actions.get("plan"),
                "facts": r.facts,
                "intent": r.actions.get("intent"),
            }

        @app.post("/prog/run")
        def api_run(d: RunIn):
            return run_code(
                d.code,
                lang=d.lang,
                filename=d.filename,
                args=d.args or [],
                timeout=d.timeout,
            )

        @app.post("/prog/quality")
        def api_quality(d: QualityIn):
            return run_quality(d.lang, d.code)

        @app.post("/prog/learn")
        def api_learn(d: LearnIn):
            return {"notes": learn_from_web(d.lang, d.topic)}

        @app.post("/prog/shell")
        def api_shell(d: ShellIn):
            return run_shell(d.cmdline, timeout=d.timeout)

        uvicorn.run(app, host="0.0.0.0", port=6969)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
