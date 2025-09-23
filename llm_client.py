from __future__ import annotations
import os, json, requests
from typing import List, Dict, Any, Optional

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=os.path.join("/workspace/mrd69", ".env"))
except Exception:
    pass

SECRETS_PATH = os.path.join("/workspace/mrd69", "secrets.json")


def _load_from_env() -> dict:
    base = (os.getenv("LLM_BASE_URL") or "").rstrip("/")
    key = os.getenv("LLM_API_KEY") or ""
    model = os.getenv("LLM_MODEL") or ""
    mini_base = (os.getenv("MINI_LLM_BASE_URL") or base).rstrip("/")
    mini_key = os.getenv("MINI_LLM_API_KEY") or key
    mini_model = os.getenv("MINI_LLM_MODEL") or "Qwen/Qwen2.5-4B-Instruct"
    return {
        "base_url": base,
        "api_key": key,
        "model": model,
        "mini_base_url": mini_base,
        "mini_api_key": mini_key,
        "mini_model": mini_model,
    }


def _load_from_secrets() -> dict:
    if not os.path.exists(SECRETS_PATH):
        return {}
    try:
        d = json.loads(open(SECRETS_PATH, encoding="utf-8").read())
    except Exception:
        return {}
    return {
        "base_url": (d.get("base_url") or "").rstrip("/"),
        "api_key": d.get("api_key") or "",
        "model": d.get("model") or "",
        "mini_base_url": (d.get("mini_base_url") or d.get("base_url") or "").rstrip(
            "/"
        ),
        "mini_api_key": d.get("mini_api_key") or d.get("api_key") or "",
        "mini_model": d.get("mini_model") or "Qwen/Qwen2.5-4B-Instruct",
    }


_cfg_env = _load_from_env()
_cfg_sec = _load_from_secrets()


def _picked(key: str) -> str:
    v = (_cfg_env.get(key) or "").strip()
    if v:
        return v
    return (_cfg_sec.get(key) or "").strip()


BASE_URL = _picked("base_url")
API_KEY = _picked("api_key")
MODEL = _picked("model")
MINI_BASE = _picked("mini_base_url") or BASE_URL
MINI_KEY = _picked("mini_api_key") or API_KEY
MINI_MODEL = _picked("mini_model") or "Qwen/Qwen2.5-4B-Instruct"

if not (BASE_URL and API_KEY and MODEL):
    raise RuntimeError(
        "Brak LLM configu: ustaw LLM_BASE_URL, LLM_API_KEY, LLM_MODEL w .env lub secrets.json"
    )

_sess = requests.Session()
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_sess.headers.update(
    {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
)
_sess.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=3,
            backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["POST"]),
        )
    ),
)


def _post(url: str, payload: dict, timeout: int = 60) -> dict:
    r = _sess.post(url, json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"LLM HTTP {r.status_code}: {r.text[:400]}")
    return r.json()


def chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.45,
    max_tokens: int = 1100,
    stream: bool = False,
) -> str:
    use_model = (model or MODEL).strip()
    url = BASE_URL.rstrip("/") + "/chat/completions"
    body: Dict[str, Any] = {
        "model": use_model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": bool(stream),
    }
    j = _post(url, body)
    ch = (j.get("choices") or [{}])[0]
    msg = (ch.get("message") or {}).get("content", "")
    if not msg:
        raise RuntimeError("LLM: pusta odpowiedź")
    return msg.strip()


def mini(prompt: str, max_tokens: int = 256, temperature: float = 0.3) -> str:
    b = (MINI_BASE or BASE_URL).rstrip("/") + "/chat/completions"
    k = MINI_KEY or API_KEY
    if not (MINI_MODEL and k and b):
        return ""
    s = requests.Session()
    s.headers.update(
        {"Authorization": f"Bearer {k}", "Content-Type": "application/json"}
    )
    j = s.post(
        b,
        json={
            "model": MINI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        },
        timeout=60,
    )
    if j.status_code >= 400:
        return ""
    jj = j.json()
    ch = (jj.get("choices") or [{}])[0]
    return ((ch.get("message") or {}).get("content", "") or "").strip()


def health() -> dict:
    return {"ok": True, "base_url": BASE_URL, "model": MODEL, "mini_model": MINI_MODEL}
