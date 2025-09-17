"""
images_client.py — v4.0 MULTI-ENGINE

Cel:
- Multi-provider obrazów: OpenAI, Stability.ai, Replicate, AUTOMATIC1111 (WebUI), HuggingFace Inference API, ComfyUI.
- Edycja (inpaint/img2img), analiza (caption), galeria (list/feedback), utility (upscale, remove bg).
- Kolejność providerów sterowana przez ENV IMG_PROVIDERS.
- Zapisy z metadanymi obok plików (JSON).

ENV (przykład):
  APP_ROOT=/workspace/a
  IMG_PROVIDERS=openai,stability,replicate,a1111,hf,comfyui
  IMG_MODEL=gpt-image-1
  IMG_SIZE=1024x1024
  IMG_TIMEOUT=90

  # OpenAI
  OPENAI_KEY=sk-proj-EmKhjmI2SHrDjK75ocjI5OuKI_Uea7qQO-d7t0
  OPENAI_BASE_URL=https://api.openai.com/v1

  # Stability
  STABILITY_KEY=-aMT2CsbRlkpqU9ZaoIPgTvrL4pa7pM8H
  STABILITY_BASE_URL=https://api.stability.ai/v1

  # Replicate
  REPLICATE_KEY=r8_FV3z4de9D7Qlpw2je2iAcpU2LcYR6hC2gwpOC
  REPLICATE_MODEL=stability-ai/stable-diffusion-xl

  # AUTOMATIC1111 (Stable Diffusion WebUI)
  A1111_BASE=http://127.0.0.1:7860

  # Hugging Face Inference API
  HF_TOKEN=hf_VTyPVDyhszhoEhIZApLQTxQBCishBbZC1T
  HF_T2I_MODEL=stabilityai/stable-diffusion-xl-base-1.0
  HF_CAPTION_MODEL=Salesforce/blip-image-captioning-large

  # ComfyUI
  COMFYUI_BASE=http://127.0.0.1:8188
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

import requests

# ──────────────────────────────────────────────
# KONFIG / ŚCIEŻKI
# ──────────────────────────────────────────────
APP_ROOT = os.getenv("APP_ROOT", "/workspace/a").rstrip("/")
DATA_DIR = Path(APP_ROOT) / "data" / "images"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SIZE = os.getenv("IMG_SIZE", "1024x1024")
TIMEOUT = int(os.getenv("IMG_TIMEOUT", "90"))
PROVIDERS = [
    p.strip()
    for p in os.getenv("IMG_PROVIDERS", "openai,stability,replicate,a1111,hf,comfyui").split(",")
    if p.strip()
]

# OpenAI
OPENAI_KEY = os.getenv("OPENAI_KEY", "")
OPENAI_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_IMG_MODEL = os.getenv("IMG_MODEL", "gpt-image-1")

# Stability
STABILITY_KEY = os.getenv("STABILITY_KEY", "")
STABILITY_BASE = os.getenv("STABILITY_BASE_URL", "https://api.stability.ai/v1").rstrip("/")

# Replicate
REPLICATE_KEY = os.getenv("REPLICATE_KEY", "")
REPLICATE_BASE = "https://api.replicate.com/v1/predictions"
REPLICATE_MODEL = os.getenv("REPLICATE_MODEL", "stability-ai/stable-diffusion-xl")

# AUTOMATIC1111 (Stable Diffusion WebUI)
A1111_BASE = os.getenv("A1111_BASE", "").rstrip("/")

# Hugging Face Inference API
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_T2I_MODEL = os.getenv("HF_T2I_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
HF_CAPTION_MODEL = os.getenv("HF_CAPTION_MODEL", "Salesforce/blip-image-captioning-large")

# ComfyUI
COMFYUI_BASE = os.getenv("COMFYUI_BASE", "").rstrip("/")


# ──────────────────────────────────────────────
# HELPERY
# ──────────────────────────────────────────────
def _ts() -> int:
    return int(time.time())


def _ext_from_mime(mime: str) -> str:
    ext = mimetypes.guess_extension(mime) or ".png"
    if ext == ".jpe":
        ext = ".jpg"
    return ext


def _save_bytes(
    raw: bytes, prompt: str, model: str, user: str = "default", mime: str = "image/png"
) -> str:
    ext = _ext_from_mime(mime)
    fname = f"{user}_{_ts()}{ext}"
    fpath = DATA_DIR / fname
    with open(fpath, "wb") as f:
        f.write(raw)
    _save_meta(fpath, prompt, model, user)
    return str(fpath)


def _save_image_b64(b64: str, prompt: str, model: str, user: str = "default") -> str:
    raw = base64.b64decode(b64)
    return _save_bytes(raw, prompt, model, user, "image/png")


def _save_image_url(url: str, prompt: str, model: str, user: str = "default") -> str:
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    mime = r.headers.get("content-type", "image/png")
    return _save_bytes(r.content, prompt, model, user, mime)


def _save_meta(fpath: Path, prompt: str, model: str, user: str) -> None:
    meta = {
        "prompt": prompt,
        "model": model,
        "user": user,
        "ts": _ts(),
        "filename": fpath.name,
    }
    with open(f"{str(fpath)}.json", "w", encoding="utf-8") as mf:
        json.dump(meta, mf, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# PROVIDER: OpenAI
# ──────────────────────────────────────────────
def _openai_generate(prompt: str, size: str, user: str) -> str | None:
    if not OPENAI_KEY:
        return None
    url = f"{OPENAI_BASE}/images/generations"
    payload = {"model": OPENAI_IMG_MODEL, "prompt": prompt, "size": size}
    headers = {"Authorization": f"Bearer {OPENAI_KEY}"}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        j = r.json()
        data = j.get("data") or []
        if not data:
            return None
        d0 = data[0]
        if "b64_json" in d0:
            return _save_image_b64(d0["b64_json"], prompt, OPENAI_IMG_MODEL, user)
        if "url" in d0:
            return _save_image_url(d0["url"], prompt, OPENAI_IMG_MODEL, user)
    except Exception as e:
        print("OpenAI generate error:", e)
    return None


def _openai_edit(image_path: str, prompt: str, user: str) -> str | None:
    if not OPENAI_KEY:
        return None
    url = f"{OPENAI_BASE}/images/edits"
    headers = {"Authorization": f"Bearer {OPENAI_KEY}"}
    files = {"image": open(image_path, "rb")}
    data = {"model": OPENAI_IMG_MODEL, "prompt": prompt}
    try:
        r = requests.post(url, data=data, files=files, headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        j = r.json()
        data = j.get("data") or []
        if data and "b64_json" in data[0]:
            return _save_image_b64(data[0]["b64_json"], prompt, OPENAI_IMG_MODEL, user)
    except Exception as e:
        print("OpenAI edit error:", e)
    return None


# ──────────────────────────────────────────────
# PROVIDER: Stability.ai
# ──────────────────────────────────────────────
def _stability_generate(prompt: str, size: str, user: str) -> str | None:
    if not STABILITY_KEY:
        return None
    url = f"{STABILITY_BASE}/generation/stable-diffusion-v1-6/text-to-image"
    try:
        w, h = [int(x) for x in size.split("x")]
    except Exception:
        w, h = 1024, 1024
    payload = {"text_prompts": [{"text": prompt}], "width": w, "height": h}
    headers = {"Authorization": f"Bearer {STABILITY_KEY}", "Accept": "application/json"}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        j = r.json()
        arts = j.get("artifacts") or []
        if arts and "base64" in arts[0]:
            return _save_image_b64(arts[0]["base64"], prompt, "stability-v1-6", user)
    except Exception as e:
        print("Stability generate error:", e)
    return None


# ──────────────────────────────────────────────
# PROVIDER: Replicate (async polling)
# ──────────────────────────────────────────────
def _replicate_generate(prompt: str, size: str, user: str) -> str | None:
    if not REPLICATE_KEY:
        return None
    headers = {"Authorization": f"Token {REPLICATE_KEY}", "Content-Type": "application/json"}
    try:
        w, h = [int(x) for x in size.split("x")]
    except Exception:
        w, h = 1024, 1024
    payload = {"version": REPLICATE_MODEL, "input": {"prompt": prompt, "width": w, "height": h}}
    try:
        r = requests.post(REPLICATE_BASE, json=payload, headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        j = r.json()
        status_url = j.get("urls", {}).get("get")
        if not status_url:
            return None
        for _ in range(max(10, TIMEOUT // 2)):
            rr = requests.get(status_url, headers=headers, timeout=TIMEOUT)
            jj = rr.json()
            st = jj.get("status")
            if st == "succeeded":
                out = jj.get("output") or []
                if out:
                    # pierwszy URL
                    return _save_image_url(out[0], prompt, REPLICATE_MODEL, user)
                break
            if st == "failed":
                break
            time.sleep(2)
    except Exception as e:
        print("Replicate generate error:", e)
    return None


# ──────────────────────────────────────────────
# PROVIDER: AUTOMATIC1111 (Stable Diffusion WebUI)
# ──────────────────────────────────────────────
def _a1111_generate(prompt: str, size: str, user: str) -> str | None:
    if not A1111_BASE:
        return None
    url = f"{A1111_BASE}/sdapi/v1/txt2img"
    try:
        w, h = [int(x) for x in size.split("x")]
    except Exception:
        w, h = 1024, 1024
    payload = {"prompt": prompt, "width": w, "height": h, "sampler_name": "Euler a", "steps": 30}
    try:
        r = requests.post(url, json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        j = r.json()
        imgs = j.get("images") or []
        if imgs:
            return _save_image_b64(imgs[0], prompt, "a1111-txt2img", user)
    except Exception as e:
        print("A1111 txt2img error:", e)
    return None


def _a1111_img2img(image_path: str, prompt: str, user: str) -> str | None:
    if not A1111_BASE:
        return None
    url = f"{A1111_BASE}/sdapi/v1/img2img"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    payload = {"init_images": [b64], "prompt": prompt, "denoising_strength": 0.5, "steps": 30}
    try:
        r = requests.post(url, json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        j = r.json()
        imgs = j.get("images") or []
        if imgs:
            return _save_image_b64(imgs[0], prompt, "a1111-img2img", user)
    except Exception as e:
        print("A1111 img2img error:", e)
    return None


# ──────────────────────────────────────────────
# PROVIDER: Hugging Face Inference API
# ──────────────────────────────────────────────
def _hf_generate(prompt: str, size: str, user: str) -> str | None:
    if not HF_TOKEN:
        return None
    url = f"https://api-inference.huggingface.co/models/{HF_T2I_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    # Uwaga: wiele modeli ignoruje width/height w tym endpoint
    payload = {"inputs": prompt}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/"):
            return _save_bytes(r.content, prompt, HF_T2I_MODEL, user, r.headers["content-type"])
        # Niektóre modele zwracają b64 w JSON
        if "application/json" in r.headers.get("content-type", ""):
            j = r.json()
            b64 = j.get("b64_json")
            if b64:
                return _save_image_b64(b64, prompt, HF_T2I_MODEL, user)
    except Exception as e:
        print("HF generate error:", e)
    return None


def _hf_caption(image_path: str) -> str | None:
    if not HF_TOKEN:
        return None
    url = f"https://api-inference.huggingface.co/models/{HF_CAPTION_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        with open(image_path, "rb") as f:
            r = requests.post(url, headers=headers, data=f.read(), timeout=TIMEOUT)
        r.raise_for_status()
        j = r.json()
        if isinstance(j, list) and j and "generated_text" in j[0]:
            return j[0]["generated_text"]
        if isinstance(j, dict) and "generated_text" in j:
            return j["generated_text"]
    except Exception as e:
        print("HF caption error:", e)
    return None


# ──────────────────────────────────────────────
# PROVIDER: ComfyUI (bardzo uproszczony trigger)
# ──────────────────────────────────────────────
def _comfyui_generate(prompt: str, size: str, user: str) -> str | None:
    if not COMFYUI_BASE:
        return None
    # Minimal: zakładamy gotowy workflow na serwerze (tu tylko ping)
    # Typowo ComfyUI wymaga JSON pipeline; tu dajemy „placeholder ping”.
    try:
        # W praktyce: wyślij JSON workflow z promptem -> odbierz image URL/path
        # Tu dla kompatybilności zwracamy None (jeśli nie masz workflowa).
        return None
    except Exception as e:
        print("ComfyUI generate error:", e)
        return None


# ──────────────────────────────────────────────
# PUBLIC API: GENERATE / EDIT / ANALYZE / LIST / FEEDBACK
# ──────────────────────────────────────────────
def generate_image(prompt: str, size: str = DEFAULT_SIZE, user: str = "default") -> dict[str, Any]:
    """Próbuje providerów w kolejności IMG_PROVIDERS aż do skutku."""
    chain = {
        "openai": lambda: _openai_generate(prompt, size, user),
        "stability": lambda: _stability_generate(prompt, size, user),
        "replicate": lambda: _replicate_generate(prompt, size, user),
        "a1111": lambda: _a1111_generate(prompt, size, user),
        "hf": lambda: _hf_generate(prompt, size, user),
        "comfyui": lambda: _comfyui_generate(prompt, size, user),
    }
    last_err = None
    for p in PROVIDERS:
        fn = chain.get(p)
        if not fn:
            continue
        try:
            path = fn()
            if path:
                return {"ok": True, "provider": p, "path": path}
        except Exception as e:
            last_err = str(e)
            print(f"provider {p} failed:", e)
    return {"ok": False, "error": last_err or "brak dostawcy"}


def edit_image(image_path: str, prompt: str, user: str = "default") -> dict[str, Any]:
    """Edycja istniejącego obrazu (priorytet: A1111 img2img → OpenAI edits)."""
    path = _a1111_img2img(image_path, prompt, user) or _openai_edit(image_path, prompt, user)
    if not path:
        return {"ok": False, "error": "no editor available"}
    return {"ok": True, "path": path}


def analyze_image(image_path: str) -> dict[str, Any]:
    """Opis obrazka (HF BLIP caption jeśli dostępny, inaczej stub)."""
    cap = _hf_caption(image_path)
    if cap:
        return {"ok": True, "desc": cap}
    return {
        "ok": True,
        "desc": f"Analiza obrazu {os.path.basename(image_path)} — (brak Vision API)",
    }


def list_images(user: str = "default") -> list[str]:
    """Lista ścieżek obrazów użytkownika."""
    return sorted(
        [
            str(p)
            for p in DATA_DIR.glob(f"{user}_*.*")
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
        ]
    )


def feedback(image_path: str, rating: int, tags: list[str] | None = None) -> dict[str, Any]:
    """Zapis opinii do meta.json obok pliku."""
    meta_path = f"{image_path}.json"
    if not os.path.exists(meta_path):
        return {"ok": False, "error": "brak meta"}
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    meta["feedback"] = {"rating": int(rating), "tags": tags or [], "ts": _ts()}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return {"ok": True, "updated": meta}


# ──────────────────────────────────────────────
# UTILS: UPSCALE + REMOVE BG
# ──────────────────────────────────────────────
def upscale_image(image_path: str, user: str = "default") -> dict[str, Any]:
    """
    Upscale przez Replicate (Real-ESRGAN). Wymaga REPLICATE_KEY.
    """
    if not REPLICATE_KEY:
        return {"ok": False, "error": "brak REPLICATE_KEY"}
    headers = {"Authorization": f"Token {REPLICATE_KEY}", "Content-Type": "application/json"}
    url = REPLICATE_BASE
    model = "nightmareai/real-esrgan"
    # Upload lokalnego pliku do tmp hosta (najprościej: base64 → data URL lub preupload do storage — tu data URL)
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    payload = {"version": model, "input": {"image": f"data:image/png;base64,{b64}"}}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        j = r.json()
        status_url = j.get("urls", {}).get("get")
        if not status_url:
            return {"ok": False, "error": "replicate no status url"}
        for _ in range(max(10, TIMEOUT // 2)):
            rr = requests.get(status_url, headers=headers, timeout=TIMEOUT)
            jj = rr.json()
            if jj.get("status") == "succeeded":
                out = jj.get("output") or []
                if out:
                    path = _save_image_url(
                        out[0], f"UPSCALE:{os.path.basename(image_path)}", model, user
                    )
                    return {"ok": True, "path": path}
                break
            if jj.get("status") == "failed":
                break
            time.sleep(2)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": False, "error": "unknown"}


def remove_background(image_path: str, user: str = "default") -> dict[str, Any]:
    """
    Remove BG przez Replicate (rembg). Wymaga REPLICATE_KEY.
    """
    if not REPLICATE_KEY:
        return {"ok": False, "error": "brak REPLICATE_KEY"}
    headers = {"Authorization": f"Token {REPLICATE_KEY}", "Content-Type": "application/json"}
    url = REPLICATE_BASE
    model = "danielgatis/rembg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    payload = {"version": model, "input": {"image": f"data:image/png;base64,{b64}"}}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        j = r.json()
        status_url = j.get("urls", {}).get("get")
        if not status_url:
            return {"ok": False, "error": "replicate no status url"}
        for _ in range(max(10, TIMEOUT // 2)):
            rr = requests.get(status_url, headers=headers, timeout=TIMEOUT)
            jj = rr.json()
            if jj.get("status") == "succeeded":
                out = jj.get("output") or []
                if out:
                    path = _save_image_url(
                        out[0], f"REMOVED_BG:{os.path.basename(image_path)}", model, user
                    )
                    return {"ok": True, "path": path}
                break
            if jj.get("status") == "failed":
                break
            time.sleep(2)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": False, "error": "unknown"}


# Production ready - no demo code
