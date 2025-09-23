"""memory_api.py — bezpieczne API dla istniejącego memory.py (bez modyfikacji memory.py).
Montowane jako router pod /api/memory.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import time

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from pydantic import BaseModel, Field

# Import Twojego modułu pamięci (nie zmieniamy go)
try:
    from .memory import get_memory  # type: ignore
except Exception:
    from memory import get_memory  # type: ignore

# ────────────────────────────────────────────────────────────────────────────
# MODELE
# ────────────────────────────────────────────────────────────────────────────


class FactIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    conf: float = Field(0.6, ge=0.0, le=1.0)
    tags: Optional[List[str]] = None
    meta: Optional[Dict[str, Any]] = None
    emb: Optional[List[float]] = None


class RecallIn(BaseModel):
    q: str = Field(..., min_length=1, max_length=1000)
    topk: int = Field(6, ge=1, le=100)


class ContextIn(BaseModel):
    q: str = Field("", max_length=1000)
    topk: int = Field(12, ge=1, le=60)
    limit: int = Field(3500, ge=200, le=20000)


class ExportIn(BaseModel):
    out: str


class ImportIn(BaseModel):
    path: str
    merge: bool = True


# ────────────────────────────────────────────────────────────────────────────
# ROUTER & HELPER
# ────────────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/api/memory", tags=["memory"])


def mem():
    return get_memory()


# ────────────────────────────────────────────────────────────────────────────
# HEALTH / STATS
# ────────────────────────────────────────────────────────────────────────────


@router.get("/health")
def health() -> Dict[str, Any]:
    try:
        st = mem().stats()
        return {"ok": True, "stats": st}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/stats")
def stats() -> Dict[str, Any]:
    return mem().stats()


# ────────────────────────────────────────────────────────────────────────────
# FACTS (proste CRUD)
# ────────────────────────────────────────────────────────────────────────────


@router.get("/facts")
def list_facts(limit: int = Query(100, ge=1, le=5000)) -> List[Dict[str, Any]]:
    rows = mem().list_facts(limit=limit)
    return rows


@router.post("/remember")
def remember(body: FactIn) -> Dict[str, Any]:
    try:
        fid = mem().add_fact(
            body.text,
            meta_data=body.meta or {},
            score=body.conf,
            emb=body.emb,
            tags=body.tags or [],
        )
        return {"ok": True, "id": fid}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ────────────────────────────────────────────────────────────────────────────
# RECALL / CONTEXT
# ────────────────────────────────────────────────────────────────────────────


@router.post("/recall")
def recall(body: RecallIn) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for txt, sc, src in mem().recall(body.q, topk=body.topk):
        out.append({"text": txt, "score": float(sc), "src": src})
    return out


@router.post("/context")
def compose_context(body: ContextIn) -> Dict[str, Any]:
    ctx = mem().compose_context(body.q, limit_chars=body.limit, topk=body.topk)
    return {"ok": True, "context": ctx}


# ────────────────────────────────────────────────────────────────────────────
# IMPORT / EXPORT / SEED
# ────────────────────────────────────────────────────────────────────────────


@router.post("/export")
def export_json(body: ExportIn) -> Dict[str, Any]:
    return mem().export_json(body.out)


@router.post("/import")
def import_json(body: ImportIn) -> Dict[str, Any]:
    return mem().import_json(body.path, merge=body.merge)


@router.post("/reload-seed")
def reload_seed(path: str = Form(...)) -> Dict[str, Any]:
    """
    Wgrywa fakty z pliku JSON.
    Dozwolone formaty:
      - {"facts": [{"text": "...", "conf": 0.8, "tags": ["a","b"]}, ...]}
      - [{"text": "...", "conf": 0.8, "tags": ["a","b"]}, ...]
    """
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Seed file not found")

    try:
        rows = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON seed format: {e}")

    if isinstance(rows, dict):
        rows = rows.get("facts") or rows.get("data") or []

    if not isinstance(rows, list):
        raise HTTPException(status_code=400, detail="Invalid JSON seed format")

    inserted = 0
    for row in rows:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        conf = float(row.get("conf", 0.6))
        tags = row.get("tags") or []
        meta = row.get("meta") or {}
        mem().add_fact(text, meta_data=meta, score=conf, tags=tags)
        inserted += 1

    return {"ok": True, "inserted": inserted, "path": str(p)}


# ────────────────────────────────────────────────────────────────────────────
# PING
# ────────────────────────────────────────────────────────────────────────────


@router.get("/ping")
def ping() -> Dict[str, Any]:
    return {"ok": True, "time": time.time(), "module": "memory_api"}
