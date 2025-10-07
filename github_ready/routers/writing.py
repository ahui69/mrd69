from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

try:
    from src.fix_writing_all_pro import ListingManager
except Exception:  # pragma: no cover - optional dependency
    ListingManager = None  # type: ignore

router = APIRouter(prefix="/api/listings", tags=["listings"])

_listing_manager: Optional[ListingManager] = None  # type: ignore[assignment]


def _get_manager() -> ListingManager:
    """Return a singleton ListingManager instance or raise 503 when unavailable."""
    global _listing_manager
    if ListingManager is None:  # type: ignore[truthy-bool]
        raise HTTPException(status_code=503, detail="Listing engine not available")
    if _listing_manager is None:
        try:
            _listing_manager = ListingManager()
        except Exception as exc:  # pragma: no cover - heavy module init
            raise HTTPException(status_code=500, detail=f"Listing init failed: {exc}")
    return _listing_manager


class ListingCreateRequest(BaseModel):
    brand: str = Field(..., min_length=1)
    item: str = Field(..., min_length=1)
    cond: str = Field(..., min_length=1, description="Condition e.g. Nowy/Bardzo dobry")
    size: str = Field(..., min_length=1)
    color: Optional[str] = ""
    material: Optional[str] = ""
    measurements: Optional[Dict[str, str]] = None
    model: Optional[str] = ""
    defects: Optional[List[str]] = None
    extras: Optional[List[str]] = None
    notes: Optional[str] = ""
    tier_hint: Optional[str] = None
    base_price: Optional[float] = Field(None, ge=0)
    price_mode: Optional[str] = None
    copy_tone: str = Field("balanced", description="balanced/sales/brand")
    profanity_level: float = Field(0.0, ge=0.0, le=1.0)
    images: Optional[List[str]] = None
    template: Optional[str] = None


class ListingUpdateRequest(BaseModel):
    brand: Optional[str] = None
    item: Optional[str] = None
    cond: Optional[str] = None
    size: Optional[str] = None
    color: Optional[str] = None
    material: Optional[str] = None
    measurements: Optional[Dict[str, str]] = None
    model: Optional[str] = None
    defects: Optional[List[str]] = None
    extras: Optional[List[str]] = None
    notes: Optional[str] = None
    tier_hint: Optional[str] = None
    base_price: Optional[float] = Field(None, ge=0)
    price: Optional[float] = Field(None, ge=0)
    price_mode: Optional[str] = None
    copy_tone: Optional[str] = None
    template: Optional[str] = None


class ListingMetricsPayload(BaseModel):
    impressions: int = Field(0, ge=0)
    clicks: int = Field(0, ge=0)
    wishlist: int = Field(0, ge=0)
    sold: int = Field(0, ge=0)
    price: Optional[int] = Field(None, ge=0)


@router.get("/health")
def listings_health() -> Dict[str, Any]:
    """Simple readiness check for the listing engine."""
    manager = _get_manager()
    return {
        "ok": True,
        "out_dir": str(getattr(manager, "outdir", "")),
    }


@router.post("/create")
def create_listing(body: ListingCreateRequest) -> Dict[str, Any]:
    manager = _get_manager()
    payload = body.dict(exclude_none=True)
    result = manager.create(**payload)
    return result


@router.post("/revise/{slug}")
def revise_listing(slug: str, body: ListingUpdateRequest) -> Dict[str, Any]:
    manager = _get_manager()
    update_payload = body.dict(exclude_none=True)
    if not update_payload:
        raise HTTPException(status_code=400, detail="No changes provided")
    result = manager.revise(slug, **update_payload)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail="Listing not found")
    return result


@router.get("/search")
def search_listings(
    brand: Optional[str] = Query(None, description="Filter by brand substring"),
    item: Optional[str] = Query(None, description="Filter by item name"),
    size: Optional[str] = Query(None, description="Filter by size"),
    text: Optional[str] = Query(None, description="Full-text search in description"),
    limit: int = Query(25, ge=1, le=200),
) -> Dict[str, Any]:
    manager = _get_manager()
    records = manager.find(
        brand=brand,
        item=item,
        size=size,
        text=text,
        limit=limit,
    )
    return {
        "items": [asdict(rec) for rec in records],
        "count": len(records),
    }


@router.get("/{slug}")
def get_listing(slug: str) -> Dict[str, Any]:
    manager = _get_manager()
    try:
        records = [rec for rec in manager._index_all() if getattr(rec, "slug", "") == slug]  # type: ignore[attr-defined]
    except AttributeError as exc:  # pragma: no cover - defensive fallback
        raise HTTPException(status_code=500, detail=f"Listing index unavailable: {exc}")
    if not records:
        raise HTTPException(status_code=404, detail="Listing not found")
    return {"item": asdict(records[-1])}

@router.post("/metrics/{slug}")
def push_metrics(slug: str, body: ListingMetricsPayload) -> Dict[str, Any]:
    manager = _get_manager()
    manager.log_metrics(slug, **body.dict(exclude_none=True))
    return {"ok": True}


@router.get("/metrics")
def get_metrics(slug: Optional[str] = Query(None)) -> Dict[str, Any]:
    manager = _get_manager()
    rows = manager.metrics(slug=slug)
    return {"items": rows, "count": len(rows)}


@router.get("/dynamic-prices")
def dynamic_prices(
    slug: Optional[str] = Query(None),
    policy: str = Query("auto", regex="^(auto|aggressive|premium)$"),
) -> Dict[str, Any]:
    manager = _get_manager()
    updates = manager.dynamic_prices(slug=slug, policy=policy)
    return {"items": updates, "count": len(updates)}
