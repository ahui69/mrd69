from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

try:
    from travelguide import (
        TripSpec,
        attractions,
        geocode,
        hotels,
        plan_trip,
        restaurants,
        static_map_url,
        tl_stops_near,
        tp_cheapest_month,
        tp_iata,
    )
except Exception:  # pragma: no cover - optional dependency
    TripSpec = None  # type: ignore
    attractions = None  # type: ignore
    geocode = None  # type: ignore
    hotels = None  # type: ignore
    plan_trip = None  # type: ignore
    restaurants = None  # type: ignore
    static_map_url = None  # type: ignore
    tl_stops_near = None  # type: ignore
    tp_cheapest_month = None  # type: ignore
    tp_iata = None  # type: ignore


router = APIRouter(prefix="/api/travel", tags=["travel"])


class TravelPlanRequest(BaseModel):
    place: str = Field(..., min_length=2)
    days: int = Field(..., ge=1, le=30)
    food: bool = True
    hotels: bool = True
    lang: str = Field("pl", min_length=2, max_length=5)
    user_id: str = Field("default", min_length=1, max_length=40)
    narrative_style: str = Field(
        "auto",
        pattern="^(auto|friendly|elegant|energetic|scholarly|casual|professional)$",
    )


def _ensure_available() -> None:
    if TripSpec is None or plan_trip is None:  # type: ignore[truthy-bool]
        raise HTTPException(status_code=503, detail="Travel engine not available")


@router.post("/plan")
def create_plan(body: TravelPlanRequest) -> Dict[str, Any]:
    _ensure_available()
    try:
        spec = TripSpec(
            place=body.place,
            days=body.days,
            food=body.food,
            hotels=body.hotels,
            lang=body.lang,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid trip spec: {exc}")
    try:
        result = plan_trip(spec, user_id=body.user_id, narrative_style=body.narrative_style)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Travel plan failed: {exc}")
    return result


@router.get("/restaurants")
def get_restaurants(
    place: str = Query(..., min_length=2),
    q: str = Query(""),
    max_results: int = Query(24, ge=1, le=80),
    open_now: bool = Query(False),
    lang: str = Query("pl", min_length=2, max_length=5),
) -> Dict[str, Any]:
    _ensure_available()
    try:
        items = restaurants(place, q, max_results, open_now, lang)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Restaurants lookup failed: {exc}")
    return {"place": place, "items": items, "count": len(items)}


@router.get("/hotels")
def get_hotels(
    place: str = Query(..., min_length=2),
    q: str = Query(""),
    max_results: int = Query(24, ge=1, le=80),
    lang: str = Query("pl", min_length=2, max_length=5),
) -> Dict[str, Any]:
    _ensure_available()
    try:
        items = hotels(place, q, max_results, lang)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Hotels lookup failed: {exc}")
    return {"place": place, "items": items, "count": len(items)}


@router.get("/attractions")
def get_attractions(
    place: str = Query(..., min_length=2),
    max_results: int = Query(60, ge=1, le=120),
    lang: str = Query("pl", min_length=2, max_length=5),
) -> Dict[str, Any]:
    _ensure_available()
    try:
        items = attractions(place, max_results, lang)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Attractions lookup failed: {exc}")
    return {"place": place, "items": items, "count": len(items)}


@router.get("/flights")
def get_flights(
    origin: str = Query(..., min_length=3),
    dest: str = Query(..., min_length=3),
    month: str = Query(..., regex="^\\d{4}-\\d{2}$"),
) -> Dict[str, Any]:
    _ensure_available()
    if tp_cheapest_month is None or tp_iata is None:
        raise HTTPException(status_code=503, detail="Flights helper not available")
    try:
        orig = origin.upper() if len(origin) == 3 else (tp_iata(origin) or origin.upper())
        dst = dest.upper() if len(dest) == 3 else (tp_iata(dest) or dest.upper())
        data = tp_cheapest_month(orig, dst, month)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Flights lookup failed: {exc}")
    return {"origin": orig, "dest": dst, "month": month, "data": data}


@router.get("/transit")
def get_transit(
    place: str = Query(..., min_length=2),
    radius: int = Query(900, ge=100, le=5000),
) -> Dict[str, Any]:
    _ensure_available()
    if geocode is None or tl_stops_near is None:
        raise HTTPException(status_code=503, detail="Transit helper not available")
    geo = geocode(place)
    if not geo:
        raise HTTPException(status_code=404, detail="Place not found")
    try:
        stops = tl_stops_near(geo["lat"], geo["lon"], radius)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transit lookup failed: {exc}")
    return {"place": place, "stops": stops, "count": len(stops)}


@router.get("/map")
def get_map(
    place: str = Query(..., min_length=2),
    max_items: int = Query(30, ge=10, le=80),
) -> Dict[str, Any]:
    _ensure_available()
    if geocode is None:
        raise HTTPException(status_code=503, detail="Geocoding not available")
    geo = geocode(place)
    if not geo:
        raise HTTPException(status_code=404, detail="Place not found")
    pin_limit = max_items
    try:
        ats = attractions(place, max_results=min(50, pin_limit))
        foods = restaurants(place, max_results=min(30, pin_limit))
        hotels_items = hotels(place, max_results=min(20, pin_limit))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lookup failed: {exc}")
    pins = []
    for it in ats[:10]:
        if it.get("lat") and it.get("lon"):
            pins.append((it["lat"], it["lon"], "A"))
    for it in foods[:6]:
        if it.get("lat") and it.get("lon"):
            pins.append((it["lat"], it["lon"], "F"))
    for it in hotels_items[:3]:
        if it.get("lat") and it.get("lon"):
            pins.append((it["lat"], it["lon"], "H"))
    map_url: Optional[str] = None
    if static_map_url is not None:
        map_url = static_map_url((geo["lat"], geo["lon"]), pins)
    return {
        "center": {"lat": geo["lat"], "lon": geo["lon"]},
        "map_url": map_url,
        "pins": pins,
        "stats": {
            "attractions": len(ats),
            "restaurants": len(foods),
            "hotels": len(hotels_items),
        },
    }
