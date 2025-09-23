# ruff: noqa: E501
# /workspace/mrd69/overmind/travelguide.py
"""
TravelGuide ULTRA — globalny planner: Google + TripAdvisor + OTM/OSM
+ Travelpayouts + Transitland + XWeather/Open-Meteo.
ENV (wystarczą te które masz):
  GOOGLE_MAPS_KEY, TRIPADVISOR_KEY, OTM_KEY, TP_TOKEN,
  TRANSITLAND_API_KEY, XWEATHER_SECRET
CLI:
  python travelguide.py plan --place "Barcelona" --days 4 --food --hotels --lang pl
  python travelguide.py food --place "Gorzów Wielkopolski" --q "pizza" \\
    --open-now --max 20
  python travelguide.py hotels --place "Berlin" --q "Mitte" --max 20
  python travelguide.py attractions --place "Tokyo" --max 60
  python travelguide.py flights --from WAW --to BCN --month 2025-10
  python travelguide.py transit --place "London" --radius 1200
  python travelguide.py map --place "Madrid" --max 30
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

# ── PATHS / IO ────────────────────────────────────────────────────────────────
# Konfiguracja ścieżek z ENV
from dotenv import load_dotenv

load_dotenv()

APP_ROOT = Path(os.getenv("APP_ROOT", Path(__file__).parent))
OUT_DIR = APP_ROOT / "out" / "travel"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_F = OUT_DIR / "_cache.json"

# Konfiguracja cache
# Cache size limit: convert MB to bytes
CACHE_SIZE_LIMIT = int(os.getenv("TRAVEL_CACHE_SIZE_MB", "100")) * 1024 * 1024
DEFAULT_TTL = int(os.getenv("TRAVEL_DEFAULT_TTL", "3600"))  # 1h default

# TTL per źródło (sekundy)
SOURCE_TTL = {
    "google": int(os.getenv("TRAVEL_GOOGLE_TTL", "3600")),  # 1h
    "tripadvisor": int(os.getenv("TRAVEL_TA_TTL", "7200")),  # 2h
    "otm": int(os.getenv("TRAVEL_OTM_TTL", "86400")),  # 24h
    "wiki": int(os.getenv("TRAVEL_WIKI_TTL", "604800")),  # 7 dni
    "osm": int(os.getenv("TRAVEL_OSM_TTL", "86400")),  # 24h
    "weather": int(os.getenv("TRAVEL_WEATHER_TTL", "3600")),  # 1h
    "transitland": int(os.getenv("TRAVEL_TRANSIT_TTL", "3600")),  # 1h
    "flights": int(os.getenv("TRAVEL_FLIGHTS_TTL", "1800")),  # 30min
}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-") or "item"


def _read_cache() -> dict:
    try:
        if not CACHE_F.exists():
            return {"entries": {}, "metadata": {"created": time.time(), "size": 0}}
        content = CACHE_F.read_text(encoding="utf-8")
        cache = json.loads(content)
        # Migracja starogo formatu
        if "entries" not in cache:
            cache = {
                "entries": cache,
                "metadata": {"created": time.time(), "size": len(content)},
            }
        return cache
    except Exception:
        return {"entries": {}, "metadata": {"created": time.time(), "size": 0}}


def _write_cache(cache: dict) -> None:
    try:
        content = json.dumps(cache, ensure_ascii=False, indent=2)
        cache["metadata"]["size"] = len(content)
        CACHE_F.write_text(content, encoding="utf-8")
    except Exception:
        pass


def _clean_cache(cache: dict) -> dict:
    """Wyczyść przeterminowane wpisy i ogranicz rozmiar cache."""
    now = time.time()
    entries = cache.get("entries", {})

    # Usuń przeterminowane wpisy
    cleaned_entries = {}
    for key, entry in entries.items():
        age = now - entry.get("ts", 0)
        source = entry.get("source", "default")
        ttl = SOURCE_TTL.get(source, DEFAULT_TTL)

        if age < ttl:
            cleaned_entries[key] = entry

    # Jeśli cache za duży, usuń najstarsze wpisy
    if cache["metadata"]["size"] > CACHE_SIZE_LIMIT:
        # Sortuj po timestamp, najstarsze pierwsze
        sorted_entries = sorted(
            cleaned_entries.items(), key=lambda x: x[1].get("ts", 0)
        )

        # Zachowaj tylko nowsze wpisy
        keep_entries = {}
        estimated_size = 0
        for key, entry in reversed(sorted_entries):  # od najnowszych
            entry_size = len(json.dumps(entry, ensure_ascii=False))
            if estimated_size + entry_size < CACHE_SIZE_LIMIT:
                keep_entries[key] = entry
                estimated_size += entry_size
            else:
                break

        cleaned_entries = keep_entries

    return {
        "entries": cleaned_entries,
        "metadata": {
            "created": cache["metadata"].get("created", now),
            "last_cleaned": now,
            "size": cache["metadata"]["size"],
        },
    }


CACHE = _read_cache()


def _save(prefix: str, md: str, meta: dict[str, Any]) -> dict[str, str]:
    p = OUT_DIR / f"{prefix}_{_now_ms()}.md"
    p.write_text(md, encoding="utf-8")
    p.with_suffix(".md.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"path": str(p), "meta": str(p.with_suffix(".md.json"))}


# ── Profile Management ─────────────────────────────────────────────────────────
PROFILES_DIR = OUT_DIR / "profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)
TRAVEL_HISTORY_DIR = OUT_DIR / "history"
TRAVEL_HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def load_traveler_profile(user_id: str) -> TravelerProfile:
    """Załaduj profil podróżnika lub utwórz nowy."""
    profile_file = PROFILES_DIR / f"{user_id}.json"

    if profile_file.exists():
        try:
            data = json.loads(profile_file.read_text(encoding="utf-8"))
            return TravelerProfile.from_dict(data)
        except Exception:
            pass

    # Utwórz nowy profil
    return TravelerProfile(user_id=user_id, name=user_id)


def save_traveler_profile(profile: TravelerProfile) -> None:
    """Zapisz profil podróżnika."""
    profile.last_updated = time.time()
    profile_file = PROFILES_DIR / f"{profile.user_id}.json"

    try:
        profile_file.write_text(
            json.dumps(profile.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"Error saving profile: {e}")


def update_profile_from_trip(
    profile: TravelerProfile, trip_data: dict[str, Any]
) -> None:
    """Aktualizuj profil na podstawie danych z podróży (learning)."""
    city = trip_data.get("city", "")
    attractions_visited = trip_data.get("attractions_visited", [])
    ratings = trip_data.get("ratings", {})  # {attraction_id: rating}

    # Aktualizuj odwiedzone miasta
    if city and city not in (profile.visited_cities or []):
        if profile.visited_cities is None:
            profile.visited_cities = []
        profile.visited_cities.append(city)

    # Analizuj preferencje na podstawie ocen
    if ratings:
        attraction_types = _categorize_attractions(attractions_visited)

        for attraction_id, rating in ratings.items():
            attraction_type = attraction_types.get(attraction_id, "unknown")
            if attraction_type == "unknown":
                continue

            # Aktualizuj scores based on rating (1-5 scale)
            normalized_rating = (rating - 3) / 2  # Convert to -1 to 1 scale
            learning_factor = 0.1  # How much to adjust preferences

            if profile.learning_scores is None:
                profile.learning_scores = {}

            current_score = profile.learning_scores.get(attraction_type, 0.5)
            new_score = current_score + (normalized_rating * learning_factor)
            profile.learning_scores[attraction_type] = max(0, min(1, new_score))

            # Update main preference fields
            if attraction_type == "museum":
                profile.likes_museums = max(
                    0,
                    min(1, profile.likes_museums + normalized_rating * learning_factor),
                )
            elif attraction_type == "nature":
                profile.likes_nature = max(
                    0,
                    min(1, profile.likes_nature + normalized_rating * learning_factor),
                )
            elif attraction_type == "nightlife":
                profile.likes_nightlife = max(
                    0,
                    min(
                        1, profile.likes_nightlife + normalized_rating * learning_factor
                    ),
                )
            elif attraction_type == "museum":
                profile.likes_museums = max(
                    0,
                    min(1, profile.likes_museums + normalized_rating * learning_factor),
                )
            elif attraction_type == "restaurant":
                profile.likes_restaurants = max(
                    0,
                    min(
                        1,
                        profile.likes_restaurants + normalized_rating * learning_factor,
                    ),
                )

    profile.trips_count += 1
    save_traveler_profile(profile)


def _categorize_attractions(attractions: list[dict[str, Any]]) -> dict[str, str]:
    """Kategoryzuj atrakcje na podstawie typów i nazw."""
    categories = {}

    for attraction in attractions:
        attraction_id = attraction.get("id", "")
        types = attraction.get("types", [])
        name = attraction.get("name", "").lower()

        category = "unknown"

        if any(t in ["museum", "art_gallery", "library"] for t in types):
            category = "museum"
        elif any(t in ["park", "natural_feature", "zoo"] for t in types):
            category = "nature"
        elif any(t in ["night_club", "bar", "casino"] for t in types):
            category = "nightlife"
        elif any(t in ["shopping_mall", "store", "market"] for t in types):
            category = "shopping"
        elif any(t in ["restaurant", "food", "cafe"] for t in types):
            category = "food"
        elif any(
            t in ["church", "mosque", "synagogue", "temple", "historical"]
            for t in types
        ):
            category = "historical"
        elif any(keyword in name for keyword in ["museum", "galeria", "muzeum"]):
            category = "museum"
        elif any(keyword in name for keyword in ["park", "ogród", "las", "plaża"]):
            category = "nature"

        categories[attraction_id] = category

    return categories


def get_contextual_recommendations(
    profile: TravelerProfile,
    weather: dict[str, Any],
    time_of_day: str,
    attractions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generuj rekomendacje kontekstowe na podstawie pogody i preferencji."""
    recommendations = []

    # Analiza pogody
    is_rainy = weather.get("condition", "").lower() in ["rain", "drizzle", "storm"]
    is_hot = weather.get("temperature", 20) > 28
    is_cold = weather.get("temperature", 20) < 5

    for attraction in attractions:
        score = 0.5  # Base score

        # Weather-based adjustments
        types = attraction.get("types", [])
        is_indoor = any(
            t in ["museum", "shopping_mall", "restaurant", "theater"] for t in types
        )
        is_outdoor = any(
            t in ["park", "beach", "hiking_area", "viewpoint"] for t in types
        )

        if is_rainy:
            if is_indoor:
                score += 0.3
            elif is_outdoor:
                score -= 0.4

        if is_hot and is_outdoor:
            score -= 0.2

        if is_cold and is_outdoor:
            score -= 0.2

        # Profile-based scoring
        attraction_type = _categorize_attractions([attraction]).get(
            attraction.get("id", ""), "unknown"
        )

        if attraction_type == "museum":
            score += profile.likes_museums - 0.5
        elif attraction_type == "nature":
            score += profile.likes_nature - 0.5
        elif attraction_type == "nightlife":
            score += profile.likes_nightlife - 0.5
        elif attraction_type == "restaurant":
            score += profile.likes_restaurants - 0.5
        elif attraction_type == "historical":
            score += profile.likes_museums - 0.5  # Historical sites similar to museums

        # Time-based adjustments
        if time_of_day == "morning" and attraction_type == "museum":
            score += 0.2
        elif time_of_day == "evening" and attraction_type in [
            "restaurant",
            "nightlife",
        ]:
            score += 0.2

        # Crowded places tolerance
        is_popular = (
            attraction.get("rating", 0) > 4.5 and attraction.get("reviews", 0) > 1000
        )
        if is_popular:
            score += (profile.crowded_places_tolerance - 0.5) * 0.3

        attraction["personalized_score"] = max(0, min(1, score))
        recommendations.append(attraction)

    # Sort by personalized score
    recommendations.sort(key=lambda x: x.get("personalized_score", 0), reverse=True)
    return recommendations


def generate_personalized_narrative(
    profile: TravelerProfile,
    city: str,
    attractions: list[dict[str, Any]],
    style: str = "friendly",
) -> str:
    """Generuj spersonalizowaną narrację planu podróży."""

    # Wybierz styl narracji na podstawie profilu
    if style == "auto":
        if profile.luxury_preference > 0.7:
            style = "elegant"
        elif profile.travel_pace == "fast":
            style = "energetic"
        elif profile.likes_historical_sites > 0.7:
            style = "scholarly"
        else:
            style = "friendly"

    # Style templates
    style_intros = {
        "friendly": f"Hej! Przygotowałem dla Ciebie plan zwiedzania {city} ✨",
        "elegant": f"Ekskluzywny przewodnik po {city} dostosowany do Twoich wymagań.",
        "energetic": f"🚀 Gotowy na przygodę w {city}? Let's go!",
        "scholarly": f"Kulturalno-historyczny przewodnik po {city} z głębokim kontekstem.",
        "casual": f"Co robimy w {city}? Sprawdźmy najlepsze miejsca!",
        "professional": f"Profesjonalny itinerary dla {city} z optymalizacją czasu i tras.",
    }

    style_transitions = {
        "friendly": ["Dalej", "Następnie", "Po tym", "Warto też"],
        "elegant": [
            "Następnie proponuję",
            "Kolejnym punktem",
            "Warto również",
            "Pozwalam sobie zasugerować",
        ],
        "energetic": ["Hop hop!", "Lecimy dalej!", "Next stop!", "Energia!"],
        "scholarly": [
            "Następnie",
            "Warto również",
            "Z historycznego punktu widzenia",
            "Kontynuując",
        ],
        "casual": ["No to", "Okej, dalej", "Btw", "A jeszcze"],
        "professional": [
            "Kolejny punkt",
            "Następnie",
            "Zgodnie z planem",
            "Optymalna kolejność",
        ],
    }

    intro = style_intros.get(style, style_intros["friendly"])
    transitions = style_transitions.get(style, style_transitions["friendly"])

    # Personalizacja na podstawie profilu
    narrative = [intro]

    if profile.name:
        narrative[0] = narrative[0].replace("Ciebie", profile.name)

    # Dodaj informacje o preferencjach
    preferences_text = _generate_preferences_text(profile)
    if preferences_text:
        narrative.append(f"\n{preferences_text}")

    # Generuj opis atrakcji z narratywą
    for i, attraction in enumerate(attractions[:8]):  # Top 8 attractions
        transition = random.choice(transitions) if i > 0 else ""
        attraction_desc = _generate_attraction_narrative(attraction, profile, style)

        if transition and i > 0:
            narrative.append(f"\n{transition}: {attraction_desc}")
        else:
            narrative.append(f"\n{attraction_desc}")

    return " ".join(narrative)


def _generate_preferences_text(profile: TravelerProfile) -> str:
    """Generuj tekst o preferencjach użytkownika."""
    parts = []

    if profile.likes_museums > 0.7:
        parts.append("uwielbiasz muzea i kulturę")
    elif profile.likes_museums < 0.3:
        parts.append("preferujesz unikać muzeów")

    if profile.likes_nature > 0.7:
        parts.append("kochasz naturę i przestrzeń")

    if profile.crowded_places_tolerance < 0.3:
        parts.append("unikasz zatłoczonych miejsc")
    elif profile.crowded_places_tolerance > 0.7:
        parts.append("nie przeszkadzają Ci tłumy")

    if profile.budget_level == "low":
        parts.append("szukasz budżetowych opcji")
    elif profile.budget_level in ["high", "luxury"]:
        parts.append("cenisz komfort i jakość")

    if parts:
        return f"Wiem, że {', '.join(parts[:3])}, więc dostosowałem plan specjalnie dla Ciebie."

    return ""


def _generate_attraction_narrative(
    attraction: dict[str, Any], profile: TravelerProfile, style: str
) -> str:
    """Generuj narracyjny opis atrakcji."""
    name = attraction.get("name", "Ta atrakcja")
    rating = attraction.get("rating", 0)
    description = attraction.get("description", "")

    # Style-specific descriptions
    if style == "elegant":
        prefix = "Pozwolę sobie polecić"
        if rating > 4.5:
            quality = "wyjątkowy"
        elif rating > 4.0:
            quality = "renomowany"
        else:
            quality = "interesujący"
    elif style == "energetic":
        prefix = "Musisz zobaczyć"
        if rating > 4.5:
            quality = "mega wypasiony"
        elif rating > 4.0:
            quality = "super"
        else:
            quality = "fajny"
    elif style == "scholarly":
        prefix = "Warto zwrócić uwagę na"
        if rating > 4.5:
            quality = "uznany"
        elif rating > 4.0:
            quality = "ważny"
        else:
            quality = "znaczący"
    else:  # friendly, casual, professional
        prefix = "Polecam"
        if rating > 4.5:
            quality = "fantastyczny"
        elif rating > 4.0:
            quality = "świetny"
        else:
            quality = "ciekawy"

    # Personalized touches
    attraction_type = _categorize_attractions([attraction]).get(
        attraction.get("id", ""), "unknown"
    )
    personal_note = ""

    if attraction_type == "museum" and profile.likes_museums > 0.7:
        personal_note = " – idealny dla Ciebie jako miłośnika kultury!"
    elif attraction_type == "nature" and profile.likes_nature > 0.7:
        personal_note = " – to miejsce na pewno Ci się spodoba!"
    elif profile.crowded_places_tolerance < 0.3 and attraction.get("reviews", 0) > 2000:
        personal_note = " (uwaga: może być zatłoczony, ale warto!"

    desc_snippet = description[:100] + "..." if len(description) > 100 else description

    return f"{prefix} {quality} {name}{personal_note} {desc_snippet}".strip()


def html_export(title: str, body_md: str) -> str:
    def md2html(md: str) -> str:
        t = html.escape(md)
        t = re.sub(r"^# (.+)$", r"<h1>\1</h1>", t, flags=re.M)
        t = re.sub(r"^## (.+)$", r"<h2>\1</h2>", t, flags=re.M)
        t = re.sub(r"^### (.+)$", r"<h3>\1</h3>", t, flags=re.M)
        t = re.sub(r"(?m)^\- (.+)$", r"<li>\1</li>", t)
        t = re.sub(r"(<li>.*</li>)", r"<ul>\1</ul>", t, flags=re.S)
        t = re.sub(
            r"\n\|([^\n]+)\|\n\|([\-:\| ]+)\|\n((?:\|.*\|\n?)+)",
            (
                r"\n<table><thead><tr><th>\1</th></tr></thead>"
                r"<tbody>\3</tbody></table>"
            ),
            t,
        )
        return t.replace("\n\n", "<br/><br/>")

    css = (
        "<style>body{font-family:system-ui,Segoe UI,Roboto,Ubuntu,sans-serif;"
        "max-width:960px;margin:40px auto;padding:0 20px;line-height:1.6}"
        "</style>"
    )
    return (
        f"<!doctype html><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>{css}"
        f"<body>{md2html(body_md)}</body>"
    )


def export_pdf(text: str, out_path: str) -> str | None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.pdfgen import canvas
    except Exception:
        return None
    c = canvas.Canvas(out_path, pagesize=A4)
    w, h = A4
    y = h - 2 * cm
    for ln in text.splitlines():
        c.drawString(2 * cm, y, ln[:110])
        y -= 14
        if y < 2 * cm:
            c.showPage()
            y = h - 2 * cm
    c.save()
    return out_path


def export_formats(
    title: str,
    body_md: str,
    meta: dict[str, Any],
    base_path: str,
    formats: list[str] | None = None,
) -> dict[str, str]:
    """Export content to multiple formats and optionally create ZIP package."""
    if formats is None:
        formats = ["md", "html", "pdf", "json"]
    import zipfile

    base_path_obj = Path(base_path)
    exported = {}

    # Markdown (base format)
    if "md" in formats:
        md_path = base_path_obj.with_suffix(".md")
        md_path.write_text(body_md, encoding="utf-8")
        exported["md"] = str(md_path)

    # HTML
    if "html" in formats:
        html_path = base_path_obj.with_suffix(".html")
        html_content = html_export(title, body_md)
        html_path.write_text(html_content, encoding="utf-8")
        exported["html"] = str(html_path)

    # PDF
    if "pdf" in formats:
        pdf_path = base_path_obj.with_suffix(".pdf")
        if export_pdf(body_md, str(pdf_path)):
            exported["pdf"] = str(pdf_path)

    # JSON metadata
    if "json" in formats:
        json_path = base_path_obj.with_suffix(".json")
        json_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        exported["json"] = str(json_path)

    # ZIP package
    if "zip" in formats and len(exported) > 1:
        zip_path = base_path_obj.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for _format_name, file_path in exported.items():
                if Path(file_path).exists():
                    zf.write(file_path, Path(file_path).name)
        exported["zip"] = str(zip_path)

    return exported


# ── ENV / HTTP ────────────────────────────────────────────────────────────────
UA = os.getenv("WEB_USER_AGENT", "Overmind/TravelGuideULTRA/6.0")
TIMEOUT = int(os.getenv("WEB_HTTP_TIMEOUT", "28"))
GOOGLE = os.getenv("GOOGLE_MAPS_KEY", "").strip()
TA_KEY = os.getenv("TRIPADVISOR_KEY", "").strip()
OTM_KEY = os.getenv("OTM_KEY", "").strip()
TP_TOKEN = os.getenv("TP_TOKEN", "").strip()
XW_SEC = os.getenv("XWEATHER_SECRET", "").strip()
TL_KEY = os.getenv("TRANSITLAND_API_KEY", "").strip()

S = requests.Session()
S.headers.update({"User-Agent": UA})


def _ck(url: str, params: dict | None = None, headers: dict | None = None) -> str:
    return hashlib.sha256(
        (
            url
            + "|"
            + json.dumps(params or {}, sort_keys=True)
            + "|"
            + json.dumps(headers or {}, sort_keys=True)
        ).encode()
    ).hexdigest()


def _get(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    ttl: int | None = None,
    source: str = "default",
    max_retries: int = 3,
) -> requests.Response:
    """
    Enhanced HTTP GET with cache, TTL per source, retry with exponential backoff.
    """
    global CACHE

    # Używaj TTL dla źródła jeśli nie podano
    if ttl is None:
        ttl = SOURCE_TTL.get(source, DEFAULT_TTL)

    key = _ck(url, params, headers)
    entries = CACHE.get("entries", {})

    # Sprawdź cache
    if key in entries:
        entry = entries[key]
        age = time.time() - entry.get("ts", 0)
        if age < ttl:
            r = requests.Response()
            r.status_code = entry["st"]
            r._content = entry["body"].encode()
            r.headers["X-Cache"] = "HIT"
            r.headers["X-Cache-Age"] = str(int(age))
            return r

    # Retry with exponential backoff
    last_exception = None
    for attempt in range(max_retries):
        try:
            r = S.get(url, params=params, headers=headers, timeout=TIMEOUT)

            # Cache successful and client error responses (not server errors)
            if r.status_code < 500:
                entries[key] = {
                    "ts": time.time(),
                    "st": r.status_code,
                    "body": r.text[:2_000_000],  # Limit response size
                    "source": source,
                }

                # Wyczyść cache co jakiś czas
                if len(entries) % 50 == 0:  # Co 50 nowych wpisów
                    CACHE = _clean_cache(CACHE)
                    entries = CACHE.get("entries", {})
                else:
                    CACHE["entries"] = entries

                _write_cache(CACHE)

            # Rate limiting handling
            if r.status_code in (429, 503):
                retry_after = int(r.headers.get("Retry-After", 60))
                if attempt < max_retries - 1:
                    sleep_time = min(retry_after, 2**attempt + random.random())
                    time.sleep(sleep_time)
                    continue

            return r

        except requests.RequestException as e:
            last_exception = e
            if attempt < max_retries - 1:
                sleep_time = 2**attempt + random.random()
                time.sleep(sleep_time)
                continue

    # Fallback response po wyczerpaniu prób
    r = requests.Response()
    r.status_code = 503
    r._content = json.dumps(
        {
            "error": "max_retries_exceeded",
            "attempts": max_retries,
            "last_error": str(last_exception),
        }
    ).encode()
    return r


def _post(url: str, data: dict | str, headers: dict | None = None) -> requests.Response:
    return S.post(url, data=data, headers=headers or {}, timeout=TIMEOUT)


# ── Optional deps (memory/psyche/autonauka) ───────────────────────────────────
def _opt(name: str):
    try:
        return __import__(name, fromlist=["*"])
    except Exception:
        return None


memory = _opt("memory")
psychika = _opt("psychika")
autonauka = _opt("autonauka")


def mem_add(title: str, payload: Any = None, tags: list[str] | None = None) -> None:
    try:
        if memory and hasattr(memory, "ltm_add_sync"):
            memory.ltm_add_sync(
                title[:200],
                sources=[payload if payload is not None else {"text": title}],
                user="global",
                tags=tags or ["travel"],
            )
    except Exception:
        pass


def psyche_mood() -> str:
    try:
        if psychika and hasattr(psychika, "psychika_preload"):
            mood = (psychika.psychika_preload("global") or {}).get("mood", "spokój")
            return mood
    except Exception:
        pass
    return "spokój"


def auto_learn(sample: dict[str, Any]) -> None:
    try:
        if autonauka:
            for fn in ("add_sample", "learn", "enqueue"):
                if hasattr(autonauka, fn):
                    getattr(autonauka, fn)(sample)
    except Exception:
        pass


# ── Geo / Timezone ────────────────────────────────────────────────────────────
def geocode(place: str) -> dict[str, Any] | None:
    r = _get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": place, "format": "json", "limit": 1},
        source="osm",
    )
    if r.status_code >= 400:
        return None
    j = r.json()
    if not j:
        return None
    it = j[0]
    lat = float(it["lat"])
    lon = float(it["lon"])
    bb = it["boundingbox"]
    south, north = float(bb[0]), float(bb[1])
    west, east = lon - 0.24, lon + 0.24
    tzr = _get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m",
            "forecast_days": 1,
            "timezone": "auto",
        },
        source="weather",
    )
    tz = (tzr.json() or {}).get("timezone", "auto")
    return {
        "lat": lat,
        "lon": lon,
        "bbox": (south, north, west, east),
        "display": it.get("display_name", ""),
        "tz": tz,
    }


# ── Helpers: distance, clustering, hours ──────────────────────────────────────
def haversine(a: tuple[float, float], b: tuple[float, float]) -> float:
    R = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    dlat = la2 - la1
    dlon = lo2 - lo1
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(h))  # km


def greedy_cluster(
    items: list[dict[str, Any]],
    days: int,
    per: int = 6,
    center: tuple[float, float] | None = None,
) -> list[list[dict[str, Any]]]:
    work = items[:]
    if center:
        work.sort(
            key=lambda it: haversine(
                center, (it.get("lat") or 0.0, it.get("lon") or 0.0)
            )
        )
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(max(1, days))]

    di = 0
    for it in work:
        buckets[di % days].append(it)
        di += 1 if len(buckets[di % days]) >= per else 0
    # round-robin redistribute leftovers
    flat = [x for x in work[days * per :]]
    for i, it in enumerate(flat):
        buckets[i % days].append(it)
    return buckets


def format_open_hours(opening: list[str] | None, day_index: int) -> str:
    if not opening:
        return ""
    idx = day_index % len(opening)
    line = opening[idx]
    line = re.sub(r"^[A-Za-ząćęłńóśźżĄĆĘŁŃÓŚŹŻ]{2,}\s*:\s*", "", line)
    return f"godz.: {line}"


# ── Data Unification Adapters ─────────────────────────────────────────────────
@dataclass
class TravelerProfile:
    """Profil podróżnika z preferencjami i historią uczenia się."""

    user_id: str
    name: str = ""

    # Preferencje podstawowe
    likes_museums: float = 0.5  # 0-1 skala
    likes_nature: float = 0.5
    likes_nightlife: float = 0.5
    likes_shopping: float = 0.5
    likes_food_experiences: float = 0.5
    likes_historical_sites: float = 0.5
    likes_modern_attractions: float = 0.5

    # Charakterystyka podróżowania
    crowded_places_tolerance: float = 0.5  # 0=nie znosi, 1=lubi tłumy
    budget_level: str = "medium"  # low, medium, high, luxury
    travel_pace: str = "moderate"  # slow, moderate, fast
    luxury_preference: float = 0.5  # 0=backpacker, 1=luksus

    # Ograniczenia i potrzeby
    mobility_restrictions: bool = False
    dietary_restrictions: list[str] | None = None
    language_preferences: list[str] | None = None

    # Historia i uczenie się
    visited_cities: list[str] | None = None
    favorite_attraction_types: list[str] | None = None
    disliked_attraction_types: list[str] | None = None

    # Preferencje kontekstowe
    weather_preferences: dict[str, Any] | None = (
        None  # {"rain": "indoor", "heat": "airconditioned"}
    )
    time_preferences: dict[str, Any] | None = (
        None  # {"morning": "museums", "evening": "restaurants"}
    )

    # Statystyki uczenia się
    trips_count: int = 0
    last_updated: float = 0.0
    learning_scores: dict[str, float] | None = (
        None  # np. {"museums": 0.8, "nature": 0.3}
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "user_id": self.user_id,
            "name": self.name,
            "likes_museums": self.likes_museums,
            "likes_nature": self.likes_nature,
            "likes_nightlife": self.likes_nightlife,
            "likes_shopping": self.likes_shopping,
            "likes_food_experiences": self.likes_food_experiences,
            "likes_historical_sites": self.likes_historical_sites,
            "likes_modern_attractions": self.likes_modern_attractions,
            "crowded_places_tolerance": self.crowded_places_tolerance,
            "budget_level": self.budget_level,
            "travel_pace": self.travel_pace,
            "luxury_preference": self.luxury_preference,
            "mobility_restrictions": self.mobility_restrictions,
            "dietary_restrictions": self.dietary_restrictions or [],
            "language_preferences": self.language_preferences or [],
            "visited_cities": self.visited_cities or [],
            "favorite_attraction_types": self.favorite_attraction_types or [],
            "disliked_attraction_types": self.disliked_attraction_types or [],
            "weather_preferences": self.weather_preferences or {},
            "time_preferences": self.time_preferences or {},
            "trips_count": self.trips_count,
            "last_updated": self.last_updated,
            "learning_scores": self.learning_scores or {},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TravelerProfile:
        """Create from dictionary."""
        return cls(
            user_id=data["user_id"],
            name=data.get("name", ""),
            likes_museums=data.get("likes_museums", 0.5),
            likes_nature=data.get("likes_nature", 0.5),
            likes_nightlife=data.get("likes_nightlife", 0.5),
            likes_shopping=data.get("likes_shopping", 0.5),
            likes_food_experiences=data.get("likes_food_experiences", 0.5),
            likes_historical_sites=data.get("likes_historical_sites", 0.5),
            likes_modern_attractions=data.get("likes_modern_attractions", 0.5),
            crowded_places_tolerance=data.get("crowded_places_tolerance", 0.5),
            budget_level=data.get("budget_level", "medium"),
            travel_pace=data.get("travel_pace", "moderate"),
            luxury_preference=data.get("luxury_preference", 0.5),
            mobility_restrictions=data.get("mobility_restrictions", False),
            dietary_restrictions=data.get("dietary_restrictions"),
            language_preferences=data.get("language_preferences"),
            visited_cities=data.get("visited_cities"),
            favorite_attraction_types=data.get("favorite_attraction_types"),
            disliked_attraction_types=data.get("disliked_attraction_types"),
            weather_preferences=data.get("weather_preferences"),
            time_preferences=data.get("time_preferences"),
            trips_count=data.get("trips_count", 0),
            last_updated=data.get("last_updated", 0.0),
            learning_scores=data.get("learning_scores"),
        )


@dataclass
class UnifiedPlace:
    """Unified data structure for all place sources."""

    name: str
    address: str = ""
    lat: float | None = None
    lon: float | None = None
    rating: float | None = None
    reviews: int = 0
    price_level: int | None = None
    url: str = ""
    phone: str = ""
    website: str = ""
    opening_hours: list[str] | None = None
    photos: list[str] | None = None
    description: str = ""
    source: str = ""
    types: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, filtering None values."""
        return {
            k: v
            for k, v in {
                "name": self.name,
                "address": self.address,
                "lat": self.lat,
                "lon": self.lon,
                "rating": self.rating,
                "reviews": self.reviews,
                "price_level": self.price_level,
                "url": self.url,
                "phone": self.phone,
                "website": self.website,
                "opening_hours": self.opening_hours,
                "photos": self.photos,
                "description": self.description,
                "source": self.source,
                "types": self.types,
            }.items()
            if v is not None and v != ""
        }


def adapt_google_place(item: dict[str, Any]) -> UnifiedPlace:
    """Adapt Google Places data to unified format."""
    return UnifiedPlace(
        name=item.get("name", ""),
        address=item.get("address", "") or item.get("formatted_address", ""),
        lat=item.get("lat"),
        lon=item.get("lon"),
        rating=item.get("rating"),
        reviews=item.get("reviews", 0) or item.get("user_ratings_total", 0),
        price_level=item.get("price_level"),
        url=item.get("map_url", ""),
        phone=item.get("phone", ""),
        website=item.get("website", ""),
        opening_hours=item.get("opening"),
        photos=item.get("photos"),
        source="google",
        types=item.get("types", []),
    )


def adapt_tripadvisor_place(item: dict[str, Any]) -> UnifiedPlace:
    """Adapt TripAdvisor data to unified format."""
    return UnifiedPlace(
        name=item.get("name", ""),
        address=item.get("address", ""),
        rating=item.get("rating"),
        reviews=item.get("reviews", 0),
        url=item.get("map_url", ""),
        description=item.get("ranking", ""),
        source="tripadvisor",
    )


def adapt_otm_place(item: dict[str, Any]) -> UnifiedPlace:
    """Adapt OpenTripMap data to unified format."""
    return UnifiedPlace(
        name=item.get("name", ""),
        lat=item.get("lat"),
        lon=item.get("lon"),
        url=item.get("url", "")
        or f"https://www.opentripmap.com/en/card/{item.get('xid', '')}",
        description=item.get("desc", ""),
        source="otm",
        types=[item.get("kind", "")],
    )


def adapt_osm_place(item: dict[str, Any]) -> UnifiedPlace:
    """Adapt OpenStreetMap data to unified format."""
    return UnifiedPlace(
        name=item.get("name", ""),
        lat=item.get("lat"),
        lon=item.get("lon"),
        website=item.get("website", ""),
        source="osm",
        types=[item.get("kind", "")],
    )


def unify_places(places: list[dict[str, Any]], source: str) -> list[UnifiedPlace]:
    """Convert list of places from any source to unified format."""
    adapters = {
        "google": adapt_google_place,
        "tripadvisor": adapt_tripadvisor_place,
        "otm": adapt_otm_place,
        "osm": adapt_osm_place,
    }

    adapter = adapters.get(source)
    if not adapter:
        # Generic adapter dla nieznanych źródeł
        return [
            UnifiedPlace(
                name=item.get("name", ""),
                address=item.get("address", ""),
                lat=item.get("lat"),
                lon=item.get("lon"),
                rating=item.get("rating"),
                reviews=item.get("reviews", 0),
                url=item.get("url", "") or item.get("map_url", ""),
                source=source,
            )
            for item in places
        ]

    return [adapter(item) for item in places if item.get("name")]


# ── Wikipedia ─────────────────────────────────────────────────────────────────
def wiki_summary(title: str, lang: str = "pl") -> dict[str, str]:
    if not title:
        return {}
    try:
        url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ','_')}"
        r = _get(url, source="wiki")
        if r.status_code == 404:
            return {}
        d = r.json()
        return {
            "title": d.get("title", ""),
            "extract": d.get("extract", ""),
            "url": ((d.get("content_urls") or {}).get("desktop") or {}).get("page", ""),
        }
    except Exception:
        return {}


# ── OpenTripMap / Overpass ───────────────────────────────────────────────────
def otm_bbox(
    bbox: tuple[float, float, float, float], kind: str, limit: int = 60
) -> list[dict[str, Any]]:
    if not OTM_KEY:
        return []
    s, n, w, e = bbox
    r = _get(
        "https://api.opentripmap.com/0.1/en/places/bbox",
        params={
            "apikey": OTM_KEY,
            "lon_min": w,
            "lat_min": s,
            "lon_max": e,
            "lat_max": n,
            "kinds": kind,
            "limit": limit,
        },
        source="otm",
    )
    if r.status_code >= 400:
        return []
    out = []
    for it in r.json() or []:
        out.append(
            {
                "name": it.get("name", ""),
                "kind": kind,
                "lat": (it.get("point") or {}).get("lat"),
                "lon": (it.get("point") or {}).get("lon"),
                "xid": it.get("xid", ""),
            }
        )
    return [x for x in out if x.get("lat") and x.get("lon") and x.get("name")]


def otm_enrich(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not OTM_KEY:
        return items
    out = []
    for it in items[:120]:
        xid = it.get("xid")
        it = dict(it)
        if xid:
            r = _get(
                f"https://api.opentripmap.com/0.1/en/places/xid/{xid}",
                params={"apikey": OTM_KEY},
                source="otm",
            )
            if r.status_code < 400:
                d = r.json()
                it["desc"] = (d.get("wikipedia_extracts") or {}).get("text", "") or (
                    d.get("info") or {}
                ).get("descr", "")
                it["url"] = d.get("otm", "")
        out.append(it)
    return out


def overpass_poi(
    bbox: tuple[float, float, float, float],
    selectors: list[str],
    limit: int = 120,
) -> list[dict[str, Any]]:
    s, n, w, e = bbox
    core = ";\n".join([f"({sel}({s},{w},{n},{e});)" for sel in selectors])
    q = f"[out:json][timeout:25];({core});out center {limit};"
    r = _post("https://overpass-api.de/api/interpreter", data={"data": q})
    if r.status_code >= 400:
        return []
    out = []
    for el in r.json().get("elements") or []:
        tags = el.get("tags") or {}
        out.append(
            {
                "name": tags.get("name") or tags.get("name:en") or "",
                "kind": tags.get("tourism")
                or tags.get("amenity")
                or tags.get("historic")
                or tags.get("natural")
                or "poi",
                "lat": el.get("lat") or (el.get("center") or {}).get("lat"),
                "lon": el.get("lon") or (el.get("center") or {}).get("lon"),
                "website": tags.get("website", ""),
            }
        )
    return [
        x
        for x in out
        if x.get("lat") and x.get("lon") and (x.get("name") or "").strip()
    ]


# ── Google Places (TextSearch + Details) ──────────────────────────────────────
def g_textsearch(
    query: str,
    typ: str | None = None,
    max_results: int = 20,
    open_now: bool = False,
    lang: str = "pl",
) -> list[dict[str, Any]]:
    if not GOOGLE:
        return []
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": query, "key": GOOGLE, "language": lang}
    if typ:
        params["type"] = typ
    if open_now:
        params["opennow"] = "true"
    out = []
    page = 0
    while True:
        page += 1
        r = _get(url, params=params, source="google")
        if r.status_code >= 400:
            break
        j = r.json()
        out.extend(
            [
                {
                    "name": it.get("name", ""),
                    "address": it.get("formatted_address", ""),
                    "rating": it.get("rating", None),
                    "reviews": it.get("user_ratings_total", 0),
                    "price_level": it.get("price_level", None),
                    "place_id": it.get("place_id", ""),
                    "types": it.get("types", []),
                    "lat": (it.get("geometry") or {}).get("location", {}).get("lat"),
                    "lon": (it.get("geometry") or {}).get("location", {}).get("lng"),
                    "map_url": (
                        "https://www.google.com/maps/place/?q=place_id:"
                        f"{it.get('place_id','')}"
                        if it.get("place_id")
                        else ""
                    ),
                }
                for it in j.get("results", [])
            ]
        )
        if len(out) >= max_results:
            break
        token = j.get("next_page_token")
        if not token or page >= 3:
            break
        time.sleep(2.1)
        params = {"pagetoken": token, "key": GOOGLE}
    return out[:max_results]


def g_details(place_id: str, lang: str = "pl") -> dict[str, Any]:
    if not (GOOGLE and place_id):
        return {}
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    r = _get(
        url,
        params={
            "place_id": place_id,
            "key": GOOGLE,
            "language": lang,
            "fields": "formatted_phone_number,opening_hours,website,"
            "price_level,photos",
        },
        source="google",
    )
    if r.status_code >= 400:
        return {}
    d = (r.json() or {}).get("result") or {}
    photos = []
    for p in (d.get("photos") or [])[:3]:
        ref = p.get("photo_reference")
        if ref:
            photos.append(
                "https://maps.googleapis.com/maps/api/place/photo"
                f"?maxwidth=1200&photo_reference={ref}&key={GOOGLE}"
            )
    return {
        "phone": d.get("formatted_phone_number", ""),
        "website": d.get("website", ""),
        "opening": (d.get("opening_hours") or {}).get("weekday_text", []),
        "price_level": d.get("price_level", None),
        "photos": photos,
    }


def restaurants_google(
    place: str,
    q: str = "",
    max_results: int = 24,
    open_now: bool = False,
    lang: str = "pl",
) -> list[dict[str, Any]]:
    base = g_textsearch(
        (f"{q} restaurant in {place}") if q else (f"restaurants in {place}"),
        "restaurant",
        max_results,
        open_now,
        lang,
    )
    out = []
    for it in base:
        it.update(g_details(it.get("place_id", ""), lang=lang))
        out.append(it)
    return out


def hotels_google(
    place: str, q: str = "", max_results: int = 24, lang: str = "pl"
) -> list[dict[str, Any]]:
    base = g_textsearch(
        (f"{q} lodging in {place}") if q else (f"hotels in {place}"),
        "lodging",
        max_results,
        False,
        lang,
    )
    out = []
    for it in base:
        it.update(g_details(it.get("place_id", ""), lang=lang))
        out.append(it)
    return out


# ── TripAdvisor (partner v2) ─────────────────────────────────────────────────
def ta_map(
    lat: float, lon: float, category: str, limit: int = 40
) -> list[dict[str, Any]]:
    if not TA_KEY:
        return []
    url = f"https://api.tripadvisor.com/api/partner/2.0/map/{lat},{lon}/{category}"
    r = _get(
        url,
        headers={"X-TripAdvisor-API-Key": TA_KEY},
        params={"limit": max(1, min(50, limit))},
        source="tripadvisor",
    )
    if r.status_code >= 400:
        return []
    items = []
    for it in r.json().get("data") or []:
        items.append(
            {
                "name": it.get("name", ""),
                "address": (it.get("address_obj") or {}).get("address_string", ""),
                "rating": (it.get("rating") or None),
                "reviews": (it.get("num_reviews") or 0),
                "map_url": it.get("web_url", ""),
                "ranking": it.get("ranking", ""),
            }
        )
    return [x for x in items if x["name"]]


def ta_restaurants(lat: float, lon: float, limit: int = 30) -> list[dict[str, Any]]:
    return ta_map(lat, lon, "restaurants", limit)


def ta_hotels(lat: float, lon: float, limit: int = 30) -> list[dict[str, Any]]:
    return ta_map(lat, lon, "hotels", limit)


def ta_attractions(lat: float, lon: float, limit: int = 50) -> list[dict[str, Any]]:
    return ta_map(lat, lon, "attractions", limit)


# ── Merge helpers ────────────────────────────────────────────────────────────
def _unique_by_name(places: list[UnifiedPlace], k: int = 60) -> list[UnifiedPlace]:
    """Remove duplicates based on name and address similarity."""
    out = []
    seen = set()
    for place in places:
        key = (
            place.name.strip().lower(),
            place.address.strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(place)
        if len(out) >= k:
            break
    return out


def merge_place_sources(
    *source_lists: tuple[list[UnifiedPlace], str]
) -> list[UnifiedPlace]:
    """
    Merge multiple place sources prioritizing by quality and completeness.
    Args: (places_list, source_name) tuples
    """
    all_places: list[UnifiedPlace] = []
    for places, _source in source_lists:
        if places:
            all_places.extend(places)

    # Sort by completeness score (more data = better)
    def completeness_score(place: UnifiedPlace) -> int:
        score = 0
        if place.rating is not None:
            score += 3
        if place.reviews > 0:
            score += 2
        if place.address:
            score += 2
        if place.phone:
            score += 1
        if place.website:
            score += 1
        if place.opening_hours:
            score += 1
        if place.description:
            score += 1
        return score

    sorted_places = sorted(all_places, key=completeness_score, reverse=True)
    return _unique_by_name(sorted_places)


# ── Food / Hotels / Attractions (combo) ───────────────────────────────────────
def restaurants(
    place: str,
    q: str = "",
    max_results: int = 24,
    open_now: bool = False,
    lang: str = "pl",
) -> list[dict[str, Any]]:
    geo = geocode(place)

    # Google Places
    google_raw = (
        restaurants_google(place, q, max_results, open_now, lang) if GOOGLE else []
    )
    google_places = unify_places(google_raw, "google")

    # TripAdvisor
    ta_raw = (
        ta_restaurants(geo["lat"], geo["lon"], max_results) if geo and TA_KEY else []
    )
    ta_places = unify_places(ta_raw, "tripadvisor")

    # Merge premium sources
    if google_places or ta_places:
        merged = merge_place_sources(
            (google_places, "google"), (ta_places, "tripadvisor")
        )[:max_results]
        return [place.to_dict() for place in merged]

    # Fallback to free sources
    if not geo:
        return []

    # OpenTripMap fallback
    otm_raw = otm_bbox(geo["bbox"], "foods", limit=max_results)
    if otm_raw:
        otm_places = unify_places(otm_raw, "otm")
        return [place.to_dict() for place in otm_places[:max_results]]

    # OSM fallback
    sels = ['node["amenity"="restaurant"]', 'node["amenity"="bar"]']
    osm_raw = overpass_poi(geo["bbox"], sels, limit=max_results)
    osm_places = unify_places(osm_raw, "osm")
    return [place.to_dict() for place in osm_places[:max_results]]


def hotels(
    place: str, q: str = "", max_results: int = 24, lang: str = "pl"
) -> list[dict[str, Any]]:
    geo = geocode(place)

    # Google Places
    google_raw = hotels_google(place, q, max_results, lang) if GOOGLE else []
    google_places = unify_places(google_raw, "google")

    # TripAdvisor
    ta_raw = ta_hotels(geo["lat"], geo["lon"], max_results) if geo and TA_KEY else []
    ta_places = unify_places(ta_raw, "tripadvisor")

    # Merge premium sources
    if google_places or ta_places:
        merged = merge_place_sources(
            (google_places, "google"), (ta_places, "tripadvisor")
        )[:max_results]
        return [place.to_dict() for place in merged]

    # Fallback to free sources
    if not geo:
        return []

    # OpenTripMap fallback
    otm_raw = otm_bbox(geo["bbox"], "other_hotels", limit=max_results)
    if otm_raw:
        otm_places = unify_places(otm_raw, "otm")
        return [place.to_dict() for place in otm_places[:max_results]]

    # OSM fallback
    sels = ['node["tourism"="hotel"]', 'node["amenity"="lodging"]']
    osm_raw = overpass_poi(geo["bbox"], sels, limit=max_results)
    osm_places = unify_places(osm_raw, "osm")
    return [place.to_dict() for place in osm_places[:max_results]]


def attractions(
    place: str, max_results: int = 80, lang: str = "pl"
) -> list[dict[str, Any]]:
    geo = geocode(place)
    if not geo:
        return []

    all_places: list[UnifiedPlace] = []

    # TripAdvisor attractions
    if TA_KEY:
        ta_raw = ta_attractions(geo["lat"], geo["lon"], limit=40)
        ta_places = unify_places(ta_raw, "tripadvisor")
        all_places.extend(ta_places)

    # OpenTripMap attractions
    kinds = [
        "interesting_places",
        "museums",
        "historic",
        "architecture",
        "natural",
        "parks",
        "beaches",
        "view_points",
    ]
    otm_raw = []
    for k in kinds:
        otm_raw.extend(otm_bbox(geo["bbox"], k, limit=16))

    if otm_raw:
        otm_enriched = otm_enrich(otm_raw)
        otm_places = unify_places(otm_enriched, "otm")
        all_places.extend(otm_places)

    # OSM fallback if no other sources
    if not all_places:
        sels = [
            'node["tourism"]',
            'node["historic"]',
            'node["leisure"="park"]',
            'node["natural"]',
        ]
        osm_raw = overpass_poi(geo["bbox"], sels, limit=120)
        osm_places = unify_places(osm_raw, "osm")
        all_places.extend(osm_places)

    # Enrich with Wikipedia
    enriched_places: list[UnifiedPlace] = []
    for place in all_places[:160]:
        if place.name:
            wiki = wiki_summary(place.name, "pl") or wiki_summary(place.name, "en")
            if wiki:
                # Create copy with wiki data
                enriched_place = UnifiedPlace(
                    name=place.name,
                    address=place.address,
                    lat=place.lat,
                    lon=place.lon,
                    rating=place.rating,
                    reviews=place.reviews,
                    price_level=place.price_level,
                    url=place.url or wiki.get("url", ""),
                    phone=place.phone,
                    website=place.website,
                    opening_hours=place.opening_hours,
                    photos=place.photos,
                    description=wiki.get("extract", place.description)
                    or place.description,
                    source=place.source,
                    types=place.types,
                )
                enriched_places.append(enriched_place)
            else:
                enriched_places.append(place)
        else:
            enriched_places.append(place)

    # Final merge and dedupe
    unique_places = _unique_by_name(enriched_places, max_results)
    return [place.to_dict() for place in unique_places]


# ── Transitland ───────────────────────────────────────────────────────────────
def tl_stops_near(lat: float, lon: float, radius_m: int = 900) -> list[dict[str, Any]]:
    params = {"lat": lat, "lon": lon, "r": max(150, radius_m)}
    if TL_KEY:
        params["apikey"] = TL_KEY
    r = _get(
        "https://transit.land/api/v2/rest/stops", params=params, source="transitland"
    )
    if r.status_code >= 400:
        return []
    return [
        {
            "name": s.get("name", ""),
            "onestop_id": s.get("onestop_id", ""),
            "lat": s.get("geometry", {}).get("coordinates", [None, None])[1],
            "lon": s.get("geometry", {}).get("coordinates", [None, None])[0],
        }
        for s in (r.json().get("data") or [])
    ]


# ── Flights (Travelpayouts) ───────────────────────────────────────────────────
def tp_iata(city: str) -> str | None:
    r = _get(
        "https://autocomplete.travelpayouts.com/places2",
        params={"term": city, "locale": "en", "types[]": "city"},
        source="flights",
    )
    if r.status_code >= 400:
        return None
    for it in r.json() or []:
        if it.get("type") == "city":
            code = it.get("code") or it.get("city_code")
            if code:
                return code
    return None


def tp_cheapest_month(origin: str, dest: str, month: str) -> dict[str, Any]:
    if not TP_TOKEN:
        return {}
    r = _get(
        "https://api.travelpayouts.com/v1/prices/cheap",
        params={"origin": origin, "destination": dest, "depart_date": month},
        headers={"x-access-token": TP_TOKEN},
        source="flights",
    )
    if r.status_code >= 400:
        return {}
    return r.json()


# ── Weather ───────────────────────────────────────────────────────────────────
def xweather_daily(lat: float, lon: float, days: int = 7) -> list[dict[str, Any]]:
    if not XW_SEC:
        return []
    r = _get(
        "https://api.weather.com/v3/wx/forecast/daily/5day",
        params={
            "geocode": f"{lat},{lon}",
            "format": "json",
            "language": "pl-PL",
            "units": "m",
            "apiKey": XW_SEC,
        },
        source="weather",
    )
    if r.status_code >= 400:
        return []
    j = r.json()
    out = []
    for i, day in enumerate(j.get("dayOfWeek", [])[: min(days, 5)]):
        out.append(
            {
                "day": day,
                "tmax": j.get("temperatureMax", [None])[i],
                "tmin": j.get("temperatureMin", [None])[i],
                "narrative": (j.get("narrative") or [""])[i],
            }
        )
    return out


def open_meteo(lat: float, lon: float, days: int = 7) -> list[dict[str, Any]]:
    r = _get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,"
            "precipitation_sum,weathercode",
            "forecast_days": max(1, min(14, days)),
            "timezone": "auto",
        },
        source="weather",
    )
    if r.status_code >= 400:
        return []
    d = r.json().get("daily") or {}
    out = []
    for i, t in enumerate(d.get("time", [])[:days]):
        out.append(
            {
                "day": t,
                "tmax": d.get("temperature_2m_max", [None])[i],
                "tmin": d.get("temperature_2m_min", [None])[i],
                "precip": d.get("precipitation_sum", [0])[i],
            }
        )
    return out


# ── Costs & Personalization ──────────────────────────────────────────────────
def estimate_cost(
    food_items: list[dict[str, Any]],
    wx: list[dict[str, Any]],
    place: str = "",
    mood: str = "spokój",
) -> tuple[int, int]:
    """Enhanced cost estimation with location and weather factors."""
    base_low, base_high = 60, 140

    # Restaurant pricing analysis
    if food_items:
        pricey = sum(1 for r in food_items[:12] if (r.get("price_level") or 0) >= 3)
        base_low = 70 + 10 * pricey
        base_high = 140 + 25 * pricey

    # Weather adjustments
    rain_days = sum(1 for d in wx if (d.get("precip", 0) or 0) > 5)
    if rain_days > len(wx) // 2:  # Mostly rainy
        base_high += 30  # Indoor activities more expensive

    # Location adjustments based on country/region
    place_lower = place.lower()
    if any(
        region in place_lower
        for region in ["norway", "switzerland", "denmark", "sweden"]
    ):
        base_low, base_high = int(base_low * 1.8), int(
            base_high * 2.2
        )  # Nordic premium
    elif any(
        region in place_lower for region in ["poland", "czech", "hungary", "slovakia"]
    ):
        base_low, base_high = int(base_low * 0.6), int(
            base_high * 0.8
        )  # Eastern Europe
    elif any(region in place_lower for region in ["spain", "portugal", "greece"]):
        base_low, base_high = int(base_low * 0.8), int(
            base_high * 1.1
        )  # Southern Europe
    elif any(
        region in place_lower
        for region in ["germany", "france", "netherlands", "belgium"]
    ):
        base_low, base_high = int(base_low * 1.2), int(
            base_high * 1.5
        )  # Western Europe

    # Mood adjustments
    if mood in ["energetyczny", "aktywny"]:
        base_high += 40  # More activities = higher cost
    elif mood in ["relaks", "zmęczony"]:
        base_low -= 20  # Fewer activities = lower cost

    return max(30, base_low), max(80, base_high)


def get_schedule_template(
    mood: str = "spokój", trip_type: str = "standard"
) -> dict[str, Any]:
    """Generate dynamic schedule based on mood and trip preferences."""
    templates = {
        "energetyczny": {
            "schedule": [
                "08:00-09:30: Śniadanie i poranny spacer",
                "10:00-12:30: Główne atrakcje i aktywności",
                "13:00-14:00: Szybki lunch",
                "14:30-17:30: Więcej zwiedzania/przygód",
                "18:00-19:00: Relaks w kawiarni",
                "19:30-21:30: Kolacja i życie nocne",
            ],
            "attractions_per_day": 8,
            "pace": "szybkie",
        },
        "relaks": {
            "schedule": [
                "09:00-10:30: Spokojne śniadanie",
                "11:00-12:30: Jedna główna atrakcja",
                "13:00-15:00: Długi lunch z widokami",
                "15:30-17:00: Spacer po parkach/ogrodach",
                "17:30-19:00: Odpoczynek w hotelu/kawiarni",
                "19:30-21:00: Kolacja bez pośpiechu",
            ],
            "attractions_per_day": 4,
            "pace": "spokojne",
        },
        "kultura": {
            "schedule": [
                "09:30-10:00: Śniadanie",
                "10:30-12:30: Muzea i galerie",
                "13:00-14:00: Lunch w lokalnej restauracji",
                "14:30-16:30: Zabytki i architektura",
                "17:00-18:30: Spacer po centrum historycznym",
                "19:00-21:00: Kolacja w tradycyjnym miejscu",
            ],
            "attractions_per_day": 6,
            "pace": "umiarkowane",
        },
        "nocne_zycie": {
            "schedule": [
                "10:00-11:00: Późne śniadanie",
                "11:30-13:30: Jedna główna atrakcja",
                "14:00-15:30: Lunch i siesta",
                "16:00-18:00: Zakupy/spacer",
                "18:30-20:00: Aperitif",
                "20:30-22:30: Kolacja",
                "23:00-02:00: Kluby i bary",
            ],
            "attractions_per_day": 5,
            "pace": "nocne",
        },
        "rodzina": {
            "schedule": [
                "08:30-09:30: Śniadanie z dziećmi",
                "10:00-12:00: Atrakcje przyjazne rodzinom",
                "12:30-13:30: Lunch (miejsce z dziecięcym menu)",
                "14:00-15:00: Odpoczynek/drzemka",
                "15:30-17:00: Parki i place zabaw",
                "17:30-19:00: Spacer i lody",
                "19:30-20:30: Wczesna kolacja",
            ],
            "attractions_per_day": 4,
            "pace": "elastyczne",
        },
    }

    # Determine trip type based on mood
    if mood in ["energetyczny", "aktywny"]:
        trip_type = "energetyczny"
    elif mood in ["relaks", "zmęczony", "spokój"]:
        trip_type = "relaks"
    elif mood in ["ciekawski", "uczący"]:
        trip_type = "kultura"

    return templates.get(trip_type, templates["relaks"])


# ── Static Map (Google + Leaflet fallback) ───────────────────────────────────
def static_map_url(
    center: tuple[float, float],
    pins: list[tuple[float, float, str]],
    zoom: int = 12,
    size: str = "800x600",
) -> str:
    """Generate map URL with Google Static Maps or Leaflet fallback."""
    if GOOGLE:
        # Google Static Maps (premium)
        parts = [
            f"center={center[0]},{center[1]}",
            f"zoom={zoom}",
            f"size={size}",
            "scale=2",
            "maptype=roadmap",
        ]

        # Enhanced pins with colors
        pin_colors = {"A": "red", "F": "green", "H": "blue", "T": "yellow"}
        for lat, lon, label in pins[:80]:
            lbl = re.sub(r"[^A-Za-z0-9]", "", (label or "X").upper())[:1] or "X"
            color = pin_colors.get(lbl, "red")
            parts.append(f"markers=color:{color}|label:{lbl}|{lat},{lon}")

        parts.append(f"key={GOOGLE}")
        return "https://maps.googleapis.com/maps/api/staticmap?" + "&".join(parts)
    else:
        # Leaflet fallback (free)
        return generate_leaflet_map_html(center, pins, zoom)


def generate_leaflet_map_html(
    center: tuple[float, float], pins: list[tuple[float, float, str]], zoom: int = 12
) -> str:
    """Generate interactive Leaflet map HTML as fallback."""
    pin_colors = {"A": "#ff0000", "F": "#00ff00", "H": "#0000ff", "T": "#ffff00"}

    markers_js = []
    for lat, lon, label in pins[:100]:
        color = pin_colors.get(label, "#ff0000")
        markers_js.append(
            f"L.circleMarker([{lat}, {lon}], {{color: '{color}', radius: 8}})"
            f".bindPopup('{label}').addTo(map);"
        )

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mapa podróży</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <style>
            #map {{ height: 600px; width: 100%; }}
            .legend {{ 
                background: white; padding: 10px; border-radius: 5px; 
                box-shadow: 0 0 15px rgba(0,0,0,0.2);
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script>
            var map = L.map('map').setView([{center[0]}, {center[1]}], {zoom});
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '© OpenStreetMap contributors'
            }}).addTo(map);
            
            // Markers
            {chr(10).join(markers_js)}
            
            // Legend
            var legend = L.control({{position: 'topright'}});
            legend.onAdd = function (map) {{
                var div = L.DomUtil.create('div', 'legend');
                div.innerHTML = '<h4>Legenda</h4>' +
                    '<span style="color: red;">●</span> Atrakcje (A)<br>' +
                    '<span style="color: green;">●</span> Restauracje (F)<br>' +
                    '<span style="color: blue;">●</span> Hotele (H)<br>' +
                    '<span style="color: orange;">●</span> Transport (T)';
                return div;
            }};
            legend.addTo(map);
        </script>
    </body>
    </html>
    """
    return html


# ── Planning ──────────────────────────────────────────────────────────────────
@dataclass
class TripSpec:
    place: str
    days: int
    food: bool = True
    hotels: bool = True
    lang: str = "pl"


def _item_md(it: dict[str, Any], day_idx: int) -> str:
    name = it.get("name") or "(bez nazwy)"
    rank = it.get("ranking", "")
    url = (
        it.get("url")
        or (
            (it.get("wiki") or {}).get("url")
            if isinstance(it.get("wiki"), dict)
            else ""
        )
        or it.get("map_url", "")
        or it.get("website", "")
    )
    desc = it.get("desc") or (
        (it.get("wiki") or {}).get("extract")
        if isinstance(it.get("wiki"), dict)
        else ""
    )
    line = f"- **{name}**" + (f" • {rank}" if rank else "")
    if it.get("opening"):
        oh = format_open_hours(it.get("opening"), day_idx)
        if oh:
            line += f" — {oh}"
    if desc:
        line += f"\n  {desc[:260]}{'...' if len(desc)>260 else ''}"
    if url:
        line += f"\n  {url}"
    return line


def plan_trip(
    spec: TripSpec, user_id: str = "default", narrative_style: str = "auto"
) -> dict[str, Any]:
    # Załaduj profil podróżnika
    profile = load_traveler_profile(user_id)

    geo = geocode(spec.place)
    if not geo:
        return {"error": "place_not_found"}
    lat, lon = geo["lat"], geo["lon"]
    tz = geo["tz"]
    mood = psyche_mood()
    schedule_template = get_schedule_template(mood)

    wx = xweather_daily(lat, lon, days=max(3, spec.days)) or open_meteo(
        lat, lon, days=max(3, spec.days)
    )
    poi = attractions(spec.place, max_results=100, lang=spec.lang)
    poi = [dict(x, opening=None) for x in poi]  # unify

    # Personalizacja atrakcji na podstawie profilu i pogody
    if poi and profile:
        weather_context = wx[0] if wx else {"condition": "clear", "temperature": 20}
        current_hour = int(time.strftime("%H"))
        time_of_day = (
            "morning"
            if current_hour < 12
            else ("afternoon" if current_hour < 18 else "evening")
        )

        poi = get_contextual_recommendations(profile, weather_context, time_of_day, poi)

    foods = (
        restaurants(spec.place, "", max_results=24, open_now=False, lang=spec.lang)
        if spec.food
        else []
    )
    hotels_l = (
        hotels(spec.place, "", max_results=24, lang=spec.lang) if spec.hotels else []
    )

    stops = tl_stops_near(lat, lon, radius_m=900)

    blocks = greedy_cluster(
        [x for x in poi if x.get("lat") and x.get("lon")],
        spec.days,
        per=schedule_template["attractions_per_day"] // spec.days,
        center=(lat, lon),
    )
    low, high = estimate_cost(foods, wx, spec.place, mood)

    # map pins
    pins = []
    for b in blocks:
        for it in b[:2]:
            pins.append((it.get("lat"), it.get("lon"), "A"))
    for r in foods[:4]:
        if r.get("lat"):
            pins.append((r["lat"], r["lon"], "F"))
    for h in hotels_l[:2]:
        if h.get("lat"):
            pins.append((h["lat"], h["lon"], "H"))
    map_url = static_map_url((lat, lon), pins) if pins else ""

    title = f"Plan podróży: {spec.place} — {spec.days} dni"
    md = [
        f"# {title}",
        f"_Strefa:_ **{tz}** • _Nastrój:_ **{mood}**  \n"
        f"_Lat/Lon:_ {lat:.4f},{lon:.4f}",
        "",
    ]

    # Dodaj spersonalizowaną narrację na początku
    if profile and poi:
        personalized_intro = generate_personalized_narrative(
            profile, spec.place, poi[:8], narrative_style
        )
        md.append("## Twój spersonalizowany plan")
        md.append(personalized_intro)
        md.append("")

    if wx:
        md.append("## Pogoda")
        for d in wx[: spec.days]:
            if "narrative" in d:
                md.append(f"- {d['day']}: {d['tmin']}-{d['tmax']}°C — {d['narrative']}")
            else:
                md.append(
                    f"- {d['day']}: {d['tmin']}-{d['tmax']}°C • opady "
                    f"{d.get('precip',0)} mm"
                )
        md.append("")

    md.append("## Szacunkowy koszt dzienny")
    md.append(f"- widełki: **{low}-{high} PLN**/os.")
    md.append("")

    if map_url:
        md.append("## Mapa (Google Static)")
        md.append(map_url)
        md.append("")

    for i, day in enumerate(blocks):
        md.append(f"## Dzień {i+1}")
        if not day:
            md.append("- rezerwa / odpoczynek\n")
            continue
        md.append(f"**Harmonogram ({schedule_template['pace']}):**")
        for schedule_item in schedule_template["schedule"]:
            md.append(f"- {schedule_item}")
        md.append("")
        md.append("**Punkty dnia:**")
        for it in day[:6]:
            md.append(_item_md(it, i))
        md.append("")

    def _fmt_item(x: dict[str, Any]) -> str:
        price = (
            ("$" * int(x["price_level"])) if (x.get("price_level") is not None) else ""
        )
        line = (
            f"- **{x['name']}** {price} • ocena {x.get('rating','?')} "
            f"({x.get('reviews',0)} opinii)"
        )
        if x.get("address"):
            line += f"\n  {x['address']}"
        if x.get("website"):
            line += f"\n  {x['website']}"
        if x.get("map_url"):
            line += f"\n  {x['map_url']}"
        return line

    if foods:
        md.append("## Restauracje (Google + TripAdvisor)")
        for r in foods[:24]:
            md.append(_fmt_item(r))
        md.append("")
    if hotels_l:
        md.append("## Hotele (Google + TripAdvisor)")
        for h in hotels_l[:24]:
            md.append(_fmt_item(h))
        md.append("")

    if stops:
        md.append("## Komunikacja — najbliższe przystanki (Transitland)")
        for s in stops[:12]:
            md.append(
                f"- {s['name']} — {s['onestop_id']} " f"({s['lat']:.4f},{s['lon']:.4f})"
            )
        md.append("")

    body = "\n".join(md).strip()
    base_path = OUT_DIR / f"trip_{_slug(spec.place)}_{_now_ms()}"

    # Enhanced export with multiple formats
    exported = export_formats(
        title,
        body,
        {
            "place": spec.place,
            "days": spec.days,
            "mood": mood,
            "schedule_type": schedule_template["pace"],
            "cost_estimate": {"low": low, "high": high},
            "restaurants_count": len(foods),
            "hotels_count": len(hotels_l),
            "attractions_count": len(poi),
        },
        str(base_path),
        formats=["md", "html", "pdf", "json", "zip"],
    )
    mem_add(
        f"[TRAVEL] {spec.place} {spec.days}d",
        {"text": body},
        tags=["travel", "plan"],
    )
    auto_learn(
        {
            "kind": "travel.plan",
            "place": spec.place,
            "days": spec.days,
            "foods": len(foods),
            "hotels": len(hotels_l),
        }
    )
    return {
        "exported_files": exported,
        "restaurants": len(foods),
        "hotels": len(hotels_l),
        "attractions": len(poi),
        "cost_estimate": {"low": low, "high": high},
        "mood": mood,
        "schedule_type": schedule_template["pace"],
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
def _print(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="TravelGuide ULTRA — Google+TripAdvisor+OTM/OSM"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan")
    p.add_argument("--place", required=True)
    p.add_argument("--days", type=int, required=True)
    p.add_argument("--food", action="store_true")
    p.add_argument("--hotels", action="store_true")
    p.add_argument("--lang", default="pl")
    p.add_argument("--user", default="default", help="User ID dla profilu podróżnika")
    p.add_argument(
        "--style",
        default="auto",
        choices=[
            "auto",
            "friendly",
            "elegant",
            "energetic",
            "scholarly",
            "casual",
            "professional",
        ],
        help="Styl narracji",
    )

    g = sub.add_parser("food")
    g.add_argument("--place", required=True)
    g.add_argument("--q", default="")
    g.add_argument("--open-now", action="store_true")
    g.add_argument("--max", type=int, default=24)
    g.add_argument("--lang", default="pl")
    h = sub.add_parser("hotels")
    h.add_argument("--place", required=True)
    h.add_argument("--q", default="")
    h.add_argument("--max", type=int, default=24)
    h.add_argument("--lang", default="pl")
    a = sub.add_parser("attractions")
    a.add_argument("--place", required=True)
    a.add_argument("--max", type=int, default=80)
    a.add_argument("--lang", default="pl")

    f = sub.add_parser("flights")
    f.add_argument("--from", dest="frm", required=True)
    f.add_argument("--to", required=True)
    f.add_argument("--month", required=True)
    t = sub.add_parser("transit")
    t.add_argument("--place", required=True)
    t.add_argument("--radius", type=int, default=900)
    m = sub.add_parser("map")
    m.add_argument("--place", required=True)
    m.add_argument("--max", type=int, default=30)

    args = ap.parse_args(argv)

    if args.cmd == "plan":
        _print(
            plan_trip(
                TripSpec(
                    place=args.place,
                    days=args.days,
                    food=args.food,
                    hotels=args.hotels,
                    lang=args.lang,
                ),
                user_id=args.user,
                narrative_style=args.style,
            )
        )
        return 0
    if args.cmd == "food":
        _print(
            {
                "place": args.place,
                "items": restaurants(
                    args.place, args.q, args.max, args.open_now, args.lang
                ),
            }
        )
        return 0
    if args.cmd == "hotels":
        _print(
            {
                "place": args.place,
                "items": hotels(args.place, args.q, args.max, args.lang),
            }
        )
        return 0
    if args.cmd == "attractions":
        _print(
            {
                "place": args.place,
                "items": attractions(args.place, args.max, args.lang),
            }
        )
        return 0
    if args.cmd == "flights":
        o = (
            args.frm.upper()
            if len(args.frm) == 3
            else (tp_iata(args.frm) or args.frm.upper())
        )
        d = (
            args.to.upper()
            if len(args.to) == 3
            else (tp_iata(args.to) or args.to.upper())
        )
        _print(
            {
                "origin": o,
                "dest": d,
                "month": args.month,
                "raw": tp_cheapest_month(o, d, args.month),
            }
        )
        return 0
    if args.cmd == "transit":
        geo = geocode(args.place)
        if not geo:
            _print({"error": "place_not_found"})
            return 0
        _print(
            {
                "place": args.place,
                "stops": tl_stops_near(geo["lat"], geo["lon"], args.radius),
            }
        )
        return 0
    if args.cmd == "map":
        geo = geocode(args.place)
        if not geo:
            _print({"error": "place_not_found"})
            return 0
        ats = attractions(args.place, max_results=min(50, args.max))
        foods = restaurants(args.place, max_results=min(30, args.max))
        hotels_l = hotels(args.place, max_results=min(20, args.max))
        pins = []
        for it in ats[:10]:
            if it.get("lat"):
                pins.append((it["lat"], it["lon"], "A"))
        for r in foods[:6]:
            if r.get("lat"):
                pins.append((r["lat"], r["lon"], "F"))
        for h in hotels_l[:3]:
            if h.get("lat"):
                pins.append((h["lat"], h["lon"], "H"))
        url = (
            static_map_url((geo["lat"], geo["lon"]), pins)
            or "(brak mapy — brak GOOGLE_MAPS_KEY)"
        )
        _print({"center": [geo["lat"], geo["lon"]], "url": url})
        return 0
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main(sys.argv[1:]))
