#!/usr/bin/env python3
"""
Demo pełnego systemu personalizacji - tworzenie profilu, nauka, generowanie planów.
"""

from travelguide import (
    generate_personalized_narrative,
    get_contextual_recommendations,
    load_traveler_profile,
    save_traveler_profile,
    update_profile_from_trip,
)


def create_expert_profile():
    """Stwórz profil doświadczonego podróżnika."""
    profile = load_traveler_profile("anna_kowalska")
    profile.name = "Anna Kowalska"
    profile.likes_museums = 0.95
    profile.likes_nature = 0.8
    profile.likes_nightlife = 0.3
    profile.likes_historical_sites = 0.9
    profile.likes_modern_attractions = 0.6
    profile.crowded_places_tolerance = 0.2  # Nie lubi tłumów
    profile.budget_level = "high"
    profile.travel_pace = "slow"  # Lubi spokojne zwiedzanie
    profile.luxury_preference = 0.8
    profile.dietary_restrictions = ["vegetarian"]
    profile.language_preferences = ["pl", "en", "fr"]

    # Symuluj historię podróży
    profile.visited_cities = ["Paryż", "Rzym", "Barcelona", "Praga", "Wiedeń"]
    profile.trips_count = 8

    save_traveler_profile(profile)
    return profile


def create_sample_attractions_detailed():
    """Utwórz szczegółowe przykładowe atrakcje."""
    return [
        {
            "id": "museum_cracow_1",
            "name": "Muzeum Książąt Czartoryskich",
            "types": ["museum", "art_gallery", "historical"],
            "rating": 4.6,
            "reviews": 1200,
            "description": "Najstarsze muzeum w Polsce z 'Damą z gronostajem' Leonarda da Vinci.",
            "lat": 50.0619,
            "lon": 19.9369,
            "opening_hours": "Wt-Nd 10:00-18:00",
            "price_level": 2,
        },
        {
            "id": "historical_cracow_1",
            "name": "Podziemia Rynku w Krakowie",
            "types": ["museum", "historical", "underground"],
            "rating": 4.4,
            "reviews": 3500,
            "description": "Średniowieczny Kraków pod ziemią - archeologiczne odkrycia i multimedia.",
            "lat": 50.0617,
            "lon": 19.9370,
            "opening_hours": "Pn-Nd 10:00-20:00",
            "price_level": 2,
        },
        {
            "id": "nature_cracow_1",
            "name": "Ogród Botaniczny UJ",
            "types": ["garden", "natural_feature", "educational"],
            "rating": 4.5,
            "reviews": 800,
            "description": "Jeden z najstarszych ogrodów botanicznych w Polsce z 5000 gatunków roślin.",
            "lat": 50.0533,
            "lon": 19.9567,
            "opening_hours": "Kwi-Paź 9:00-19:00",
            "price_level": 1,
        },
        {
            "id": "shopping_cracow_1",
            "name": "Sukiennice",
            "types": ["historical", "shopping", "market"],
            "rating": 4.3,
            "reviews": 2800,
            "description": "Historyczne centrum handlowe z XIV wieku - pamiątki i rękodzieło.",
            "lat": 50.0617,
            "lon": 19.9370,
            "opening_hours": "Pn-Nd 9:00-18:00",
            "price_level": 2,
        },
        {
            "id": "nightlife_cracow_1",
            "name": "Alchemia",
            "types": ["bar", "pub", "cultural"],
            "rating": 4.4,
            "reviews": 900,
            "description": "Kultowy bar w żydowskiej dzielnicy Kazimierz z klimatem lat 20.",
            "lat": 50.0512,
            "lon": 19.9455,
            "opening_hours": "Pn-Nd 16:00-2:00",
            "price_level": 2,
        },
    ]


def demo_full_personalization():
    """Demonstracja pełnej personalizacji."""
    print("🎨 DEMO SYSTEMU PERSONALIZACJI TRAVELGUIDE")
    print("=" * 55)

    # 1. Stwórz profil eksperta
    profile = create_expert_profile()
    print(f"👩‍🎓 Profil: {profile.name}")
    print(f"   • Lubi muzea: {profile.likes_museums:.1%}")
    print(f"   • Toleruje tłumy: {profile.crowded_places_tolerance:.1%}")
    print(f"   • Budżet: {profile.budget_level}")
    print(f"   • Tempo: {profile.travel_pace}")

    # 2. Przygotuj atrakcje
    attractions = create_sample_attractions_detailed()
    print(f"\n🏛️ Dostępne atrakcje: {len(attractions)}")

    # 3. Test rekomendacji w różnych warunkach
    print("\n🌦️ REKOMENDACJE KONTEKSTOWE:")

    # Deszczowy dzień
    rainy_weather = {"condition": "rain", "temperature": 12}
    rainy_recs = get_contextual_recommendations(profile, rainy_weather, "afternoon", attractions)

    print("   🌧️ Deszczowy dzień (preferencje: muzea + indoor):")
    for i, rec in enumerate(rainy_recs[:3], 1):
        score = rec.get("personalized_score", 0)
        print(f"      {i}. {rec['name']} (score: {score:.2f})")

    # 4. Generowanie narracji w różnych stylach
    print("\n📝 PERSONALIZOWANE NARRACJE:")

    styles = [
        ("elegant", "🎩 Elegancki styl dla wymagającego podróżnika"),
        ("scholarly", "📚 Naukowy styl z kontekstem historycznym"),
        ("friendly", "😊 Przyjazny styl personalny"),
    ]

    for style, desc in styles:
        print(f"\n   {desc}:")
        narrative = generate_personalized_narrative(profile, "Kraków", rainy_recs[:3], style)
        # Pokaż pierwszy fragment
        lines = narrative.split("\n")
        for line in lines[:3]:
            if line.strip():
                print(f'      "{line.strip()}"')
                break

    # 5. Symuluj uczenie się
    print("\n🧠 SYSTEM UCZENIA SIĘ:")
    print(f"   Przed podróżą: muzea={profile.likes_museums:.2f}, natura={profile.likes_nature:.2f}")

    # Symuluj bardzo pozytywną ocenę natury
    trip_data = {
        "city": "Kraków",
        "attractions_visited": attractions,
        "ratings": {
            "nature_cracow_1": 5,  # Ogród botaniczny zachwycił!
            "museum_cracow_1": 4,  # Muzeum dobre, ale nic nowego
            "historical_cracow_1": 5,  # Podziemia fascynujące
            "shopping_cracow_1": 3,  # Sukiennice przeciętne
            "nightlife_cracow_1": 2,  # Bar za głośny
        },
    }

    update_profile_from_trip(profile, trip_data)

    print(f"   Po podróży: muzea={profile.likes_museums:.2f}, natura={profile.likes_nature:.2f}")
    print("   🎯 System nauczył się, że Anna bardziej docenia naturę!")

    # 6. Final test z nowym profilem
    print("\n🔄 NOWE REKOMENDACJE PO NAUCE:")
    final_recs = get_contextual_recommendations(
        profile, {"condition": "clear", "temperature": 22}, "morning", attractions
    )

    print("   ☀️ Słoneczny poranek (po dostosowaniu preferencji):")
    for i, rec in enumerate(final_recs[:3], 1):
        score = rec.get("personalized_score", 0)
        print(f"      {i}. {rec['name']} (score: {score:.2f})")

    print("\n✨ PODSUMOWANIE:")
    print(f"   📊 Profil został zaktualizowany na podstawie {profile.trips_count} podróży")
    print(f"   🗺️ Odwiedzone miasta: {', '.join(profile.visited_cities[-3:])}")
    print("   🎯 System adaptuje się do rzeczywistych preferencji użytkownika")

    # Pokaż końcowy profil
    print(f"\n💾 Końcowy profil zapisany w: out/travel/profiles/{profile.user_id}.json")


if __name__ == "__main__":
    demo_full_personalization()
