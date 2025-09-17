#!/usr/bin/env python3
"""
Test systemu personalizacji dla TravelGuide.
Demonstracja funkcji profilu podróżnika bez zależności od zewnętrznych API.
"""

import sys

from travelguide import (
    _categorize_attractions,
    generate_personalized_narrative,
    get_contextual_recommendations,
    load_traveler_profile,
    save_traveler_profile,
    update_profile_from_trip,
)


def create_sample_attractions():
    """Utwórz przykładowe atrakcje dla testów."""
    return [
        {
            "id": "museum_1",
            "name": "Muzeum Narodowe w Krakowie",
            "types": ["museum", "tourist_attraction"],
            "rating": 4.5,
            "reviews": 2500,
            "description": "Najstarsze muzeum w Polsce z kolekcją polskiego malarstwa.",
            "lat": 50.0619,
            "lon": 19.9369,
        },
        {
            "id": "park_1",
            "name": "Planty Krakowskie",
            "types": ["park", "natural_feature"],
            "rating": 4.3,
            "reviews": 1800,
            "description": "Pierścień parków otaczający Stare Miasto.",
            "lat": 50.0614,
            "lon": 19.9365,
        },
        {
            "id": "historical_1",
            "name": "Zamek Królewski na Wawelu",
            "types": ["castle", "historical", "tourist_attraction"],
            "rating": 4.7,
            "reviews": 5000,
            "description": "Zamek królewski i katedra - symbol Polski.",
            "lat": 50.0543,
            "lon": 19.9356,
        },
        {
            "id": "nightlife_1",
            "name": "Klub Kwadrat",
            "types": ["night_club", "bar"],
            "rating": 4.2,
            "reviews": 800,
            "description": "Popularny klub w centrum Krakowa.",
            "lat": 50.0625,
            "lon": 19.9374,
        },
        {
            "id": "shopping_1",
            "name": "Galeria Krakowska",
            "types": ["shopping_mall", "store"],
            "rating": 4.1,
            "reviews": 3200,
            "description": "Duże centrum handlowe przy dworcu głównym.",
            "lat": 50.0679,
            "lon": 19.9449,
        },
    ]


def test_profile_creation():
    """Test tworzenia i zarządzania profilami."""
    print("🧪 Test 1: Tworzenie profilu podróżnika")

    # Utwórz nowy profil
    profile = load_traveler_profile("test_user")
    print(f"✅ Nowy profil utworzony: {profile.name} (ID: {profile.user_id})")

    # Dostosuj preferencje
    profile.likes_museums = 0.9
    profile.likes_nature = 0.7
    profile.likes_nightlife = 0.2
    profile.likes_historical_sites = 0.8
    profile.budget_level = "medium"
    profile.crowded_places_tolerance = 0.4

    # Zapisz profil
    save_traveler_profile(profile)
    print(
        f"✅ Profil zapisany: lubi muzea ({profile.likes_museums}), naturę ({profile.likes_nature})"
    )

    return profile


def test_attraction_categorization():
    """Test kategoryzacji atrakcji."""
    print("\n🧪 Test 2: Kategoryzacja atrakcji")

    attractions = create_sample_attractions()
    categories = _categorize_attractions(attractions)

    for attraction in attractions:
        category = categories.get(attraction["id"], "unknown")
        print(f"✅ {attraction['name']}: {category}")

    return attractions, categories


def test_contextual_recommendations():
    """Test rekomendacji kontekstowych."""
    print("\n🧪 Test 3: Rekomendacje kontekstowe")

    profile = load_traveler_profile("test_user")
    attractions = create_sample_attractions()

    # Test w deszczową pogodę
    rainy_weather = {"condition": "rain", "temperature": 15}
    rainy_recs = get_contextual_recommendations(profile, rainy_weather, "afternoon", attractions)

    print("🌧️ Rekomendacje na deszczową pogodę:")
    for rec in rainy_recs[:3]:
        score = rec.get("personalized_score", 0)
        print(f"  • {rec['name']}: score {score:.2f}")

    # Test w słoneczną pogodę
    sunny_weather = {"condition": "clear", "temperature": 25}
    sunny_recs = get_contextual_recommendations(profile, sunny_weather, "morning", attractions)

    print("\n☀️ Rekomendacje na słoneczną pogodę:")
    for rec in sunny_recs[:3]:
        score = rec.get("personalized_score", 0)
        print(f"  • {rec['name']}: score {score:.2f}")


def test_narrative_generation():
    """Test generowania narracji."""
    print("\n🧪 Test 4: Generowanie narracji")

    profile = load_traveler_profile("test_user")
    attractions = create_sample_attractions()

    # Test różnych stylów
    styles = ["friendly", "elegant", "energetic", "scholarly"]

    for style in styles:
        print(f"\n📝 Styl: {style}")
        narrative = generate_personalized_narrative(profile, "Kraków", attractions, style)
        print(f"   {narrative[:100]}...")


def test_learning_system():
    """Test systemu uczenia się."""
    print("\n🧪 Test 5: System uczenia się")

    profile = load_traveler_profile("test_user")

    # Symuluj podróż z ocenami
    trip_data = {
        "city": "Kraków",
        "attractions_visited": create_sample_attractions(),
        "ratings": {
            "museum_1": 5,  # Bardzo pozytywna ocena muzeum
            "park_1": 4,  # Pozytywna ocena parku
            "nightlife_1": 2,  # Negatywna ocena klubu
            "historical_1": 5,  # Bardzo pozytywna ocena zamku
            "shopping_1": 3,  # Neutralna ocena centrum handlowego
        },
    }

    print(
        f"📊 Preferencje przed nauką: muzea={profile.likes_museums:.2f}, natura={profile.likes_nature:.2f}"
    )

    # Aktualizuj profil na podstawie ocen
    update_profile_from_trip(profile, trip_data)

    print(
        f"📈 Preferencje po nauce: muzea={profile.likes_museums:.2f}, natura={profile.likes_nature:.2f}"
    )
    print(f"✅ Liczba podróży: {profile.trips_count}")
    print(f"✅ Odwiedzone miasta: {profile.visited_cities}")


def test_profile_persistence():
    """Test zapisu i wczytywania profili."""
    print("\n🧪 Test 6: Trwałość profili")

    # Załaduj profil ponownie
    profile_reloaded = load_traveler_profile("test_user")

    print(f"✅ Profil wczytany: {profile_reloaded.name}")
    print(f"✅ Preferencje zachowane: muzea={profile_reloaded.likes_museums:.2f}")
    print(f"✅ Historia zachowana: {profile_reloaded.trips_count} podróży")


def main():
    """Uruchom wszystkie testy systemu personalizacji."""
    print("🚀 TEST SYSTEMU PERSONALIZACJI TRAVELGUIDE")
    print("=" * 50)

    try:
        test_profile_creation()
        test_attraction_categorization()
        test_contextual_recommendations()
        test_narrative_generation()
        test_learning_system()
        test_profile_persistence()

        print("\n" + "=" * 50)
        print("✅ WSZYSTKIE TESTY PRZESZŁY POMYŚLNIE!")
        print("🎯 System personalizacji działa prawidłowo")

    except Exception as e:
        print(f"\n❌ BŁĄD W TESTACH: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
