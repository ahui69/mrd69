# ruff: noqa: E501
"""
seed_memory.py - Skrypt do zasilania pamięci LTM wiedzą początkową.

Ten skrypt pobiera informacje z zewnętrznych źródeł (DuckDuckGo, Wikipedia),
przetwarza je za pomocą LLM w celu ekstrakcji kluczowych faktów, a następnie
zapisuje te fakty w pamięci długoterminowej (LTM) wraz z linkiem do źródła.
"""

import json
import time

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from readability import Document

from llm_simple import chat as llm_chat
from memory import get_memory

# Konfiguracja
MAX_RESULTS_PER_TOPIC = 7  # Zwiększono liczbę analizowanych wyników
LLM_MAX_TOKENS_FOR_SUMMARY = 1500  # Tokeny dla LLM do podsumowania artykułu


def _fetch_article_text(url: str) -> str:
    """Pobiera i ekstrahuje główną treść artykułu z podanego URL."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        doc = Document(response.content)
        soup = BeautifulSoup(doc.summary(), "html.parser")
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        print(f"  [!] Błąd podczas pobierania {url}: {e}")
        return ""


def _extract_facts_from_text(text: str, topic: str) -> list[str]:
    """Używa LLM do ekstrakcji kluczowych faktów z tekstu."""
    if not text or not text.strip():
        return []

    system_prompt = f"""Jesteś analitykiem danych.
Twoim zadaniem jest przeanalizowanie poniższego tekstu na temat '{topic}'
i wyekstrahowanie z niego od 5 do 10 najważniejszych, kluczowych faktów.
Zwróć fakty jako listę w formacie JSON.
Każdy fakt powinien być zwięzłym, samodzielnym zdaniem.

Przykład formatu wyjściowego:
["Fakt 1.", "Fakt 2.", "Fakt 3."]
"""
    try:
        response_raw = llm_chat(
            user_text=text[:8000],  # Ograniczenie tekstu wejściowego dla LLM
            system_text=system_prompt,
            max_tokens=LLM_MAX_TOKENS_FOR_SUMMARY,
        )
        # Proste czyszczenie odpowiedzi, aby uzyskać poprawny JSON
        start_index = response_raw.find("[")
        end_index = response_raw.rfind("]") + 1
        json_str = response_raw[start_index:end_index]
        facts = json.loads(json_str)
        return facts if isinstance(facts, list) else []
    except Exception as e:
        print(f"  [!] Błąd podczas ekstrakcji faktów przez LLM: {e}")
        return []


def seed_topic(topic: str, category: str):
    """Wyszukuje info na dany temat, ekstrahuje fakty i zapisuje w pamięci."""
    print(f"\n--- Zasilanie pamięci wiedzą na temat: " f"'{topic}' (Kategoria: {category}) ---")
    mem = get_memory()
    processed_urls = set()
    total_facts_added = 0

    try:
        with DDGS() as ddgs:
            # Używamy 'keywords' aby lepiej sprecyzować zapytanie
            keywords = f"{topic} {category}"
            results = list(ddgs.text(keywords, max_results=MAX_RESULTS_PER_TOPIC))
    except Exception as e:
        print(f"  [!] Błąd podczas wyszukiwania w DuckDuckGo: {e}")
        return

    print(f"[+] Znaleziono {len(results)} wyników dla zapytania '{keywords}'.")

    for i, result in enumerate(results):
        url = result.get("href")
        if not url or url in processed_urls:
            continue

        print(f"\n[{i+1}/{len(results)}] Przetwarzanie: " f"{result.get('title', 'Brak tytułu')}")
        print(f"  -> Źródło: {url}")

        article_text = _fetch_article_text(url)
        if not article_text:
            continue

        facts = _extract_facts_from_text(article_text, topic)
        if not facts:
            print("  [!] Nie udało się wyekstrahować faktów.")
            continue

        print(f"  [+] Wyekstrahowano {len(facts)} faktów.")
        for fact in facts:
            if not isinstance(fact, str) or not fact.strip():
                continue

            # Dodajemy fakt do pamięci z wysokim 'conf' i tagami źródła
            fact_with_source = f"{fact.strip()} (Źródło: {url})"
            mem.add_fact(
                text=fact_with_source,
                conf=0.90,  # Wysoka pewność dla wiedzy początkowej
                tags=[
                    "seed",
                    category.lower().replace(" ", "_"),
                    topic.lower().replace(" ", "_"),
                ],
            )
            total_facts_added += 1

        processed_urls.add(url)
        time.sleep(1)  # Unikamy zbyt szybkich zapytań

    print(
        f"\n--- Zakończono zasilanie dla tematu '{topic}'. "
        f"Dodano {total_facts_added} nowych faktów. ---"
    )


if __name__ == "__main__":
    # === GŁÓWNA KONFIGURACJA KATEGORII I TEMATÓW ===
    # Rozbudowana lista tematów do zasilenia pamięci, zgodnie z prośbą.
    # Dla każdej kategorii zostanie przetworzonych po kilka-kilkanaście
    # tematów.

    knowledge_base = {
        "Krypto i tokeny": [
            "Podstawy technologii blockchain",
            "Czym jest Bitcoin i Ethereum",
            "Zdecentralizowane finanse (DeFi) dla początkujących",
            "Niewymienialne tokeny (NFT) i ich zastosowania",
            "Jak działa kopanie kryptowalut (Proof of Work vs Proof of Stake)",
            "Bezpieczeństwo portfeli kryptowalutowych",
            "Analiza techniczna na rynku krypto",
            "Czym są smart kontrakty",
            "Popularne altcoiny i ich projekty",
            "Ryzyka inwestycyjne w kryptowaluty",
            "Czym są Oracles w blockchain (np. Chainlink)",
            "Skalowalność blockchain: Layer 2 (Polygon, Arbitrum)",
            "Interoperacyjność blockchain (Polkadot, Cosmos)",
            "Zdecentralizowane organizacje autonomiczne (DAO)",
            "Yield farming i liquidity mining w DeFi",
            "Stablecoiny: rodzaje i mechanizmy (USDT, USDC, DAI)",
            "Tokenomia: projektowanie ekonomii tokenów",
            "Web3: zdecentralizowany internet",
            "Ataki na smart kontrakty i ich audyty",
            "Podatki od kryptowalut w Polsce",
            "Historia kryptowalut przed Bitcoinem",
            "Czym jest halving Bitcoina",
            "Rola giełd CEX vs DEX",
            "Metaverse i rola kryptowalut",
            "Soulbound Tokens (SBT)",
            "Real World Assets (RWA) na blockchainie",
            "Prywatność w kryptowalutach (Monero, Zcash)",
            "Staking i jego rodzaje",
            "Launchpady i IDO (Initial DEX Offering)",
            "Analiza on-chain",
        ],
        "Programowanie i kod": [
            "Wprowadzenie do języka Python",
            "Podstawy JavaScript i jego rola w web development",
            "Zasady czystego kodu (Clean Code)",
            "Paradygmaty programowania: obiektowe, funkcyjne, proceduralne",
            "Czym jest sztuczna inteligencja i uczenie maszynowe",
            "Systemy kontroli wersji: Git i GitHub",
            "Podstawy tworzenia API (REST vs GraphQL)",
            "Struktury danych i algorytmy",
            "Testowanie oprogramowania: testy jednostkowe i integracyjne",
            "Wprowadzenie do konteneryzacji (Docker)",
            "Programowanie asynchroniczne w Pythonie (asyncio)",
            "Tworzenie aplikacji webowych (Flask, Django)",
            "Frontendowe frameworki (React, Vue, Angular)",
            "TypeScript jako nadzbiór JavaScript",
            "Architektura mikroserwisów",
            "Bazy danych SQL vs NoSQL (PostgreSQL, MongoDB)",
            "Czym jest DevOps i CI/CD",
            "Pisanie wydajnego kodu w Pythonie",
            "Wzorce projektowe (Design Patterns)",
            "Bezpieczeństwo aplikacji webowych (OWASP Top 10)",
            "Programowanie niskopoziomowe (C, C++)",
            "Język programowania Rust i jego zalety",
            "WebAssembly (WASM)",
            "GraphQL w praktyce",
            "Zarządzanie zależnościami (pip, npm, yarn)",
            "Infrastruktura jako kod (Terraform, Ansible)",
            "Podstawy chmury obliczeniowej (AWS, Azure, GCP)",
            "Big Data i narzędzia (Spark, Hadoop)",
            "Tworzenie gier komputerowych (Unity, Unreal Engine)",
            "Etyczny hacking i cyberbezpieczeństwo",
        ],
        "Psychologia": [
            "Jak czytać mowę ciała i mikromimikę",
            "Psychologia kreatywności i proces twórczy",
            "Czym jest inteligencja emocjonalna (EQ)",
            "Mechanizmy obronne ego w teorii Freuda",
            "Błędy poznawcze i jak ich unikać",
            "Psychologia perswazji i wywierania wpływu",
            "Rola sarkazmu i ironii w komunikacji",
            "Podstawy terapii poznawczo-behawioralnej (CBT)",
            "Efekt Dunninga-Krugera",
            "Motywacja wewnętrzna i zewnętrzna",
            "Dysonans poznawczy i jego skutki",
            "Teoria przywiązania Johna Bowlby'ego",
            "Psychologia tłumu i zachowania grupowe",
            "Efekt placebo i nocebo",
            "Zaburzenia osobowości (np. borderline, narcystyczne)",
            "Stres i techniki radzenia sobie z nim",
            "Psychologia snu i marzeń sennych",
            "Uzależnienia behawioralne (np. od internetu, hazardu)",
            "Syndrom sztokholmski",
            "Psychologia pozytywna i dobrostan",
            "Wpływ muzyki na mózg i emocje",
            "Pamięć i jej rodzaje (krótkotrwała, długotrwała)",
            "Heurystyki wydawania sądów",
            "Psychologia kolorów i jej wpływ na marketing",
            "Rozwój moralny według Kohlberga",
            "Teoria umysłu i empatia",
            "Zjawisko flow (przepływu) Mihaly Csikszentmihalyi",
            "Temperament a osobowość",
            "Psychologia ewolucyjna",
            "Wypalenie zawodowe: przyczyny i zapobieganie",
        ],
        "Grafika i design": [
            "Podstawowe zasady kompozycji w projektowaniu graficznym",
            "Teoria kolorów i jej zastosowanie (koło barw, harmonia)",
            "Różnice między grafiką wektorową a rastrową",
            "Typografia: dobór i łączenie fontów",
            "Czym jest User Interface (UI) i User Experience (UX)",
            "Projektowanie logo i tożsamości wizualnej marki",
            "Podstawy obsługi Adobe Photoshop i Illustrator",
            "Trendy w web designie",
            "Dostępność cyfrowa (WCAG) w projektowaniu",
            "Tworzenie moodboardów w procesie kreatywnym",
            "Złoty podział i jego zastosowanie w designie",
            "Projektowanie responsywnych stron internetowych (RWD)",
            "Tworzenie systemów projektowych (Design Systems)",
            "Animacja i motion design (After Effects)",
            "Modelowanie 3D i rendering (Blender, Cinema 4D)",
            "Grid systems w projektowaniu layoutów",
            "Psychologia Gestalt w projektowaniu wizualnym",
            "Projektowanie ikon",
            "Przygotowanie plików do druku (DTP)",
            "Licencje na czcionki i obrazy",
            "Minimalizm w designie",
            "Brutalizm w web designie",
            "Projektowanie dla urządzeń mobilnych (Mobile-First)",
            "Narzędzia do prototypowania (Figma, Adobe XD)",
            "Gamifikacja w UX",
            "Testy A/B w projektowaniu",
            "Projektowanie zorientowane na użytkownika (User-Centered Design)",
            "Dobra praktyka w projektowaniu formularzy",
            "Tworzenie portfolio projektanta",
            "Sztuka feedbacku w procesie projektowym",
        ],
        "Pisarstwo i storytelling": [
            "Struktura trzech aktów w narracji",
            "Podróż bohatera (Monomyth) Josepha Campbella",
            "Jak tworzyć wiarygodne i złożone postacie",
            "Technika 'Show, don't tell' (Pokaż, nie opowiadaj)",
            "Budowanie napięcia i suspensu w opowieści",
            "Znaczenie dialogów w prozie",
            "Kreatywne pisanie: ćwiczenia na pobudzenie wyobraźni",
            "Copywriting: pisanie tekstów, które sprzedają",
            "Różne typy narratorów (pierwszoosobowy, trzecioosobowy)",
            "Storytelling w marketingu i budowaniu marki",
            "Konflikt jako motor napędowy fabuły",
            "Tworzenie światów (World-building) w fantasy i sci-fi",
            "Pisanie scenariuszy filmowych i telewizyjnych",
            "Poezja: formy i środki stylistyczne",
            "Redakcja i korekta własnego tekstu",
            "Jak znaleźć własny styl pisarski",
            "Pisanie bloga i angażowanie czytelników",
            "Ghostwriting: pisanie dla kogoś",
            "Prawo autorskie dla pisarzy",
            "Jak pokonać blokadę pisarską",
            "Techniki researchu dla pisarzy",
            "Pisanie literatury faktu (non-fiction)",
            "Tworzenie opisów postaci i miejsc",
            "Tempo i rytm w prozie",
            "Symbolika i motywy w literaturze",
            "Adaptacja książki na scenariusz",
            "Pisanie opowiadań (short stories)",
            "Budowanie marki osobistej jako autor",
            "Self-publishing vs. wydawnictwo tradycyjne",
            "Sztuka pisania dobrych zakończeń",
        ],
        "Sport i zdrowy styl życia": [
            "Podstawy treningu siłowego w kulturystyce",
            "Zasady walki w MMA: stójka, parter, zapasy",
            "Rola białka, węglowodanów i tłuszczów w diecie sportowca",
            "Trening interwałowy o wysokiej intensywności (HIIT)",
            "Regeneracja po treningu: sen, stretching, odnowa biologiczna",
            "Suplementacja w sporcie: kreatyna, BCAA, białko serwatkowe",
            "Historia i zasady piłki nożnej",
            ("Biomechanika podstawowych ćwiczeń siłowych " "(przysiad, martwy ciąg)"),
            "Psychologia sportu: motywacja i radzenie sobie z presją",
            "Najczęstsze kontuzje sportowe i ich prewencja",
            "Trening funkcjonalny i jego zalety",
            "Kalistenika: trening z masą własnego ciała",
            "Dieta ketogeniczna i jej wpływ na organizm",
            "Post przerywany (Intermittent Fasting)",
            "Joga i jej wpływ na ciało i umysł",
            "Doping w sporcie: rodzaje i konsekwencje",
            "Trening cardio: bieganie, pływanie, rower",
            "Rola mikroskładników (witaminy, minerały) w diecie",
            "Periodyzacja treningu",
            "Trening mobilności i elastyczności",
            "Historia Igrzysk Olimpijskich",
            "Analiza składu ciała (tkanka tłuszczowa, masa mięśniowa)",
            "CrossFit: metodologia i podstawowe ćwiczenia",
            "Sporty ekstremalne: snowboarding, wspinaczka, surfing",
            "Trening mentalny i wizualizacja w sporcie",
            "Rola nawodnienia organizmu",
            "Zdrowie stawów i tkanki łącznej",
            "Morsowanie i jego wpływ na zdrowie",
            "Badania lekarskie dla sportowców",
            "Różnice w treningu kobiet i mężczyzn",
        ],
        "Mechanika i technika": [
            "Zasada działania silnika czterosuwowego",
            "Podstawy chłodnictwa i cykl termodynamiczny pompy ciepła",
            "Różnice między silnikiem diesla a benzynowym",
            "Podstawowe narzędzia w warsztacie mechanicznym",
            "Jak działa układ hamulcowy w samochodzie",
            "Spawanie: metody MIG, TIG, MMA",
            "Podstawy elektroniki: prawo Ohma, tranzystory, kondensatory",
            "Diagnostyka komputerowa pojazdów (OBD-II)",
            "Energia odnawialna: panele fotowoltaiczne i turbiny wiatrowe",
            "Druk 3D: technologie FDM, SLA, SLS",
            "Układ napędowy: skrzynia biegów, sprzęgło, dyferencjał",
            "Pneumatyka i hydraulika siłowa",
            "Obróbka skrawaniem: toczenie, frezowanie, szlifowanie",
            "Materiały inżynierskie: stale, stopy aluminium, kompozyty",
            "Systemy klimatyzacji w budynkach i pojazdach",
            "Automatyka i robotyka przemysłowa",
            "Silniki elektryczne: budowa i zasada działania",
            "Technologia CNC (Computerized Numerical Control)",
            "Podstawy metrologii warsztatowej",
            "Korozja i metody ochrony metali",
            "Systemy zawieszenia w pojazdach",
            "Technologia opon samochodowych",
            "Lutowanie w elektronice",
            "Programowalne sterowniki logiczne (PLC)",
            "Zasada działania turbosprężarki",
            "Akumulatory i systemy magazynowania energii",
            "Sieci komputerowe: topologie, protokoły, sprzęt",
            "Technologia GPS i systemy nawigacji",
            "Drony: budowa i zastosowanie",
            "Internet Rzeczy (IoT) w przemyśle",
        ],
        "Moda i styl": [
            "Historia mody XX wieku",
            "Jak znaleźć perełki w sklepach z odzieżą używaną (lumpeksach)",
            "Podstawy budowania garderoby kapsułowej",
            "Różne style w modzie męskiej (casual, smart casual, business)",
            "Wpływ subkultur na modę (punk, hip-hop, grunge)",
            "Slow fashion vs fast fashion",
            "Jak dbać o ubrania, by służyły dłużej",
            "Dobór ubioru do sylwetki",
            "Kultowe modele butów sportowych (sneakersów)",
            "Znaczenie dodatków w stylizacji",
            "Historia dżinsu",
            ("Wielcy projektanci mody (Coco Chanel, Yves Saint Laurent, " "Alexander McQueen)"),
            ("Tygodnie mody (Fashion Weeks) w Paryżu, Mediolanie, " "Londynie, Nowym Jorku"),
            "Moda vintage i retro",
            ("Materiały w modzie: bawełna, wełna, jedwab, " "materiały syntetyczne"),
            "Dress code w biznesie i na oficjalnych uroczystościach",
            "Streetwear: historia i kluczowe marki",
            "Pielęgnacja obuwia skórzanego",
            "Trendy w modzie na przestrzeni dekad",
            "Etyka w modzie: zrównoważona produkcja i sprawiedliwy handel",
            "Sztuka łączenia wzorów i kolorów",
            "Moda uniseks i gender fluid",
            "Historia garnituru męskiego",
            "Wpływ kina i muzyki na modę",
            "Personalizacja ubrań (customization)",
            "Jak czytać metki i składy ubrań",
            "Upcycling i recykling w modzie",
            "Krawiectwo miarowe (bespoke)",
            "Analiza kolorystyczna i typy urody",
            "Wpływ social mediów na trendy w modzie",
        ],
        "Podróże i turystyka": [
            "Jak tanio planować podróże i szukać lotów",
            "Różnice między hotelami, hostelami i Airbnb",
            "Bezpieczeństwo w podróży solo",
            "Pakowanie plecaka na długą podróż (backpacking)",
            "Jak korzystać z transportu publicznego w obcym mieście",
            "Kultura i etykieta w krajach Azji Południowo-Wschodniej",
            "Najpiękniejsze parki narodowe w USA",
            "Wskazówki dotyczące podróżowania z ograniczonym budżetem",
            "Cyfrowi nomadzi: praca zdalna w podróży",
            "Ubezpieczenie turystyczne: co powinno zawierać",
            "Programy lojalnościowe linii lotniczych i hoteli",
            "House sitting i couchsurfing jako alternatywne formy noclegu",
            "Jak radzić sobie z jet lagiem",
            "Podróżowanie z dziećmi: porady i wskazówki",
            "Wymiana walut i korzystanie z kart płatniczych za granicą",
            "Wymagane wizy i dokumenty podróżne",
            "Szczepienia i zdrowie w podróży tropikalnej",
            "Turystyka kulinarna: odkrywanie smaków świata",
            "Ekoturystyka i podróżowanie w sposób zrównoważony",
            "Najsłynniejsze szlaki trekkingowe na świecie",
            "Podróżowanie kamperem (van life)",
            "Jak robić dobre zdjęcia w podróży",
            "Aplikacje mobilne przydatne w podróży",
            "Podróżowanie pociągiem po Europie (Interrail)",
            "Wolontariat międzynarodowy (workaway, WWOOF)",
            "Jak nauczyć się podstaw lokalnego języka przed wyjazdem",
            "Radzenie sobie z szokiem kulturowym",
            "Oszustwa turystyczne i jak ich unikać",
            "Podróżowanie poza utartym szlakiem (off-the-beaten-path)",
            "Pamiątki z podróży: co warto przywieźć",
        ],
        "Biznes i analityka": [
            "Analiza SWOT: mocne i słabe strony, szanse i zagrożenia",
            "Lejek sprzedażowy w marketingu online",
            "Podstawy SEO (Search Engine Optimization)",
            "Jak działają aukcje internetowe (np. model aukcji angielskiej)",
            "Kluczowe wskaźniki efektywności (KPI) w biznesie",
            "Model biznesowy Canvas",
            "Podstawy analizy danych w Pythonie (biblioteka Pandas)",
            "Negocjacje biznesowe: techniki i strategie",
            "Marketing w mediach społecznościowych",
            "Zarządzanie projektami: metodyki Agile i Scrum",
            "Analiza konkurencji",
            "Customer Relationship Management (CRM)",
            "Badania rynku i ankiety",
            "Content marketing i jego rola",
            "E-mail marketing i automatyzacja",
            "Google Analytics i analiza ruchu na stronie",
            "Budowanie marki osobistej (personal branding)",
            "Modele subskrypcyjne w biznesie",
            "Finansowanie startupów: venture capital, aniołowie biznesu",
            "Analiza finansowa przedsiębiorstwa",
            "Zarządzanie ryzykiem w projekcie",
            "Myślenie projektowe (Design Thinking) w biznesie",
            "Franczyza jako model biznesowy",
            "E-commerce: platformy i strategie (Shopify, WooCommerce)",
            "Public relations (PR) i komunikacja z mediami",
            "Networking i budowanie relacji biznesowych",
            "Prawo gospodarcze dla przedsiębiorców",
            "Optymalizacja konwersji (CRO)",
            "Sztuka prezentacji i wystąpień publicznych",
            "Zarządzanie zespołem zdalnym",
        ],
        "Rozwój osobisty i produktywność": [
            "Technika Pomodoro w zarządzaniu czasem",
            "Metoda GTD (Getting Things Done) Davida Allena",
            "Jak budować nawyki i pozbywać się złych",
            "Mindfulness i medytacja dla początkujących",
            "Ustalanie celów metodą SMART",
            "Sztuka asertywności w komunikacji",
            "Koncepcja 'deep work' (pracy głębokiej) Cala Newporta",
            "Jak efektywnie się uczyć i zapamiętywać informacje",
            "Rola odpoczynku w produktywności",
            "Syndrom oszusta i jak sobie z nim radzić",
            "Macierz Eisenhowera do priorytetyzacji zadań",
            "Prowadzenie dziennika (journaling) i jego korzyści",
            "Sztuka mówienia 'nie'",
            "Inteligencja finansowa i podstawy oszczędzania",
            "Szybkie czytanie i techniki notowania (mapy myśli)",
            "Budowanie odporności psychicznej (rezyliencji)",
            "Work-life balance i jego znaczenie",
            "Minimalizm jako filozofia życia",
            "Znajdowanie swojego 'dlaczego' (purpose)",
            "Efektywne słuchanie aktywne",
            "Jak dawać i przyjmować konstruktywny feedback",
            "Pokonywanie prokrastynacji",
            "Rola snu w funkcjonowaniu mózgu",
            "Uczenie się przez całe życie (lifelong learning)",
            "Zarządzanie stresem i techniki relaksacyjne",
            "Budowanie pewności siebie",
            "Znaczenie porannej rutyny",
            "Cyfrowy detoks i higiena cyfrowa",
            "Myślenie krytyczne i rozwiązywanie problemów",
            "Stoicyzm jako praktyczna filozofia na co dzień",
        ],
        "Filozofia i idee": [
            "Wprowadzenie do stoicyzmu: Epiktet, Seneka, Marek Aureliusz",
            "Egzystencjalizm: Sartre, Camus i wolność wyboru",
            "Imperatyw kategoryczny Immanuela Kanta",
            "Alegoria jaskini Platona i teoria idei",
            "Nihilizm i 'wola mocy' Nietzschego",
            "Utylitaryzm: zasada największego szczęścia",
            "Myśl polityczna Machiavellego",
            "Taoizm i koncepcja Wu Wei",
            "Debata natura czy wychowanie (nature vs. nurture)",
            "Problem zła w filozofii",
            "Umowa społeczna (Hobbes, Locke, Rousseau)",
            "Kartezjańskie 'Cogito, ergo sum'",
            "Absurdyzm Alberta Camusa",
            "Fenomenologia Edmunda Husserla",
            "Etyka cnoty Arystotelesa",
            "Filozofia dialogu (Buber, Levinas)",
            "Poststrukturalizm (Foucault, Derrida)",
            "Szkoła frankfurcka i teoria krytyczna",
            "Filozofia języka Wittgensteina",
            "Eksperyment myślowy 'mózg w naczyniu'",
            "Determinizm vs. wolna wola",
            "Problem ciała i umysłu (dualizm vs. monizm)",
            "Filozofia nauki Karla Poppera (falsyfikacjonizm)",
            "Cynizm i Diogenes z Synopy",
            "Epikureizm i dążenie do szczęścia",
            "Transhumanizm i przyszłość ludzkości",
            "Filozofia Wschodu: buddyzm, hinduizm, konfucjanizm",
            "Problem tożsamości osobowej",
            "Estetyka: czym jest piękno?",
            "Filozofia polityczna: liberalizm, konserwatyzm, socjalizm",
        ],
        "Nauka i wszechświat": [
            "Teoria Wielkiego Wybuchu i ewolucja wszechświata",
            "Ogólna i szczególna teoria względności Einsteina",
            "Podstawy mechaniki kwantowej",
            "Czarne dziury i horyzont zdarzeń",
            "Struktura DNA i podstawy genetyki",
            "Ewolucja gatunków według Darwina",
            "Ciemna materia i ciemna energia",
            "Standardowy Model cząstek elementarnych",
            "Poszukiwanie życia pozaziemskiego (paradoks Fermiego)",
            "Zmiany klimatyczne: przyczyny i skutki",
            "Tektonika płyt i cykl skalny",
            "Fotosynteza i oddychanie komórkowe",
            "Ludzki mózg: budowa i funkcje",
            "Układ Słoneczny i jego planety",
            "Fale grawitacyjne i ich detekcja (LIGO)",
            "Edycja genów CRISPR-Cas9",
            "Sztuczna inteligencja: sieci neuronowe i głębokie uczenie",
            "Antybiotyki i problem oporności bakterii",
            "Cykl życia gwiazd",
            "Promieniowanie tła kosmicznego",
            "Zasady termodynamiki",
            "Historia odkryć naukowych",
            "Metoda naukowa w praktyce",
            "Energia jądrowa: fuzja i rozszczepienie",
            "Teleskop Hubble'a i Jamesa Webba",
            "Kwantowa teoria pola",
            "Pochodzenie życia na Ziemi (abiogeneza)",
            "Wirusy i ich rola w ekosystemie",
            "Psychodeliki i ich wpływ na mózg z perspektywy neurobiologii",
            "Komputery kwantowe: zasada działania",
        ],
        "Sztuka i kultura": [
            "Główne nurty w malarstwie: od renesansu do współczesności",
            "Historia kina: kluczowe filmy i reżyserzy",
            "Czym jest muzyka klasyczna: epoki i kompozytorzy",
            "Architektura gotycka, barokowa i modernistyczna",
            "Wpływ mitologii greckiej na kulturę zachodnią",
            "Street art: od graffiti do murali",
            "Podstawy fotografii: kompozycja, światło, ekspozycja",
            "Historia jazzu i jego najważniejsi przedstawiciele",
            "Teatr absurdu: Beckett i Ionesco",
            "Postmodernizm w sztuce i literaturze",
            "Impresjonizm i jego rewolucja w malarstwie",
            "Kubizm i Pablo Picasso",
            "Surrealizm i Salvador Dalí",
            "Pop-art i Andy Warhol",
            "Bauhaus: szkoła i styl w designie",
            "Opera: historia i najważniejsze dzieła",
            "Historia rock and rolla",
            "Kino Nowej Fali (francuskie, japońskie)",
            "Sztuka performance i happening",
            "Land art: sztuka ziemi",
            "Fotografia dokumentalna i reportażowa",
            "Rola muzeum w dzisiejszym świecie",
            "Konserwacja dzieł sztuki",
            "Rynek sztuki i aukcje",
            "Sztuka cyfrowa i generatywna",
            "Kultura remiksu i jej wpływ na twórczość",
            "Antropologia kulturowa: badanie różnorodności kultur",
            "Memetyka i kultura internetowa",
            "Ikony popkultury XX wieku",
            "Wpływ globalizacji na kultury lokalne",
        ],
        "Gotowanie i kulinaria": [
            "Pięć smaków podstawowych i umami",
            "Podstawowe techniki krojenia warzyw (julienne, brunoise)",
            "Jak działa proces fermentacji (kiszonki, chleb na zakwasie)",
            "Różnice między kuchnią włoską, francuską i chińską",
            "Sztuka parzenia kawy: metody przelewowe i ciśnieniowe",
            "Podstawy tworzenia sosów (emulsje, redukcje)",
            "Dobieranie wina do potraw (food pairing)",
            "Bezpieczeństwo żywności i temperatury przechowywania",
            "Historia przypraw i ich rola w kuchni",
            "Slow food: filozofia i praktyka",
            "Reakcja Maillarda i karmelizacja",
            "Sous-vide: gotowanie w niskiej temperaturze",
            "Kuchnia molekularna: techniki i przykłady",
            "Sztuka wypieku chleba",
            "Cukiernictwo: podstawy i techniki (np. temperowanie czekolady)",
            "Kultura jedzenia ulicznego (street food) na świecie",
            "Dieta śródziemnomorska i jej korzyści zdrowotne",
            "Wegetarianizm i weganizm: zasady i zamienniki",
            "Przewodnik po serach: rodzaje i produkcja",
            "Oliwa z oliwek: rodzaje i zastosowanie",
            "Sztuka robienia sushi",
            "Grillowanie i BBQ: techniki i marynaty",
            "Koktajle i miksologia",
            "Historia pizzy",
            "Superfoods: fakty i mity",
            "Przechowywanie żywności i zero waste w kuchni",
            "Noże kuchenne: rodzaje i pielęgnacja",
            "Rola soli w gotowaniu",
            "Tradycyjna kuchnia polska",
            "Gwiazdki Michelin: historia i kryteria",
        ],
    }

    for category, topics in knowledge_base.items():
        print(f"\n\n{'='*20} ROZPOCZYNANIE KATEGORII: " f"{category.upper()} {'='*20}")
        for topic in topics:
            seed_topic(topic, category)
            time.sleep(5)  # Dłuższa przerwa między tematami

    print("\n\n✅✅✅ ZAKOŃCZONO CAŁY PROCES ZASILANIA PAMIĘCI. ✅✅✅")
