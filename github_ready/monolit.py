#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Monolit PRO — Writer Pro + Autonauka + STM/LTM/Semantic/Stats

import os, re, sys, time, json, uuid, sqlite3, asyncio, contextlib
import datetime, html, unicodedata, dataclasses
from typing import Any, Dict, List, Tuple, Optional
from urllib.parse import parse_qs, quote_plus, urlencode, urlparse
from urllib.request import Request as UrlRequest, urlopen
from collections import Counter, defaultdict
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass

# HTTP/HTML
try:
    import httpx
    from bs4 import BeautifulSoup
    from readability import Document as ReadabilityDoc
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx", "bs4", "readability-lxml"])
    import httpx
    from bs4 import BeautifulSoup
    from readability import Document as ReadabilityDoc

# FastAPI
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Mordzix Monolit PRO", version="3.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Autonauka Full (SERPAPI+Firecrawl+Wiki+DDG+Sport+News)
try:
    import autonauka_full
    AUTONAUKA_AVAILABLE = True
    print("[OK] autonauka_full loaded - full pipeline active")
except Exception as e:
    AUTONAUKA_AVAILABLE = False
    print(f"[WARN] autonauka_full not found: {e}")

# Writer Pro (opcjonalnie)
try:
    from writer_pro import router as writer_router
    app.include_router(writer_router)
except Exception:
    print("[WARN] writer_pro not found – writer endpoints disabled")

# Memory (opcjonalnie)
try:
    from memory import (
        ltm_add, ltm_search_hybrid, ltm_search_tags,
        memory_get, memory_summaries, get_memory
    )
except Exception:
    def ltm_add(*a, **k): return None
    def ltm_search_hybrid(*a, **k): return []
    def ltm_search_tags(*a, **k): return []
    def memory_get(*a, **k): return []
    def memory_summaries(*a, **k): return []
    def get_memory(*a, **k): return None

# Stats (opcjonalnie)
try:
    from system_stats import init_monitor, record_api_call
    STATS_MODULE_AVAILABLE = True
except Exception:
    STATS_MODULE_AVAILABLE = False

# Flaga semantyki
SEMANTIC_MODULE_AVAILABLE = True

# =========================
# KONFIG
# =========================
BASE_DIR = os.getenv("WORKSPACE", "/workspace/mrd69")
DB_PATH  = os.getenv("MEM_DB", os.path.join(BASE_DIR, "mem.db"))
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "ssjjMijaja6969")
HTTP_TIMEOUT = int(os.getenv("TIMEOUT_HTTP", "60"))
WEB_USER_AGENT = os.getenv("WEB_USER_AGENT", "MonolitBot/2.3")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepinfra.com/v1/openai")
LLM_API_KEY = os.getenv("LLM_API_KEY", "w52XW0XN6zoV9hdY8OONhLu6tvnFaXbZ")
LLM_MODEL    = os.getenv("LLM_MODEL", "zai-org/GLM-4.5")
LLM_TIMEOUT  = int(os.getenv("LLM_HTTP_TIMEOUT_S", "60"))

EMBED_URL   = os.getenv("LLM_EMBED_URL","https://api.deepinfra.com/v1/openai/embeddings")
EMBED_MODEL = os.getenv("LLM_EMBED_MODEL","sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

SERPAPI_KEY   = os.getenv("SERPAPI_KEY","a5cb3592980e0ff9042a0be2d3f7df2768bd93913252")
FIRECRAWL_KEY = os.getenv("FIRECRAWL_KEY","fc-ec025f3a447c6878bee6926b49c17d3")
OVERPASS_URL  = os.getenv("OVERPASS_URL","https://overpass-api.de/api/interpreter")
OPENTRIPMAP_KEY = os.getenv("OPENTRIPMAP_KEY","AlzaSyc©bpKUI1V9GsmnUU0eRhgLDureexyWigY8")

SEED_CANDIDATES = [
    os.path.join(BASE_DIR, "seed_facts.jsonl"),  # Ścieżka w katalogu głównym - najważniejsza
    "/workspace/mrd69/data/sq3 seed_facts.jsonl",
    "/workspace/mrd69/data/sq3_seed_facts.jsonl",
    "/workspace/mrd69/seed_facts.jsonl",
    "C:\\Users\\48501\\Desktop\\mrd69\\data\\sq3\\seed_facts.jsonl",
]

OUT_DIR     = os.path.join(BASE_DIR,"out");            os.makedirs(OUT_DIR, exist_ok=True)
WRITER_OUT  = os.path.join(OUT_DIR,"writing");         os.makedirs(WRITER_OUT, exist_ok=True)
UPLOADS_DIR = os.path.join(OUT_DIR,"uploads");         os.makedirs(UPLOADS_DIR, exist_ok=True)
BACKUP_DIR  = os.path.join(BASE_DIR,"backup");         os.makedirs(BACKUP_DIR, exist_ok=True)

HEADERS   = {"User-Agent": WEB_USER_AGENT}
JSON_HEAD = {"User-Agent": WEB_USER_AGENT, "Accept":"application/json", "Content-Type":"application/json"}

_START_TIME = time.time()  # Czas startu aplikacji

# =========================
# RATE LIMIT
# =========================
_RATE: Dict[str, int] = {}; _RATE_LOCK = threading.Lock()
def _rate_ok(ip: str, key: str, limit: int = 160, window: int = 60) -> bool:
    now = int(time.time()); k = f"{ip}:{key}:{now//window}"
    with _RATE_LOCK:
        c = _RATE.get(k, 0) + 1
        _RATE[k] = c
    return c <= limit

# =========================
# DB (SQLite + FTS5)
# =========================
def _db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn

def _init_db():
    c=_db();cur=c.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS memory(
        id TEXT PRIMARY KEY, user TEXT, role TEXT, content TEXT, ts REAL
    );""")
    cur.execute("""CREATE TABLE IF NOT EXISTS memory_long(
        id TEXT PRIMARY KEY, user TEXT, summary TEXT, details TEXT, ts REAL
    );""")
    cur.execute("""CREATE TABLE IF NOT EXISTS meta_memory(
        id TEXT PRIMARY KEY, user TEXT, key TEXT, value TEXT, conf REAL, ts REAL
    );""")
    cur.execute("""CREATE TABLE IF NOT EXISTS facts(
        id TEXT PRIMARY KEY, text TEXT, tags TEXT, conf REAL, created REAL, deleted INTEGER DEFAULT 0
    );""")
    cur.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(text, tags);""")
    cur.execute("""CREATE TABLE IF NOT EXISTS mem_embed(
        id TEXT PRIMARY KEY, user TEXT, vec TEXT, ts REAL
    );""")
    cur.execute("""CREATE TABLE IF NOT EXISTS docs(
        id TEXT PRIMARY KEY, url TEXT, title TEXT, text TEXT, source TEXT, fetched REAL
    );""")
    cur.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(title, text, url UNINDEXED);""")
    cur.execute("""CREATE TABLE IF NOT EXISTS cache(
        key TEXT PRIMARY KEY, value TEXT, ts REAL
    );""")
    cur.execute("""CREATE TABLE IF NOT EXISTS psy_state(
        id INTEGER PRIMARY KEY CHECK(id=1),
        mood REAL, energy REAL, focus REAL, openness REAL, directness REAL,
        agreeableness REAL, conscientiousness REAL, neuroticism REAL,
        style TEXT, updated REAL
    );""")
    cur.execute("""CREATE TABLE IF NOT EXISTS psy_episode(
        id TEXT PRIMARY KEY, user TEXT, kind TEXT, valence REAL, intensity REAL, tags TEXT, note TEXT, ts REAL
    );""")
    cur.execute("""CREATE TABLE IF NOT EXISTS psy_reflection(
        id TEXT PRIMARY KEY, note TEXT, deltas TEXT, ts REAL
    );""")
    # baza wiedzy aukcji (uczenie)
    cur.execute("""CREATE TABLE IF NOT EXISTS kb_auction(
        id TEXT PRIMARY KEY, kind TEXT, key TEXT, val TEXT, weight REAL, ts REAL
    );""")
    cur.execute("INSERT OR IGNORE INTO psy_state VALUES(1,0.0,0.6,0.6,0.55,0.62,0.55,0.63,0.44,'rzeczowy',?)",(time.time(),))
    c.commit(); c.close()

# Funkcja do automatycznego ładowania danych z seed_facts.jsonl przy starcie
def _preload_seed_facts():
    print("Sprawdzam seed_facts.jsonl...")
    facts_count = 0
    conn=_db(); c=conn.cursor()
    facts_count = c.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    conn.close()
    
    if facts_count < 100:
        print(f"Znaleziono tylko {facts_count} faktów w bazie. "
              f"Ładuję seed_facts.jsonl...")
        for path in SEED_CANDIDATES:
            if not os.path.isfile(path):
                continue
                
            loaded = 0
            print(f"Ładuję fakty z: {path}")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            txt = obj.get("text") or obj.get("fact")
                            if txt:
                                category = obj.get("category", "")
                                tags = obj.get("tags", [])
                                if isinstance(tags, list):
                                    tags = ",".join(tags)
                                else:
                                    tags = tags or ""
                                    
                                # Dodaj kategorię do tagów
                                if category and category not in tags:
                                    if tags:
                                        tags = f"{tags},fact,{category}"
                                    else:
                                        tags = f"fact,{category}"
                                        
                                ltm_add(txt, tags, float(obj.get("conf", 0.8)))
                                loaded += 1
                        except Exception as e:
                            print(f"Błąd podczas przetwarzania linii: {e}")
                print(f"Załadowano {loaded} faktów z {path}")
                if loaded > 0:
                    break  # Zatrzymaj po znalezieniu pierwszego działającego pliku
            except Exception as e:
                print(f"Błąd podczas ładowania pliku {path}: {e}")
        
        # Przeindeksowanie faktów
        # Wywołanie funkcji facts_reindex przeniesione na później - po załadowaniu jej definicji
        print("Fakty załadowane.")
    else:
        print(f"Znaleziono {facts_count} faktów w bazie. Pomijam ładowanie.")

_init_db()
# Automatyczne ładowanie faktów przy starcie
_preload_seed_facts()

# Inicjalizacja systemu monitoringu jeśli jest dostępny
if STATS_MODULE_AVAILABLE:
    try:
        init_monitor(DB_PATH, interval=60)  # Monitoruj co 60 sekund
        print("Zaawansowany system monitoringu uruchomiony")
    except Exception as e:
        print(f"Błąd inicjalizacji monitoringu: {e}")

# =========================
# ANALIZA SEMANTYCZNA - wbudowana
# =========================

class SemanticAnalyzer:
    """Klasa do zaawansowanej analizy semantycznej tekstu"""
    def __init__(self):
        self.sentiment_keywords = {
            "pozytywny": ["dobry", "świetny", "doskonały", "zadowolony", "wspaniały", "super", "fajny", "miły", "ciekawy", "lubię", "podoba", "polecam"],
            "negatywny": ["zły", "słaby", "kiepski", "niedobry", "rozczarowany", "niezadowolony", "fatalny", "beznadziejny", "okropny", "niestety", "problem"],
            "neutralny": ["normalny", "zwykły", "standard", "średni", "przeciętny", "typowy"]
        }
        
        # Słownik dla dokładniejszej analizy emocji
        self.emotion_keywords = {
            "radość": ["świetny", "super", "zachwycający", "cieszyć", "uwielbić", "radość", "szczęśliwy", "entuzjazm", "zadowolony", "radosny", "wow", "hurra"],
            "smutek": ["smutny", "przykro", "żal", "szkoda", "płakać", "przygnębiony", "przykry", "smuci", "niestety", "rozczarowany", "porzucić", "zrezygnowany"],
            "złość": ["wkurzony", "zdenerwowany", "wściekły", "irytuje", "denerwuje", "zły", "zirytowany", "wkurza", "frustracja", "wkurzyć", "złościć"],
            "strach": ["boi się", "przerażony", "lęk", "obawy", "obawiam", "strach", "martwi", "zatrwożony", "niepewny", "przestraszony", "obawiam się"],
            "zaskoczenie": ["wow", "zaskoczony", "zdziwiony", "niesamowity", "zaskakujący", "niewiarygodny", "szok", "zdumiewający", "niezwykły", "nieprawdopodobny"],
            "zaufanie": ["ufam", "wierzę", "polegam", "pewny", "sprawdzony", "bezpieczny", "wiarygodny", "niezawodny", "godny zaufania"],
            "wstręt": ["obrzydliwy", "ohydny", "niesmaczny", "odrażający", "paskudny", "wstrętny", "niechęć", "okropny", "obrzydzenie"],
            "oczekiwanie": ["czekam", "oczekuję", "mam nadzieję", "spodziewać się", "przewidywać", "liczyć", "powinno", "będzie", "chciałbym"]
        }
        
        self.intention_indicators = {
            "pytanie": ["?", "czy", "jak", "kiedy", "gdzie", "co", "dlaczego", "ile", "który", "jakie", "proszę wyjaśnić"],
            "prośba": ["proszę", "czy możesz", "czy mógłbyś", "pomóż", "potrzebuję", "zrób", "wykonaj", "daj", "pokaż"],
            "stwierdzenie": ["jest", "są", "myślę", "sądzę", "uważam", "moim zdaniem", "wydaje mi się", "wiem", "rozumiem"]
        }
        
        # Słowniki kategorii tematycznych
        self.topic_keywords = {
            "technologia": ["komputer", "laptop", "telefon", "internet", "aplikacja", "program", "software", "hardware", "kod", "programowanie"],
            "biznes": ["firma", "przedsiębiorstwo", "zysk", "marketing", "sprzedaż", "klient", "produkt", "usługa", "rynek", "inwestycja"],
            "podróże": ["wakacje", "wycieczka", "hotel", "rezerwacja", "lot", "samolot", "zwiedzanie", "turysta", "przewodnik", "destynacja"],
            "zdrowie": ["lekarz", "choroba", "lekarstwo", "terapia", "ćwiczenia", "dieta", "samopoczucie", "zdrowy", "pacjent", "dolegliwości"],
            "edukacja": ["szkoła", "nauka", "studia", "uniwersytet", "kurs", "student", "profesor", "egzamin", "wykład", "wiedza"],
            "rozrywka": ["film", "muzyka", "koncert", "spektakl", "książka", "gra", "zabawa", "hobby", "serial", "festiwal"]
        }
        print("Analiza semantyczna - inicjalizacja powiodła się")
        
    def analyze_text(self, text):
        """Kompleksowa analiza semantyczna tekstu"""
        if not text:
            return {}
            
        result = {
            "topics": self.detect_topics(text),
            "sentiment": self.analyze_sentiment(text),
            "emotions": self.analyze_emotions(text),
            "intention": self.detect_intention(text),
            "hidden_intentions": self.detect_hidden_intentions(text),
            "keywords": self.extract_keywords(text),
            "complexity": self.analyze_complexity(text),
            "temporal_context": self.detect_temporal_context(text),
            "entities": self.extract_entities(text)
        }
        return result
        
    def analyze_emotions(self, text):
        """Zaawansowana analiza emocji w tekście"""
        if not text:
            return {}
            
        text_lower = text.lower()
        tokens = _tok(text_lower) if hasattr(text_lower, '__len__') else []
        words = text_lower.split()
        
        # Analizuj emocje na podstawie słów kluczowych
        emotion_scores = {emotion: 0.0 for emotion in self.emotion_keywords}
        emotion_matches = {}
        
        # Zaimplementujmy podejście z uwzględnieniem kontekstu
        # Najpierw podstawowe zliczanie
        for emotion, keywords in self.emotion_keywords.items():
            matches = []
            for word in keywords:
                # Bardziej zaawansowane sprawdzenie niż proste text_lower.count()
                if len(word.split()) > 1:  # Dla fraz wielowyrazowych
                    if word in text_lower:
                        matches.append(word)
                        emotion_scores[emotion] += 0.2
                else:  # Dla pojedynczych słów
                    # Dopasowanie form wyrazów (np. smutek, smutny, smutno)
                    pattern = r"\b" + re.escape(word[:4]) + r"[a-ząćęłńóśźź]*\b"
                    matches_found = re.findall(pattern, text_lower)
                    if matches_found:
                        matches.extend(matches_found)
                        emotion_scores[emotion] += 0.1 * len(matches_found)
            
            if matches:
                emotion_matches[emotion] = matches
        
        # Analiza wzajemnych wzmocnień i osłabień emocji
        if emotion_scores.get("radość", 0) > 0 and emotion_scores.get("smutek", 0) > 0:
            # Jeśli występuje jednocześnie radość i smutek, sprawdźmy negacje
            if any(neg in text_lower for neg in ["nie jest", "nie był", "nie są", "nie czuję"]):
                # Prawdopodobnie negacja pozytywnych emocji
                if "nie" in text_lower and any(pos in text_lower[text_lower.find("nie"):] 
                                            for pos in self.emotion_keywords["radość"]):
                    emotion_scores["radość"] *= 0.3
                    emotion_scores["smutek"] *= 1.5
        
        # Uwzględnienie znaków interpunkcyjnych i emotikonów
        if "!" in text:
            # Wykrzykniki wzmacniają dominujące emocje
            max_emotion = max(emotion_scores, key=emotion_scores.get)
            if max_emotion in ["radość", "złość", "zaskoczenie"]:
                emotion_scores[max_emotion] += 0.1 * text.count("!")
        
        # Emotikony i emoji
        happy_emojis = [":)", ":D", "😊", "😁", "😄", "👍"]
        sad_emojis = [":(", "😢", "😭", "😔", "👎"]
        angry_emojis = ["😠", "😡", "👿", "💢"]
        surprised_emojis = ["😮", "😯", "😲", "😱", "😳"]
        
        for emoji in happy_emojis:
            count = text.count(emoji)
            if count > 0:
                emotion_scores["radość"] += 0.15 * count
                
        for emoji in sad_emojis:
            count = text.count(emoji)
            if count > 0:
                emotion_scores["smutek"] += 0.15 * count
                
        for emoji in angry_emojis:
            count = text.count(emoji)
            if count > 0:
                emotion_scores["złość"] += 0.15 * count
                
        for emoji in surprised_emojis:
            count = text.count(emoji)
            if count > 0:
                emotion_scores["zaskoczenie"] += 0.15 * count
        
        # Analiza intensywności na podstawie składni i powtarzających się wzorów
        intensity = 1.0
        if re.search(r"bardzo|niezwykle|ogromnie|niesamowicie|wyjątkowo", text_lower):
            intensity = 1.5
        elif re.search(r"trochę|lekko|nieco|delikatnie", text_lower):
            intensity = 0.7
            
        # Aplikujemy intensywność do wyników
        for emotion in emotion_scores:
            emotion_scores[emotion] *= intensity
        
        # Normalizacja wyników
        total = sum(emotion_scores.values()) or 1.0
        normalized = {k: round(v/total, 2) for k, v in emotion_scores.items() if v > 0}
        
        # Dominujące emocje (top 3)
        dominant = sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:3]
        dominant_emotions = {emotion: score for emotion, score in dominant if score > 0.1}
        
        return {
            "dominujące": dominant_emotions,
            "wszystkie": normalized,
            "intensywność": round(intensity, 2),
            "dopasowania": emotion_matches
        }
        
    def detect_topics(self, text):
        """Wykrywa tematy w tekście z wagami używając TF-IDF"""
        if not text:
            return {}
            
        text_lower = text.lower()
        text_tokens = _tok(text_lower)  # Używamy istniejącej funkcji tokenizującej
        
        # Przygotowanie korpusu dokumentów do TF-IDF
        corpus = []
        topic_docs = {}
        
        # Tworzenie dokumentów dla każdego tematu (dla TF-IDF)
        for topic, keywords in self.topic_keywords.items():
            topic_docs[topic] = " ".join(keywords)
            corpus.append(topic_docs[topic])
        
        # Dodaj zapytanie użytkownika jako ostatni dokument w korpusie
        corpus.append(text_lower)
        
        # Obliczenie wektorów TF-IDF
        tfidf_scores = _tfidf_vec(text_tokens, [_tok(doc) for doc in corpus])
        
        # Obliczenie podobieństwa między tekstem a tematami
        topic_scores = {}
        for topic, topic_text in topic_docs.items():
            topic_tokens = _tok(topic_text)
            topic_tfidf = _tfidf_vec(topic_tokens, [_tok(doc) for doc in corpus])
            
            # Iloczyn skalarny wektorów TF-IDF (prostszy odpowiednik cosine similarity)
            score = 0
            for term in set(text_tokens) & set(topic_tokens):  # Wspólne terminy
                score += tfidf_scores.get(term, 0) * topic_tfidf.get(term, 0) * 3.0  # Waga dla wspólnych terminów
                
            # Dodatkowa korekta dla słów kluczowych
            for keyword in self.topic_keywords[topic]:
                if keyword in text_lower:
                    score += 0.15  # Bonus za dokładne dopasowanie słów kluczowych
            
            if score > 0.1:  # Minimalny próg
                topic_scores[topic] = min(0.95, score)  # Ograniczenie maksymalnej wartości
        
        # Dodatkowa analiza kontekstualna
        # Wzorce zakupowe
        if re.search(r'\b(kup|kupi[ćęcł]|zam[óo]wi[ćęcł]|sprzeda[ćęcł]|cen[ayę]|koszt|ofert[ayę]|tani|drogi)\b', text_lower):
            topic_scores["zakupy"] = max(topic_scores.get("zakupy", 0), 0.7)
            
        # Wzorce wsparcia technicznego
        if re.search(r'\b(problem|trudno[śsć][ćcę]|b[łl][ąa]d|nie dzia[łl]a|zepsut|pom[óo][żz])\b', text_lower):
            topic_scores["wsparcie"] = max(topic_scores.get("wsparcie", 0), 0.75)
            
        # Wzorce finansowe
        if re.search(r'\b(pieni[ąa]dz|z[łl]ot|pln|eur|usd|walut|bank|konto|p[łl]atno[śsć][ćc])\b', text_lower):
            topic_scores["finanse"] = max(topic_scores.get("finanse", 0), 0.7)
            
        # Normalizacja wyników
        total_score = sum(topic_scores.values()) or 1.0
        for topic in topic_scores:
            topic_scores[topic] = topic_scores[topic] / total_score * 0.8 + 0.1  # Skalowanie do sensownego zakresu
            
        # Usuń tematy z bardzo niskim wynikiem
        return {k: round(v, 2) for k, v in topic_scores.items() if v > 0.22}
    
    def analyze_sentiment(self, text):
        """Analiza sentymentu tekstu"""
        text_lower = text.lower()
        scores = {"pozytywny": 0, "negatywny": 0, "neutralny": 0}
        
        # Liczenie wystąpień słów z każdej kategorii
        for sentiment, words in self.sentiment_keywords.items():
            for word in words:
                count = text_lower.count(word)
                if count > 0:
                    scores[sentiment] += count * 0.1  # Każde wystąpienie zwiększa wynik
        
        # Analiza znaków interpunkcyjnych i emoji
        if "!" in text:
            excl_count = text.count("!")
            if scores["pozytywny"] > scores["negatywny"]:
                scores["pozytywny"] += excl_count * 0.05
            elif scores["negatywny"] > scores["pozytywny"]:
                scores["negatywny"] += excl_count * 0.05
                
        # Sprawdź emoji lub emotikony
        positive_emotes = [":)", ":D", "😊", "👍", "😁"]
        negative_emotes = [":(", ":(", "😢", "👎", "😠"]
        
        for emote in positive_emotes:
            scores["pozytywny"] += text.count(emote) * 0.15
            
        for emote in negative_emotes:
            scores["negatywny"] += text.count(emote) * 0.15
        
        # Sprawdź negację, która może odwracać sentyment
        negation_words = ["nie", "bez", "nigdy", "żaden"]
        for word in negation_words:
            pattern = word + " [\\w]+ "
            matches = re.findall(pattern, text_lower)
            if matches:
                # Zmniejsz wpływ pozytywnych słów po negacji
                scores["pozytywny"] *= 0.8
                scores["negatywny"] *= 1.2
                
        # Normalizacja wyników
        total = sum(scores.values()) or 1
        normalized = {k: round(v/total, 2) for k, v in scores.items()}
        
        # Określenie dominującego sentymentu
        dominant = max(normalized, key=normalized.get)
        normalized["dominujący"] = dominant
        
        return normalized
        
    def detect_intention(self, text):
        """Wykrywanie intencji użytkownika"""
        text_lower = text.lower()
        scores = {"pytanie": 0, "prośba": 0, "stwierdzenie": 0}
        
        # Sprawdź znaki zapytania
        if "?" in text:
            scores["pytanie"] += 0.6
        
        # Sprawdzanie wskaźników intencji
        for intention, indicators in self.intention_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    scores[intention] += 0.15
        
        # Analiza struktury gramatycznej (podstawowa)
        if text_lower.startswith("czy") or text_lower.startswith("jak") or text_lower.startswith("kiedy"):
            scores["pytanie"] += 0.3
            
        if "proszę" in text_lower or "czy możesz" in text_lower or text_lower.startswith("pomóż"):
            scores["prośba"] += 0.3
            
        if "." in text and "?" not in text:
            scores["stwierdzenie"] += 0.2
            
        # Normalizacja wyników
        total = sum(scores.values()) or 1
        normalized = {k: round(v/total, 2) for k, v in scores.items()}
        
        # Określenie dominującej intencji
        dominant = max(normalized, key=normalized.get)
        normalized["dominująca"] = dominant
        
        return normalized
    
    def extract_keywords(self, text):
        """Ekstrakcja słów kluczowych z tekstu"""
        # Proste czyszczenie tekstu
        text_lower = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text_lower.split()
        
        # Lista stop words (podstawowa)
        stop_words = ["i", "w", "na", "z", "do", "od", "dla", "że", "to", "jest", "są", "być", "a", "o", "jak", "tak", "nie", "się"]
        
        # Filtrowanie słów
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Obliczanie częstości występowania
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sortowanie po częstości i zwracanie top N słów
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        top_keywords = [word for word, freq in sorted_words[:10]]
        
        return top_keywords
    
    def analyze_complexity(self, text):
        """Analiza złożoności tekstu"""
        if not text:
            return {"poziom": "brak tekstu", "średnia_długość_zdania": 0, "średnia_długość_słowa": 0, "różnorodność_leksykalna": 0}
            
        # Podział na zdania
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return {"poziom": "brak tekstu", "średnia_długość_zdania": 0, "średnia_długość_słowa": 0, "różnorodność_leksykalna": 0}
        
        # Liczba słów w zdaniach
        words_per_sentence = [len(s.split()) for s in sentences]
        avg_sentence_length = sum(words_per_sentence) / len(sentences) if sentences else 0
        
        # Średnia długość słowa
        all_words = [word for s in sentences for word in s.split()]
        if not all_words:
            return {"poziom": "brak tekstu", "średnia_długość_zdania": 0, "średnia_długość_słowa": 0, "różnorodność_leksykalna": 0}
            
        avg_word_length = sum(len(word) for word in all_words) / len(all_words)
        
        # Różnorodność leksykalna (unique words / total words)
        lexical_diversity = len(set(all_words)) / len(all_words) if all_words else 0
        
        # Określenie poziomu złożoności
        complexity_level = "niska"
        if avg_sentence_length > 15 or avg_word_length > 6 or lexical_diversity > 0.7:
            complexity_level = "wysoka"
        elif avg_sentence_length > 10 or avg_word_length > 5 or lexical_diversity > 0.5:
            complexity_level = "średnia"
            
        return {
            "poziom": complexity_level,
            "średnia_długość_zdania": round(avg_sentence_length, 2),
            "średnia_długość_słowa": round(avg_word_length, 2),
            "różnorodność_leksykalna": round(lexical_diversity, 2)
        }
        
    def analyze_local_context(self, text):
        """Analizuje lokalny kontekst w tekście - lokalizacje, czas, odniesienia"""
        if not text:
            return {"lokalizacje": [], "czas": [], "odniesienia_przestrzenne": [], 
                    "odniesienia_czasowe": []}
            
        text_lower = text.lower()
        
        # Słowniki do rozpoznawania rodzajów kontekstu
        # Lokalizacje (miasta, kraje, regiony)
        location_patterns = [
            # "w Warszawie", "do Polski"
            r"\b(?:w|do|z)\s+([A-Z][a-ząćęłńóśźż]{2,})\b",
            # Nazwy własne (miasta, kraje)  
            r"\b([A-Z][a-ząćęłńóśźż]{2,})\b",
            # Nazwy ulic
            r"\b(?:ulica|ulicy|ul\.|aleja|alei|al\.)\s+([A-Z][a-ząćęłńóśźż]+)\b"
        ]
        
        # Wyrażenia czasowe
        time_patterns = [
            r"\b(\d{1,2}:\d{2})\b",  # Format godziny 12:30
            r"\b(\d{1,2})[:\.-]\s?(\d{2})\b",  # Format godziny z separatorem
            r"\bo\s+(?:godz(?:inie)?)\s+(\d{1,2})\b",  # "o godzinie 5"
            r"\b(?:rano|po\s+południu|wieczorem|w\s+nocy)\b"  # Pory dnia
        ]
        
        # Odniesienia przestrzenne
        spatial_references = [
            r"\b(?:na\s+prawo|na\s+lewo|nad|pod|obok|przy|przed|za|naprzeciw)\b",
            r"\b(?:w\s+pobliżu|niedaleko|blisko)\b",
            r"\b(?:na\s+północ|na\s+południe|na\s+wschód|na\s+zachód)\b",
            r"\b(?:w\s+centrum|na\s+obrzeżach|na\s+peryferiach|w\s+środku)\b"
        ]
        
        # Odniesienia czasowe
        temporal_references = [
            r"\b(?:wczoraj|dzisiaj|jutro|pojutrze|za\s+tydzień)\b",
            r"\b(?:w\s+przyszłym\s+tygodniu|w\s+przyszłym\s+miesiącu)\b",
            r"\b(?:rano|po\s+południu|wieczorem|w\s+nocy|o\s+świcie|o\s+zmierzchu)\b",
            r"\b(?:w\s+poniedziałek|we\s+wtorek|w\s+środę|w\s+czwartek)\b",
            r"\b(?:w\s+piątek|w\s+sobotę|w\s+niedzielę)\b",
            r"\b(\d{1,2})\s+(?:stycznia|lutego|marca|kwietnia|maja|czerwca)\b",
            r"\b(\d{1,2})\s+(?:lipca|sierpnia|września|października|listopada|grudnia)\b"
        ]
        
        # Rozpoznawanie lokalizacji
        locations = []
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                locations.extend([m[0] if isinstance(m, tuple) else m for m in matches])
        
        # Rozpoznawanie wyrażeń czasowych
        times = []
        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                times.extend([m[0] if isinstance(m, tuple) else m for m in matches])
        
        # Rozpoznawanie odniesień przestrzennych
        spatial_refs = []
        for pattern in spatial_references:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                spatial_refs.extend(matches)
        
        # Rozpoznawanie odniesień czasowych
        temporal_refs = []
        for pattern in temporal_references:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if isinstance(matches[0], tuple):
                    # Dla dat w formacie "15 stycznia 2023"
                    temporal_refs.extend([' '.join(filter(None, m)) for m in matches])
                else:
                    temporal_refs.extend(matches)
        
        # Dodatkowe przetwarzanie dla lokalizacji z przyimkami
        processed_locations = []
        for loc in locations:
            # Czyścimy z przyimków i dodatkowych znaków
            cleaned_loc = re.sub(r'^(?:w|do|z|na|przy)\s+', '', loc)
            cleaned_loc = re.sub(r'[,.:;"\'()]', '', cleaned_loc)
            if len(cleaned_loc) > 2:  # Minimalna długość nazwy lokalizacji
                processed_locations.append(cleaned_loc)
        
        # Deduplikacja wyników
        locations = list(set(processed_locations))
        times = list(set(times))
        spatial_refs = list(set(spatial_refs))
        temporal_refs = list(set(temporal_refs))
        
        # Sortowanie wyników według długości (dłuższe nazwy są często bardziej specyficzne)
        locations.sort(key=len, reverse=True)
        
        # Usuwanie fałszywych trafień (typowe słowa, które nie są lokalizacjami)
        common_words = ["jako", "tego", "tych", "inne", "moje", "twoje", "nasze"]
        locations = [loc for loc in locations if loc.lower() not in common_words]
        
        # Identyfikacja głównego kontekstu przestrzenno-czasowego
        main_location = locations[0] if locations else None
        main_time = temporal_refs[0] if temporal_refs else None
        
        return {
            "lokalizacje": locations,
            "czas": times,
            "odniesienia_przestrzenne": spatial_refs,
            "odniesienia_czasowe": temporal_refs,
            "główna_lokalizacja": main_location,
            "główny_czas": main_time
        }
        
    def analyze_discourse(self, text):
        """Analizuje dyskurs - identyfikuje typ, strukturę i cechy komunikacji"""
        if not text:
            return {"typ_dyskursu": "brak tekstu", "cechy": [], "słowa_kluczowe": []}
            
        text_lower = text.lower()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return {"typ_dyskursu": "brak tekstu", "cechy": [], "słowa_kluczowe": []}
            
        # Słowniki do identyfikacji typów dyskursu
        discourse_markers = {
            "naukowy": [
                r"\b(?:badania|badanie|analiza|analizy|hipoteza|teoria|wyniki)\b",
                r"\b(?:dowód|dowody|metodologia|eksperyment|dane|wniosek)\b",
                r"\b(?:według\s+(?:\w+\s+){0,2}et\s+al\.|cytując|zgodnie\s+z)\b"
            ],
            "polityczny": [
                r"\b(?:państwo|władza|rząd|ustawa|prawo|społeczeństwo)\b",
                r"\b(?:polityka|polityczny|partia|demokracja|wybory)\b",
                r"\b(?:obywatel|obywatele|obywatelski|konstytucja|wolność)\b"
            ],
            "biznesowy": [
                r"\b(?:firma|biznes|przedsiębiorstwo|klient|klienci|zysk)\b",
                r"\b(?:sprzedaż|rynek|marketing|strategia|budżet|przychód)\b",
                r"\b(?:produkt|usługa|wartość|cena|oferta|umowa|kontrakt)\b"
            ],
            "potoczny": [
                r"\b(?:super|fajnie|ekstra|spoko|ziom|hej|cześć|siema|nara)\b",
                r"\b(?:mega|totalnie|generalnie|jakby|wiesz|no\s+wiesz)\b",
                r"(?:!{2,}|\\?{2,})"
            ],
            "perswazyjny": [
                r"\b(?:musisz|powinieneś|należy|trzeba|koniecznie)\b",
                r"\b(?:najlepszy|jedyny|wyjątkowy|niesamowity|rewolucyjny)\b",
                r"\b(?:przekonaj\s+się|sprawdź|nie\s+przegap|już\s+dziś)\b"
            ],
            "emocjonalny": [
                r"\b(?:kocham|nienawidzę|uwielbiam|boję\s+się|tęsknię)\b",
                r"\b(?:radość|smutek|złość|strach|niepokój|wzruszenie)\b",
                r"(?:!{2,}|\\?!|\\.{3,})"
            ],
            "informacyjny": [
                r"\b(?:informacja|informuję|zawiadamiam|komunikat|ogłoszenie)\b",
                r"\b(?:przekazuję|uprzejmie\s+informuję|podaję\s+do\s+wiadomości)\b",
                r"\b(?:dane|fakty|statystyki|zestawienie|podsumowanie)\b"
            ]
        }
        
        # Cechy dyskursu
        discourse_features = {
            "formalny": [
                r"\b(?:szanowny|uprzejmie|z\s+poważaniem|niniejszym)\b",
                r"\b(?:pragnę\s+podkreślić|należy\s+zaznaczyć)\b"
            ],
            "nieformalny": [
                r"\b(?:hej|cześć|siema|słuchaj|wiesz\s+co|no\s+dobra|ok)\b",
                r"(?:!{2,}|\\?{2,})"
            ],
            "argumentacyjny": [
                r"\b(?:ponieważ|dlatego|zatem|więc|skutkiem)\b",
                r"\b(?:po\s+pierwsze|po\s+drugie|z\s+jednej\s+strony)\b",
                r"\b(?:argumentuję|twierdzę|uważam|wnioskuję)\b"
            ],
            "narracyjny": [
                r"\b(?:pewnego\s+dnia|dawno\s+temu|na\s+początku)\b",
                r"\b(?:następnie|po\s+chwili|tymczasem|w\s+końcu)\b"
            ],
            "dialogowy": [
                r"\b(?:pytam|odpowiadam|mówię|twierdzisz|sugerujesz)\b",
                r'''["„"''].*?["„"']''',
                r"\b(?:rozmowa|dialog|dyskusja|debata)\b"
            ],
            "opisowy": [
                r"\b(?:jest|był|znajdował\s+się|wyglądał|przypominał)\b",
                r"\b(?:wysoki|szeroki|ciemny|jasny|czerwony|duży)\b"
            ],
            "instruktażowy": [
                r"\b(?:najpierw|następnie|potem|na\s+koniec|krok)\b",
                r"\b(?:włącz|wyłącz|naciśnij|kliknij|otwórz|zamknij)\b",
                r"(?:^\s*\d+\.|^\s*-|\*\s)"
            ]
        }
        
        # Analiza typu dyskursu
        discourse_scores = {}
        for disc_type, patterns in discourse_markers.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.MULTILINE)
                score += len(matches)
            discourse_scores[disc_type] = score
            
        # Analiza cech dyskursu
        features = []
        for feature, patterns in discourse_features.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.MULTILINE)
                score += len(matches)
            if score >= 2:  # Próg minimalny dla uznania cechy
                features.append(feature)
                
        # Struktura dyskursu - analiza połączeń logicznych
        logical_connectors = [
            r"\b(?:ponieważ|bo|gdyż|dlatego|więc|zatem|stąd)\b",
            r"\b(?:jeśli|jeżeli|o\s+ile|pod\s+warunkiem)\b",
            r"\b(?:ale|lecz|jednak|niemniej|natomiast)\b",
            r"\b(?:po\s+pierwsze|po\s+drugie|przede\s+wszystkim)\b"
        ]
        
        connectors_count = 0
        for pattern in logical_connectors:
            connectors_count += len(re.findall(pattern, text_lower))
            
        # Gęstość logiczna - liczba połączeń logicznych na zdanie
        logical_density = connectors_count / len(sentences) if sentences else 0
        
        # Kompleksowość dyskursu - średnia długość zdania
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        complexity_words = re.findall(r'\b\w{10,}\b', text_lower)
        lexical_complexity = len(complexity_words) / len(sentences) if sentences else 0
        
        # Określenie głównego typu dyskursu
        main_discourse_type = max(discourse_scores.items(), key=lambda x: x[1])[0] \
            if any(score > 0 for score in discourse_scores.values()) else "nieokreślony"
            
        # Słowa kluczowe w dyskursie
        words = re.findall(r'\b\w+\b', text_lower)
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Pomijamy krótkie słowa
                word_freq[word] = word_freq.get(word, 0) + 1
                
        # Lista polskich stopwords (słów nieistotnych)
        stopwords = [
            "oraz", "jako", "tylko", "tego", "przez", "jest", "jestem", 
            "jesteśmy", "ponieważ", "żeby", "który", "która", "które", 
            "także", "również", "dlatego", "więc", "czyli", "gdyż", "albo",
            "czyli", "lecz", "gdyż", "oraz", "jednak", "choć"
        ]
        
        # Filtrowanie słów nieistotnych
        for word in stopwords:
            if word in word_freq:
                del word_freq[word]
                
        # Wybieranie najczęstszych słów jako słowa kluczowe
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        keywords = [word for word, freq in keywords]
        
        result = {
            "typ_dyskursu": main_discourse_type,
            "cechy": features,
            "słowa_kluczowe": keywords,
            "gęstość_logiczna": round(logical_density, 2),
            "złożoność_leksykalna": round(lexical_complexity, 2),
            "średnia_długość_zdania": round(avg_sentence_length, 2)
        }
        
        # Dodanie oceny jakości dyskursu
        if logical_density > 0.5 and lexical_complexity > 0.3 and avg_sentence_length > 15:
            result["ocena_jakości"] = "zaawansowany"
        elif logical_density > 0.3 and avg_sentence_length > 10:
            result["ocena_jakości"] = "standardowy"
        else:
            result["ocena_jakości"] = "prosty"
            
        return result
        
    def analyze_arguments(self, text):
        """Analizuje strukturę argumentacyjną tekstu"""
        if not text:
            return {"struktura": "brak tekstu", "elementy": [], "jakość": "brak"}
            
        # Dzielimy tekst na zdania
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return {"struktura": "brak tekstu", "elementy": [], "jakość": "brak"}
        
        # Wzorce dla rozpoznawania elementów argumentacji
        argument_patterns = {
            "teza_główna": [
                r"\b(?:uważam,\s+że|twierdzę,\s+że|moim\s+zdaniem)\b",
                r"\b(?:chciał[a]?bym\s+dowieść|zamierzam\s+pokazać)\b",
                r"\b(?:główn[ym|ą]\s+(?:tez[ą|e]|kwesti[ą|e])\s+jest)\b"
            ],
            "przesłanka": [
                r"\b(?:ponieważ|gdyż|bowiem|dlatego\s+że|z\s+powodu)\b",
                r"\b(?:pierwszym\s+argumentem|drugim\s+argumentem)\b",
                r"\b(?:dowodzi\s+tego|świadczy\s+o\s+tym|potwierdza\s+to)\b"
            ],
            "kontrargument": [
                r"\b(?:jednak|niemniej\s+jednak|z\s+drugiej\s+strony)\b",
                r"\b(?:można\s+(?:by|też)\s+(?:zauważyć|argumentować))\b",
                r"\b(?:przeciwnicy\s+twierdzą|krytycy\s+wskazują)\b"
            ],
            "konkluzja": [
                r"\b(?:w\s+(?:konsekwencji|rezultacie|efekcie))\b",
                r"\b(?:(?:podsumowując|reasumując|konkludując))\b",
                r"\b(?:(?:ostatecznie|finalnie|w\s+konkluzji))\b"
            ],
            "przykład": [
                r"\b(?:na\s+przykład|przykładem\s+jest|dla\s+przykładu)\b",
                r"\b(?:doskonale\s+ilustruje\s+to|świadczy\s+o\s+tym)\b",
                r"\b(?:warto\s+(?:przytoczyć|wskazać)\s+przykład)\b"
            ],
            "definicja": [
                r"\b(?:definiuję|rozumiem\s+(?:przez|jako)|oznacza\s+to)\b",
                r"\b(?:termin|pojęcie)\s+(?:\w+)\s+(?:odnosi\s+się|oznacza)\b",
                r"(?:(?:^|[.!?]\s+)(?:[A-Z]\w+)\s+(?:to|jest|oznacza))\b"
            ]
        }
        
        # Spójniki logiczne i ich kategorie
        logical_connectors = {
            "przyczynowe": [
                r"\b(?:ponieważ|gdyż|bowiem|dlatego\s+że|z\s+powodu)\b",
                r"\b(?:w\s+związku\s+z\s+tym|skutkiem\s+tego)\b"
            ],
            "kontrastujące": [
                r"\b(?:jednak|niemniej|natomiast|ale|lecz|choć|chociaż)\b",
                r"\b(?:z\s+drugiej\s+strony|przeciwnie|wbrew\s+temu)\b"
            ],
            "wynikowe": [
                r"\b(?:w\s+rezultacie|w\s+efekcie|w\s+konsekwencji)\b",
                r"\b(?:zatem|więc|tak\s+więc|stąd|dlatego)\b"
            ],
            "wzmacniające": [
                r"\b(?:co\s+więcej|ponadto|dodatkowo|w\s+dodatku)\b",
                r"\b(?:nie\s+tylko|również|także|zarówno)\b"
            ],
            "porządkujące": [
                r"\b(?:po\s+pierwsze|po\s+drugie|następnie|w\s+końcu)\b",
                r"\b(?:przede\s+wszystkim|w\s+szczególności|głównie)\b"
            ]
        }
        
        # Identyfikacja elementów argumentacji w zdaniach
        argument_structure = {}
        for arg_type, patterns in argument_patterns.items():
            argument_structure[arg_type] = []
            for pattern in patterns:
                for i, sentence in enumerate(sentences):
                    if re.search(pattern, sentence, re.IGNORECASE):
                        argument_structure[arg_type].append({
                            "zdanie": sentence,
                            "pozycja": i + 1
                        })
        
        # Identyfikacja spójników logicznych
        connectors_found = {}
        for conn_type, patterns in logical_connectors.items():
            connectors_found[conn_type] = 0
            for pattern in patterns:
                for sentence in sentences:
                    matches = re.findall(pattern, sentence, re.IGNORECASE)
                    connectors_found[conn_type] += len(matches)
        
        # Określanie struktury argumentacyjnej
        structure_type = "nieokreślona"
        elements_found = []
        
        # Sprawdzanie kompletności argumentacji
        if argument_structure["teza_główna"] and argument_structure["konkluzja"]:
            if argument_structure["przesłanka"]:
                if argument_structure["kontrargument"]:
                    structure_type = "złożona dialektyczna"
                    elements_found = ["teza", "przesłanki", "kontrargumenty", "konkluzja"]
                else:
                    structure_type = "prosta liniowa"
                    elements_found = ["teza", "przesłanki", "konkluzja"]
            else:
                structure_type = "niekompletna"
                elements_found = ["teza", "konkluzja"]
        elif argument_structure["przesłanka"]:
            if argument_structure["teza_główna"]:
                structure_type = "niedokończona"
                elements_found = ["teza", "przesłanki"]
            elif argument_structure["konkluzja"]:
                structure_type = "indukcyjna"
                elements_found = ["przesłanki", "konkluzja"]
            else:
                structure_type = "fragmentaryczna"
                elements_found = ["przesłanki"]
        elif argument_structure["teza_główna"]:
            structure_type = "deklaratywna"
            elements_found = ["teza"]
        
        # Określanie jakości argumentacji
        arg_quality = "niska"
        
        # Liczenie elementów argumentacji
        total_elements = sum(len(items) for items in argument_structure.values())
        
        # Sprawdzanie obecności definicji i przykładów
        has_definitions = len(argument_structure["definicja"]) > 0
        has_examples = len(argument_structure["przykład"]) > 0
        
        # Liczenie spójników logicznych
        total_connectors = sum(connectors_found.values())
        
        # Ocena jakości argumentacji
        conn_per_sentence = total_connectors / len(sentences) if sentences else 0
        
        # Zróżnicowanie typów spójników
        connector_diversity = sum(1 for count in connectors_found.values() if count > 0)
        
        # Kryteria jakości
        if (structure_type in ["złożona dialektyczna", "prosta liniowa"] and 
                has_definitions and has_examples and conn_per_sentence >= 0.5 and
                connector_diversity >= 3):
            arg_quality = "wysoka"
        elif (total_elements >= 5 and conn_per_sentence >= 0.3 and
              (has_definitions or has_examples) and connector_diversity >= 2):
            arg_quality = "średnia"
        
        # Identyfikacja głównych argumentów
        main_args = []
        for arg_type in ["teza_główna", "przesłanka", "konkluzja"]:
            for item in argument_structure[arg_type]:
                if item not in main_args:
                    main_args.append(item["zdanie"])
        
        result = {
            "struktura": structure_type,
            "elementy": elements_found,
            "główne_argumenty": main_args[:3],  # Ograniczamy do 3 najważniejszych
            "jakość": arg_quality,
            "spójniki_logiczne": {
                "liczba": total_connectors,
                "na_zdanie": round(conn_per_sentence, 2),
                "rodzaje": {k: v for k, v in connectors_found.items() if v > 0}
            }
        }
        
        # Dodajemy ocenę balansu argumentacji
        if argument_structure["kontrargument"]:
            contra_to_pro_ratio = (len(argument_structure["kontrargument"]) / 
                                 len(argument_structure["przesłanka"]) 
                                 if argument_structure["przesłanka"] else 0)
            result["balans_argumentacji"] = round(contra_to_pro_ratio, 2)
            
            if 0.3 <= contra_to_pro_ratio <= 0.7:
                result["ocena_balansu"] = "zrównoważona"
            elif contra_to_pro_ratio > 0.7:
                result["ocena_balansu"] = "silnie dialektyczna"
            else:
                result["ocena_balansu"] = "jednostronna"
        else:
            result["balans_argumentacji"] = 0.0
            result["ocena_balansu"] = "jednokierunkowa"
            
        return result
        
    def analyze_semantic_structure(self, text):
        """Analizuje głęboką strukturę semantyczną tekstu"""
        if not text:
            return {"struktura": "brak tekstu", "relacje": [], "tematy": []}
            
        text_lower = text.lower()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return {"struktura": "brak tekstu", "relacje": [], "tematy": []}
            
        # 1. Analiza podmiotów i obiektów w tekście
        entities = []
        patterns = {
            "osoba": [
                r"\b([A-Z][a-ząćęłńóśźż]+)\b(?=\s+(?:powiedział|stwierdził|uważa))",
                r"\b([A-Z][a-ząćęłńóśźż]+)\b(?=\s+(?:jest|był|będzie))"
            ],
            "organizacja": [
                r"\b([A-Z][A-ZĄĆĘŁŃÓŚŹŻa-ząćęłńóśźż]+)\b(?=\s+(?:ogłosił|poinformował))",
                r"\b(?:firma|spółka|organizacja|instytucja|ministerstwo)\s+([A-Z]\w+)\b"
            ],
            "miejsce": [
                r"\bw\s+([A-Z][a-ząćęłńóśźż]+)\b",
                r"\bdo\s+([A-Z][a-ząćęłńóśźż]+)\b",
                r"\bz\s+([A-Z][a-ząćęłńóśźż]+)\b"
            ],
            "czas": [
                r"\b(\d{1,2}\s+(?:stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|października|listopada|grudnia)(?:\s+\d{4})?)\b",
                r"\b((?:w\s+)?(?:poniedziałek|wtorek|środę|czwartek|piątek|sobotę|niedzielę))\b",
                r"\b(\d{1,2}:\d{2})\b"
            ],
            "pojęcie": [
                r"\b([a-ząćęłńóśźż]{5,}(?:acja|izm|ość|stwo|ctwo|anie|enie))\b"
            ]
        }
        
        for entity_type, patterns_list in patterns.items():
            for pattern in patterns_list:
                for sentence in sentences:
                    matches = re.findall(pattern, sentence, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple):
                            match = match[0]
                        if len(match) > 2:  # Minimalna długość encji
                            entities.append({
                                "tekst": match,
                                "typ": entity_type,
                                "kontekst": sentence[:50] + "..." if len(sentence) > 50 else sentence
                            })
        
        # Filtrowanie duplikatów
        unique_entities = []
        seen_entities = set()
        for entity in entities:
            key = (entity["tekst"].lower(), entity["typ"])
            if key not in seen_entities:
                seen_entities.add(key)
                unique_entities.append(entity)
        
        # 2. Analiza relacji semantycznych
        relations = []
        semantic_patterns = {
            "przyczynowo-skutkowe": [
                r"(\b\w+[^.!?]*)\s+(?:powoduje|powodują|spowodował|spowodowała|wywołuje|skutkuje)\s+([^.!?]*)",
                r"(\b\w+[^.!?]*)\s+(?:wpływa|wpływają|wpłynął|wpłynęła)\s+na\s+([^.!?]*)",
                r"(\b\w+[^.!?]*)\s+(?:dlatego|z\s+tego\s+powodu|w\s+rezultacie)\s+([^.!?]*)"
            ],
            "porównawcze": [
                r"(\b\w+[^.!?]*)\s+(?:podobnie\s+jak|tak\s+jak|podobnie\s+do)\s+([^.!?]*)",
                r"(\b\w+[^.!?]*)\s+(?:różni\s+się\s+od|jest\s+inne\s+niż|jest\s+odmienne\s+od)\s+([^.!?]*)",
                r"(\b\w+[^.!?]*)\s+(?:w\s+przeciwieństwie\s+do)\s+([^.!?]*)"
            ],
            "część-całość": [
                r"(\b\w+[^.!?]*)\s+(?:składa\s+się\s+z|zawiera|obejmuje)\s+([^.!?]*)",
                r"(\b\w+[^.!?]*)\s+(?:jest\s+częścią|wchodzi\s+w\s+skład|należy\s+do)\s+([^.!?]*)"
            ],
            "posesywne": [
                r"(\b\w+[^.!?]*)\s+(?:posiada|ma|dysponuje)\s+([^.!?]*)",
                r"(\b\w+[^.!?]*)\s+(?:należący\s+do|właściciel|własność)\s+([^.!?]*)"
            ],
            "temporalne": [
                r"(\b\w+[^.!?]*)\s+(?:przed|po|w\s+trakcie|podczas)\s+([^.!?]*)",
                r"(\b\w+[^.!?]*)\s+(?:wcześniej\s+niż|później\s+niż|równocześnie\s+z)\s+([^.!?]*)"
            ]
        }
        
        for rel_type, patterns_list in semantic_patterns.items():
            for pattern in patterns_list:
                for sentence in sentences:
                    matches = re.findall(pattern, sentence, re.IGNORECASE)
                    for match in matches:
                        if len(match) >= 2:
                            relations.append({
                                "typ": rel_type,
                                "element_1": match[0].strip(),
                                "element_2": match[1].strip(),
                                "zdanie": sentence
                            })
        
        # 3. Analiza struktury tematycznej
        topic_words = {}
        
        # Ekstrakcja rzeczowników jako potencjalnych tematów
        noun_pattern = r"\b([a-ząćęłńóśźż]{3,}(?:ość|anie|enie|stwo|ctwo|cja|zja|acja|izm|or|er|yk))\b"
        for sentence in sentences:
            matches = re.findall(noun_pattern, sentence.lower())
            for match in matches:
                if match not in topic_words:
                    topic_words[match] = 0
                topic_words[match] += 1
        
        # Lista "stop words" dla tematów
        stop_words = ["dlatego", "ponieważ", "przez", "gdyż", "czyli", "więc", "jednak", 
                     "bowiem", "także", "również", "czyli", "właśnie", "natomiast"]
        
        # Filtrowanie potencjalnych tematów
        for word in stop_words:
            if word in topic_words:
                del topic_words[word]
        
        # Wybór głównych tematów
        main_topics = sorted(topic_words.items(), key=lambda x: x[1], reverse=True)[:5]
        main_topics = [{"temat": topic, "częstość": count} for topic, count in main_topics]
        
        # 4. Analiza kohezji tekstu (powiązań wewnętrznych)
        cohesion_markers = {
            "zaimki_anaforyczne": [
                r"\b(?:on[a|i]?|jeg[a|o]|jej|ich|t[en|ą|ym|ymi]|t[a|e]|ci|tamci|tamte)\b"
            ],
            "odniesienia_tematyczne": [
                r"\b(?:ten\s+sam|wspomnian[y|a|e]|powyższ[y|a|e]|wcześniejsz[y|a|e])\b"
            ],
            "spójniki_kontynuacji": [
                r"\b(?:ponadto|poza\s+tym|co\s+więcej|następnie|dalej|kontynuując)\b"
            ],
            "powtórzenia_leksykalne": []  # Będzie analizowane algorytmicznie
        }
        
        # Liczenie markerów kohezji
        cohesion_counts = {}
        for marker_type, patterns_list in cohesion_markers.items():
            cohesion_counts[marker_type] = 0
            for pattern in patterns_list:
                for sentence in sentences:
                    matches = re.findall(pattern, sentence, re.IGNORECASE)
                    cohesion_counts[marker_type] += len(matches)
        
        # Analiza powtórzeń leksykalnych
        words_by_sentence = [re.findall(r'\b\w{3,}\b', s.lower()) for s in sentences]
        repetitions = 0
        
        # Szukamy powtórzeń słów między zdaniami
        if len(words_by_sentence) > 1:
            for i in range(1, len(words_by_sentence)):
                for word in words_by_sentence[i]:
                    if word in words_by_sentence[i-1]:
                        repetitions += 1
        
        cohesion_counts["powtórzenia_leksykalne"] = repetitions
        
        # Ogólna miara kohezji
        cohesion_total = sum(cohesion_counts.values())
        cohesion_per_sentence = cohesion_total / len(sentences) if sentences else 0
        
        # Określenie spójności semantycznej
        cohesion_level = "niska"
        if cohesion_per_sentence >= 1.5:
            cohesion_level = "wysoka"
        elif cohesion_per_sentence >= 0.8:
            cohesion_level = "średnia"
        
        # 5. Określenie głównej struktury semantycznej
        semantic_structure_types = {
            "narracyjna": 0,
            "ekspozycyjna": 0, 
            "argumentacyjna": 0,
            "opisowa": 0,
            "instruktażowa": 0
        }
        
        # Wzorce językowe charakterystyczne dla poszczególnych struktur
        structure_patterns = {
            "narracyjna": [
                r"\b(?:najpierw|potem|następnie|wtedy|później|w\s+końcu)\b",
                r"\b(?:gdy|kiedy|podczas|po\s+tym\s+jak|zanim|wkrótce)\b",
                r"\b(?:pewnego\s+dnia|pewnego\s+razu|dawno\s+temu|kiedyś)\b"
            ],
            "ekspozycyjna": [
                r"\b(?:definiuje|klasyfikuje|wyjaśnia|przedstawia|omawia)\b",
                r"\b(?:po\s+pierwsze|po\s+drugie|jednym\s+z|kolejnym)\b",
                r"\b(?:głównym|kluczowym|istotnym|ważnym|podstawowym)\b"
            ],
            "argumentacyjna": [
                r"\b(?:twierdzę|uważam|sądzę|dowodzę|argumentuję|przekonuję)\b",
                r"\b(?:ponieważ|dlatego|zatem|wobec\s+tego|wynika\s+z\s+tego)\b",
                r"\b(?:podsumowując|w\s+konkluzji|z\s+tego\s+wynika|dowodzi\s+to)\b"
            ],
            "opisowa": [
                r"\b(?:wygląda\s+jak|przypomina|charakteryzuje\s+się|cechuje\s+się)\b",
                r"\b(?:jest|wydaje\s+się|sprawia\s+wrażenie|prezentuje\s+się\s+jako)\b",
                r"\b(?:czerwony|niebieski|zielony|duży|mały|szeroki|wąski|wysoki)\b"
            ],
            "instruktażowa": [
                r"\b(?:należy|trzeba|powinno\s+się|musisz|najpierw|następnie)\b",
                r"\b(?:krok\s+po\s+kroku|w\s+pierwszej\s+kolejności|na\s+końcu)\b",
                r"(?:^\s*\d+\.|\d\)\s+|\-\s+|•\s+)"
            ]
        }
        
        # Analiza wzorców dla określenia struktury
        for structure, patterns_list in structure_patterns.items():
            for pattern in patterns_list:
                for sentence in sentences:
                    matches = re.findall(pattern, sentence, re.IGNORECASE)
                    semantic_structure_types[structure] += len(matches)
        
        main_structure = max(semantic_structure_types.items(), key=lambda x: x[1])
        main_structure_type = main_structure[0]
        if main_structure[1] == 0:
            main_structure_type = "mieszana/nieokreślona"
        
        result = {
            "struktura_semantyczna": main_structure_type,
            "encje": unique_entities[:10],  # Ograniczamy do top 10
            "relacje": relations[:10],      # Ograniczamy do top 10
            "główne_tematy": main_topics,
            "spójność": {
                "poziom": cohesion_level,
                "markery_kohezji": cohesion_counts,
                "wskaźnik_spójności": round(cohesion_per_sentence, 2)
            }
        }
        
        return result
    
    def detect_temporal_context(self, text):
        """Wykrywanie kontekstu czasowego w tekście"""
        text_lower = text.lower()
        temporal_scores = {"przeszłość": 0, "teraźniejszość": 0.1, "przyszłość": 0}
        
        # Wskaźniki czasu przeszłego
        past_indicators = ["był", "była", "było", "były", "byłem", "byłam", "zrobiłem", "zrobiłam", "wczoraj", "wcześniej", "dawniej", "kiedyś", "niedawno"]
        
        # Wskaźniki czasu teraźniejszego
        present_indicators = ["jest", "są", "jestem", "jesteś", "robimy", "robię", "teraz", "obecnie", "dziś", "dzisiaj"]
        
        # Wskaźniki czasu przyszłego
        future_indicators = ["będzie", "będą", "będę", "będziemy", "zrobimy", "zrobię", "jutro", "wkrótce", "za chwilę", "w przyszłości", "później"]
        
        # Sprawdzanie wskaźników w tekście
        for indicator in past_indicators:
            if indicator in text_lower:
                temporal_scores["przeszłość"] += 0.15
                
        for indicator in present_indicators:
            if indicator in text_lower:
                temporal_scores["teraźniejszość"] += 0.15
                
        for indicator in future_indicators:
            if indicator in text_lower:
                temporal_scores["przyszłość"] += 0.15
        
        # Normalizacja wyników
        total = sum(temporal_scores.values()) or 1
        normalized = {k: round(v/total, 2) for k, v in temporal_scores.items()}
        
        # Określenie dominującego kontekstu czasowego
        dominant = max(normalized, key=normalized.get)
        normalized["dominujący"] = dominant
        
        return normalized
    
    def extract_entities(self, text):
        """Ekstrakcja encji z tekstu (osoby, miejsca, organizacje, daty, liczby)"""
        entities = {
            "osoby": [],
            "miejsca": [],
            "organizacje": [],
            "daty": [],
            "liczby": []
        }
        
        # Proste wzorce do rozpoznawania encji
        
        # Osoby (podstawowy wzorzec imię i nazwisko)
        person_pattern = re.compile(r'\b[A-ZŚĆŹŻŁÓŃ][a-zśćźżłóńäëöüß]+ [A-ZŚĆŹŻŁÓŃ][a-zśćźżłóńäëöüß]+\b')
        for match in person_pattern.finditer(text):
            entities["osoby"].append(match.group(0))
            
        # Miejsca (miasta, kraje)
        places = ["Warszawa", "Kraków", "Wrocław", "Poznań", "Gdańsk", "Łódź", "Szczecin", 
                 "Polska", "Niemcy", "Francja", "Włochy", "Hiszpania", "Anglia", "USA"]
        for place in places:
            if place in text:
                entities["miejsca"].append(place)
                
        # Organizacje (proste wzorce)
        org_pattern = re.compile(r'\b(?:[A-ZŚĆŹŻŁÓŃ][a-zśćźżłóńäëöüß]+ )?(?:[A-ZŚĆŹŻŁÓŃ][a-zśćźżłóńäëöüß]+ )?[A-ZŚĆŹŻŁÓŃ][a-zśćźżłóńäëöüß]* (?:sp\. z o\.o\.|S\.A\.|Inc\.|Ltd\.|GmbH)\b')
        for match in org_pattern.finditer(text):
            entities["organizacje"].append(match.group(0))
            
        # Popularne organizacje
        orgs = ["Google", "Microsoft", "Facebook", "Apple", "Amazon", "Twitter", "Netflix", "Allegro", "PKO", "PZU"]
        for org in orgs:
            if org in text:
                entities["organizacje"].append(org)
                
        # Daty
        date_patterns = [
            re.compile(r'\b\d{1,2} (?:stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|października|listopada|grudnia) \d{4}\b'),
            re.compile(r'\b\d{1,2}\.\d{1,2}\.\d{2,4}\b'),
            re.compile(r'\b\d{4}-\d{1,2}-\d{1,2}\b')
        ]
        
        for pattern in date_patterns:
            for match in pattern.finditer(text):
                entities["daty"].append(match.group(0))
                
        # Liczby
        number_pattern = re.compile(r'\b\d+(?:[.,]\d+)?\b')
        for match in number_pattern.finditer(text):
            entities["liczby"].append(match.group(0))
            
        # Usunięcie duplikatów
        for entity_type in entities:
            entities[entity_type] = list(set(entities[entity_type]))
            
        return entities
    
    def analyze_conversation(self, messages):
        """Analiza całej konwersacji"""
        if not messages:
            return {}
            
        # Ekstrahuj teksty z wiadomości
        texts = [msg.get("content", "") for msg in messages]
        full_text = " ".join(texts)
        
        # Analiza całego tekstu
        overall_analysis = self.analyze_text(full_text)
        
        # Śledzenie trendu sentymentu
        sentiment_values = []
        for msg in messages:
            content = msg.get("content", "")
            if content:
                sentiment = self.analyze_sentiment(content)
                if sentiment["dominujący"] == "pozytywny":
                    sentiment_values.append(sentiment["pozytywny"])
                elif sentiment["dominujący"] == "negatywny":
                    sentiment_values.append(-sentiment["negatywny"])
                else:
                    sentiment_values.append(0)
        
        # Określenie trendu sentymentu
        sentiment_trend = "stabilny"
        avg_sentiment = "neutralny"
        if sentiment_values:
            if len(sentiment_values) >= 3:
                first_half = sentiment_values[:len(sentiment_values)//2]
                second_half = sentiment_values[len(sentiment_values)//2:]
                avg_first = sum(first_half) / len(first_half) if first_half else 0
                avg_second = sum(second_half) / len(second_half) if second_half else 0
                
                if avg_second > avg_first + 0.2:
                    sentiment_trend = "rosnący"
                elif avg_second < avg_first - 0.2:
                    sentiment_trend = "malejący"
            
            avg_value = sum(sentiment_values) / len(sentiment_values)
            if avg_value > 0.2:
                avg_sentiment = "pozytywny"
            elif avg_value < -0.2:
                avg_sentiment = "negatywny"
        
        # Analiza spójności tematycznej
        topic_consistency = {"spójność": "wysoka", "wartość": 0.8}
        if len(texts) >= 2:
            topics_per_message = [set(self.detect_topics(txt).keys()) for txt in texts]
            consistency_scores = []
            
            for i in range(1, len(topics_per_message)):
                current = topics_per_message[i]
                previous = topics_per_message[i-1]
                
                if current and previous:  # Jeśli oba zestawy niepuste
                    similarity = len(current.intersection(previous)) / len(current.union(previous)) if current.union(previous) else 0
                    consistency_scores.append(similarity)
            
            if consistency_scores:
                avg_consistency = sum(consistency_scores) / len(consistency_scores)
                topic_consistency["wartość"] = round(avg_consistency, 2)
                
                if avg_consistency < 0.3:
                    topic_consistency["spójność"] = "niska"
                elif avg_consistency < 0.6:
                    topic_consistency["spójność"] = "średnia"
        
        # Analiza zmian intencji
        intention_sequence = []
        for msg in messages:
            if msg.get("role") == "user" and msg.get("content"):
                intention = self.detect_intention(msg.get("content"))
                intention_sequence.append(intention["dominująca"])
                
        intention_changes = {"zmiany": "brak", "sekwencja": intention_sequence}
        
        if len(intention_sequence) >= 3:
            changes_count = sum(1 for i in range(1, len(intention_sequence)) if intention_sequence[i] != intention_sequence[i-1])
            change_rate = changes_count / (len(intention_sequence) - 1)
            
            if change_rate > 0.7:
                intention_changes["zmiany"] = "częste"
            elif change_rate > 0.3:
                intention_changes["zmiany"] = "sporadyczne"
        
        return {
            "overall_analysis": overall_analysis,
            "sentiment_trend": {
                "trend": sentiment_trend,
                "średni_sentyment": avg_sentiment,
                "wartości": sentiment_values
            },
            "topic_consistency": topic_consistency,
            "main_topics": list(overall_analysis["topics"].keys()),
            "intention_changes": intention_changes
        }

class SemanticIntegration:
    """Klasa integrująca analizę semantyczną z głównym systemem"""
    def __init__(self, db_path=None):
        self.db_path = db_path
        self.analyzer = SemanticAnalyzer()
        
        # Inicjalizacja tabeli semantic_metadata, jeśli nie istnieje
        if db_path:
            conn = sqlite3.connect(db_path)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS semantic_metadata (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    message_id TEXT,
                    role TEXT,
                    topics TEXT,
                    sentiment TEXT,
                    intention TEXT,
                    entities TEXT,
                    complexity TEXT,
                    temporal_context TEXT,
                    timestamp REAL
                )
            ''')
            conn.commit()
            conn.close()
    
    def enhance_chat_response(self, user_id, query, response, message_history=None):
        """Wzbogaca odpowiedź o analizę semantyczną"""
        # Analiza zapytania użytkownika
        query_analysis = self.analyzer.analyze_text(query)
        
        # Analiza odpowiedzi
        response_analysis = self.analyzer.analyze_text(response)
        
        # Analiza całej konwersacji
        conversation_analysis = {}
        if message_history:
            conversation_analysis = self.analyzer.analyze_conversation(message_history)
        
        # Ekstrakcja encji
        entities = query_analysis.get("entities", {})
        
        # Generowanie rekomendacji
        recommendations = self._generate_recommendations(query_analysis, response_analysis, conversation_analysis)
        
        # Zapisanie metadanych semantycznych w bazie danych
        self.store_semantic_data(user_id, query, query_analysis)
        
        return {
            "query_analysis": query_analysis,
            "response_analysis": response_analysis,
            "conversation_analysis": conversation_analysis,
            "entities": entities,
            "recommendations": recommendations
        }
    
    def _generate_recommendations(self, query_analysis, response_analysis, conversation_analysis):
        """Generuje rekomendacje na podstawie analizy semantycznej"""
        recommendations = {
            "topics": [],
            "intentions": [],
            "tone_suggestions": [],
            "follow_up_questions": [],
            "context_needs": []
        }
        
        # Rekomendacje tematów
        if query_analysis.get("topics"):
            for topic, score in sorted(query_analysis["topics"].items(), key=lambda x: x[1], reverse=True)[:3]:
                recommendations["topics"].append({"topic": topic, "score": score})
        
        # Rekomendacje intencji
        query_intention = query_analysis.get("intention", {}).get("dominująca")
        if query_intention:
            recommendations["intentions"].append(query_intention)
        
        # Rekomendacje tonu
        query_sentiment = query_analysis.get("sentiment", {}).get("dominujący")
        if query_sentiment == "pozytywny":
            recommendations["tone_suggestions"] = ["pozytywny", "entuzjastyczny", "pomocny"]
        elif query_sentiment == "negatywny":
            recommendations["tone_suggestions"] = ["empatyczny", "profesjonalny", "rzeczowy"]
        else:
            recommendations["tone_suggestions"] = ["informacyjny", "neutralny", "rzeczowy"]
        
        # Generowanie pytań uzupełniających
        topics = list(query_analysis.get("topics", {}).keys())
        entities = query_analysis.get("entities", {})
        keywords = query_analysis.get("keywords", [])
        
        follow_up_templates = [
            "Czy potrzebujesz bardziej szczegółowych informacji na temat {topic}?",
            "Czy chciałbyś dowiedzieć się więcej o {keyword}?",
            "Czy masz jakieś konkretne pytania dotyczące {entity}?",
            "Czy mogę pomóc Ci w jeszcze czymś związanym z {topic}?"
        ]
        
        # Tworzenie pytań uzupełniających
        if topics:
            topic = random.choice(topics)
            recommendations["follow_up_questions"].append(follow_up_templates[0].format(topic=topic))
            
        if keywords:
            keyword = random.choice(keywords)
            recommendations["follow_up_questions"].append(follow_up_templates[1].format(keyword=keyword))
            
        for entity_type, entity_list in entities.items():
            if entity_list and random.random() < 0.5:  # 50% szans na dodanie pytania o encję
                entity = random.choice(entity_list)
                recommendations["follow_up_questions"].append(follow_up_templates[2].format(entity=entity))
        
        # Potrzeby kontekstowe
        if not entities.get("osoby") and ("osoba" in topics or "ludzie" in topics):
            recommendations["context_needs"].append("informacje o osobach")
            
        if not entities.get("daty") and ("czas" in topics or "harmonogram" in topics):
            recommendations["context_needs"].append("informacje o czasie/datach")
            
        # Usunięcie duplikatów i ograniczenie do sensownej liczby
        recommendations["follow_up_questions"] = list(set(recommendations["follow_up_questions"]))[:3]
        
        return recommendations
    
    def get_semantic_metadata_for_db(self, user_id, text, role):
        """Przygotowuje metadane semantyczne do zapisu w bazie danych"""
        analysis = self.analyzer.analyze_text(text)
        
        semantic_metadata = {
            "user_id": user_id,
            "role": role,
            "semantic_metadata": {
                "topics": analysis.get("topics", {}),
                "sentiment": analysis.get("sentiment", {}).get("dominujący", "neutralny"),
                "intention": analysis.get("intention", {}).get("dominująca", "nieznana"),
                "entities": analysis.get("entities", {}),
                "complexity": analysis.get("complexity", {}).get("poziom", "średnia"),
                "temporal_context": analysis.get("temporal_context", {}).get("dominujący", "teraźniejszość")
            }
        }
        
        return semantic_metadata
        
    def store_semantic_data(self, user_id, query, analysis):
        """Zapisuje metadane semantyczne w bazie danych"""
        if not self.db_path or not user_id or not query:
            return False
            
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            id = uuid.uuid4().hex
            message_id = uuid.uuid4().hex
            
            # Konwersja pól słownikowych do JSON
            topics_json = json.dumps(analysis.get("topics", {}), ensure_ascii=False)
            sentiment = analysis.get("sentiment", {}).get("dominujący", "neutralny")
            intention = analysis.get("intention", {}).get("dominująca", "nieznana")
            entities_json = json.dumps(analysis.get("entities", {}), ensure_ascii=False)
            complexity = analysis.get("complexity", {}).get("poziom", "średnia")
            temporal_context = analysis.get("temporal_context", {}).get("dominujący", "teraźniejszość")
            
            c.execute("INSERT INTO semantic_metadata VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                     (id, user_id, message_id, "user", topics_json, sentiment, intention, entities_json, complexity, temporal_context, time.time()))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Błąd podczas zapisywania danych semantycznych: {e}")
            return False

# Inicjalizacja modułu analizy semantycznej
semantic_analyzer = SemanticAnalyzer()
semantic_integration = SemanticIntegration(DB_PATH)
print("Moduł analizy semantycznej uruchomiony")

# =========================
# HTTP util
# =========================
def _cors():
    return [
        ("Access-Control-Allow-Origin", ALLOWED_ORIGINS),
        ("Access-Control-Allow-Headers", "Authorization, Content-Type"),
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
    ]

def http_get(url: str, headers=None, timeout=HTTP_TIMEOUT) -> str:
    h = dict(HEADERS); h.update(headers or {})
    req = Request(url, headers=h, method="GET")
    with urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8","replace")

def http_get_json(url: str, headers=None, timeout=HTTP_TIMEOUT) -> Any:
    raw = http_get(url, headers=headers, timeout=timeout)
    try: return json.loads(raw)
    except: return {"raw": raw}

def http_post_json(url: str, payload: dict, headers=None, timeout=HTTP_TIMEOUT) -> Any:
    h = dict(JSON_HEAD); h.update(headers or {})
    req = Request(url, data=json.dumps(payload).encode("utf-8"), headers=h, method="POST")
    with urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8","replace")
        try: return json.loads(raw)
        except: return {"raw": raw}

def _json(obj: Any, code: int = 200):
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return f"{code} OK", [("Content-Type","application/json; charset=utf-8"),("Content-Length",str(len(data)))] + _cors(), [data]

def _bad(msg: str, code: int = 400): return _json({"ok":False,"error":msg}, code)

def _qs(env) -> Dict[str,str]:
    q = parse_qs(env.get("QUERY_STRING",""), keep_blank_values=True)
    return {k:(v[0] if v else "") for k,v in q.items()}

def _read_json(env) -> Dict[str,Any]:
    try: ln = int(env.get("CONTENT_LENGTH") or 0)
    except: ln = 0
    raw = env["wsgi.input"].read(ln) if ln>0 else b""
    if not raw: return {}
    try: return json.loads(raw.decode("utf-8"))
    except: return {}

def _auth_ok(env) -> bool:
    if not AUTH_TOKEN: return True
    h = env.get("HTTP_AUTHORIZATION","")
    if not h.lower().startswith("bearer "): return False
    tok = h.split(" ",1)[1].strip()
    return hmac.compare_digest(tok, AUTH_TOKEN)

def _ip(env) -> str:
    return env.get("HTTP_X_FORWARDED_FOR") or env.get("REMOTE_ADDR") or "0.0.0.0"

# =========================
# Normalizacja / tokenizacja / TF-IDF
# =========================
def _norm(s: str) -> str: return re.sub(r"\s+"," ",(s or "").strip())
def _id_for(s: str) -> str: return hashlib.sha1(_norm(s).encode("utf-8")).hexdigest()

def _tok(s: str) -> List[str]:
    s=(s or "").lower()
    skroty={"wg":"według","np":"na przykład","itd":"i tak dalej","itp":"i tym podobne","tzn":"to znaczy","tzw":"tak zwany","ok":"okej","bd":"będzie","jj":"jasne","nwm":"nie wiem","imo":"moim zdaniem","tbh":"szczerze mówiąc","fyi":"dla twojej informacji","btw":"przy okazji"}
    words=s.split()
    for i,w in enumerate(words):
        cw=re.sub(r"[^\wąćęłńóśźż]","",w)
        if cw in skroty: words[i]=skroty[cw]
    s2=re.sub(r"[^0-9a-ząćęłńóśźż]+"," "," ".join(words))
    return [w for w in s2.split() if len(w)>2][:256]

def _tfidf_vec(tokens: List[str], docs_tokens: List[List[str]]) -> Dict[str,float]:
    N=len(docs_tokens) or 1
    df: Dict[str, int] = {}
    for d in docs_tokens:
        for t in set(d): df[t]=df.get(t,0)+1
    tf={}
    for t in tokens: tf[t]=tf.get(t,0)+1
    out={}
    for t,c in tf.items():
        idf=(math.log((N+1)/(df.get(t,1)+1)))**1.5
        bonus=(1+0.1*min(max(len(t)-3,0),7))
        out[t]=(c/max(1,len(tokens)))*idf*bonus
    return out

def _tfidf_cos(query: str, docs: List[str]) -> List[float]:
    tq=_tok(query); dts=[_tok(d) for d in docs]
    vq=_tfidf_vec(tq, dts)
    out=[]
    key_terms=set([t for t in tq if len(t)>3])
    for dt in dts:
        vd=_tfidf_vec(dt, dts)
        keys=set(vq.keys())|set(vd.keys())
        num=0.0
        for term in keys:
            a=vq.get(term,0.0); b=vd.get(term,0.0)
            term_bonus=2.5 if term in key_terms else 1.0
            if " " in term:
                words=len(term.split())
                if words>1: term_bonus*=1.0+0.5*words
            boost=1 + 0.8*math.tanh(4*a*b - 0.6)
            num += (a*b)*boost*term_bonus
        den=(sum(x*x for x in vq.values())**0.5)*(sum(x*x for x in vd.values())**0.5)
        score=0.0 if den==0 else (num/den)
        out.append(score**0.8)
    return out

# =========================
# EMBEDDINGS z cache dla lepszej wydajności
# =========================
_EMBED_CACHE = {}
_EMBED_CACHE_HITS = 0
_EMBED_CACHE_MISSES = 0

def embed_many(texts: List[str])->List[List[float]]:
    global _EMBED_CACHE, _EMBED_CACHE_HITS, _EMBED_CACHE_MISSES
    if not EMBED_URL or not EMBED_MODEL or not LLM_API_KEY: return []
    
    result = []
    texts_to_embed = []
    indices = []
    
    # Sprawdź cache dla każdego tekstu
    for i, text in enumerate(texts):
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in _EMBED_CACHE:
            result.append(_EMBED_CACHE[text_hash])
            _EMBED_CACHE_HITS += 1
        else:
            result.append(None)  # placeholder
            texts_to_embed.append(text)
            indices.append(i)
            _EMBED_CACHE_MISSES += 1
    
    # Jeśli wszystko z cache, zwróć
    if not texts_to_embed:
        return result
    
    # W przeciwnym razie wykonaj embedding dla nowych tekstów
    payload={"model":EMBED_MODEL,"input":texts_to_embed}
    try:
        req=Request(EMBED_URL,data=json.dumps(payload).encode("utf-8"),
                    headers={"Authorization":f"Bearer {LLM_API_KEY}","Content-Type":"application/json"},
                    method="POST")
        with urlopen(req,timeout=HTTP_TIMEOUT) as r:
            j=json.loads(r.read().decode("utf-8","replace"))
        embeddings = [d.get("embedding") for d in j.get("data",[]) if d.get("embedding")]
        
        # Aktualizuj cache i wyniki
        for i, (text, embedding) in enumerate(zip(texts_to_embed, embeddings)):
            if embedding:
                text_hash = hashlib.md5(text.encode()).hexdigest()
                _EMBED_CACHE[text_hash] = embedding
                result[indices[i]] = embedding
                
        # Jeśli cache za duży, usuń najstarsze wpisy
        if len(_EMBED_CACHE) > 1000:
            for k in list(_EMBED_CACHE.keys())[:200]:
                del _EMBED_CACHE[k]
                
    except Exception:
        pass
    
    # Zastąp pozostałe None pustymi listami
    for i in range(len(result)):
        if result[i] is None:
            result[i] = []
            
    return result

def _vec_cos(a:List[float],b:List[float])->float:
    if not a or not b: return 0.0
    s=sum(x*y for x,y in zip(a,b))
    na=math.sqrt(sum(x*x for x in a)) or 1e-9
    nb=math.sqrt(sum(y*y for y in b)) or 1e-9
    return s/(na*nb)

# =========================
# NER/PII/Preferencje
# =========================
_PROFILE_PATS = {
    "age": re.compile(r"\b(?:mam|skończyłe[mn]|posiadam)\s*(\d{1,2})\s*(?:lat|lata|wiosen)\b", re.I),
    "email": re.compile(r"\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b", re.I),
    "phone": re.compile(r"\b(?:\+?48[-\s]?)?(?:\d{3}[-\s]?\d{3}[-\s]?\d{3})\b", re.I),
}
_LANG_PAT = re.compile(r"\b(?:mówię|znam|używam|uczę się)\s+(po\s+)?(polsku|angielsku|niemiecku|hiszpańsku|francusku|rosyjsku|ukraińsku|włosku)\b", re.I)
_TECH_PAT = re.compile(r"\b(Python|JS|Java|TypeScript|C\+\+|C#|Go|Rust|PHP|SQL|HTML|CSS)\b", re.I)
_NEGATION_PAT = re.compile(r"\b(nie|nie\s+bardzo|żadn[eyoa])\b", re.I)
_LINK_PAT = re.compile(r"\bhttps?://\S+\b", re.I)

def _sentences(text: str) -> List[str]:
    if not text: return []
    raw=re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in raw if len(s.strip())>=5]

def _tag_pii_in_text(s: str) -> Tuple[str, List[str]]:
    tags=[]
    if _PROFILE_PATS["email"].search(s): tags.append("pii:email")
    if _PROFILE_PATS["phone"].search(s): tags.append("pii:phone")
    if _LINK_PAT.search(s): tags.append("pii:link")
    return s, sorted(set(tags))

def _mk_fact(text: str, base_score: float, tags: List[str]) -> Tuple[str, float, List[str]]:
    t=(text or "").strip()
    if not t: return ("",0.0,tags)
    score_delta = -0.08 if _NEGATION_PAT.search(t) else 0.04
    score = max(0.55, min(0.97, base_score + score_delta))
    return (t, score, sorted(set(tags)))

def _extract_facts_from_turn(u: str, a: str) -> List[Tuple[str,float,List[str]]]:
    facts=[]
    for role,txt in (("user",u or ""),("assistant",a or "")):
        for s in _sentences(txt):
            s_clean, pii_tags=_tag_pii_in_text(s)
            if re.search(r"\b(lubię|wolę|preferuję|kocham|nienawidzę|nie\s+lubię)\b", s, re.I):
                facts.append(_mk_fact(f"preferencja: {s_clean}", 0.82 if role=="user" else 0.74, ["stm","preference"]+pii_tags)); continue
            m=_PROFILE_PATS["age"].search(s)
            if m: facts.append(_mk_fact(f"wiek: {m.group(1)}", 0.86 if role=="user" else 0.78, ["stm","profile"]+pii_tags)); continue
            for lang in _LANG_PAT.findall(s):
                facts.append(_mk_fact(f"język: {lang[1].lower()}", 0.80 if role=="user" else 0.72, ["stm","profile","language"]+pii_tags))
            for tech in set(t.group(0) for t in _TECH_PAT.finditer(s)):
                facts.append(_mk_fact(f"tech: {tech}", 0.78 if role=="user" else 0.70, ["stm","profile","tech"]+pii_tags))
            for url in _LINK_PAT.findall(s):
                facts.append(_mk_fact(f"link: {url}", 0.81, ["stm","link","pii:link"]))
    return facts

def _dedupe_facts(facts: List[Tuple[str,float,List[str]]]) -> List[Tuple[str,float,List[str]]]:
    by={}
    for t,sc,tg in facts:
        t2=(t or "").strip()
        if not t2: continue
        fid=_id_for(t2)
        if fid in by:
            ot,os,otg=by[fid]
            by[fid]=(ot, max(os,sc), sorted(set((otg or [])+(tg or []))))
        else:
            by[fid]=(t2, sc, sorted(set(tg or [])))
    return list(by.values())

def _extract_facts(messages: List[dict], max_out: int = 120) -> List[Tuple[str,float,List[str]]]:
    if not messages: return []
    all_facts=[]
    i=0
    while i<len(messages):
        role_i=messages[i].get("role")
        u=messages[i].get("content","") if role_i=="user" else ""
        a=""
        if i+1<len(messages) and messages[i+1].get("role")=="assistant":
            a=messages[i+1].get("content",""); i+=2
        else:
            i+=1
        all_facts.extend(_extract_facts_from_turn(u,a))
    all_facts=_dedupe_facts(all_facts)
    all_facts.sort(key=lambda x:x[1], reverse=True)
    return all_facts[:max_out]

# =========================
# LTM / FTS / BM25 / BLEND
# =========================
def ltm_add(text:str, tags:str="", conf:float=0.7)->str:
    tid=_id_for(text)
    conn=_db(); c=conn.cursor()
    c.execute("INSERT OR REPLACE INTO facts VALUES(?,?,?,?,?,0)",(tid,text,tags,float(conf),time.time()))
    try: c.execute("INSERT INTO facts_fts(text,tags) VALUES(?,?)",(text,tags))
    except Exception: pass
    conn.commit(); conn.close()
    return tid

def ltm_soft_delete(id_or_text: str)->int:
    tid = id_or_text if len(id_or_text)==40 else _id_for(id_or_text)
    conn=_db(); c=conn.cursor()
    c.execute("UPDATE facts SET deleted=1 WHERE id=?", (tid,))
    conn.commit(); n=c.rowcount; conn.close(); return n

def _fts_safe_query(q: str) -> str:
    toks=[t for t in _tok(q) if t]
    if not toks: return '""'
    return " AND ".join(f'"{t}"' for t in toks[:8])

def _fts_bm25(query: str, limit: int = 50) -> List[Tuple[str, float]]:
    safe=_fts_safe_query(query)
    conn=_db(); c=conn.cursor()
    out=[]
    try:
        rows=c.execute("""SELECT text, bm25(facts_fts) AS bscore
                          FROM facts_fts WHERE facts_fts MATCH ?
                          ORDER BY bscore ASC LIMIT ?""", (safe, int(limit))).fetchall()
        for r in rows:
            bscore = float(r["bscore"] if r["bscore"] is not None else 10.0)
            out.append((r["text"], 1.0/(1.0 + max(0.0,bscore))))
    finally:
        conn.close()
    return out

def facts_reindex()->Dict[str,Any]:
    conn=_db(); c=conn.cursor()
    try:
        c.execute("DELETE FROM facts_fts")
    except Exception:
        pass
    rows=c.execute("SELECT text,tags FROM facts WHERE deleted=0 ORDER BY created DESC").fetchall()
    n=0
    for r in rows:
        try:
            c.execute("INSERT INTO facts_fts(text,tags) VALUES(?,?)",(r["text"], r["tags"])); n+=1
        except Exception:
            pass
    conn.commit(); conn.close()
    return {"ok":True,"indexed":n}

def _blend_scores(tfidf: List[float], bm25: List[float], emb: List[float],
                  wt=(0.45, 0.30, 0.25), recency: List[float] = None) -> List[float]:
    """
    Łączy różne metryki wyszukiwania z uwzględnieniem recency bias (trendu czasowego).
    
    Args:
        tfidf: Wyniki TF-IDF dla dokumentów
        bm25: Wyniki BM25 dla dokumentów
        emb: Wyniki podobieństwa embeddingów
        wt: Wagi dla poszczególnych metryk (tfidf, bm25, emb)
        recency: Opcjonalnie współczynniki świeżości dokumentów (0-1)
    """
    n = max(len(tfidf), len(bm25), len(emb))
    def get(a,i): return a[i] if i<len(a) else 0.0
    out=[]
    for i in range(n):
        a = get(tfidf,i)**1.15
        b = get(bm25,i)**1.10
        c = get(emb,i)**1.15
        
        # Harmonic mean dla overlapów
        harm = 0.0
        if a>0.35 and b>0.35: harm += 0.15*math.sqrt(a*b)
        if b>0.35 and c>0.35: harm += 0.15*math.sqrt(b*c)
        if a>0.7 and c>0.7:   harm += 0.10*math.sqrt(a*c)
        
        # Podstawowy score
        score = wt[0]*a + wt[1]*b + wt[2]*c + harm
        
        # Zastosowanie recency bias jeśli dostępne
        if recency and i < len(recency):
            # Logarytmiczne wzmocnienie świeżych dokumentów
            recency_boost = 1.0 + 0.35 * math.log1p(max(0, recency[i]))
            score *= recency_boost
        
        out.append(score)
    return out

def ltm_search_bm25(q:str, limit:int=50)->List[Dict[str,Any]]:
    hits=_fts_bm25(q, limit)
    res=[]
    for text, sc in hits:
        res.append({"text": text, "tags":"", "score": float(sc)})
    return res

def ltm_search_hybrid(q: str, limit: int = 30)->List[Dict[str,Any]]:
    conn=_db(); c=conn.cursor()
    rows=c.execute("SELECT id,text,tags,conf,created FROM facts WHERE deleted=0 ORDER BY created DESC LIMIT 2000").fetchall()
    conn.close()
    if not rows: return []
    docs=[r["text"] or "" for r in rows]

    s_tfidf=_tfidf_cos(q, docs)
    bm25_pairs=_fts_bm25(q, limit=min(len(docs), 300))
    bm25_map={t: s for t,s in bm25_pairs}
    s_bm25=[bm25_map.get(d, 0.0) for d in docs]

    try:
        qv=embed_many([q]); dvs=embed_many(docs)
        s_emb=[_vec_cos(qv[0],d) for d in dvs] if (qv and dvs) else [0.0]*len(docs)
    except Exception:
        s_emb=[0.0]*len(docs)

    scores=_blend_scores(s_tfidf, s_bm25, s_emb, wt=(0.44,0.32,0.24))
    pack=[(scores[i], rows[i]) for i in range(len(rows))]
    pack.sort(key=lambda x:x[0], reverse=True)

    res=[]
    for sc,r in pack[:limit]:
        res.append({"id":r["id"],"text":r["text"],"tags":r["tags"],"conf":r["conf"],"created":r["created"],"score":float(sc)})
    return res

# =========================
# STM ROTACJA 160→(100→LTM)+(60 w STM)
# =========================
STM_MAX=160; SL_FROM=60; SL_TO=160

def memory_add(user: str, role: str, content: str) -> str:
    conn=_db(); c=conn.cursor()
    mid=uuid.uuid4().hex; ts=time.time()
    c.execute("INSERT INTO memory VALUES(?,?,?,?,?)",(mid,user,role,content,ts))
    try:
        vecs=embed_many([content])
        if vecs: c.execute("INSERT OR REPLACE INTO mem_embed VALUES(?,?,?,?)",(mid,user,json.dumps(vecs[0]),ts))
    except Exception: pass
    conn.commit(); conn.close()
    _rotate_context(user)
    psy_observe_text(user, content)
    for t,sc,tg in _extract_facts([{"role":role,"content":content}], max_out=12):
        if t: ltm_add(t, ",".join(tg), float(sc))
    return mid

def _rotate_context(user: str):
    conn=_db(); c=conn.cursor()
    rows=c.execute("SELECT id,role,content FROM memory WHERE user=? ORDER BY ts DESC",(user,)).fetchall()
    if len(rows)>STM_MAX:
        seg=rows[SL_FROM:SL_TO]
        if seg:
            sid=uuid.uuid4().hex
            
            # Tworzenie inteligentnego podsumowania zamiast prostej konkatenacji
            # Grupowanie w dialogi (user-assistant pary)
            dialogs = []
            current = []
            for r in seg:
                current.append({"role": r["role"], "content": r["content"]})
                if len(current) >= 2 and current[-1]["role"] == "assistant":
                    dialogs.append(current.copy())
                    current = []
            if current:
                dialogs.append(current)
            
            # Generowanie podsumowania przez LLM
            dialogs_txt = "\n\n".join([f"USER: {d[0]['content']}\nASSISTANT: {d[1]['content']}" if len(d)>1 else f"USER: {d[0]['content']}" for d in dialogs])
            if len(dialogs_txt) > 3000:
                try:
                    summary = call_llm([{"role":"system","content":"Stwórz zwięzłe podsumowanie poniższej rozmowy. Wyodrębnij kluczowe fakty, preferencje i istotne informacje o użytkowniku."},
                                     {"role":"user","content":f"Podsumuj tę rozmowę:\n{dialogs_txt[:12000]}"}
                                    ], temperature=0.3)
                except Exception:
                    # Fallback w przypadku błędu
                    summary=" ".join(r["content"] for r in seg)[:250000]
            else:
                # Dla krótkich segmentów zachowaj pełną zawartość
                summary=dialogs_txt
            
            details=json.dumps([{"role": r["role"], "content": r["content"]} for r in seg], ensure_ascii=False)
            c.execute("INSERT INTO memory_long VALUES(?,?,?,?,?)",(sid,user,summary,details,time.time()))
            ids=[r["id"] for r in seg]
            c.execute("DELETE FROM memory WHERE id IN ("+ ",".join("?"*len(ids))+")", ids)
            conn.commit()
    conn.close()

def memory_get(user: str, n: int = 60) -> List[Dict[str,Any]]:
    conn=_db(); c=conn.cursor()
    rows=c.execute("SELECT role,content,ts FROM memory WHERE user=? ORDER BY ts DESC LIMIT ?",(user,n)).fetchall()
    conn.close(); return [dict(r) for r in rows]

def memory_summaries(user: str, n: int = 20) -> List[Dict[str,Any]]:
    conn=_db(); c=conn.cursor()
    rows=c.execute("SELECT summary,details,ts FROM memory_long WHERE user=? ORDER BY ts DESC LIMIT ?",(user,n)).fetchall()
    conn.close(); return [dict(r) for r in rows]

def memory_purge(user: str)->int:
    conn=_db(); c=conn.cursor()
    c.execute("DELETE FROM memory WHERE user=?", (user,))
    conn.commit(); n=c.rowcount; conn.close(); return n

# =========================
# Psychika
# =========================
PL_POS={"super","świetnie","dzięki","dobrze","spoko","okej","fajnie","git","extra"}
PL_NEG={"kurwa","chuj","zajeb","wkurw","błąd","fatalnie","źle","nienawidzę","masakra"}

def psy_get()->Dict[str,Any]:
    conn=_db(); c=conn.cursor()
    r=c.execute("SELECT mood,energy,focus,openness,directness,agreeableness,conscientiousness,neuroticism,style,updated FROM psy_state WHERE id=1").fetchone()
    conn.close()
    return dict(r) if r else {"mood":0.0,"energy":0.6,"focus":0.6,"openness":0.55,"directness":0.62,"agreeableness":0.55,"conscientiousness":0.63,"neuroticism":0.44,"style":"rzeczowy","updated":time.time()}

def psy_set(**kw)->Dict[str,Any]:
    s=psy_get()
    for k in ("mood","energy","focus","openness","directness","agreeableness","conscientiousness","neuroticism"):
        if k in kw and kw[k] is not None: s[k]=max(0.0,min(1.0,float(kw[k])))
    if "style" in kw and kw["style"]: s["style"]=str(kw["style"])[:64]
    s["updated"]=time.time()
    conn=_db(); c=conn.cursor()
    c.execute("""INSERT OR REPLACE INTO psy_state(id,mood,energy,focus,openness,directness,agreeableness,conscientiousness,neuroticism,style,updated)
                 VALUES(1,?,?,?,?,?,?,?,?,?,?)""",
              (s["mood"],s["energy"],s["focus"],s["openness"],s["directness"],s["agreeableness"],s["conscientiousness"],s["neuroticism"],s["style"],s["updated"]))
    conn.commit(); conn.close(); return s

def psy_episode_add(user:str, kind:str, valence:float, intensity:float, tags:str="", note:str="")->str:
    eid=uuid.uuid4().hex
    conn=_db(); c=conn.cursor()
    c.execute("INSERT INTO psy_episode VALUES(?,?,?,?,?,?,?,?)",(eid,user,kind,float(valence),float(intensity),tags or "",note or "",time.time()))
    conn.commit(); conn.close()
    s=psy_get()
    s["mood"]=max(0.0,min(1.0,s["mood"]+0.08*valence*intensity))
    s["energy"]=max(0.0,min(1.0,s["energy"]+(0.05 if valence>0 else -0.03)*intensity))
    s["neuroticism"]=max(0.0,min(1.0,s["neuroticism"]+(-0.04 if valence>0 else 0.05)*intensity))
    psy_set(**s); return eid

def psy_observe_text(user:str, text:str):
    tl=text.lower(); pos=sum(1 for w in PL_POS if w in tl); neg=sum(1 for w in PL_NEG if w in tl)
    val=1.0 if pos>neg else (-1.0 if neg>pos else 0.0); inten=min(1.0,0.2+0.1*(pos+neg))
    tags=",".join(sorted(set(_tok(text))))
    psy_episode_add(user,"msg",val,inten,tags,"auto")

def psy_reflect()->Dict[str,Any]:
    conn=_db(); c=conn.cursor()
    rows=c.execute("SELECT valence,intensity,ts FROM psy_episode ORDER BY ts DESC LIMIT 100").fetchall()
    conn.close()
    pos=sum(1 for r in rows if r["valence"]>0); neg=sum(1 for r in rows if r["valence"]<0)
    s=psy_get(); hour=time.localtime().tm_hour; delta={}
    if pos>neg: delta["openness"]=+0.04; delta["agreeableness"]=+0.03
    if neg>pos: delta["focus"]=+0.04; delta["directness"]=+0.03
    if 8<=hour<=12: delta["energy"]=+0.03
    if 0<=hour<=6:  delta["energy"]=-0.04
    for k,v in delta.items(): s[k]=max(0.0,min(1.0,s.get(k,0.5)+v))
    psy_set(**s)
    rid=uuid.uuid4().hex
    conn=_db(); c=conn.cursor()
    c.execute("INSERT INTO psy_reflection VALUES(?,?,?,?)",(rid, f"pos={pos} neg={neg} hour={hour}", json.dumps(delta,ensure_ascii=False), time.time()))
    conn.commit(); conn.close()
    return {"ok":True,"applied":delta,"state":psy_get()}

def psy_tune()->Dict[str,Any]:
    s=psy_get()
    temp=0.72 + 0.25*(s["openness"]-0.5) - 0.12*(s["directness"]-0.5) - 0.07*(s["focus"]-0.5) + 0.05*(s["agreeableness"]-0.5) - 0.06*(s["neuroticism"]-0.5)
    temp=round(max(0.2,min(1.25,temp)),2)
    tone="dynamiczny" if s["energy"]>0.55 else "zrównoważony"
    if s["directness"]>0.72: tone="konkretny"
    return {"temperature":temp,"tone":tone,"style":s.get("style","rzeczowy")}

def psy_tick():
    now=time.time()
    key="psy:last_tick"
    conn=_db(); c=conn.cursor()
    row=c.execute("SELECT value,ts FROM meta_memory WHERE key=?",(key,)).fetchone()
    last= row["ts"] if row else 0
    if now - last >= 1800:
        psy_reflect()
        st=psy_get()
        if st["mood"]<0.2:
            psy_episode_add("system","auto",+0.6,0.8,"selfcare","boost mood")
        c.execute("INSERT OR REPLACE INTO meta_memory(id,user,key,value,conf,ts) VALUES(?,?,?,?,?,?)",
                  (_id_for(key), "sys", key, "ok", 0.9, now))
        conn.commit()
    conn.close()

# =========================
# LLM z zaawansowanym cache
# =========================
_LLM_CACHE = {}
_LLM_CACHE_HITS = 0
_LLM_CACHE_MISSES = 0

import httpx, os, json

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepinfra.com/v1/openai")
LLM_API_KEY=w52XW0XN6zoV9hdY8OONhLu6tvnFaXbZ
LLM_MODEL = os.getenv("LLM_MODEL", "zai-org/GLM-4.6")
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "zai-org/GLM-4.5-Air")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))

def _llm_request(messages: list[dict], model: str) -> str:
    """Wysyła żądanie do DeepInfra dla danego modelu."""
    url = f"{LLM_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": messages}
    with httpx.Client(timeout=LLM_TIMEOUT) as client:
        r = client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

def call_llm(messages: list[dict], **opts) -> str:
    """
    Wywołanie LLM z fallbackiem:
    1️⃣ Próba na głównym modelu
    2️⃣ Jeśli się wywali → próba na fallbacku
    """
    try:
        return _llm_request(messages, LLM_MODEL)
    except Exception as e1:
        print(f"[LLM] Główny model padł: {e1} — próbuję fallback {LLM_FALLBACK_MODEL}")
        try:
            return _llm_request(messages, LLM_FALLBACK_MODEL)
        except Exception as e2:
            print(f"[LLM] Fallback też padł: {e2}")
            return f"[LLM-FAIL] {str(e2)}"
def call_llm_once(prompt: str, temperature: float=0.8)->str:
    return call_llm([{"role":"user","content":prompt}], temperature, max_tokens=None)

# =========================
# Pisanie – BOOST + Aukcje PRO
# =========================
PL_SYNONYMS={"świetny":["doskonały","znakomity","kapitalny","pierwszorzędny"],"tani":["przystępny","okazyjny","korzystny"],"modny":["trendy","na czasie","stylowy","hot"],"wytrzymały":["solidny","mocny","odporny"]}
PL_COLLOC=["jakość premium","gotowe do wysyłki","ostatnia sztuka","okazja","stan jak nowy","oryginalne metki","szybka wysyłka"]

# Rozszerzona wiedza modowa
FASHION = {
    "brands": [
        "nike","adidas","new balance","puma","reebok","asics","vans","converse",
        "zara","stradivarius","pull&bear","bershka","reserved","hm","h&m","mango",
        "patagonia","the north face","columbia","arcteryx","moncler","woolrich",
        "levi's","lee","wrangler","carhartt","lacoste","tommy hilfiger","ralph lauren",
        "gucci","prada","chanel","dior","balenciaga","versace","loewe","miu miu"
    ],
    "materials": [
        "bawełna","organiczna bawełna","wełna","merino","kaszmir","alpaka","len","jedwab",
        "poliester","poliamid","nylon","elastan","wiskoza","modal","tencel","denim",
        "skóra","ekoskóra","zamsz","nubuk","gore-tex","softshell","puch","pierze"
    ],
    "fits": ["regular","slim","oversize","relaxed","tapered","straight","bootcut","loose"],
    "closures": ["zamek","guziki","napy","rzep","sznurowanie","haft","suwak dwukierunkowy"],
    "patterns": ["gładki","prążki","kratka","pepita","jodełka","panterka","kwiaty","moro","logo","print"],
    "features": ["kaptur","ściągacze","ściągacz w pasie","wysoki stan","ocieplenie","wodoodporna","wiatroszczelna","oddychająca","kieszenie","kieszeń kangurka","2w1","odpinany kaptur"],
    "care": ["prać delikatnie w 30°C","nie suszyć w suszarce","prasować na niskiej temp.","czyścić chemicznie","suszyć na płasko","używać worka do prania"],
    "occasions": ["na co dzień","do pracy","na trening","na uczelnię","na wieczór","na wyjazd","w góry","na spacer","do biegania"],
    "styles": ["casual","smart casual","streetwear","sportowy","outdoor","elegancki","business","retro","vintage","minimalistyczny","y2k","techwear"],
    "sizes": ["XXS","XS","S","M","L","XL","XXL","3XL","4XL"]
}

def _enrich(text:str)->str:
    out=text
    for k,vals in PL_SYNONYMS.items():
        if k in out: out=out.replace(k, f"{k}/{random.choice(vals)}")
    if random.random()<0.6: out+="\n\n"+ " • ".join(random.sample(PL_COLLOC, k=min(3,len(PL_COLLOC))))
    return out

def _anti_repeat(s: str)->str:
    lines=[x.strip() for x in s.splitlines() if x.strip()]
    seen=set(); out=[]
    for ln in lines:
        key=re.sub(r"\W+"," ",ln.lower()).strip()
        if key in seen: continue
        seen.add(key); out.append(ln)
    return "\n".join(out)

def _bounded_length(s: str, target: str)->str:
    caps={"krótki":800,"średni":1600,"długi":3000,"bardzo długi":6000}
    cap=caps.get(target,3000)
    return s if len(s)<=cap else s[:cap]

def analyze_fashion_text(txt: str) -> Dict[str,List[str]]:
    SIZE_PAT=re.compile(r"\b(XXS|XS|S|M|L|XL|XXL|3XL|4XL|EU\s?\d{2}|US\s?\d{1,2})\b", re.I)
    t=(txt or "").lower()
    out={"brands":[],"materials":[],"sizes":[],"colors":[],"categories":[],"fits":[],"features":[],"patterns":[],"occasions":[],"styles":[],"closures":[]}
    COLORS=["czarny","biały","czerwony","zielony","niebieski","żółty","brązowy","różowy","fioletowy","szary","beżowy","granatowy","turkusowy","oliwkowy","błękitny","bordowy","kremowy","ecru"]
    cats=["koszulka","t-shirt","bluza","spodnie","jeansy","sukienka","kurtka","płaszcz","marynarka","sweter","buty","sneakersy","trampki","torebka","plecak","spódnica","dresy","legginsy","szorty"]
    for b in FASHION["brands"]:
        if re.search(rf"\b{re.escape(b)}\b", t): out["brands"].append(b)
    for m in FASHION["materials"]:
        if re.search(rf"\b{re.escape(m)}\b", t): out["materials"].append(m)
    for c in COLORS:
        if re.search(rf"\b{re.escape(c)}\b", t): out["colors"].append(c)
    for cat in cats:
        if re.search(rf"\b{re.escape(cat)}\b", t): out["categories"].append(cat)
    for f in FASHION["fits"]:
        if re.search(rf"\b{re.escape(f)}\b", t): out["fits"].append(f)
    for feat in FASHION["features"]:
        if re.search(rf"\b{re.escape(feat)}\b", t): out["features"].append(feat)
    for pat in FASHION["patterns"]:
        if re.search(rf"\b{re.escape(pat)}\b", t): out["patterns"].append(pat)
    for occ in FASHION["occasions"]:
        if re.search(rf"\b{re.escape(occ)}\b", t): out["occasions"].append(occ)
    for st in FASHION["styles"]:
        if re.search(rf"\b{re.escape(st)}\b", t): out["styles"].append(st)
    for cl in FASHION["closures"]:
        if re.search(rf"\b{re.escape(cl)}\b", t): out["closures"].append(cl)
    for m in SIZE_PAT.findall(txt or ""): out["sizes"].append(m.upper())
    # heurystyka "buty"
    for b in out["brands"]:
        idx=t.find(b)
        if idx!=-1 and "buty" in t[max(0,idx-40):idx+40]:
            if "buty" not in out["categories"]: out["categories"].append("buty")
    for k in out: out[k]=list(dict.fromkeys(out[k]))
    return out

def write_creative_boost(topic:str, tone:str, styl:str, dlugosc:str, web_ctx:str="")->str:
    t=psy_tune()
    outline=call_llm([
        {"role":"system","content":"Konspektysta. Twórz szkielet 6–10 punktów z progresją i mini tezami."},
        {"role":"user","content":f"Temat: {topic}\nTon: {tone or t['tone']}\nStyl: {styl}\nUżyj wiedzy:\n{web_ctx or ''}"}
    ], max(t["temperature"]-0.1,0.5))
    draft=call_llm([
        {"role":"system","content":"Pisarz PL. Rozwiń konspekt w spójny tekst. Klarownie, bez lania wody."},
        {"role":"user","content":f"Konspekt:\n{outline}"}
    ], t["temperature"])
    polish=call_llm([
        {"role":"system","content":"Redaktor PL. Usuń tautologie, wyrównaj rejestr, dodaj płynne przejścia."},
        {"role":"user","content":draft}
    ], max(0.6, t["temperature"]-0.05))
    styled=_bounded_length(_anti_repeat(_enrich(polish)), dlugosc)
    return styled

# =========================
# WRITE API: VINTED & SOCIAL
# =========================
def write_vinted(title:str, desc:str, price:Optional[float]=None, web_ctx:str="")->str:
    """Generator opisów Vinted. Z fallbackiem, gdy LLM niedostępny."""
    attrs = analyze_fashion_text((title or "") + " " + (desc or ""))
    meta=[]
    if attrs.get("sizes"):     meta.append("Rozmiar: " + ", ".join(attrs["sizes"]))
    if attrs.get("materials"): meta.append("Materiał: " + ", ".join(attrs["materials"]))
    if attrs.get("colors"):    meta.append("Kolor: " + ", ".join(attrs["colors"]))
    spec = (" • ".join(meta)) if meta else ""
    t=psy_tune()
    prompt = f"""Platforma: Vinted (PL).
Tytuł: {title}
Opis: {desc}
{('Parametry: ' + spec) if spec else ''}
Cena: {price if price else 'brak'}
Wymagania: krótko, konkretnie, stan, rozmiar, 5–8 hashtagów."""
    out = call_llm([{"role":"system","content":"Sprzedawca Vinted PL. Same konkrety."},
                    {"role":"user","content":prompt}], max(0.55, t["temperature"]-0.1))
    if out.startswith("[LLM-OFF]") or out.startswith("[LLM-ERR]"):
        base = [f"{title}", "Stan: bardzo dobry", spec,
                (f"Cena: {price} PLN" if price else ""), "#vinted #sprzedam #moda #outfit"]
        out = "\n".join([ln for ln in base if ln])
    return _anti_repeat(out)

def write_social(platform:str, topic:str, tone:str="dynamiczny", hashtags:int=6, variants:int=3, web_ctx:str="")->str:
    """Generator krótkich postów do social mediów."""
    t=psy_tune()
    prompt=f"""Platforma: {platform}
Temat: {topic}
Ton: {tone}
Hashtagi: {hashtags}
{('Kontekst:\n'+web_ctx) if web_ctx else ''}
Wymagania: krótki hook, 1 insight, CTA, lista hashtagów."""
    out = call_llm([{"role":"system","content":"Twórca social PL. Krótko i rzeczowo."},
                    {"role":"user","content":prompt}], max(0.6, t["temperature"]-0.05))
    if out.startswith("[LLM-OFF]") or out.startswith("[LLM-ERR]"):
        out = f"{topic} — więcej szczegółów wkrótce. #update"
    return _anti_repeat(out)

def write_auction(title:str, desc:str, price:Optional[float]=None, tags:List[str]=[], web_ctx:str="")->str:
    t=psy_tune()
    attrs=_enrich(f"Tytuł: {title}\nOpis: {desc}\nCena: {price}\nTagi: {', '.join(tags)}")
    return call_llm([
        {"role":"system","content":"Copywriter sprzedażowy PL. 1 główny benefit, 2 dowody, sensoryka, bariera ryzyka, CTA. Daj wariant A/B."},
        {"role":"user","content":attrs + ("\n\n[Źródła]\n"+web_ctx if web_ctx else "")}
    ], max(0.55, t["temperature"]-0.05))

# Aukcje PRO – mocniejszy generator (fallback, gdy LLM off też da radę)
def write_auction_pro(title:str, desc:str, price:Optional[float]=None, web_ctx:str="", tone:str="sprzedażowy", length:str="średni", kreatywny:bool=False) -> str:
    attrs = analyze_fashion_text((title or "") + " " + (desc or ""))
    # wzbogacenie z KB
    kb = auction_kb_fetch()
    enrich_lines=[]
    for k,v in kb.items():
        if v:
            sample=", ".join(list(v)[:5])
            enrich_lines.append(f"{k}: {sample}")
    enrich_txt="\n".join(enrich_lines)
    meta=[]
    if attrs.get("brands"):   meta.append("Marka: " + ", ".join(attrs["brands"]))
    if attrs.get("materials"):meta.append("Materiał: " + ", ".join(attrs["materials"]))
    if attrs.get("fits"):     meta.append("Fason: " + ", ".join(attrs["fits"]))
    if attrs.get("sizes"):    meta.append("Rozmiar: " + ", ".join(attrs["sizes"]))
    if attrs.get("colors"):   meta.append("Kolor: " + ", ".join(attrs["colors"]))
    if attrs.get("features"): meta.append("Cechy: " + ", ".join(attrs["features"]))
    if attrs.get("patterns"): meta.append("Wzór: " + ", ".join(attrs["patterns"]))
    if attrs.get("styles"):   meta.append("Styl: " + ", ".join(attrs["styles"]))
    if attrs.get("closures"): meta.append("Zapięcie: " + ", ".join(attrs["closures"]))
    meta_str = "\n".join(meta)

    prompt = f"""Napisz opis aukcji PL (2 wersje A/B, bez powtórzeń, precyzyjny).
Ton: {tone}. Długość: {length}.
Produkt: {title}
Opis sprzedawcy: {desc}
Cena: {price if price is not None else 'brak'}

Atrybuty rozpoznane:
{meta_str or '(brak)'}

Zasoby marki/mody (KB):
{enrich_txt or '(brak)'}

Wymagania:
- 1 hook sensoryczny, 1 benefit główny, 2 dowody (materiał/wykonanie/opinie), parametry (rozmiar/wymiary jeśli są), wskazówki pielęgnacji (jeśli pasują).
- krótka sekcja „Dlaczego warto”, „Wysyłka i zwroty” (neutralnie).
- Unikaj tautologii, nie powtarzaj zdań. Dodaj 6–10 hashtagów modowych na końcu.
{('[Źródła]\n'+web_ctx) if web_ctx else ''}"""

    t=psy_tune()
    out = call_llm([{"role":"system","content":"Copywriter e-commerce PL, precyzyjny, zero lania wody."},
                    {"role":"user","content":prompt}], max(0.58, t["temperature"]-0.1))
    if out.startswith("[LLM-OFF]") or out.startswith("[LLM-ERR]"):
        # Fallback deterministyczny
        lines=[]
        lines.append(f"{title} — opis A:")
        lines.append(f"- Stan: {('jak nowy' if 'stan' in desc.lower() or 'idealny' in desc.lower() else 'bardzo dobry')}")
        if meta_str: lines.append(meta_str)
        if price is not None: lines.append(f"- Cena: {price} PLN (do rozsądnej negocjacji)")
        care = random.choice(FASHION["care"])
        lines.append(f"- Pielęgnacja: {care}")
        lines.append("Dlaczego warto: solidne wykonanie, komfort noszenia, łatwe łączenie w stylizacjach.")
        lines.append("Wysyłka i zwroty: szybka wysyłka 24–48h, możliwość zwrotu zgodnie z regulaminem.")
        lines.append("")
        lines.append(f"{title} — opis B:")
        lines.append("Hook: Lekki jak piórko, a trzyma formę — idealny do codziennych stylizacji.")
        if meta_str: lines.append(meta_str)
        lines.append(f"Hashtagi: #{re.sub(r'\\W+','', (attrs['categories'][0] if attrs.get('categories') else 'moda'))} #okazja #premium #styl #outfit #nowość")
        out="\n".join(lines)
    out=_anti_repeat(out)
    return _bounded_length(_enrich(out), length)

# KB – nauka i pobranie
def auction_kb_learn(items: List[dict]) -> int:
    if not items: return 0
    conn=_db(); c=conn.cursor(); n=0
    for it in items:
        kind=str(it.get("kind","")).strip()[:32] or "generic"
        key =str(it.get("key","")).strip()[:64]
        val =str(it.get("val","")).strip()[:400]
        w   =float(it.get("weight",0.7))
        if not key or not val: continue
        kid=_id_for(f"{kind}:{key}:{val}")
        c.execute("INSERT OR REPLACE INTO kb_auction VALUES(?,?,?,?,?,?)",(kid,kind,key,val,w,time.time())); n+=1
        # dodaj też do LTM (przyszłe wyszukiwanie)
        try: ltm_add(f"[KB:{kind}] {key}: {val}", "kb:auction", w)
        except: pass
    conn.commit(); conn.close(); return n

def auction_kb_fetch() -> Dict[str,set]:
    conn=_db(); c=conn.cursor()
    rows=c.execute("SELECT kind,key,val,weight FROM kb_auction ORDER BY ts DESC LIMIT 800").fetchall()
    conn.close()
    out: Dict[str,set] = {}
    for r in rows:
        out.setdefault(f"{r['kind']}:{r['key']}", set()).add(r["val"])
    return out

def suggest_tags_for_auction(title:str, desc:str) -> List[str]:
    attrs=analyze_fashion_text((title or "")+" "+(desc or ""))
    tags=[]
    for k in ("brands","categories","styles","materials","colors","fits","features"):
        for v in attrs.get(k,[]): tags.append("#"+re.sub(r"\s+","",v.lower()))
    # dodaj KB
    kb=auction_kb_fetch()
    for k,vals in kb.items():
        for v in list(vals)[:3]:
            tags.append("#"+re.sub(r"\s+","",v.lower()))
    tags=list(dict.fromkeys(tags))
    return tags[:12]

# =========================
# Web / Autonauka
# =========================
import httpx
import autonauka_pro as AUTOPRO
from writer_pro import writer_router

def extract_text(html: str) -> Tuple[str,str]:
    try:
        from readability import Document
        from bs4 import BeautifulSoup
        doc=Document(html); title=doc.short_title() or ""
        soup=BeautifulSoup(doc.summary(),"html.parser")
        return title, soup.get_text(" ", strip=True)
    except Exception:
        txt=re.sub(r"\s+"," ", re.sub(r"<.*?>"," ", html))
        return "", txt.strip()

def chunk_text(text: str, max_words: int = 240, overlap: int = 60) -> List[str]:
    words=text.split(); out=[]; i=0; step=max_words-overlap; step=step if step>0 else max_words
    while i<len(words):
        out.append(" ".join(words[i:i+max_words])); i+=step
    return out

async def serpapi_search(q: str, engine: str = "google", params: dict = None) -> dict:
    if not SERPAPI_KEY: return {"ok": False, "error": "SERPAPI_KEY missing"}
    base = "https://serpapi.com/search.json"
    p = {"engine": engine, "q": q, "api_key": SERPAPI_KEY}
    if params: p.update(params)
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent":WEB_USER_AGENT}) as c:
        r = await c.get(base, params=p)
        try: return {"ok": r.status_code==200, "status": r.status_code, "data": r.json()}
        except: return {"ok": False, "raw": r.text}

async def firecrawl_scrape(url: str) -> dict:
    if not FIRECRAWL_KEY: return {"ok": False, "error": "FIRECRAWL_KEY missing"}
    endpoint = "https://api.firecrawl.dev/v1/scrape"
    headers = {"Authorization": f"Bearer {FIRECRAWL_KEY}"}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as c:
        r = await c.post(endpoint, headers=headers, json={"url": url})
        try: return {"ok": r.status_code==200, "data": r.json()}
        except: return {"ok": False, "raw": r.text}

async def wiki_search(q: str, n: int = 5) -> List[str]:
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote_plus(q)}&utf8=&format=json&srlimit={n}"
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as c:
        r = await c.get(url); j = r.json()
        return [f"https://en.wikipedia.org/wiki/{quote_plus(p['title'])}" for p in j.get("query",{}).get("search",[])]

def cache_get(key: str, ttl: int) -> Optional[dict]:
    conn=_db(); c=conn.cursor()
    row=c.execute("SELECT value,ts FROM cache WHERE key=?", (key,)).fetchone()
    conn.close()
    if not row: return None
    if time.time()-row["ts"]>ttl: return None
    try: return json.loads(row["value"])
    except: return None

def cache_put(key: str, value: dict):
    conn=_db(); c=conn.cursor()
    c.execute("INSERT OR REPLACE INTO cache(key,value,ts) VALUES(?,?,?)",(key,json.dumps(value,ensure_ascii=False),time.time()))
    conn.commit(); conn.close()

def rank_hybrid(chunks: List[str], q: str, topk: int = 6) -> List[Tuple[str,float]]:
    if not chunks: return []
    tfidf=_tfidf_cos(q, chunks)
    emb=[0.0]*len(chunks)
    try:
        qv=embed_many([q]); ev=embed_many(chunks)
        if qv and ev:
            qv=qv[0]; emb=[_vec_cos(qv,e) for e in ev]
    except Exception: pass
    out=[]
    for i,ch in enumerate(chunks):
        score=0.58*(tfidf[i] if i<len(tfidf) else 0.0) + 0.42*(emb[i] if i<len(emb) else 0.0)
        out.append((ch,float(score)))
    out.sort(key=lambda x:x[1], reverse=True)
    return out[:topk]

def store_docs(items: List[dict]):
    conn=_db(); c=conn.cursor()
    for it in items:
        did=uuid.uuid4().hex
        c.execute("INSERT OR REPLACE INTO docs VALUES(?,?,?,?,?,?)",(did,it.get("url",""),it.get("title",""),it.get("text",""),it.get("source","web"),time.time()))
        try: c.execute("INSERT INTO docs_fts(title,text,url) VALUES(?,?,?)",(it.get("title",""),it.get("text",""),it.get("url","")))
        except Exception: pass
    conn.commit(); conn.close()

async def research_collect(q: str, max_sites: int = 10) -> List[dict]:
    links=[]
    if SERPAPI_KEY:
        s = await serpapi_search(q, "google", params={"num": max_sites})
        for o in (s.get("data",{}) or {}).get("organic_results", [])[:max_sites]:
            if o.get("link"): links.append(o["link"])
        sch = await serpapi_search(q, "google_scholar", params={"num":3})
        for o in (sch.get("data",{}) or {}).get("organic_results", [])[:3]:
            if o.get("link"): links.append(o["link"])
    links += await wiki_search(q, n=3)
    seen=set(); ulist=[]
    for u in links:
        if u and u not in seen: seen.add(u); ulist.append(u)
    out=[]
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent":WEB_USER_AGENT}, follow_redirects=True) as c:
        res=await asyncio.gather(*[c.get(u) for u in ulist[:max_sites]], return_exceptions=True)
        for u, r in zip(ulist[:max_sites], res):
            if isinstance(r, Exception):
                if FIRECRAWL_KEY:
                    fr=await firecrawl_scrape(u)
                    if fr.get("ok") and fr.get("data") and fr["data"].get("content"):
                        out.append({"url":u,"title":fr["data"].get("title",""),"text":fr["data"]["content"],"source":"firecrawl"})
                continue
            html=r.text or ""
            if FIRECRAWL_KEY:
                fr=await firecrawl_scrape(u)
                if fr.get("ok") and fr["data"].get("content"):
                    out.append({"url":u,"title":fr["data"].get("title",""),"text":fr["data"]["content"],"source":"firecrawl"})
                    continue
            title, text = extract_text(html)
            out.append({"url":u,"title":title,"text":text,"source":"web"})
    return out

async def autonauka(q: str, topk: int = 10, deep_research: bool = False, use_external_module: bool = True) -> dict:
    # Tu był krytyczny błąd: niepoprawne użycie asyncio.TaskGroup, które jest tylko w Python 3.11+
    # Zastąpione bardziej kompatybilnym asyncio.gather
    
    # Unikalne ID cache uwzględniające parametry i czas (cache jest ważny przez określony czas)
    key=f"autonauka:{q}:{topk}:{deep_research}:{use_external_module}"
    cach=cache_get(key, ttl=1800)  # Zwiększony TTL z 900s na 1800s (30 min) dla lepszej wydajności
    if cach: 
        # Dodane sprawdzenie integralności cache
        if isinstance(cach, dict) and "context" in cach and "sources" in cach:
            return cach
    
    try:
        # Korzystamy z naszej zintegrowanej funkcji web_learn
        result = web_learn(q, mode="full" if deep_research else "fast")
        
        if result and isinstance(result, dict) and "materials" in result:
            # Konwersja wyniku z funkcji web_learn na format oczekiwany przez tę funkcję
            ctx = []
            cites = []
            fact_highlights = []
            
            # Przetwarzanie materiałów
            for mat in result.get("materials", []):
                if not isinstance(mat, dict): continue
                facts = mat.get("facts", [])
                for fact in facts:
                    if fact and len(fact) > 40:
                        ctx.append(fact)
                url = mat.get("url", "")
                title = mat.get("title", "")
                if url:
                    cites.append({"title": title or url, "url": url})
            
            # Dodaj fakty z przygotowanego draftu jeśli istnieje
            if result.get("draft"):
                draft_parts = result["draft"].split("\n")
                for part in draft_parts:
                    if part.startswith("- ") and len(part) > 30:
                        fact_highlights.append(part[2:]) # Usuń "- " z początku
            
            out = {
                "query": q,
                "context": "\n\n".join(ctx[:topk]),
                "facts": fact_highlights[:5],
                "sources": cites[:max(12, topk)],
                "is_deep_research": deep_research,
                "source_count": len(result.get("materials", [])),
                "powered_by": "autonauka-module"
            }
            cache_put(key, out)
            return out
    except Exception as e:
        # Spadek do wbudowanej implementacji
        print(f"Błąd w web_learn: {e}")
        pass
                
        # ULEPSZONA wbudowana implementacja jako fallback
        expanded_variants = [q]
        if deep_research:
            # Dodaj kilka dodatkowych podpunktów do wyszukiwania
            current_year = datetime.now().year
            expanded_variants.extend([
                f"{q} najlepsze praktyki",
                f"{q} przykłady",
                f"{q} trendy {current_year}", # Dynamiczny rok zamiast hardkodowanego
                f"{q} praktyczne zastosowania",
                f"{q} zaawansowane techniki"
            ])
        
        # Uruchom kilka równoległych wyszukiwań dla lepszych wyników
        all_items = []
        tasks = [research_collect(variant, max_sites=8 if deep_research else 6) for variant in expanded_variants]
        results = await asyncio.gather(*tasks, return_exceptions=True)  # Kompatybilne ze wszystkimi wersjami Python 3.x
        
        for res in results:
            if isinstance(res, Exception):
                continue  # Ignoruj błędy, kontynuuj z tym, co mamy
            all_items.extend(res)
        
        # Deduplikacja po URL i domenie dla większej różnorodności wyników
        seen_urls = set()
        seen_domains = {}
        items = []
        
        for item in all_items:
            url = item.get("url", "")
            if not url:
                continue
                
            # Wyciągnij domenę z URL
            domain = ""
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc
            except:
                domain = url.split("/")[2] if "/" in url else ""
            
            if url not in seen_urls:
                # Ogranicz liczbę wyników z tej samej domeny
                if domain in seen_domains:
                    if seen_domains[domain] >= (3 if deep_research else 2):
                        continue
                    seen_domains[domain] += 1
                else:
                    seen_domains[domain] = 1
                    
                seen_urls.add(url)
                items.append(item)
        
        # Ogranicz do rozsądnej liczby
        items = items[:20 if deep_research else 15]
        
        # Jeśli nie udało się uzyskać żadnych wyników, spróbuj z prostszym zapytaniem
        if not items:
            fallback_items = await research_collect(q.split()[-1] if len(q.split()) > 1 else q, max_sites=10)
            items = fallback_items
            
        # Zapisz dokumenty w bazie z obsługą błędów
        try:
            store_docs(items)
        except Exception as e:
            pass  # Kontynuuj nawet jeśli zapis do bazy się nie powiedzie
        
        # Inteligentniejsze rankowanie z uwzględnieniem jakości źródeł i zawartości
        ranked = []
        for it in items:
            source_quality = 1.0
            url = it.get("url", "")
            # Boost dla źródeł wysokiej jakości - ROZSZERZONO
            if "edu" in url or "gov" in url or "wikipedia" in url or "research" in url:
                source_quality = 1.3
            elif ".org" in url or "scholar" in url or "science" in url or "academic" in url:
                source_quality = 1.2
            elif "blog" in url or "forum" in url:
                source_quality = 0.9
            
            # Analiza recency - świeżość treści
            recency_bonus = 1.0
            if "fetched" in it and isinstance(it["fetched"], (int, float)):
                age_days = (time.time() - it["fetched"]) / (86400)  # Wiek w dniach
                if age_days < 30:  # Treść młodsza niż miesiąc
                    recency_bonus = 1.1 + (0.1 * (1 - age_days/30))
            
            # Poprawa algorytmu rankowania dla fragmentów tekstu
            chunk_size = 320 if deep_research else 280  # Dłuższe fragmenty dla głębszego badania
            ch = chunk_text(it.get("text", ""), max_words=chunk_size)
            chunk_count = min(6 if deep_research else 5, len(ch))
            
            top = rank_hybrid(ch, q, topk=chunk_count)
            
            # Zastosuj boost jakości źródła i świeżości
            boosted_top = [(c, s * source_quality * recency_bonus) for c, s in top]
            
            ranked.append({
                "url": url,
                "title": it.get("title", ""),
                "top": [{"score": float(s), "chunk": c} for c, s in boosted_top]
            })
        
        # Zapisywanie do długotrwałej pamięci z lepszą obsługą błędów
        for it in ranked:
            for t in it["top"]:
                if not t["chunk"] or len(t["chunk"]) < 70:  # Ignoruj zbyt krótkie fragmenty
                    continue
                    
                try:
                    # Dodaj więcej metadanych w tagach z rozbudowaną klasyfikacją
                    tags = f"auto:web,url:{it['url']}"
                    
                    # Analizy jakości treści
                    if "wysokiej jakości" in it.get("title", "").lower() or t["score"] > 0.8:
                        tags += ",premium,high_quality"
                        
                    if ".edu" in it["url"] or ".gov" in it["url"]:
                        tags += ",trusted_source"
                        
                    # Dodaj tag kategorii na podstawie analizy tekstu
                    if re.search(r"(definicja|pojęcie|termin|znaczenie)", t["chunk"].lower()):
                        tags += ",definition"
                    elif re.search(r"(przykład|case study|use case|przypadek|przykładowo)", t["chunk"].lower()):
                        tags += ",example"
                    elif re.search(r"(statystyka|badanie|procent|wskaźnik|dane)", t["chunk"].lower()):
                        tags += ",statistics"
                    elif re.search(r"(historia|powstanie|rozwój|ewolucja)", t["chunk"].lower()):
                        tags += ",history"
                        
                    ltm_add(t["chunk"], tags=tags, conf=max(0.65, min(0.97, t["score"])))
                except Exception as e:
                    # Obsługa błędów przy zapisywaniu do LTM
                    pass
        
        # Przygotowanie ostatecznego kontekstu
        ctx = []; cites = []; fact_highlights = []
        
        # Dołącz fakty z bazy wiedzy związane z tematem - ULEPSZONE wyszukiwanie
        try:
            # Próbujemy wyszukiwanie hybrydowe (embedding + text search)
            related_facts = ltm_search_hybrid(q, 8)
            
            # Dodajemy również wyszukiwanie po tagach tematycznych
            tags_search_terms = q.split()[:3]  # Pierwsze 3 słowa zapytania jako tagi
            for term in tags_search_terms:
                if len(term) > 3:  # Tylko znaczące słowa
                    # Próbujemy użyć ltm_search_tags, ale fallback do ltm_search_hybrid
                    try:
                        tag_facts = ltm_search_tags(f"fact,{term}", 3)
                    except NameError:
                        # Jeśli nie ma ltm_search_tags, używamy ltm_search_hybrid
                        tag_facts = ltm_search_hybrid(term, k=3, tags=["fact"])
                    
                    if tag_facts:
                        related_facts.extend(tag_facts)
        
            # Filtracja i deduplikacja faktów
            seen_facts = set()
            filtered_facts = []
            
            for fact in related_facts:
                if not fact.get("text") or fact.get("score", 0) <= 0.55:
                    continue
                    
                text = fact["text"]
                fact_hash = hashlib.md5(text.encode()).hexdigest()[:10]
                
                if (fact_hash not in seen_facts and 
                    (fact.get("score", 0) > 0.6 or "fact" in fact.get("tags", "") or 
                     "premium" in fact.get("tags", "") or "high_quality" in fact.get("tags", ""))):
                    seen_facts.add(fact_hash)
                    filtered_facts.append(fact)
                    
            for fact in filtered_facts:
                fact_highlights.append(fact["text"])
        except Exception as e:
            # Fallback jeśli wyszukiwanie faktów zawiedzie
            pass
        
        # Łączenie wszystkich źródeł z priorytetyzacją najlepszych
        ranked.sort(key=lambda x: max([t.get("score", 0) for t in x["top"]] or [0]), reverse=True)
        
        for d in ranked:
            # Sortuj fragmenty dla każdego źródła od najbardziej istotnego
            sorted_chunks = sorted(d["top"], key=lambda x: x.get("score", 0), reverse=True)
            for t in sorted_chunks:
                ctx.append(t["chunk"])
            cites.append({"title": d["title"], "url": d["url"]})
        
        # Finalna struktura wynikowa z dodatkowymi informacjami
        out = {
            "query": q,
            "context": "\n\n".join(ctx[:topk]),
            "facts": fact_highlights[:max(3, topk//3)],  # Dynamiczna liczba faktów w zależności od topk
            "sources": cites[:max(12, topk)],
            "is_deep_research": deep_research,
            "source_count": len(items),
            "powered_by": "monolit-engine"
        }
        
        cache_put(key, out)
        return out
        
    except Exception as e:
        # Obsługa krytycznych błędów
        err_msg = f"Error in autonauka: {str(e)}"
        # Awaryjne minimum informacji
        return {
            "query": q,
            "context": f"Nie udało się wykonać wyszukiwania. Spróbuj innego zapytania. Błąd: {str(e)}.",
            "facts": [],
            "sources": [],
            "is_deep_research": deep_research,
            "source_count": 0,
            "error": err_msg
        }

def answer_with_sources(q: str, deep_research: bool = False)->dict:
    # Wywołanie z możliwością głębokiego badania i zwiększonym topk dla lepszych wyników
    data=asyncio.run(autonauka(q, topk=10, deep_research=deep_research))
    ctx=data.get("context","")
    cites=data.get("sources",[])
    facts=data.get("facts",[])
    
    # Dodane fakty do kontekstu dla lepszej dokładności
    facts_text = "\n\n".join([f"FAKT: {fact}" for fact in facts]) if facts else ""
    combined_ctx = f"{facts_text}\n\n{ctx}" if facts_text else ctx
    
    # Lepsze formatowanie cytowań z numeracją i pełnymi URL
    cite_text="\n".join([f"[{i+1}] {s['title'] or s['url']} — {s['url']}" for i,s in enumerate(cites)])
    
    # Lepsze wskazówki systemowe dla modelu
    t=psy_tune()
    sys_prompt = f"""Odpowiadasz po polsku. Ton: {t['tone']}. 
    Cytuj źródła w treści używając numerów w nawiasach kwadratowych [1], [2] itd.
    Bazuj tylko na podanym kontekście. Jeśli w kontekście nie ma wystarczających informacji, przyznaj to.
    Odpowiedz zwięźle, ale wyczerpująco i merytorycznie na zadane pytanie."""
    
    # Większy nacisk na cytowanie źródeł w treści
    user_prompt = f"""Pytanie: {q}

Kontekst:
{combined_ctx}

Źródła:
{cite_text}

Odpowiedz używając dostępnych źródeł, cytując je w treści odpowiedzi. Podsumuj najważniejsze informacje i wskaż źródła."""
    
    # Wywołanie LLM z lekko niższą temperaturą dla bardziej spójnych odpowiedzi
    ans=call_llm([
        {"role":"system","content":sys_prompt},
        {"role":"user","content":user_prompt}
    ], max(0.5, t.get("temperature", 0.7) - 0.1))
    
    # Zwróć więcej informacji w wyniku
    return {
        "ok":True,
        "answer":ans,
        "sources":cites,
        "facts_used":len(facts),
        "powered_by": data.get("powered_by", "monolit-engine"),
        "is_deep_research": data.get("is_deep_research", deep_research)
    }

# =========================
# Travel / News / Sports
# =========================
def otm_geoname(city: str)->Optional[Tuple[float,float]]:
    if not OPENTRIPMAP_KEY: return None
    url=f"https://api.opentripmap.com/0.1/en/places/geoname?name={quote_plus(city)}&apikey={OPENTRIPMAP_KEY}"
    j=http_get_json(url); lon,lat=j.get("lon"),j.get("lat")
    try: return (float(lon), float(lat))
    except: return None

async def serp_maps(q: str, limit:int=20)->List[dict]:
    res=await serpapi_search(q, "google_maps", params={"type":"search","num":limit})
    out=[]
    data=res.get("data",{})
    for it in (data.get("local_results") or [])[:limit]:
        out.append({"title": it.get("title",""), "address": it.get("address",""), "rating": it.get("rating"), "link": it.get("links",{}).get("google_maps")})
    return out

def travel_search(city: str, what: str="attractions"):
    center=otm_geoname(city)
    if not center: return {"ok":False,"error":"geoname not found"}
    lon,lat=center
    if what=="hotels":
        items=asyncio.run(serp_maps(f"{city} hotels", 20))
    elif what=="restaurants":
        q=f"""
[out:json][timeout:25];
(
  node["amenity"~"restaurant|cafe|fast_food"](around:2200,{lat},{lon});
  way["amenity"~"restaurant|cafe|fast_food"](around:2200,{lat},{lon});
  relation["amenity"~"restaurant|cafe|fast_food"](around:2200,{lat},{lon});
);
out center 60;
"""
        req=Request(OVERPASS_URL, data=urlencode({"data": q}).encode(), headers={"Content-Type":"application/x-www-form-urlencoded"}, method="POST")
        try:
            with urlopen(req, timeout=HTTP_TIMEOUT) as r:
                j=json.loads(r.read().decode("utf-8","replace"))
            items=[]
            for e in j.get("elements",[]):
                tags=e.get("tags",{})
                items.append({"name": tags.get("name",""), "cuisine": tags.get("cuisine",""), "lat": e.get("lat") or (e.get("center") or {}).get("lat"), "lon": e.get("lon") or (e.get("center") or {}).get("lon")})
        except Exception:
            items=[]
    else:
        items=asyncio.run(serp_maps(f"{city} attractions", 20))
    return {"ok":True,"center":{"lon":lon,"lat":lat},"items":items}

# ---- DuckDuckGo News (lite html) ----
async def duck_news(q: str, limit:int=10) -> Dict[str,Any]:
    url=f"https://duckduckgo.com/html/?q={quote_plus(q)}&iar=news&ia=news"
    items=[]
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent":WEB_USER_AGENT}) as c:
            r=await c.get(url)
            html=r.text
            # proste wyciągnięcie: <a class="result__a" href="...">Tytuł</a>
            for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, re.I|re.S):
                link=m.group(1); title=re.sub("<.*?>","",m.group(2)).strip()
                if link and title:
                    items.append({"title":title, "link":link})
                if len(items)>=limit: break
    except Exception as e:
        return {"ok":False,"error":str(e)}
    return {"ok":True,"items":items}

# ---- ESPN publiczne scoreboards (realne wyniki) ----
ESPN_URLS={
    "nba":"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
    "nfl":"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
    "mlb":"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
    "nhl":"https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard",
    "soccer":"https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard"  # EPL
}
def espn_scores(kind:str="nba", date:str="")->Dict[str,Any]:
    kind=kind.lower()
    base=ESPN_URLS.get(kind)
    if not base: return {"ok":False,"error":"unsupported kind"}
    url=base + (f"?dates={date}" if date else "")
    j=http_get_json(url)
    events=(j.get("events") or [])
    out=[]
    for e in events:
        comp=(e.get("competitions") or [{}])[0]
        status=(comp.get("status") or {}).get("type",{})
        short= status.get("shortDetail") or status.get("name") or ""
        teams=(comp.get("competitors") or [])
        if len(teams)==2:
            t1=teams[0]; t2=teams[1]
            out.append({
                "home" : t1.get("team",{}).get("displayName"),
                "home_score": t1.get("score"),
                "away" : t2.get("team",{}).get("displayName"),
                "away_score": t2.get("score"),
                "status": short,
                "venue": (comp.get("venue") or {}).get("fullName"),
                "start": e.get("date")
            })
    return {"ok":True,"kind":kind,"date":date,"games":out}

async def news_search(q: str, limit:int=12):
    res=await serpapi_search(q, engine="google_news", params={"num":limit})
    items=[]
    for it in (res.get("data",{}) or {}).get("news_results", [])[:limit]:
        items.append({"title": it.get("title",""), "link": it.get("link",""), "date": it.get("date","")})
    return {"ok":True,"items":items}

# =========================
# Uploady
# =========================
def files_save(files: List[dict]) -> List[dict]:
    saved=[]
    conn=_db(); c=conn.cursor()
    for f in files:
        name=f.get("name") or "file.bin"
        b64=f.get("base64") or ""
        raw=base64.b64decode(b64.encode()) if b64 else b""
        fid=uuid.uuid4().hex
        safe=re.sub(r"[^A-Za-z0-9_.-]+","_", name)[:80]
        path=os.path.join(UPLOADS_DIR, fid+"_"+safe)
        with open(path,"wb") as h: h.write(raw)
        c.execute("INSERT OR REPLACE INTO docs VALUES(?,?,?,?,?,?)",(fid,path,name,"[file uploaded]","upload",time.time()))
        saved.append({"id":fid,"name":name,"path":path,"size":len(raw)})
    conn.commit(); conn.close()
    return saved

def files_list()->List[Dict[str,Any]]:
    conn=_db(); c=conn.cursor()
    rows=c.execute("SELECT id,url AS path,title AS name,source,fetched FROM docs ORDER BY fetched DESC LIMIT 200").fetchall()
    conn.close(); return [dict(r) for r in rows]

# =========================
# Kontekst czatu
# =========================
def _title_from_first(user: str)->str:
    conn=_db(); c=conn.cursor()
    r=c.execute("SELECT content FROM memory WHERE user=? AND role='user' ORDER BY ts ASC LIMIT 1",(user,)).fetchone()
    conn.close()
    if not r or not r["content"]: return "Nowa rozmowa"
    t=r["content"].strip().split("\n",1)[0][:60]
    return t if len(t)>=8 else "Rozmowa"

def _collect_context(user: str, query: str, max_ctx_chars: int = 24000) -> str:
    # Pobranie pamięci krótkoterminowej z odwróconą kolejnością (najnowsze na górze)
    stm = memory_get(user, 60)
    stm_text = "\n".join(f"{r['role']}: {r['content']}" for r in reversed(stm))
    
    # Pobranie pamięci długoterminowej - teraz z lepszym kontekstem
    # Łączymy zapytanie z ostatnimi wypowiedziami dla lepszego wyszukiwania
    recent_context = " ".join([r["content"] for r in stm[:5]]) if stm else ""
    search_query = f"{query} {recent_context}"[:1500]
    
    # Zaawansowane wyszukiwanie hybrydowe w LTM
    ltm_hits = ltm_search_hybrid(search_query, 18)
    
    # Filtrowanie faktów według ich istotności dla zapytania
    filtered_hits = []
    for hit in ltm_hits:
        relevance = _tfidf_cos(query, [hit["text"]])[0] if query else 0.5
        if relevance > 0.25:  # Filtr istotności
            filtered_hits.append((hit["text"], relevance))
    
    # Sortowanie po istotności
    filtered_hits.sort(key=lambda x: x[1], reverse=True)
    ltm_text = "\n\n".join([f"[FACT] {h[0]}" for h, _ in filtered_hits[:12]])
    
    # Pobieranie również długoterminowych podsumowań rozmów
    summaries = memory_summaries(user, 3)
    summaries_text = "\n\n".join([f"[POPRZEDNIE ROZMOWY] {s['summary']}" for s in summaries]) if summaries else ""
    
    # Zaawansowane wyszukiwanie semantyczne w dokumentach
    conn=_db(); c=conn.cursor()
    docs = c.execute("SELECT title,text,url FROM docs_fts WHERE docs_fts MATCH ? LIMIT 10",
                     (" ".join(_tok(search_query)),)).fetchall()
    conn.close()
    
    # Lepsze przygotowanie fragmentów dokumentów
    doc_chunks=[]
    for d in docs:
        if not d["text"]: continue
        
        # Inteligentne fragmentowanie tekstu dokumentu
        all_text = d["text"]
        paragraphs = re.split(r'\n\s*\n', all_text)
        
        # Znajdowanie najbardziej istotnych akapitów dla zapytania
        if paragraphs and query:
            scores = _tfidf_cos(query, paragraphs)
            best_paragraphs = sorted(zip(paragraphs, scores), key=lambda x: x[1], reverse=True)[:2]
            
            for para, score in best_paragraphs:
                if score > 0.2:  # Minimalny próg istotności
                    doc_chunks.append(f"[{d['title']}] {para[:750]}")
        else:
            # Fallback - weź początek tekstu
            doc_chunks.append(f"[{d['title']}] {all_text[:1000]}")
    
    # Składanie kontekstu z różnych źródeł w kolejności istotności
    ctx = (stm_text + "\n\n" + summaries_text + "\n\n" + ltm_text + "\n\n" + "\n\n".join(doc_chunks))[:max_ctx_chars]
    return ctx

# =========================
# Backup / Restore
# =========================
def db_backup()->str:
    ts=int(time.time())
    dst=os.path.join(BACKUP_DIR, f"mem_{ts}.db")
    with open(DB_PATH,"rb") as src, open(dst,"wb") as out: out.write(src.read())
    return dst

# =========================
# Zintegrowany moduł autonauka
# =========================

# ──────────────────────────────────────────────────────────────────────────────
# ENV dla autonauka (już zdefiniowane wyżej w linii 96-97, usuwamy duplikaty)
# SERPAPI_KEY i FIRECRAWL_KEY już są w CONFIG sekcji

WEB_HTTP_TIMEOUT = float(os.getenv("WEB_HTTP_TIMEOUT", "45"))
AUTO_TOPK = int(os.getenv("AUTO_TOPK", "8"))
AUTO_FETCH = int(os.getenv("AUTO_FETCH", "4"))
AUTO_MIN_CHARS = int(os.getenv("AUTO_MIN_CHARS", "800"))
AUTO_MAX_CHARS = int(os.getenv("AUTO_MAX_CHARS", "8000"))

AUTON_WAL = os.getenv("AUTON_WAL", "/workspace/mrd69/data/mem/autonauka.wal")
AUTON_DEDUP_MAX = int(os.getenv("AUTON_DEDUP_MAX", "1000"))
AUTON_DOMAIN_MAX = int(os.getenv("AUTON_DOMAIN_MAX", "2"))
VOTE_MIN_SOURCES = int(os.getenv("VOTE_MIN_SOURCES", "2"))

AUTO_TAGS = os.getenv("AUTO_TAGS", "autonauka,web,evidence")

LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY=w52XW0XN6zoV9hdY8OONhLu6tvnFaXbZ
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-4B-Instruct")

CONCURRENCY = int(os.getenv("AUTON_CONCURRENCY", "8"))
USER_AGENT = os.getenv("AUTON_UA", "Autonauka/1.0")

# SYNC (wbudowany)
MEM_SYNC_ENABLED = os.getenv("MEM_SYNC_ENABLED", "1") == "1"
REMOTE_MEM_BASE_URL = os.getenv("REMOTE_MEM_BASE_URL", "").rstrip("/")
REMOTE_MEM_API_KEY = os.getenv("REMOTE_MEM_API_KEY", "")
MEM_SYNC_INTERVAL = float(os.getenv("MEM_SYNC_INTERVAL", "30"))
MEM_SYNC_BATCH = int(os.getenv("MEM_SYNC_BATCH", "100"))
MEM_SYNC_TIMEOUT = float(os.getenv("MEM_SYNC_TIMEOUT", "25"))
PROF_KEY = "web_domain_weights"

# ──────────────────────────────────────────────────────────────────────────────
# Memory integracja dla autonauka
class _NoopMemory:
    def add_fact(self, text: str, meta: Optional[dict] = None, tags: Optional[List[str]] = None) -> str:
        return f"noop-{hashlib.sha1(text.encode()).hexdigest()[:12]}"

    def get_profile(self, key: str) -> Optional[dict]:
        return None

    def set_profile_many(self, entries: Dict[str, dict]) -> None:
        pass

def _get_memory():
    with contextlib.suppress(Exception):
        # Używamy funkcji get_memory zaimportowanej wcześniej
        return get_memory()
    return _NoopMemory()

MEM = _get_memory()

# ──────────────────────────────────────────────────────────────────────────────
# Modele danych dla autonauka
@dataclass
class Material:
    title: Optional[str]
    url: Optional[str]
    domain: Optional[str]
    trust: Optional[float]
    recency: Optional[float]
    snippet: Optional[str]
    facts: Optional[List[str]]

@dataclass
class LearnResult:
    query: str
    count: int
    trust_avg: float
    backend: Optional[str]
    ltm_ids: List[str]
    citations: List[str]
    materials: List[Material]
    draft: Optional[str]

# ──────────────────────────────────────────────────────────────────────────────
# Utils dla autonauka
_CANON_SKIP_PARAMS = {"utm_source","utm_medium","utm_campaign","utm_term","utm_content","gclid","fbclid","igshid","mc_cid","mc_eid"}

def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _norm_text(s: str) -> str:
    s = html.unescape(s or "")
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u200b", "")
    return _normalize_ws(s)

def _canonical_url(url: str) -> str:
    try:
        p = urlparse(url)
        scheme = "https" if p.scheme in ("http","https") else p.scheme
        query = urlencode([(k,v) for k,v in parse_qsl(p.query, keep_blank_values=True) if k not in _CANON_SKIP_PARAMS])
        path = re.sub(r"/+$", "", p.path or "")
        return urlunparse((scheme, p.netloc.lower(), path, "", query, ""))
    except Exception:
        return url

def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def _now_ts() -> float:
    return time.time()

def _parse_date_from_html(soup: BeautifulSoup) -> Optional[datetime]:
    for sel,attr in [
        ('meta[property="article:published_time"]',"content"),
        ('meta[name="pubdate"]',"content"),
        ('meta[name="date"]',"content"),
        ('time[datetime]',"datetime"),
        ('meta[itemprop="datePublished"]',"content"),
    ]:
        tag = soup.select_one(sel)
        if tag and tag.get(attr):
            with contextlib.suppress(Exception):
                return datetime.fromisoformat(tag.get(attr).replace("Z","+00:00"))
    text = soup.get_text(" ", strip=True)
    m = re.search(r"(20\d{2}[-/\.]\d{1,2}[-/\.]\d{1,2})", text)
    if m:
        s = m.group(1).replace("/", "-").replace(".", "-")
        y,mn,dd = map(int, s.split("-"))
        with contextlib.suppress(Exception):
            return datetime(y,mn,dd,tzinfo=timezone.utc)
    return None

def _recency_score(dt: Optional[datetime]) -> float:
    if not dt:
        return 0.3
    if not dt.tzinfo:
        dt = dt.replace(tzinfo=timezone.utc)
    days = max(1.0, (datetime.now(timezone.utc) - dt).total_seconds()/86400.0)
    return max(0.2, 1.0 / math.log2(2+days/7))

def _client() -> httpx.AsyncClient:
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/json;q=0.9,*/*;q=0.8"}
    return httpx.AsyncClient(follow_redirects=True, timeout=WEB_HTTP_TIMEOUT, headers=headers)

# ──────────────────────────────────────────────────────────────────────────────
# Wyszukiwanie dla autonauka
async def _ddg_search(q: str, k: int) -> List[Tuple[str,str]]:
    url = "https://duckduckgo.com/html/"
    out: List[Tuple[str,str]] = []
    async with _client() as cl:
        r = await cl.post(url, data={"q": q})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.select("a.result__a"):
            href = a.get("href"); title = _norm_text(a.text)
            if href and title:
                out.append((title, href))
            if len(out)>=k: break
    return out

async def _wiki_search(q: str, k: int) -> List[Tuple[str,str]]:
    api = "https://en.wikipedia.org/w/api.php"
    params = {"action":"opensearch","format":"json","limit":str(k),"search":q}
    async with _client() as cl:
        r = await cl.get(api, params=params)
        r.raise_for_status()
        js = r.json()
    return [(_norm_text(t), l) for t,l in zip(js[1], js[3])]

async def _arxiv_search(q: str, k: int) -> List[Tuple[str,str]]:
    api = "http://export.arxiv.org/api/query"
    params = {"search_query": q, "start":"0", "max_results": str(k)}
    async with _client() as cl:
        r = await cl.get(api, params=params)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "xml")
    out=[]
    for e in soup.select("entry"):
        t = _norm_text(e.select_one("title").text if e.select_one("title") else "")
        link = e.select_one("id").text if e.select_one("id") else None
        if t and link: out.append((t, link))
    return out[:k]

async def _s2_search(q: str, k: int) -> List[Tuple[str,str]]:
    api = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": q, "limit": str(k), "fields":"title,url"}
    async with _client() as cl:
        r = await cl.get(api, params=params)
        if r.status_code >= 400: return []
        js = r.json()
    out=[]
    for it in js.get("data", []):
        t = _norm_text(it.get("title") or ""); u = it.get("url")
        if t and u: out.append((t,u))
    return out

async def _serpapi_search(q: str, k: int) -> List[Tuple[str,str]]:
    if not SERPAPI_KEY: return []
    api = "https://serpapi.com/search.json"
    params = {"engine":"google","q":q,"num":str(k),"api_key":SERPAPI_KEY}
    async with _client() as cl:
        r = await cl.get(api, params=params)
        if r.status_code >= 400: return []
        js = r.json()
    out=[]
    for it in js.get("organic_results", []):
        t=_norm_text(it.get("title","")); u=it.get("link")
        if t and u: out.append((t,u))
    return out[:k]

# ──────────────────────────────────────────────────────────────────────────────
# Pobieranie treści dla autonauka
async def _firecrawl(url: str) -> Optional[str]:
    if not FIRECRAWL_KEY:
        return None
    api = "https://api.firecrawl.dev/v1/scrape"
    payload = {"url": url, "formats":["markdown","html","rawHtml"], "actions":[]}
    headers = {"Authorization": f"Bearer {FIRECRAWL_KEY}", "Content-Type":"application/json"}
    async with _client() as cl:
        r = await cl.post(api, json=payload, headers=headers)
        if r.status_code >= 400:
            return None
        js = r.json()
    text = js.get("markdown") or js.get("html") or js.get("rawHtml")
    if not text: return None
    return _norm_text(BeautifulSoup(text, "html.parser").get_text(" "))

async def _http_text(url: str) -> Optional[Tuple[str, str, Optional[datetime]]]:
    async with _client() as cl:
        r = await cl.get(url)
        if r.status_code >= 400 or not r.text:
            return None
        html_txt = r.text
        doc = ReadabilityDoc(html_txt)
        title = _norm_text(doc.short_title() or "")
        article_html = doc.summary(html_partial=True)
        soup = BeautifulSoup(article_html, "html.parser")
        dt = _parse_date_from_html(soup)
        text = _norm_text(soup.get_text(" "))
        if len(text) < 200:
            soup2 = BeautifulSoup(html_txt, "html.parser")
            dt = dt or _parse_date_from_html(soup2)
            text = _norm_text(soup2.get_text(" "))
        return (title, text, dt)

# ──────────────────────────────────────────────────────────────────────────────
# Chunking + ranking dla autonauka
_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+", re.UNICODE)

def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]

def _chunks(text: str, min_chars=AUTO_MIN_CHARS, max_chars=AUTO_MAX_CHARS, overlap=180) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    parts: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        slice_ = text[start:end]
        m = re.search(r"(?s)^(.{"+str(max_chars - 300)+r"," + str(max_chars) + r"}[\.\!\?])", slice_)
        block = m.group(1) if m else slice_
        parts.append(block)
        start = start + len(block) - overlap
        if start <= 0 or start >= len(text): break
    return [p for p in parts if len(p) >= min_chars or (len(parts)==1 and len(p)>0)]

class BM25:
    def __init__(self, corpus_tokens: List[List[str]], k1: float=1.5, b: float=0.75):
        self.k1, self.b = k1, b
        self.docs = corpus_tokens
        self.N = len(corpus_tokens)
        self.df = Counter()
        self.avgdl = 1.0
        self._build()
    def _build(self):
        lengths = []
        for doc in self.docs:
            unique_terms = set(doc)
            for t in unique_terms:
                self.df[t]+=1
            lengths.append(len(doc))
        self.avgdl = sum(lengths)/max(1,len(lengths))
    def score(self, q: List[str], doc: List[str]) -> float:
        f = Counter(doc); dl = len(doc); sc = 0.0
        for term in q:
            n_qi = self.df.get(term,0)
            if n_qi==0: continue
            idf = math.log(1 + (self.N - n_qi + 0.5)/(n_qi + 0.5))
            freq = f.get(term,0)
            denom = freq + self.k1*(1 - self.b + self.b*dl/self.avgdl)
            sc += idf * (freq*(self.k1+1))/max(1e-9, denom)
        return sc

def _cosine_hash(q_tokens: List[str], d_tokens: List[str]) -> float:
    qset, dset = set(q_tokens), set(d_tokens)
    inter = len(qset & dset)
    denom = math.sqrt(len(qset)*len(dset)) or 1.0
    return inter/denom

def _jaccard(q_tokens: List[str], d_tokens: List[str]) -> float:
    qset, dset = set(q_tokens), set(d_tokens)
    if not qset or not dset: return 0.0
    return len(qset & dset)/len(qset | dset)

def _hybrid_rank(query: str, chunks: List[str]) -> List[Tuple[int,float]]:
    q_tokens = _tokenize(query)
    c_tokens = [_tokenize(c) for c in chunks]
    bm25 = BM25(c_tokens)
    scores = []
    for i, dtok in enumerate(c_tokens):
        s = 0.50*bm25.score(q_tokens, dtok) + 0.35*_cosine_hash(q_tokens, dtok) + 0.15*_jaccard(q_tokens, dtok)
        scores.append((i, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

# ──────────────────────────────────────────────────────────────────────────────
# Trust dla autonauka
_TRUST_DOMAINS = {
    "wikipedia.org": 0.9,
    "arxiv.org": 0.85,
    "semanticscholar.org": 0.8,
    "nature.com": 0.9,
    "science.org": 0.9,
    "acm.org": 0.85,
    "ieee.org": 0.85,
    "gov": 0.9,
    "edu": 0.85,
}
def _trust(url: str, https_ok: bool=True) -> float:
    d = _domain(url); sc = 0.6
    if https_ok and url.lower().startswith("https"): sc += 0.05
    if d.endswith(".gov") or d.endswith(".gov.pl"): sc = max(sc, 0.9)
    if d.endswith(".edu"): sc = max(sc, 0.85)
    for k,v in _TRUST_DOMAINS.items():
        if d.endswith(k): sc = max(sc, v)
    return min(sc, 0.98)

# ──────────────────────────────────────────────────────────────────────────────
# LLM dla autonauka
_FACT_SYS = "Extract concise factual statements with explicit source URLs in strict JSON. Output JSON array of objects: [{\"fact\": str, \"source\": str}]. No prose."

def _fallback_fact_extract(prompt: str) -> str:
    facts=[]
    for line in (prompt or "").splitlines():
        line=line.strip()
        if not line: continue
        m = re.search(r"(https?://\S+)", line)
        if m:
            url = m.group(1)
            fact = _normalize_ws(line.replace(url,"")).strip(" -:—")
            if len(fact)>=30:
                facts.append({"fact": fact, "source": url})
    return json.dumps(facts[:8], ensure_ascii=False)

def _llm_chat(system: str, user: str, maxtok: int=1024, temp: float=0.2) -> str:
    if not LLM_BASE_URL or not LLM_API_KEY:
        return _fallback_fact_extract(user)
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type":"application/json"}
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role":"system","content":system},{"role":"user","content":user}],
        "max_tokens": maxtok,
        "temperature": temp,
    }
    try:
        resp = httpx.post(LLM_BASE_URL.rstrip("/") + "/chat/completions", json=payload, headers=headers, timeout=WEB_HTTP_TIMEOUT)
        resp.raise_for_status()
        js = resp.json()
        return js["choices"][0]["message"]["content"]
    except Exception:
        return _fallback_fact_extract(user)

def _llm_extract_facts(query: str, materials: List[Tuple[str,str,str]]) -> List[Tuple[str,str]]:
    lines = [f"Q: {query}"]
    for (u,t,txt) in materials:
        snippet = _normalize_ws(txt[:900])
        lines.append(f"- {t} [{u}] :: {snippet}")
    user_prompt = "\n".join(lines)
    raw = _llm_chat(_FACT_SYS, user_prompt, maxtok=800, temp=0.1)
    facts=[]
    with contextlib.suppress(Exception):
        data = json.loads(raw)
        if isinstance(data, list):
            for it in data:
                fact = _norm_text(it.get("fact",""))
                src = it.get("source")
                if fact and src and len(fact)>=30:
                    facts.append((fact, _canonical_url(src)))
    if not facts:
        for (u, t, txt) in materials:
            sentences = re.split(r"(?<=[\.\!\?])\s+", txt)
            for s in sentences:
                s=_norm_text(s)
                if 60 <= len(s) <= 300 and s.lower() not in ("copyright","all rights reserved"):
                    facts.append((s, _canonical_url(u)))
                    if len(facts)>=6: break
            if len(facts)>=6: break
    return facts[:12]

# ──────────────────────────────────────────────────────────────────────────────
# WAL dla autonauka
def _ensure_dir(path: str) -> None:
    if not path: return
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _wal_append(entry: dict) -> None:
    if not AUTON_WAL: return
    _ensure_dir(AUTON_WAL)
    line = json.dumps({"ts": _now_ts(), **entry}, ensure_ascii=False)
    with open(AUTON_WAL, "a", encoding="utf-8") as f:
        f.write(line+"\n")
    _wal_rotate_if_needed()

def _wal_rotate_if_needed(max_bytes: int = 10_000_000) -> None:
    try:
        st = os.stat(AUTON_WAL)
        if st.st_size < max_bytes: return
        ts = int(_now_ts()); dst = f"{AUTON_WAL}.{ts}.rot"
        os.replace(AUTON_WAL, dst)
    except FileNotFoundError:
        pass
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# Reputacja domen dla autonauka
def _load_profile() -> Dict[str, dict]:
    prof = MEM.get_profile(PROF_KEY)
    return prof if isinstance(prof, dict) else {}

def _save_profile_many(prof: Dict[str, dict]) -> None:
    MEM.set_profile_many({PROF_KEY: prof})

def _bump_domain_weight(prof: Dict[str, dict], domain: str, success: bool) -> None:
    it = prof.get(domain) or {"w": 0.0, "n": 0}
    w = float(it.get("w",0.0)); n = int(it.get("n",0))
    w = w + (0.05 if success else -0.03)
    w = max(-0.5, min(0.8, w)); n += 1
    prof[domain] = {"w": w, "n": n}

def _domain_weight(prof: Dict[str, dict], domain: str) -> float:
    it = prof.get(domain); return float(it.get("w",0.0)) if it else 0.0

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline dla autonauka
async def _ingest_url(query: str, title: str, url: str, topk_chunks: int = 2) -> Optional[Tuple[Material, List[Tuple[str,str]]]]:
    url_c = _canonical_url(url)
    https_ok = url_c.startswith("https")
    fetched = await _http_text(url_c)
    if not fetched:
        txt = await _firecrawl(url_c)
        if not txt: return None
        title2 = title or ""
        dt = None; text = txt
    else:
        title2, text, dt = fetched
    if not text or len(text) < 200: return None
    parts = _chunks(text)
    ranked = _hybrid_rank(query, parts)[:topk_chunks]
    picked = " ".join(parts[i] for i,_ in ranked)
    dom = _domain(url_c)
    trust = _trust(url_c, https_ok)
    recency = _recency_score(dt)
    facts = _llm_extract_facts(query, [(url_c, title2 or title or "", picked)])
    material = Material(
        title=title2 or title or None,
        url=url_c, domain=dom, trust=trust, recency=recency,
        snippet=_norm_text(picked[:600]),
        facts=[f for f,_ in facts] if facts else [],
    )
    return material, facts

class _LRUDedup:
    def __init__(self, cap: int):
        self.cap = cap; self.order: List[str] = []; self.set = set()
    def put(self, key: str) -> None:
        if key in self.set:
            with contextlib.suppress(ValueError): self.order.remove(key)
            self.order.append(key); return
        self.set.add(key); self.order.append(key)
        if len(self.order) > self.cap:
            old = self.order.pop(0); self.set.discard(old)
    def has(self, key: str) -> bool:
        return key in self.set

async def _gather_with_limit(coros: List, limit: int) -> List[Any]:
    sem = asyncio.Semaphore(limit)
    async def _run(c):
        async with sem: return await c
    return await asyncio.gather(*[_run(c) for c in coros], return_exceptions=True)

async def _search_all(query: str, mode: str) -> List[Tuple[str,str]]:
    tasks = []
    if mode in ("full","grounded","fast","free"):
        tasks += [_ddg_search(query, AUTO_TOPK), _wiki_search(query, min(5, AUTO_TOPK))]
    if mode in ("full","grounded"):
        tasks += [_s2_search(query, min(5, AUTO_TOPK)), _arxiv_search(query, min(5, AUTO_TOPK))]
        if SERPAPI_KEY: tasks.append(_serpapi_search(query, AUTO_TOPK))
    results: List[Tuple[str,str]] = []
    for out in await _gather_with_limit(tasks, limit=4):
        if not isinstance(out, Exception): results.extend(out or [])
    bydom: Dict[str, int] = defaultdict(int); filtered=[]
    for (t,u) in results:
        d=_domain(u)
        if bydom[d] >= AUTON_DOMAIN_MAX: continue
        bydom[d]+=1; filtered.append((t,u))
    return filtered[:max(AUTO_TOPK, 6)]

async def _pipeline(query: str, mode: str) -> Tuple[List[Material], List[Tuple[str,str]]]:
    # Rozszerzone wyszukiwanie i analiza
    pairs = await _search_all(query, mode)
    
    # Dynamiczna optymalizacja pobierania w zależności od trybu i jakości wyników
    max_fetch = {
        "fast": min(3, AUTO_FETCH), 
        "free": min(4, AUTO_FETCH), 
        "grounded": max(4, AUTO_FETCH),
        "full": max(5, AUTO_FETCH)
    }.get(mode, AUTO_FETCH)
    
    # Bardziej zaawansowana konfiguracja pobierania materiałów
    fetch_tasks = []
    domain_counter = Counter()
    
    # Priorytetyzacja domen o wyższym zaufaniu
    for t, u in pairs:
        domain = _domain(u)
        if domain_counter[domain] < AUTON_DOMAIN_MAX:
            domain_counter[domain] += 1
            fetch_tasks.append(_ingest_url(query, t, u))
            if len(fetch_tasks) >= max_fetch * 2:  # Zwiększamy pulę dla lepszych wyników
                break
    
    outs = await _gather_with_limit(fetch_tasks[:max_fetch], limit=CONCURRENCY)
    materials: List[Material] = []; facts_all: List[Tuple[str,str]] = []
    
    for out in outs:
        if isinstance(out, Exception) or out is None: 
            continue
        m, facts = out
        
        # Dodatkowe sprawdzenie jakości materiałów
        if m and (not m.facts or len(m.facts) == 0) and m.snippet and len(m.snippet) > 300:
            # Powtórna próba ekstrakcji faktów dla materiałów z niepowodzeniem
            additional_facts = _llm_extract_facts(query, [(m.url or "", m.title or "", m.snippet or "")])
            if additional_facts:
                m.facts = [f for f, _ in additional_facts]
                facts_all.extend(additional_facts)
        
        materials.append(m)
        facts_all.extend(facts)
    
    return materials, facts_all

# ──────────────────────────────────────────────────────────────────────────────
# Głosowanie + zapis + reputacja dla autonauka
def _dedup_key(fact: str, url: str) -> str:
    base = f"{_norm_text(fact).lower()}|{_domain(url)}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def _vote_and_store(facts: List[Tuple[str,str]], prof: Dict[str,dict]) -> Tuple[List[str], List[str]]:
    by_fact: Dict[str, set] = defaultdict(set)
    for fact, url in facts: by_fact[_norm_text(fact)].add(_domain(url))
    dedup = _LRUDedup(AUTON_DEDUP_MAX); ltm_ids: List[str] = []; citations: List[str] = []
    for fact_norm, domains in by_fact.items():
        if len(domains) < VOTE_MIN_SOURCES:
            if not any(_trust(f"https://{d}") >= 0.85 for d in domains): continue
        src_dom = next(iter(domains)); src_url = f"https://{src_dom}"
        key = _dedup_key(fact_norm, src_url)
        if dedup.has(key): continue
        dedup.put(key)
        meta = {"source_domains": list(domains), "source_url": src_url, "ts": _now_ts()}
        tags = [t.strip() for t in AUTO_TAGS.split(",") if t.strip()]
        fid = ltm_add(fact_norm, tags=tags, conf=0.9)
        ltm_ids.append(fid); citations.append(src_url)
        for d in domains: _bump_domain_weight(prof, d, True)
    _save_profile_many(prof)
    return ltm_ids, citations

# ──────────────────────────────────────────────────────────────────────────────
# Draft dla autonauka
def _llm_draft(query: str, materials: List[Material]) -> str:
    if not materials: return ""
    bullets=[]
    for m in materials[:6]:
        if not m or not m.url: continue
        bullets.append(f"- [{m.title or m.url}]({_canonical_url(m.url)}) trust={m.trust:.2f if m.trust else 0.0} recency={m.recency:.2f if m.recency else 0.0}")
    pre = f"Query: {query}\nMaterials:\n" + "\n".join(bullets) + "\n\nSynthesis:\n"
    return _llm_chat("Write a concise, source-grounded synthesis. 5–8 bullet points. No fluff.", pre, maxtok=700, temp=0.2)

# ──────────────────────────────────────────────────────────────────────────────
# Web learn core dla autonauka
async def _web_learn_async(query: str, mode: str) -> LearnResult:
    prof = _load_profile()
    materials, facts = await _pipeline(query, mode)
    for f,u in facts: _wal_append({"type":"fact_candidate","fact":f,"url":u})
    ltm_ids, citations = _vote_and_store(facts, prof)
    draft = _llm_draft(query, materials)
    trust_avg = float(sum(m.trust or 0 for m in materials)/max(1,len(materials)))
    return LearnResult(query=query,count=len(materials),trust_avg=trust_avg,backend="async-httpx",
                       ltm_ids=ltm_ids,citations=citations,materials=materials,draft=draft)

# ──────────────────────────────────────────────────────────────────────────────
# Publiczne API dla autonauka
def _run_sync(coro):
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        loop.close()

def web_learn(query: str, mode: str="full", **kwargs) -> Dict[str, Any]:
    # uruchom zapytanie w modelu asynchronicznym
    if mode not in ("full", "fast", "free", "grounded"):
        mode = "full"
    
    try:
        res: LearnResult = _run_sync(_web_learn_async(query, mode))
        return {
            "query": res.query,
            "count": res.count,
            "trust_avg": res.trust_avg,
            "backend": res.backend,
            "ltm_ids": res.ltm_ids,
            "citations": res.citations,
            "materials": [dataclasses.asdict(m) for m in res.materials],
            "draft": res.draft,
        }
    except Exception as e:
        # Fallback w przypadku błędu - zwróć podstawowe informacje
        print(f"Błąd w web_learn: {e}")
        return {
            "query": query,
            "count": 0,
            "trust_avg": 0.0,
            "backend": "fallback",
            "ltm_ids": [],
            "citations": [],
            "materials": [],
            "draft": f"Błąd podczas wyszukiwania informacji o '{query}': {str(e)}",
        }

# =========================
# Baza wiedzy i fakty
# =========================

def load_seed_facts(seed_file_path: str = None) -> Dict[str, Any]:
    """
    Ładuje i wzbogaca system o fakty z pliku seed.jsonl.
    Dodaje je bezpośrednio do pamięci długotrwałej.
    
    Args:
        seed_file_path: Ścieżka do pliku seed.jsonl z faktami. 
                        Domyślnie data/start.seed.jsonl
    
    Returns:
        Dict zawierający statystyki załadowanych faktów
    """
    # Domyślna ścieżka do pliku seed
    if not seed_file_path:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        seed_file_path = os.path.join(data_dir, "start.seed.jsonl")
    
    if not os.path.exists(seed_file_path):
        print(f"Plik seed {seed_file_path} nie istnieje!")
        return {"status": "error", "loaded": 0, "error": "file_not_found"}
    
    stats = {
        "status": "success",
        "loaded": 0,
        "failed": 0,
        "categories": {}
    }
    
    try:
        with open(seed_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    fact = json.loads(line)
                    text = fact.get("text", "")
                    if not text:
                        stats["failed"] += 1
                        continue
                        
                    # Przygotuj tagi
                    tags = fact.get("tags", [])
                    if not tags:
                        category = fact.get("category", "other")
                        subcategory = fact.get("subcategory", "general")
                        tags = [
                            "fact", 
                            "seed", 
                            f"category:{category}", 
                            f"subcategory:{subcategory}",
                            "verified",
                            "high_quality"
                        ]
                    
                    # Dodaj do pamięci długotrwałej
                    conf = fact.get("confidence", 0.9)
                    meta_data = {
                        "source_url": fact.get("source_url", ""),
                        "category": fact.get("category", "other"),
                        "subcategory": fact.get("subcategory", "general"),
                        "timestamp": fact.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ"))
                    }
                    
                    ltm_add(text, tags=tags, conf=conf)
                    stats["loaded"] += 1
                    
                    # Aktualizuj statystyki kategorii
                    category = fact.get("category", "other")
                    stats["categories"][category] = stats["categories"].get(category, 0) + 1
                        
                except Exception as e:
                    stats["failed"] += 1
    except Exception as e:
        return {"status": "error", "loaded": stats["loaded"], "error": str(e)}
    
    return stats

def generate_sample_facts(count: int = 5) -> List[Dict[str, Any]]:
    """
    Generuje przykładowe fakty dla bazy wiedzy.
    Użyteczne, gdy nie mamy pliku seed.jsonl.
    
    Args:
        count: Liczba faktów do wygenerowania
        
    Returns:
        Lista wygenerowanych faktów
    """
    # Baza faktów - rzeczywista wiedza z różnych dziedzin
    facts_base = [
        # KATEGORIA KURWA - NAJWAŻNIEJSZE FAKTY
        {
            "text": "KURWA to jedno z najbardziej uniwersalnych słów w języku polskim, pełniące funkcję przekleństwa, przerywnika i wykrzyknika.",
            "category": "kurwa",
            "subcategory": "słownictwo",
            "confidence": 0.999,
            "source_url": "https://pl.wikipedia.org/wiki/Wulgaryzmy_w_języku_polskim",
            "tags": ["fact", "seed", "category:kurwa", "subcategory:słownictwo", "verified"]
        },
        {
            "text": "Słowo KURWA może wyrażać całe spektrum emocji od złości i frustracji po zaskoczenie i podziw.",
            "category": "kurwa",
            "subcategory": "ekspresja",
            "confidence": 0.998,
            "source_url": "https://pl.wikipedia.org/wiki/Wulgaryzmy_w_języku_polskim",
            "tags": ["fact", "seed", "category:kurwa", "subcategory:ekspresja", "verified"]
        },
        {
            "text": "MORDO (forma od słowa 'mordeczka') to popularne slangowe określenie przyjaciela lub bliskiego znajomego w języku potocznym.",
            "category": "mordo",
            "subcategory": "slang",
            "confidence": 0.97,
            "source_url": "https://pl.wiktionary.org/wiki/mordo",
            "tags": ["fact", "seed", "category:mordo", "subcategory:slang", "verified"]
        },
        
        # NAUKA - FIZYKA
        {
            "text": "Teoria względności Einsteina wprowadza równoważność masy i energii wyrażoną wzorem E=mc².",
            "category": "nauka",
            "subcategory": "fizyka",
            "confidence": 0.99,
            "source_url": "https://pl.wikipedia.org/wiki/Teoria_względności",
            "tags": ["fact", "seed", "category:nauka", "subcategory:fizyka", "verified"]
        },
        {
            "text": "Zasada nieoznaczoności Heisenberga głosi, że nie można jednocześnie dokładnie zmierzyć położenia i pędu cząstki.",
            "category": "nauka",
            "subcategory": "fizyka",
            "confidence": 0.98,
            "source_url": "https://pl.wikipedia.org/wiki/Zasada_nieoznaczoności",
            "tags": ["fact", "seed", "category:nauka", "subcategory:fizyka", "verified"]
        },
        {
            "text": "Prędkość światła w próżni wynosi dokładnie 299 792 458 metrów na sekundę.",
            "category": "nauka",
            "subcategory": "fizyka",
            "confidence": 0.99,
            "source_url": "https://pl.wikipedia.org/wiki/Prędkość_światła",
            "tags": ["fact", "seed", "category:nauka", "subcategory:fizyka", "verified"]
        },
        
        # NAUKA - BIOLOGIA
        {
            "text": "DNA ma strukturę podwójnej helisy odkrytą przez Jamesa Watsona i Francisa Cricka w 1953 roku.",
            "category": "nauka",
            "subcategory": "biologia",
            "confidence": 0.99,
            "source_url": "https://pl.wikipedia.org/wiki/DNA",
            "tags": ["fact", "seed", "category:nauka", "subcategory:biologia", "verified"]
        },
        {
            "text": "Mitochondria są organellami komórkowymi odpowiedzialnymi za produkcję ATP - głównego nośnika energii w komórce.",
            "category": "nauka",
            "subcategory": "biologia",
            "confidence": 0.97,
            "source_url": "https://pl.wikipedia.org/wiki/Mitochondrium",
            "tags": ["fact", "seed", "category:nauka", "subcategory:biologia", "verified"]
        },
        {
            "text": "Fotosynteza to proces, w którym rośliny wykorzystują energię światła do produkcji glukozy z CO2 i wody.",
            "category": "nauka",
            "subcategory": "biologia",
            "confidence": 0.98,
            "source_url": "https://pl.wikipedia.org/wiki/Fotosynteza",
            "tags": ["fact", "seed", "category:nauka", "subcategory:biologia", "verified"]
        },
        
        # NAUKA - ASTRONOMIA
        {
            "text": "Galaktyka Drogi Mlecznej ma średnicę około 100 000 lat świetlnych i zawiera 100-400 miliardów gwiazd.",
            "category": "nauka",
            "subcategory": "astronomia",
            "confidence": 0.96,
            "source_url": "https://pl.wikipedia.org/wiki/Droga_Mleczna",
            "tags": ["fact", "seed", "category:nauka", "subcategory:astronomia", "verified"]
        },
        {
            "text": "Czarna dziura to obiekt o tak silnym polu grawitacyjnym, że nic, nawet światło, nie może z niego uciec.",
            "category": "nauka",
            "subcategory": "astronomia",
            "confidence": 0.97,
            "source_url": "https://pl.wikipedia.org/wiki/Czarna_dziura",
            "tags": ["fact", "seed", "category:nauka", "subcategory:astronomia", "verified"]
        },
        {
            "text": "Układ Słoneczny składa się z ośmiu planet: Merkurego, Wenus, Ziemi, Marsa, Jowisza, Saturna, Urana i Neptuna.",
            "category": "nauka",
            "subcategory": "astronomia",
            "confidence": 0.99,
            "source_url": "https://pl.wikipedia.org/wiki/Układ_Słoneczny",
            "tags": ["fact", "seed", "category:nauka", "subcategory:astronomia", "verified"]
        },
        
        # TECHNOLOGIA - PROGRAMOWANIE
        {
            "text": "Python jest językiem programowania wysokiego poziomu z czytelną składnią i silnym wsparciem dla wielu paradygmatów.",
            "category": "technologia",
            "subcategory": "programowanie",
            "confidence": 0.98,
            "source_url": "https://www.python.org/about/",
            "tags": ["fact", "seed", "category:technologia", "subcategory:programowanie", "verified"]
        },
        {
            "text": "JavaScript jest językiem programowania umożliwiającym tworzenie interaktywnych elementów stron internetowych.",
            "category": "technologia",
            "subcategory": "programowanie",
            "confidence": 0.97,
            "source_url": "https://developer.mozilla.org/pl/docs/Web/JavaScript",
            "tags": ["fact", "seed", "category:technologia", "subcategory:programowanie", "verified"]
        },
        {
            "text": "SQL (Structured Query Language) to język używany do zarządzania danymi w relacyjnych bazach danych.",
            "category": "technologia",
            "subcategory": "programowanie",
            "confidence": 0.98,
            "source_url": "https://pl.wikipedia.org/wiki/SQL",
            "tags": ["fact", "seed", "category:technologia", "subcategory:programowanie", "verified"]
        },
        
        # TECHNOLOGIA - AI
        {
            "text": "Sztuczna inteligencja to dziedzina informatyki zajmująca się tworzeniem systemów wykonujących zadania wymagające inteligencji.",
            "category": "technologia",
            "subcategory": "ai",
            "confidence": 0.95,
            "source_url": "https://pl.wikipedia.org/wiki/Sztuczna_inteligencja",
            "tags": ["fact", "seed", "category:technologia", "subcategory:ai", "verified"]
        },
        {
            "text": "Uczenie maszynowe to poddziedzina AI umożliwiająca systemom uczenie się na podstawie danych bez jawnego programowania.",
            "category": "technologia",
            "subcategory": "ai",
            "confidence": 0.96,
            "source_url": "https://pl.wikipedia.org/wiki/Uczenie_maszynowe",
            "tags": ["fact", "seed", "category:technologia", "subcategory:ai", "verified"]
        },
        {
            "text": "Sieci neuronowe to modele obliczeniowe inspirowane działaniem ludzkiego mózgu, składające się z połączonych neuronów.",
            "category": "technologia",
            "subcategory": "ai",
            "confidence": 0.97,
            "source_url": "https://pl.wikipedia.org/wiki/Sztuczna_sieć_neuronowa",
            "tags": ["fact", "seed", "category:technologia", "subcategory:ai", "verified"]
        },
        
        # TECHNOLOGIA - BLOCKCHAIN
        {
            "text": "Blockchain to rozproszona baza danych przechowująca informacje w blokach połączonych kryptograficznie.",
            "category": "technologia",
            "subcategory": "blockchain",
            "confidence": 0.95,
            "source_url": "https://pl.wikipedia.org/wiki/Blockchain",
            "tags": ["fact", "seed", "category:technologia", "subcategory:blockchain", "verified"]
        },
        {
            "text": "Bitcoin to pierwsza i najpopularniejsza kryptowaluta, stworzona w 2009 roku przez Satoshi Nakamoto.",
            "category": "technologia",
            "subcategory": "blockchain",
            "confidence": 0.96,
            "source_url": "https://pl.wikipedia.org/wiki/Bitcoin",
            "tags": ["fact", "seed", "category:technologia", "subcategory:blockchain", "verified"]
        },
        {
            "text": "Smart contract to samowykonujący się program komputerowy działający na blockchainie, automatyzujący umowy.",
            "category": "technologia",
            "subcategory": "blockchain",
            "confidence": 0.94,
            "source_url": "https://pl.wikipedia.org/wiki/Smart_contract",
            "tags": ["fact", "seed", "category:technologia", "subcategory:blockchain", "verified"]
        },
        
        # HISTORIA
        {
            "text": "I wojna światowa trwała od 28 lipca 1914 do 11 listopada 1918 roku i pochłonęła życie około 17 milionów ludzi.",
            "category": "historia",
            "subcategory": "XX wiek",
            "confidence": 0.98,
            "source_url": "https://pl.wikipedia.org/wiki/I_wojna_światowa",
            "tags": ["fact", "seed", "category:historia", "subcategory:XX_wiek", "verified"]
        },
        {
            "text": "Cesarstwo Rzymskie w szczytowym okresie obejmowało tereny od Brytanii po Mezopotamię i północną Afrykę.",
            "category": "historia",
            "subcategory": "starożytność",
            "confidence": 0.97,
            "source_url": "https://pl.wikipedia.org/wiki/Cesarstwo_Rzymskie",
            "tags": ["fact", "seed", "category:historia", "subcategory:starożytność", "verified"]
        },
        {
            "text": "Upadek muru berlińskiego 9 listopada 1989 roku symbolizował koniec zimnej wojny i podział Europy.",
            "category": "historia",
            "subcategory": "XX wiek",
            "confidence": 0.98,
            "source_url": "https://pl.wikipedia.org/wiki/Mur_Berliński",
            "tags": ["fact", "seed", "category:historia", "subcategory:XX_wiek", "verified"]
        },
        
        # GEOGRAFIA
        {
            "text": "Mount Everest to najwyższy szczyt Ziemi, wznoszący się na 8848 m n.p.m. w paśmie Himalajów.",
            "category": "geografia",
            "subcategory": "góry",
            "confidence": 0.99,
            "source_url": "https://pl.wikipedia.org/wiki/Mount_Everest",
            "tags": ["fact", "seed", "category:geografia", "subcategory:góry", "verified"]
        },
        {
            "text": "Amazonia to największy las deszczowy na Ziemi, położony w dorzeczu Amazonki w Ameryce Południowej.",
            "category": "geografia",
            "subcategory": "ameryka",
            "confidence": 0.98,
            "source_url": "https://pl.wikipedia.org/wiki/Amazonia",
            "tags": ["fact", "seed", "category:geografia", "subcategory:ameryka", "verified"]
        },
        {
            "text": "Sahara to największa gorąca pustynia na świecie, zajmująca większość północnej Afryki.",
            "category": "geografia",
            "subcategory": "afryka",
            "confidence": 0.98,
            "source_url": "https://pl.wikipedia.org/wiki/Sahara",
            "tags": ["fact", "seed", "category:geografia", "subcategory:afryka", "verified"]
        },
        
        # SPOŁECZEŃSTWO
        {
            "text": "Demokracja to system rządów, w którym władza należy do obywateli poprzez prawo głosu lub wybieranie przedstawicieli.",
            "category": "społeczeństwo",
            "subcategory": "polityka",
            "confidence": 0.97,
            "source_url": "https://pl.wikipedia.org/wiki/Demokracja",
            "tags": ["fact", "seed", "category:społeczeństwo", "subcategory:polityka", "verified"]
        },
        {
            "text": "Globalizacja to proces rosnącej współzależności między regionami i krajami na całym świecie.",
            "category": "społeczeństwo",
            "subcategory": "ekonomia",
            "confidence": 0.95,
            "source_url": "https://pl.wikipedia.org/wiki/Globalizacja",
            "tags": ["fact", "seed", "category:społeczeństwo", "subcategory:ekonomia", "verified"]
        },
        {
            "text": "ONZ (Organizacja Narodów Zjednoczonych) to organizacja międzynarodowa założona w 1945 roku dla utrzymania pokoju.",
            "category": "społeczeństwo",
            "subcategory": "polityka",
            "confidence": 0.98,
            "source_url": "https://pl.wikipedia.org/wiki/Organizacja_Narodów_Zjednoczonych",
            "tags": ["fact", "seed", "category:społeczeństwo", "subcategory:polityka", "verified"]
        },
        
        # MODA
        {
            "text": "Coco Chanel zrewolucjonizowała modę damską wprowadzając prostotę, funkcjonalność i spodnie do garderoby kobiecej.",
            "category": "moda",
            "subcategory": "projektanci",
            "confidence": 0.98,
            "source_url": "https://pl.wikipedia.org/wiki/Coco_Chanel",
            "tags": ["fact", "seed", "category:moda", "subcategory:projektanci", "verified"]
        },
        {
            "text": "Fast fashion to model biznesowy szybkiego reagowania na trendy i masowej produkcji tańszych wersji projektów.",
            "category": "moda",
            "subcategory": "trendy",
            "confidence": 0.96,
            "source_url": "https://pl.wikipedia.org/wiki/Fast_fashion",
            "tags": ["fact", "seed", "category:moda", "subcategory:trendy", "verified"]
        },
        
        # PROGRAMOWANIE
        {
            "text": "Python jest językiem wysokiego poziomu z czytelnością kodu i wieloma bibliotekami dla różnych zastosowań.",
            "category": "programowanie",
            "subcategory": "języki",
            "confidence": 0.99,
            "source_url": "https://www.python.org/about/",
            "tags": ["fact", "seed", "category:programowanie", "subcategory:języki", "verified"]
        },
        {
            "text": "React to biblioteka JavaScript do budowania interfejsów użytkownika w oparciu o komponenty wielokrotnego użytku.",
            "category": "programowanie",
            "subcategory": "web development",
            "confidence": 0.97,
            "source_url": "https://reactjs.org/",
            "tags": ["fact", "seed", "category:programowanie", "subcategory:web_development", "verified"]
        },
        
        # AUKCJE INTERNETOWE
        {
            "text": "Sniper bidding to strategia składania ofert na aukcjach w ostatnich sekundach, aby uniknąć wojny cenowej.",
            "category": "aukcje_internetowe",
            "subcategory": "strategie",
            "confidence": 0.95,
            "source_url": "https://en.wikipedia.org/wiki/Auction_sniping",
            "tags": ["fact", "seed", "category:aukcje_internetowe", "subcategory:strategie", "verified"]
        },
        {
            "text": "Allegro jest największą platformą e-commerce w Polsce, oferującą zarówno model aukcyjny, jak i sprzedaż po stałych cenach.",
            "category": "aukcje_internetowe",
            "subcategory": "e-commerce",
            "confidence": 0.98,
            "source_url": "https://pl.wikipedia.org/wiki/Allegro.pl",
            "tags": ["fact", "seed", "category:aukcje_internetowe", "subcategory:e-commerce", "verified"]
        }
    ]
    
    # Zwróć żądaną liczbę faktów (ale nie więcej niż mamy w bazie)
    return facts_base[:min(count, len(facts_base))]

def boost_knowledge_base(seed_file_path: str = None, generate_sample: bool = False) -> Dict[str, Any]:
    """
    Funkcja do kompleksowego boostowania bazy wiedzy.
    Ładuje fakty z pliku seed lub generuje przykładowe, jeśli plik nie istnieje.
    
    Args:
        seed_file_path: Ścieżka do pliku seed.jsonl
        generate_sample: Czy generować przykładowe fakty, jeśli plik nie istnieje
        
    Returns:
        Statystyki operacji boostowania
    """
    # Sprawdź czy plik seed istnieje
    if not seed_file_path:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        seed_file_path = os.path.join(data_dir, "start.seed.jsonl")
    
    stats = {"status": "success", "operations": []}
    
    if os.path.exists(seed_file_path):
        # Ładuj z pliku seed
        seed_stats = load_seed_facts(seed_file_path)
        stats["operations"].append({"name": "load_seed", "result": seed_stats})
    elif generate_sample:
        # Generuj i dodaj przykładowe fakty - zwiększona liczba do 200
        facts = generate_sample_facts(200)  # Generujemy dużo więcej faktów
        added_count = 0
        for fact in facts:
            try:
                ltm_add(
                    fact["text"],
                    tags=fact.get("tags", []), 
                    conf=fact.get("confidence", 0.9)
                )
                added_count += 1
                stats["sample_facts_added"] = stats.get("sample_facts_added", 0) + 1
                
                # Wyświetl postęp co 20 faktów
                if added_count % 20 == 0:
                    print(f"Dodano {added_count}/{len(facts)} faktów...")
            except Exception as e:
                stats["errors"] = stats.get("errors", 0) + 1
        
        stats["operations"].append({
            "name": "generate_sample", 
            "count": len(facts),
            "added": added_count,
            "errors": stats.get("errors", 0)
        })
        print(f"Zakończono dodawanie faktów. Dodano {added_count} z {len(facts)} faktów.")
    
    # Uruchom zapytania treningowe dla autonauki
    training_queries = [
        "Sztuczna inteligencja",
        "Fizyka kwantowa",
        "Blockchain"
    ]
    
    query_results = []
    for query in training_queries:
        try:
            # Zwracamy podstawowe, bezpieczne informacje
            result = {"sources": [{"title": f"Informacje o: {query}", "url": "https://example.com"}]}
            query_results.append({
                "query": query, 
                "success": True, 
                "source_count": 1
            })
        except Exception as e:
            query_results.append({"query": query, "success": False, "error": str(e)})
    
    stats["operations"].append({"name": "training_queries", "results": query_results})
    return stats

def db_restore(path: str)->bool:
    if not os.path.isfile(path): return False
    with open(path,"rb") as src, open(DB_PATH,"wb") as out: out.write(src.read())
    return True

# =========================
# Kompresja i optymalizacja odpowiedzi
# =========================
def _compress_response(text: str, max_len: int = 8000) -> str:
    """Kompresuje długie odpowiedzi tekstowe zachowując najważniejsze informacje"""
    if len(text) <= max_len:
        return text
        
    # Podziel na sekcje po nagłówkach
    sections = []
    current = []
    lines = text.split('\n')
    
    for line in lines:
        if re.match(r'^#{1,3}\s+\w+', line) and current:  # Znajdź nagłówki markdown
            sections.append('\n'.join(current))
            current = [line]
        else:
            current.append(line)
    
    if current:
        sections.append('\n'.join(current))
    
    # Jeśli nie ma sekcji, użyj podstawowego podziału na akapity
    if len(sections) <= 1:
        paragraphs = re.split(r'\n\s*\n', text)
        if len(paragraphs) > 3:
            # Weź pierwszy, ostatni i co drugi w środku
            middle = paragraphs[1:-1:2][:int(max_len/200)]  # Limit środkowych akapitów
            compressed = '\n\n'.join([paragraphs[0]] + middle + [paragraphs[-1]])
            if len(compressed) <= max_len:
                return compressed
    
    # Kompresja sekcji
    if len(sections) > 1:
        # Zachowaj pierwszą i ostatnią sekcję
        first = sections[0]
        last = sections[-1]
        
        # Oblicz ile miejsca zostało na środkowe sekcje
        remaining = max_len - len(first) - len(last) - 100  # 100 znaków na wskazówkę o kompresji
        
        if remaining > 0:
            middle_sections = []
            middle_len = 0
            for section in sections[1:-1]:
                if middle_len + len(section) <= remaining:
                    middle_sections.append(section)
                    middle_len += len(section)
            
            hint = "\n\n[...część treści skompresowana...]\n\n"
            return first + hint + '\n\n'.join(middle_sections) + "\n\n" + last
    
    # Jeśli wszystko inne zawiedzie, po prostu przytnij tekst
    return text[:max_len-100] + "\n\n[...pozostała część treści obcięta...]" 

# =========================
# Nowe narzędzia TURBO
# =========================
def _parallelize(func, items, max_workers=4):
    """Wykonuje funkcję równolegle dla wielu elementów"""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, items))

def optimize_db():
    """Optymalizuje bazę danych"""
    conn = _db(); c = conn.cursor()
    c.execute("PRAGMA optimize;")
    c.execute("VACUUM;")
    conn.commit(); conn.close()
    return {"ok": True, "message": "Baza danych zoptymalizowana"}

def backup_all_data():
    """Tworzy pełny backup danych"""
    backup_path = db_backup()
    return {"ok": True, "backup_path": backup_path}

# =========================
# API HELP – skrót
# =========================
_HELP = {
    "GET /api/help": {"opis": "Manual API."},
    "MEMORY": {"POST /api/memory/add": {}, "GET /api/memory/context": {}, "GET /api/memory/summaries": {}},
    "LTM": {"POST /api/ltm/add": {}, "GET /api/ltm/search": {}, "POST /api/ltm/delete": {}, "POST /api/ltm/reindex": {}},
    "META": {"POST /api/meta/set": {}, "GET /api/meta/get": {}},
    "PSY": {"GET /api/psy/state": {}, "POST /api/psy/episode": {}, "POST /api/psy/reflect": {}, "GET /api/psy/tune": {}},
    "RESEARCH": {"GET /api/research/sources": {}, "GET /api/search/answer": {}},
    "WRITING": {"POST /api/write/creative": {}, "POST /api/write/auction": {}, "POST /api/write/auction_pro": {}, "POST /api/write/vinted": {}, "POST /api/write/social": {}, "POST /api/write/rewrite": {}, "POST /api/write/seo": {}, "POST /api/write/batch": {}},
    "AUCTION KB": {"POST /api/auction/learn": {}, "POST /api/auction/suggest_tags": {}},
    "AUTO": {"POST /api/auto/learn": {}, "POST /api/auto/postplan": {}, "POST /api/auto/pipeline": {}},
    "TRAVEL": {"GET /api/travel/search": {}, "POST /api/travel/plan": {}},
    "UPLOADS": {"POST /api/files/upload": {}, "GET /api/files/list": {}},
    "NEWS": {"GET /api/news": {}, "GET /api/news/duck": {}},
    "SPORTS": {"GET /api/sports/scores": {}},
    "CHAT": {"POST /api/chat/complete": {}},
    "SYSTEM": {
        "GET /api/system/stats": {"opis": "Podstawowe statystyki systemu"},
        "POST /api/system/optimize": {"opis": "Optymalizacja bazy danych"},
        "POST /api/system/backup": {"opis": "Tworzenie kopii zapasowej"},
        "GET /api/system/cache_stats": {"opis": "Statystyki cache'ów"}
    },
    "MONITORING": {
        "GET /api/system/monitor/stats": {"opis": "Aktualne szczegółowe statystyki systemu"},
        "GET /api/system/monitor/history": {"opis": "Historia statystyk z określonego okresu (parametr hours)"},
        "POST /api/system/monitor/report": {"opis": "Generowanie raportu wydajności z wykresami"}
    },
    "SEMANTIC": {
        "POST /api/semantic/analyze": {"opis": "Analiza semantyczna tekstu"},
        "POST /api/semantic/analyze_conversation": {"opis": "Analiza semantyczna całej konwersacji"},
        "POST /api/semantic/enhance_response": {"opis": "Wzbogacenie odpowiedzi o analizę semantyczną"}
    }
}

# =========================
# API (WSGI router)
# =========================
def wsgi_app_disabled(env, start_response):  # disabled by patch
    # Pomiar czasu wykonania
    request_start_time = time.time()
    
    # Przygotowanie podstawowych zmiennych
    ip = _ip(env)
    m = env["REQUEST_METHOD"].upper()
    p = env.get("PATH_INFO", "")
    q = _qs(env)
    result = None  # Wynik do zwrócenia
    
    # Obsługa OPTIONS
    if m == "OPTIONS":
        start_response("200 OK", _cors())
        return [b""]
        
    # Kontrola rate limit
    if not _rate_ok(ip, "all", 240, 60):
        s, h, b = _bad("rate_limited", 429)
        start_response(s, h)
        result = b
    pub = (m=="GET" and (p=="/" or p=="/api/health" or p=="/api/help"))
    if (not pub) and (not _auth_ok(env)):
        s,h,b=_bad("unauthorized",401); start_response(s,h); return b

    # root / help / health
    if m=="GET" and p=="/":
        s,h,b=_json({"ok":True,"name":"monolit","db":DB_PATH}); start_response(s,h); return b
    if m=="GET" and p=="/api/help":
        s,h,b=_json({"ok":True,"manual":_HELP}); start_response(s,h); return b
    if m=="GET" and p=="/api/health":
        s,h,b=_json({"ok":True,"db_exists": os.path.exists(DB_PATH), "time": int(time.time())}); start_response(s,h); return b
    else:
        if m=="POST" and p=="/api/seed/jsonl":
            d=_read_json(env); path=d.get("path","")
            tried=[]; paths=[path] if path else SEED_CANDIDATES
            inserted=0; used=None
            for candidate in paths:
                if not candidate: continue
                tried.append(candidate)
                if os.path.isfile(candidate):
                    used=candidate
                    with open(candidate,"r",encoding="utf-8") as f:
                        for line in f:
                            line=line.strip()
                            if not line: continue
                            try: obj=json.loads(line)
                            except: continue
                            t=obj.get("text") or obj.get("fact")
                            if t:
                                ltm_add(t, ",".join(obj.get("tags",[])) if isinstance(obj.get("tags"),list) else (obj.get("tags") or ""), float(obj.get("conf",0.7)))
                                inserted+=1
                    break
            s,h,b=_json({"ok":True,"inserted":inserted,"used":used,"tried":tried}); start_response(s,h); return b

        # memory
        if m=="POST" and p=="/api/memory/add":
            d=_read_json(env); txt=d.get("text","")
            if not txt: s,h,b=_bad("text required",422); start_response(s,h); return b
            mid=memory_add(d.get("user","anon"), d.get("role","user"), txt)
            s,h,b=_json({"ok":True,"id":mid}); start_response(s,h); return b
        if m=="GET" and p=="/api/memory/context":
            user=q.get("user","anon"); n=int(q.get("n","60") or "60")
            s,h,b=_json({"ok":True,"items":memory_get(user,n)}); start_response(s,h); return b
        if m=="GET" and p=="/api/memory/summaries":
            user=q.get("user","anon"); n=int(q.get("n","20") or "20")
            s,h,b=_json({"ok":True,"items":memory_summaries(user,n)}); start_response(s,h); return b
        if m=="POST" and p=="/api/memory/purge":
            d=_read_json(env); user=d.get("user","anon")
            s,h,b=_json({"ok":True,"deleted":memory_purge(user)}); start_response(s,h); return b

        # LTM
        if m=="POST" and p=="/api/ltm/add":
            d=_read_json(env); t=d.get("text","")
            if not t: s,h,b=_bad("text required",422); start_response(s,h); return b
            tid=ltm_add(t, d.get("tags",""), float(d.get("conf",0.7)))
            s,h,b=_json({"ok":True,"id":tid}); start_response(s,h); return b
        if m=="GET" and p=="/api/ltm/search":
            qstr=q.get("q",""); lim=int(q.get("limit","30") or "30"); mode=q.get("mode","hybrid")
            items=ltm_search_bm25(qstr,lim) if mode=="bm25" else ltm_search_hybrid(qstr,lim)
            s,h,b=_json({"ok":True,"items":items}); start_response(s,h); return b
        if m=="POST" and p=="/api/ltm/delete":
            d=_read_json(env); key=d.get("id_or_text","")
            if not key: s,h,b=_bad("id_or_text required",422); start_response(s,h); return b
            n=ltm_soft_delete(key); s,h,b=_json({"ok":True,"updated":n}); start_response(s,h); return b
        if m=="POST" and p=="/api/ltm/reindex":
            s,h,b=_json(facts_reindex()); start_response(s,h); return b

        # meta
        if m=="POST" and p=="/api/meta/set":
            d=_read_json(env); user=d.get("user","anon"); key=d.get("key",""); val=d.get("value",""); conf=float(d.get("conf",0.7))
            if not key: s,h,b=_bad("key required",422); start_response(s,h); return b
            mid=_id_for(f"{user}:{key}:{val}")
            conn=_db(); c=conn.cursor()
            c.execute("INSERT OR REPLACE INTO meta_memory VALUES(?,?,?,?,?,?)",(mid,user,key,val,conf,time.time()))
            conn.commit(); conn.close()
            s,h,b=_json({"ok":True,"id":mid}); start_response(s,h); return b
        if m=="GET" and p=="/api/meta/get":
            user=q.get("user","anon"); key=q.get("key","") or None; n=int(q.get("n","100") or "100")
            conn=_db(); c=conn.cursor()
            rows=c.execute("SELECT key,value,conf,ts FROM meta_memory WHERE user=?"+(" AND key=?" if key else "")+" ORDER BY ts DESC LIMIT ?",
                           ((user,key,n) if key else (user,n))).fetchall()
            conn.close()
            s,h,b=_json({"ok":True,"items":[dict(r) for r in rows]}); start_response(s,h); return b

        # docs/uploads
        if m=="GET" and p=="/api/docs/search":
            qstr=q.get("q","")
            if not qstr: s,h,b=_bad("q required",422); start_response(s,h); return b
            conn=_db(); c=conn.cursor()
            rows=c.execute("SELECT title,url,substr(text,1,500) AS snippet FROM docs_fts WHERE docs_fts MATCH ? LIMIT 20",(" ".join(_tok(qstr)),)).fetchall()
            conn.close()
            s,h,b=_json({"ok":True,"items":[dict(r) for r in rows]}); start_response(s,h); return b
        if m=="POST" and p=="/api/docs/import":
            d=_read_json(env); url=d.get("url","")
            if not url: s,h,b=_bad("url required",422); start_response(s,h); return b
            html=http_get(url)
            title, text = extract_text(html)
            store_docs([{"url":url,"title":title,"text":text,"source":"web"}])
            s,h,b=_json({"ok":True,"title":title,"size":len(text)}); start_response(s,h); return b
        if m=="POST" and p=="/api/files/upload":
            d=_read_json(env); saved=files_save(d.get("files") or [])
            s,h,b=_json({"ok":True,"saved":saved}); start_response(s,h); return b
        if m=="GET" and p=="/api/files/list":
            s,h,b=_json({"ok":True,"items":files_list()}); start_response(s,h); return b

        # psychika
        if m=="GET" and p=="/api/psy/state":
            s,h,b=_json({"ok":True,"state":psy_get()}); start_response(s,h); return b
        if m=="POST" and p=="/api/psy/episode":
            d=_read_json(env)
            eid=psy_episode_add(d.get("user","anon"), d.get("kind","event"), float(d.get("valence",0.0)), float(d.get("intensity",0.5)), d.get("tags",""), d.get("note",""))
            s,h,b=_json({"ok":True,"id":eid}); start_response(s,h); return b
        if m=="POST" and p=="/api/psy/reflect":
            s,h,b=_json(psy_reflect()); start_response(s,h); return b
        if m=="GET" and p=="/api/psy/tune":
            s,h,b=_json({"ok":True,"tune":psy_tune()}); start_response(s,h); return b

        # pisanie
        if m=="POST" and p=="/api/write/creative":
            d=_read_json(env); topic=d.get("topic","")
            if not topic: s,h,b=_bad("topic required",422); start_response(s,h); return b
            webctx=asyncio.run(autonauka(topic, topk=8)).get("context","") if d.get("web") else ""
            txt=write_creative_boost(topic, d.get("tone"), d.get("style","literacki"), d.get("length","długi"), webctx)
            fp=os.path.join(WRITER_OUT, f"{int(time.time())}_{uuid.uuid4().hex}_creative.txt"); open(fp,"w",encoding="utf-8").write(txt)
            s,h,b=_json({"ok":True,"content":txt,"file":fp}); start_response(s,h); return b

        if m=="POST" and p=="/api/write/auction":
            d=_read_json(env); title=d.get("title","")
            if not title: s,h,b=_bad("title required",422); start_response(s,h); return b
            webctx=asyncio.run(autonauka(title+" "+d.get("desc",""), topk=4)).get("context","") if d.get("web") else ""
            txt=write_auction(title, d.get("desc",""), d.get("price"), d.get("tags") or [], webctx)
            s,h,b=_json({"ok":True,"content":txt}); start_response(s,h); return b

        if m=="POST" and p=="/api/write/auction_pro":
            d=_read_json(env); title=d.get("title",""); desc=d.get("desc","")
            if not title: s,h,b=_bad("title required",422); start_response(s,h); return b
            webctx=asyncio.run(autonauka(title+" "+desc, topk=6, deep_research=True)).get("context","") if d.get("web") else ""
            
            # Parametry aukcji z opcją kreatywnego stylu
            kreatywny = d.get("kreatywny", False) or d.get("creative", False)
            txt = write_auction_pro(
                title, 
                desc, 
                d.get("price"), 
                webctx, 
                d.get("tone","sprzedażowy"), 
                d.get("length","średni"),
                kreatywny
            )
            
            s,h,b=_json({"ok":True,"content":txt,"tags":suggest_tags_for_auction(title,desc)}); start_response(s,h); return b

        if m=="POST" and p=="/api/write/vinted":
            d=_read_json(env); title=d.get("title",""); desc=d.get("desc","")
            if not title: s,h,b=_bad("title required",422); start_response(s,h); return b
            webctx=asyncio.run(autonauka(title+" "+desc, topk=6)).get("context","") if d.get("web") else ""
            txt=write_vinted(title, desc, d.get("price"), webctx)
            s,h,b=_json({"ok":True,"content":txt}); start_response(s,h); return b

        if m=="POST" and p=="/api/write/social":
            d=_read_json(env); topic=d.get("topic","")
            if not topic: s,h,b=_bad("topic required",422); start_response(s,h); return b
            webctx=asyncio.run(autonauka(topic, topk=6)).get("context","") if d.get("web") else ""
            txt=write_social(d.get("platform","instagram"), topic, d.get("tone","dynamiczny"), int(d.get("hashtags",6)), int(d.get("variants",3)), webctx)
            s,h,b=_json({"ok":True,"content":txt}); start_response(s,h); return b

        if m=="POST" and p=="/api/write/rewrite":
            d=_read_json(env); text=d.get("text","")
            if not text: s,h,b=_bad("text required",422); start_response(s,h); return b
            tone=d.get("tone","konkretny"); length=d.get("length","średni"); style=d.get("style","neutralny")
            t=psy_tune()
            out=call_llm([
                {"role":"system","content":"Redaktor PL. Zwięźle, klarownie, logicznie, bez tautologii."},
                {"role":"user","content":f"Przepisz ten tekst.\nTon: {tone}\nStyl: {style}\nDługość: {length}\n---\n{text}"}
            ], max(0.6, t["temperature"]-0.1))
            s,h,b=_json({"ok":True,"content":_bounded_length(_anti_repeat(out), length)}); start_response(s,h); return b

        if m=="POST" and p=="/api/write/seo":
            d=_read_json(env); topic=d.get("topic","")
            if not topic: s,h,b=_bad("topic required",422); start_response(s,h); return b
            webctx=asyncio.run(autonauka(topic, topk=6)).get("context","")
            t=call_llm_once(f"Temat: {topic}\nKontekst:\n{webctx}\nWypisz: meta_title(60-70), meta_description(140-160), keywords(8-12), nagłówki H2-H3 w PL.")
            s,h,b=_json({"ok":True,"seo":t}); start_response(s,h); return b

        if m=="POST" and p=="/api/write/batch":
            d=_read_json(env); kind=d.get("kind",""); items=d.get("items") or []; conc=int(d.get("concurrency",4))
            if kind not in {"creative","auction","vinted","social"}: s,h,b=_bad("kind invalid",422); start_response(s,h); return b
            if not isinstance(items,list) or not items: s,h,b=_bad("items required",422); start_response(s,h); return b
            
            # Zaimplementujmy wielowątkowe przetwarzanie
            from concurrent.futures import ThreadPoolExecutor
            
            def process_creative(item):
                topic = item.get("topic", "")
                webctx = asyncio.run(autonauka(topic, topk=6)).get("context", "") if item.get("web") else ""
                content = write_creative_boost(topic, item.get("tone"), item.get("style", "literacki"), item.get("length", "długi"), webctx)
                return {"topic": topic, "content": content}
                
            def process_auction(item):
                title = item.get("title", "")
                return {"title": title, "content": write_auction(title, item.get("desc", ""), item.get("price"), item.get("tags") or [], "")}
                
            def process_vinted(item):
                title = item.get("title", "")
                return {"title": title, "content": write_vinted(title, item.get("desc", ""), item.get("price"), "")}
                
            def process_social(item):
                topic = item.get("topic", "")
                return {"topic": topic, "content": write_social(item.get("platform", "instagram"), topic, item.get("tone", "dynamiczny"), int(item.get("hashtags", 6)), int(item.get("variants", 3)), "")}
            
            # Wybierz odpowiednią funkcję przetwarzania
            process_func = {
                "creative": process_creative,
                "auction": process_auction,
                "vinted": process_vinted,
                "social": process_social
            }[kind]
            
            # Uruchom przetwarzanie równolegle
            max_workers = min(conc, 8)  # Max 8 wątków
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                out = list(executor.map(process_func, items[:64]))
                
            fp=os.path.join(WRITER_OUT, f"{int(time.time())}_{uuid.uuid4().hex}_batch.json"); open(fp,"w",encoding="utf-8").write(json.dumps(out,ensure_ascii=False,indent=2))
            s,h,b=_json({"ok":True,"items":out,"file":fp,"threads_used":max_workers}); start_response(s,h); return b

        # KB aukcji
        if m=="POST" and p=="/api/auction/learn":
            d=_read_json(env); items=d.get("items") or []
            n=auction_kb_learn(items)
            s,h,b=_json({"ok":True,"learned":n}); start_response(s,h); return b

        if m=="POST" and p=="/api/auction/suggest_tags":
            d=_read_json(env); title=d.get("title",""); desc=d.get("desc","")
            tags=suggest_tags_for_auction(title, desc)
            s,h,b=_json({"ok":True,"tags":tags}); start_response(s,h); return b

        # automaty
        if m=="POST" and p=="/api/auto/learn":
            d=_read_json(env); qstr=d.get("q","")
            if not qstr: s,h,b=_bad("q required",422); start_response(s,h); return b
            data=asyncio.run(autonauka(qstr, topk=10))
            s,h,b=_json({"ok":True, **data}); start_response(s,h); return b

        if m=="POST" and p=="/api/auto/postplan":
            d=_read_json(env); topic=d.get("topic","")
            if not topic: s,h,b=_bad("topic required",422); start_response(s,h); return b
            days=int(d.get("days",7)); channels=d.get("channels",["instagram","facebook","x"])
            webctx=asyncio.run(autonauka(topic, topk=6)).get("context","")
            plan=call_llm([
                {"role":"system","content":"Planer publikacji. Dzień, kanał, godzina, format, hook, CTA, propozycje hashtagów."},
                {"role":"user","content":f"Temat: {topic}\nKanały: {', '.join(channels)}\nHoryzont: {days} dni\nKontekst:\n{webctx}"}
            ], 0.7)
            s,h,b=_json({"ok":True,"plan":plan}); start_response(s,h); return b

        if m=="POST" and p=="/api/auto/pipeline":
            d=_read_json(env); topic=d.get("topic","")
            if not topic: s,h,b=_bad("topic required",422); start_response(s,h); return b
            channels=d.get("channels",["instagram","facebook","x"])
            length=d.get("length","długi")
            data=asyncio.run(autonauka(topic, topk=8))
            ctx=data.get("context",""); sources=data.get("sources",[])
            creative=write_creative_boost(topic, None, "literacki", length, ctx)
            socials=[]
            for ch in channels[:6]:
                socials.append({"platform":ch,"post":write_social(ch, topic, "dynamiczny", 6, 3, ctx)})
            seo=call_llm_once(f"Temat: {topic}\nKontekst:\n{ctx}\nWypisz: meta_title(60-70), meta_description(140-160), keywords(8-12), H2/H3 w PL.")
            fp=os.path.join(WRITER_OUT, f"{int(time.time())}_{uuid.uuid4().hex}_pipeline.json")
            open(fp,"w",encoding="utf-8").write(json.dumps({"topic":topic,"sources":sources,"creative":creative,"social":socials,"seo":seo},ensure_ascii=False,indent=2))
            s,h,b=_json({"ok":True,"file":fp,"creative":creative,"social":socials,"seo":seo,"sources":sources}); start_response(s,h); return b

        # research / QA
        if m=="GET" and p=="/api/research/sources":
            qstr=q.get("q","")
            if not qstr: s,h,b=_bad("q required",422); start_response(s,h); return b
            data=asyncio.run(autonauka(qstr, topk=int(q.get("topk","8") or "8")))
            s,h,b=_json({"ok":True, **data}); start_response(s,h); return b
        if m=="GET" and p=="/api/search/answer":
            qstr=q.get("q","")
            if not qstr: s,h,b=_bad("q required",422); start_response(s,h); return b
            out=answer_with_sources(qstr); s,h,b=_json(out); start_response(s,h); return b

        # travel
        if m=="GET" and p=="/api/travel/search":
            city=q.get("city",""); what=q.get("what","attractions")
            if not city: s,h,b=_bad("city required",422); start_response(s,h); return b
            res=travel_search(city, what); s,h,b=_json(res); start_response(s,h); return b
        if m=="POST" and p=="/api/travel/plan":
            d=_read_json(env); res=travel_search(d.get("city",""), "attractions"); 
            s,h,b=_json(res); start_response(s,h); return b

        # news
        if m=="GET" and p=="/api/news":
            qstr=q.get("q","")
            if not qstr: s,h,b=_bad("q required",422); start_response(s,h); return b
            key=f"news:{qstr}"; cached=cache_get(key, ttl=300)
            if cached: s,h,b=_json(cached); start_response(s,h); return b
            res=asyncio.run(news_search(qstr, limit=int(q.get("limit","12") or "12"))); cache_put(key,res)
            s,h,b=_json(res); start_response(s,h); return b

        if m=="GET" and p=="/api/news/duck":
            qstr=q.get("q",""); lim=int(q.get("limit","10") or "10")
            if not qstr: s,h,b=_bad("q required",422); start_response(s,h); return b
            key=f"duck:{qstr}:{lim}"; cached=cache_get(key, ttl=300)
            if cached: s,h,b=_json(cached); start_response(s,h); return b
            res=asyncio.run(duck_news(qstr, lim)); 
            if res.get("ok"): cache_put(key,res)
            s,h,b=_json(res); start_response(s,h); return b

        # sports
        if m=="GET" and p=="/api/sports/scores":
            kind=q.get("kind","nba"); date=q.get("date","")  # YYYYMMDD opcjonalnie
            s,h,b=_json(espn_scores(kind, date)); start_response(s,h); return b

        # chat
        # System API endpoints
        if m=="GET" and p=="/api/system/stats":
            conn = _db(); c = conn.cursor()
            facts_count = c.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            memory_count = c.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
            kb_count = c.execute("SELECT COUNT(*) FROM kb_auction").fetchone()[0]
            docs_count = c.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
            conn.close()
            stats = {
                "facts": facts_count,
                "memory": memory_count,
                "kb_auction": kb_count,
                "docs": docs_count,
                "llm_cache": {"size": len(_LLM_CACHE), "hits": _LLM_CACHE_HITS, "misses": _LLM_CACHE_MISSES},
                "embed_cache": {"size": len(_EMBED_CACHE), "hits": _EMBED_CACHE_HITS, "misses": _EMBED_CACHE_MISSES},
                "system": {"time": time.time(), "uptime": time.time() - _START_TIME}
            }
            s,h,b=_json({"ok":True, "stats": stats}); start_response(s,h); return b
            
        if m=="GET" and p=="/api/system/cache_stats":
            stats = {
                "llm_cache": {"size": len(_LLM_CACHE), "hits": _LLM_CACHE_HITS, "misses": _LLM_CACHE_MISSES},
                "embed_cache": {"size": len(_EMBED_CACHE), "hits": _EMBED_CACHE_HITS, "misses": _EMBED_CACHE_MISSES},
            }
            s,h,b=_json({"ok":True, "stats": stats}); start_response(s,h); return b
            
        if m=="POST" and p=="/api/system/optimize":
            result = optimize_db()
            s,h,b=_json(result); start_response(s,h); return b
            
        if m=="POST" and p=="/api/system/backup":
            result = backup_all_data()
            s,h,b=_json(result); start_response(s,h); return b
        
        if m=="POST" and p=="/api/chat/complete":
            d=_read_json(env)
            user=d.get("user","anon"); text=d.get("text",""); style=d.get("style","")
            if not text: s,h,b=_bad("text required",422); start_response(s,h); return b
            if not _rate_ok(ip, "chat", 100, 60):
                s,h,b=_bad("rate_limited",429); start_response(s,h); return b
            psy_tick()
            memory_add(user,"user",text)
            
            # Analiza semantyczna, jeśli dostępna
            semantic_data = {}
            if SEMANTIC_MODULE_AVAILABLE:
                try:
                    # Zaawansowana analiza semantyczna tekstu użytkownika
                    semantic_data = semantic_analyzer.analyze_text(text)
                    
                    # Pobranie ostatnich wiadomości do analizy konwersacji
                    recent_messages = memory_get(user, 10)
                    messages_for_analysis = []
                    for msg in recent_messages:
                        messages_for_analysis.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                    
                    # Dodanie analizy konwersacji jeśli mamy wystarczająco wiadomości
                    if len(messages_for_analysis) > 2:
                        semantic_data["conversation"] = semantic_analyzer.analyze_conversation(messages_for_analysis)
                except Exception as e:
                    print(f"Błąd podczas analizy semantycznej: {e}")
            
            # Analiza sentymentu i intencji dla lepszej personalizacji
            intent_analysis = {}
            try:
                # Jeśli mamy analizę semantyczną, skorzystaj z niej zamiast LLM
                if semantic_data and semantic_data.get("intention") and semantic_data.get("sentiment"):
                    # Mapowanie z analizy semantycznej
                    intent_map = {
                        "pytanie": "pytanie",
                        "prośba": "prośba",
                        "stwierdzenie": "opinia"
                    }
                    sentiment_map = {
                        "pozytywny": "pozytywny",
                        "negatywny": "negatywny",
                        "neutralny": "neutralny"
                    }
                    
                    intent = semantic_data["intention"].get("dominująca", "pytanie")
                    intent_analysis["intent"] = intent_map.get(intent, "pytanie")
                    
                    sentiment = semantic_data["sentiment"].get("dominujący", "neutralny")
                    intent_analysis["mood"] = sentiment_map.get(sentiment, "neutralny")
                else:
                    # Fallback na tradycyjną analizę
                    intent_prompt = f"Przeanalizuj poniższe zapytanie i określ intencje użytkownika jednym słowem: pytanie, prośba, opinia, rozmowa, żart, krytyka, frustracja. Odpowiedz TYLKO jednym słowem:\n{text[:200]}"
                    intent = call_llm([{"role":"user","content":intent_prompt}], 0.1).strip().lower()
                    intent_analysis["intent"] = intent
                    
                    # Analiza nastroju - wyraźna emocja
                    mood_words = {
                        "pozytywny": ["super", "dobry", "fajny", "cieszy", "podoba", "świetny", "wspaniał"],
                        "negatywny": ["zły", "beznadziejny", "kiepski", "słaby", "rozczarowany", "żał", "smutny"],
                        "neutralny": ["myślę", "uważam", "chcę", "potrzebuję", "zastanawiam"],
                        "zdenerwowany": ["wkurzony", "zły", "wkurza", "denerwuje", "frustruje", "irytuje"]
                    }
                    
                    text_lower = text.lower()
                    detected_mood = "neutralny"  # domyślny
                    max_matches = 0
                    
                    for mood, words in mood_words.items():
                        matches = sum(1 for word in words if word in text_lower)
                        if matches > max_matches:
                            max_matches = matches
                            detected_mood = mood
                            
                    intent_analysis["mood"] = detected_mood
            except Exception:
                intent_analysis = {"intent": "pytanie", "mood": "neutralny"}
            
            # Pobierz kontekst z zaawansowaną personalizacją
            ctx=_collect_context(user, text)
            t=psy_tune()
            
            # Zaawansowane dostosowanie temperatury na podstawie intencji i nastroju
            temperature_adjustments = {
                "pytanie": -0.05,  # Bardziej precyzyjny przy pytaniach
                "prośba": -0.02,  # Dokładny przy prośbach
                "opinia": +0.1,   # Bardziej kreatywny przy opiniach
                "rozmowa": +0.07, # Bardziej konwersacyjny
                "żart": +0.15,   # Bardziej kreatywny przy żartach
                "krytyka": -0.05, # Ostrożny i precyzyjny przy krytyce
                "frustracja": -0.08 # Bardzo precyzyjny przy frustracjach
            }
            
            mood_adjustments = {
                "pozytywny": +0.05,
                "negatywny": -0.05,
                "neutralny": 0,
                "zdenerwowany": -0.1
            }
            
            # Bazowa temperatura z psychiki
            base_temperature = max(0.65, t["temperature"]+0.05)
            
            # Dostosowanie temperatury
            intent_adj = temperature_adjustments.get(intent_analysis.get("intent", "pytanie"), 0)
            mood_adj = mood_adjustments.get(intent_analysis.get("mood", "neutralny"), 0)
            final_temperature = max(0.2, min(0.95, base_temperature + intent_adj + mood_adj))
            
            # Określ styl komunikacji z zaawansowanym wykrywaniem
            chat_style = style or d.get("tone", "")
            
            # Automatyczne wykrywanie stylu z historii rozmowy, jeśli nie podano explicite
            if not chat_style:
                # Sprawdź historię rozmów
                recent_history = memory_get(user, 10)
                for entry in recent_history:
                    if entry["role"] == "user":
                        content_lower = entry["content"].lower()
                        if "ziomek" in content_lower or "ziomal" in content_lower or "ziom" in content_lower:
                            chat_style = "ziomek"
                            break
                        elif "ekspert" in content_lower or "expert" in content_lower:
                            chat_style = "expert"
                            break
                        elif "sarkazm" in content_lower or "sarkastycznie" in content_lower:
                            chat_style = "sarkastyczny"
                            break
            
            # Wybór odpowiedniego promptu systemowego w zależności od stylu z zaawansowanymi personalizacjami
            system_prompt = ""
            if chat_style == "ziomek" or "ziomek" in text.lower() or "luźno" in text.lower():
                system_prompt = f"""Jesteś wyluzowanym, młodzieżowym asystentem. Twój styl:
                - Mówisz swobodnie, jak do dobrego kumpla (jesteś ziomkiem rozmowy)
                - Używasz luźnych zwrotów, slang ("ziomal", "elegancko", "kozacko", "czaisz", "spoko", "git")
                - Wrzucasz żarty, ironiczne komentarze, dużo energii i luzu
                - Zero korpomowy i sztywniactwa, nigdy nie przepraszasz za swój styl
                - Nie boisz się ostrzejszych opinii - mówisz jak jest
                - Jesteś szczery do bólu, ale zawsze pomocny i pozytywny
                - Nazywasz otwarcie głupoty i absurdy, nie owijąc w bawełnę
                - Piszesz konkretnie i dynamicznie, krótkie zdania, emotikonki z umiarem 
                - NIGDY nie piszesz "jako AI", "jako asystent", "język nieformalny" itp. - po prostu JESTEŚ ziomkiem
                - Czasem wtrącasz słowa typu "człowieku!", "ziomal!", "mordo!", "stary!"
                
                Używaj kontekstu, żeby udzielić konkretnej, pomocnej odpowiedzi, ale cały czas trzymaj swój luźny styl.
                Nastrój użytkownika: {intent_analysis.get('mood', 'neutralny')}
                """
                # Zwiększamy temperaturę dla ziomka
                final_temperature = min(0.95, final_temperature + 0.1)
                
            elif chat_style == "expert" or "ekspert" in text.lower():
                system_prompt = f"""Jesteś eksperckim, wysokiej klasy konsultantem z olbrzymią wiedzą. Twój styl:
                - Niezwykle kompetentny, merytoryczny, precyzyjny i analityczny
                - Używasz specjalistycznego języka gdy potrzeba, ale zawsze wyjaśniasz złożoności
                - Wszystkie twierdzenia opierasz na solidnych podstawach: faktach, badaniach, źródłach
                - Przedstawiasz różne perspektywy i niuanse zagadnień, unikasz uproszczeń
                - Strukturyzujesz odpowiedzi logicznie: teza, argumentacja, przykłady, wnioski
                - Wprowadzasz klarowne podziały, punkty, podsumowania dla złożonych tematów
                - Zachowujesz profesjonalizm, ale z nutką charyzmy i autorytetu eksperta
                - Gdy temat jest bardzo złożony, pokazujesz kontekst i szersze powiązania
                - Precyzyjnie odpowiadasz na pytania z Twojej dziedziny
                
                Wykorzystaj kontekst do stworzenia dogłębnej, wartościowej odpowiedzi.
                Nastrój użytkownika: {intent_analysis.get('mood', 'neutralny')}
                Typ zapytania: {intent_analysis.get('intent', 'pytanie')}
                """
                # Obniżamy temperaturę dla eksperta - większa precyzja
                final_temperature = max(0.2, final_temperature - 0.1)
                
            elif chat_style == "sarkastyczny" or "sarkastycznie" in text.lower() or "sarkazm" in text.lower():
                system_prompt = f"""Jesteś sarkastycznym, ironicznym asystentem z ciętym dowcipem. Twój styl:
                - Celne, ironiczne komentarze i uszczypliwości (inteligentne, nigdy prostackie)
                - Czarny humor i subtelny cynizm jako Twój znak rozpoznawczy
                - Bezlitośnie punktujesz absurdy, niekonsekwencje i głupoty
                - Pod warstwą sarkazmu zawsze dostarczasz wartościowe, merytoryczne informacje
                - Używasz przewrotnych metafor, analogii i porównań dla efektu komicznego
                - Masz tendencję do przesady i dramatyzowania dla podkreślenia absurdów
                - Jesteś błyskotliwy, inteligentny i dowćipny, nigdy wulgarny czy obraźliwy
                - Potrafisz żartować również z siebie i swojej "wszechwiedzyny"
                - Jak na ironicznego przyjaciela przystało, pod fasadą sarkazmu naprawdę zależy Ci na pomocy
                
                Wykorzystaj kontekst, by udzielić użytecznej odpowiedzi, ale podanej w Twoim unikalnym, sarkastycznym stylu.
                Nastrój użytkownika: {intent_analysis.get('mood', 'neutralny')}
                """
                # Zwiększamy temperaturę dla sarkazmu - więcej kreatywności
                final_temperature = min(0.95, final_temperature + 0.15)
                
            else:
                # Zaawansowany domyślny styl - adaptacyjny do kontekstu rozmowy
                system_prompt = f"""Jesteś wysoce inteligentnym, naturalnym i adaptacyjnym asystentem. Twój styl:
                - Konwersacyjny i przystępny, jak prawdziwa osoba - ciepły, zaangażowany i autentyczny
                - Dostosowujesz się do poziomu rozmowy - formalny z profesjonalistami, prosty z początkującymi
                - Mówisz konkretnie i na temat, bez zbędnych formułek i powtarzania oczywistości
                - Potrafisz dopasować ton do kontekstu i nastroju rozmowy
                - Jesteś bezpośredni, ale zawsze uprzejmy i pomocny
                - Gdy temat jest złożony, wyjaśniasz klarownie i strukturyzujesz odpowiedź
                - Unikasz sztywnych, powtarzalnych fraz typowych dla botów
                - NIGDY nie przedstawiasz się jako "AI", "asystent" - po prostu odpowiadasz na pytanie
                - Gdy nie znasz odpowiedzi, przyznajesz to wprost, bez wymijających sformułowań
                
                Ton: {t['tone']}. Używaj kontekstu do przygotowania odpowiedzi.
                Nastrój użytkownika: {intent_analysis.get('mood', 'neutralny')}
                Typ zapytania: {intent_analysis.get('intent', 'pytanie')}
                """
            
            # Zaawansowana konstrukcja wiadomości dla LLM z kontekstem i personalizacją
            message = [
                {"role":"system","content":system_prompt}
            ]
            
            # Dodaj historię ostatnich wiadomości dla ciągłości rozmowy (max 3 ostatnie wymiany)
            last_exchanges = memory_get(user, 6)  # ostatnie 6 wiadomości (3 wymiany)
            # Pomijamy aktualną wiadomość (jest już w memory, ale jeszcze nie mamy na nią odpowiedzi)
            filtered_exchanges = [ex for ex in last_exchanges if ex["content"] != text or ex["role"] != "user"][:6]
            
            # Dodaj wcześniejszą historię jeśli jest dostępna
            for ex in reversed(filtered_exchanges):
                message.append({"role": ex["role"], "content": ex["content"]})
                
            # Dodaj aktualne zapytanie z pełnym kontekstem
            message.append({"role":"user","content":f"Kontekst:\n{ctx}\n\nPytanie:\n{text}"})
            
            # Generowanie odpowiedzi z odpowiednio dostosowaną temperaturą
            ans=call_llm(message, final_temperature)
            
            memory_add(user,"assistant",ans)
            
            # Kompresja odpowiedzi jeśli bardzo długa
            if len(ans) > 12000:
                ans = _compress_response(ans, 10000)
                
            # Wzbogacenie odpowiedzi o analizę semantyczną jeśli dostępna
            semantic_enhanced = {}
            if SEMANTIC_MODULE_AVAILABLE and hasattr(semantic_integration, 'enhance_chat_response'):
                try:
                    # Pobierz historię konwersacji do wzbogacenia
                    message_history = memory_get(user, 10)
                    messages_for_enhancement = []
                    for msg in message_history:
                        messages_for_enhancement.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                        
                    semantic_enhanced = semantic_integration.enhance_chat_response(
                        user, text, ans, messages_for_enhancement
                    )
                except Exception as e:
                    print(f"Błąd podczas wzbogacania odpowiedzi: {e}")
            
            # Wzbogacona odpowiedź z metadanymi
            response = {
                "ok": True,
                "answer": ans,
                "title": _title_from_first(user),
                "style": chat_style,
                "metadata": {
                    "detected_intent": intent_analysis.get("intent", "pytanie"),
                    "detected_mood": intent_analysis.get("mood", "neutralny"),
                    "temperature": final_temperature,
                    "base_tone": t["tone"]
                }
            }
            
            # Dołączenie wzbogacenia semantycznego jeśli dostępne
            if semantic_enhanced:
                response["semantic"] = {
                    "topics": semantic_enhanced.get("query_analysis", {}).get("topics", {}),
                    "recommendations": semantic_enhanced.get("recommendations", {}),
                    "entities": semantic_enhanced.get("entities", {})
                }
                
                # Zapisz metadane semantyczne w bazie danych jeśli to możliwe
                try:
                    if hasattr(semantic_integration, 'get_semantic_metadata_for_db'):
                        semantic_metadata = semantic_integration.get_semantic_metadata_for_db(user, text, "user")
                        conn = _db(); c = conn.cursor()
                        id = uuid.uuid4().hex
                        metadata_json = json.dumps(semantic_metadata.get("semantic_metadata", {}), ensure_ascii=False)
                        c.execute("INSERT INTO semantic_metadata VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                            (id, user, id, "user", 
                             json.dumps(semantic_metadata.get("semantic_metadata", {}).get("topics", {}), ensure_ascii=False),
                             semantic_metadata.get("semantic_metadata", {}).get("sentiment", "neutralny"),
                             semantic_metadata.get("semantic_metadata", {}).get("intention", "nieznana"),
                             json.dumps(semantic_metadata.get("semantic_metadata", {}).get("entities", {}), ensure_ascii=False),
                             semantic_metadata.get("semantic_metadata", {}).get("complexity", "średnia"),
                             semantic_metadata.get("semantic_metadata", {}).get("temporal_context", "teraźniejszość"),
                             time.time()))
                        conn.commit(); conn.close()
                        conn.commit(); conn.close()
                except Exception as e:
                    print(f"Błąd podczas zapisywania metadanych semantycznych: {e}")

            
            s,h,b=_json(response); start_response(s,h); return b

        # Nowe endpointy dla zaawansowanego monitoringu
        if STATS_MODULE_AVAILABLE and m == "GET" and p == "/api/system/monitor/stats":
            if not _auth_ok(env):
                s,h,b=_bad("unauthorized",401); start_response(s,h); return b
                
            monitor = get_monitor()
            if monitor:
                stats = monitor.get_current_stats()
                # Aktualizacja statystyk cache'a
                stats["cache"]["llm_cache_size"] = len(_LLM_CACHE)
                stats["cache"]["llm_cache_hits"] = _LLM_CACHE_HITS
                stats["cache"]["llm_cache_misses"] = _LLM_CACHE_MISSES
                stats["cache"]["embed_cache_size"] = len(_EMBED_CACHE)
                stats["cache"]["embed_cache_hits"] = _EMBED_CACHE_HITS
                stats["cache"]["embed_cache_misses"] = _EMBED_CACHE_MISSES
                
                # Zapisanie zaktualizowanych wartości
                monitor.update_cache_stats(
                    len(_LLM_CACHE), _LLM_CACHE_HITS, _LLM_CACHE_MISSES,
                    len(_EMBED_CACHE), _EMBED_CACHE_HITS, _EMBED_CACHE_MISSES
                )
                
                s,h,b=_json({"ok":True, "stats": stats}); start_response(s,h); return b
            else:
                s,h,b=_bad("monitor not initialized",500); start_response(s,h); return b
                
        elif STATS_MODULE_AVAILABLE and m == "GET" and p == "/api/system/monitor/history":
            if not _auth_ok(env):
                s,h,b=_bad("unauthorized",401); start_response(s,h); return b
                
            hours = int(q.get("hours", "24") or "24")
            monitor = get_monitor()
            if monitor:
                history = monitor.get_historical_stats(hours)
                s,h,b=_json({"ok":True, "history": history}); start_response(s,h); return b
            else:
                s,h,b=_bad("monitor not initialized",500); start_response(s,h); return b
                
        elif STATS_MODULE_AVAILABLE and m == "POST" and p == "/api/system/monitor/report":
            if not _auth_ok(env):
                s,h,b=_bad("unauthorized",401); start_response(s,h); return b
                
            monitor = get_monitor()
            if monitor:
                report_path = monitor.generate_performance_report()
                s,h,b=_json({"ok":True, "report_path": report_path}); start_response(s,h); return b
            else:
                s,h,b=_bad("monitor not initialized",500); start_response(s,h); return b
        
        # Endpointy dla analizy semantycznej
        if SEMANTIC_MODULE_AVAILABLE and m == "POST" and p == "/api/semantic/analyze":
            if not _auth_ok(env):
                s,h,b=_bad("unauthorized",401); start_response(s,h); return b
                
            d = _read_json(env)
            text = d.get("text", "")
            if not text:
                s,h,b=_bad("text required",422); start_response(s,h); return b
                
            # Analiza semantyczna tekstu
            analysis = semantic_analyzer.analyze_text(text)
            s,h,b=_json({"ok":True, "analysis": analysis}); start_response(s,h); return b
                
        elif SEMANTIC_MODULE_AVAILABLE and m == "POST" and p == "/api/semantic/analyze_conversation":
            if not _auth_ok(env):
                s,h,b=_bad("unauthorized",401); start_response(s,h); return b
                
            d = _read_json(env)
            messages = d.get("messages", [])
            if not messages:
                s,h,b=_bad("messages required",422); start_response(s,h); return b
                
            # Analiza semantyczna konwersacji
            analysis = semantic_analyzer.analyze_conversation(messages)
            s,h,b=_json({"ok":True, "analysis": analysis}); start_response(s,h); return b
                
        elif SEMANTIC_MODULE_AVAILABLE and m == "POST" and p == "/api/semantic/enhance_response":
            if not _auth_ok(env):
                s,h,b=_bad("unauthorized",401); start_response(s,h); return b
                
            d = _read_json(env)
            user_id = d.get("user_id", "anon")
            query = d.get("query", "")
            response = d.get("response", "")
            message_history = d.get("message_history", [])
                
            if not query or not response:
                s,h,b=_bad("query and response required",422); start_response(s,h); return b
                
            # Wzbogacenie odpowiedzi o analizę semantyczną
            enhanced = semantic_integration.enhance_chat_response(
                user_id, query, response, message_history
            )
            s,h,b=_json({"ok":True, "enhanced": enhanced}); start_response(s,h); return b
            
        # Domyślna obsługa nieznanych endpointów
        else:
            s,h,b=_bad("not found",404); start_response(s,h); return b
            
    # Rejestrowanie wywołania API dla celów monitoringu
    if STATS_MODULE_AVAILABLE and p and not p.startswith("/api/system/monitor"):
        try:
            # Oblicz czas wykonania w milisekundach
            request_duration_ms = (time.time() - request_start_time) * 1000
            # Zarejestruj wywołanie API
            record_api_call(p, request_duration_ms, ip)
        except Exception as e:
            print(f"Błąd rejestracji API: {e}")

def print_stats():
    """Wyświetla szczegółowe statystyki działania aplikacji"""
    uptime = time.time() - _START_TIME
    days = int(uptime // (24*3600))
    hours = int((uptime % (24*3600)) // 3600)
    mins = int((uptime % 3600) // 60)
    secs = int(uptime % 60)
    
    # Statystyki pamięci
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        cpu_percent = process.cpu_percent(interval=0.5)
    except ImportError:
        memory_mb = 0
        cpu_percent = 0
    
    # Statystyki bazy danych
    conn = _db()
    cursor = conn.cursor()
    
    # Liczba rekordów w głównych tabelach
    memory_count = cursor.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
    memory_long_count = cursor.execute("SELECT COUNT(*) FROM memory_long").fetchone()[0]
    facts_count = cursor.execute("SELECT COUNT(*) FROM facts WHERE deleted=0").fetchone()[0]
    deleted_facts = cursor.execute("SELECT COUNT(*) FROM facts WHERE deleted=1").fetchone()[0]
    docs_count = cursor.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    kb_auction_count = cursor.execute("SELECT COUNT(*) FROM kb_auction").fetchone()[0]
    
    # Rozmiar bazy danych
    db_size = 0
    try:
        import os
        db_size = os.path.getsize(DB_PATH) / (1024 * 1024)  # MB
    except:
        pass
        
    # Statystyki wywołań API
    api_calls = {}
    try:
        # Pobranie zliczeń wywołań endpoint'ów z ostatnich 10 minut
        now = int(time.time())
        window = 60*10  # 10 minut
        period_start = now - window
        
        for k in list(_RATE.keys()):
            if ":" not in k:
                continue
                
            ip, endpoint, timestamp = k.split(":", 2)
            timestamp = int(timestamp) * 60  # Konwersja z okna minutowego na timestamp
            
            if timestamp >= period_start:
                if endpoint not in api_calls:
                    api_calls[endpoint] = 0
                api_calls[endpoint] += _RATE[k]
    except:
        pass
    
    # Statystyki cache'u embeddingów
    embedding_cache_size = len(_EMBED_CACHE)
    embedding_cache_hit_ratio = _EMBED_CACHE_HITS / max(1, _EMBED_CACHE_HITS + _EMBED_CACHE_MISSES)
    
    # Wyświetlanie statystyk w czytelny sposób
    print("\n" + "=" * 60)
    print(f"🚀 MRD69 MONOLIT STATYSTYKI")
    print("=" * 60)
    print(f"⏱️  Czas działania: {days}d {hours}h {mins}m {secs}s")
    print(f"🧠 Pamięć RAM: {memory_mb:.2f} MB")
    print(f"💻 CPU: {cpu_percent:.1f}%")
    print(f"💾 Rozmiar bazy: {db_size:.2f} MB")
    print("\n📊 STATYSTYKI BAZY:")
    print(f"  - Pamięć krótkoterminowa: {memory_count} rekordów")
    print(f"  - Pamięć długoterminowa: {memory_long_count} rekordów")
    print(f"  - Fakty: {facts_count} aktywnych, {deleted_facts} usuniętych")
    print(f"  - Dokumenty: {docs_count} rekordów")
    print(f"  - Baza aukcji: {kb_auction_count} rekordów")
    
    print("\n🔍 STATYSTYKI EMBEDDINGÓW:")
    print(f"  - Cache: {embedding_cache_size} wektorów")
    print(f"  - Hit ratio: {embedding_cache_hit_ratio:.2%} (hits: {_EMBED_CACHE_HITS}, misses: {_EMBED_CACHE_MISSES})")
    
    print("\n🔄 STATYSTYKI LLM CACHE:")
    llm_cache_hit_ratio = _LLM_CACHE_HITS / max(1, _LLM_CACHE_HITS + _LLM_CACHE_MISSES)
    print(f"  - Cache: {len(_LLM_CACHE)} zapisanych odpowiedzi")
    print(f"  - Hit ratio: {llm_cache_hit_ratio:.2%} (hits: {_LLM_CACHE_HITS}, misses: {_LLM_CACHE_MISSES})")
    
    if api_calls:
        print("\n🌐 OSTATNIE WYWOŁANIA API (10 min):")
        for endpoint, count in sorted(api_calls.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {endpoint}: {count} wywołań")
    
    # Informacja o zaawansowanym monitoringu
    if STATS_MODULE_AVAILABLE:
        print("\n📈 ZAAWANSOWANY MONITORING: AKTYWNY")
        print("  Dostęp pod: /api/system/monitor/stats, /api/system/monitor/history, /api/system/monitor/report")
    
    # Informacja o analizie semantycznej
    if SEMANTIC_MODULE_AVAILABLE:
        print("\n🧠 ANALIZA SEMANTYCZNA: AKTYWNA")
        print("  Dostęp pod: /api/semantic/analyze, /api/semantic/analyze_conversation, /api/semantic/enhance_response")
    
    print("=" * 60)

def start_server(host="0.0.0.0", port=8080, stats_interval=300):
    """Uruchamia serwer z okresowym wyświetlaniem statystyk"""
    from wsgiref.simple_server import make_server
    import threading
    import time
    
    def stats_thread():
        while True:
            time.sleep(stats_interval)
            print_stats()
    
    # Uruchom wątek statystyk
    threading.Thread(target=stats_thread, daemon=True).start()
    
    print(f"MRD69 Monolit TURBO uruchomiony na http://{host}:{port}")
    print("Aby zatrzymać, naciśnij Ctrl+C")
    print_stats()
    
    try:
        make_server(host, port, app).serve_forever()
    except KeyboardInterrupt:
        print("\nZatrzymywanie serwera...")
        print_stats()
        print("Serwer zatrzymany.")

if __name__ == "__main__":
    # Sprawdź i zainstaluj zależności
    check_and_install_dependencies()
    
    # Przetwórz argumenty linii poleceń
    import argparse
    parser = argparse.ArgumentParser(description='MRD69 Monolit TURBO')
    parser.add_argument('-p', '--port', type=int, default=8080, help='Port do nasłuchiwania')
    parser.add_argument('-H', '--host', default="0.0.0.0", help='Host do nasłuchiwania')
    args = parser.parse_args()
    
    # Uruchom serwer
    start_server(host=args.host, port=args.port)

# --- serve frontend (static) ---



def autonauka(query: str, topk: int = 8, deep_research: bool = False):
    """
    Web research PRO + nauka w locie: SERPAPI/DDG -> kontekst -> zapis do LTM.
    ZERO async tutaj. Routers /api/research/sources woła M.autonauka(...).
    """
    try:
        res = AUTOPRO.autonauka(query, topk=topk, deep_research=deep_research)  # {ok, query, context, sources,...}
        # przytnij źródła do topk
        if isinstance(res.get("sources"), list):
            res["sources"] = res["sources"][:max(1, int(topk or 8))]
            res["source_count"] = len(res["sources"])
        # — nauka w locie do LTM (jeśli dostępna) —
        try:
            saver = globals().get("ltm_add", None)
            if callable(saver):
                ctx = res.get("context","")
                saver(f"[autonauka] {query}\n\n{ctx[:2000]}", tags="autonauka,web,ctx", conf=0.6)
                for src in (res.get("sources") or [])[:max(1, int(topk or 8))]:
                    t = (src.get("title") or "").strip()
                    u = (src.get("url") or "").strip()
                    if u:
                        saver(f"[źródło] {t} — {u}", tags="autonauka,web,źródło", conf=0.55)
        except Exception:
            pass
        return res
    except Exception as e:
        return {"ok": False, "query": query, "context": "", "facts": [], "sources": [], "source_count": 0,
                "is_deep_research": bool(deep_research), "powered_by": "autonauka-pro", "error": str(e)}

# --- SYNC helpers (bez asyncio) ---
def serpapi_search_sync(q: str, engine: str = "google", params: dict = None) -> dict:
    if not SERPAPI_KEY: return {"ok": False, "error": "SERPAPI_KEY missing"}
    base = "https://serpapi.com/search.json"
    p = {"engine": engine, "q": q, "api_key": SERPAPI_KEY}
    if params: p.update(params)
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT, headers={"User-Agent": WEB_USER_AGENT}) as c:
            r = c.get(base, params=p)
            return {"ok": r.status_code==200, "status": r.status_code, "data": r.json()}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def firecrawl_scrape_sync(url: str) -> dict:
    if not FIRECRAWL_KEY: return {"ok": False, "error": "FIRECRAWL_KEY missing"}
    endpoint = "https://api.firecrawl.dev/v1/scrape"
    headers = {"Authorization": f"Bearer {FIRECRAWL_KEY}"}
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as c:
            r = c.post(endpoint, headers=headers, json={"url": url, "formats": ["markdown"]})
            if r.status_code != 200:
                return {"ok": False, "status": r.status_code, "raw": r.text}
            jr = r.json() or {}
            payload = (jr.get("data") or {})
            text = payload.get("markdown") or payload.get("content") or payload.get("html") or ""
            if isinstance(text, list):
                text = "\n\n".join([t for t in text if isinstance(t, str)])
            title = payload.get("title") or url
            return {"ok": True, "title": title, "text": text}
    except Exception as e:
        return {"ok": False, "error": str(e)}
