#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTONAUKA FULL - Wszystkie źródła, pipeline, LTM, newsy, sport
- SERPAPI (płatne ale masz wykupione!)
- Firecrawl (masz wykupione!)
- Wikipedia (FREE)
- DuckDuckGo (FREE)
- Zapis do LTM (nauka w locie)
- Streszczenia przez LLM
- ESPN dla wyników meczów
- Google News dla newsów
"""

import os, re, time, json, hashlib, asyncio
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import quote_plus
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
FIRECRAWL_KEY = os.getenv("FIRECRAWL_KEY", "") or os.getenv("FIRECRAWL_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepinfra.com/v1/openai")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct")

HTTP_TIMEOUT = int(os.getenv("WEB_HTTP_TIMEOUT", "45"))
AUTO_TOPK = int(os.getenv("AUTO_TOPK", "8"))
AUTO_FETCH = int(os.getenv("AUTO_FETCH", "6"))

# ═══════════════════════════════════════════════════════
# HTTP CLIENT
# ═══════════════════════════════════════════════════════

import httpx

def get_client():
    return httpx.AsyncClient(
        timeout=HTTP_TIMEOUT,
        headers={"User-Agent": "MRD69-Autonauka/3.0"},
        follow_redirects=True
    )

# ═══════════════════════════════════════════════════════
# LLM - Streszczenia
# ═══════════════════════════════════════════════════════

async def llm_summarize(text: str, query: str) -> str:
    """LLM tworzy streszczenie z źródeł"""
    if not LLM_API_KEY:
        return text[:1500]  # Fallback - przytnij
    
    prompt = f"""Pytanie użytkownika: {query}

Informacje ze źródeł:
{text[:8000]}

Zadanie: Stwórz zwięzłe, konkretne streszczenie odpowiadające na pytanie.
- Cytuj źródła używając [1], [2] etc
- Wyciągnij najważniejsze fakty
- Po polsku, 3-5 akapitów max
"""
    
    try:
        async with get_client() as client:
            resp = await client.post(
                LLM_BASE_URL.rstrip("/") + "/chat/completions",
                headers={
                    "Authorization": f"Bearer {LLM_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": "Jesteś asystentem research. Tworzysz streszczenia z cytowaniem źródeł."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1200
                },
                timeout=60
            )
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"LLM error: {e}")
        return text[:1500]

# ═══════════════════════════════════════════════════════
# ŹRÓDŁA - SERPAPI
# ═══════════════════════════════════════════════════════

async def serpapi_search(query: str, limit: int = 10) -> List[Dict]:
    """SERPAPI - najlepsze wyniki Google"""
    if not SERPAPI_KEY:
        return []
    
    try:
        async with get_client() as client:
            resp = await client.get(
                "https://serpapi.com/search.json",
                params={
                    "q": query,
                    "api_key": SERPAPI_KEY,
                    "num": limit,
                    "hl": "pl"
                }
            )
            data = resp.json()
            
            results = []
            for item in data.get("organic_results", [])[:limit]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "serpapi"
                })
            return results
    except Exception as e:
        print(f"SERPAPI error: {e}")
        return []

# ═══════════════════════════════════════════════════════
# ŹRÓDŁA - DUCKDUCKGO
# ═══════════════════════════════════════════════════════

async def duckduckgo_search(query: str, limit: int = 10) -> List[Dict]:
    """DuckDuckGo - FREE backup"""
    try:
        from duckduckgo_search import DDGS
        
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=limit):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                    "source": "duckduckgo"
                })
        return results
    except Exception as e:
        print(f"DDG error: {e}")
        return []

# ═══════════════════════════════════════════════════════
# ŹRÓDŁA - WIKIPEDIA
# ═══════════════════════════════════════════════════════

async def wikipedia_search(query: str, limit: int = 3) -> List[Dict]:
    """Wikipedia PL + EN"""
    results = []
    
    for lang in ['pl', 'en']:
        try:
            async with get_client() as client:
                resp = await client.get(
                    f"https://{lang}.wikipedia.org/w/api.php",
                    params={
                        "action": "opensearch",
                        "format": "json",
                        "search": query,
                        "limit": limit
                    }
                )
                data = resp.json()
                
                titles = data[1] if len(data) > 1 else []
                descriptions = data[2] if len(data) > 2 else []
                urls = data[3] if len(data) > 3 else []
                
                for i in range(len(titles)):
                    results.append({
                        "title": titles[i] if i < len(titles) else "",
                        "snippet": descriptions[i] if i < len(descriptions) else "",
                        "url": urls[i] if i < len(urls) else "",
                        "source": f"wikipedia-{lang}"
                    })
        except Exception as e:
            print(f"Wiki {lang} error: {e}")
    
    return results[:limit]

# ═══════════════════════════════════════════════════════
# SCRAPING - FIRECRAWL
# ═══════════════════════════════════════════════════════

async def firecrawl_scrape(url: str) -> Optional[str]:
    """Firecrawl - pobiera pełną treść strony"""
    if not FIRECRAWL_KEY:
        return None
    
    try:
        async with get_client() as client:
            resp = await client.post(
                "https://api.firecrawl.dev/v1/scrape",
                headers={"Authorization": f"Bearer {FIRECRAWL_KEY}"},
                json={
                    "url": url,
                    "formats": ["markdown", "html"]
                },
                timeout=30
            )
            
            if resp.status_code != 200:
                return None
            
            data = resp.json()
            content = data.get("data", {})
            text = content.get("markdown") or content.get("content") or content.get("html") or ""
            
            # Czyszczenie
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text[:10000]  # Max 10k znaków
    except Exception as e:
        print(f"Firecrawl error for {url}: {e}")
        return None

# ═══════════════════════════════════════════════════════
# PIPELINE - Zbierz ze wszystkich źródeł
# ═══════════════════════════════════════════════════════

async def research_pipeline(query: str, deep: bool = False) -> Dict[str, Any]:
    """
    GŁÓWNY PIPELINE:
    1. Szukaj we wszystkich źródłach równolegle
    2. Pobierz treści (Firecrawl dla top wyników)
    3. Ranking + dedup
    4. Streszczenie przez LLM
    5. Zapis do LTM
    """
    
    print(f"🔍 Research pipeline: {query}")
    
    # 1. WYSZUKIWANIE - wszystkie źródła równolegle!
    tasks = [
        serpapi_search(query, limit=12),
        duckduckgo_search(query, limit=10),
        wikipedia_search(query, limit=5)
    ]
    
    if deep:
        # Dodatkowe zapytania dla głębokiego research
        tasks.append(serpapi_search(f"{query} tutorial", limit=5))
        tasks.append(serpapi_search(f"{query} examples", limit=5))
    
    all_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Połącz wyniki
    combined = []
    for result in all_results:
        if isinstance(result, list):
            combined.extend(result)
    
    # 2. DEDUP po URL
    seen_urls = set()
    unique_results = []
    
    for r in combined:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(r)
    
    print(f"📊 Znaleziono {len(unique_results)} unikalnych źródeł")
    
    # 3. SCRAPING - pobierz pełne treści top wyników
    top_results = unique_results[:AUTO_FETCH]
    
    scrape_tasks = []
    for r in top_results:
        if r.get("source") == "serpapi":  # Priorytet dla SERPAPI
            scrape_tasks.append(firecrawl_scrape(r["url"]))
    
    scraped_contents = await asyncio.gather(*scrape_tasks, return_exceptions=True)
    
    # Dodaj scraped content do results
    for i, content in enumerate(scraped_contents):
        if isinstance(content, str) and content:
            top_results[i]["full_text"] = content
    
    # 4. BUDUJ KONTEKST
    context_parts = []
    sources = []
    facts = []
    
    for idx, r in enumerate(top_results[:AUTO_TOPK], 1):
        # Snippet lub full text
        text = r.get("full_text") or r.get("snippet") or ""
        
        if text and len(text) > 50:
            # Dodaj do kontekstu z numerem źródła
            context_parts.append(f"[{idx}] {text[:1500]}")
            
            # Wyciągnij fakty (zdania)
            sentences = re.split(r'[.!?]+', text)
            for sent in sentences[:3]:
                sent = sent.strip()
                if len(sent) > 60:
                    facts.append(sent)
        
        # Źródło
        sources.append({
            "id": idx,
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "source_type": r.get("source", "")
        })
    
    context = "\n\n".join(context_parts)
    
    # 5. STRESZCZENIE przez LLM
    summary = await llm_summarize(context, query)
    
    # 6. ZAPIS DO LTM (jeśli moduł dostępny)
    ltm_saved = []
    try:
        # Import monolit dla dostępu do ltm_add
        import sys
        sys.path.insert(0, '/workspace/mrd69')
        from monolit import ltm_add
        
        # Zapisz top fakty
        for fact in facts[:10]:
            if len(fact) > 60:
                fact_id = ltm_add(fact, tags=f"autonauka,web,{query[:30]}", conf=0.75)
                ltm_saved.append(fact_id)
        
        # Zapisz streszczenie
        ltm_add(f"[RESEARCH] {query}: {summary[:500]}", tags="autonauka,summary,research", conf=0.85)
        
        print(f"💾 Zapisano {len(ltm_saved)} faktów do LTM")
    except Exception as e:
        print(f"LTM save error: {e}")
    
    return {
        "ok": True,
        "query": query,
        "summary": summary,
        "context": context,
        "facts": facts[:15],
        "sources": sources,
        "source_count": len(top_results),
        "ltm_saved": len(ltm_saved),
        "powered_by": "autonauka-full-pipeline"
    }

# ═══════════════════════════════════════════════════════
# NEWSY - Google News przez SERPAPI
# ═══════════════════════════════════════════════════════

async def get_news(query: str = "świat", limit: int = 10) -> Dict[str, Any]:
    """Pobierz najnowsze newsy"""
    
    if not SERPAPI_KEY:
        return {"ok": False, "error": "Brak SERPAPI_KEY"}
    
    try:
        async with get_client() as client:
            resp = await client.get(
                "https://serpapi.com/search.json",
                params={
                    "engine": "google_news",
                    "q": query,
                    "api_key": SERPAPI_KEY,
                    "hl": "pl",
                    "gl": "pl",
                    "num": limit
                }
            )
            data = resp.json()
            
            news = []
            for item in data.get("news_results", [])[:limit]:
                news.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", ""),
                    "source": item.get("source", ""),
                    "date": item.get("date", ""),
                    "thumbnail": item.get("thumbnail", "")
                })
            
            return {
                "ok": True,
                "query": query,
                "news": news,
                "count": len(news)
            }
    except Exception as e:
        print(f"News error: {e}")
        return {"ok": False, "error": str(e)}

# ═════════════════════════════════════════════════════
# SPORT - Wyniki meczów (ESPN API - FREE!)
# ═══════════════════════════════════════════════════════

async def get_football_scores(league: str = "ita.1") -> Dict[str, Any]:
    """
    Pobierz wyniki meczów piłkarskich
    Ligi:
    - ita.1 = Serie A (Juventus!)
    - eng.1 = Premier League
    - esp.1 = La Liga
    - ger.1 = Bundesliga
    - uefa.champions = Champions League
    """
    
    try:
        async with get_client() as client:
            resp = await client.get(
                f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard"
            )
            data = resp.json()
            
            games = []
            for event in data.get("events", []):
                competition = event.get("competitions", [{}])[0]
                status = competition.get("status", {}).get("type", {})
                
                teams = competition.get("competitors", [])
                if len(teams) >= 2:
                    home = teams[0]
                    away = teams[1]
                    
                    game_info = {
                        "home_team": home.get("team", {}).get("displayName", ""),
                        "home_score": home.get("score", ""),
                        "away_team": away.get("team", {}).get("displayName", ""),
                        "away_score": away.get("score", ""),
                        "status": status.get("shortDetail", ""),
                        "state": status.get("state", ""),
                        "start_date": event.get("date", ""),
                        "venue": competition.get("venue", {}).get("fullName", "")
                    }
                    
                    games.append(game_info)
            
            return {
                "ok": True,
                "league": league,
                "games": games,
                "count": len(games)
            }
    except Exception as e:
        print(f"Football scores error: {e}")
        return {"ok": False, "error": str(e)}

# ═══════════════════════════════════════════════════════
# JUVENTUS TRACKER - Specjalnie dla Ciebie!
# ═══════════════════════════════════════════════════════

async def track_juventus() -> Dict[str, Any]:
    """
    Śledź Juventus:
    - Ostatni mecz + wynik
    - Następny mecz
    - Pozycja w tabeli
    """
    
    try:
        # Pobierz wyniki Serie A
        scores = await get_football_scores("ita.1")
        
        if not scores.get("ok"):
            return scores
        
        juventus_games = []
        last_game = None
        next_game = None
        
        for game in scores.get("games", []):
            # Sprawdź czy Juventus gra
            is_juve_home = "Juventus" in game.get("home_team", "")
            is_juve_away = "Juventus" in game.get("away_team", "")
            
            if is_juve_home or is_juve_away:
                juventus_games.append(game)
                
                # Ostatni mecz (zakończony)
                if game.get("state") == "post" and not last_game:
                    last_game = game
                
                # Następny mecz
                if game.get("state") in ["pre", "in"] and not next_game:
                    next_game = game
        
        # Przygotuj komunikat
        message = "⚽ **JUVENTUS UPDATE**\n\n"
        
        if last_game:
            home = last_game["home_team"]
            away = last_game["away_team"]
            score_home = last_game["home_score"]
            score_away = last_game["away_score"]
            
            is_juve_home = "Juventus" in home
            juve_score = score_home if is_juve_home else score_away
            opponent_score = score_away if is_juve_home else score_home
            opponent = away if is_juve_home else home
            
            result = "WYGRANA" if juve_score > opponent_score else ("REMIS" if juve_score == opponent_score else "PRZEGRANA")
            
            message += f"🏁 Ostatni mecz: {result}\n"
            message += f"Juventus {juve_score} - {opponent_score} {opponent}\n"
            message += f"Status: {last_game.get('status', '')}\n\n"
        
        if next_game:
            home = next_game["home_team"]
            away = next_game["away_team"]
            start = next_game.get("start_date", "")
            
            message += f"📅 Następny mecz:\n"
            message += f"{home} vs {away}\n"
            if start:
                try:
                    dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                    message += f"Data: {dt.strftime('%d.%m.%Y %H:%M')}\n"
                except:
                    pass
            message += f"Stadion: {next_game.get('venue', 'TBA')}\n"
        
        return {
            "ok": True,
            "message": message,
            "last_game": last_game,
            "next_game": next_game,
            "all_games": juventus_games
        }
        
    except Exception as e:
        print(f"Juventus tracker error: {e}")
        return {"ok": False, "error": str(e)}

# ═══════════════════════════════════════════════════════
# MAIN API - autonauka()
# ═══════════════════════════════════════════════════════

def autonauka(query: str, topk: int = AUTO_TOPK, deep_research: bool = False, **kwargs) -> Dict[str, Any]:
    """
    GŁÓWNA FUNKCJA - research z wszystkich źródeł
    
    Args:
        query: Zapytanie
        topk: Ile źródeł zwrócić
        deep_research: Czy głęboki research (więcej źródeł)
    
    Returns:
        {ok, query, summary, context, facts, sources, ...}
    """
    
    # Run async pipeline
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(research_pipeline(query, deep=deep_research))
        return result
    finally:
        loop.close()

# ═══════════════════════════════════════════════════════
# SYNC WRAPPERS (dla monolit.py)
# ═══════════════════════════════════════════════════════

def get_news_sync(query: str = "świat", limit: int = 10) -> Dict[str, Any]:
    """Newsy - sync wrapper"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(get_news(query, limit))
    finally:
        loop.close()

def track_juventus_sync() -> Dict[str, Any]:
    """Juventus tracker - sync wrapper"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(track_juventus())
    finally:
        loop.close()

def get_football_scores_sync(league: str = "ita.1") -> Dict[str, Any]:
    """Football scores - sync wrapper"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(get_football_scores(league))
    finally:
        loop.close()

# ═══════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "Python programowanie"
    
    print(f"🔍 Testuję autonauka: {query}\n")
    
    result = autonauka(query, topk=5)
    
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # Test Juventus
    print("\n\n⚽ JUVENTUS TRACKER:\n")
    juve = track_juventus_sync()
    print(juve.get("message", "Brak danych"))
