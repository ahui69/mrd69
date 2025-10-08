#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTONAUKA KTÓRA DZIAŁA - bez płatnych API, bez gówna!
- DuckDuckGo (FREE!)
- Wikipedia (FREE!)
- Prawdziwe źródła
- Działa od razu
"""

import re, time, json
from typing import List, Dict, Any
from urllib.parse import quote

def autonauka(query: str, topk: int = 8, **kwargs) -> Dict[str, Any]:
    """
    Ucz się z internetu - DZIAŁAJĄCA wersja!
    Zwraca: {ok, query, context, facts, sources, ...}
    """
    
    # 1. DUCKDUCKGO - wyszukiwanie (FREE!)
    results = []
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=topk):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
    except Exception as e:
        print(f"DDG error: {e}")
    
    # 2. WIKIPEDIA - jako backup (FREE!)
    if len(results) < 3:
        wiki = _wiki_search(query)
        results.extend(wiki[:5])
    
    if not results:
        return {
            "ok": True,
            "query": query,
            "context": "(brak wyników wyszukiwania)",
            "facts": [],
            "sources": [],
            "source_count": 0,
            "powered_by": "autonauka-dziala"
        }
    
    # 3. Buduj kontekst
    facts = []
    sources = []
    
    for i, r in enumerate(results[:topk], 1):
        # Snippet jako fakt
        snippet = r.get("snippet", "").strip()
        if snippet and len(snippet) > 50:
            facts.append(f"{snippet} [{i}]")
        
        # Źródło
        sources.append({
            "title": r.get("title", ""),
            "url": r.get("url", "")
        })
    
    context = "\n\n".join(facts)
    
    return {
        "ok": True,
        "query": query,
        "context": context,
        "facts": [f.replace(f" [{i+1}]", "") for i, f in enumerate(facts)],
        "sources": sources,
        "source_count": len(results),
        "is_deep_research": kwargs.get("deep_research", False),
        "powered_by": "autonauka-dziala"
    }


def _wiki_search(query: str) -> List[Dict]:
    """Wyszukaj w Wikipedii (FREE!)"""
    import requests
    
    try:
        # API Wikipedia
        url = f"https://pl.wikipedia.org/w/api.php"
        params = {
            "action": "opensearch",
            "format": "json",
            "search": query,
            "limit": 5
        }
        
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        
        # Wynik: [query, [titles], [descriptions], [urls]]
        titles = data[1] if len(data) > 1 else []
        descriptions = data[2] if len(data) > 2 else []
        urls = data[3] if len(data) > 3 else []
        
        results = []
        for i in range(len(titles)):
            results.append({
                "title": titles[i] if i < len(titles) else "",
                "snippet": descriptions[i] if i < len(descriptions) else "",
                "url": urls[i] if i < len(urls) else ""
            })
        
        return results
    except Exception as e:
        print(f"Wiki error: {e}")
        return []


# TEST (uruchom: python3 autonauka_dziala.py)
if __name__ == "__main__":
    result = autonauka("Python programowanie", topk=5)
    print(json.dumps(result, ensure_ascii=False, indent=2))
