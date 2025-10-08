"""
DODAJ TE ENDPOINTY DO monolit.py

Wklej po linii gdzie są inne endpointy (około linia 2500+)
Przed: if __name__ == "__main__":
"""

# ═══════════════════════════════════════════════════════
# IMPORT na początku pliku (dodaj do importów)
# ═══════════════════════════════════════════════════════
"""
try:
    import autonauka_full
    AUTONAUKA_AVAILABLE = True
except Exception:
    AUTONAUKA_AVAILABLE = False
"""

# ═══════════════════════════════════════════════════════
# ENDPOINTY - dodaj do FastAPI app
# ═══════════════════════════════════════════════════════

@app.post("/api/auto/learn")
async def auto_learn_endpoint(request: Request):
    """Web learning - ucz się z internetu"""
    try:
        body = await request.json()
        query = body.get("q") or body.get("query", "")
        
        if not query:
            raise HTTPException(status_code=400, detail="Brak parametru 'q' lub 'query'")
        
        deep = body.get("deep_research", False) or body.get("deep", False)
        
        if AUTONAUKA_AVAILABLE:
            result = autonauka_full.autonauka(query, topk=10, deep_research=deep)
            return result
        else:
            # Fallback do wbudowanej autonauka jeśli jest
            result = await autonauka(query, topk=10, deep_research=deep)
            return result
            
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/api/news")
async def news_endpoint(q: str = "świat", limit: int = 10):
    """Pobierz najnowsze newsy"""
    try:
        if AUTONAUKA_AVAILABLE:
            result = autonauka_full.get_news_sync(query=q, limit=limit)
            return result
        else:
            return {"ok": False, "error": "Moduł autonauka_full niedostępny"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/api/sport/football")
async def football_scores_endpoint(league: str = "ita.1"):
    """
    Wyniki meczów piłkarskich
    
    Ligi:
    - ita.1 = Serie A (Juventus!)
    - eng.1 = Premier League  
    - esp.1 = La Liga
    - ger.1 = Bundesliga
    - uefa.champions = Liga Mistrzów
    """
    try:
        if AUTONAUKA_AVAILABLE:
            result = autonauka_full.get_football_scores_sync(league=league)
            return result
        else:
            return {"ok": False, "error": "Moduł autonauka_full niedostępny"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/api/sport/juventus")
async def juventus_tracker_endpoint():
    """
    JUVENTUS TRACKER - specjalnie dla Ciebie!
    Zwraca: ostatni mecz, następny mecz, podsumowanie
    """
    try:
        if AUTONAUKA_AVAILABLE:
            result = autonauka_full.track_juventus_sync()
            
            # BONUS: Zapisz do LTM jeśli był mecz
            if result.get("ok") and result.get("last_game"):
                try:
                    from monolit import ltm_add
                    ltm_add(
                        result.get("message", ""),
                        tags="sport,juventus,football,serie_a",
                        conf=0.9
                    )
                except:
                    pass
            
            return result
        else:
            return {"ok": False, "error": "Moduł autonauka_full niedostępny"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ═══════════════════════════════════════════════════════
# WEBHOOK - Auto-check Juventus co godzinę
# ═══════════════════════════════════════════════════════

import threading
import time as _time

def juventus_auto_checker():
    """Background task - sprawdza Juventus co godzinę"""
    while True:
        try:
            _time.sleep(3600)  # Co godzinę
            
            if AUTONAUKA_AVAILABLE:
                result = autonauka_full.track_juventus_sync()
                
                if result.get("ok") and result.get("last_game"):
                    # Zapisz do LTM
                    try:
                        from monolit import ltm_add, psy_episode_add
                        ltm_add(
                            f"[AUTO] {result.get('message', '')}",
                            tags="sport,juventus,auto_update",
                            conf=0.85
                        )
                        
                        # Dodaj jako psyche event
                        psy_episode_add(
                            "system",
                            "juventus_update",
                            valence=1.0,  # Pozytywny event (dla fana!)
                            intensity=0.7,
                            tags="sport,juventus",
                            note=result.get("message", "")[:200]
                        )
                        
                        print(f"⚽ AUTO-UPDATE: Juventus tracked and saved to LTM")
                    except Exception as e:
                        print(f"Auto-save error: {e}")
        except Exception as e:
            print(f"Juventus auto-checker error: {e}")

# Start background thread
_juve_thread = threading.Thread(target=juventus_auto_checker, daemon=True)
_juve_thread.start()
print("⚽ Juventus auto-tracker uruchomiony (sprawdza co godzinę)")
