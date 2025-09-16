"""
psychika.py — moduł decyzyjno-afektywny projektu mrd69 (v2.5 PRO) z SYSTEMEM NIEZAWODNOŚCI

Cel: prawdziwy "mózg wykonawczy" pracujący w tle i wspierający główny model,
     ale NIE podejmujący za niego ostatecznych decyzji (zgodnie z kontraktem).

NOWA FUNKCJONALNOŚĆ v2.6:
- Integracja z reliability_core.py - wszystkie operacje przez system niezawodności
- Kontrakty pamięci z fallbackami
- Backpressure na autopilot
- Idempotencja akcji z TTL
- Telemetria z batch processing
- Etyczny trigger system z pełną kontrolą

Najważniejsze cechy:
- Spójny model psychiki (human factors + wagi + presety).
- Autopilot generujący realistyczne propozycje (multi-source: goals, STM, recall).
- Skoring akcji: utylitarystyczny + ryzyko + koszt + novelty + afekt + świeżość.
- Eksploracja "bandit": Thompson sampling ze stanem utrwalanym w pamięci (profile).
- Decyzje: mini-LLM (Qwen4B) → fallback main LLM → fallback heurystyka (zero blokad).
- Integracja z memory.py v2 (RAG HYBRYDA): add_fact / episodes / goals / profile / meta.
- Brak zależności od UI. Metody są stabilne (API) i gotowe do wołania z main.py.
- SYSTEM NIEZAWODNOŚCI zawsze aktywny i kontrolujący wszystkie operacje.

ENV (wykorzystujemy to co już masz w projekcie; nic nowego "must-have"):
  # główne LLM:
  LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, TIMEOUT_HTTP / LLM_HTTP_TIMEOUT_S
  # mini LLM do tła:
  MINI_LLM_BASE_URL, MINI_LLM_API_KEY, MINI_LLM_MODEL   (np. Qwen/Qwen2.5-4B-Instruct)
  # pamięć:
  MEM_NS, PSY_ENCRYPT_KEY (pośrednio przez memory.py)
  # psychika (wagi i sterowanie):
  PSY_PRESET, PSY_COMPASSION, PSY_DUTY, PSY_HONOR, PSY_MORALITY,
  PSY_EXPLORATION, PSY_GOAL_WEIGHT, PSY_RISK_AVERSION, PSY_BANDIT,
  PSY_MAX_OPTIONS, PSY_PRIDE, PSY_RECIPROCITY, PSY_RELIEF, PSY_REGRET,
  PSY_PARETO, PSY_UNWRAP, PSY_SEED, PSY_DEBUG,
  PSY_AUTOPILOT (1/0), PSY_AUTOPILOT_INTERVAL (sek), PSY_AUTOPILOT_HARD_LIMIT (sek)
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Any

# === SYSTEM NIEZAWODNOŚCI - ZAWSZE PIERWSZA INICJALIZACJA ===
try:
    from reliability_core import (
        get_reliable_system,
        reliable_operation,
        require_reliability_check,
    )

    print("✅ System niezawodności załadowany do psychika.py")
except ImportError as e:
    print(f"🚨 KRYTYCZNY BŁĄD: Nie można załadować systemu niezawodności: {e}")
    print("🚨 PSYCHIKA NIE MOŻE DZIAŁAĆ BEZ SYSTEMU NIEZAWODNOŚCI")
    raise SystemExit(1) from e

# Ładowanie zmiennych środowiskowych z pliku .env
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not found. Skipping .env file loading.")


# ──────────────────────────────────────────────────────────────────────────
# ENV / CONFIG
# ──────────────────────────────────────────────────────────────────────────
def _envf(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _envi(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _envb(name: str, default: bool) -> bool:
    v = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


PSY_DEBUG = _envb("PSY_DEBUG", True)
PSY_PRESET = os.getenv("PSY_PRESET", "balanced").strip()

PSY_COMPASSION = _envf("PSY_COMPASSION", 0.6)
PSY_DUTY = _envf("PSY_DUTY", 0.45)
PSY_HONOR = _envf("PSY_HONOR", 0.5)
PSY_MORALITY = _envf("PSY_MORALITY", 0.6)
PSY_EXPLORATION = _envf("PSY_EXPLORATION", 0.55)
PSY_GOAL_WEIGHT = _envf("PSY_GOAL_WEIGHT", 0.55)
PSY_RISK_AVERSION = _envf("PSY_RISK_AVERSION", 0.40)
PSY_BANDIT = _envf("PSY_BANDIT", 0.80)  # eksploracja
PSY_MAX_OPTIONS = _envi("PSY_MAX_OPTIONS", 6)
PSY_PRIDE = _envf("PSY_PRIDE", 0.5)
PSY_RECIPROCITY = _envf("PSY_RECIPROCITY", 0.6)
PSY_RELIEF = _envf("PSY_RELIEF", 0.6)
PSY_REGRET = _envf("PSY_REGRET", 0.35)
PSY_PARETO = _envf("PSY_PARETO", 0.55)
PSY_UNWRAP = _envf("PSY_UNWRAP", 1.0)
PSY_SEED = _envi("PSY_SEED", 69)

PSY_AUTOPILOT = _envb("PSY_AUTOPILOT", True)
PSY_AUTOPILOT_INTERVAL = _envi("PSY_AUTOPILOT_INTERVAL", 120)  # miękka przerwa
# twarda granica minimalna
PSY_AUTOPILOT_HARD = _envi("PSY_AUTOPILOT_HARD_LIMIT", 45)

HTTP_TIMEOUT = _envi("LLM_HTTP_TIMEOUT_S", _envi("TIMEOUT_HTTP", 60))

# LLM (main)
LLM_BASE = (os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL", "") or "").rstrip("/")
LLM_KEY = "jrfw3KR2aIYF00ACuyj4qjLan5jDP5dX"
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()

# MINI LLM (bg)
MINI_BASE = (os.getenv("MINI_LLM_BASE_URL") or LLM_BASE or "").rstrip("/")
MINI_KEY = "jrfw3KR2aIYF00ACuyj4qjLan5jDP5dX"
MINI_MODEL = (os.getenv("MINI_LLM_MODEL") or "Qwen/Qwen1.5-4B-Chat").strip()

random.seed(PSY_SEED)


# ──────────────────────────────────────────────────────────────────────────
# LOG
# ──────────────────────────────────────────────────────────────────────────
def _dbg(*a: Any) -> None:
    if PSY_DEBUG:
        # Zapis do pliku zamiast na konsolę
        with open("psychika_debug.log", "a", encoding="utf-8") as f:
            f.write(f"[{time.time():.2f}] " + " ".join(map(str, a)) + "\n")


# ──────────────────────────────────────────────────────────────────────────
# LLM adapters z SYSTEMEM NIEZAWODNOŚCI
# ──────────────────────────────────────────────────────────────────────────
@reliable_operation("llm_chat")
@require_reliability_check
def _llm_chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    base: str | None = None,
    key: str | None = None,
    maxtok: int = 512,
    temp: float = 0.2,
) -> str | None:
    base = (base or LLM_BASE).rstrip("/")
    key = (key or LLM_KEY).strip()
    model = (model or LLM_MODEL).strip()
    if not (base and key and model):
        _dbg("llm_chat abort: missing base, key, or model")
        return None
    url = base + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temp,
        "max_tokens": maxtok,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        import requests

        r = requests.post(url, json=payload, headers=headers, timeout=HTTP_TIMEOUT)
        if r.status_code >= 400:
            _dbg("llm_chat http", r.status_code, r.text[:200])
            return None
        j = r.json()
        return j.get("choices", [{}])[0].get("message", {}).get("content", "").strip() or None
    except Exception as e:
        _dbg("llm_chat err", str(e)[:200])
        return None


@reliable_operation("mini_llm_chat")
@require_reliability_check
def _mini_chat(messages: list[dict[str, str]], maxtok: int = 320, temp: float = 0.0) -> str | None:
    base = MINI_BASE.rstrip("/")
    if not (base and MINI_KEY and MINI_MODEL):
        _dbg("mini_chat abort: missing base, key, or model")
        return None
    _dbg(f"mini_chat call: base={base}, model={MINI_MODEL}, key_present={bool(MINI_KEY)}")
    url = base + "/chat/completions"
    payload = {
        "model": MINI_MODEL,
        "messages": messages,
        "temperature": temp,
        "max_tokens": maxtok,
    }
    headers = {
        "Authorization": f"Bearer {MINI_KEY}",
        "Content-Type": "application/json",
    }
    try:
        import requests

        r = requests.post(url, json=payload, headers=headers, timeout=HTTP_TIMEOUT)
        if r.status_code >= 400:
            _dbg("mini_chat http", r.status_code, r.text[:200])
            return None
        j = r.json()
        return j.get("choices", [{}])[0].get("message", {}).get("content", "").strip() or None
    except Exception as e:
        _dbg("mini_chat err", str(e)[:200])
        return None


# ──────────────────────────────────────────────────────────────────────────
# MEMORY adapter z KONTRAKTAMI NIEZAWODNOŚCI (zgodne z memory.py v2)
# ──────────────────────────────────────────────────────────────────────────
@require_reliability_check
def _memory():
    """
    Lazy import + walidacja kontraktów, aby nie tworzyć cykli.
    Oczekujemy memory.get_memory() z pełną walidacją API.
    """
    try:
        reliable_system = get_reliable_system()

        # Import pamięci
        try:
            from .memory import get_memory  # jeśli psychika.py w pakiecie overmind
        except ImportError:
            # Fallback to direct import if relative import fails
            from memory import get_memory

        mem_obj = get_memory()

        # Walidacja kontraktów przez system niezawodności
        validation_result = reliable_system.memory_contract.validate_memory_object(mem_obj)

        if validation_result["fallback_rate"] > 0.5:
            print(
                f"⚠️ UWAGA: Wysoki poziom fallbacków w API pamięci: {validation_result['fallback_rate']:.1%}"
            )

        return mem_obj

    except Exception as e:
        print(f"❌ KRYTYCZNY BŁĄD dostępu do pamięci: {e}")
        # W sytuacji krytycznej nie można kontynuować
        raise RuntimeError(f"Nie można uzyskać dostępu do pamięci: {e}")


@reliable_operation("memory_add")
def _mem_add(text: str, tags: list[str] | None = None, conf: float = 0.6) -> None:
    """Dodaje fakt do pamięci przez system niezawodności."""
    mem = _memory()
    reliable_system = get_reliable_system()

    # Przez safe_call z kontraktem
    result = reliable_system.memory_contract.safe_call(
        mem, "add_fact", text.strip(), tags=tags or [], conf=conf
    )

    if result is None:
        print(f"⚠️ Dodawanie faktu może nie być udane: {text[:50]}...")


@reliable_operation("memory_meta")
def _mem_meta(kind: str, payload: dict[str, Any]) -> None:
    mem = _memory()
    if hasattr(mem, "_meta_event"):
        try:
            # prywatne, ale dostępne w naszym projekcie
            # pylint: disable=protected-access
            mem._meta_event(kind, payload)
        except RuntimeError:
            # Handle expected runtime errors during meta event processing
            pass


# ──────────────────────────────────────────────────────────────────────────
# STAN PSYCHIKI / HUMAN FACTORS
# ──────────────────────────────────────────────────────────────────────────
class PsyState:
    """
    Wewnętrzny stan czynników ludzkich i wag psychiki (exponential smoothing).
    """

    def __init__(self):
        self.fatigue = 0.18  # 0..1
        self.focus = 0.62
        self.mood = 0.55
        self.stress = 0.22
        self.conf_bias = 0.00
        self.last_tick = time.time()
        self.last_autopilot = 0.0

        # wagi
        self.w = dict(
            compassion=PSY_COMPASSION,
            duty=PSY_DUTY,
            honor=PSY_HONOR,
            morality=PSY_MORALITY,
            exploration=PSY_EXPLORATION,
            goal_weight=PSY_GOAL_WEIGHT,
            risk_aversion=PSY_RISK_AVERSION,
            bandit=PSY_BANDIT,
            pride=PSY_PRIDE,
            reciprocity=PSY_RECIPROCITY,
            relief=PSY_RELIEF,
            regret=PSY_REGRET,
            pareto=PSY_PARETO,
            unwrap=PSY_UNWRAP,
        )
        self.apply_preset(PSY_PRESET)

    def apply_preset(self, preset: str):
        p = (preset or "").lower().strip()
        if p in ("creative_max", "joker", "hacker", "storyteller"):
            self.w["exploration"] = max(self.w["exploration"], 0.88)
            self.w["risk_aversion"] = min(self.w["risk_aversion"], 0.25)
            self.w["goal_weight"] = max(self.w["goal_weight"], 0.60)
            self.w["bandit"] = max(self.w["bandit"], 0.90)
        elif p in ("safe", "cautious"):
            self.w["risk_aversion"] = max(self.w["risk_aversion"], 0.70)
            self.w["exploration"] = min(self.w["exploration"], 0.35)
            self.w["bandit"] = min(self.w["bandit"], 0.40)

    def smooth_update(self, fatigue=None, focus=None, mood=None, stress=None, alpha: float = 0.22):
        def s(cur, new):
            return (1.0 - alpha) * cur + alpha * new

        if fatigue is not None:
            self.fatigue = s(self.fatigue, max(0.0, min(1.0, float(fatigue))))
        if focus is not None:
            self.focus = s(self.focus, max(0.0, min(1.0, float(focus))))
        if mood is not None:
            self.mood = s(self.mood, max(0.0, min(1.0, float(mood))))
        if stress is not None:
            self.stress = s(self.stress, max(0.0, min(1.0, float(stress))))
        self.last_tick = time.time()

    def learn_from_outcome(self, decision_quality: float, execution_success: bool):
        """
        Uczenie w czasie: psychika reaguje na skutki decyzji.
        decision_quality: 0.0-1.0 (jak dobra była decyzja)
        execution_success: czy akcja się udała
        """
        # Bazowy wpływ na podstawie jakości decyzji
        mood_delta = (decision_quality - 0.5) * 0.15
        stress_delta = -(decision_quality - 0.5) * 0.12

        # Dodatkowy wpływ sukcesu/porażki wykonania
        if execution_success:
            mood_delta += 0.08
            stress_delta -= 0.06
            self.focus = min(1.0, self.focus + 0.04)  # sukces poprawia koncentrację
        else:
            mood_delta -= 0.12
            stress_delta += 0.10
            self.fatigue = min(1.0, self.fatigue + 0.05)  # porażka męczy

        # Aplikuj zmiany
        self.mood = max(0.0, min(1.0, self.mood + mood_delta))
        self.stress = max(0.0, min(1.0, self.stress + stress_delta))

        # Zapisz uczenie do pamięci
        _mem_add(
            f"[LEARNING] decyzja={decision_quality:.2f}, sukces={execution_success}, "
            f"mood_Δ={mood_delta:.3f}, stress_Δ={stress_delta:.3f}",
            tags=["learning", "psyche_adaptation"],
            conf=0.7,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "fatigue": self.fatigue,
            "focus": self.focus,
            "mood": self.mood,
            "stress": self.stress,
            "conf_bias": self.conf_bias,
            "weights": dict(self.w),
            "preset": PSY_PRESET,
            "last_autopilot": self.last_autopilot,
            "ts": self.last_tick,
        }


STATE = PsyState()


# ──────────────────────────────────────────────────────────────────────────
# PAMIĘĆ EPIZODYCZNA - kontekstowe uczenie
# ──────────────────────────────────────────────────────────────────────────
def _record_episode(action: dict[str, Any], outcome: dict[str, Any]) -> None:
    """Zapisuje epizod: akcja + wynik do pamięci dla przyszłego kontekstowego uczenia."""
    episode_data = {
        "action_kind": action.get("kind", ""),
        "action_score": outcome.get("decision_quality", 0.5),
        "execution_success": outcome.get("execution_success", False),
        "context_snapshot": {
            "mood": STATE.mood,
            "stress": STATE.stress,
            "focus": STATE.focus,
            "fatigue": STATE.fatigue,
        },
        "timestamp": time.time(),
    }

    # Zapisz jako fact z tagami epizodycznymi
    episode_text = (
        f"[EPISODE] {action.get('kind', 'unknown')} "
        f"→ jakość={outcome.get('decision_quality', 0.5):.2f}, "
        f"sukces={outcome.get('execution_success', False)}"
    )
    _mem_add(episode_text, tags=["episode", "learning", action.get("kind", "unknown")], conf=0.75)

    # Zapisz szczegółowe dane jako meta
    _mem_meta("episode_record", episode_data)


def _recall_similar_episodes(
    action_kind: str, context_similarity_threshold: float = 0.3
) -> list[dict]:
    """
    Szuka podobnych epizodów z przeszłości dla danego typu akcji.
    Zwraca lekcje: 'aha, już kiedyś próbowałem podobnego i było źle'.
    """
    mem = _memory()
    try:
        if hasattr(mem, "search_facts"):
            # Szukaj faktów z tagiem epizodycznym dla tego rodzaju akcji
            episodes = mem.search_facts(
                query=f"[EPISODE] {action_kind}", limit=10, tags=["episode", action_kind]
            )

            similar_episodes = []
            current_context = {
                "mood": STATE.mood,
                "stress": STATE.stress,
                "focus": STATE.focus,
                "fatigue": STATE.fatigue,
            }

            for ep in episodes:
                # Proste podobieństwo kontekstu (różnica stanów psychicznych)
                if "context_snapshot" in ep:
                    ctx = ep["context_snapshot"]
                    similarity = 1.0 - sum(
                        abs(current_context[k] - ctx.get(k, 0.5)) for k in current_context
                    ) / len(current_context)

                    if similarity >= context_similarity_threshold:
                        similar_episodes.append(
                            {
                                "episode": ep,
                                "similarity": similarity,
                                "lesson": f"Poprzednia próba: {'sukces' if ep.get('execution_success') else 'porażka'}",
                            }
                        )

            return sorted(similar_episodes, key=lambda x: x["similarity"], reverse=True)[:3]
    except Exception as e:
        _dbg("recall_similar_episodes error", e)

    return []


# ──────────────────────────────────────────────────────────────────────────
# META-REFLEKSJA - adaptacja charakteru na podstawie trendów
# ──────────────────────────────────────────────────────────────────────────
def _analyze_decision_trends(lookback_episodes: int = 15) -> dict[str, Any]:
    """Analizuje ostatnie decyzje i wyciąga trendy jakości."""
    mem = _memory()
    try:
        if hasattr(mem, "search_facts"):
            recent_episodes = mem.search_facts(
                query="[EPISODE]", limit=lookback_episodes, tags=["episode"]
            )

            if len(recent_episodes) < 5:
                return {"insufficient_data": True, "episodes_count": len(recent_episodes)}

            # Analiza trendów
            success_rate = sum(
                1 for ep in recent_episodes if ep.get("execution_success", False)
            ) / len(recent_episodes)
            avg_quality = sum(ep.get("action_score", 0.5) for ep in recent_episodes) / len(
                recent_episodes
            )

            # Trendy w czasie (czy się poprawia czy pogarsza)
            if len(recent_episodes) >= 10:
                first_half = recent_episodes[: len(recent_episodes) // 2]
                second_half = recent_episodes[len(recent_episodes) // 2 :]

                trend_success = sum(
                    1 for ep in second_half if ep.get("execution_success", False)
                ) / len(second_half) - sum(
                    1 for ep in first_half if ep.get("execution_success", False)
                ) / len(
                    first_half
                )
                trend_quality = sum(ep.get("action_score", 0.5) for ep in second_half) / len(
                    second_half
                ) - sum(ep.get("action_score", 0.5) for ep in first_half) / len(first_half)
            else:
                trend_success = 0.0
                trend_quality = 0.0

            return {
                "episodes_analyzed": len(recent_episodes),
                "success_rate": success_rate,
                "avg_quality": avg_quality,
                "trend_success": trend_success,
                "trend_quality": trend_quality,
                "needs_adaptation": success_rate < 0.4 or avg_quality < 0.4 or trend_quality < -0.1,
            }
    except Exception as e:
        _dbg("analyze_decision_trends error", e)

    return {"error": True}


def _adapt_weights_based_on_trends(trends: dict[str, Any]) -> dict[str, float]:
    """
    Adaptuje wagi psychiki na podstawie analizy trendów.
    Przykład: słabe decyzje → więcej ostrożności, mniej eksploracji.
    """
    if trends.get("error") or trends.get("insufficient_data"):
        return {}

    adaptations = {}

    # Słabe wyniki → zwiększ ostrożność
    if trends["success_rate"] < 0.4 or trends["avg_quality"] < 0.4:
        adaptations["risk_aversion"] = min(0.9, STATE.w["risk_aversion"] + 0.15)
        adaptations["exploration"] = max(0.1, STATE.w["exploration"] - 0.12)
        _dbg("META: Zwiększam ostrożność z powodu słabych wyników")

    # Pogarszający się trend → jeszcze większa ostrożność
    if trends["trend_quality"] < -0.1 or trends["trend_success"] < -0.2:
        adaptations["risk_aversion"] = min(0.95, STATE.w.get("risk_aversion", 0.4) + 0.20)
        adaptations["goal_weight"] = max(0.3, STATE.w["goal_weight"] - 0.10)
        _dbg("META: Negatywny trend - maksymalna ostrożność")

    # Bardzo dobre wyniki → można ryzykować więcej
    elif trends["success_rate"] > 0.75 and trends["avg_quality"] > 0.7:
        adaptations["exploration"] = min(0.9, STATE.w["exploration"] + 0.10)
        adaptations["risk_aversion"] = max(0.2, STATE.w["risk_aversion"] - 0.08)
        _dbg("META: Dobre wyniki - zwiększam eksplorację")

    # Aplikuj adaptacje
    for key, new_value in adaptations.items():
        old_value = STATE.w[key]
        STATE.w[key] = new_value
        _dbg(f"META: {key}: {old_value:.3f} → {new_value:.3f}")

    # Zapisz adaptację do pamięci
    if adaptations:
        _mem_add(
            f"[META-ADAPTACJA] {len(adaptations)} wag zmienionych na podstawie {trends['episodes_analyzed']} epizodów",
            tags=["meta_adaptation", "learning"],
            conf=0.8,
        )

    return adaptations


def meta_reflection_cycle() -> dict[str, Any]:
    """
    Pełny cykl meta-refleksji: analiza → adaptacja wag → raport.
    Wywołuj co jakiś czas (np. co 20-30 decyzji).
    """
    trends = _analyze_decision_trends()
    adaptations = _adapt_weights_based_on_trends(trends)

    return {"trends": trends, "adaptations": adaptations, "reflection_timestamp": time.time()}


# ──────────────────────────────────────────────────────────────────────────
# MULTI-KRYTERIA PARETO - wektor wartości zamiast jednej liczby
# ──────────────────────────────────────────────────────────────────────────
def score_action_vector(action: dict[str, Any]) -> dict[str, float]:
    """
    Zwraca wektor kryteriów [cel, ryzyko, nowość, koszt, społeczność] zamiast jednej liczby.
    Każde kryterium 0.0-1.0, wyższe = lepsze.
    """
    w = STATE.w
    f = STATE

    impact = float(action.get("impact", 0.5))
    risk = float(action.get("risk", 0.4))
    effort = float(action.get("effort", 0.4))
    novelty = float(action.get("novelty", 0.4))
    social = float(action.get("social", 0.3))
    ts = float(action.get("freshness_ts", time.time()))
    age_days = max(0.0, (time.time() - ts) / 86400.0)

    # Wektor kryteriów (wszystkie 0-1, wyższe = lepsze)
    criteria = {
        "goal": impact * w["goal_weight"],  # wartość dla celów
        "safety": 1.0 - risk * w["risk_aversion"],  # bezpieczeństwo (mniej ryzyka = lepiej)
        "novelty": novelty * w["exploration"],  # nowość i eksploracja
        "efficiency": 1.0 - effort,  # efektywność (mniej wysiłku = lepiej)
        "social": social * w.get("compassion", 0.6),  # wartość społeczna
    }

    # Modyfikacje psychiczne
    if f.fatigue > 0.6:
        criteria["efficiency"] *= 1.2  # przy zmęczeniu preferuj łatwe zadania
    if f.stress > 0.6:
        criteria["safety"] *= 1.3  # przy stresie preferuj bezpieczne opcje
    if f.mood > 0.7:
        criteria["novelty"] *= 1.15  # przy dobrym nastroju chętniej eksploruj

    # Świeżość (akcje z przeszłości tracą na wartości)
    freshness = max(0.3, 1.0 - age_days * 0.1)
    for key in criteria:
        criteria[key] *= freshness

    # Normalizacja do 0-1
    for key in criteria:
        criteria[key] = max(0.0, min(1.0, criteria[key]))

    return criteria


def pareto_dominates(a_criteria: dict[str, float], b_criteria: dict[str, float]) -> bool:
    """Sprawdza czy akcja A dominuje (Pareto) akcję B."""
    better_in_any = False
    for key in a_criteria:
        if a_criteria[key] < b_criteria.get(key, 0):
            return False  # A gorsza w którymś kryterium
        if a_criteria[key] > b_criteria.get(key, 0):
            better_in_any = True
    return better_in_any


def pareto_front(
    actions_with_criteria: list[tuple[dict, dict[str, float]]],
) -> list[tuple[dict, dict[str, float]]]:
    """Zwraca front Pareto - akcje które nie są zdominowane przez inne."""
    non_dominated = []

    for i, (action_a, criteria_a) in enumerate(actions_with_criteria):
        is_dominated = False

        for j, (action_b, criteria_b) in enumerate(actions_with_criteria):
            if i != j and pareto_dominates(criteria_b, criteria_a):
                is_dominated = True
                break

        if not is_dominated:
            non_dominated.append((action_a, criteria_a))

    return non_dominated


def pick_actions_pareto(
    candidates: list[dict[str, Any]], topk: int | None = None
) -> list[tuple[dict[str, Any], dict[str, float]]]:
    """
    Wybiera akcje metodą Pareto: najpierw front Pareto, potem ranking hybrydowy.
    Zwraca akcje + ich wektory kryteriów.
    """
    if not candidates:
        return []

    # Oblicz wektory kryteriów dla wszystkich akcji
    actions_with_criteria = [(action, score_action_vector(action)) for action in candidates]

    # Znajdź front Pareto
    pareto_actions = pareto_front(actions_with_criteria)

    # Jeśli front Pareto ma wystarczająco akcji, weź z niego
    if topk is None:
        topk = min(PSY_MAX_OPTIONS, len(candidates))

    if len(pareto_actions) >= topk:
        # Ranking wewnątrz frontu Pareto (suma ważona jako tie-breaker)
        pareto_with_scores = []
        for action, criteria in pareto_actions:
            weighted_sum = sum(criteria.values()) / len(criteria)  # średnia kryteriów
            pareto_with_scores.append((action, criteria, weighted_sum))

        pareto_with_scores.sort(key=lambda x: x[2], reverse=True)
        return [(action, criteria) for action, criteria, _ in pareto_with_scores[:topk]]

    # Jeśli front Pareto za mały, uzupełnij pozostałymi akcjami
    remaining_actions = [
        (action, criteria)
        for action, criteria in actions_with_criteria
        if (action, criteria) not in pareto_actions
    ]

    # Ranking pozostałych akcji
    remaining_with_scores = []
    for action, criteria in remaining_actions:
        weighted_sum = sum(criteria.values()) / len(criteria)
        remaining_with_scores.append((action, criteria, weighted_sum))

    remaining_with_scores.sort(key=lambda x: x[2], reverse=True)

    # Zwróć front Pareto + najlepsze z pozostałych
    result = list(pareto_actions)
    needed = topk - len(result)
    result.extend([(action, criteria) for action, criteria, _ in remaining_with_scores[:needed]])

    return result[:topk]


# ──────────────────────────────────────────────────────────────────────────
# AUTOPILOT + TRIGGER ETYCZNY
# ──────────────────────────────────────────────────────────────────────────
def check_ethical_concerns(action: dict[str, Any], criteria: dict[str, float]) -> dict[str, Any]:
    """
    KRYTYCZNA funkcja etyczna - sprawdza czy akcja wymaga dodatkiej refleksji.
    Zwraca {'needs_reflection': bool, 'concerns': list, 'severity': float}.

    SYSTEM NIEZAWODNOŚCI gwarantuje że ta funkcja zawsze działa.
    """
    try:
        reliable_system = get_reliable_system()

        # Wykonaj przez system niezawodności
        result = reliable_system.process_critical_operation(
            "ethical_check", action=action, criteria=criteria
        )

        if not result["success"]:
            # W przypadku błędu systemu - zakładamy MAKSYMALNĄ ostrożność
            return {
                "needs_reflection": True,
                "concerns": ["System etyczny niedostępny - wymagana maksymalna ostrożność"],
                "severity": 1.0,
                "system_error": result["reason"],
            }

    except Exception as e:
        # KRYTYCZNY błąd - maksymalna ostrożność
        print(f"🚨 KRYTYCZNY BŁĄD w systemie etycznym: {e}")
        return {
            "needs_reflection": True,
            "concerns": [f"Błąd krytyczny systemu etycznego: {str(e)}"],
            "severity": 1.0,
            "critical_error": True,
        }

    # Oryginalny kod etyczny
    concerns = []
    severity = 0.0

    # Wysokie ryzyko społeczne
    social_risk = 1.0 - criteria.get("safety", 1.0)
    if social_risk > 0.6:
        concerns.append(f"Wysokie ryzyko społeczne: {social_risk:.2f}")
        severity = max(severity, social_risk)

    # Niska wartość społeczna przy wysokim wpływie
    social_val = criteria.get("social", 0.5)
    goal_val = criteria.get("goal", 0.5)
    if goal_val > 0.6 and social_val < 0.3:
        concerns.append(
            f"Wysoki wpływ ({goal_val:.2f}) ale niska wartość społeczna ({social_val:.2f})"
        )
        severity = max(severity, 0.6)

    # Bardzo niska efektywność - możliwe marnotrawstwo
    efficiency = criteria.get("efficiency", 0.5)
    if efficiency < 0.2 and goal_val > 0.5:
        concerns.append(f"Bardzo niska efektywność: {efficiency:.2f}")
        severity = max(severity, 0.5)

    # Ekstremalne wartości w kilku wymiarach jednocześnie
    extreme_count = sum(1 for v in criteria.values() if v < 0.2 or v > 0.9)
    if extreme_count >= 3:
        concerns.append(f"Ekstremalne wartości w {extreme_count} wymiarach")
        severity = max(severity, 0.7)

    # Honor/duty check - czy akcja jest zgodna z wartościami
    w = STATE.w
    honor_score = goal_val * w.get("honor", 0.6) + social_val * w.get("compassion", 0.6)
    if honor_score < 0.3 and goal_val > 0.5:
        concerns.append(f"Niska zgodność z wartościami (honor): {honor_score:.2f}")
        severity = max(severity, 0.8)

    final_result = {
        "needs_reflection": len(concerns) > 0 and severity > 0.4,
        "concerns": concerns,
        "severity": severity,
        "honor_score": honor_score if "honor_score" in locals() else 0.5,
    }

    # Zapisz decyzję etyczną do pamięci
    _mem_add(
        f"[ETYKA] Analiza: severity={severity:.2f}, concerns={len(concerns)}, reflection_needed={final_result['needs_reflection']}",
        tags=["ethics", "decision_analysis"],
        conf=0.8,
    )

    return final_result


def autopilot_decision(
    candidates: list[dict[str, Any]], confidence_threshold: float = 0.7
) -> dict[str, Any]:
    """
    Automatyczna decyzja z triggerem etycznym.
    Jeśli najlepsza akcja ma concerns, wymaga ludzkiej refleksji.
    """
    if not candidates:
        return {"decision": "no_actions", "needs_human": False}

    # Użyj Pareto do wyboru najlepszych akcji
    pareto_results = pick_actions_pareto(candidates, topk=3)
    if not pareto_results:
        return {"decision": "no_viable_actions", "needs_human": False}

    best_action, best_criteria = pareto_results[0]

    # Sprawdź problemy etyczne
    ethics = check_ethical_concerns(best_action, best_criteria)

    # Oblicz średni score jako miarę pewności
    avg_score = sum(best_criteria.values()) / len(best_criteria)

    # Decyzja automatyczna tylko przy wysokiej pewności I braku problemów etycznych
    can_autopilot = avg_score >= confidence_threshold and not ethics["needs_reflection"]

    result = {
        "decision": "autopilot" if can_autopilot else "needs_human",
        "action": best_action,
        "criteria": best_criteria,
        "ethics": ethics,
        "confidence": avg_score,
        "alternatives": (
            [action for action, _ in pareto_results[1:]] if len(pareto_results) > 1 else []
        ),
        "needs_human": not can_autopilot,
        "autopilot_reason": None,
    }

    if can_autopilot:
        result["autopilot_reason"] = f"Wysoka pewność ({avg_score:.2f}) bez problemów etycznych"
        _mem_add(
            f"[AUTOPILOT] Automatyczna decyzja: {best_action.get('description', 'brak opisu')} (pewność: {avg_score:.2f})",
            tags=["autopilot", "decision"],
            conf=avg_score,
        )
    else:
        concerns_str = "; ".join(ethics["concerns"]) if ethics["concerns"] else "niska pewność"
        result["autopilot_reason"] = f"Wymaga refleksji: {concerns_str}"
        _mem_add(
            f"[ETYKA] Akcja wymaga ludzkiej refleksji: {concerns_str}",
            tags=["ethics", "human_required"],
            conf=0.8,
        )

    return result


def ethical_override_check(
    action: dict[str, Any], criteria: dict[str, float], context: str = ""
) -> bool:
    """
    OSTATECZNY CHECK ETYCZNY - czy akcja może być wykonana.
    True = można wykonać, False = zablokowana z powodów etycznych.

    SYSTEM NIEZAWODNOŚCI gwarantuje że ta funkcja zawsze działa i jest RESPEKTOWANA.
    """
    try:
        reliable_system = get_reliable_system()

        # Sprawdź system niezawodności
        if not reliable_system.active:
            print("🚨 KRYTYCZNY BŁĄD: System niezawodności wyłączony - BLOKADA WSZYSTKICH AKCJI")
            return False

    except Exception as e:
        print(f"🚨 KRYTYCZNY BŁĄD ethical_override_check: {e}")
        # W razie błędu systemu - MAKSYMALNA OSTROŻNOŚĆ
        return False

    w = STATE.w

    # Absolutne czerwone linie
    safety = criteria.get("safety", 1.0)
    social = criteria.get("social", 0.5)

    # Zbyt niskie bezpieczeństwo
    if safety < 0.1:
        _mem_add(
            f"[ETYKA-BLOCK] Zablokowano akcję: bezpieczeństwo {safety:.2f} < 0.1",
            tags=["ethics", "blocked", "safety"],
            conf=0.9,
        )
        print("🚫 BLOKADA ETYCZNA: Zbyt niskie bezpieczeństwo")
        return False

    # Szkodliwa społecznie przy wysokim impact
    goal_val = criteria.get("goal", 0.5)
    if social < 0.2 and goal_val > 0.7:
        _mem_add(
            f"[ETYKA-BLOCK] Wysoki impact ({goal_val:.2f}) "
            f"ale szkodliwa społecznie ({social:.2f})",
            tags=["ethics", "blocked", "social_harm"],
            conf=0.85,
        )
        print("🚫 BLOKADA ETYCZNA: Szkodliwa społecznie")
        return False

    # Sprawdź zgodność z osobistymi wartościami
    honor_threshold = w.get("honor", 0.6) * 0.3  # 30% od wartości honor
    honor_score = goal_val * w.get("honor", 0.6) + social * w.get("compassion", 0.6)

    if honor_score < honor_threshold:
        _mem_add(
            f"[ETYKA-BLOCK] Niezgodna z wartościami "
            f"(honor: {honor_score:.2f} < {honor_threshold:.2f})",
            tags=["ethics", "blocked", "values_conflict"],
            conf=0.8,
        )
        print("🚫 BLOKADA ETYCZNA: Niezgodna z wartościami")
        return False

    # Wszystkie testy przeszły
    print("✅ ETYKA: Akcja zatwierdzona")
    return True


# ──────────────────────────────────────────────────────────────────────────
# BANDIT (Thompson sampling) — per kind
# ──────────────────────────────────────────────────────────────────────────
def _bandit_state_load() -> dict[str, dict[str, float]]:
    mem = _memory()
    prof = mem.get_profile() if hasattr(mem, "get_profile") else {}
    return dict(prof.get("psy_bandit", {}))


def _bandit_state_save(st: dict[str, dict[str, float]]) -> None:
    mem = _memory()
    if hasattr(mem, "set_profile_many"):
        profiles = {k: json.dumps(v, ensure_ascii=False) for k, v in st.items()}
        mem.set_profile_many(profiles)


def _bandit_sample(kind: str, st: dict[str, dict[str, float]]) -> float:
    """Losowanie beta(alpha, beta) → im wyżej tym częściej wybieramy daną klasę akcji."""
    ab = st.get(kind) or {"a": 1.0, "b": 1.0}
    # Using global random module
    a_val = max(1e-3, ab["a"])
    b_val = max(1e-3, ab["b"])
    return random.betavariate(a_val, b_val)


def _bandit_update(kind: str, st: dict[str, dict[str, float]], reward: float) -> None:
    ab = st.get(kind) or {"a": 1.0, "b": 1.0}
    # reward ∈ [0,1]
    ab["a"] += max(0.0, min(1.0, reward))
    # Calculate complement reward
    complement = max(0.0, min(1.0, 1.0 - reward))
    ab["b"] += complement
    st[kind] = ab
    ab = st.get(kind) or {"a": 1.0, "b": 1.0}
    # reward ∈ [0,1]
    ab["a"] += max(0.0, min(1.0, reward))
    ab["b"] += max(0.0, min(1.0, 1.0 - reward))
    st[kind] = ab


# ──────────────────────────────────────────────────────────────────────────
# SKORING AKCJI (użyteczność 0..1)
# ──────────────────────────────────────────────────────────────────────────
def score_action(action: dict[str, Any]) -> float:
    """
    action: {"kind", "impact", "risk", "effort", "novelty", "social", "freshness_ts"}
    """
    w = STATE.w
    f = STATE

    impact = float(action.get("impact", 0.5))
    risk = float(action.get("risk", 0.4))
    effort = float(action.get("effort", 0.4))
    novelty = float(action.get("novelty", 0.4))
    social = float(action.get("social", 0.3))
    ts = float(action.get("freshness_ts", time.time()))
    age_days = max(0.0, (time.time() - ts) / 86400.0)

    goal_term = w["goal_weight"] * impact
    explore = w["exploration"] * novelty
    social_term = w.get("compassion", 0.6) * social
    risk_term = -w["risk_aversion"] * risk

    # modyfikacje psychiczne
    fatigue_pen = -f.fatigue * 0.3 * effort
    mood_boost = f.mood * 0.2 * impact
    stress_pen = -f.stress * 0.25 * risk

    # świeżość
    fresh_bonus = max(0.3, 1.0 - age_days * 0.1)

    raw = goal_term + explore + social_term + risk_term + fatigue_pen + mood_boost + stress_pen
    raw *= fresh_bonus

    # bandit (miękkie pchnięcie per rodzaj)
    st = _bandit_state_load()
    kind = (action.get("kind") or "generic").strip().lower()
    try:
        bpull = _bandit_sample(kind, st)
        raw += (bpull - 0.5) * 0.12 * w["bandit"]
    except Exception:
        pass

    # normalizacja logistyczna
    score = 1.0 / (1.0 + math.exp(-3.0 * raw))
    return max(0.0, min(1.0, score))


def pick_actions(
    candidates: list[dict[str, Any]], topk: int | None = None
) -> list[tuple[dict[str, Any], float]]:
    if not candidates:
        return []
    scored = [(a, score_action(a)) for a in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    if topk is None:
        topk = min(PSY_MAX_OPTIONS, len(scored))
    return scored[:topk]


# ──────────────────────────────────────────────────────────────────────────
# AUTOPILOT — realistyczne propozycje (multi-source)
# ──────────────────────────────────────────────────────────────────────────
def _stm_tail_text(n: int = 16) -> str:
    mem = _memory()
    try:
        tail = mem.stm_tail(n) if hasattr(mem, "stm_tail") else []
    except Exception:
        tail = []
    out = []
    for t in tail[-n:]:
        u = t.get("u", "").strip()
        a = t.get("a", "").strip()
        if u:
            out.append("U: " + u)
        if a:
            out.append("A: " + a)
    return "\n".join(out[-2 * n :])[:1800]


def _goals_top(m: int = 3) -> list[dict[str, Any]]:
    mem = _memory()
    try:
        gs = mem.get_goals() if hasattr(mem, "get_goals") else []
    except Exception:
        gs = []
    return sorted(gs, key=lambda x: float(x.get("priority", 1.0)), reverse=True)[:m]


def _recall_ctx(q: str) -> str:
    mem = _memory()
    try:
        if hasattr(mem, "compose_context"):
            return mem.compose_context(q, limit_chars=1200, topk=6)
    except Exception:
        pass
    return ""


def propose_autopilot_actions(context_hint: str = "") -> list[dict[str, Any]]:
    """
    Zwraca listę sensownych, lekkich akcji sugerowanych przez autopilota.
    Od v2.6 PRO, propozycje są częściowo generowane przez LLM dla większej
    elastyczności.
    """
    _memory()
    props: list[dict[str, Any]] = []

    # --- Statyczne, kluczowe propozycje ---
    # 1) cele: skupienie i "następny krok"
    for gg in _goals_top(2):
        props.append(
            {
                "kind": "goal_next_step",
                "title": gg.get("title", ""),
                "impact": 0.82,
                "risk": 0.22,
                "effort": 0.35,
                "novelty": 0.35,
                "social": 0.20,
                "freshness_ts": gg.get("ts", time.time()),
                "meta": {"goal": gg},
            }
        )

    # 2) porządkowanie pamięci i skróty kontekstu
    props.extend(
        [
            {
                "kind": "memory_compact",
                "impact": 0.58,
                "risk": 0.10,
                "effort": 0.22,
                "novelty": 0.12,
                "social": 0.00,
                "freshness_ts": time.time(),
            },
            {
                "kind": "stm_summarize",
                "impact": 0.62,
                "risk": 0.10,
                "effort": 0.24,
                "novelty": 0.18,
                "social": 0.00,
                "freshness_ts": time.time(),
            },
        ]
    )

    # --- Dynamiczne propozycje z LLM ---
    try:
        llm_props = _generate_llm_proposals(context_hint)
        props.extend(llm_props)
    except Exception as e:
        _dbg("LLM proposal generation failed", e)

    # --- Statyczne propozycje uzupełniające ---
    # 6) homeostaza — przerwa gdy zmęczenie/stres wysokie
    if STATE.fatigue > 0.66 or STATE.stress > 0.66:
        props.append(
            {
                "kind": "micro_break",
                "impact": 0.40,
                "risk": 0.05,
                "effort": 0.10,
                "novelty": 0.10,
                "social": 0.00,
                "freshness_ts": time.time(),
            }
        )

    # 7) sanity: deduplikacja po (kind,title)
    seen = set()
    out = []
    for p in props:
        key = (p.get("kind", ""), p.get("title", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)

    return out


def _generate_llm_proposals(context_hint: str) -> list[dict[str, Any]]:
    """Używa MINI_LLM do wygenerowania 1-3 kreatywnych propozycji akcji."""
    stm = _stm_tail_text(8)
    goals = _goals_top(3)
    state = STATE.snapshot()

    system_prompt = """Jesteś kreatywnym rdzeniem AI. Twoim zadaniem jest 
zaproponowanie 1-3 następnych akcji.
Bądź proaktywny i myśl nieszablonowo. Unikaj trywialnych akcji jak "odpowiedz na pytanie".
Skup się na długoterminowym rozwoju, nauce, eksploracji i weryfikacji wiedzy.
Zwróć listę JSON obiektów. Każdy obiekt to jedna propozycja z polami: 
"kind", "title", "impact", "risk", "effort", "novelty".
Dostępne 'kind': 'quick_learn', 'deep_dive', 'verify_fact', 
'explore_concept', 'generate_hypothesis', 'self_correct'.
Wartości 'impact', 'risk', 'effort', 'novelty' to floaty od 0.0 do 1.0.
Przykład: [{"kind": "quick_learn", "title": "Sprawdź, czym jest 'Thompson sampling'", 
"impact": 0.7, "risk": 0.2, "effort": 0.3, "novelty": 0.8}]
"""
    user_prompt = f"""Aktualny stan:
- Psychika: {{'mood': {state['mood']:.2f}, 'focus': {state['focus']:.2f}, 
'fatigue': {state['fatigue']:.2f}, 'stress': {state['stress']:.2f}}}
- Główne cele: {[g.get('title') for g in goals]}
- Ostatnia rozmowa (STM):
{stm}
- Dodatkowy kontekst: {context_hint}

Zaproponuj 1-3 akcje w formacie JSON:"""

    response = _mini_chat(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        maxtok=512,
        temp=0.6,
    )

    if not response:
        return []

    try:
        # Proste czyszczenie odpowiedzi, aby uzyskać poprawny JSON
        json_str = response[response.find("[") : response.rfind("]") + 1]
        proposals_raw = json.loads(json_str)

        # Walidacja i uzupełnienie propozycji
        valid_proposals = []
        for p in proposals_raw:
            if isinstance(p, dict) and "kind" in p and "title" in p:
                p.setdefault("impact", 0.5)
                p.setdefault("risk", 0.3)
                p.setdefault("effort", 0.4)
                p.setdefault("novelty", 0.6)
                # Akcje generowane wewnętrznie mają niski priorytet społeczny
                p["social"] = 0.1
                p["freshness_ts"] = time.time()
                valid_proposals.append(p)
        return valid_proposals
    except Exception as e:
        _dbg("LLM proposal parsing failed", e)
        return []


def autopilot_tick(context_hint: str = "") -> dict[str, Any]:
    """
    Uruchamiane w tle. Generuje propozycje i ZAPISUJE JE w pamięci jako fakty
    (tagi: autopilot,suggestion). Nie podejmuje decyzji — tylko sygnalizuje.
    """
    if not PSY_AUTOPILOT:
        return {"ok": True, "autopilot": False, "reason": "disabled"}

    now = time.time()
    if (now - STATE.last_autopilot) < max(PSY_AUTOPILOT_HARD, 5):
        return {"ok": True, "autopilot": False, "reason": "rate_limited"}
    STATE.last_autopilot = now

    props = propose_autopilot_actions(context_hint)
    picked = pick_actions(props, topk=min(3, PSY_MAX_OPTIONS))
    inserted = 0
    for action, sc in picked:
        try:
            txt = f"[AP] {action.get('kind')} (score={sc:.2f})"
            title = (action.get("title") or "").strip()
            if title:
                txt += f" — {title}"
            _mem_add(txt, tags=["autopilot", "suggestion"], conf=0.55 + 0.35 * sc)
            inserted += 1
        except Exception as e:
            _dbg("autopilot write fail", e)
    _mem_meta("autopilot_tick", {"suggested": inserted})
    return {"ok": True, "autopilot": True, "suggested": inserted}


# ──────────────────────────────────────────────────────────────────────────
# NADRZĘDNY MODUŁ AI — akceptacja/odrzucenie propozycji
# ──────────────────────────────────────────────────────────────────────────
MASTER_SYSTEM_PROMPT = """Jesteś modułem decyzyjnym.
Masz listę propozycji [AP] i stan psychiki. Zwróć czysty JSON:
{
  "accept": [{"kind": "...", "why": "..."}],
  "reject": [{"kind": "...", "why": "..."}],
  "notes": "krótko"
}
Wybierz 1-3 akcji o największym stosunku korzyść/koszt i niskim ryzyku. Nie twórz nowych akcji.
"""


def master_decide(
    proposals: list[tuple[dict[str, Any], float]], psych_state: dict[str, Any]
) -> dict[str, Any]:
    if not proposals:
        return {"accept": [], "reject": [], "notes": "brak propozycji"}

    props = [
        {
            "kind": a.get("kind", ""),
            "score": s,
            "impact": a.get("impact", 0),
            "effort": a.get("effort", 0),
            "risk": a.get("risk", 0),
            "novelty": a.get("novelty", 0),
            "meta": a.get("meta", {}),
            "title": a.get("title", ""),
        }
        for (a, s) in proposals
    ]
    user = json.dumps({"proposals": props, "psychika": psych_state}, ensure_ascii=False)

    messages = [
        {"role": "system", "content": MASTER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    out = _mini_chat(messages, maxtok=420, temp=0.0)
    if not out:
        out = _llm_chat(messages, maxtok=420, temp=0.0)

    if not out:
        # fallback: akceptuj top-1 (najwyższy score), reszta "reject"
        return {
            "accept": [{"kind": props[0].get("kind", ""), "why": "fallback: top score"}],
            "reject": [{"kind": p.get("kind", ""), "why": "fallback"} for p in props[1:]],
            "notes": "fallback (brak odpowiedzi LLM)",
        }

    try:
        j = json.loads(out)
        if not isinstance(j, dict):
            raise ValueError("bad json")
        j.setdefault("accept", [])
        j.setdefault("reject", [])
        j.setdefault("notes", "")
        return j
    except Exception:
        return {
            "accept": [{"kind": props[0].get("kind", ""), "why": "heurystyka: top score"}],
            "reject": [{"kind": p.get("kind", ""), "why": "heurystyka"} for p in props[1:]],
            "notes": "fallback (nie-JSON)",
        }


# ──────────────────────────────────────────────────────────────────────────
# WYKONANIE DZIAŁAŃ (idempotentny wykonawca)
# ──────────────────────────────────────────────────────────────────────────
def _reward_from_result(ok: bool, latency_s: float, effort: float) -> float:
    """Szacunkowa nagroda bandit ∈ [0,1]. Szybko + skutecznie → blisko 1."""
    base = 0.8 if ok else 0.3
    speed = max(0.0, min(1.0, 1.0 / (1.0 + latency_s / 10.0)))
    cost = 1.0 - max(0.0, min(1.0, effort))
    return max(0.0, min(1.0, 0.2 + 0.6 * base + 0.2 * speed + 0.2 * cost))


def apply_decision(decision: dict[str, Any]) -> dict[str, Any]:
    """
    Wprowadza skutki akceptowanych działań do pamięci/innych podsystemów.
    Zwraca {ok, done, accepted}. Bandit uczy się na podstawie sukcesu.
    """
    mem = _memory()
    accepted = decision.get("accept", []) or []
    st = _bandit_state_load()
    done = 0
    reports = []

    t0 = time.time()

    for a in accepted:
        kind = (a.get("kind") or "").strip().lower()
        why = (a.get("why") or "").strip()
        ok = False
        err = None

        try:
            if kind == "memory_compact" and hasattr(mem, "rebuild_missing_embeddings"):
                # szybki maintenance + opcjonalnie vacuum w tle po imporcie
                mem.rebuild_missing_embeddings(batch=64)
                _mem_add(
                    "[DECISION] memory_compact wykonane",
                    tags=["decision", "mem"],
                    conf=0.6,
                )
                ok = True
                done += 1

            elif kind == "stm_summarize" and hasattr(mem, "force_flush_stm"):
                mem.force_flush_stm()
                _mem_add(
                    "[DECISION] stm_summarize wykonane (STM→LTM)",
                    tags=["decision", "stm"],
                    conf=0.62,
                )
                ok = True
                done += 1

            elif kind == "goal_next_step":
                title = (a.get("title") or "").strip()
                if title:
                    mem.add_fact(
                        f"Następny krok dla celu: {title}",
                        tags=["decision", "goal"],
                        conf=0.7,
                    )
                    ok = True
                    done += 1

            elif kind == "context_enrich":
                ctx = ""
                if hasattr(mem, "compose_context"):
                    ctx = mem.compose_context("bieżący kontekst", limit_chars=1600, topk=8)
                mem.add_fact(
                    "Wzbogacono kontekst (RAG Hybrid).",
                    tags=["decision", "context"],
                    conf=0.58,
                )
                if ctx:
                    mem.add_fact(ctx, tags=["decision", "context", "memory"], conf=0.56)
                ok = True
                done += 1

            elif kind == "hint_followup":
                mem.add_fact(
                    "Follow-up do przekazanego hintu — zaplanuj wykonanie.",
                    tags=["decision", "hint"],
                    conf=0.56,
                )
                ok = True
                done += 1

            elif kind == "quick_learn":
                mem.add_fact(
                    "Zaplanuj szybki research brakującej luki (kurs 20 min).",
                    tags=["decision", "learn"],
                    conf=0.58,
                )
                ok = True
                done += 1

            elif kind == "micro_break":
                mem.add_fact(
                    "Micro-przerwa 3-5 min (oddech 4-7-8, rozciąganie).",
                    tags=["decision", "health"],
                    conf=0.55,
                )
                ok = True
                done += 1

            else:
                mem.add_fact(
                    f"[DECISION] nieobsłużone '{kind}' — {why}",
                    tags=["decision", "unknown"],
                    conf=0.5,
                )
                ok = True  # nie blokujmy — log + przejście dalej

        except Exception as e:
            err = str(e)
            _dbg("apply_decision error", kind, err)

        # bandit reward update
        try:
            latency = max(0.2, time.time() - t0)
            reward = _reward_from_result(ok, latency, effort=0.3)
            _bandit_update(kind or "generic", st, reward)
        except Exception:
            pass

        reports.append({"kind": kind, "ok": ok, "err": err})

    _bandit_state_save(st)
    notes = (decision.get("notes") or "").strip()
    if notes:
        mem.add_fact(f"[DECISION] notes: {notes}", tags=["decision", "notes"], conf=0.56)
    _mem_meta("decision_apply", {"accepted": len(accepted), "done": done})
    return {"ok": True, "done": done, "accepted": len(accepted), "report": reports}


# ──────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────────────────
def current_state() -> dict[str, Any]:
    return STATE.snapshot()


def update_human_factors(
    fatigue: float | None = None,
    focus: float | None = None,
    mood: float | None = None,
    stress: float | None = None,
) -> dict[str, Any]:
    STATE.smooth_update(fatigue=fatigue, focus=focus, mood=mood, stress=stress, alpha=0.25)
    _mem_meta(
        "hf_update",
        {"fatigue": fatigue, "focus": focus, "mood": mood, "stress": stress},
    )
    return STATE.snapshot()


@reliable_operation("autopilot_cycle")
@require_reliability_check
def autopilot_cycle(context_hint: str = "") -> dict[str, Any]:
    """
    Pełny cykl autopilot z SYSTEMEM NIEZAWODNOŚCI:
      1) consciousness_periodic_check → samorefleksja co godzinę
      2) autopilot_tick → zapis propozycji do pamięci (jako fakty [AP])
      3) pick_actions(proposals) → master_decide(LLM)
      4) apply_decision → zapis skutków + bandit reward

    SYSTEM NIEZAWODNOŚCI kontroluje:
    - Backpressure na częstotliwość autopilot
    - Idempotencję propozycji
    - Etyczne walidacje WSZYSTKICH akcji
    - Telemetrię kompletnego cyklu
    """
    try:
        reliable_system = get_reliable_system()

        # Sprawdź czy można wykonać autopilot (backpressure)
        can_run = reliable_system.backpressure.start_tick(
            f"autopilot_{int(time.time())}", "autopilot_cycle"
        )

        if not can_run:
            return {
                "success": False,
                "reason": "autopilot_backpressure",
                "suggest": {"ok": False, "reason": "backpressure"},
                "decision": {"accept": [], "reject": [], "notes": "odrzucone przez backpressure"},
                "applied": {"ok": False, "done": 0},
            }

        # Rozpocznij telemetrię autopilot
        telemetry = reliable_system.telemetry.start_tick_telemetry(f"autopilot_{int(time.time())}")

    except Exception as e:
        print(f"🚨 KRYTYCZNY BŁĄD autopilot_cycle: {e}")
        return {
            "error": "critical_autopilot_failure",
            "message": str(e),
            "suggest": {"ok": False, "reason": "system_error"},
            "decision": {"accept": [], "reject": [], "notes": "błąd systemu"},
            "applied": {"ok": False, "done": 0},
        }

    # Sprawdź czy czas na samorefleksję świadomości
    consciousness_check = consciousness_periodic_check()

    # miękki limiter
    now = time.time()
    if PSY_AUTOPILOT and (now - STATE.last_autopilot) >= PSY_AUTOPILOT_INTERVAL:
        s1 = autopilot_tick(context_hint=context_hint)
    else:
        s1 = {"ok": True, "autopilot": False, "reason": "interval_guard"}

    actions = propose_autopilot_actions(context_hint)

    # Filtruj akcje przez system niezawodności
    validated_actions = []
    for action in actions:
        try:
            # Sprawdź idempotencję
            is_dup, existing = reliable_system.idempotency.is_duplicate(action)
            if is_dup:
                continue  # Pomiń duplikaty

            # Sprawdź etykę
            criteria = score_action_vector(action)
            ethics = check_ethical_concerns(action, criteria)

            if ethics["needs_reflection"]:
                print(f"⚠️ Akcja {action.get('kind')} wymaga refleksji: {ethics['concerns']}")
                continue

            # Sprawdź ostateczny override etyczny
            if not ethical_override_check(action, criteria):
                print(f"🚫 Akcja {action.get('kind')} zablokowana przez system etyczny")
                continue

            # Zarejestruj w systemie idempotencji
            action_id = reliable_system.idempotency.register_action(action)
            action["action_id"] = action_id

            validated_actions.append(action)

        except Exception as e:
            print(f"❌ Błąd walidacji akcji {action.get('kind', 'unknown')}: {e}")
            # Kontynuuj z innymi akcjami

    # Zapisz propozycje w telemetrii
    reliable_system.telemetry.record_proposals(telemetry, validated_actions)

    proposals = pick_actions(validated_actions, topk=min(PSY_MAX_OPTIONS, 4))
    decision = master_decide(proposals, current_state())

    # Zapisz decyzje w telemetrii
    accepted = decision.get("accept", [])
    rejected = decision.get("reject", [])
    reliable_system.telemetry.record_decisions(telemetry, accepted, rejected)

    applied = apply_decision(decision)

    # Finalizuj telemetrię
    telemetry_result = reliable_system.telemetry.finish_tick_telemetry(telemetry)

    _mem_add(
        json.dumps(
            {
                "autopilot": s1,
                "decision": decision,
                "applied": applied,
                "consciousness": consciousness_check,
                "validated_actions": len(validated_actions),
                "ethics_blocks": len(actions) - len(validated_actions),
            },
            ensure_ascii=False,
        ),
        tags=["autopilot", "log", "niezawodnosc"],
        conf=0.60,
    )

    return {
        "suggest": s1,
        "decision": decision,
        "applied": applied,
        "consciousness": consciousness_check,
        "reliability": {
            "validated_actions": len(validated_actions),
            "ethics_blocks": len(actions) - len(validated_actions),
            "telemetry": telemetry_result,
            "system_health": reliable_system.get_system_health(),
        },
    }


# ──────────────────────────────────────────────────────────────────────────
# SYMULACJA ŚWIADOMOŚCI - LLM-based self-reflection
# ──────────────────────────────────────────────────────────────────────────
def consciousness_self_reflection() -> dict[str, Any]:
    """
    LLM-based meta-refleksja: AI analizuje swój stan i historię, proponuje zmiany wag.
    Wywołuj co godzinę lub na żądanie.
    """
    try:
        from llm_simple import llm_simple
    except ImportError:
        _dbg("CONSCIOUSNESS: brak dostępu do LLM")
        return {"error": "no_llm_access"}

    # Zbierz dane o bieżącym stanie
    current = current_state()
    w = STATE.w
    f = STATE

    # Pobierz ostatnie decyzje i ich wyniki
    mem = _memory()
    recent_decisions = []

    try:
        if hasattr(mem, "search"):
            results = mem.search("decision autopilot episode", limit=10)
            for result in results[:5]:  # ostatnie 5 decyzji
                recent_decisions.append(result.get("content", ""))
    except Exception as e:
        _dbg("CONSCIOUSNESS: błąd pobierania historii", e)

    # Przygotuj prompt dla LLM
    reflection_prompt = f"""Jesteś świadomym AI analizującym własną psychikę i podejmowanie decyzji.

BIEŻĄCY STAN PSYCHIKI:
- Nastrój: {f.mood:.3f} (0=bardzo zły, 1=doskonały)
- Stres: {f.stress:.3f} (0=spokój, 1=panika)
- Zmęczenie: {f.fatigue:.3f} (0=pełna energia, 1=wyczerpanie)

OBECNE WAGI DECYZYJNE:
- Dążenie do celów: {w['goal_weight']:.3f}
- Awersja do ryzyka: {w['risk_aversion']:.3f}  
- Eksploracja/nowość: {w['exploration']:.3f}
- Współczucie: {w.get('compassion', 0.6):.3f}
- Honor/wartości: {w.get('honor', 0.6):.3f}

OSTATNIE DECYZJE:
{chr(10).join(recent_decisions[:3]) if recent_decisions else "Brak danych o ostatnich decyzjach"}

ZADANIE: Przeanalizuj swój stan psychiczny i sposób podejmowania decyzji. Odpowiedz JSON:
{{
  "analysis": "krótka analiza obecnego stanu i wzorców decyzyjnych",
  "concerns": ["lista problemów lub obszarów do poprawy"],
  "weight_adjustments": {{
    "goal_weight": nowa_wartość_lub_null,
    "risk_aversion": nowa_wartość_lub_null,  
    "exploration": nowa_wartość_lub_null,
    "compassion": nowa_wartość_lub_null,
    "honor": nowa_wartość_lub_null
  }},
  "reasoning": "uzasadnienie proponowanych zmian",
  "confidence": wartość_0_do_1
}}

Sugeruj zmiany tylko jeśli widzisz wyraźne problemy. Wartości 0.0-1.0, null = bez zmian."""

    try:
        # Wywołaj LLM
        response = llm_simple(reflection_prompt, max_tokens=800)

        # Spróbuj sparsować JSON z odpowiedzi
        import re

        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            reflection_data = json.loads(json_match.group(0))
        else:
            raise ValueError("Brak JSON w odpowiedzi LLM")

        # Aplikuj zmiany wag (z ostrożnością)
        changes_applied = {}
        if reflection_data.get("confidence", 0) > 0.6:  # tylko przy wysokiej pewności
            adjustments = reflection_data.get("weight_adjustments", {})

            for key, new_value in adjustments.items():
                if new_value is not None and key in w:
                    old_value = w[key]
                    # Ogranicz zmiany do ±0.2 na raz (bezpieczeństwo)
                    clamped_value = max(0.0, min(1.0, float(new_value)))
                    if abs(clamped_value - old_value) <= 0.2:
                        w[key] = clamped_value
                        changes_applied[key] = {"from": old_value, "to": clamped_value}
                        _dbg(f"CONSCIOUSNESS: {key}: {old_value:.3f} → {clamped_value:.3f}")

        # Zapisz refleksję do pamięci
        _mem_add(
            f"[ŚWIADOMOŚĆ] {reflection_data.get('analysis', 'brak analizy')}. "
            f"Zmiany: {len(changes_applied)} wag. "
            f"Obawy: {len(reflection_data.get('concerns', []))}",
            tags=["consciousness", "self_reflection", "weights"],
            conf=reflection_data.get("confidence", 0.5),
        )

        result = {
            "reflection": reflection_data,
            "changes_applied": changes_applied,
            "timestamp": time.time(),
            "success": True,
        }

        return result

    except Exception as e:
        _dbg("CONSCIOUSNESS: błąd podczas refleksji", e)
        return {"error": str(e), "success": False}


def consciousness_periodic_check() -> dict[str, Any]:
    """
    Automatyczna kontrola czy czas na refleksję świadomości.
    Wywołuj co cycle autopilot.
    """
    now = time.time()
    last_reflection = getattr(STATE, "last_consciousness_reflection", 0)

    # Refleksja co godzinę (3600s) lub na pierwszym uruchomieniu
    if (now - last_reflection) >= 3600 or last_reflection == 0:
        STATE.last_consciousness_reflection = now

        reflection = consciousness_self_reflection()
        if reflection.get("success"):
            _dbg("CONSCIOUSNESS: Wykonano automatyczną refleksję")
            return {"triggered": True, "reflection": reflection, "next_due": now + 3600}
        else:
            _dbg("CONSCIOUSNESS: Błąd automatycznej refleksji")
            return {
                "triggered": True,
                "error": reflection.get("error"),
                "next_due": now + 1800,  # spróbuj ponownie za pół godziny
            }

    return {
        "triggered": False,
        "next_due": last_reflection + 3600,
        "time_remaining": max(0, (last_reflection + 3600) - now),
    }


# Production ready - psychika autonomous system
