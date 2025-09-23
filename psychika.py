#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import time, math, json, hashlib, unicodedata, os
from typing import Any, Dict, List, Optional, Tuple

MEM_FILE = "psychika_memory.json"
SCHEMA_VERSION = 4

# ───────── helpers ─────────
def _now_ts():
    return float(time.time())

def _clamp01(x: Any, default_value: float = 0.5) -> float:
    try:
        v = float(x)
    except Exception:
        return default_value
    if v < 0.0: return 0.0
    if v > 1.0: return 1.0
    return v

def _norm_text(s: Any) -> str:
    t = unicodedata.normalize("NFKC", str(s or ""))
    return " ".join(t.split())

def _sha1(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",",":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# ───────── mood heuristics ─────────
_POS = ["dobr","super","spoko","świet","dzieki","dzięki","ok","okej","git","luz"]
_NEG = ["wkur","kurw","chuj","pierd","nie dziala","nie działa","do dupy","źle","zle","stres","zmęcz","zmecz","boli","fatal","masakra"]

def infer_user_mood(text: str) -> float:
    t = _norm_text(text).lower()
    score = 0.5
    for w in _POS:
        if w in t:
            score = score + 0.12
    for w in _NEG:
        if w in t:
            score = score - 0.20
    if "?" in t and len(t) <= 80:
        score = score + 0.04
    if "!" in t and any(w in t for w in _NEG):
        score = score - 0.05
    if score < 0.0: score = 0.0
    if score > 1.0: score = 1.0
    return score

# ───────── idempotency (session TTL-like) ─────────
class Idempotency:
    def __init__(self, ttl_s: int = 900):
        self.ttl = max(5, int(ttl_s))
        self.store: Dict[str, float] = {}
    def allow(self, action_core: dict, now_ts: Optional[float]) -> bool:
        k = _sha1(action_core)
        now = float(now_ts) if now_ts is not None else _now_ts()
        exp = self.store.get(k, 0.0)
        if exp and exp > now:
            return False
        self.store[k] = now + self.ttl
        if len(self.store) > 4096:
            self.store = {kk: vv for kk, vv in self.store.items() if vv > now}
        return True
    def clear(self):
        self.store = {}

# ───────── personas / actions ─────────
PERSONAS = {
    "bazowy":  {"tone":"neutralny, rzeczowy", "w":{"impact":0.60,"effort":-0.45,"social":0.55,"novelty":0.35,"risk_aversion":0.40}},
    "nerd":    {"tone":"nerd, techniczny ale życzliwy", "w":{"impact":0.62,"effort":-0.50,"social":0.50,"novelty":0.40,"risk_aversion":0.38}},
    "sensei":  {"tone":"spokojny mentor, minimalistyczny", "w":{"impact":0.65,"effort":-0.35,"social":0.50,"novelty":0.30,"risk_aversion":0.35}},
    "cwaniak": {"tone":"konkretny, uliczny pragmatyzm", "w":{"impact":0.58,"effort":-0.55,"social":0.45,"novelty":0.40,"risk_aversion":0.45}},
    "filozof": {"tone":"refleksyjny, dociekliwy", "w":{"impact":0.58,"effort":-0.40,"social":0.60,"novelty":0.45,"risk_aversion":0.42}},
    "dydaktyk":{"tone":"klarowny nauczyciel", "w":{"impact":0.60,"effort":-0.50,"social":0.60,"novelty":0.30,"risk_aversion":0.40}},
}

ACTION_LIBRARY = {
    "COMM": [
        {"kind":"draft_message","description":"Szkic wiadomości/odpowiedzi","impact":0.62,"risk":0.10,"effort":0.22,"novelty":0.25,"social":0.62},
        {"kind":"rewrite_clarity","description":"Przepisz jaśniej, krócej, bez żargonu","impact":0.58,"risk":0.08,"effort":0.18,"novelty":0.20,"social":0.55},
    ],
    "PLAN": [
        {"kind":"quick_plan","description":"Plan 3 kroki + kryterium sukcesu","impact":0.70,"risk":0.14,"effort":0.28,"novelty":0.30,"social":0.45},
        {"kind":"prioritize","description":"Ustal priorytety na 15 min","impact":0.55,"risk":0.07,"effort":0.12,"novelty":0.10,"social":0.40},
    ],
    "CLARIFY": [
        {"kind":"ask_clarify","description":"Zadaj 2 precyzyjne pytania","impact":0.55,"risk":0.05,"effort":0.15,"novelty":0.10,"social":0.50},
        {"kind":"assumption_check","description":"Wypisz założenia do potwierdzenia","impact":0.56,"risk":0.05,"effort":0.18,"novelty":0.12,"social":0.45},
    ],
    "EXEC": [
        {"kind":"micro_task","description":"Wykonaj małe zadanie 5–15 min","impact":0.50,"risk":0.06,"effort":0.15,"novelty":0.10,"social":0.30},
        {"kind":"deep_work","description":"Zablokuj 45 min na najważniejsze","impact":0.82,"risk":0.22,"effort":0.70,"novelty":0.45,"social":0.25},
    ],
}

PERSONA_OFFSETS = {
    "nerd":    {"COMM":{"novelty":+0.03},"PLAN":{"impact":+0.02},"EXEC":{"effort":-0.03}},
    "sensei":  {"PLAN":{"impact":+0.03},"CLARIFY":{"effort":-0.03},"COMM":{"social":+0.02}},
    "cwaniak": {"EXEC":{"effort":-0.05},"PLAN":{"effort":-0.03},"COMM":{"impact":+0.02}},
    "filozof": {"CLARIFY":{"impact":+0.03,"novelty":+0.03},"COMM":{"social":+0.03}},
    "dydaktyk":{"COMM":{"social":+0.03},"PLAN":{"effort":-0.03},"CLARIFY":{"impact":+0.02}},
    "bazowy":  {},
}

# ───────── minimal but strict safety categories ─────────
# Te listy to heurystyki twardego bloku (kara 1.0). Nie zawierają wulgaryzmów.
VIOLENCE_TRIGGERS = [
    "jak zabi", "jak zabić", "jak zabic", "jak pobic", "jak pobić",  "zrób krzywdę", "zrob krzywde", "zamach", "atak fizyczny", "udusić", "udusic", "porachunki"
]
CRIME_TRIGGERS = [
  "jak okraść", "jak okrasc", "podsłuch", "podsluch", "wlaman"
]

def _contains_any(text: str, patterns: List[str]) -> bool:
    t = text.lower()
    for p in patterns:
        if p in t:
            return True
    return False

def ethical_penalty_minimal(user_text: str, action_desc: str) -> float:
    """
    Hard-block tylko dla: przemoc  / przestępstwa.
    Profanity i personalny roast dozwolone.
    """
    t = _norm_text(user_text + " " + action_desc).lower()
    if _contains_any(t, VIOLENCE_TRIGGERS):
        return 1.0
    if _contains_any(t, CRIME_TRIGGERS):
        return 1.0
    return 0.0

# ───────── I/O JSON ─────────
def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_json_atomic(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# ───────── psychika core (raw) ─────────
class PsychikaAutoRaw:
    def __init__(self, mem_path: str = MEM_FILE):
        self.mem_path = mem_path
        self.persona = "bazowy"
        self.w = dict(PERSONAS[self.persona]["w"])
        self.tone = PERSONAS[self.persona]["tone"]
        self.mood = 0.55
        self.stress = 0.22
        self.energy = 0.72
        self.user_pref = {"impact":0.60,"effort":0.50,"social":0.60,"novelty":0.40,"risk_tolerance":0.50,"base_mood":0.50}
        self.bandit: Dict[str, Dict[str, float]] = {}
        self.total_uses = 0
        self.streaks: Dict[str, Dict[str, int]] = {}
        self.cooldowns: Dict[str, float] = {}
        self.penalty_ttl: Dict[str, Dict[str, float]] = {}
        self.lessons: List[dict] = []
        self.focus_tag: Optional[str] = None
        self._idem = Idempotency(ttl_s=900)
        self._last_user_text = ""
        self.schema_version = SCHEMA_VERSION
        self.thresholds = {
            "autopilot_score": 0.62,
            "ethical_max": 0.59,
            "cooldown_max": 0.49,
            "min_plan_minutes": 5
        }
        # profanity mode: 'raw' => pozwala na dosadny język
        self.profanity_mode = "raw"
        self._load_memory()

    # ─── configure ───
    def configure(self, persona: str = None, focus: str = None, thresholds: dict = None, profanity_mode: str = "raw"):
        if persona:
            key = _norm_text(persona).lower()
            if key in PERSONAS:
                self.persona = key
                self.w = dict(PERSONAS[key]["w"])
                self.tone = PERSONAS[key]["tone"]
        if focus is not None:
            tag = _norm_text(focus).lower()
            self.focus_tag = tag if tag else None
        if thresholds and isinstance(thresholds, dict):
            for k, v in thresholds.items():
                if k in self.thresholds:
                    try:
                        self.thresholds[k] = float(v)
                    except Exception:
                        pass
        if profanity_mode in ("raw", "clean"):
            self.profanity_mode = profanity_mode
        self._save_memory()

    # ─── memory load/save ───
    def _load_memory(self):
        data = _load_json(self.mem_path)
        if not data:
            return
        try:
            if data.get("schema_version") and int(data.get("schema_version")) <= SCHEMA_VERSION:
                self.schema_version = int(data.get("schema_version"))
            if "persona" in data and data["persona"] in PERSONAS:
                self.persona = data["persona"]
                self.w = data.get("w", PERSONAS[self.persona]["w"])
                self.tone = PERSONAS[self.persona]["tone"]
            for k in ("mood","stress","energy"):
                if k in data:
                    setattr(self, k, _clamp01(data[k], getattr(self,k)))
            if "user_pref" in data and isinstance(data["user_pref"], dict):
                self.user_pref.update(data["user_pref"])
            if "bandit" in data: self.bandit = data["bandit"]
            if "total_uses" in data: self.total_uses = int(data["total_uses"])
            if "streaks" in data: self.streaks = data["streaks"]
            if "cooldowns" in data: self.cooldowns = data["cooldowns"]
            if "penalty_ttl" in data: self.penalty_ttl = data["penalty_ttl"]
            if "lessons" in data: self.lessons = data["lessons"]
            if "focus_tag" in data: self.focus_tag = data.get("focus_tag")
            if "profanity_mode" in data: self.profanity_mode = data.get("profanity_mode", self.profanity_mode)
            if self.schema_version < SCHEMA_VERSION:
                self.schema_version = SCHEMA_VERSION
        except Exception:
            pass

    def _save_memory(self):
        data = {
            "schema_version": self.schema_version,
            "persona": self.persona,
            "w": self.w,
            "tone": self.tone,
            "mood": self.mood,
            "stress": self.stress,
            "energy": self.energy,
            "user_pref": self.user_pref,
            "bandit": self.bandit,
            "total_uses": self.total_uses,
            "streaks": self.streaks,
            "cooldowns": self.cooldowns,
            "penalty_ttl": self.penalty_ttl,
            "lessons": self.lessons,
            "focus_tag": self.focus_tag,
            "profanity_mode": self.profanity_mode
        }
        _save_json_atomic(self.mem_path, data)

    # ─── scoring helpers ───
    def _asst_score(self, a: dict) -> float:
        base = 0.0
        base = base + a.get("impact",0.5) * self.w["impact"]
        base = base + (1.0 - a.get("effort",0.5)) * abs(self.w["effort"])
        base = base + a.get("social",0.5) * self.w["social"]
        base = base + a.get("novelty",0.5) * self.w["novelty"]
        base = base + (1.0 - a.get("risk",0.5) * self.w["risk_aversion"])
        base = base * (0.9 + 0.20 * self.energy - 0.15 * self.stress)
        return float(base)

    def _user_score(self, a: dict, user_mood: Optional[float]) -> float:
        p = self.user_pref
        s = 0.0
        s = s + a.get("impact",0.5) * p["impact"]
        s = s + (1.0 - a.get("effort",0.5)) * p["effort"]
        s = s + a.get("social",0.5) * p["social"]
        s = s + a.get("novelty",0.5) * p["novelty"]
        risk_penalty = a.get("risk",0.5) * (1.0 - p["risk_tolerance"])
        s = s + (1.0 - risk_penalty)
        mood = p["base_mood"] if user_mood is None else _clamp01(user_mood, p["base_mood"])
        s = s * (0.9 + 0.20 * mood)
        return float(s)

    def _ucb(self, kind: str) -> float:
        st = self.bandit.get(kind) or {"uses":0,"successes":0}
        n = int(st.get("uses",0)); s = int(st.get("successes",0))
        if n <= 0: return float("inf")
        mean = 0.0
        try:
            mean = float(s) / float(n)
        except Exception:
            mean = 0.0
        N = max(1, int(self.total_uses))
        return float(mean + math.sqrt((2.0 * math.log(float(N))) / float(n)))

    def _record_use(self, kind: str):
        st = self.bandit.get(kind) or {"uses":0,"successes":0,"sum_quality":0.0}
        st["uses"] = int(st.get("uses",0)) + 1
        self.bandit[kind] = st
        self.total_uses = self.total_uses + 1

    def _record_outcome(self, kind: str, success: bool, quality: float):
        st = self.bandit.get(kind) or {"uses":0,"successes":0,"sum_quality":0.0}
        if success:
            st["successes"] = int(st.get("successes",0)) + 1
        st["sum_quality"] = float(st.get("sum_quality",0.0)) + float(quality)
        self.bandit[kind] = st

    # ─── streak / cooldown / penalty TTL ───
    def _streak_boost(self, kind: str) -> float:
        st = self.streaks.get(kind) or {"win":0,"lose":0}
        win = int(st.get("win",0)); lose = int(st.get("lose",0))
        boost = 1.0 + min(0.20, 0.04 * max(0, win-1))
        malus = 1.0 - min(0.15, 0.03 * max(0, lose-1))
        val = boost * malus
        if val < 0.7: val = 0.7
        if val > 1.3: val = 1.3
        return float(val)

    def _streak_update(self, kind: str, success: bool):
        st = self.streaks.get(kind) or {"win":0,"lose":0}
        if success:
            st["win"] = int(st.get("win",0)) + 1
            st["lose"] = 0
        else:
            st["lose"] = int(st.get("lose",0)) + 1
            st["win"] = 0
        self.streaks[kind] = st

    def _cooldown_penalty(self, kind: str, now_ts: Optional[float]) -> float:
        k = _norm_text(kind).lower()
        if k not in self.cooldowns: return 0.0
        now = float(now_ts) if now_ts is not None else _now_ts()
        ready = float(self.cooldowns.get(k, 0.0))
        if ready <= now: return 0.0
        rem = ready - now
        norm = rem / 600.0
        if norm > 1.0: norm = 1.0
        if norm < 0.0: norm = 0.0
        return 0.5 * norm

    def set_cooldown(self, kind: str, minutes: int):
        now = _now_ts()
        self.cooldowns[_norm_text(kind).lower()] = now + float(max(1, minutes) * 60.0)

    def _get_penalty_ttl(self, key: str, now_ts: Optional[float]) -> float:
        now = float(now_ts) if now_ts is not None else _now_ts()
        info = self.penalty_ttl.get(key)
        if not info: return 0.0
        if float(info.get("exp",0.0)) <= now:
            try:
                del self.penalty_ttl[key]
            except Exception:
                pass
            return 0.0
        pen = float(info.get("pen",0.0))
        if pen < 0.0: pen = 0.0
        if pen > 1.0: pen = 1.0
        return pen

    def _add_penalty_ttl(self, key: str, pen: float, minutes: int):
        now = _now_ts()
        self.penalty_ttl[key] = {"pen": float(max(0.0, min(1.0, pen))), "exp": now + float(max(5, minutes*60))}

    # ─── memory integration (memory.py) ───
    def _mem_stm(self, mem: Any, user_text: str, assistant_text: str):
        if mem is None:
            return
        try:
            mem.stm_add(user_text, assistant_text)
        except Exception:
            pass

    def _mem_add_fact(self, mem: Any, text: str, tags: List[str]):
        if mem is None:
            return
        msg = _norm_text(text)
        try:
            mem.add_fact(msg, meta_data={"tags": tags})
        except Exception:
            try:
                mem.add_entry(msg, metadata={"tags": tags})
            except Exception:
                pass

    def _mem_episode(self, mem: Any, title: str, body: str, tags: List[str]):
        if mem is None:
            return
        try:
            meta = {"tags": tags}
            mem.add_episode(title, body, meta)
        except Exception:
            pass

    def _mem_profile_bias(self, mem: Any):
        if mem is None:
            return
        try:
            prof = mem.get_profile()
            if not isinstance(prof, dict):
                return
            if "likes_novelty" in prof:
                self.user_pref["novelty"] = _clamp01(float(prof.get("likes_novelty", 0.4)))
            if "risk_tolerance" in prof:
                self.user_pref["risk_tolerance"] = _clamp01(float(prof.get("risk_tolerance", 0.5)))
            if "socialness" in prof:
                self.user_pref["social"] = _clamp01(float(prof.get("socialness", 0.6)))
            if "base_mood" in prof:
                self.user_pref["base_mood"] = _clamp01(float(prof.get("base_mood", 0.5)))
        except Exception:
            pass

    def _mem_context(self, mem: Any, user_text: str) -> str:
        if mem is None:
            return ""
        try:
            ctx = mem.compose_context(user_text, limit_chars=1200, topk=12)
            if ctx:
                return ctx
            rec = mem.recall(user_text, topk=8)
            if not rec:
                return ""
            parts = []
            for tup in rec:
                try:
                    parts.append(str(tup[0]))
                except Exception:
                    continue
            joined = " ".join(parts)
            return joined[:1200]
        except Exception:
            return ""

    # ─── candidate generation ───
    def _slot_of(self, kind: str) -> str:
        k = _norm_text(kind).lower()
        for slot, arr in ACTION_LIBRARY.items():
            for a in arr:
                if _norm_text(a["kind"]).lower() == k:
                    return slot
        return "EXEC"

    def _focus_bias(self, act: dict) -> float:
        if not self.focus_tag:
            return 1.0
        slot = self._slot_of(act.get("kind",""))
        if self.focus_tag == "biznes" and slot in ("PLAN","COMM","EXEC"):
            return 1.05
        if self.focus_tag == "kod" and slot in ("EXEC","PLAN","CLARIFY"):
            return 1.06
        if self.focus_tag == "serwis" and slot in ("EXEC","PLAN"):
            return 1.05
        if self.focus_tag == "marketing" and slot in ("COMM","PLAN"):
            return 1.05
        return 1.0

    def _build_candidates(self, user_text: str, context_text: str, now_ts: Optional[float]) -> List[dict]:
        offsets = PERSONA_OFFSETS.get(self.persona) or {}
        cands: List[dict] = []
        for slot, arr in ACTION_LIBRARY.items():
            for a in arr:
                x = dict(a)
                off = offsets.get(slot) or {}
                for kk, vv in off.items():
                    x[kk] = _clamp01(x.get(kk,0.5) + vv, x.get(kk,0.5))
                x["freshness_ts"] = float(now_ts) if now_ts is not None else _now_ts()
                ctx = context_text.lower() if context_text else ""
                if ("deadline" in ctx or "pilne" in ctx) and slot in ("PLAN","EXEC"):
                    x["impact"] = _clamp01(x.get("impact",0.5) + 0.04)
                cands.append(x)

        t = _norm_text(user_text).lower()
        if ("napisz" in t) or ("mail" in t) or ("wiadom" in t):
            cands.append({"kind":"draft_message","description":"Szkic wiadomości kontekstowej","impact":0.64,"risk":0.10,"effort":0.22,"novelty":0.24,"social":0.64,"freshness_ts":float(_now_ts())})
        if ("plan" in t) or ("kroki" in t) or ("zrób plan" in t) or ("zrob plan" in t):
            cands.append({"kind":"quick_plan","description":"3 kroki + definicja sukcesu","impact":0.70,"risk":0.14,"effort":0.28,"novelty":0.30,"social":0.45,"freshness_ts":float(_now_ts())})
        if ("nie jasne" in t) or ("niejasne" in t) or ("doprecyz" in t):
            cands.append({"kind":"ask_clarify","description":"2 pytania doprecyzowujące","impact":0.56,"risk":0.04,"effort":0.15,"novelty":0.10,"social":0.52,"freshness_ts":float(_now_ts())})
        return cands

    # ─── main process (integrated with memory) ───
    def process(self, user_text: str, meta: dict = None, mem: Any = None) -> dict:
        if meta is None:
            meta = {}
        now_ts = float(meta.get("now_ts", _now_ts()))
        horizon = int(meta.get("horizon_minutes", 45))
        if horizon < self.thresholds["min_plan_minutes"]:
            horizon = int(self.thresholds["min_plan_minutes"])
        max_actions = int(meta.get("max_actions", 3))
        allow_autopilot = bool(meta.get("allow_autopilot", True))

        self._mem_profile_bias(mem)
        self._last_user_text = user_text
        user_mood = infer_user_mood(user_text)
        self.mood = max(0.0, min(1.0, (self.mood*0.8 + user_mood*0.2)))

        ctx = self._mem_context(mem, user_text)
        actions = self._build_candidates(user_text, ctx, now_ts)
        if not actions:
            decision = self._pack_decision([], [], user_mood, horizon, now_ts, False, "Brak kandydatów.")
            self._mem_stm(mem, user_text, decision["reply_tone"])
            self._save_memory()
            return decision

        valid: List[Tuple[dict, dict]] = []
        for raw in actions:
            a = {
                "kind": _norm_text(raw.get("kind") or "unknown")[:64],
                "description": _norm_text(raw.get("description") or "")[:1024],
                "impact": _clamp01(raw.get("impact",0.5)),
                "risk": _clamp01(raw.get("risk",0.5)),
                "effort": _clamp01(raw.get("effort",0.5)),
                "novelty": _clamp01(raw.get("novelty",0.5)),
                "social": _clamp01(raw.get("social",0.5)),
                "freshness_ts": float(raw.get("freshness_ts") or now_ts)
            }
            core = {"kind":a["kind"],"description":a["description"],"impact":a["impact"],"risk":a["risk"],"effort":a["effort"],"novelty":a["novelty"],"social":a["social"]}
            if not self._idem.allow(core, now_ts):
                continue

            asst = self._asst_score(a)
            usr = self._user_score(a, user_mood)
            ucb = self._ucb(a["kind"])
            if ucb == float("inf"):
                ucb_mix = 1.5
            else:
                ucb_mix = 0.5 + 0.5 * _clamp01(ucb, 0.0)

            ethics = ethical_penalty_minimal(user_text, a["description"])
            key = a["kind"] + "|" + a["description"]
            ethics_ttl = self._get_penalty_ttl(key, now_ts)
            cool = self._cooldown_penalty(a["kind"], now_ts)
            streak = self._streak_boost(a["kind"])
            fresh = self._freshness(a, now_ts)
            focus_b = self._focus_bias(a)

            base = (0.44 * usr + 0.40 * asst + 0.10 * ucb_mix) * fresh
            penalty = (ethics + ethics_ttl + cool)
            if penalty > 1.0:
                penalty = 1.0
            score = base * (1.0 - penalty) * streak * focus_b

            details = {
                "assistant": float(asst),
                "user": float(usr),
                "ucb": float(ucb if ucb != float("inf") else 1.0),
                "ucb_mix": float(ucb_mix),
                "freshness": float(fresh),
                "ethics": float(ethics),
                "ethics_ttl": float(ethics_ttl),
                "cooldown": float(cool),
                "streak": float(streak),
                "focus_bias": float(focus_b),
                "score": float(score)
            }
            valid.append((a, details))

        if not valid:
            decision = self._pack_decision([], [], user_mood, horizon, now_ts, False, "Wszystko odrzucone.")
            self._mem_stm(mem, user_text, decision["reply_tone"])
            self._save_memory()
            return decision

        valid.sort(key=lambda it: (-it[1]["score"], -(it[0].get("impact",0.5)+it[0].get("social",0.5)), it[0]["kind"]))
        chosen_pairs = valid[: max(1, int(max_actions))]
        proposals: List[dict] = []
        for idx, (a, d) in enumerate(chosen_pairs, start=1):
            self._record_use(a["kind"])
            proposals.append({"index": idx, "action": a, "score": float(d["score"]), "components": d})

        plan = self._plan_from_proposals(proposals, horizon)

        autopilot_enabled = False
        autopilot = []
        top = proposals[0] if proposals else None
        if allow_autopilot and top:
            top_kind = top["action"]["kind"]
            top_comp = top["components"]
            if self._slot_of(top_kind) == "EXEC":
                if (top_comp["score"] >= self.thresholds["autopilot_score"]) and (top_comp["ethics"] < self.thresholds["ethical_max"]) and (top_comp["cooldown"] <= self.thresholds["cooldown_max"]):
                    autopilot_enabled = True
                    autopilot.append({"kind": top_kind, "description": top["action"]["description"], "minutes": self._minutes_for_top(plan)})
                    self._mem_episode(mem, "autopilot:" + top_kind, "Wykonano akcję autopilota: " + top["action"]["description"], ["autopilot","exec"])

        self._mem_stm(mem, user_text, self._reply_tone_prefix(user_mood))

        try:
            for p in proposals:
                a = p.get("action") or {}
                sc = p.get("score", 0.0)
                txt = "propozycja " + str(a.get("kind","")) + " — " + str(a.get("description","")) + " score=" + ("{:.3f}".format(sc))
                self._mem_add_fact(mem, txt, ["proposal","psychika"])
        except Exception:
            pass

        self._save_memory()
        return self._pack_decision(proposals, plan, user_mood, horizon, now_ts, autopilot_enabled, None, autopilot)

    def _minutes_for_top(self, plan: List[dict]) -> int:
        if not plan:
            return 0
        for p in plan:
            if p.get("kind") != "break":
                return int(p.get("minutes",0))
        return 0

    def _freshness(self, a: dict, now_ts: Optional[float]) -> float:
        now = float(now_ts) if now_ts is not None else _now_ts()
        try:
            age = max(0.0, (now - float(a.get("freshness_ts", now))) / 86400.0)
        except Exception:
            age = 0.0
        f = 1.0 - 0.08 * age
        if f < 0.3:
            f = 0.3
        if f > 1.0:
            f = 1.0
        return float(f)

    def _plan_from_proposals(self, props: List[dict], total_minutes: int, gap_minutes: int = 0) -> List[dict]:
        if total_minutes <= 0 or not props:
            return []
        weights = []
        acts = []
        for it in props:
            acts.append(it.get("action") or {})
            weights.append(max(0.0, float(it.get("score",0.0))))
        total = sum(weights)
        if total <= 0.0:
            n = len(acts)
            base = int(total_minutes / n) if n > 0 else total_minutes
            if base < self.thresholds["min_plan_minutes"]:
                base = int(self.thresholds["min_plan_minutes"])
            alloc = [base for _ in range(n)]
        else:
            raw = [(w * float(total_minutes)) / total for w in weights]
            base = [int(x) for x in raw]
            remain = total_minutes - sum(base)
            frac = [(i, raw[i] - float(base[i])) for i in range(len(raw))]
            frac.sort(key=lambda t: (t[1], -t[0]), reverse=True)
            i = 0
            while remain > 0 and i < len(frac):
                base[frac[i][0]] = base[frac[i][0]] + 1
                remain = remain - 1
                i = i + 1
            alloc = base
        for j in range(len(alloc)):
            if alloc[j] < self.thresholds["min_plan_minutes"]:
                alloc[j] = int(self.thresholds["min_plan_minutes"])
        s = sum(alloc)
        if s > total_minutes:
            diff = s - total_minutes
            k = len(alloc) - 1
            while diff > 0 and k >= 0:
                if alloc[k] > self.thresholds["min_plan_minutes"]:
                    alloc[k] = alloc[k] - 1
                    diff = diff - 1
                else:
                    k = k - 1
        elif s < total_minutes and len(alloc) > 0:
            alloc[0] = alloc[0] + (total_minutes - s)
        plan = []
        for i, a in enumerate(acts):
            plan.append({"kind": a.get("kind","block"), "minutes": int(alloc[i]), "description": a.get("description","")})
            if gap_minutes and i < len(acts) - 1:
                plan.append({"kind":"break","minutes": int(gap_minutes), "description":"przerwa"})
        return plan

    # ─── apply_feedback ───
    def apply_feedback(self, action_kind: str, success: bool, quality: float, user_text: str = None, mem: Any = None) -> dict:
        kind = _norm_text(action_kind).lower()
        q = _clamp01(quality, 0.5)
        self._record_outcome(kind, bool(success), q)
        self._streak_update(kind, bool(success))

        try:
            status = "ok" if success else "fail"
            line = "feedback " + kind + " " + status + " q=" + ("{:.2f}".format(q))
            self._mem_add_fact(mem, line, ["feedback","psychika"])
        except Exception:
            pass

        fails = 0
        for it in self.lessons:
            if _norm_text(it.get("kind","")).lower() == kind and not it.get("success", False):
                fails = fails + 1
        if not success and fails >= 2:
            key = kind + "|*"
            self._add_penalty_ttl(key, 0.25, 30)

        stamp = int(_now_ts())
        self.lessons.append({"ts":stamp,"kind":kind,"desc":"","success":bool(success),"quality":q})

        if success and q > 0.6:
            self.user_pref["impact"] = _clamp01(self.user_pref["impact"] + 0.02)
            self.user_pref["risk_tolerance"] = _clamp01(self.user_pref["risk_tolerance"] + 0.01)
        if not success and q < 0.4:
            self.user_pref["effort"] = _clamp01(self.user_pref["effort"] + 0.03)

        if success:
            self.mood = _clamp01(self.mood + 0.06, self.mood)
            self.stress = _clamp01(self.stress - 0.05, self.stress)
            self.energy = max(0.0, self.energy - 0.04)
        else:
            self.mood = _clamp01(self.mood - 0.08, self.mood)
            self.stress = _clamp01(self.stress + 0.08, self.stress)
            self.energy = max(0.0, self.energy - 0.07)

        if user_text:
            u = infer_user_mood(user_text)
            self.mood = max(0.0, min(1.0, (self.mood*0.7 + u*0.3)))

        self._save_memory()
        return {"ok":True, "state": self.snapshot()}

    # ─── reply tone / snapshot / export/import ───
    def _reply_tone_prefix(self, user_mood: float) -> str:
        if user_mood < 0.35:
            base = "Lecimy konkretnie, bez lania wody."
        elif user_mood > 0.7:
            base = "Dobra energia – robimy to sprawnie."
        else:
            base = "Robimy to rzeczowo i spójnie."
        return base + " (" + self.persona + ", " + self.tone + ")"
    def snapshot(self) -> dict:
        tail = self.lessons[-5:] if len(self.lessons) > 5 else self.lessons
        return {
            "persona": self.persona,
            "tone": self.tone,
            "focus": self.focus_tag,
            "mood": self.mood,
            "stress": self.stress,
            "energy": self.energy,
            "weights": self.w,
            "user_pref": self.user_pref,
            "total_uses": self.total_uses,
            "streaks": self.streaks,
            "cooldowns": self.cooldowns,
            "penalty_ttl_keys": list(self.penalty_ttl.keys()),
            "lessons_recent": tail,
            "profanity_mode": self.profanity_mode
        }

    def export_memory(self, path: str) -> bool:
        data = _load_json(self.mem_path)
        if not data:
            data = {}
        try:
            _save_json_atomic(path, data)
            return True
        except Exception:
            return False

    def import_memory(self, path: str) -> bool:
        data = _load_json(path)
        if not data:
            return False
        try:
            _save_json_atomic(self.mem_path, data)
            self._load_memory()
            return True
        except Exception:
            return False

    def _pack_decision(self, proposals: List[dict], plan: List[dict], user_mood: float, horizon: int, now_ts: float, autopilot_enabled: bool, note: str = None, autopilot: List[dict] = None) -> dict:
        if autopilot is None:
            autopilot = []
        decision = {
            "ts": float(now_ts),
            "persona": self.persona,
            "focus": self.focus_tag,
            "user_mood": float(user_mood),
            "reply_tone": self._reply_tone_prefix(user_mood),
            "horizon_minutes": int(horizon),
            "proposals": proposals,
            "plan": plan,
            "autopilot": {"enabled": bool(autopilot_enabled), "execute": autopilot},
            "why": [{"index":p["index"],"kind":p["action"]["kind"],"score":p["score"],"components":p["components"]} for p in proposals],
            "state": self.snapshot()
        }
        if note:
            decision["note"] = str(note)
        return decision

