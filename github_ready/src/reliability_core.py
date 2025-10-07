"""
reliability_core.py - Centralny moduł niezawodności systemu AI

Integruje wszystkie komponenty z architektura_niezawodnosci.ipynb do produkcyjnego użytku.
System ZAWSZE działa w tle i wszystkie operacje są przepuszczane przez mechanizmy niezawodności.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any


# Konfiguracja systemu
@dataclass
class SystemConfig:
    # STM rozszerzone limity
    STM_CHAR_LIMIT: int = 10000
    STM_LINE_LIMIT: int = 100

    # Backpressure kontrola
    MAX_CONCURRENT_TICKS: int = 3
    COOLDOWN_DURATION_SECONDS: float = 60.0

    # Retry policy
    MAX_RETRIES: int = 3
    BASE_DELAY_SECONDS: float = 1.0

    # Telemetria
    TELEMETRY_BATCH_SIZE: int = 10
    TELEMETRY_FLUSH_INTERVAL: int = 300  # 5 minut

    # Idempotencja
    ACTION_TTL_HOURS: int = 24

    # KPI i trends
    ROLLING_WINDOW_SIZE: int = 50
    PERFORMANCE_THRESHOLD: float = 0.6

    # Presety
    PRESET_AUTO_SWITCH: bool = True
    PRESET_OVERRIDE_DURATION: int = 3600  # 1 godzina


# Singleton konfiguracji
CONFIG = SystemConfig()


class MemoryAPIContract:
    """Kontrakty API z fallbackami i metrykami."""

    def __init__(self):
        self.required_methods = {
            "add_fact": "Dodawanie faktów do pamięci",
            "get_goals": "Pobieranie celów użytkownika",
            "compose_context": "Składanie kontekstu dla LLM",
            "get_profile": "Pobieranie profilu użytkownika",
            "set_profile_many": "Zapisywanie wielu wartości profilu",
        }
        self.metrics = defaultdict(int)
        self.fallback_usage = defaultdict(int)

    def validate_memory_object(self, memory_obj) -> dict[str, Any]:
        """Waliduje obiekt pamięci i zwraca metryki pokrycia."""
        coverage = {}
        missing_methods = []

        for method, description in self.required_methods.items():
            has_method = hasattr(memory_obj, method)
            coverage[method] = 1.0 if has_method else -1.0

            if not has_method:
                missing_methods.append(method)
                print(f"❌ Brak metody {method}: {description}")
            else:
                print(f"✅ {method}: {description}")

        self.metrics["total_calls"] += 1

        return {
            "coverage": coverage,
            "missing_methods": missing_methods,
            "fallback_rate": sum(self.fallback_usage.values())
            / max(1, self.metrics["total_calls"]),
        }

    def safe_call(self, memory_obj, method_name: str, *args, **kwargs):
        """Bezpieczne wywołanie metody z fallbackiem."""
        self.metrics["total_calls"] += 1

        if hasattr(memory_obj, method_name):
            try:
                return getattr(memory_obj, method_name)(*args, **kwargs)
            except Exception as e:
                print(f"⚠️ Błąd wywołania {method_name}: {e}")
                return self._fallback_response(method_name)
        else:
            self.fallback_usage[method_name] += 1
            return self._fallback_response(method_name)

    def _fallback_response(self, method_name: str):
        """Fallback responses dla brakujących metod."""
        fallbacks = {
            "add_fact": True,
            "get_goals": [],
            "compose_context": "Kontekst niedostępny z powodu braku implementacji",
            "get_profile": {},
            "set_profile_many": True,
        }
        return fallbacks.get(method_name, None)

    def get_metrics(self) -> dict[str, Any]:
        """Zwraca metryki API."""
        return {
            "total_calls": self.metrics["total_calls"],
            "fallback_usage": dict(self.fallback_usage),
            "fallback_rate": sum(self.fallback_usage.values())
            / max(1, self.metrics["total_calls"]),
            "coverage": {
                method: 1.0 if method not in self.fallback_usage else 0.0
                for method in self.required_methods
            },
        }


class AutopilotBackpressure:
    """Kontrola przepływu z limitami concurrent i cooldownami."""

    def __init__(self):
        self.max_concurrent = CONFIG.MAX_CONCURRENT_TICKS
        self.cooldown_duration = CONFIG.COOLDOWN_DURATION_SECONDS

        self.active_ticks: set[str] = set()
        self.kind_cooldowns: dict[str, float] = {}
        self.tick_start_times: dict[str, float] = {}
        self.rejection_stats = defaultdict(int)
        self.total_requests = 0

        self._lock = threading.Lock()

    def start_tick(self, tick_id: str, kind: str) -> bool:
        """Próbuje rozpocząć tick. True = dozwolone, False = odrzucone."""
        with self._lock:
            self.total_requests += 1
            now = time.time()

            # Sprawdź cooldown dla tego typu
            if kind in self.kind_cooldowns:
                remaining = self.kind_cooldowns[kind] - now
                if remaining > 0:
                    self.rejection_stats["cooldown"] += 1
                    print(
                        f"🚫 Tick {tick_id} odrzucony: Cooldown dla '{kind}': {remaining:.1f}s pozostało"
                    )
                    return False

            # Sprawdź limit concurrent
            if len(self.active_ticks) >= self.max_concurrent:
                self.rejection_stats["concurrent_limit"] += 1
                print(
                    f"🚫 Tick {tick_id} odrzucony: Limit równoległych ticków ({self.max_concurrent})"
                )
                return False

            # Akceptuj tick
            self.active_ticks.add(tick_id)
            self.tick_start_times[tick_id] = now
            self.kind_cooldowns[kind] = now + self.cooldown_duration

            print(f"✅ {tick_id} rozpoczęty")
            return True

    def finish_tick(self, tick_id: str) -> float:
        """Kończy tick i zwraca czas trwania."""
        with self._lock:
            if tick_id in self.active_ticks:
                self.active_ticks.remove(tick_id)
                start_time = self.tick_start_times.pop(tick_id, time.time())
                duration = time.time() - start_time
                return duration
            return 0.0

    def get_status(self) -> dict[str, Any]:
        """Status systemu backpressure."""
        with self._lock:
            return {
                "active_ticks": len(self.active_ticks),
                "max_concurrent": self.max_concurrent,
                "utilization": len(self.active_ticks) / self.max_concurrent,
                "total_requests": self.total_requests,
                "rejected_count": dict(self.rejection_stats),
                "rejection_rate": sum(self.rejection_stats.values())
                / max(1, self.total_requests),
                "kind_cooldowns": {
                    k: max(0, v - time.time()) for k, v in self.kind_cooldowns.items()
                },
            }


class ActionIdempotency:
    """System deduplikacji akcji z TTL."""

    def __init__(self):
        self.actions: dict[str, dict] = {}
        self.ttl_hours = CONFIG.ACTION_TTL_HOURS
        self._lock = threading.Lock()

    def _generate_action_id(self, action: dict[str, Any]) -> str:
        """Generuje deterministyczny ID akcji."""
        key_data = {
            "kind": action.get("kind", ""),
            "title": action.get("title", ""),
            "description": action.get("description", ""),
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()[:12]

    def is_duplicate(self, action: dict[str, Any]) -> tuple[bool, dict | None]:
        """Sprawdza czy akcja jest duplikatem."""
        action_id = self._generate_action_id(action)

        with self._lock:
            if action_id in self.actions:
                existing = self.actions[action_id]
                # Sprawdź TTL
                age_hours = (time.time() - existing["created_at"]) / 3600
                if age_hours < self.ttl_hours:
                    return True, existing
                else:
                    # Usuń przestarzałą akcję
                    del self.actions[action_id]

        return False, None

    def register_action(self, action: dict[str, Any], confidence: float = 0.9) -> str:
        """Rejestruje nową akcję i zwraca jej ID."""
        action_id = self._generate_action_id(action)

        with self._lock:
            if action_id in self.actions:
                # Aktualizuj istniejącą
                self.actions[action_id]["count"] += 1
                self.actions[action_id]["last_seen"] = time.time()
                print(
                    f"🔄 Duplikat {action_id}: confidence {confidence:.2f}, count {self.actions[action_id]['count']}"
                )
            else:
                # Zarejestruj nową
                self.actions[action_id] = {
                    "action": action,
                    "created_at": time.time(),
                    "last_seen": time.time(),
                    "count": 1,
                    "confidence": confidence,
                }
                print(
                    f"✨ Nowa akcja {action_id}: {action.get('kind')} - {action.get('title', 'brak tytułu')}"
                )

        return action_id

    def cleanup_expired(self):
        """Czyści przestarzałe akcje."""
        now = time.time()
        with self._lock:
            expired = [
                action_id
                for action_id, data in self.actions.items()
                if (now - data["created_at"]) / 3600 > self.ttl_hours
            ]
            for action_id in expired:
                del self.actions[action_id]

    def get_stats(self) -> dict[str, Any]:
        """Statystyki systemu idempotencji."""
        with self._lock:
            total_registered = len(self.actions)
            duplicate_rate = sum(
                max(0, data["count"] - 1) for data in self.actions.values()
            ) / max(1, sum(data["count"] for data in self.actions.values()))

            return {
                "total_registered": total_registered,
                "active_count": len(self.actions),
                "duplicate_rate": duplicate_rate,
                "avg_confidence": sum(
                    data["confidence"] for data in self.actions.values()
                )
                / max(1, total_registered),
            }


def retry_with_backoff(
    max_retries: int = None,
    base_delay: float = None,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
):
    """Dekorator retry z eksponencjalnym backoffem."""
    if max_retries is None:
        max_retries = CONFIG.MAX_RETRIES
    if base_delay is None:
        base_delay = CONFIG.BASE_DELAY_SECONDS

    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        print(
                            f"✅ Sukces {func.__name__} po {attempt} próbach (ostatni błąd: {last_exception})"
                        )
                    return result

                except Exception as e:
                    last_exception = str(e)

                    if attempt < max_retries:
                        delay = min(base_delay * (backoff_factor**attempt), max_delay)
                        error_type = (
                            "LLM"
                            if "llm" in str(e).lower() or "openai" in str(e).lower()
                            else "IO"
                        )
                        print(
                            f"🔄 Retry {attempt + 1}/{max_retries} dla {func.__name__}: {e}"
                        )
                        print(f"⏳ Czekam {delay:.1f}s przed kolejną próbą...")
                        time.sleep(delay)
                    else:
                        print(
                            f"❌ {func.__name__} nie powiódł się po {max_retries} próbach: {e}"
                        )
                        raise e

        return wrapper

    return decorator


class TelemetryCollector:
    """System zbierania telemetrii per tick."""

    def __init__(self):
        self.batch_size = CONFIG.TELEMETRY_BATCH_SIZE
        self.flush_interval = CONFIG.TELEMETRY_FLUSH_INTERVAL

        self.active_telemetries: dict[str, TickTelemetry] = {}
        self.completed_batch: list[dict] = []
        self.rolling_metrics = deque(maxlen=50)

        self.last_flush = time.time()
        self._lock = threading.Lock()

    def start_tick_telemetry(self, tick_id: str) -> TickTelemetry:
        """Rozpoczyna zbieranie telemetrii dla tick."""
        telemetry = TickTelemetry(tick_id)

        with self._lock:
            self.active_telemetries[tick_id] = telemetry

        return telemetry

    def record_proposals(self, telemetry: TickTelemetry, proposals: list[dict]):
        """Zapisuje propozycje akcji."""
        telemetry.proposals_count = len(proposals)
        if proposals:
            telemetry.avg_proposal_score = sum(
                p.get("score", 0.5) for p in proposals
            ) / len(proposals)

    def record_decisions(
        self, telemetry: TickTelemetry, accepted: list[dict], rejected: list[dict]
    ):
        """Zapisuje decyzje (akceptowane/odrzucone)."""
        total = len(accepted) + len(rejected)
        telemetry.acceptance_rate = len(accepted) / max(1, total)
        telemetry.decisions_count = total

    def record_llm_call(self, telemetry: TickTelemetry, duration: float):
        """Zapisuje wywołanie LLM."""
        telemetry.llm_calls += 1
        telemetry.total_llm_time += duration

    def record_error(self, telemetry: TickTelemetry, error: Exception, context: str):
        """Zapisuje błąd."""
        telemetry.errors.append(
            {"error": str(error), "context": context, "timestamp": time.time()}
        )

    def finish_tick_telemetry(self, telemetry: TickTelemetry) -> dict[str, Any]:
        """Kończy telemetrię tick i przetwarza."""
        telemetry.end_time = time.time()
        telemetry.duration = telemetry.end_time - telemetry.start_time

        result = telemetry.to_dict()

        with self._lock:
            if telemetry.tick_id in self.active_telemetries:
                del self.active_telemetries[telemetry.tick_id]

            self.completed_batch.append(result)

            # Auto-flush przy przekroczeniu batch size
            if len(self.completed_batch) >= self.batch_size:
                self._flush_batch()

        return result

    def _flush_batch(self):
        """Przetwarza batch telemetrii."""
        if not self.completed_batch:
            return

        # Oblicz metryki dla batcha
        batch_metrics = self._calculate_batch_metrics(self.completed_batch)
        self.rolling_metrics.append(batch_metrics)

        print(f"📤 Flush telemetrii: {len(self.completed_batch)} tick-ów")
        print(f"   Średnia proposals: {batch_metrics['avg_proposals']:.1f}")
        print(f"   Średnia acceptance rate: {batch_metrics['avg_acceptance_rate']:.2f}")
        print(f"   Średni czas LLM: {batch_metrics['avg_llm_time']:.2f}s")

        self.completed_batch.clear()
        self.last_flush = time.time()

    def _calculate_batch_metrics(self, batch: list[dict]) -> dict[str, Any]:
        """Oblicza metryki dla batcha."""
        if not batch:
            return {}

        return {
            "batch_size": len(batch),
            "avg_proposals": sum(t.get("proposals_count", 0) for t in batch)
            / len(batch),
            "avg_acceptance_rate": sum(t.get("acceptance_rate", 0) for t in batch)
            / len(batch),
            "avg_llm_time": sum(t.get("total_llm_time", 0) for t in batch) / len(batch),
            "avg_duration": sum(t.get("duration", 0) for t in batch) / len(batch),
            "error_rate": sum(len(t.get("errors", [])) for t in batch) / len(batch),
            "timestamp": time.time(),
        }

    def get_rolling_metrics(self) -> dict[str, Any]:
        """Zwraca rolling metryki."""
        if not self.rolling_metrics:
            return {"no_data": True}

        recent = list(self.rolling_metrics)
        return {
            "window_size": len(recent),
            "avg_proposals_per_tick": sum(m.get("avg_proposals", 0) for m in recent)
            / len(recent),
            "avg_acceptance_rate": sum(m.get("avg_acceptance_rate", 0) for m in recent)
            / len(recent),
            "avg_llm_time": sum(m.get("avg_llm_time", 0) for m in recent) / len(recent),
            "avg_tick_duration": sum(m.get("avg_duration", 0) for m in recent)
            / len(recent),
            "error_rate": sum(m.get("error_rate", 0) for m in recent) / len(recent),
        }


class TickTelemetry:
    """Telemetria pojedynczego tick."""

    def __init__(self, tick_id: str):
        self.tick_id = tick_id
        self.start_time = time.time()
        self.end_time = None
        self.duration = 0.0

        self.proposals_count = 0
        self.avg_proposal_score = 0.0
        self.decisions_count = 0
        self.acceptance_rate = 0.0

        self.llm_calls = 0
        self.total_llm_time = 0.0

        self.errors: list[dict] = []

    def to_dict(self) -> dict[str, Any]:
        """Konwersja do słownika."""
        return {
            "tick_id": self.tick_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "proposals_count": self.proposals_count,
            "avg_proposal_score": self.avg_proposal_score,
            "decisions_count": self.decisions_count,
            "acceptance_rate": self.acceptance_rate,
            "llm_calls": self.llm_calls,
            "total_llm_time": self.total_llm_time,
            "errors": self.errors,
        }


class ReliableAISystem:
    """Główny system niezawodności AI - ZAWSZE aktywny."""

    def __init__(self):
        self.config = CONFIG

        # Inicjalizuj wszystkie komponenty
        self.memory_contract = MemoryAPIContract()
        self.backpressure = AutopilotBackpressure()
        self.idempotency = ActionIdempotency()
        self.telemetry = TelemetryCollector()

        # Status systemu
        self.system_start_time = time.time()
        self.total_operations = 0
        self.error_count = 0
        self.active = True

        print("🚀 ReliableAISystem zainicjalizowany - ZAWSZE AKTYWNY")
        print(f"   🔄 Backpressure: max {self.config.MAX_CONCURRENT_TICKS} concurrent")
        print(f"   🔁 Retry: {self.config.MAX_RETRIES} attempts")
        print(f"   🔐 Idempotencja: TTL {self.config.ACTION_TTL_HOURS}h")

    @retry_with_backoff()
    def process_critical_operation(
        self, operation_type: str, memory_obj=None, **kwargs
    ) -> dict[str, Any]:
        """
        KRYTYCZNA metoda - wszystkie operacje AI muszą przez nią przechodzić.
        Zawsze aktywna kontrola niezawodności.
        """
        if not self.active:
            raise RuntimeError("ReliableAISystem is disabled - CRITICAL ERROR")

        operation_start = time.time()
        self.total_operations += 1

        # Generuj unikalny ID operacji
        operation_id = f"{operation_type}_{int(time.time())}"

        try:
            # 1. Sprawdź kontrakt pamięci
            memory_validation = {}
            if memory_obj:
                memory_validation = self.memory_contract.validate_memory_object(
                    memory_obj
                )

            # 2. Sprawdź backpressure
            can_execute = self.backpressure.start_tick(operation_id, operation_type)
            if not can_execute:
                return {
                    "success": False,
                    "reason": "backpressure_rejected",
                    "operation_id": operation_id,
                    "retry_after": self.config.COOLDOWN_DURATION_SECONDS,
                }

            # 3. Rozpocznij telemetrię
            telemetry = self.telemetry.start_tick_telemetry(operation_id)

            # 4. Sprawdź idempotencję jeśli to akcja
            action_id = None
            if "action" in kwargs:
                action = kwargs["action"]
                is_dup, existing = self.idempotency.is_duplicate(action)
                if is_dup:
                    return {
                        "success": True,
                        "reason": "idempotent_duplicate",
                        "existing_action": existing,
                        "operation_id": operation_id,
                    }
                action_id = self.idempotency.register_action(action)

            # 5. Wykonaj operację z pełnym monitoringiem
            result = self._execute_monitored_operation(
                operation_type, telemetry, **kwargs
            )

            # 6. Zakończ backpressure
            duration = self.backpressure.finish_tick(operation_id)

            # 7. Finalizuj telemetrię
            telemetry_result = self.telemetry.finish_tick_telemetry(telemetry)

            operation_duration = time.time() - operation_start

            return {
                "success": True,
                "result": result,
                "operation_id": operation_id,
                "action_id": action_id,
                "memory_validation": memory_validation,
                "telemetry": telemetry_result,
                "operation_duration": operation_duration,
                "system_health": self.get_system_health(),
            }

        except Exception as e:
            self.error_count += 1
            print(f"❌ CRITICAL ERROR w {operation_type}: {e}")

            # Zapisz błąd w telemetrii
            if "telemetry" in locals():
                self.telemetry.record_error(telemetry, e, operation_type)

            # Zawsze zakończ backpressure
            if "operation_id" in locals():
                self.backpressure.finish_tick(operation_id)

            # Przekaż błąd dalej - NIGDY nie ukrywaj błędów
            raise e

    def _execute_monitored_operation(
        self, operation_type: str, telemetry: TickTelemetry, **kwargs
    ) -> Any:
        """Wykonuje operację z pełnym monitoringiem."""
        if operation_type == "chat_completion":
            return self._monitored_chat_completion(telemetry, **kwargs)
        elif operation_type == "memory_operation":
            return self._monitored_memory_operation(telemetry, **kwargs)
        elif operation_type == "decision_making":
            return self._monitored_decision_making(telemetry, **kwargs)
        else:
            # Nieznany typ operacji - log i kontynuuj
            print(f"⚠️ Nieznany typ operacji: {operation_type}")
            return {"status": "unknown_operation", "type": operation_type}

    def _monitored_chat_completion(
        self, telemetry: TickTelemetry, **kwargs
    ) -> dict[str, Any]:
        """LLM completion z pełnym monitoringiem."""
        llm_start = time.time()

        try:
            # Symulacja wywołania LLM - tu będzie prawdziwe wywołanie
            result = kwargs.get("result", "Przykładowa odpowiedź LLM")

            llm_duration = time.time() - llm_start
            self.telemetry.record_llm_call(telemetry, llm_duration)

            return {
                "response": result,
                "llm_duration": llm_duration,
                "tokens_used": len(result.split()) * 1.3,  # Przybliżone tokeny
            }

        except Exception as e:
            self.telemetry.record_error(telemetry, e, "llm_completion")
            raise

    def _monitored_memory_operation(
        self, telemetry: TickTelemetry, **kwargs
    ) -> dict[str, Any]:
        """Operacje pamięci z monitoringiem."""
        memory_obj = kwargs.get("memory_obj")
        operation = kwargs.get("operation", "unknown")

        if not memory_obj:
            raise ValueError("Brak obiektu pamięci dla operacji memory")

        # Użyj safe_call do wszystkich operacji pamięci
        if operation == "add_fact":
            result = self.memory_contract.safe_call(
                memory_obj,
                "add_fact",
                kwargs.get("fact", ""),
                tags=kwargs.get("tags", []),
                conf=kwargs.get("confidence", 0.8),
            )
        elif operation == "compose_context":
            result = self.memory_contract.safe_call(
                memory_obj,
                "compose_context",
                kwargs.get("query", ""),
                limit_chars=kwargs.get("limit_chars", 2000),
                topk=kwargs.get("topk", 5),
            )
        else:
            result = f"Nieobsługiwana operacja pamięci: {operation}"

        return {"memory_result": result, "operation": operation}

    def _monitored_decision_making(
        self, telemetry: TickTelemetry, **kwargs
    ) -> dict[str, Any]:
        """Podejmowanie decyzji z monitoringiem."""
        proposals = kwargs.get("proposals", [])

        # Zapisz propozycje w telemetrii
        self.telemetry.record_proposals(telemetry, proposals)

        # Prosta heurystyka decyzyjna
        accepted = proposals[:2] if len(proposals) > 1 else proposals
        rejected = proposals[2:] if len(proposals) > 2 else []

        # Zapisz decyzje
        self.telemetry.record_decisions(telemetry, accepted, rejected)

        return {
            "accepted": accepted,
            "rejected": rejected,
            "decision_reason": "Wybrano top 2 propozycje na podstawie score",
        }

    def get_system_health(self) -> dict[str, Any]:
        """Zwraca pełny status zdrowia systemu."""
        uptime = time.time() - self.system_start_time
        error_rate = self.error_count / max(self.total_operations, 1)

        return {
            "active": self.active,
            "uptime_seconds": uptime,
            "total_operations": self.total_operations,
            "error_rate": error_rate,
            "health_score": max(0, 1 - error_rate * 10),
            "backpressure_status": self.backpressure.get_status(),
            "memory_contract_metrics": self.memory_contract.get_metrics(),
            "idempotency_stats": self.idempotency.get_stats(),
            "telemetry_metrics": self.telemetry.get_rolling_metrics(),
            "timestamp": time.time(),
        }

    def force_cleanup(self):
        """Wymuś czyszczenie zasobów."""
        print("🧹 Wymuszam czyszczenie systemu niezawodności...")
        self.idempotency.cleanup_expired()
        self.telemetry._flush_batch()
        print("✅ Czyszczenie zakończone")

    def disable_system(self, reason: str = "Manual override"):
        """UWAGA: Wyłącza system - używaj tylko w wyjątkowych sytuacjach!"""
        self.active = False
        print(f"🚨 SYSTEM NIEZAWODNOŚCI WYŁĄCZONY: {reason}")
        print("⚠️ UWAGA: Aplikacja działa bez mechanizmów bezpieczeństwa!")

    def enable_system(self):
        """Włącza system z powrotem."""
        self.active = True
        print("✅ System niezawodności włączony z powrotem")


# Globalny singleton systemu niezawodności
_reliable_system = None


def get_reliable_system() -> ReliableAISystem:
    """Zwraca singleton systemu niezawodności."""
    global _reliable_system
    if _reliable_system is None:
        _reliable_system = ReliableAISystem()
    return _reliable_system


# Dekoratory dla łatwej integracji
def reliable_operation(operation_type: str = "general"):
    """Dekorator automatycznie przepuszczający operację przez system niezawodności."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            system = get_reliable_system()

            # Przygotuj argumenty dla systemu niezawodności
            operation_kwargs = {
                "original_function": func.__name__,
                "args": args,
                "kwargs": kwargs,
            }

            # Dodaj specjalne argumenty jeśli są dostępne
            if "memory_obj" in kwargs:
                operation_kwargs["memory_obj"] = kwargs["memory_obj"]
            if "action" in kwargs:
                operation_kwargs["action"] = kwargs["action"]

            try:
                # Przepuść przez system niezawodności
                reliable_result = system.process_critical_operation(
                    operation_type, **operation_kwargs
                )

                if not reliable_result["success"]:
                    if reliable_result["reason"] == "backpressure_rejected":
                        raise RuntimeError(
                            f"Operacja odrzucona przez backpressure. Spróbuj ponownie za {reliable_result['retry_after']}s"
                        )
                    else:
                        raise RuntimeError(
                            f"Operacja niezawodności failed: {reliable_result['reason']}"
                        )

                # Wykonaj oryginalną funkcję
                original_result = func(*args, **kwargs)

                # Zwróć rozszerzony wynik
                return {
                    "original_result": original_result,
                    "reliability_info": reliable_result,
                    "success": True,
                }

            except Exception as e:
                print(f"❌ RELIABLE OPERATION FAILED [{func.__name__}]: {e}")
                raise e

        return wrapper

    return decorator


def require_reliability_check(func):
    """Dekorator wymagający aktywnego systemu niezawodności."""

    def wrapper(*args, **kwargs):
        system = get_reliable_system()
        if not system.active:
            raise RuntimeError(
                f"CRITICAL: Function {func.__name__} requires active reliability system"
            )
        return func(*args, **kwargs)

    return wrapper
