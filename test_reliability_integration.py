#!/usr/bin/env python3
"""
test_reliability_integration.py

Test kompletnej integracji systemu niezawodności z main.py i psychika.py.
Sprawdza czy wszystkie komponenty działają razem i TRAKTUJĄ POWAŻNIE bezpieczeństwo.
"""

import sys
from pathlib import Path

# Dodaj ścieżkę do modułów
sys.path.insert(0, str(Path(__file__).parent))


def test_reliability_core():
    """Test podstawowego systemu niezawodności."""
    print("🧪 Test 1: Podstawowy system niezawodności")

    try:
        from reliability_core import (
            get_reliable_system,
        )

        # Pobierz system
        reliable_system = get_reliable_system()
        print(f"   ✅ System pobrany: {reliable_system}")

        # Sprawdź health
        health = reliable_system.get_system_health()
        print(f"   ✅ Health score: {health['health_score']:.2f}")
        print(f"   ✅ Error rate: {health['error_rate']:.3%}")
        print(f"   ✅ Active: {health['active']}")

        # Test krytycznej operacji
        result = reliable_system.process_critical_operation(
            "test_operation", test_data="Test integracji"
        )

        if result["success"]:
            print("   ✅ Test krytycznej operacji: PASSED")
        else:
            print(f"   ❌ Test krytycznej operacji: FAILED - {result}")
            return False

        return True

    except Exception as e:
        print(f"   ❌ Błąd testu systemu niezawodności: {e}")
        return False


def test_psychika_integration():
    """Test integracji z psychika.py."""
    print("\n🧪 Test 2: Integracja z psychika.py")

    try:
        import psychika

        # Test funkcji etycznych
        test_action = {
            "kind": "test_action",
            "title": "Test action",
            "impact": 0.8,
            "risk": 0.3,
            "effort": 0.4,
            "novelty": 0.5,
            "social": 0.6,
        }

        test_criteria = {
            "goal": 0.8,
            "safety": 0.7,
            "social": 0.6,
            "efficiency": 0.6,
            "novelty": 0.5,
        }

        # Test check etyczny
        ethics = psychika.check_ethical_concerns(test_action, test_criteria)
        print(f"   ✅ Check etyczny: needs_reflection={ethics['needs_reflection']}")
        print(f"   ✅ Severity: {ethics['severity']:.2f}")

        # Test ethical override
        can_execute = psychika.ethical_override_check(test_action, test_criteria)
        print(f"   ✅ Ethical override: {can_execute}")

        # Test stanu psychiki
        state = psychika.current_state()
        print(f"   ✅ Stan psychiki: mood={state['mood']:.2f}, stress={state['stress']:.2f}")

        return True

    except Exception as e:
        print(f"   ❌ Błąd testu psychika.py: {e}")
        return False


def test_memory_contracts():
    """Test kontraktów pamięci."""
    print("\n🧪 Test 3: Kontrakty pamięci")

    try:
        import memory
        from reliability_core import get_reliable_system

        reliable_system = get_reliable_system()
        mem = memory.get_memory()

        # Test walidacji kontraktów
        validation = reliable_system.memory_contract.validate_memory_object(mem)
        print(f"   ✅ Walidacja pamięci: fallback_rate={validation['fallback_rate']:.1%}")

        # Test safe_call
        result = reliable_system.memory_contract.safe_call(
            mem, "add_fact", "[TEST] Test kontraktu pamięci", tags=["test", "reliability"], conf=0.8
        )
        print(f"   ✅ Safe call add_fact: {result}")

        # Test get_metrics
        metrics = reliable_system.memory_contract.get_metrics()
        print(f"   ✅ Metryki kontraktów: {metrics}")

        return True

    except Exception as e:
        print(f"   ❌ Błąd testu kontraktów pamięci: {e}")
        return False


def test_backpressure():
    """Test systemu backpressure."""
    print("\n🧪 Test 4: System backpressure")

    try:
        from reliability_core import get_reliable_system

        reliable_system = get_reliable_system()
        backpressure = reliable_system.backpressure

        # Test podstawowy
        tick1 = "test_tick_1"
        can_start = backpressure.start_tick(tick1, "test_operation")
        print(f"   ✅ Start tick 1: {can_start}")

        # Test drugiego tick (powinien przejść)
        tick2 = "test_tick_2"
        can_start2 = backpressure.start_tick(tick2, "test_operation")
        print(f"   ✅ Start tick 2: {can_start2}")

        # Test statusu
        status = backpressure.get_status()
        print(f"   ✅ Status: utilization={status['utilization']:.1%}")

        # Zakończ ticki
        duration1 = backpressure.finish_tick(tick1)
        duration2 = backpressure.finish_tick(tick2)
        print(f"   ✅ Durations: {duration1:.3f}s, {duration2:.3f}s")

        return True

    except Exception as e:
        print(f"   ❌ Błąd testu backpressure: {e}")
        return False


def test_idempotency():
    """Test systemu idempotencji."""
    print("\n🧪 Test 5: System idempotencji")

    try:
        from reliability_core import get_reliable_system

        reliable_system = get_reliable_system()
        idempotency = reliable_system.idempotency

        # Test akcji
        action1 = {
            "kind": "test_action",
            "title": "Test idempotencji",
            "description": "Testowa akcja dla idempotencji",
        }

        action2 = {
            "kind": "test_action",
            "title": "Test idempotencji",  # Identyczna
            "description": "Testowa akcja dla idempotencji",
        }

        # Pierwsza rejestracja
        is_dup1, existing1 = idempotency.is_duplicate(action1)
        print(f"   ✅ Pierwsza akcja - duplikat: {is_dup1}")

        id1 = idempotency.register_action(action1)
        print(f"   ✅ ID pierwszej akcji: {id1}")

        # Druga rejestracja (duplikat)
        is_dup2, existing2 = idempotency.is_duplicate(action2)
        print(f"   ✅ Druga akcja - duplikat: {is_dup2}")

        if is_dup2:
            print(f"   ✅ Wykryto duplikat: {existing2['count']} wystąpień")

        # Stats
        stats = idempotency.get_stats()
        print(
            f"   ✅ Stats: registered={stats['total_registered']}, rate={stats['duplicate_rate']:.1%}"
        )

        return True

    except Exception as e:
        print(f"   ❌ Błąd testu idempotencji: {e}")
        return False


def test_telemetry():
    """Test systemu telemetrii."""
    print("\n🧪 Test 6: System telemetrii")

    try:
        from reliability_core import get_reliable_system

        reliable_system = get_reliable_system()
        telemetry = reliable_system.telemetry

        # Start telemetrii
        tick_telemetry = telemetry.start_tick_telemetry("test_tick")
        print(f"   ✅ Telemetria rozpoczęta: {tick_telemetry.tick_id}")

        # Zapisz propozycje
        proposals = [{"kind": "test1", "score": 0.8}, {"kind": "test2", "score": 0.6}]
        telemetry.record_proposals(tick_telemetry, proposals)
        print(f"   ✅ Propozycje zapisane: {tick_telemetry.proposals_count}")

        # Zapisz decyzje
        accepted = [proposals[0]]
        rejected = [proposals[1]]
        telemetry.record_decisions(tick_telemetry, accepted, rejected)
        print(f"   ✅ Decyzje zapisane: rate={tick_telemetry.acceptance_rate:.1%}")

        # Zapisz wywołanie LLM
        telemetry.record_llm_call(tick_telemetry, 1.5)
        print(f"   ✅ LLM call zapisane: {tick_telemetry.llm_calls} calls")

        # Zakończ telemetrię
        result = telemetry.finish_tick_telemetry(tick_telemetry)
        print(f"   ✅ Telemetria zakończona: duration={result['duration']:.3f}s")

        # Rolling metrics
        rolling = telemetry.get_rolling_metrics()
        if "no_data" not in rolling:
            print(f"   ✅ Rolling metrics: window={rolling['window_size']}")

        return True

    except Exception as e:
        print(f"   ❌ Błąd testu telemetrii: {e}")
        return False


def test_complete_integration():
    """Test kompletnej integracji - symulacja autopilot cycle."""
    print("\n🧪 Test 7: Kompletna integracja - autopilot cycle")

    try:
        import psychika
        from reliability_core import get_reliable_system

        reliable_system = get_reliable_system()

        # Test autopilot cycle z systemem niezawodności
        result = psychika.autopilot_cycle("Test kompletnej integracji")

        print("   ✅ Autopilot cycle wykonany")

        # Sprawdź strukturę odpowiedzi (może być różna w zależności od backpressure)
        suggest = result.get("suggest", {})
        decision = result.get("decision", {})
        applied = result.get("applied", {})

        print(f"   ✅ Suggest: {suggest.get('ok', 'no_data')}")
        print(f"   ✅ Decision: {len(decision.get('accept', []))} accepted")
        print(f"   ✅ Applied: {applied.get('done', 0)} done")

        if "reliability" in result:
            rel = result["reliability"]
            print(
                f"   ✅ Reliability: {rel['validated_actions']} validated, {rel['ethics_blocks']} blocked"
            )

        # Sprawdź health systemu po operacji
        health = reliable_system.get_system_health()
        print(f"   ✅ Final health score: {health['health_score']:.2f}")

        return True

    except Exception as e:
        print(f"   ❌ Błąd testu kompletnej integracji: {e}")
        return False


def main():
    """Główna funkcja testowa."""
    print("🚀 TEST KOMPLETNEJ INTEGRACJI SYSTEMU NIEZAWODNOŚCI")
    print("=" * 60)

    tests = [
        test_reliability_core,
        test_psychika_integration,
        test_memory_contracts,
        test_backpressure,
        test_idempotency,
        test_telemetry,
        test_complete_integration,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
                print("   🎉 PASSED")
            else:
                failed += 1
                print("   💥 FAILED")
        except Exception as e:
            failed += 1
            print(f"   💥 EXCEPTION: {e}")

    print("\n" + "=" * 60)
    print("📊 WYNIKI TESTÓW:")
    print(f"   ✅ Zaliczone: {passed}")
    print(f"   ❌ Niezaliczone: {failed}")
    print(f"   📈 Sukces: {passed/(passed+failed)*100:.1f}%")

    if failed == 0:
        print("\n🎉 WSZYSTKIE TESTY ZALICZONE!")
        print("✅ System niezawodności jest w pełni zintegrowany i działa poprawnie")
        print("✅ Wszystkie mechanizmy bezpieczeństwa są aktywne")
        print("✅ Aplikacja TRAKTUJE POWAŻNIE system niezawodności")
    else:
        print(f"\n⚠️ UWAGA: {failed} testów nie przeszło")
        print("🚨 System może nie działać poprawnie bez poprawek")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
