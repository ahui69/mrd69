import importlib, inspect, sys
m = importlib.import_module("seed_memory")
path = sys.argv[1] if len(sys.argv) > 1 else r"data\sq3\seed.jsonl"

def try_call(fn):
    try:
        fn(path); return True
    except TypeError:
        try:
            fn(); return True
        except Exception:
            return False
    except Exception:
        return False

# 1) kandydaci heurystyczni (nazwa zawiera seed/json/jsonl/load/ingest/import)
candidates = []
for name, fn in inspect.getmembers(m, inspect.isfunction):
    if fn.__module__ != m.__name__: 
        continue
    n = name.lower()
    if any(k in n for k in ("seed","json","jsonl","load","ingest","import")):
        candidates.append((name, fn))

# 2) priorytety znanych nazw
priority = ("seed_from_jsonl","seed_from_json","seed","load_seed","load","ingest","main","run","import_seed")
for pname in priority:
    if hasattr(m, pname):
        print(f"[seed_runner] trying: {pname}('{path}')")
        if try_call(getattr(m, pname)):
            print("[seed_runner] OK"); sys.exit(0)

# 3) przeleć kandydatów
for name, fn in candidates:
    print(f"[seed_runner] trying: {name}('{path}')")
    if try_call(fn):
        print("[seed_runner] OK"); sys.exit(0)

print("[seed_runner] Nie znalazłem działającej funkcji do zasiania.")
sys.exit(2)
