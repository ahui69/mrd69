#!/usr/bin/env python3
import importlib, sys
mod = sys.argv[1] if len(sys.argv)>1 else "server"
m = importlib.import_module(mod)
app = getattr(m, "app", None)
if app is None:
    print("BRAK: w module", mod, "nie znaleziono 'app'")
    raise SystemExit(1)
for r in app.routes:
    methods = ",".join(sorted(getattr(r, "methods", []) or []))
    path = getattr(r, "path", getattr(r, "path_format", ""))
    name = getattr(r, "name", "")
    print(f"{methods:10s} {path}  {name}")
