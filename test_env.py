#!/usr/bin/env python3
"""Test wczytywania zmiennych środowiskowych"""

import os

from dotenv import load_dotenv

# Wczytaj .env
load_dotenv()

# Sprawdź kluczowe zmienne
vars_to_check = [
    "LLM_API_KEY",
    "SERPAPI_KEY",
    "OPENAI_API_KEY",
    "STABILITY_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
]

print("=== Test zmiennych środowiskowych ===")
for var in vars_to_check:
    value = os.getenv(var, "NOT_FOUND")
    if value and value != "NOT_FOUND":
        print(f"✅ {var}: {value[:20]}...")
    else:
        print(f"❌ {var}: NOT_FOUND")

print("\n=== Test importu konfiguracji ===")
try:
    import config

    print("✅ config.py imported successfully")

    # Test dostępu do zmiennych przez config
    print(f"✅ HOST: {config.HOST}")
    print(f"✅ PORT: {config.PORT}")

except Exception as e:
    print(f"❌ Error importing config: {e}")

print("\n=== Test importu modułów z API ===")
modules_to_test = ["search_client", "autonauka", "kimi_client"]

for module_name in modules_to_test:
    try:
        module = __import__(module_name)
        print(f"✅ {module_name} imported successfully")
    except Exception as e:
        print(f"❌ {module_name} failed: {e}")
