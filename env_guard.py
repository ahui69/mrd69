import os, json

need = {
    "LLM_BASE_URL": True,
    "LLM_API_KEY": True,
    "LLM_MODEL": True,
    "MINI_LLM_MODEL": False,
    "WEB_HTTP_TIMEOUT": False,
    "MEM_ROOT": False,
}
miss = [k for k, req in need.items() if req and not os.getenv(k)]
print(
    json.dumps(
        {"missing_required": miss, "ok": len(miss) == 0}, ensure_ascii=False, indent=2
    )
)
