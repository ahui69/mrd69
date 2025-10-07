#!/usr/bin/env bash
set -euo pipefail
API_BASE="${API_BASE:-http://127.0.0.1:8000}"
AUTH=""
[ -n "${LLM_API_KEY:-}" ] && AUTH="-H Authorization: Bearer ${LLM_API_KEY}"

post_json () {
  local path="$1"; shift
  curl -sS -X POST \
    -H 'Content-Type: application/json' \
    $AUTH \
    --data "$*" \
    "$API_BASE$path"
}

MESSAGES='[{"role":"system","content":"Jesteś pomocnym asystentem."}]'

ask () {
  local u="$1"
  MESSAGES=$(python3 - "$MESSAGES" "$u" <<'PY'
import sys, json
msgs=json.loads(sys.argv[1])
user=sys.argv[2]
msgs.append({"role":"user","content":user})
print(json.dumps(msgs, ensure_ascii=False))
PY
)
  BODY=$(python3 - "$MESSAGES" <<'PY'
import sys, json
msgs=json.loads(sys.argv[1])
print(json.dumps({"messages": msgs, "temperature":0.7, "max_tokens":200}, ensure_ascii=False))
PY
)
  R=$(post_json "/api/chat" "$BODY")
  echo "👤 $u"
  echo "🤖 $(python3 - <<PY
import sys, json
print(json.loads(sys.stdin.read()).get("reply",""))
PY
<<<"$R")"
  # dopisz assistant do kontekstu
  REPLY=$(python3 - <<PY
import sys, json
print(json.loads(sys.stdin.read()).get("reply",""))
PY
<<<"$R")
  MESSAGES=$(python3 - "$MESSAGES" "$REPLY" <<'PY'
import sys, json
msgs=json.loads(sys.argv[1])
rep=sys.argv[2]
msgs.append({"role":"assistant","content":rep})
print(json.dumps(msgs, ensure_ascii=False))
PY
)
}

ask "Elo, jak mogę szybko zwiedzić Kraków w 1 dzień?"
ask "Dorzuć plan wieczoru, budżet 150 zł."
ask "Brzmi spoko. A jutro rano gdzie na śniadanie?"
