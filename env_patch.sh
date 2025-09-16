#!/usr/bin/env bash
# env_patch.sh — dopisz brakujące ENV-y dla memory.py + psychika.py (DeepInfra + Runpod)

set -euo pipefail

TARGET_FILE="\$HOME/.bashrc"
PRINT_ONLY=0

while [[ \$# -gt 0 ]]; do
  case "\$1" in
    --file) TARGET_FILE="\$2"; shift 2 ;;
    --print) PRINT_ONLY=1; shift ;;
    *) echo "Nieznana flaga: \$1" >&2; exit 1 ;;
  esac
done

mkdir -p "\$(dirname "\$TARGET_FILE")"
touch "\$TARGET_FILE"

backup_file="\${TARGET_FILE}.bak.\$(date +%Y%m%d-%H%M%S)"
cp "\$TARGET_FILE" "\$backup_file"

RUNPOD_DEFAULT=""
if [[ -d "/workspace/volume" ]]; then
  RUNPOD_DEFAULT="/workspace/volume"
elif [[ -d "/runpod-volume" ]]; then
  RUNPOD_DEFAULT="/runpod-volume"
else
  RUNPOD_DEFAULT="/workspace/mrd69"
fi

ensure_export() {
  local var="\$1"
  local val="\$2"
  local file="\$3"
  if [[ -n "\${!var-}" ]]; then
    return 0
  fi
  if grep -Eq "^[[:space:]]*export[[:space:]]+\$var=" "\$file"; then
    return 0
  fi
  local esc
  esc=\$(printf "%q" "\$val")
  if [[ "\$PRINT_ONLY" -eq 1 ]]; then
    printf 'export %s=%s\n' "\$var" "\$esc"
  else
    printf 'export %s=%s\n' "\$var" "\$esc" >> "\$file"
  fi
}

ensure_export "LLM_BASE_URL" "https://api.deepinfra.com/v1/openai" "\$TARGET_FILE"
ensure_export "LLM_API_KEY"  "DI_your_secret_key"                 "\$TARGET_FILE"
ensure_export "LLM_MODEL"    "Qwen/Qwen2.5-72B-Instruct"          "\$TARGET_FILE"

ensure_export "MINI_LLM_BASE_URL" "\${LLM_BASE_URL}"               "\$TARGET_FILE"
ensure_export "MINI_LLM_API_KEY"  "\${LLM_API_KEY}"                "\$TARGET_FILE"
ensure_export "MINI_LLM_MODEL"    "Qwen/Qwen2.5-4B-Instruct"       "\$TARGET_FILE"

# RunPod Integration
ensure_export "RUNPOD_PERSIST_DIR" "\$RUNPOD_DEFAULT"             "\$TARGET_FILE"
ensure_export "USE_RUNPOD"        "True"                          "\$TARGET_FILE"
ensure_export "RUNPOD_API_KEY"    "your_runpod_api_key"           "\$TARGET_FILE"
ensure_export "RUNPOD_ENDPOINT_ID" "your_endpoint_id"             "\$TARGET_FILE"

# Memory Configuration
ensure_export "MEM_NS"             "runpod"                       "\$TARGET_FILE"
ensure_export "PSY_ENCRYPT_KEY"    "zmien_to_na_tajny_klucz"       "\$TARGET_FILE"
ensure_export "LTM_MIN_CONF"       "0.25"                          "\$TARGET_FILE"
ensure_export "MAX_LTM_FACTS"      "2000000"                       "\$TARGET_FILE"
ensure_export "RECALL_TOPK_PER_SRC" "100"                          "\$TARGET_FILE"
ensure_export "STM_MAX_TURNS"      "400"                           "\$TARGET_FILE"
ensure_export "STM_KEEP_TAIL"      "100"                           "\$TARGET_FILE"

# Embeddings Configuration 
ensure_export "LLM_EMBED_URL"      "https://api.openai.com/v1/embeddings" "\$TARGET_FILE"
ensure_export "LLM_EMBED_MODEL"    "text-embedding-3-large"         "\$TARGET_FILE"
ensure_export "OPENAI_API_KEY"     "your_openai_api_key"           "\$TARGET_FILE"

ensure_export "LLM_HTTP_TIMEOUT_S" "60"                            "\$TARGET_FILE"
ensure_export "DB_DURABLE"         "0"                             "\$TARGET_FILE"

ensure_export "PSY_PRESET"        "balanced"                      "\$TARGET_FILE"
ensure_export "PSY_AUTOPILOT"     "1"                             "\$TARGET_FILE"
ensure_export "PSY_AUTOPILOT_INTERVAL"    "120"                   "\$TARGET_FILE"
ensure_export "PSY_AUTOPILOT_HARD_LIMIT"  "45"                    "\$TARGET_FILE"
ensure_export "PSY_DEBUG"         "0"                             "\$TARGET_FILE"

if [[ "\$PRINT_ONLY" -eq 1 ]]; then
  echo
  echo "# (podgląd) Zmienne zostały WYDRUKOWANE — plik nie został zmieniony."
  echo "# Docelowy plik: \$TARGET_FILE"
  echo "# Backup: \$backup_file"
else
  echo "# Dopisano brakujące zmienne do: \$TARGET_FILE"
  echo "# Backup wcześniejszego pliku:  \$backup_file"
  echo "# Załaduj zmiany w tej sesji:"
  echo "  source \"\$TARGET_FILE\""
fi
