#!/usr/bin/env bash
set -euo pipefail
cd /workspace/mrd69
python3 -m venv .venv || true
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
