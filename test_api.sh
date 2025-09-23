#!/usr/bin/env bash
set -euo pipefail

/bin/echo "==> Testuję API pod http://127.0.0.1:8000"

/usr/bin/curl -s -X GET http://127.0.0.1:8000/api/health
/bin/echo
/usr/bin/curl -s -X GET http://127.0.0.1:8000/api/memory/health
/bin/echo
/usr/bin/curl -s -X GET "http://127.0.0.1:8000/api/memory/facts?limit=3"
/bin/echo
/usr/bin/curl -s -X POST http://127.0.0.1:8000/api/memory/recall \
     -H "Content-Type: application/json" \
     -d '{"q":"Fakt","topk":5}'
/bin/echo
/usr/bin/curl -s -X POST http://127.0.0.1:8000/api/memory/context \
     -H "Content-Type: application/json" \
     -d '{"q":"Kontekst","topk":3,"limit":500}'
/bin/echo
/usr/bin/curl -s -X POST http://127.0.0.1:8000/api/memory/facts \
     -H "Content-Type: application/json" \
     -d '{"text":"Test fact","conf":0.9,"tags":["api"]}'
/bin/echo
/usr/bin/curl -s -X POST http://127.0.0.1:8000/api/chat \
     -H "Content-Type: application/json" \
     -d '{"chatId":null,"newMessage":{"role":"user","content":"elo"},"messages":[]}'
/bin/echo
/usr/bin/curl -s -X POST http://127.0.0.1:8000/api/memory/reload-seed \
     -H "Content-Type: application/json" \
     -d '{"path":"/workspace/mrd69/data/seed.json"}'
/bin/echo
