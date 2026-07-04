#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "===== health ====="
curl -sS http://127.0.0.1:8000/health | python3 -m json.tool

echo
echo "===== hey ====="
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"symoneural_moe","stream":false,"messages":[{"role":"user","content":"hey"}],"max_tokens":128}' \
  | python3 -m json.tool

echo
echo "===== code ====="
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"symoneural_moe","stream":false,"messages":[{"role":"user","content":"write a tiny bash script that prints SyMoNeuRaL is alive"}],"max_tokens":512}' \
  | python3 -m json.tool
