#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "===== health ====="
curl -sS http://127.0.0.1:8000/health 2>/dev/null | python3 -m json.tool || echo "manager not responding"

echo
echo "===== ports ====="
ss -ltnp | grep -E ':8000|:18080|:18081|:18082' || echo "no SyMoNeuRaL ports listening"

echo
echo "===== processes ====="
ps aux | grep -E 'uvicorn|symon-llm-server|symon-image-server|llama-server' | grep -v grep || echo "no SyMoNeuRaL processes"

echo
echo "===== GPU ====="
nvidia-smi
