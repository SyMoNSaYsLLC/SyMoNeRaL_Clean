#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p data/runtime data/output

LLM_SERVER_BIN="$(cat data/runtime/llm_server_bin.txt 2>/dev/null || true)"
if [ -z "$LLM_SERVER_BIN" ] || [ ! -x "$LLM_SERVER_BIN" ]; then
  LLM_SERVER_BIN="$(find "$ROOT/bin" "$HOME/SyMoNeuRaL_MCP/bin" -maxdepth 1 -type f \( -name 'llama-server' -o -name 'symon-llm-server' -o -name '*llm*server*' \) | grep -v image | head -n1 || true)"
fi

if [ -z "$LLM_SERVER_BIN" ]; then
  echo "ERROR: no llama-server/symon-llm-server binary found."
  exit 1
fi

QWEN_MODEL="$ROOT/models/llm/qwen2.5-coder-1.5b-instruct-q8_0.gguf"
GEMMA_MODEL="$ROOT/models/llm/gemma-4-12b-it-Q4_K_M.gguf"

echo "Using LLM server: $LLM_SERVER_BIN"

echo "Starting resident Qwen on 127.0.0.1:18080..."
nohup env LD_LIBRARY_PATH="$ROOT/bin:$HOME/SyMoNeuRaL_MCP/bin" CUDA_VISIBLE_DEVICES=0 \
  "$LLM_SERVER_BIN" \
  -m "$QWEN_MODEL" \
  --host 127.0.0.1 \
  --port 18080 \
  -ngl 999 \
  -c 4096 \
  > data/runtime/qwen-server.log 2>&1 &
echo "$!" > data/runtime/qwen-server.pid

echo "Starting resident Gemma on 127.0.0.1:18081..."
nohup env LD_LIBRARY_PATH="$ROOT/bin:$HOME/SyMoNeuRaL_MCP/bin" CUDA_VISIBLE_DEVICES=0 \
  "$LLM_SERVER_BIN" \
  -m "$GEMMA_MODEL" \
  --host 127.0.0.1 \
  --port 18081 \
  -ngl 999 \
  -c 8192 \
  > data/runtime/gemma-server.log 2>&1 &
echo "$!" > data/runtime/gemma-server.pid

echo "Waiting for Qwen and Gemma..."
for i in $(seq 1 120); do
  QWEN_OK=0
  GEMMA_OK=0

  curl -fsS http://127.0.0.1:18080/v1/models >/dev/null 2>&1 && QWEN_OK=1
  curl -fsS http://127.0.0.1:18081/v1/models >/dev/null 2>&1 && GEMMA_OK=1

  if [ "$QWEN_OK" = "1" ] && [ "$GEMMA_OK" = "1" ]; then
    echo "Qwen and Gemma are up."
    break
  fi

  if ! kill -0 "$(cat data/runtime/qwen-server.pid)" 2>/dev/null; then
    echo "Qwen exited early:"
    tail -120 data/runtime/qwen-server.log || true
    exit 1
  fi

  if ! kill -0 "$(cat data/runtime/gemma-server.pid)" 2>/dev/null; then
    echo "Gemma exited early:"
    tail -120 data/runtime/gemma-server.log || true
    exit 1
  fi

  sleep 1
done

echo "Starting clean SyMoNeuRaL UI/API on 127.0.0.1:8000..."
nohup "$ROOT/.venv/bin/python" -m uvicorn app.main:app \
  --host 127.0.0.1 \
  --port 8000 \
  > data/runtime/symon-manager.log 2>&1 &
echo "$!" > data/runtime/symon-manager.pid

for i in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8000/health | grep -q '"qwen"'; then
    echo "Clean SyMoNeuRaL is up."
    curl -fsS http://127.0.0.1:8000/health
    echo
    echo "Open: http://127.0.0.1:8000/?v=clean"
    exit 0
  fi
  sleep 1
done

echo "Clean SyMoNeuRaL API did not start:"
tail -160 data/runtime/symon-manager.log || true
exit 1
