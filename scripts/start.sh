#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p data/runtime data/output

usage() {
  cat <<USAGE
SyMoNeuRaL start options:

  ./scripts/start.sh
  ./scripts/start.sh symoneural_moe
      Start manager + Qwen + Gemma + Stable Diffusion.

  ./scripts/start.sh qwen
      Start manager + Qwen only.

  ./scripts/start.sh gemma
      Start manager + Gemma only.

  ./scripts/start.sh stable
  ./scripts/start.sh image
      Start manager + Stable Diffusion image server only.

  ./scripts/start.sh all
      Start manager + Qwen + Gemma + Stable Diffusion.

  ./scripts/start.sh qwen gemma stable
      Start exactly the listed engines plus manager.

Ports:
  8000  SyMoNeuRaL frontend/API
  18080 Qwen resident server
  18081 Gemma resident server
  18082 Stable Diffusion image server
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

START_MANAGER=1
START_QWEN=0
START_GEMMA=0
START_STABLE=0

if [ "$#" -eq 0 ]; then
  set -- symoneural_moe
fi

for arg in "$@"; do
  case "$arg" in
    symoneural_moe|moe)
      START_QWEN=1
      START_GEMMA=1
      START_STABLE=1
      ;;
    qwen|coder)
      START_QWEN=1
      ;;
    gemma|manager|writer)
      START_GEMMA=1
      ;;
    stable|image|sd|v1-5|v1-5-pruned-emaonly)
      START_STABLE=1
      ;;
    all)
      START_QWEN=1
      START_GEMMA=1
      START_STABLE=1
      ;;
    api|frontend|ui)
      START_MANAGER=1
      ;;
    *)
      echo "Unknown option: $arg"
      usage
      exit 1
      ;;
  esac
done

LLM_SERVER_BIN="$(cat data/runtime/llm_server_bin.txt 2>/dev/null || true)"
if [ -z "$LLM_SERVER_BIN" ] || [ ! -x "$LLM_SERVER_BIN" ]; then
  LLM_SERVER_BIN="$(find "$ROOT/bin" "$HOME/SyMoNeuRaL_MCP/bin" -maxdepth 1 -type f \( -name 'llama-server' -o -name 'symon-llm-server' -o -name '*llm*server*' \) | grep -v image | head -n1 || true)"
fi

QWEN_MODEL="$ROOT/models/llm/qwen2.5-coder-1.5b-instruct-q8_0.gguf"
GEMMA_MODEL="$ROOT/models/llm/gemma-4-12b-it-Q4_K_M.gguf"
STABLE_MODEL="$ROOT/models/sd/v1-5-pruned-emaonly.safetensors"
IMAGE_SERVER_BIN="$ROOT/bin/symon-image-server"

start_llm() {
  local name="$1"
  local port="$2"
  local model="$3"
  local ctx="$4"
  local pid_file="data/runtime/${name}-server.pid"
  local log_file="data/runtime/${name}-server.log"

  if ss -ltnp | grep -q ":${port} "; then
    echo "$name already listening on 127.0.0.1:${port}"
    return 0
  fi

  if [ -z "$LLM_SERVER_BIN" ]; then
    echo "ERROR: no llama-server/symon-llm-server binary found."
    exit 1
  fi

  if [ ! -f "$model" ]; then
    echo "ERROR: missing model: $model"
    exit 1
  fi

  echo "Starting resident $name on 127.0.0.1:${port}..."
  nohup env LD_LIBRARY_PATH="$ROOT/bin:$HOME/SyMoNeuRaL_MCP/bin" CUDA_VISIBLE_DEVICES=0 \
    "$LLM_SERVER_BIN" \
    -m "$model" \
    --host 127.0.0.1 \
    --port "$port" \
    -ngl 999 \
    -c "$ctx" \
    > "$log_file" 2>&1 &

  echo "$!" > "$pid_file"

  for i in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 || curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "$name is up."
      return 0
    fi

    if ! kill -0 "$(cat "$pid_file")" 2>/dev/null; then
      echo "$name exited early:"
      tail -120 "$log_file" || true
      exit 1
    fi

    sleep 1
  done

  echo "$name did not become ready:"
  tail -120 "$log_file" || true
  exit 1
}

start_stable() {
  local pid_file="data/runtime/symon-image-server.pid"
  local log_file="data/runtime/symon-image-server.log"

  if ss -ltnp | grep -q ':18082 '; then
    echo "Stable Diffusion already listening on 127.0.0.1:18082"
    return 0
  fi

  if [ ! -x "$IMAGE_SERVER_BIN" ]; then
    echo "ERROR: missing image server: $IMAGE_SERVER_BIN"
    exit 1
  fi

  if [ ! -f "$STABLE_MODEL" ]; then
    echo "ERROR: missing Stable model: $STABLE_MODEL"
    exit 1
  fi

  echo "Starting Stable Diffusion on 127.0.0.1:18082..."
  nohup env LD_LIBRARY_PATH="$ROOT/bin:$HOME/SyMoNeuRaL_MCP/bin" CUDA_VISIBLE_DEVICES=0 \
    "$IMAGE_SERVER_BIN" \
    --listen-ip 127.0.0.1 \
    --listen-port 18082 \
    -m "$STABLE_MODEL" \
    --max-vram 14 \
    --mmap \
    --vae-tiling \
    --fa \
    > "$log_file" 2>&1 &

  echo "$!" > "$pid_file"

  for i in $(seq 1 120); do
    if ss -ltnp | grep -q ':18082 '; then
      echo "Stable Diffusion is up."
      return 0
    fi

    if ! kill -0 "$(cat "$pid_file")" 2>/dev/null; then
      echo "Stable Diffusion exited early:"
      tail -120 "$log_file" || true
      exit 1
    fi

    sleep 1
  done

  echo "Stable Diffusion did not become ready:"
  tail -120 "$log_file" || true
  exit 1
}

start_manager() {
  local pid_file="data/runtime/symon-manager.pid"
  local log_file="data/runtime/symon-manager.log"

  if ss -ltnp | grep -q ':8000 '; then
    echo "Manager already listening on 127.0.0.1:8000"
    return 0
  fi

  echo "Starting SyMoNeuRaL UI/API on 127.0.0.1:8000..."
  nohup "$ROOT/.venv/bin/python" -m uvicorn app.main:app \
    --host 127.0.0.1 \
    --port 8000 \
    > "$log_file" 2>&1 &

  echo "$!" > "$pid_file"

  for i in $(seq 1 30); do
    if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
      echo "SyMoNeuRaL manager is up."
      curl -fsS http://127.0.0.1:8000/health
      echo
      return 0
    fi

    if ! kill -0 "$(cat "$pid_file")" 2>/dev/null; then
      echo "Manager exited early:"
      tail -120 "$log_file" || true
      exit 1
    fi

    sleep 1
  done

  echo "Manager did not become ready:"
  tail -120 "$log_file" || true
  exit 1
}

echo "===== SyMoNeuRaL start plan ====="
echo "Qwen:   $START_QWEN"
echo "Gemma:  $START_GEMMA"
echo "Stable: $START_STABLE"
echo "API/UI: $START_MANAGER"
echo

if [ "$START_QWEN" = "1" ]; then
  start_llm qwen 18080 "$QWEN_MODEL" 4096
fi

if [ "$START_GEMMA" = "1" ]; then
  start_llm gemma 18081 "$GEMMA_MODEL" 8192
fi

if [ "$START_STABLE" = "1" ]; then
  start_stable
fi

if [ "$START_MANAGER" = "1" ]; then
  start_manager
fi

echo
echo "===== status ====="
ss -ltnp | grep -E ':8000|:18080|:18081|:18082' || true
echo
curl -sS http://127.0.0.1:8000/health 2>/dev/null || true
echo
echo "Open: http://127.0.0.1:8000/?v=start-options"
