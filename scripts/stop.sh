#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for f in data/runtime/symon-manager.pid data/runtime/qwen-server.pid data/runtime/gemma-server.pid data/runtime/symon-image-server.pid; do
  if [ -f "$f" ]; then
    PID="$(cat "$f" 2>/dev/null || true)"
    if [ -n "${PID:-}" ] && kill -0 "$PID" 2>/dev/null; then
      echo "Stopping $f PID $PID"
      kill "$PID" 2>/dev/null || true
    fi
    rm -f "$f"
  fi
done

pkill -f "$ROOT/.venv/bin/python -m uvicorn app.main:app" 2>/dev/null || true
pkill -f "$ROOT/bin/.*llama-server" 2>/dev/null || true
pkill -f "$ROOT/bin/.*llm.*server" 2>/dev/null || true
pkill -f "$ROOT/bin/symon-image-server" 2>/dev/null || true
