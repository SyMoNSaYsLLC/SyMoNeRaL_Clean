from pathlib import Path

# remove stale verifier/test scripts that still curl/check old workers
for name in [
    "scripts/status.sh",
    "scripts/test.sh",
    "verify_symoneural_updates.py",
    "verify_venv_deps.py",
    "update_vram_guard.py",
]:
    p = Path(name)
    if p.exists():
        p.unlink()

# remove leftover words from worker placeholder messages
for name in ["app/workers/gemma.py", "app/workers/qwen.py", "app/workers/safetensor.py"]:
    p = Path(name)
    s = p.read_text()
    s = s.replace(" No llama.cpp or worker HTTP fallback was used.", "")
    p.write_text(s)

# add clean status script for the only public service
Path("scripts/status.sh").write_text(r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "===== symoneural manager ====="
PID="$(cat data/runtime/symon-manager.pid 2>/dev/null || true)"
if [ -n "${PID:-}" ] && kill -0 "$PID" 2>/dev/null; then
  echo "running pid=$PID"
else
  echo "not running"
fi

echo
echo "===== public port ====="
ss -ltnp | grep ':8000' || echo "8000 not listening"

echo
echo "===== compile ====="
"$ROOT/.venv/bin/python" -m py_compile app/main.py app/symoneural.py app/workers/*.py app/comms/*.py app/runtime/*.py
''')
Path("scripts/status.sh").chmod(0o755)

print("Cleaned stale URL/llama leftovers.")
