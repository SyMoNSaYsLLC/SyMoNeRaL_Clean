from pathlib import Path

ROOT = Path.cwd()
for d in ["app/workers", "app/comms", "app/runtime"]:
    Path(d).mkdir(parents=True, exist_ok=True)
    Path(d, "__init__.py").touch()

Path("app/runtime/worker_types.py").write_text(r'''from dataclasses import dataclass

@dataclass
class WorkerResult:
    worker: str
    content: str
    ok: bool = True
    error: str | None = None
''')

Path("app/workers/vram_guard.py").write_text(r'''import subprocess

VRAM_TARGET_GB = 14.0
VRAM_HARD_LIMIT_GB = 15.2

def gpu_vram_gb() -> tuple[float, float]:
    out = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ], text=True).strip().splitlines()[0]
    used_mb, total_mb = [float(x.strip()) for x in out.split(",")]
    return used_mb / 1024.0, total_mb / 1024.0

def status() -> dict:
    used, total = gpu_vram_gb()
    return {
        "used_gb": round(used, 2),
        "total_gb": round(total, 2),
        "target_gb": VRAM_TARGET_GB,
        "hard_limit_gb": VRAM_HARD_LIMIT_GB,
        "ok": used < VRAM_TARGET_GB,
        "critical": used >= VRAM_HARD_LIMIT_GB,
    }
''')

Path("app/workers/gemma.py").write_text(r'''from app.runtime.worker_types import WorkerResult
from app.workers import vram_guard

NAME = "gemma"

def ready() -> bool:
    return True

def decide(prompt: str) -> dict:
    p = prompt.lower()
    if any(x in p for x in ["draw", "image", "picture", "photo", "render", "safetensor", "stable"]):
        return {"worker": "safetensor", "task": prompt, "check": "image_request"}
    if any(x in p for x in ["code", "script", "python", "bash", "rust", "cpp", "c++", "debug", "repo", "bitbake", ".bb"]):
        return {"worker": "qwen", "task": prompt, "check": "code_request"}
    return {"worker": "gemma", "task": prompt, "check": "writer_request"}

def write(prompt: str, max_tokens: int = 768) -> WorkerResult:
    _ = vram_guard.status()
    return WorkerResult(
        worker=NAME,
        content=(
            "Gemma native worker module is selected, but the repo-owned Gemma inference body "
            "has not been wired yet. No llama.cpp or worker HTTP fallback was used."
        ),
    )

def final_check(original: str, draft: str) -> WorkerResult:
    return WorkerResult(worker=NAME, content=draft)
''')

Path("app/workers/qwen.py").write_text(r'''from app.runtime.worker_types import WorkerResult
from app.workers import vram_guard

NAME = "qwen"

def ready() -> bool:
    return True

def code(task: str, max_tokens: int = 1024) -> WorkerResult:
    _ = vram_guard.status()
    return WorkerResult(
        worker=NAME,
        content=(
            "Qwen native coder worker module is selected, but the repo-owned Qwen inference body "
            "has not been wired yet. No llama.cpp or worker HTTP fallback was used."
        ),
    )
''')

Path("app/workers/safetensor.py").write_text(r'''from app.runtime.worker_types import WorkerResult
from app.workers import vram_guard

NAME = "safetensor"

def ready() -> bool:
    return True

def image(prompt: str) -> WorkerResult:
    _ = vram_guard.status()
    return WorkerResult(
        worker=NAME,
        content=(
            "Safetensor native image worker module is selected, but the repo-owned safetensor "
            "inference body has not been wired yet. No llama.cpp or worker HTTP fallback was used."
        ),
    )
''')

Path("app/comms/gemmacodercomms.py").write_text(r'''from app.workers import gemma, qwen

def run(original_prompt: str, manager_task: str, max_tokens: int):
    draft = qwen.code(manager_task, max_tokens=max_tokens)
    checked = gemma.final_check(original_prompt, draft.content)
    return checked
''')

Path("app/comms/gemmaimagingcomms.py").write_text(r'''from app.workers import gemma, safetensor

def run(original_prompt: str, manager_task: str):
    result = safetensor.image(manager_task or original_prompt)
    checked = gemma.final_check(original_prompt, result.content)
    return checked
''')

Path("app/symoneural.py").write_text(r'''from app.workers import gemma, qwen, safetensor
from app.comms import gemmacodercomms, gemmaimagingcomms

def answer(prompt: str, max_tokens: int = 768) -> tuple[str, str]:
    decision = gemma.decide(prompt)
    worker = decision["worker"]
    task = decision["task"]

    if worker == "qwen":
        result = gemmacodercomms.run(prompt, task, max_tokens)
    elif worker == "safetensor":
        result = gemmaimagingcomms.run(prompt, task)
    else:
        result = gemma.write(task, max_tokens)

    return result.worker, result.content
''')

Path("app/main.py").write_text(r'''from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import symoneural
from app.workers import gemma, qwen, safetensor, vram_guard

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/output"
OUTPUT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SyMoNeuRaL MoE")

if (ROOT / "apps/web/static").exists():
    app.mount("/static", StaticFiles(directory=str(ROOT / "apps/web/static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT)), name="outputs")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str | None = "symoneural_moe"
    messages: list[ChatMessage]
    max_tokens: int | None = 768
    stream: bool | None = False

def latest_user(messages: list[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content.strip()
    return ""

@app.get("/")
async def root():
    return FileResponse(ROOT / "apps/web/index.html")

@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": "symoneural_moe",
        "vram": vram_guard.status(),
        "workers": {
            "gemma": gemma.ready(),
            "qwen": qwen.ready(),
            "safetensor": safetensor.ready(),
        },
    }

@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "symoneural_moe", "object": "model"}]}

@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    prompt = latest_user(req.messages)
    if not prompt:
        raise HTTPException(400, "missing user prompt")

    model, content = symoneural.answer(prompt, req.max_tokens or 768)

    return {
        "id": "symon-chatcmpl-local",
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "requested_max_tokens": req.max_tokens,
        },
    }

@app.post("/completion")
async def completion(payload: dict[str, Any]):
    prompt = str(payload.get("prompt") or payload.get("content") or "").strip()
    model, content = symoneural.answer(prompt, 768)
    return {"content": content, "model": model, "stop": True}

@app.get("/props")
async def props():
    return {"model_path": "symoneural_moe", "modalities": ["text", "image"], "total_slots": 1}

@app.get("/slots")
async def slots():
    return [{"id": 0, "state": 0}]

@app.get("/tools")
async def tools():
    return {"tools": [], "tool_choice": "none"}

@app.post("/v1/streams/lookup")
async def streams_lookup(payload: dict[str, Any]):
    return {"id": payload.get("id"), "found": False, "stream": None}

@app.get("/metrics")
async def metrics():
    return PlainTextResponse("symoneural_manager_up 1\n")
''')

Path("scripts/start.sh").write_text(r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p data/runtime data/output

if [ ! -x "$ROOT/.venv/bin/python" ]; then
  echo "ERROR: missing .venv python"
  exit 1
fi

source "$ROOT/.venv/bin/activate"

PID="$(cat data/runtime/symon-manager.pid 2>/dev/null || true)"
if [ -n "${PID:-}" ] && kill -0 "$PID" 2>/dev/null; then
  kill "$PID" 2>/dev/null || true
fi

nohup "$ROOT/.venv/bin/python" -m uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  > data/runtime/symon-manager.log 2>&1 &

echo "$!" > data/runtime/symon-manager.pid
echo "SyMoNeuRaL public manager started on port 8000"
''')
Path("scripts/start.sh").chmod(0o755)

Path("scripts/stop.sh").write_text(r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PID="$(cat data/runtime/symon-manager.pid 2>/dev/null || true)"
if [ -n "${PID:-}" ] && kill -0 "$PID" 2>/dev/null; then
  kill "$PID" 2>/dev/null || true
fi
rm -f data/runtime/symon-manager.pid
echo "Stopped SyMoNeuRaL public manager."
''')
Path("scripts/stop.sh").chmod(0o755)

print("Applied pure repo-owned name-based worker architecture. No llama.cpp worker code.")
