from pathlib import Path
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
