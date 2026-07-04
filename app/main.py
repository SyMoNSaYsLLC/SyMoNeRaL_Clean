import base64
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "data/runtime"
OUTPUT = ROOT / "data/output"
OUTPUT.mkdir(parents=True, exist_ok=True)
RUNTIME.mkdir(parents=True, exist_ok=True)

QWEN_PORT = 18080
GEMMA_PORT = 18081
IMAGE_PORT = 18082
QWEN_URL = f"http://127.0.0.1:{QWEN_PORT}"
GEMMA_URL = f"http://127.0.0.1:{GEMMA_PORT}"
IMAGE_URL = f"http://127.0.0.1:{IMAGE_PORT}"

app = FastAPI(title="SyMoNeuRaL MoE")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT)), name="outputs")
if (ROOT / "apps/web/static").exists():
    app.mount("/static", StaticFiles(directory=str(ROOT / "apps/web/static")), name="static")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str | None = "symoneural_moe"
    messages: list[ChatMessage]
    max_tokens: int | None = 768
    stream: bool | None = False

def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.4):
            return True
    except OSError:
        return False

def norm(s: str) -> str:
    return s.lower().replace("-", " ").replace("_", " ").replace("\n", " ")

def looks_image(prompt: str) -> bool:
    p = norm(prompt)
    return any(x in p for x in [
        "generate an image", "generate image", "make an image", "create an image",
        "draw", "render image", "render an image", "stable diffusion",
        "picture of", "image of", "photo of", "txt2img", "text to image",
    ])

def latest_user(messages: list[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""

async def post_json(url: str, payload: dict[str, Any], timeout: float = 180.0) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.json()

async def call_text_model(prompt: str, max_tokens: int, prefer_gemma: bool = True) -> tuple[str, str]:
    use_gemma = prefer_gemma and port_open(GEMMA_PORT)

    if use_gemma:
        base_url = GEMMA_URL
        model_name = "gemma-4-12b-it-Q4_K_M"
        system = (
            "You are SyMoNeuRaL Manager. Follow the user's instructions exactly. "
            "For coding requests, produce valid, complete, runnable code. "
            "Do not invent invalid syntax. Do not skip requested methods."
        )
        temperature = 0.25
    else:
        if not port_open(QWEN_PORT):
            raise HTTPException(503, "No resident text server is running")
        base_url = QWEN_URL
        model_name = "qwen2.5-coder-1.5b-instruct-q8_0"
        system = (
            "You are SyMoNeuRaL Coder. Follow the requested format exactly. "
            "Return valid, complete, runnable code."
        )
        temperature = 0.25

    payload = {
        "model": model_name,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        data = await post_json(f"{base_url}/v1/chat/completions", payload)
        return model_name, data["choices"][0]["message"]["content"]
    except Exception:
        data = await post_json(
            f"{base_url}/completion",
            {
                "prompt": f"{system}\n\nUser:\n{prompt}\n\nAssistant:\n",
                "n_predict": max_tokens,
                "temperature": temperature,
            },
        )
        return model_name, data.get("content", json.dumps(data))

def start_image_server_if_needed() -> None:
    if port_open(IMAGE_PORT):
        return

    image_bin = ROOT / "bin/symon-image-server"
    model = ROOT / "models/sd/v1-5-pruned-emaonly.safetensors"

    if not image_bin.exists():
        raise RuntimeError(f"missing {image_bin}")
    if not model.exists():
        raise RuntimeError(f"missing {model}")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(ROOT / "bin") + ":" + env.get("LD_LIBRARY_PATH", "")

    log = open(RUNTIME / "symon-image-server.log", "ab", buffering=0)
    cmd = [
        str(image_bin),
        "--listen-ip", "127.0.0.1",
        "--listen-port", str(IMAGE_PORT),
        "-m", str(model),
        "--max-vram", "14",
        "--mmap",
        "--vae-tiling",
        "--fa",
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    (RUNTIME / "symon-image-server.pid").write_text(str(proc.pid))

    deadline = time.time() + 90
    while time.time() < deadline:
        if port_open(IMAGE_PORT):
            return
        if proc.poll() is not None:
            raise RuntimeError("image server exited early; see data/runtime/symon-image-server.log")
        time.sleep(1)

    raise RuntimeError("image server did not start")

async def generate_image(prompt: str) -> str:
    start_image_server_if_needed()

    data = await post_json(
        f"{IMAGE_URL}/sdapi/v1/txt2img",
        {
            "prompt": prompt,
            "negative_prompt": "low quality, blurry, distorted, malformed, ugly, watermark, text",
            "width": 512,
            "height": 512,
            "steps": 20,
            "cfg_scale": 7.0,
            "seed": -1,
            "batch_size": 1,
            "n_iter": 1,
            "sampler_name": "Euler a",
        },
        timeout=420.0,
    )

    images = data.get("images") or []
    if not images:
        raise RuntimeError(f"image server returned no images: {data}")

    b64 = images[0].split(",", 1)[-1]
    raw = base64.b64decode(b64)
    name = f"symon_image_{int(time.time())}.png"
    path = OUTPUT / name
    path.write_bytes(raw)

    return f"Generated image with SyMoNeuRaL image worker.\n\n![Generated image](/outputs/{name})\n\nFile: `data/output/{name}`"

@app.get("/")
async def root():
    return FileResponse(ROOT / "apps/web/index.html")

@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": "SyMoNeuRaL MoE",
        "server": "127.0.0.1:8000",
        "qwen": port_open(QWEN_PORT),
        "gemma": port_open(GEMMA_PORT),
        "image": port_open(IMAGE_PORT),
    }

@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {"id": "symoneural_moe", "object": "model", "role": "router"},
            {"id": "qwen2.5-coder-1.5b-instruct-q8_0", "object": "model", "role": "resident_fast_fallback"},
            {"id": "gemma-4-12b-it-Q4_K_M", "object": "model", "role": "resident_manager_coder"},
            {"id": "v1-5-pruned-emaonly", "object": "model", "role": "image_generator"},
        ],
    }

@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    prompt = latest_user(req.messages).strip()
    if not prompt:
        raise HTTPException(400, "missing user prompt")

    requested_model = (req.model or "symoneural_moe").lower()

    force_image = (
        "image" in requested_model
        or "stable" in requested_model
        or "diffusion" in requested_model
        or "v1-5" in requested_model
        or requested_model == "sd"
    )

    if force_image or looks_image(prompt):
        content = await generate_image(prompt)
        model = "v1-5-pruned-emaonly"
    else:
        if "qwen" in requested_model or "coder" in requested_model:
            model, content = await call_text_model(prompt, req.max_tokens or 1024, prefer_gemma=False)
        elif "gemma" in requested_model:
            model, content = await call_text_model(prompt, req.max_tokens or 1024, prefer_gemma=True)
        else:
            model, content = await call_text_model(prompt, req.max_tokens or 1024, prefer_gemma=True)

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
    result = await chat(ChatRequest(messages=[ChatMessage(role="user", content=prompt)]))
    return {"content": result["choices"][0]["message"]["content"], "model": result["model"], "stop": True}

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
