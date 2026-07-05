from app.runtime.worker_types import WorkerResult
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
            "has not been wired yet."
        ),
    )

def final_check(original: str, draft: str) -> WorkerResult:
    return WorkerResult(worker=NAME, content=draft)
