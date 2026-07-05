from app.runtime.worker_types import WorkerResult
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
            "inference body has not been wired yet."
        ),
    )
