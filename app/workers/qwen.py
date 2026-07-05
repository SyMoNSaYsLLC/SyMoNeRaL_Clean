from app.runtime.worker_types import WorkerResult
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
            "has not been wired yet."
        ),
    )
