from dataclasses import dataclass

@dataclass
class WorkerResult:
    worker: str
    content: str
    ok: bool = True
    error: str | None = None
