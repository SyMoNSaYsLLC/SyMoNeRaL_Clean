import subprocess

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
