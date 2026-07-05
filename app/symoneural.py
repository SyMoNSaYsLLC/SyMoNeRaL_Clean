from app.workers import gemma, qwen, safetensor
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
