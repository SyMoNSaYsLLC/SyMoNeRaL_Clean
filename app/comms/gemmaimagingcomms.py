from app.workers import gemma, safetensor

def run(original_prompt: str, manager_task: str):
    result = safetensor.image(manager_task or original_prompt)
    checked = gemma.final_check(original_prompt, result.content)
    return checked
