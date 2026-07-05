from app.workers import gemma, qwen

def run(original_prompt: str, manager_task: str, max_tokens: int):
    draft = qwen.code(manager_task, max_tokens=max_tokens)
    checked = gemma.final_check(original_prompt, draft.content)
    return checked
