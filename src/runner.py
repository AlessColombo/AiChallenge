from .config import MODEL_ID
from .langfuse_runtime import run_llm_call as _run_llm_call


def run_llm_call(session_id: str, agent, prompt: str) -> str:
    """Backward-compatible wrapper around the new langfuse runtime module."""
    return _run_llm_call(session_id=session_id, model_id=MODEL_ID, agent=agent, prompt=prompt)
