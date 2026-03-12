import os

from langfuse import observe

from .config import langfuse_client, MODEL_ID


@observe(as_type="generation")
def run_llm_call(session_id: str, agent, prompt: str) -> str:
    """Invoke a Strands agent while sending tracing data to Langfuse.

    The caller must pass the active ``session_id`` and ``agent``.  The
    ``MODEL_ID`` constant is imported from :mod:`config` so that the trace
    matches the model used by the agent.
    """

    # debug: show configuration and call info before sending anything
    try:
        host_info = getattr(langfuse_client, "_base_url", None) or os.getenv("LANGFUSE_HOST")
    except Exception:
        host_info = os.getenv("LANGFUSE_HOST")
    print("[Langfuse] host:", host_info)
    print("[Langfuse] public key:", os.getenv("LANGFUSE_PUBLIC_KEY"))
    print("[Langfuse] secret key:", os.getenv("LANGFUSE_SECRET_KEY"))
    print(f"[Langfuse] session_id: {session_id}")
    print(f"[Langfuse] model: {MODEL_ID}")
    print(f"[Langfuse] prompt: {prompt}\n")

    # associate the trace with the session identifier
    langfuse_client.update_current_trace(session_id=session_id)

    # record the generation input
    langfuse_client.update_current_generation(
        model=MODEL_ID,
        input=[{"role": "user", "content": prompt}],
    )

    # perform the call
    result = agent(prompt)
    response = str(result)

    # extract token usage from the invocation metrics
    invocation = result.metrics.latest_agent_invocation
    usage = invocation.usage if invocation else result.metrics.accumulated_usage

    # send the usage and output back to Langfuse
    langfuse_client.update_current_generation(
        model=MODEL_ID,
        output=response,
        usage_details={
            "input": usage.get("inputTokens", 0),
            "output": usage.get("outputTokens", 0),
            "total": usage.get("totalTokens", 0),
        },
    )

    return response