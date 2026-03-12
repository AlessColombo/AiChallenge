import os

import ulid
from langfuse import Langfuse, observe
from strands import Agent

# Initialize Langfuse client
langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"),
)


def generate_session_id() -> str:
    """Generate a unique session ID using TEAM_NAME and ULID."""
    team_name = os.getenv("TEAM_NAME", "tutorial")
    return f"{team_name}-{ulid.new().str}"


def print_langfuse_startup_info() -> None:
    print("✓ Langfuse initialized successfully")
    pk = os.getenv("LANGFUSE_PUBLIC_KEY", "Not set")
    preview = f"{pk[:20]}..." if pk != "Not set" else pk
    print(f"✓ Public key: {preview}")
    print("✓ Helper functions ready: generate_session_id()")


def create_agent(model, system_prompt: str = "You are a helpful AI assistant that provides clear and concise answers."):
    """Create a fresh Strands agent (new conversation context)."""
    return Agent(model=model, system_prompt=system_prompt)


@observe(as_type="generation")
def run_llm_call(session_id: str, model_id: str, agent, prompt: str) -> str:
    """Run a single Strands agent call and track it in Langfuse."""
    langfuse_client.update_current_trace(session_id=session_id)
    langfuse_client.update_current_generation(
        model=model_id,
        input=[{"role": "user", "content": prompt}],
    )

    result = agent(prompt)
    response = str(result)

    invocation = result.metrics.latest_agent_invocation
    usage = invocation.usage if invocation else result.metrics.accumulated_usage

    langfuse_client.update_current_generation(
        model=model_id,
        output=response,
        usage_details={
            "input": usage.get("inputTokens", 0),
            "output": usage.get("outputTokens", 0),
            "total": usage.get("totalTokens", 0),
        },
    )
    return response
