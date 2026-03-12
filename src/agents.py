from strands import Agent

from .config import model


def create_solver_agent() -> Agent:
    """Create the agent that solves the dataset and returns only IDs."""
    from .tools import solve_public_dataset, write_result

    return Agent(
        model=model,
        system_prompt=(
            "You solve the challenge dataset with tools.\n"
            "Rules:\n"
            "- Use solve_public_dataset to get the IDs.\n"
            "- Use write_result to save the result file.\n"
            "- Final answer must contain ONLY IDs, one per line.\n"
            "- Do not include explanations, labels, markdown, or punctuation."
        ),
        tools=[solve_public_dataset, write_result],
    )
