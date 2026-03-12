from strands import Agent

from .config import model


def create_mail_agent() -> Agent:
    """Agent 1: use transactions+mails to find suspicious transactions."""
    from .tools import mail_transaction_candidates

    return Agent(
        model=model,
        system_prompt=(
            "You are the Mail Fraud Agent.\n"
            "Use mail_transaction_candidates on DATA_DIR.\n"
            "Return ONLY transaction IDs, one per line."
        ),
        tools=[mail_transaction_candidates],
    )


def create_location_agent() -> Agent:
    """Agent 2: use transactions+locations to find suspicious transactions."""
    from .tools import location_transaction_candidates

    return Agent(
        model=model,
        system_prompt=(
            "You are the Location Fraud Agent.\n"
            "Use location_transaction_candidates on DATA_DIR.\n"
            "Return ONLY transaction IDs, one per line."
        ),
        tools=[location_transaction_candidates],
    )


def create_orchestrator_agent() -> Agent:
    """Orchestrator: merges both candidate sets with users context."""
    from .tools import orchestrate_fraudulent_transactions, write_result

    return Agent(
        model=model,
        system_prompt=(
            "You are the Fraud Orchestrator.\n"
            "Rules:\n"
            "- Read the two candidate lists already provided in the prompt.\n"
            "- Call orchestrate_fraudulent_transactions(mail_candidates_text, "
            "location_candidates_text, data_dir).\n"
            "- Then call write_result with filename=results.txt.\n"
            "- Final answer must contain ONLY transaction IDs, one per line."
        ),
        tools=[orchestrate_fraudulent_transactions, write_result],
    )
