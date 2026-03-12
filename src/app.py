import argparse
import os

from strands.types.exceptions import MaxTokensReachedException

from .agents import create_location_agent, create_mail_agent, create_orchestrator_agent
from .config import DATA_DIR, DATA_DIRS, MODEL_ID
from .langfuse_runtime import (
    generate_session_id,
    langfuse_client,
    print_langfuse_startup_info,
    run_llm_call,
)
from .tools import (
    detect_location_candidates_for_dataset,
    detect_mail_candidates_for_dataset,
    normalize_ids_text,
    orchestrate_fraud_ids_for_dataset,
    write_result_file,
)

RESULTS_FILENAME = "results.txt"


def _save_session_id(session_id: str, output_dir: str) -> None:
    session_file = os.path.join(output_dir, "session_id.txt")
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(session_file, "w", encoding="utf-8") as f:
            f.write(session_id)
        print(f"Session ID saved to {session_file}")
    except Exception:
        print("Warning: unable to write session_id file")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fraud orchestrator on one dataset directory or configured ones."
    )
    parser.add_argument(
        "--data-dir",
        default="",
        help="Process only this dataset directory.",
    )
    return parser.parse_args()


def _parse_dataset_dirs(cli_data_dir: str = "") -> list[str]:
    if cli_data_dir.strip():
        return [os.path.abspath(cli_data_dir)]
    if os.getenv("DATA_DIR"):
        return [DATA_DIR]
    if DATA_DIRS.strip():
        raw_dirs = [chunk.strip() for chunk in DATA_DIRS.split(",")]
        dirs = [os.path.abspath(d) for d in raw_dirs if d]
        if dirs:
            return dirs
    return [DATA_DIR]


def _mail_prompt(data_dir: str) -> str:
    return (
        "Use transactions and mails to detect potentially fraudulent transactions.\n"
        "Call mail_transaction_candidates with DATA_DIR and return ONLY IDs.\n"
        f"DATA_DIR={data_dir}"
    )


def _location_prompt(data_dir: str) -> str:
    return (
        "Use transactions and locations to detect potentially fraudulent transactions.\n"
        "Call location_transaction_candidates with DATA_DIR and return ONLY IDs.\n"
        f"DATA_DIR={data_dir}"
    )


def _orchestrator_prompt(data_dir: str, mail_ids: list[str], location_ids: list[str]) -> str:
    return (
        "Merge the two candidate sets using users context.\n"
        "Call orchestrate_fraudulent_transactions with:\n"
        f"- data_dir={data_dir}\n"
        f"- mail_candidates_text={chr(10).join(mail_ids)}\n"
        f"- location_candidates_text={chr(10).join(location_ids)}\n"
        f"Then call write_result with filename={RESULTS_FILENAME}.\n"
        "Return ONLY final transaction IDs."
    )


def _run_mail_agent(session_id: str, data_dir: str) -> list[str]:
    try:
        response = run_llm_call(
            session_id=session_id,
            model_id=MODEL_ID,
            agent=create_mail_agent(),
            prompt=_mail_prompt(data_dir),
        )
        ids = normalize_ids_text(response)
        if ids:
            return ids
    except MaxTokensReachedException:
        print("Mail agent: max tokens reached. Using deterministic fallback.")
    except Exception as exc:
        print(f"Mail agent failed: {exc}. Using deterministic fallback.")
    return detect_mail_candidates_for_dataset(data_dir)


def _run_location_agent(session_id: str, data_dir: str) -> list[str]:
    try:
        response = run_llm_call(
            session_id=session_id,
            model_id=MODEL_ID,
            agent=create_location_agent(),
            prompt=_location_prompt(data_dir),
        )
        ids = normalize_ids_text(response)
        if ids:
            return ids
    except MaxTokensReachedException:
        print("Location agent: max tokens reached. Using deterministic fallback.")
    except Exception as exc:
        print(f"Location agent failed: {exc}. Using deterministic fallback.")
    return detect_location_candidates_for_dataset(data_dir)


def _run_orchestrator_agent(
    session_id: str,
    data_dir: str,
    mail_ids: list[str],
    location_ids: list[str],
) -> list[str]:
    try:
        response = run_llm_call(
            session_id=session_id,
            model_id=MODEL_ID,
            agent=create_orchestrator_agent(),
            prompt=_orchestrator_prompt(data_dir, mail_ids, location_ids),
        )
        ids = normalize_ids_text(response)
        if ids:
            return ids
    except MaxTokensReachedException:
        print("Orchestrator agent: max tokens reached. Using deterministic fallback.")
    except Exception as exc:
        print(f"Orchestrator agent failed: {exc}. Using deterministic fallback.")
    return orchestrate_fraud_ids_for_dataset(data_dir, mail_ids, location_ids)


def main():
    args = _parse_args()
    dataset_dirs = _parse_dataset_dirs(args.data_dir)
    print("Using dataset directories:")
    for path in dataset_dirs:
        print(f"- {path}")

    session_id = generate_session_id()
    print_langfuse_startup_info()
    print(f"Session ID: {session_id}")

    for dataset_dir in dataset_dirs:
        print(f"\nProcessing dataset: {dataset_dir}")
        mail_ids = _run_mail_agent(session_id, dataset_dir)
        print(f"Mail agent candidates: {len(mail_ids)}")

        location_ids = _run_location_agent(session_id, dataset_dir)
        print(f"Location agent candidates: {len(location_ids)}")

        final_ids = _run_orchestrator_agent(session_id, dataset_dir, mail_ids, location_ids)
        output_path = write_result_file(final_ids, dataset_dir, RESULTS_FILENAME)
        _save_session_id(session_id, os.path.dirname(output_path))

        print(f"Result file written: {output_path}")
        print(f"Fraud IDs written ({len(final_ids)}): {', '.join(final_ids)}")

    print("\n[Langfuse] About to flush traces")
    print("[Langfuse] host:", os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"))
    langfuse_client.flush()


if __name__ == "__main__":
    main()
