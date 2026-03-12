import os

from strands.types.exceptions import MaxTokensReachedException

from .agents import create_solver_agent
from .config import (
    DATA_DIR,
    DATA_DIRS,
    RESULT_FILENAME,
    generate_session_id,
    langfuse_client,
)
from .runner import run_llm_call
from .tools import normalize_ids_text, solve_ids_for_dataset, write_result_file


def _save_session_id(session_id: str) -> None:
    session_file = os.path.join(os.path.dirname(__file__), "session_id.txt")
    try:
        with open(session_file, "w", encoding="utf-8") as f:
            f.write(session_id)
        print(f"Session ID saved to {session_file}")
    except Exception:
        print("Warning: unable to write session_id file")


def _prompt_for_solver() -> str:
    return (
        "Read the challenge input files from DATA_DIR and produce the final IDs.\n"
        "Then call write_result with those IDs.\n"
        "Return ONLY the IDs, one per line."
    )


def _parse_dataset_dirs() -> list[str]:
    if DATA_DIRS.strip():
        raw_dirs = [chunk.strip() for chunk in DATA_DIRS.split(",")]
        dirs = [os.path.abspath(d) for d in raw_dirs if d]
        if dirs:
            return dirs
    return [DATA_DIR]


def main():
    dataset_dirs = _parse_dataset_dirs()
    print("Using dataset directories:")
    for d in dataset_dirs:
        print(f"- {d}")

    session_id = generate_session_id()
    print(f"Session ID: {session_id}")
    _save_session_id(session_id)

    for dataset_dir in dataset_dirs:
        print(f"\nProcessing dataset: {dataset_dir}")
        ids: list[str] = []
        used_fallback = False

        try:
            agent = create_solver_agent()
            response = run_llm_call(
                session_id=session_id,
                agent=agent,
                prompt=f"{_prompt_for_solver()}\nDATA_DIR={dataset_dir}",
            )
            ids = normalize_ids_text(response)
        except MaxTokensReachedException:
            print("LLM call failed: max tokens reached. Using deterministic fallback.")
            used_fallback = True
        except Exception as e:
            print(f"LLM call failed: {e}. Using deterministic fallback.")
            used_fallback = True

        if not ids:
            ids = solve_ids_for_dataset(dataset_dir)
            used_fallback = True

        output_path = write_result_file(ids, dataset_dir, RESULT_FILENAME)
        print(f"Result file written: {output_path}")
        print(f"IDs written ({len(ids)}): {', '.join(ids)}")
        if used_fallback:
            print("Result generated via deterministic dataset parser.")

    print("\n[Langfuse] About to flush traces")
    print("[Langfuse] host:", os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))
    langfuse_client.flush()


if __name__ == "__main__":
    main()
