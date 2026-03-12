import csv
import glob
import json
import os
import re
from typing import Iterable

from strands.tools import tool

from .config import DATA_DIR, RESULT_FILENAME

ID_PATTERN = re.compile(r"\b[A-Z0-9]{8}\b")


def _resolve_data_dir(data_dir: str | None = None) -> str:
    if data_dir and data_dir.strip():
        return os.path.abspath(data_dir)
    return DATA_DIR


def _pick_first_path(data_dir: str, patterns: Iterable[str]) -> str | None:
    for pattern in patterns:
        matches = sorted(glob.glob(os.path.join(data_dir, "**", pattern), recursive=True))
        if matches:
            for match in matches:
                normalized = match.replace("\\", "/")
                base_name = os.path.basename(match)
                if "/__MACOSX/" in normalized:
                    continue
                if base_name.startswith("._"):
                    continue
                return match
    return None


def _load_users_ids(users_path: str) -> set[str]:
    with open(users_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    ids = set()
    for row in payload:
        user_id = row.get("_user_id") or row.get("user_id")
        if isinstance(user_id, str) and user_id:
            ids.add(user_id.strip())
    return ids


def _load_status_ids(status_path: str) -> set[str]:
    ids = set()
    with open(status_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("CitizenID") or row.get("user_id")
            if isinstance(raw, str) and raw:
                ids.add(raw.strip())
    return ids


def _load_test_ids(data_dir: str) -> list[str]:
    test_path = _pick_first_path(data_dir, ("test.txt",))
    if not test_path or not os.path.isfile(test_path):
        return []
    with open(test_path, "r", encoding="utf-8") as f:
        content = f.read()
    return ID_PATTERN.findall(content)


def normalize_ids_text(text: str) -> list[str]:
    return sorted(set(ID_PATTERN.findall(text or "")))


def solve_ids_for_dataset(data_dir: str | None = None) -> list[str]:
    resolved = _resolve_data_dir(data_dir)

    users_path = _pick_first_path(resolved, ("users*.json",))
    status_path = _pick_first_path(resolved, ("status*.csv",))
    if not users_path or not status_path:
        raise FileNotFoundError(
            "Missing required files. Expected users*.json and status*.csv in DATA_DIR."
        )

    users_ids = _load_users_ids(users_path)
    status_ids = _load_status_ids(status_path)
    common_ids = sorted(users_ids.intersection(status_ids))

    test_ids = _load_test_ids(resolved)
    if test_ids:
        filtered = [i for i in test_ids if i in users_ids or i in status_ids]
        if filtered:
            return sorted(set(filtered))
        return sorted(set(test_ids))

    return common_ids


def write_result_file(
    ids: list[str], data_dir: str | None = None, filename: str | None = None
) -> str:
    resolved = _resolve_data_dir(data_dir)
    os.makedirs(resolved, exist_ok=True)
    output_name = filename or RESULT_FILENAME
    output_path = os.path.join(resolved, output_name)
    cleaned = [i.strip() for i in ids if i and i.strip()]
    payload = "\n".join(cleaned)
    if payload:
        payload += "\n"
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(payload)
    return output_path


@tool
def solve_public_dataset(data_dir: str = "") -> str:
    """Resolve challenge IDs from the configured data directory."""
    ids = solve_ids_for_dataset(data_dir or None)
    return "\n".join(ids)


@tool
def write_result(ids_text: str, data_dir: str = "") -> str:
    """Write result.txt with one ID per line and no extra text."""
    ids = normalize_ids_text(ids_text)
    return write_result_file(ids, data_dir or None)
