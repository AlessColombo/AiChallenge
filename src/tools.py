import csv
import glob
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from statistics import median
from typing import Iterable

from strands.tools import tool

from .config import DATA_DIR, RESULT_FILENAME

UUID_PATTERN = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
LEGACY_ID_PATTERN = re.compile(r"\b[A-Z0-9]{8}\b")

MAIL_ALERT_TERMS = [
    "urgent",
    "verify",
    "reset password",
    "security alert",
    "winner",
    "claim",
    "wallet",
    "crypto",
    "bit.ly",
    "tinyurl",
]
TX_ALERT_TERMS = ["gift card", "crypto", "wallet", "urgent", "refund", "support", "verify"]
HIGH_RISK_METHODS = {"mobiledevice", "smartwatch", "paypal", "googlepay"}


def _resolve_data_dir(data_dir: str | None = None) -> str:
    if data_dir and data_dir.strip():
        return os.path.abspath(data_dir)
    return DATA_DIR


def _pick_first_path(data_dir: str, patterns: Iterable[str]) -> str | None:
    for pattern in patterns:
        matches = sorted(glob.glob(os.path.join(data_dir, "**", pattern), recursive=True))
        for match in matches:
            normalized = match.replace("\\", "/")
            if "/__MACOSX/" in normalized:
                continue
            if os.path.basename(match).startswith("._"):
                continue
            return match
    return None


def _read_json_array(path: str | None) -> list[dict]:
    if not path or not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, list) else []


def _read_csv_rows(path: str | None) -> list[dict]:
    if not path or not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _norm(text: str | None) -> str:
    raw = text or ""
    folded = unicodedata.normalize("NFKD", raw)
    return folded.encode("ascii", "ignore").decode("ascii").lower().strip()


def _to_float(value: str | float | int | None) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_hour(ts: str | None) -> int | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts).hour
    except ValueError:
        return None


def normalize_ids_text(text: str) -> list[str]:
    if not text:
        return []
    uuids = UUID_PATTERN.findall(text)
    if uuids:
        seen = set()
        out: list[str] = []
        for item in uuids:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out
    return sorted(set(LEGACY_ID_PATTERN.findall(text)))


def _collect_dataset(data_dir: str) -> dict[str, list[dict]]:
    users_path = _pick_first_path(data_dir, ("users*.json",))
    tx_path = _pick_first_path(data_dir, ("transactions*.csv",))
    mail_path = _pick_first_path(data_dir, ("mails*.json", "mail*.json"))
    loc_path = _pick_first_path(data_dir, ("locations*.json",))
    if not users_path or not tx_path:
        raise FileNotFoundError("Expected users*.json and transactions*.csv in DATA_DIR.")
    return {
        "users": _read_json_array(users_path),
        "transactions": _read_csv_rows(tx_path),
        "mails": _read_json_array(mail_path),
        "locations": _read_json_array(loc_path),
    }


def _build_user_maps(users: list[dict], tx_rows: list[dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    by_iban = {
        str(user.get("iban", "")).strip(): user
        for user in users
        if str(user.get("iban", "")).strip()
    }
    by_biotag: dict[str, dict] = {}
    for row in tx_rows:
        sender_id = str(row.get("sender_id", "")).strip()
        recipient_id = str(row.get("recipient_id", "")).strip()
        sender_iban = str(row.get("sender_iban", "")).strip()
        recipient_iban = str(row.get("recipient_iban", "")).strip()
        if sender_id and "-" in sender_id and sender_iban in by_iban:
            by_biotag[sender_id] = by_iban[sender_iban]
        if recipient_id and "-" in recipient_id and recipient_iban in by_iban:
            by_biotag[recipient_id] = by_iban[recipient_iban]
    return by_iban, by_biotag


def _user_risk_from_profile(user: dict) -> int:
    score = 0
    description = _norm(user.get("description"))
    if "phishing" in description or "hameconnage" in description:
        score += 3
    if "clicked" in description or "too trusting" in description:
        score += 2
    if "not always vigilant" in description or "pas toujours vigilante" in description:
        score += 1

    year = user.get("birth_year")
    if isinstance(year, int) and (2087 - year) >= 75:
        score += 2

    salary = user.get("salary")
    if isinstance(salary, (int, float)) and salary < 12000:
        score += 1
    return score


def _mail_risk_by_iban(users: list[dict], mails: list[dict]) -> dict[str, int]:
    scores: dict[str, int] = defaultdict(int)
    for row in mails:
        content = _norm(str(row.get("mail", "")))
        if not content:
            continue
        hits = sum(1 for term in MAIL_ALERT_TERMS if term in content)
        if hits == 0:
            continue
        risk = min(3, hits)
        for user in users:
            full = _norm(f"{user.get('first_name', '')} {user.get('last_name', '')}")
            first = _norm(user.get("first_name"))
            if (full and full in content) or (first and first in content):
                iban = str(user.get("iban", "")).strip()
                if iban:
                    scores[iban] += risk
    return scores


def _sender_amount_stats(tx_rows: list[dict]) -> dict[str, tuple[float, float]]:
    by_sender: dict[str, list[float]] = defaultdict(list)
    for row in tx_rows:
        sender = str(row.get("sender_id", "")).strip()
        amount = _to_float(row.get("amount"))
        if sender and amount is not None:
            by_sender[sender].append(amount)
    stats: dict[str, tuple[float, float]] = {}
    for sender, values in by_sender.items():
        if not values:
            continue
        med = median(values)
        p90 = sorted(values)[max(0, int(0.9 * (len(values) - 1)))]
        stats[sender] = (med, p90)
    return stats


def _tx_user_score(row: dict, by_biotag: dict[str, dict], mail_scores: dict[str, int]) -> int:
    sender = str(row.get("sender_id", "")).strip()
    recipient = str(row.get("recipient_id", "")).strip()
    sender_user = by_biotag.get(sender)
    recipient_user = by_biotag.get(recipient)
    sender_score = _user_risk_from_profile(sender_user) if sender_user else 0
    recipient_score = _user_risk_from_profile(recipient_user) if recipient_user else 0
    if sender_user:
        sender_score += mail_scores.get(str(sender_user.get("iban", "")).strip(), 0)
    if recipient_user:
        recipient_score += mail_scores.get(str(recipient_user.get("iban", "")).strip(), 0)
    return max(sender_score, recipient_score)


def detect_mail_candidates_for_dataset(data_dir: str | None = None) -> list[str]:
    resolved = _resolve_data_dir(data_dir)
    dataset = _collect_dataset(resolved)
    users = dataset["users"]
    tx_rows = dataset["transactions"]
    mails = dataset["mails"]
    _, by_biotag = _build_user_maps(users, tx_rows)
    mail_scores = _mail_risk_by_iban(users, mails)
    amount_stats = _sender_amount_stats(tx_rows)

    scored: list[tuple[str, int]] = []
    for row in tx_rows:
        tx_id = str(row.get("transaction_id", "")).strip()
        if not tx_id:
            continue
        score = 0
        tx_type = _norm(str(row.get("transaction_type", "")))
        method = _norm(str(row.get("payment_method", "")))
        description = _norm(str(row.get("description", "")))
        sender = str(row.get("sender_id", "")).strip()
        recipient = str(row.get("recipient_id", "")).strip()
        amount = _to_float(row.get("amount"))

        user_score = _tx_user_score(row, by_biotag, mail_scores)
        if user_score >= 6:
            score += 2
        elif user_score >= 3:
            score += 1

        if method in HIGH_RISK_METHODS:
            score += 1
        if "e-commerce" in tx_type or "online" in tx_type:
            score += 1
        if description and any(term in description for term in TX_ALERT_TERMS):
            score += 2
        if sender.startswith("EXT") or recipient.startswith("EXT"):
            score += 1

        if amount is not None and sender in amount_stats:
            med, p90 = amount_stats[sender]
            if med > 0 and amount >= med * 2.5 and amount >= 300:
                score += 2
            elif amount > p90 and amount >= 500:
                score += 1

        if score >= 3:
            scored.append((tx_id, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    max_flags = max(1, int(len(tx_rows) * 0.35))
    return [tx_id for tx_id, _ in scored[:max_flags]]


def _dominant_city_by_biotag(locations: list[dict]) -> dict[str, str]:
    counters: dict[str, Counter] = defaultdict(Counter)
    for row in locations:
        biotag = str(row.get("biotag", "")).strip()
        city = str(row.get("city", "")).strip()
        if biotag and city:
            counters[biotag][city] += 1
    return {tag: counts.most_common(1)[0][0] for tag, counts in counters.items()}


def detect_location_candidates_for_dataset(data_dir: str | None = None) -> list[str]:
    resolved = _resolve_data_dir(data_dir)
    dataset = _collect_dataset(resolved)
    tx_rows = dataset["transactions"]
    users = dataset["users"]
    _, by_biotag = _build_user_maps(users, tx_rows)
    dominant_city = _dominant_city_by_biotag(dataset["locations"])
    amount_stats = _sender_amount_stats(tx_rows)

    scored: list[tuple[str, int]] = []
    for row in tx_rows:
        tx_id = str(row.get("transaction_id", "")).strip()
        if not tx_id:
            continue
        score = 0
        sender = str(row.get("sender_id", "")).strip()
        tx_type = _norm(str(row.get("transaction_type", "")))
        location = str(row.get("location", "")).strip()
        hour = _parse_hour(str(row.get("timestamp", "")))
        amount = _to_float(row.get("amount"))

        if location and sender in dominant_city and dominant_city[sender] != location:
            score += 3

        if "in-person" in tx_type and hour is not None and hour <= 5:
            score += 1

        if amount is not None and sender in amount_stats:
            med, p90 = amount_stats[sender]
            if med > 0 and amount >= med * 2.5 and amount >= 300:
                score += 1
            elif amount > p90 and amount >= 500:
                score += 1

        user = by_biotag.get(sender)
        if user and _user_risk_from_profile(user) >= 5:
            score += 1

        if score >= 3:
            scored.append((tx_id, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    max_flags = max(1, int(len(tx_rows) * 0.35))
    return [tx_id for tx_id, _ in scored[:max_flags]]


def orchestrate_fraud_ids_for_dataset(
    data_dir: str | None = None,
    mail_candidates: list[str] | None = None,
    location_candidates: list[str] | None = None,
) -> list[str]:
    resolved = _resolve_data_dir(data_dir)
    dataset = _collect_dataset(resolved)
    tx_rows = dataset["transactions"]
    users = dataset["users"]
    _, by_biotag = _build_user_maps(users, tx_rows)
    amount_stats = _sender_amount_stats(tx_rows)

    mail_set = set(mail_candidates or detect_mail_candidates_for_dataset(resolved))
    loc_set = set(location_candidates or detect_location_candidates_for_dataset(resolved))
    union_ids = mail_set.union(loc_set)
    if not union_ids:
        return []

    tx_by_id = {str(row.get("transaction_id", "")).strip(): row for row in tx_rows}
    scored: list[tuple[str, int]] = []

    for tx_id in union_ids:
        row = tx_by_id.get(tx_id)
        if not row:
            continue
        score = 0
        if tx_id in mail_set:
            score += 2
        if tx_id in loc_set:
            score += 2

        sender = str(row.get("sender_id", "")).strip()
        recipient = str(row.get("recipient_id", "")).strip()
        sender_user = by_biotag.get(sender)
        recipient_user = by_biotag.get(recipient)
        if sender_user:
            score += 1 if _user_risk_from_profile(sender_user) >= 4 else 0
        if recipient_user:
            score += 1 if _user_risk_from_profile(recipient_user) >= 4 else 0

        amount = _to_float(row.get("amount"))
        if amount is not None and sender in amount_stats:
            _, p90 = amount_stats[sender]
            if amount > p90 and amount >= 500:
                score += 1

        scored.append((tx_id, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    filtered = [tx_id for tx_id, score in scored if score >= 3]
    if not filtered:
        filtered = [tx_id for tx_id, _ in scored[: max(1, len(scored) // 3)]]

    max_flags = max(1, int(len(tx_rows) * 0.40))
    return filtered[:max_flags]


def solve_ids_for_dataset(data_dir: str | None = None) -> list[str]:
    return orchestrate_fraud_ids_for_dataset(data_dir)


def write_result_file(
    ids: list[str], data_dir: str | None = None, filename: str | None = None
) -> str:
    resolved = _resolve_data_dir(data_dir)
    os.makedirs(resolved, exist_ok=True)
    output_name = filename or RESULT_FILENAME
    output_path = os.path.join(resolved, output_name)
    cleaned = [item.strip() for item in ids if item and item.strip()]
    payload = "\n".join(cleaned)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(payload)
    return output_path


@tool
def mail_transaction_candidates(data_dir: str = "") -> str:
    """Agent A: suspicious transactions from transactions+mails."""
    ids = detect_mail_candidates_for_dataset(data_dir or None)
    return "\n".join(ids)


@tool
def location_transaction_candidates(data_dir: str = "") -> str:
    """Agent B: suspicious transactions from transactions+locations."""
    ids = detect_location_candidates_for_dataset(data_dir or None)
    return "\n".join(ids)


@tool
def orchestrate_fraudulent_transactions(
    mail_candidates_text: str = "",
    location_candidates_text: str = "",
    data_dir: str = "",
) -> str:
    """Merge agent outputs with users profile and return final fraud IDs."""
    mail_ids = normalize_ids_text(mail_candidates_text)
    loc_ids = normalize_ids_text(location_candidates_text)
    ids = orchestrate_fraud_ids_for_dataset(data_dir or None, mail_ids, loc_ids)
    return "\n".join(ids)


@tool
def solve_public_dataset(data_dir: str = "") -> str:
    """Run orchestrator workflow and return one transaction_id per line."""
    ids = solve_ids_for_dataset(data_dir or None)
    return "\n".join(ids)


@tool
def write_result(ids_text: str, data_dir: str = "", filename: str = "") -> str:
    """Write result file with one transaction_id per line."""
    ids = normalize_ids_text(ids_text)
    return write_result_file(ids, data_dir or None, filename or None)
