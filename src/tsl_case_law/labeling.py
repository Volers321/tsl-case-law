"""Expert labelling: export a sample of cases to CSV, read the filled-in sheet back.

The domain expert (a TSL company owner) checks each sampled case in a spreadsheet:
is it really a transport/forwarding dispute, what kind of dispute, and is the rule-based
outcome correct. These labels are the gold standard that every automatic step —
relevance filter, outcome rules, LLM extraction, outcome prediction — is measured against.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

DISPUTE_TYPES = (
    "zaplata_frachtu",  # carrier sues for unpaid freight
    "szkoda_w_ladunku",  # loss of or damage to cargo
    "opoznienie",  # late delivery
    "odpowiedzialnosc_spedytora",  # forwarder's liability / choice of carrier
    "kary_umowne",  # contractual penalties
    "przedawnienie_reklamacja",  # limitation periods, complaints procedure
    "ubezpieczenie_oc",  # carrier's liability insurance
    "inne",
)

EXPERT_COLUMNS = ("is_tsl", "dispute_type", "outcome_rule_ok", "outcome_gold", "notes")
EXPORT_COLUMNS = (
    "id",
    "case_number",
    "court",
    "date",
    "instance",
    "outcome_rule",
    "sentencja",
    "facts_excerpt",
    *EXPERT_COLUMNS,
)


def _excerpt(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"


def stratified_sample(records: list[dict], n: int, *, seed: int = 13) -> list[dict]:
    """Sample across rule outcomes so rare classes (e.g. SET_ASIDE) are represented."""
    rng = random.Random(seed)
    by_outcome: dict[str, list[dict]] = {}
    for r in records:
        by_outcome.setdefault(r["outcome_rule"], []).append(r)
    for group in by_outcome.values():
        rng.shuffle(group)
    sample: list[dict] = []
    while len(sample) < n and any(by_outcome.values()):
        for key in sorted(by_outcome):  # round-robin over outcome classes
            if by_outcome[key] and len(sample) < n:
                sample.append(by_outcome[key].pop())
    return sample


def export_for_labeling(records: list[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel / Numbers open Polish characters correctly
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=EXPORT_COLUMNS, delimiter=";")
        w.writeheader()
        for r in records:
            w.writerow(
                {
                    "id": r["id"],
                    "case_number": r["case_number"],
                    "court": r["court"],
                    "date": r["date"],
                    "instance": r["instance"],
                    "outcome_rule": r["outcome_rule"],
                    "sentencja": _excerpt(r["sentencja"], 1200),
                    "facts_excerpt": _excerpt(r["facts"], 1500),
                    **dict.fromkeys(EXPERT_COLUMNS, ""),
                }
            )
    return len(records)


def read_labels(path: str | Path) -> dict[int, dict]:
    """Read the expert's sheet. Accepts ';' or ',' (spreadsheets differ by locale)."""
    raw = Path(path).read_text(encoding="utf-8-sig")
    dialect = csv.Sniffer().sniff(raw.splitlines()[0], delimiters=";,")
    labels = {}
    for row in csv.DictReader(raw.splitlines(), dialect=dialect):
        if not (row.get("is_tsl") or "").strip():
            continue  # not labelled yet
        labels[int(row["id"])] = {k: (row.get(k) or "").strip() for k in EXPERT_COLUMNS}
    return labels
