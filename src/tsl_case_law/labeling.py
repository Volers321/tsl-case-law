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
    "outcome_source",
    "sentencja_end",
    "conclusion",
    "facts_excerpt",
    "batch",
    *EXPERT_COLUMNS,
)


def _excerpt(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"


def _tail(text: str, n: int) -> str:
    """End of a text: the operative points come after the court header in a sentencja,
    and the court's conclusion closes the reasons."""
    text = " ".join(text.split())
    return text if len(text) <= n else "… " + text[-n:].split(" ", 1)[-1]


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


def export_for_labeling(
    records: list[dict], path: str | Path, *, append: bool = False, batch: int = 1
) -> int:
    """Write a label sheet; with ``append`` add rows as a new batch, keeping earlier labels.

    Batches matter for honest evaluation: anything tuned after looking at batch 1 must be
    measured on a later batch.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old_rows: list[dict] = []
    if append and path.exists():
        _, old_rows = _read_sheet(path)
        for r in old_rows:
            r["batch"] = r.get("batch") or "1"
    # utf-8-sig so Excel / Numbers open Polish characters correctly
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=EXPORT_COLUMNS, delimiter=";", extrasaction="ignore")
        w.writeheader()
        w.writerows(old_rows)
        for r in records:
            w.writerow(
                {
                    "id": r["id"],
                    "case_number": r["case_number"],
                    "court": r["court"],
                    "date": r["date"],
                    "instance": r["instance"],
                    "outcome_rule": r["outcome_rule"],
                    "outcome_source": r.get("outcome_source", ""),
                    "sentencja_end": _tail(r["sentencja"], 900),
                    "conclusion": _tail(r.get("assessment", ""), 700),
                    "facts_excerpt": _excerpt(r["facts"], 1500),
                    "batch": batch,
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
        labels[int(row["id"])]["batch"] = int((row.get("batch") or "1").strip() or 1)
        # The outcome the expert saw and confirmed — rules may change later.
        labels[int(row["id"])]["shown_outcome"] = (row.get("outcome_rule") or "").strip()
    return labels


# ------------------------------------------------------------------ interactive labelling

OUTCOMES_FIRST = ("GRANTED", "PARTIAL", "DISMISSED", "DISCONTINUED", "UNKNOWN")
OUTCOMES_APPEAL = ("APPEAL_DISMISSED", "CHANGED", "SET_ASIDE", "UNKNOWN")
OUTCOME_PL = {
    "GRANTED": "powództwo uwzględnione w całości",
    "PARTIAL": "uwzględnione w części",
    "DISMISSED": "powództwo oddalone",
    "DISCONTINUED": "postępowanie umorzone",
    "APPEAL_DISMISSED": "apelacja oddalona / odrzucona",
    "CHANGED": "wyrok zmieniony",
    "SET_ASIDE": "wyrok uchylony",
    "UNKNOWN": "nie da się ustalić",
}
DISPUTE_PL = {
    "zaplata_frachtu": "zapłata frachtu",
    "szkoda_w_ladunku": "szkoda / utrata ładunku",
    "opoznienie": "opóźnienie dostawy",
    "odpowiedzialnosc_spedytora": "odpowiedzialność spedytora",
    "kary_umowne": "kary umowne",
    "przedawnienie_reklamacja": "przedawnienie / reklamacja",
    "ubezpieczenie_oc": "ubezpieczenie OC przewoźnika",
    "inne": "inne",
}
YES_ANSWERS = {"", "t", "tak", "y"}


class Quit(Exception):
    pass


def _read_sheet(path: Path) -> tuple[list[str], list[dict]]:
    raw = path.read_text(encoding="utf-8-sig")
    dialect = csv.Sniffer().sniff(raw.splitlines()[0], delimiters=";,")
    reader = csv.DictReader(raw.splitlines(), dialect=dialect)
    return list(reader.fieldnames or []), list(reader)


def _write_sheet(path: Path, fields: list[str], rows: list[dict]) -> None:
    tmp = path.with_suffix(".part")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)  # saved after every case: closing the window loses nothing


def _choose(ask, prompt: str, options: tuple[str, ...], labels: dict[str, str]) -> str:
    menu = "\n".join(f"   {i}  {labels.get(o, o)}" for i, o in enumerate(options, start=1))
    while True:
        answer = ask(f"{prompt}\n{menu}\n > ").strip().lower()
        if answer == "q":
            raise Quit
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]


def _yes_no(ask, prompt: str) -> bool:
    while True:
        answer = ask(prompt).strip().lower()
        if answer == "q":
            raise Quit
        if answer in YES_ANSWERS:
            return True
        if answer in ("n", "nie"):
            return False


def label_interactively(
    path: str | Path,
    *,
    ask=input,
    show=print,
    width: int = 500,
    review: dict[int, str] | None = None,
) -> int:
    """Walk through unlabelled rows one at a time; Enter = "tak", q = save and quit.

    With ``review`` ({id: hint}) only those rows are shown again, already labelled or not,
    together with the hint (e.g. the LLM's summary where it disagrees with the expert), so
    disagreements can be adjudicated. Returns the number of rows labelled in this session.
    Progress is written to the same CSV after every case.
    """
    path = Path(path)
    fields, rows = _read_sheet(path)
    if review is not None:
        # Rows keep their old labels until answered again, so quitting (or closing the
        # window) at any point never loses an earlier label.
        todo = [r for r in rows if int(r["id"]) in review]
        done_before = 0
        rows_total = len(todo)
    else:
        todo = [r for r in rows if not (r.get("is_tsl") or "").strip()]
        done_before = len(rows) - len(todo)
        rows_total = len(rows)
    show(f"\nOcenione: {done_before}/{rows_total}. Enter = tak, n = nie, q = zapisz i wyjdź.\n")
    labelled = 0
    for row in todo:
        n = done_before + labelled + 1
        appeal = row.get("instance") == "appeal"
        show("=" * 78)
        show(f"[{n}/{rows_total}] {row['case_number']} · {row['court']} · {row['date']}")
        show(f"Instancja: {'APELACJA / kasacja' if appeal else 'pierwsza'}")
        show(f"\nFAKTY (początek): {_excerpt(row.get('facts_excerpt', ''), width)}")
        show(f"\nROZSTRZYGNIĘCIE (koniec sentencji): {_tail(row.get('sentencja_end', ''), width)}")
        if row.get("conclusion"):
            show(f"\nWNIOSEK SĄDU (koniec uzasadnienia): {_tail(row['conclusion'], 300)}")
        if review is not None:
            show(f"\nSTRESZCZENIE LLM: {review[int(row['id'])]}")
            show(f"Twoja wcześniejsza odpowiedź na pytanie 1: {row.get('is_tsl') or '—'}")
        show("")
        try:
            if not _yes_no(ask, "1) Czy to spór z przewozu / spedycji? [Enter=tak / n] > "):
                row.update(is_tsl="nie", dispute_type="", outcome_rule_ok="", outcome_gold="")
            else:
                dispute = _choose(ask, "2) Rodzaj sporu:", DISPUTE_TYPES, DISPUTE_PL)
                rule = row.get("outcome_rule", "UNKNOWN")
                ok = _yes_no(
                    ask,
                    f"3) Program odczytał wynik: {rule} = {OUTCOME_PL.get(rule, rule)}. "
                    "Zgadza się? [Enter=tak / n] > ",
                )
                gold = ""
                if not ok:
                    options = OUTCOMES_APPEAL if appeal else OUTCOMES_FIRST
                    gold = _choose(ask, "   Jaki jest poprawny wynik?", options, OUTCOME_PL)
                row.update(
                    is_tsl="tak",
                    dispute_type=dispute,
                    outcome_rule_ok="tak" if ok else "nie",
                    outcome_gold=gold,
                )
        except (Quit, EOFError, KeyboardInterrupt):
            show(f"\nZapisano. Ocenione: {done_before + labelled}/{rows_total}.")
            return labelled
        _write_sheet(path, fields, rows)
        labelled += 1
    show(f"\nGotowe! Ocenione wszystkie {rows_total} sprawy. Teraz: tsl check-labels")
    return labelled
