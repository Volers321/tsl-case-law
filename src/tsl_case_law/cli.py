"""Command line: ``tsl <command>``.

Typical flow:
    tsl download --from 2015-01-01     # search SAOS and fetch full judgments
    tsl build                          # raw JSON -> data/cases.jsonl (+ filters)
    tsl stats                          # what we've got
    tsl sample -n 50                   # CSV for the expert to label
    tsl check-labels                   # how good are the rules vs. the expert?
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .labeling import export_for_labeling, read_labels, stratified_sample
from .processing import to_record

RAW = Path("data/raw")
CASES = Path("data/cases.jsonl")
IDS = RAW / "_ids.json"
LABEL_SHEET = Path("data/labels/sample.csv")


def load_cases(path: Path = CASES) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} not found — run `tsl build` first.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def cmd_download(args) -> None:
    from .saos import SaosClient, fetch_judgments, search_ids

    client = SaosClient(delay=args.delay)
    RAW.mkdir(parents=True, exist_ok=True)
    if IDS.exists() and not args.refresh:
        found = {int(k): v for k, v in json.loads(IDS.read_text(encoding="utf-8")).items()}
        print(f"Using {len(found)} ids from {IDS} (pass --refresh to search again)")
    else:
        found = search_ids(client, date_from=args.date_from, date_to=args.date_to)
        IDS.write_text(json.dumps(found, ensure_ascii=False, indent=1), encoding="utf-8")
    n = fetch_judgments(client, sorted(found, reverse=True), RAW, limit=args.limit)
    total = len(list(RAW.glob("[0-9]*.json")))
    print(f"Downloaded {n} new judgments; {total} on disk in {RAW}/")


def cmd_build(args) -> None:
    files = sorted(RAW.glob("[0-9]*.json"))
    if not files:
        raise SystemExit(f"No judgments in {RAW}/ — run `tsl download` first.")
    kept, dropped = [], Counter()
    for f in files:
        rec = to_record(json.loads(f.read_text(encoding="utf-8")))
        if rec.tsl_score < args.min_score:
            dropped["low TSL score"] += 1
        elif not rec.facts:
            dropped["no reasons / facts section"] += 1
        else:
            kept.append(rec.to_dict())
    kept.sort(key=lambda r: r["date"])
    CASES.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8"
    )
    print(f"Kept {len(kept)} of {len(files)} judgments -> {CASES}")
    for reason, n in dropped.most_common():
        print(f"  dropped {n:5d}: {reason}")


def cmd_stats(args) -> None:
    cases = load_cases()
    print(f"cases: {len(cases)}   dates: {cases[0]['date']} … {cases[-1]['date']}")
    for key in ("court_type", "instance", "outcome_rule"):
        counts = Counter(c[key] for c in cases)
        print(f"\n{key}:")
        for value, n in counts.most_common():
            print(f"  {value or '—':20s} {n:5d}  {n / len(cases):6.1%}")
    years = Counter(c["date"][:4] for c in cases)
    print("\nper year: " + "  ".join(f"{y}:{n}" for y, n in sorted(years.items())))
    facts_len = sorted(len(c["facts"].split()) for c in cases)
    print(f"facts length (words): median {facts_len[len(facts_len) // 2]}, max {facts_len[-1]}")


def cmd_sample(args) -> None:
    cases = load_cases()
    if LABEL_SHEET.exists() and not args.force:
        raise SystemExit(f"{LABEL_SHEET} exists — it may contain your labels. Use --force.")
    sample = stratified_sample(cases, args.n, seed=args.seed)
    export_for_labeling(sample, LABEL_SHEET)
    print(f"Wrote {len(sample)} cases to {LABEL_SHEET}")
    print("Open it in Numbers/Excel, fill the last 5 columns, save as CSV (same name).")


def cmd_check_labels(args) -> None:
    labels = read_labels(LABEL_SHEET)
    if not labels:
        raise SystemExit(f"No labelled rows in {LABEL_SHEET} yet.")
    cases = {c["id"]: c for c in load_cases()}
    yes = {"tak", "t", "yes", "y", "1", "true"}
    tsl = [i for i, lab in labels.items() if lab["is_tsl"].lower() in yes]
    print(f"labelled: {len(labels)}   relevant to TSL: {len(tsl)} ({len(tsl) / len(labels):.0%})")
    judged = [i for i in tsl if labels[i]["outcome_rule_ok"]]
    ok = [i for i in judged if labels[i]["outcome_rule_ok"].lower() in yes]
    if judged:
        print(f"rule-based outcome correct: {len(ok)}/{len(judged)} ({len(ok) / len(judged):.0%})")
    wrong = [i for i in judged if i not in ok]
    for i in wrong:
        c = cases.get(i, {})
        print(
            f"  ✗ {c.get('case_number', i)}: rule={c.get('outcome_rule')} gold={labels[i]['outcome_gold']}"
        )
    types = Counter(labels[i]["dispute_type"] or "—" for i in tsl)
    print("dispute types: " + ", ".join(f"{t}:{n}" for t, n in types.most_common()))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="tsl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="search SAOS and download full judgments")
    d.add_argument("--from", dest="date_from", default="2015-01-01")
    d.add_argument("--to", dest="date_to")
    d.add_argument("--limit", type=int, help="max judgments to fetch this run")
    d.add_argument("--delay", type=float, default=0.3, help="seconds between requests")
    d.add_argument("--refresh", action="store_true", help="re-run the searches")
    d.set_defaults(func=cmd_download)

    b = sub.add_parser("build", help="raw judgments -> data/cases.jsonl")
    b.add_argument("--min-score", type=float, default=6.0, help="TSL relevance threshold")
    b.set_defaults(func=cmd_build)

    sub.add_parser("stats", help="dataset statistics").set_defaults(func=cmd_stats)

    s = sub.add_parser("sample", help="export a stratified sample for expert labelling")
    s.add_argument("-n", type=int, default=50)
    s.add_argument("--seed", type=int, default=13)
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_sample)

    sub.add_parser("check-labels", help="compare rules with expert labels").set_defaults(
        func=cmd_check_labels
    )

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
