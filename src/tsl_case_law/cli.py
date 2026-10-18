"""Command line: ``tsl <command>``.

Typical flow:
    tsl download --from 2015-01-01     # search SAOS and fetch full judgments
    tsl build                          # raw JSON -> data/cases.jsonl (+ filters)
    tsl stats                          # what we've got
    tsl sample -n 50                   # CSV for the expert to label
    tsl check-labels                   # how good are the rules vs. the expert?
    tsl extract --ids-from-labels      # LLM features for the labelled cases (needs API key)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .labeling import export_for_labeling, read_labels, stratified_sample
from .processing import CIVIL_REGISTERS, to_record

RAW = Path("data/raw")
CASES = Path("data/cases.jsonl")
IDS = RAW / "_ids.json"
LABEL_SHEET = Path("data/labels/sample.csv")
EXTRACTED = Path("data/extracted.jsonl")


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
        found = search_ids(
            client,
            date_from=args.date_from,
            date_to=args.date_to,
            state_path=RAW / "_search_state.json",
        )
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
        if rec.repertorium and rec.repertorium not in CIVIL_REGISTERS:
            dropped[f"not a civil case ({rec.repertorium})"] += 1
        elif not rec.repertorium and rec.facts.lower().count("oskarżon") >= 3:
            dropped["not a civil case (criminal wording)"] += 1
        elif rec.tsl_score < args.min_score:
            dropped["low TSL score"] += 1
        elif not rec.facts:
            dropped["no reasons / facts section"] += 1
        elif not rec.assessment:
            # Without a found start of the court's assessment the "facts" are the whole
            # reasons, outcome included — unusable as prediction input.
            dropped["no assessment cut (leakage risk)"] += 1
        else:
            kept.append(rec.to_dict())
    kept.sort(key=lambda r: r["date"])
    CASES.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8"
    )
    print(f"Kept {len(kept)} of {len(files)} judgments -> {CASES}")
    other = Counter()
    for reason, n in dropped.most_common():
        if reason.startswith("not a civil case (") and n < 10:
            other[reason] += n  # keep the summary short: rare registers in one line
        else:
            print(f"  dropped {n:5d}: {reason}")
    if other:
        regs = ", ".join(r[18:-1] for r in other)
        print(f"  dropped {sum(other.values()):5d}: not a civil case (other: {regs})")


def cmd_stats(args) -> None:
    cases = load_cases()
    print(f"cases: {len(cases)}   dates: {cases[0]['date']} … {cases[-1]['date']}")
    for key in ("court_type", "instance", "outcome_rule", "outcome_source"):
        counts = Counter(c[key] for c in cases)
        print(f"\n{key}:")
        for value, n in counts.most_common():
            print(f"  {value or '—':20s} {n:5d}  {n / len(cases):6.1%}")
    years = Counter(c["date"][:4] for c in cases)
    print("\nper year: " + "  ".join(f"{y}:{n}" for y, n in sorted(years.items())))
    facts_len = sorted(len(c["facts"].split()) for c in cases)
    print(f"facts length (words): median {facts_len[len(facts_len) // 2]}, max {facts_len[-1]}")


def _snip(text: str, n: int, tail: bool = False) -> str:
    flat = " ".join(text.split())
    return ("…" + flat[-n:]) if tail and len(flat) > n else flat[:n]


def cmd_inspect(args) -> None:
    """Diagnostics for the processing rules on real data (compact, paste-able output)."""
    cases = load_cases()
    print(f"cases: {len(cases)}")

    print("\noutcome_rule by judgment_type:")
    by_type = Counter((c["judgment_type"], c["outcome_rule"]) for c in cases)
    for (jt, out), n in sorted(by_type.items()):
        print(f"  {jt or '—':10s} {out:18s} {n:5d}")

    empty_sent = [c for c in cases if not c["sentencja"].strip()]
    no_assess = [c for c in cases if not c["assessment"].strip()]
    print(f"\nempty sentencja:   {len(empty_sent)}")
    print(f"no assessment cut: {len(no_assess)}  (facts = whole reasons -> possible leakage)")

    numbers = Counter(c["case_number"] for c in cases if c["case_number"])
    dupes = {k: v for k, v in numbers.items() if v > 1}
    print(f"case numbers on >1 record: {len(dupes)} (e.g. SENTENCE + REASONS of one case)")
    for k in list(dupes)[:3]:
        recs = [c for c in cases if c["case_number"] == k]
        print(f"  {k}: " + ", ".join(f"{c['id']}/{c['judgment_type']}" for c in recs))

    print(f"\nUNKNOWN with a sentencja — first {args.n}:")
    unknown = [c for c in cases if c["outcome_rule"] == "UNKNOWN" and c["sentencja"].strip()]
    for c in unknown[: args.n]:  # the end of the sentencja holds the operative points
        print(f"  [{c['id']}] {c['case_number']}: {_snip(c['sentencja'], 350, tail=True)}")

    print(
        "\nrepertorium: "
        + ", ".join(
            f"{r or '—'}:{n}" for r, n in Counter(c["repertorium"] for c in cases).most_common()
        )
    )
    print(f"\noutcome from reasons — first {min(args.n, 5)} (check these are right):")
    from_reasons = [c for c in cases if c["outcome_source"] == "reasons"]
    for c in from_reasons[: min(args.n, 5)]:
        print(f"  [{c['id']}] {c['outcome_rule']}: {_snip(c['assessment'], 250, tail=True)}")

    print(f"\nstart of text when sentencja is empty — first {min(args.n, 4)}:")
    for c in empty_sent[: min(args.n, 4)]:
        print(f"  [{c['id']}] {_snip(c['facts'], 200)}")

    print(f"\nend of facts when no assessment cut — first {args.n}:")
    for c in no_assess[: args.n]:
        print(f"  [{c['id']}] {_snip(c['facts'], 200, tail=True)}")

    longest = sorted(cases, key=lambda c: -len(c["facts"].split()))[:3]
    print("\nlongest facts (words) and the 150 chars before the assessment cut:")
    for c in longest:
        print(f"  [{c['id']}] {len(c['facts'].split())} w: {_snip(c['facts'], 150, tail=True)}")
        print(f"      assessment starts: {_snip(c['assessment'], 100) or '—'}")


def cmd_sample(args) -> None:
    cases = load_cases()
    if LABEL_SHEET.exists() and not args.force:
        raise SystemExit(f"{LABEL_SHEET} exists — it may contain your labels. Use --force.")
    sample = stratified_sample(cases, args.n, seed=args.seed)
    export_for_labeling(sample, LABEL_SHEET)
    print(f"Wrote {len(sample)} cases to {LABEL_SHEET}")
    print("Open it in Numbers/Excel, fill the last 5 columns, save as CSV (same name).")


YES = {"tak", "t", "yes", "y", "1", "true"}


def _gold_outcome(label: dict, case: dict) -> str | None:
    """Expert's outcome: the rule's label if they confirmed it, else what they wrote."""
    if label["outcome_rule_ok"].lower() in YES:
        return case.get("outcome_rule")
    return label["outcome_gold"].upper() or None


def _acc(pairs: list[tuple]) -> str:
    ok = sum(a == b for a, b in pairs)
    return f"{ok}/{len(pairs)} ({ok / len(pairs):.0%})" if pairs else "—"


def cmd_check_labels(args) -> None:
    labels = read_labels(LABEL_SHEET)
    if not labels:
        raise SystemExit(f"No labelled rows in {LABEL_SHEET} yet.")
    cases = {c["id"]: c for c in load_cases()}
    llm = {}
    if EXTRACTED.exists():
        for line in EXTRACTED.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            llm[r["id"]] = r

    tsl = [i for i, lab in labels.items() if lab["is_tsl"].lower() in YES]
    print(f"labelled: {len(labels)}   relevant to TSL: {len(tsl)} ({len(tsl) / len(labels):.0%})")
    print("  -> precision of the keyword relevance filter (`tsl build --min-score`)")

    gold = {i: _gold_outcome(labels[i], cases.get(i, {})) for i in tsl}
    gold = {i: g for i, g in gold.items() if g}
    print("\noutcome accuracy vs expert:")
    print(
        f"  rules: {_acc([(cases[i]['outcome_rule'], g) for i, g in gold.items() if i in cases])}"
    )
    if llm:
        pairs = [(llm[i]["outcome_llm"].get("outcome"), g) for i, g in gold.items() if i in llm]
        print(f"  LLM:   {_acc(pairs)}")
        rel = [(llm[i]["facts_llm"].get("is_tsl", False), i in tsl) for i in labels if i in llm]
        print(f"\nLLM relevance (is_tsl) agreement: {_acc(rel)}")
        typed = [
            (llm[i]["facts_llm"].get("dispute_type"), labels[i]["dispute_type"])
            for i in tsl
            if i in llm and labels[i]["dispute_type"]
        ]
        print(f"LLM dispute type agreement:       {_acc(typed)}")

    wrong = [i for i, g in gold.items() if i in cases and cases[i]["outcome_rule"] != g]
    if wrong:
        print("\nrule errors:")
    for i in wrong:
        c = cases[i]
        print(
            f"  ✗ {c['case_number']}: rule={c['outcome_rule']} gold={gold[i]}"
            + (f" llm={llm[i]['outcome_llm'].get('outcome')}" if i in llm else "")
        )
    types = Counter(labels[i]["dispute_type"] or "—" for i in tsl)
    print("\ndispute types: " + ", ".join(f"{t}:{n}" for t, n in types.most_common()))


def cmd_extract(args) -> None:
    from .extract import PROMPT_VERSION, extract_all
    from .llm import LLM

    cases = load_cases()
    if args.ids_from_labels:
        wanted = set(read_labels(LABEL_SHEET))
        cases = [c for c in cases if c["id"] in wanted]
    cases = cases[: args.limit] if args.limit else cases
    llm = LLM.from_env()
    print(
        f"Extracting {len(cases)} cases with {llm.provider}/{llm.model} (prompt {PROMPT_VERSION})"
    )
    rows = extract_all(llm, cases)
    EXTRACTED.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    failed = sum(not (r["facts_llm"]["ok"] and r["outcome_llm"]["ok"]) for r in rows)
    print(f"Wrote {len(rows)} rows to {EXTRACTED}; {failed} with unparseable answers")


def main(argv: list[str] | None = None) -> None:
    # Show progress immediately even when piped through `tee` (stdout is block-buffered then).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
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

    i = sub.add_parser("inspect", help="check section split and outcome rules on real data")
    i.add_argument("-n", type=int, default=8, help="examples per diagnostic")
    i.set_defaults(func=cmd_inspect)

    s = sub.add_parser("sample", help="export a stratified sample for expert labelling")
    s.add_argument("-n", type=int, default=50)
    s.add_argument("--seed", type=int, default=13)
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_sample)

    sub.add_parser("check-labels", help="compare rules with expert labels").set_defaults(
        func=cmd_check_labels
    )

    e = sub.add_parser("extract", help="LLM: party roles, dispute type, claim, defences, outcome")
    e.add_argument("--limit", type=int, help="only the first N cases (cost control)")
    e.add_argument("--ids-from-labels", action="store_true", help="only cases in the label sheet")
    e.set_defaults(func=cmd_extract)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
