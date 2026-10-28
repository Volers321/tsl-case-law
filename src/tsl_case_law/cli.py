"""Command line: ``tsl <command>``.

Typical flow:
    tsl download --from 2015-01-01     # search SAOS and fetch full judgments
    tsl build                          # raw JSON -> data/cases.jsonl (+ filters)
    tsl stats                          # what we've got
    tsl sample -n 50                   # CSV for the expert to label
    tsl label                          # expert reviews the sample, one case at a time
    tsl check-labels                   # how good are the rules vs. the expert?
    tsl extract --ids-from-labels      # LLM features for the labelled cases (needs API key)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .labeling import export_for_labeling, label_interactively, read_labels, stratified_sample
from .processing import CIVIL_REGISTERS, to_record

RAW = Path("data/raw")
CASES = Path("data/cases.jsonl")
IDS = RAW / "_ids.json"
LABEL_SHEET = Path("data/labels/sample.csv")
EXTRACTED = Path("data/extracted.jsonl")
NOTES = Path("data/commentary.jsonl")
APPEALS = Path("data/appeals.json")


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
    # One case can be published twice (e.g. two SENTENCE documents); keep the fuller one.
    by_number: dict[str, dict] = {}
    for r in kept:
        key = r["case_number"] or f"id:{r['id']}"
        if key not in by_number or len(r["facts"]) > len(by_number[key]["facts"]):
            if key in by_number:
                dropped["duplicate case number"] += 1
            by_number[key] = r
        else:
            dropped["duplicate case number"] += 1
    kept = sorted(by_number.values(), key=lambda r: r["date"])
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
    if args.add:
        if not LABEL_SHEET.exists():
            raise SystemExit(f"{LABEL_SHEET} not found — run `tsl sample` first.")
        labels = read_labels(LABEL_SHEET)
        from .labeling import _read_sheet

        _, rows = _read_sheet(LABEL_SHEET)
        seen = {int(r["id"]) for r in rows}
        batch = max([lab["batch"] for lab in labels.values()] or [1]) + 1
        fresh = [c for c in cases if c["id"] not in seen]
        sample = stratified_sample(fresh, args.add, seed=args.seed + batch)
        export_for_labeling(sample, LABEL_SHEET, append=True, batch=batch)
        print(f"Added {len(sample)} new cases as batch {batch} to {LABEL_SHEET}. Now: tsl label")
        return
    if LABEL_SHEET.exists() and not args.force:
        raise SystemExit(f"{LABEL_SHEET} exists — it may contain your labels. Use --force.")
    sample = stratified_sample(cases, args.n, seed=args.seed)
    export_for_labeling(sample, LABEL_SHEET)
    print(f"Wrote {len(sample)} cases to {LABEL_SHEET}. Now: tsl label")


YES = {"tak", "t", "yes", "y", "1", "true"}


def _gold_outcome(label: dict, case: dict) -> str | None:
    """Expert's outcome: the rule's label if they confirmed it, else what they wrote."""
    if label["outcome_rule_ok"].lower() in YES:
        return label.get("shown_outcome") or case.get("outcome_rule")
    return label["outcome_gold"].upper() or None


def _acc(pairs: list[tuple]) -> str:
    ok = sum(a == b for a, b in pairs)
    return f"{ok}/{len(pairs)} ({ok / len(pairs):.0%})" if pairs else "—"


def cmd_label(args) -> None:
    if not LABEL_SHEET.exists():
        raise SystemExit(f"{LABEL_SHEET} not found — run `tsl sample` first.")
    review = None
    if args.disputed:
        if not EXTRACTED.exists():
            raise SystemExit(f"{EXTRACTED} not found — run `tsl extract --ids-from-labels`.")
        llm = {r["id"]: r for r in map(json.loads, EXTRACTED.read_text("utf-8").splitlines())}
        labels = read_labels(LABEL_SHEET)
        review = {}
        for i, lab in labels.items():
            f = llm.get(i, {}).get("facts_llm", {})
            if f.get("ok") and f.get("is_tsl") != (lab["is_tsl"].lower() in YES):
                verdict = "TSL" if f.get("is_tsl") else "nie TSL"
                review[i] = f"[LLM uważa: {verdict}] {_snip(f.get('summary', ''), 600)}"
        if not review:
            raise SystemExit("No disagreements between the LLM and your labels.")
    label_interactively(LABEL_SHEET, review=review)


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
    undecidable = sum(g == "UNKNOWN" for g in gold.values())
    # "UNKNOWN" from the expert means the excerpt does not show the outcome — such cases
    # say nothing about whether a rule or the LLM is right, so they are left out.
    gold = {i: g for i, g in gold.items() if g and g != "UNKNOWN"}
    print(f"\noutcome accuracy vs expert ({undecidable} undecidable from the excerpt, skipped):")
    print(
        f"  rules: {_acc([(cases[i]['outcome_rule'], g) for i, g in gold.items() if i in cases])}"
    )
    for src in ("sentencja", "reasons"):
        pairs = [
            (cases[i]["outcome_rule"], g)
            for i, g in gold.items()
            if i in cases and cases[i].get("outcome_source") == src
        ]
        print(f"    from {src:9s} {_acc(pairs)}")
    if llm:
        pairs = [(llm[i]["outcome_llm"].get("outcome"), g) for i, g in gold.items() if i in llm]
        print(f"  LLM:   {_acc(pairs)}")
        rel = [(llm[i]["facts_llm"].get("is_tsl", False), i in tsl) for i in labels if i in llm]
        print(f"\nLLM relevance (is_tsl) agreement: {_acc(rel)}")
        batches = sorted({lab["batch"] for lab in labels.values()})
        if len(batches) > 1:
            for b in batches:
                part = [
                    (llm[i]["facts_llm"].get("is_tsl", False), i in tsl)
                    for i in labels
                    if i in llm and labels[i]["batch"] == b
                ]
                note = " (prompt tuned on it, adjudicated)" if b == 1 else " (unseen: honest)"
                print(f"  batch {b}: {_acc(part)}{note}")
        tp = sum(p and g for p, g in rel)
        fp = sum(p and not g for p, g in rel)
        fn = sum(g and not p for p, g in rel)
        print(f"  LLM said TSL: {tp + fp} (correct {tp}); missed TSL cases: {fn}")
        print(f"  precision {tp / max(tp + fp, 1):.0%}, recall {tp / max(tp + fn, 1):.0%}")
        disagree = [
            i for i in labels if i in llm and llm[i]["facts_llm"].get("is_tsl") != (i in tsl)
        ]
        if disagree:
            print("\nLLM vs expert on relevance (LLM summary of the facts):")
        for i in disagree:
            f = llm[i]["facts_llm"]
            who = "LLM: TSL, expert: no" if f.get("is_tsl") else "LLM: no, expert: TSL"
            num = cases.get(i, {}).get("case_number", i)
            print(f"  {num} [{who}] {f.get('dispute_type')}: {_snip(f.get('summary', ''), 260)}")
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

    # Why does the keyword filter let non-TSL cases through? Score and topic of each.
    def score(i):
        return cases.get(i, {}).get("tsl_score", 0)

    yes = sorted(score(i) for i in tsl)
    no = sorted(score(i) for i in labels if i not in tsl)
    if yes and no:
        print(f"\nTSL score — relevant: {yes}")
        print(f"TSL score — not TSL:  {no}")
        print("keyword threshold:  precision / recall on the labelled cases")
        for t in (6, 10, 15, 20, 25, 30):
            kept_yes = sum(x >= t for x in yes)
            kept_no = sum(x >= t for x in no)
            prec = kept_yes / max(kept_yes + kept_no, 1)
            print(f"  >= {t:2d}: {prec:4.0%} / {kept_yes / len(yes):4.0%}")
    not_tsl = [i for i in labels if i not in tsl and i in cases]
    if not_tsl:
        print("\nnot TSL according to the expert:")
    for i in not_tsl:
        c = cases[i]
        regs = "; ".join(r[:40] for r in c.get("regulations", [])[:2])
        print(
            f"  {c['case_number']} (score {c.get('tsl_score', 0):.0f}) {_snip(c.get('facts', ''), 160)}"
        )
        if regs:
            print(f"      regulations: {regs}")


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
    rows = extract_all(llm, cases, workers=args.workers)
    EXTRACTED.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    failed = sum(not (r["facts_llm"]["ok"] and r["outcome_llm"]["ok"]) for r in rows)
    print(f"Wrote {len(rows)} rows to {EXTRACTED}; {failed} with unparseable answers")


def _load_extracted() -> dict[int, dict]:
    if not EXTRACTED.exists():
        return {}
    rows = map(json.loads, EXTRACTED.read_text(encoding="utf-8").splitlines())
    return {r["id"]: r for r in rows}


def _load_notes() -> dict[int, dict]:
    if not NOTES.exists():
        return {}
    rows = map(json.loads, NOTES.read_text(encoding="utf-8").splitlines())
    return {r["id"]: r for r in rows}


def _load_links() -> dict[int, dict]:
    if not APPEALS.exists():
        return {}
    return {int(k): v for k, v in json.loads(APPEALS.read_text(encoding="utf-8")).items()}


def _examples(keep_reversed: bool = False):
    from .predict import build_examples

    extracted = _load_extracted()
    links = _load_links()
    cases = load_cases()
    examples = build_examples(
        cases, extracted, _load_notes(), links, drop_reversed=not keep_reversed
    )
    note = "LLM relevance filter" if extracted else "keyword filter only — run `tsl extract`"
    if links:
        everything = build_examples(cases, extracted, None, links, drop_reversed=False)
        dropped = len(everything) - len(examples)
        note += f"; {dropped} changed/set aside on appeal excluded" if dropped else ""
    else:
        note += "; appeals not linked — run `tsl appeals`"
    print(f"{len(examples)} first-instance TSL cases with a known outcome ({note})")
    return examples


REPORT = Path("data/predict_report.json")


def cmd_predict_eval(args) -> None:
    from . import predict as P

    examples = _examples(keep_reversed=args.keep_reversed)
    train, test = P.temporal_split(examples, args.test_frac)
    if args.dev:
        # Development runs never touch the test period: the newest part of the training
        # data plays the role of the test set. Tune here, report the test set once.
        train, test = P.temporal_split(train, args.test_frac)
        print("DEV MODE: validation split inside the training period")
    print(
        f"train {len(train)} ({train[0].date} … {train[-1].date})   "
        f"test {len(test)} ({test[0].date} … {test[-1].date})"
    )
    print(
        "test labels: " + ", ".join(f"{k}:{v}" for k, v in Counter(e.label for e in test).items())
    )
    gold = [e.label for e in test]
    methods = args.methods.split(",")
    results, llm, spread = {}, None, {}
    for m in methods:
        if m == "majority":
            probs = P.predict_majority(train, test)
        elif m == "tfidf":
            probs = P.predict_tfidf(train, test)
        elif m in ("llm", "llm-rag", "llm-rag-issue"):
            from .llm import LLM

            if m == "llm-rag-issue" and not any(e.note for e in train):
                print("  skip llm-rag-issue: no commentary yet — run `tsl annotate`")
                continue
            llm = llm or LLM.from_env()
            runs = [
                P.predict_llm(
                    llm,
                    train,
                    test,
                    k=0 if m == "llm" else args.k,
                    by_issue=m == "llm-rag-issue",
                    repeat=rep,
                )
                for rep in range(args.repeats)
            ]
            if len(runs) > 1:
                per_run = [P.evaluate(gold, r)["accuracy"] for r in runs]
                spread[m] = (min(per_run), max(per_run), sum(per_run) / len(per_run))
            probs = P.average_probs(runs)
        else:
            raise SystemExit(f"unknown method {m!r}")
        r = P.evaluate(gold, probs) | {"accuracy_ci95": P.bootstrap_ci(gold, probs)}
        if m in spread:
            r["runs_accuracy_min_max_mean"] = spread[m]
        results[m] = r
    print("\n3 classes (win / partial / lose)            | claimant gets anything?")
    print(
        f"{'method':10s} {'accuracy':>8s} {'95% CI':>13s} {'F1':>6s} {'Brier':>6s} | {'acc':>5s} {'Brier':>6s}"
    )
    for m, r in results.items():
        lo, hi = r["accuracy_ci95"]
        print(
            f"{m:10s} {r['accuracy']:8.3f}  [{lo:.2f}, {hi:.2f}] {r['macro_f1']:6.3f} "
            f"{r['brier']:6.3f} | {r['any_win_accuracy']:5.3f} {r['any_win_brier']:6.3f}"
        )
    print("Brier: 0 = perfect probabilities, lower is better; majority is the bar to beat.")
    if spread:
        print(f"\nLLM rows above average {args.repeats} runs. Accuracy of single runs:")
        for m, (lo, hi, mean) in spread.items():
            print(f"  {m:14s} min {lo:.3f}  max {hi:.3f}  mean {mean:.3f}  (spread {hi - lo:.3f})")
        print("A difference between methods smaller than the spread is noise.")
    if args.dev:
        return  # dev results are for tuning only; the report holds test results
    REPORT.write_text(
        json.dumps(
            {
                "n_train": len(train),
                "n_test": len(test),
                "split_date": test[0].date,
                "results": results,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"report -> {REPORT}")


def cmd_similar(args) -> None:
    from .predict import LABEL_PL, SimilarCases

    hits = SimilarCases(_examples()).search(args.text, args.k)
    for e, score in hits:
        mark = "  ✓ utrzymany w apelacji" if e.appeal == "upheld" else ""
        print(f"\n{score:.2f}  {e.case_number} ({e.date})  -> {LABEL_PL[e.label]}{mark}")
        print(f"      {_snip(e.summary, 300)}")
        if e.note:
            print(f"      Teza: {e.note['teza']}")
            print(f"      Wskazówka: {e.note['wskazowka']}")


def cmd_predict(args) -> None:
    from . import predict as P
    from .extract import extract_facts
    from .llm import LLM

    if args.file:
        args.text = Path(args.file).read_text(encoding="utf-8")
    if not (args.text or "").strip():
        raise SystemExit('Describe the dispute: tsl predict "…" or tsl predict -f casus.txt')
    examples = _examples()
    llm = LLM.from_env()
    # Describe the new case the way the database describes past ones (neutral summary,
    # dispute type, defences, conduct), so retrieval compares like with like.
    facts = extract_facts(llm, {"facts": args.text})
    if facts.get("ok") and not facts.get("is_tsl"):
        print("⚠ The model does not see a freight/forwarding dispute here — results may not fit.")
    summary = facts.get("summary") or args.text
    features = {k: facts.get(k) for k in ("dispute_type", "defences", "conduct")}
    by_issue = any(e.note for e in examples)
    query = " ".join(
        [summary, str(features.get("dispute_type") or ""), *(features.get("defences") or [])]
    )
    hits = P.SimilarCases(examples, by_issue=by_issue).search(query, args.k)
    raw = llm.complete(
        P.PREDICT_SYSTEM,
        P.precedents_prompt(hits, with_notes=by_issue) + P.case_prompt(args.text, features),
        max_tokens=2048,
    )
    p = P.parse_probs(raw)
    if not p:
        raise SystemExit("Could not parse the model's answer:\n" + raw)
    memo = None
    for _ in range(2):
        memo = P.parse_memo(
            llm.complete(P.MEMO_SYSTEM, P.memo_prompt(args.text, p, hits), max_tokens=4096),
            len(hits),
        )
        if memo:
            break
    if not memo:
        raise SystemExit("Could not build the memo — run the command again.")
    text = P.render_memo(memo, p, hits)
    print("\n" + text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\nZapisano ocenę do pliku {args.out}")


def cmd_conduct_stats(args) -> None:
    from .predict import conduct_table, wilson

    examples = _examples()
    if not any(e.features.get("conduct") for e in examples):
        raise SystemExit("No conduct features — run `tsl extract` with the current prompt.")
    base = sum(e.label != "lose" for e in examples) / len(examples)
    print(f"claimant wins (fully or partly) in {base:.0%} of {len(examples)} cases\n")
    print(
        f"{'pre-trial conduct':32s} {'n':>4s} {'claimant wins':>14s} {'95% CI':>13s} {'vs all':>7s}"
    )
    for r in conduct_table(examples, min_n=args.min_n):
        lo, hi = wilson(r["claimant_wins"], r["n"])
        name = f"{r['feature']} = {'yes' if r['value'] else 'no'}"
        print(
            f"{name:32s} {r['n']:4d} {r['claimant_wins']:13.0%}  [{lo:.0%}, {hi:.0%}] "
            f"{r['vs_base']:+7.0%}"
        )
    print("\nCorrelation, not causation; groups overlap; facts as described by the court.")


def cmd_annotate(args) -> None:
    from .commentary import ANNOTATE_VERSION, annotate_all
    from .llm import LLM

    extracted = _load_extracted()
    if not extracted:
        raise SystemExit("Run `tsl extract` first — notes are written for TSL cases only.")
    cases = [
        c
        for c in load_cases()
        if (extracted.get(c["id"], {}).get("facts_llm") or {}).get("is_tsl") and c["assessment"]
    ]
    cases = cases[: args.limit] if args.limit else cases
    llm = LLM.from_env()
    print(f"Annotating {len(cases)} TSL judgments with {llm.model} (prompt {ANNOTATE_VERSION})")
    notes = annotate_all(llm, cases, workers=args.workers)
    NOTES.write_text("".join(json.dumps(n, ensure_ascii=False) + "\n" for n in notes), "utf-8")
    ok = [n for n in notes if n.get("ok")]
    flags = [g for n in ok for g in n.get("przepisy_grounded", [])]
    print(f"Wrote {len(notes)} notes to {NOTES}; {len(notes) - len(ok)} failed")
    if flags:
        print(
            f"provisions found in the judgment text: {sum(flags)}/{len(flags)} "
            f"({sum(flags) / len(flags):.0%}) — the rest may be paraphrased or invented"
        )


def cmd_commentary(args) -> None:
    from .commentary import NoteIndex, overview
    from .llm import LLM

    notes = list(_load_notes().values())
    if not notes:
        raise SystemExit("No commentary yet — run `tsl annotate` first.")
    cases = {c["id"]: c for c in load_cases()}
    index = NoteIndex(cases, notes, _load_links())
    result = overview(LLM.from_env(), args.issue, index, k=args.n)
    if not result["hits"]:
        raise SystemExit("No judgments found for this issue — try other words.")
    print(f"\nPRZEGLĄD ORZECZNICTWA: {args.issue}\n")
    print(result["text"])
    print("\nWyroki uwzględnione:")
    for case, note, _score in result["hits"]:
        print(
            f"  {case['case_number']} ({case['date'][:10]}, {case.get('court', '')}) — {_snip(note['teza'], 140)}"
        )
    chk = result["check"]
    print(
        f"\nKontrola cytowań: {chk['valid']}/{chk['citations']} sygnatur z listy, "
        f"{chk['distinct_cases']} różnych wyroków, twierdzenia z cytowaniem {chk['claims_with_citation']}"
    )
    if chk["invalid"]:
        print(f"  ⚠ sygnatury spoza listy (możliwe halucynacje): {', '.join(chk['invalid'])}")
    print("To nie jest porada prawna; komentarz powstał automatycznie z treści wyroków.")


def cmd_appeals(args) -> None:
    from .appeals import REVERSED, appeal_status, link_appeals, reversal_stats
    from .predict import gold_outcome

    cases = load_cases()
    extracted = _load_extracted()
    links = link_appeals(cases, outcome_of=lambda c: gold_outcome(c, extracted.get(c["id"])))
    APPEALS.write_text(json.dumps(links, ensure_ascii=False, indent=1), encoding="utf-8")
    n_appeals = sum(c["instance"] == "appeal" for c in cases)
    print(f"{n_appeals} appellate judgments; {len(links)} linked to a first-instance case here")
    print(f"links -> {APPEALS}")

    def is_tsl(c):
        return (extracted.get(c["id"], {}).get("facts_llm") or {}).get("is_tsl", True)

    tsl_cases = [c for c in cases if is_tsl(c)]

    def level(c):
        return "Sąd Rejonowy" if "Rejonow" in c.get("court", "") else "Sąd Okręgowy"

    def dispute(c):
        return (extracted.get(c["id"], {}).get("facts_llm") or {}).get("dispute_type") or "—"

    header = (
        f"  {'group':28s} {'appealed':>8s} {'upheld':>7s} {'changed':>8s} "
        f"{'set aside':>10s} {'reversed':>9s}"
    )
    for title, key in (("all TSL", None), ("by court", level), ("by dispute type", dispute)):
        print(f"\n{title}:\n{header}")
        stats = reversal_stats(tsl_cases, links, key)
        for g, r in sorted(stats.items(), key=lambda kv: -kv[1]["appealed"]):
            if r["appealed"] < (1 if key is None else args.min_n):
                continue
            rate = f"{r['reversal_rate']:.0%}" if r["reversal_rate"] is not None else "—"
            print(
                f"  {g:28s} {r['appealed']:8d} {r.get('upheld', 0):7d} {r.get('changed', 0):8d} "
                f"{r.get('set_aside', 0):10d} {rate:>9s}"
            )
    print(
        "\n'not_found' does not mean final: many appeals are not published in SAOS, so the\n"
        "reversal rate is measured on the appealed cases we can see, not on all judgments."
    )
    # The direct measure: every appellate judgment says whether the ruling below stood.
    appeals = [c for c in tsl_cases if c["instance"] == "appeal"]
    verdicts = Counter(gold_outcome(c, extracted.get(c["id"])) for c in appeals)
    known = verdicts["APPEAL_DISMISSED"] + verdicts["CHANGED"] + verdicts["SET_ASIDE"]
    if known:
        from .predict import wilson

        rev = (verdicts["CHANGED"] + verdicts["SET_ASIDE"]) / known
        lo, hi = wilson(rev, known)
        print(
            f"\nALL {len(appeals)} TSL appellate judgments (no linking needed): {known} with a "
            f"known outcome — upheld {verdicts['APPEAL_DISMISSED']}, changed "
            f"{verdicts['CHANGED']}, set aside {verdicts['SET_ASIDE']}"
        )
        print(
            f"  -> the ruling below was changed or set aside in {rev:.0%} [95% CI {lo:.0%}–{hi:.0%}]"
        )
        by_type: dict[str, Counter] = {}
        for c in appeals:
            by_type.setdefault(dispute(c), Counter())[gold_outcome(c, extracted.get(c["id"]))] += 1
        print(f"  {'dispute type':28s} {'appeals':>7s} {'reversed':>9s}")
        for t, cnt in sorted(by_type.items(), key=lambda kv: -sum(kv[1].values())):
            n = cnt["APPEAL_DISMISSED"] + cnt["CHANGED"] + cnt["SET_ASIDE"]
            if n >= args.min_n:
                print(f"  {t:28s} {n:7d} {(cnt['CHANGED'] + cnt['SET_ASIDE']) / n:9.0%}")
        print("  (only appealed judgments: a reversal rate among appeals, not among all rulings)")

    linked = {link["appeal_id"] for link in links.values()}
    matched = sum(c["id"] in linked for c in appeals)
    print(
        f"\nlinking: {matched} of {len(appeals)} TSL appeals matched a first-instance judgment "
        "in this dataset (SAOS rarely publishes both judgments of one case)"
    )
    reversed_ = [c for c in tsl_cases if appeal_status(c["id"], links) in REVERSED]
    if reversed_:
        print("\nexamples of reversed judgments (excluded from prediction and precedents):")
    for c in reversed_[:5]:
        link = links[c["id"]]
        print(
            f"  {c['case_number']} ({c['court']}) -> {link['status']} by {link['appeal_case_number']}"
        )


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
    s.add_argument("--add", type=int, metavar="N", help="append N new cases as a new batch")
    s.set_defaults(func=cmd_sample)

    lab = sub.add_parser("label", help="review the sample case by case (Enter = yes)")
    lab.add_argument(
        "--disputed", action="store_true", help="re-review only cases where the LLM disagrees"
    )
    lab.set_defaults(func=cmd_label)
    sub.add_parser("check-labels", help="compare rules with expert labels").set_defaults(
        func=cmd_check_labels
    )

    e = sub.add_parser("extract", help="LLM: party roles, dispute type, claim, defences, outcome")
    e.add_argument("--limit", type=int, help="only the first N cases (cost control)")
    e.add_argument("--ids-from-labels", action="store_true", help="only cases in the label sheet")
    e.add_argument("--workers", type=int, default=6, help="parallel API calls")
    e.set_defaults(func=cmd_extract)

    pe = sub.add_parser("predict-eval", help="compare outcome predictors on the newest cases")
    pe.add_argument("--methods", default="majority,tfidf,llm,llm-rag,llm-rag-issue")
    pe.add_argument("--test-frac", type=float, default=0.2)
    pe.add_argument("-k", type=int, default=5, help="precedents for llm-rag")
    pe.add_argument("--dev", action="store_true", help="evaluate on a validation split instead")
    pe.add_argument("--repeats", type=int, default=1, help="LLM runs per method (averaged)")
    pe.add_argument(
        "--keep-reversed", action="store_true", help="keep judgments changed/set aside on appeal"
    )
    pe.set_defaults(func=cmd_predict_eval)

    si = sub.add_parser("similar", help="find similar past cases for a description")
    si.add_argument("text")
    si.add_argument("-k", type=int, default=5)
    si.set_defaults(func=cmd_similar)

    pr = sub.add_parser("predict", help="estimate the outcome of a described dispute")
    pr.add_argument("text", nargs="?", help="facts of the dispute, in Polish")
    pr.add_argument("-f", "--file", help="read the facts from a text file instead")
    pr.add_argument("-o", "--out", help="also save the memo as a Markdown file")
    pr.add_argument("-k", type=int, default=5)
    pr.set_defaults(func=cmd_predict)

    cs = sub.add_parser("conduct-stats", help="claimant win rate by pre-trial conduct")
    cs.add_argument("--min-n", type=int, default=8)
    cs.set_defaults(func=cmd_conduct_stats)

    ap = sub.add_parser("appeals", help="link judgments to their appeals; reversal rates")
    ap.add_argument("--min-n", type=int, default=5, help="smallest group to show")
    ap.set_defaults(func=cmd_appeals)

    an = sub.add_parser("annotate", help="LLM commentary note for every TSL judgment")
    an.add_argument("--limit", type=int)
    an.add_argument("--workers", type=int, default=6)
    an.set_defaults(func=cmd_annotate)

    co = sub.add_parser("commentary", help="case-law overview of a legal issue, with citations")
    co.add_argument("issue", help='e.g. "przedawnienie roszczeń z umowy przewozu"')
    co.add_argument("-n", type=int, default=12, help="judgments to draw on")
    co.set_defaults(func=cmd_commentary)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
