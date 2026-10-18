# tsl-case-law

**Case-law assistant for Polish transport & freight-forwarding (TSL) disputes.** Given the facts of a dispute — unpaid freight, damaged cargo, late delivery, a forwarder's liability — it finds similar rulings of Polish courts and estimates the likely outcome, citing the precedents it relied on and how uncertain it is.

🇵🇱 *Asystent orzecznictwa TSL: wyszukuje podobne wyroki polskich sądów (SAOS) i prognozuje wynik sporu przewozowego lub spedycyjnego z cytowanymi precedensami i miarą niepewności.*

![CI](https://github.com/Volers321/tsl-case-law/actions/workflows/ci.yml/badge.svg)

> ⚠️ Decision support for businesses assessing a dispute — **not** legal advice and not a tool for courts. AI used by judicial authorities to research and interpret facts and law is a high-risk category under the EU AI Act; this project deliberately stays on the side of parties estimating their chances.

## Why this project

I have run a transport and forwarding company for 10 years. Disputes with carriers, forwarders and shippers follow recurring patterns, and the first question is always *"will we win this in court?"*. Answering it today means a lawyer reading dozens of judgments. This project tests how far retrieval + LLMs get on that question — **measured against real court outcomes**, not vibes.

## Approach

```
SAOS API ──► download ──► build ───────────────► cases.jsonl
(public court             HTML → text             sentencja | facts | assessment
 judgments)               TSL relevance filter    rule-based outcome label
                          split by section        amount, court, instance
                                   │
                    expert labels (50 cases, by a TSL practitioner)
                                   │
          ┌────────────────────────┼─────────────────────────┐
          ▼                        ▼                         ▼
   similar-case retrieval   outcome prediction        evaluation on held-out,
   (hybrid + rerank)        from FACTS ONLY           later judgments
                            majority · classic ML ·   accuracy · macro-F1 ·
                            LLM · LLM + precedents    calibration
```

**Leakage control is the core design decision.** A judgment's operative part (*sentencja*) and the court's legal assessment (*"Sąd zważył, co następuje…"*) both reveal the outcome. The model only ever sees the **facts** section; the split is tested explicitly. Evaluation uses a **temporal split** — train/retrieve on older rulings, test on the newest — so the model cannot have seen test cases, and precedents always predate the case.

## Status

- [x] SAOS client: topical searches (statutes cited: Prawo przewozowe, CMR Convention; TSL vocabulary), full-judgment download, resumable, polite rate limiting
- [x] Processing: HTML cleanup, TSL relevance score, section split, rule-based outcome labels (cost orders ignored), amounts
- [x] Expert-labelling workflow: stratified sample → spreadsheet → `tsl check-labels`
- [ ] Expert labels for 50 cases + accuracy of rules and relevance filter
- [ ] LLM extraction of structured facts (party roles, dispute type, claim, defence)
- [ ] Similar-case retrieval and outcome prediction baselines
- [ ] Evaluation on held-out judgments; MCP server

## Quickstart

```bash
git clone https://github.com/Volers321/tsl-case-law && cd tsl-case-law
python3 -m venv .venv && source .venv/bin/activate
make install
tsl download --from 2015-01-01      # searches SAOS, then fetches full judgments (resumable)
tsl build                           # -> data/cases.jsonl
tsl stats
tsl sample -n 50                    # -> data/labels/sample.csv for expert review
tsl check-labels                    # rules vs. expert, after filling the sheet
```

## Data

Judgments come from [SAOS](https://www.saos.org.pl) (System Analizy Orzeczeń Sądowych), which publishes anonymised rulings of Polish common courts and the Supreme Court. Court rulings are not subject to copyright in Poland (art. 4 of the Copyright Act). Raw downloads stay out of git and are re-created with `tsl download`.

## License

MIT
