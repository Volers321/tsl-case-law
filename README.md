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

## Results so far

### Dataset

1492 judgments downloaded from SAOS (2015–2026) → **1080 civil/commercial cases** after processing:

| Step | Dropped | Why |
|---|---:|---|
| Case register (repertorium) not civil | 228 | TSL words also appear in criminal (K, Ka), labour (P, Pa), social-insurance (U, AUa) and consumer-clause (AmC) cases |
| Keyword relevance score < 6 | 131 | |
| No reasons, or start of the court's assessment not found | 53 | without the cut, "facts" would contain the court's reasoning — **outcome leakage** |

The first version of the pipeline kept 1305 cases, 24% with an unknown outcome and a 46 000-word "facts" section (a criminal trial). Diagnostics on real data (`tsl inspect`) drove each fix above.

### Measured against a TSL practitioner

30 cases, stratified by outcome, labelled by me (10 years running a transport & forwarding company) with `tsl label`, a one-case-at-a-time terminal tool.

| What | Result |
|---|---|
| Relevance — keyword filter (score ≥ 6) | 20/30 TSL (precision 67%) |
| Relevance — LLM (`claude-haiku-5-5`, facts only) | 30/30 agree ⚠️ optimistic, see below |
| Outcome — regex rules on the sentencja | 16/18 |
| Outcome — LLM on the sentencja | 17/18 |
| Dispute type — LLM | 16/20 |

**What went wrong, and what it taught me**

- *My own first labels were wrong.* The first pass said only 11/30 cases were TSL. The LLM disagreed on 13; reviewing them side by side (`tsl label --disputed`) I changed all 13 — 11 TSL cases I had marked "no" (e.g. unpaid freight for six container runs, which I had rushed past) and 2 passenger-ticket cases I had marked "yes". Expert labels need adjudication too.
- *"Prawo przewozowe" covers passengers.* The main false positives are fare-evasion suits and passenger injuries. The LLM prompt now defines TSL as carriage of **goods**, forwarding and carrier/forwarder liability insurance.
- *Why the 30/30 is optimistic.* The prompt definition was refined after seeing these labels, and disagreements were adjudicated with the LLM summary on screen (anchoring). An unbiased estimate needs a fresh batch labelled blind (`tsl sample --add 20`, reported per batch by `tsl check-labels`); with n=30 any figure has a wide confidence interval anyway.
- *Gold labels are frozen.* A confirmed outcome stores the label the expert saw, so later rule changes cannot silently rewrite the gold set.

## Status

- [x] SAOS client: topical searches, full-judgment download, resumable, retries, polite rate limiting
- [x] Processing: civil-register filter, section split with leakage guard, rule-based outcomes (cost orders ignored, outcome from reasons when the sentencja is missing), amounts
- [x] Expert labelling (`tsl label`, `--disputed`, batches) and scoring (`tsl check-labels`)
- [x] LLM extraction (prompt-versioned cache, parallel): relevance, party roles, dispute type, claim, defences, neutral summary — from facts only; outcome from the sentencja only
- [x] Pre-trial conduct features (demand, acknowledgement, partial payment, settlement attempt, timely complaint, delivery accepted without reservations) — `tsl conduct-stats` with Wilson intervals
- [x] Similar-case retrieval (TF-IDF) and four outcome predictors on a temporal split, with bootstrap CIs and Brier score — `tsl predict-eval`
- [x] `tsl predict "<facts>"`: probabilities, reasoning and the cited precedents
- [ ] Results on the full dataset (needs `tsl extract` over all cases)
- [ ] Hybrid retrieval + reranking from rag-eval-pl; MCP server; demo

## Quickstart

```bash
git clone https://github.com/Volers321/tsl-case-law && cd tsl-case-law
python3 -m venv .venv && source .venv/bin/activate
make install
tsl download --from 2015-01-01      # searches SAOS, then fetches full judgments (resumable)
tsl build                           # -> data/cases.jsonl
tsl stats
tsl inspect                         # diagnostics of the split and outcome rules
tsl sample -n 30                    # -> data/labels/sample.csv
tsl label                           # expert review, one case at a time (Enter = yes)
cp .env.example .env                # add ANTHROPIC_API_KEY
tsl extract --ids-from-labels       # LLM features for the labelled cases
tsl check-labels                    # now also LLM vs. expert
tsl extract                         # all cases (parallel, cached)
tsl predict-eval                    # majority · tfidf · llm · llm-rag on the newest 20%
tsl conduct-stats                   # does paying part of the debt predict losing the rest?
tsl predict "Przewoźnik żąda frachtu 4 500 zł, zleceniodawca potrącił karę za opóźnienie…"
```

## Data

Judgments come from [SAOS](https://www.saos.org.pl) (System Analizy Orzeczeń Sądowych), which publishes anonymised rulings of Polish common courts and the Supreme Court. Court rulings are not subject to copyright in Poland (art. 4 of the Copyright Act). Raw downloads stay out of git and are re-created with `tsl download`.

## License

MIT
