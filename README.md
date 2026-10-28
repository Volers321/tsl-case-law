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

### Outcome prediction (first instance, temporal split)

365 first-instance cases that the LLM classed as TSL and whose outcome is known. Methods learn from / retrieve among **292 judgments up to April 2021** and are tested on the **73 newest (May 2021 – April 2026)**. The model sees only the facts section.

| Method | Accuracy (3 classes) | 95% CI | Macro-F1 | Brier ↓ |
|---|---:|---:|---:|---:|
| majority class | 0.274 | 0.18–0.37 | 0.143 | 0.724 |
| TF-IDF + logistic regression | **0.411** | 0.30–0.52 | **0.406** | **0.652** |
| LLM (`claude-haiku-5-5`), facts summary | 0.397 | 0.29–0.51 | 0.394 | 0.696 |
| LLM + 5 similar earlier judgments | 0.342 | 0.23–0.44 | 0.339 | 0.703 |

**Reading the table honestly**

- Every method beats the majority baseline, but **the three are statistically indistinguishable**: with 73 test cases the 95% intervals overlap almost entirely. This is a small-data result, not a leaderboard.
- **Distribution shift.** The majority class in 2015–2021 (claimant wins) is not the majority in 2021–2026 (claimant loses, 41% of the test set), which is why "always predict the usual outcome" scores only 27%.
- **Precedents did not help — they hurt slightly.** Likely causes, to test next: TF-IDF similarity matches wording, not legal issue (two freight-payment cases look alike whether the defence is limitation or set-off); precedents come from an older period with a different outcome mix, and the model anchors on them.
- **The LLM is worse calibrated than logistic regression** (Brier 0.696 vs 0.652): its probabilities are more confident than its accuracy justifies.
- Three classes is a hard target: "partial" covers anything from 5% to 95% of the claim.

**LLM run-to-run variance is as large as the gaps between methods.** Excluding two judgments reversed on appeal changed the precedent pool, which forced fresh LLM calls on the *same* 73 test cases: `llm-rag` moved from 0.274 to 0.411 and `llm-rag-issue` from 0.384 to 0.356, while the cached `llm` stayed at 0.356. So single-run comparisons of LLM methods here are noise. `tsl predict-eval --repeats N` now runs every LLM method N times, reports the min / max / mean accuracy of single runs and scores the averaged probabilities (a small ensemble).

All tuning from here on runs on a validation split inside the training period (`tsl predict-eval --dev`); the test set above is reported, not optimised against.

### Pre-trial conduct and outcomes

Share of first-instance cases in which the claimant won at least partly (all cases: 67%, n = 365). Extracted by the LLM from the facts; `null` (not mentioned) is excluded, not counted as "no".

| Conduct (from the facts) | n | Claimant wins | 95% CI (Wilson) |
|---|---:|---:|---:|
| defendant acknowledged the debt | 31 | 81% | 64–91% |
| goods accepted **without** reservations | 71 | 77% | 66–86% |
| goods accepted **with** reservations | 36 | 56% | 40–70% |
| complaint **not** filed in time | 31 | 90% | 75–97% |
| complaint filed in time | 27 | 59% | 41–75% |
| attempt to settle | 42 | 62% | 47–75% |
| partial payment made | 86 | 70% | 59–78% |

The striking "late complaint → claimant wins 90%" is counter-intuitive. A likely explanation (a hypothesis, not yet checked) is party roles: if the claimant is a **carrier suing for freight**, a shipper who did not complain about damage in time cannot set it off. Conduct features only mean something **relative to who is suing whom** — the next version of the table splits by claimant role. Correlation, not causation; small groups, overlapping categories.

### How often appeals succeed

Of **279 TSL appellate judgments** with a known outcome, the court of second instance **changed or set aside the ruling below in 34%** (95 of 279; Wilson 95% CI 29–40%). By dispute type (LLM-classified, groups with ≥ 7 appeals):

| Dispute | Appeals | Changed or set aside |
|---|---:|---:|
| unpaid freight | 107 | 27% |
| cargo loss / damage | 81 | 38% |
| carrier's liability insurance | 51 | 37% |
| forwarder's liability | 13 | 54% |
| other | 12 | 42% |
| late delivery | 8 | 12% |
| contractual penalties | 7 | 43% |

Freight-payment suits, usually documentary, are overturned least; disputes about who was the forwarder and who the carrier most — though with 13 appeals that figure is shaky. This is a rate **among appealed judgments**, not among all rulings, and only for appeals published in SAOS.

Linking each first-instance judgment to its own appeal (by the lower court's case number and city) worked for only 13 of 285 TSL appeals: SAOS rarely publishes both judgments of one case. The 2 first-instance judgments found to be changed are excluded from prediction and precedents; upheld ones are marked as confirmed.

### Case-law commentary

Commentaries and glosses in Lex/Legalis are copyrighted; judgments are not. So `tsl annotate` writes a note for each of the **675 TSL judgments** from the court's own reasoning — holding, key arguments, provisions relied on, a practical takeaway — and `tsl commentary "<issue>"` turns the notes into a case-law overview in which every claim cites a case number.

| Automatic check | Result |
|---|---|
| Provisions in the notes that literally appear in the judgment text | **4208 / 4439 (95%)** |
| Case numbers cited in two overviews ("przedawnienie roszczeń z umowy przewozu", "odpowiedzialność przewoźnika za szkodę w ładunku CMR") that belong to the judgments supplied | **39 / 39**, 0 invented |
| Claims in those overviews backed by at least one citation | **20 / 20** |

*A bug the real output exposed:* the first version of the citation check flagged 9 "hallucinated" case numbers in the first overview. All were genuine — the model had put several in one bracket (`[V GC 388/14; IX GC 1392/16]`) and the checker read the bracket as a single unknown number. The checker had a false-alarm problem, not the model; it now splits brackets and has a regression test built from that output.

The remaining 5% of provisions are not found verbatim — they may be paraphrased or invented; a manual check of a sample is still to do. The per-note flag (`przepisy_grounded`) lets a UI show unverified provisions differently.

## Status

- [x] SAOS client: topical searches, full-judgment download, resumable, retries, polite rate limiting
- [x] Processing: civil-register filter, section split with leakage guard, rule-based outcomes (cost orders ignored, outcome from reasons when the sentencja is missing), amounts
- [x] Expert labelling (`tsl label`, `--disputed`, batches) and scoring (`tsl check-labels`)
- [x] LLM extraction (prompt-versioned cache, parallel): relevance, party roles, dispute type, claim, defences, neutral summary — from facts only; outcome from the sentencja only
- [x] Pre-trial conduct features (demand, acknowledgement, partial payment, settlement attempt, timely complaint, delivery accepted without reservations) — `tsl conduct-stats` with Wilson intervals
- [x] Similar-case retrieval (TF-IDF) and four outcome predictors on a temporal split, with bootstrap CIs and Brier score — `tsl predict-eval`
- [x] `tsl predict "<facts>"`: probabilities, reasoning and the cited precedents
- [x] Results on the full dataset (above)
- [x] Case-law commentary from the judgments themselves: a note per judgment (holding, arguments, provisions, practical takeaway) — `tsl annotate`; issue overviews with case-number citations — `tsl commentary "<zagadnienie>"`; provisions and citations checked automatically
- [x] Appeal linking: each first-instance judgment is matched to the appeal against it (signature + lower-court city, appeal must postdate it); judgments **changed or set aside on appeal are excluded** from training, testing, precedents and overviews, upheld ones are marked as confirmed — `tsl appeals` reports reversal rates by court and dispute type
- [ ] Does commentary improve prediction? `llm-rag-issue` retrieves precedents by legal issue and shows their holdings — measure on `--dev`, then test
- [ ] Retrieval by legal issue (dense + rerank from rag-eval-pl, same dispute type), calibration, conduct by claimant role — developed on the validation split
- [ ] FastAPI + Docker, MCP server, demo

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
tsl appeals                         # link judgments to appeals, reversal rates; reversed ones excluded
tsl annotate                        # commentary note per TSL judgment (holding, provisions, tip)
tsl commentary "przedawnienie roszczeń z umowy przewozu"   # case-law overview with citations
tsl predict "Przewoźnik żąda frachtu 4 500 zł, zleceniodawca potrącił karę za opóźnienie…"
```

## Data

Judgments come from [SAOS](https://www.saos.org.pl) (System Analizy Orzeczeń Sądowych), which publishes anonymised rulings of Polish common courts and the Supreme Court. Court rulings are not subject to copyright in Poland (art. 4 of the Copyright Act). Raw downloads stay out of git and are re-created with `tsl download`.

## License

MIT
