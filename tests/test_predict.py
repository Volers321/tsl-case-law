import json

import pytest

from tsl_case_law import cli
from tsl_case_law import predict as P

# Three synthetic dispute patterns with a clear signal in the wording of the facts.
PATTERNS = {
    "win": "przewoźnik wykonał przewóz fracht faktura pozwany nie zapłacił uznał dług",
    "lose": "szkoda w ładunku reklamacja po terminie przedawnienie roszczenia upłynął rok",
    "partial": "opóźnienie dostawy limit odpowiedzialności przewoźnego CMR artykuł 23",
}


def make_cases(n_per_year=6, years=range(2015, 2025)):
    cases, extracted, i = [], {}, 0
    labels = list(PATTERNS)
    for year in years:
        for j in range(n_per_year):
            i += 1
            label = labels[(i + j) % 3]
            outcome = {"win": "GRANTED", "lose": "DISMISSED", "partial": "PARTIAL"}[label]
            cases.append(
                {
                    "id": i,
                    "case_number": f"V GC {i}/{year % 100}",
                    "date": f"{year}-0{1 + j % 9}-15",
                    "instance": "first",
                    "outcome_rule": outcome,
                    "facts": f"{PATTERNS[label]} sprawa numer {i} " * 5,
                }
            )
            extracted[i] = {
                "id": i,
                "facts_llm": {
                    "ok": True,
                    "is_tsl": i % 10 != 0,  # every 10th case is not TSL
                    "dispute_type": "zaplata_frachtu",
                    "defences": ["przedawnienie"] if label == "lose" else [],
                    "conduct": {"acknowledged_debt": label == "win", "partial_payment": None},
                    "summary": PATTERNS[label],
                },
                "outcome_llm": {"ok": True, "outcome": outcome},
            }
    return cases, extracted


class FakeLLM:
    provider = model = "fake"

    def __init__(self):
        self.prompts = []

    def complete(self, system, user, max_tokens=2048):
        self.prompts.append(user)
        guess = "lose" if "przedawnienie" in user else "win"
        p = {lab: 0.1 for lab in P.LABELS} | {guess: 0.8}
        return json.dumps(p | {"reason": "test", "precedents": [1]})


def test_examples_keep_only_tsl_first_instance_with_known_outcome():
    cases, extracted = make_cases()
    cases.append(dict(cases[0], id=999, instance="appeal"))
    cases.append(dict(cases[1], id=998, outcome_rule="UNKNOWN"))
    extracted[998] = dict(extracted[2], outcome_llm={"outcome": "DISMISSED"})
    ex = P.build_examples(cases, extracted)
    ids = {e.id for e in ex}
    assert 999 not in ids and 10 not in ids
    assert next(e for e in ex if e.id == 998).label == "lose", "LLM reading fills UNKNOWN"
    assert [e.date for e in ex] == sorted(e.date for e in ex)


def test_temporal_split_tests_on_newest():
    ex = P.build_examples(*make_cases())
    train, test = P.temporal_split(ex, 0.2)
    assert max(e.date for e in train) <= min(e.date for e in test)
    assert len(test) == round(len(ex) * 0.2)


def test_similar_cases_finds_same_pattern():
    ex = P.build_examples(*make_cases())
    hits = P.SimilarCases(ex).search("reklamacja po terminie i przedawnienie", 3)
    assert all(e.label == "lose" for e, _ in hits)


def test_tfidf_beats_majority_on_learnable_signal():
    ex = P.build_examples(*make_cases())
    train, test = P.temporal_split(ex)
    gold = [e.label for e in test]
    major = P.evaluate(gold, P.predict_majority(train, test))
    tfidf = P.evaluate(gold, P.predict_tfidf(train, test))
    assert tfidf["accuracy"] > major["accuracy"]
    assert tfidf["brier"] < major["brier"]


def test_llm_rag_uses_only_earlier_precedents_and_caches(tmp_path):
    ex = P.build_examples(*make_cases())
    train, test = P.temporal_split(ex)
    llm = FakeLLM()
    out = P.predict_llm(llm, train, test, k=3, cache_dir=tmp_path, progress=lambda _: None)
    train_ids = {e.id for e in train}
    split = min(e.date for e in test)
    for r in out:
        assert set(r["retrieved"]) <= train_ids, "precedent leaked from the test period"
    assert all(e.date <= split for e in train)
    assert "PODOBNE SPRAWY" in llm.prompts[0] and "wynik:" in llm.prompts[0]
    n_calls = len(llm.prompts)
    P.predict_llm(llm, train, test, k=3, cache_dir=tmp_path, progress=lambda _: None)
    assert len(llm.prompts) == n_calls, "second run must come from the cache"


def test_parse_probs_normalises_and_rejects_garbage():
    p = P.parse_probs('{"win": 2, "partial": 1, "lose": 1, "reason": "x"}')
    assert p["win"] == pytest.approx(0.5) and sum(p[lab] for lab in P.LABELS) == pytest.approx(1)
    assert P.parse_probs("nie wiem") is None
    assert P.parse_probs('{"win": 0, "partial": 0, "lose": 0}') is None


def test_evaluate_and_intervals():
    gold = ["win", "lose", "win", "partial"]
    perfect = [{lab: float(lab == g) for lab in P.LABELS} for g in gold]
    assert P.evaluate(gold, perfect) == {"n": 4, "accuracy": 1.0, "macro_f1": 1.0, "brier": 0.0}
    lo, hi = P.wilson(0.5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8


def test_conduct_table():
    ex = P.build_examples(*make_cases())
    rows = {(r["feature"], r["value"]): r for r in P.conduct_table(ex, min_n=3)}
    assert rows[("acknowledged_debt", True)]["claimant_wins"] == 1.0
    assert ("partial_payment", True) not in rows, "unknown (null) is not counted as no"


def test_cli_predict_eval_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    cases, extracted = make_cases()
    (tmp_path / "data/cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
    (tmp_path / "data/extracted.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in extracted.values())
    )
    cli.main(["predict-eval", "--methods", "majority,tfidf"])
    out = capsys.readouterr().out
    assert "majority" in out and "tfidf" in out and "95% CI" in out
    report = json.loads((tmp_path / "data/predict_report.json").read_text())
    assert set(report["results"]) == {"majority", "tfidf"}
    cli.main(["conduct-stats", "--min-n", "3"])
    assert "acknowledged_debt = yes" in capsys.readouterr().out
    cli.main(["similar", "przedawnienie reklamacja", "-k", "2"])
    assert "przegrana" in capsys.readouterr().out
