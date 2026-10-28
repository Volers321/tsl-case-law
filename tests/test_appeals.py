import json

from tsl_case_law import cli
from tsl_case_law import predict as P
from tsl_case_law.appeals import link_appeals, lower_court_refs, norm_sig, reversal_stats
from tsl_case_law.commentary import NoteIndex


def first(id_, sig, court, date="2015-03-01", outcome="GRANTED"):
    return {
        "id": id_,
        "case_number": sig,
        "court": court,
        "date": date,
        "instance": "first",
        "outcome_rule": outcome,
        "sentencja": "",
        "facts": "przewoźnik fracht " * 5,
    }


def appeal(id_, sig, text, outcome, date="2016-01-10", facts=""):
    return {
        "id": id_,
        "case_number": sig,
        "court": "Sąd Okręgowy w Radomiu",
        "date": date,
        "instance": "appeal",
        "outcome_rule": outcome,
        "sentencja": text,
        "facts": facts,
    }


CASES = [
    first(1, "V GC 657/14", "Sąd Rejonowy w Radomiu"),
    first(2, "V GC 657/14", "Sąd Rejonowy w Kielcach"),  # same signature, other court
    first(3, "VI GC 100/15", "Sąd Rejonowy dla Łodzi-Śródmieścia w Łodzi"),
    first(4, "I C 5/15", "Sąd Rejonowy w Gdyni"),
    appeal(
        10,
        "IX Ga 525/15",
        "na skutek apelacji powoda od wyroku Sądu Rejonowego w Radomiu "
        "z dnia 14 września 2015 roku, sygn. akt V GC 657/14 I. oddala apelację",
        "APPEAL_DISMISSED",
    ),
    appeal(
        11,
        "XIII Ga 1/16",
        "od wyroku Sądu Rejonowego dla Łodzi-Śródmieścia w Łodzi "
        "z dnia 1 marca 2015 r. sygn. akt VI GC 100/15 zmienia zaskarżony wyrok",
        "CHANGED",
    ),
    appeal(
        12,
        "III Ca 9/16",
        "Sygn. akt III Ca 9/16",
        "SET_ASIDE",
        facts="Wyrokiem z dnia 1 marca 2015 r. Sąd Rejonowy w Gdyni (sygn. akt I C 5/15) zasądził",
    ),
]


def test_signature_normalisation_and_extraction():
    assert norm_sig("V  GC 657 / 14") == "V GC 657/14"
    refs = lower_court_refs(CASES[4])
    assert refs == [("V GC 657/14", "radomiu")]


def test_links_need_the_same_court_when_the_signature_is_ambiguous():
    links = link_appeals(CASES)
    assert links[1]["status"] == "upheld" and links[1]["appeal_case_number"] == "IX Ga 525/15"
    assert 2 not in links, "same signature in Kielce must not be linked"
    assert links[3]["status"] == "changed"
    assert links[4]["status"] == "set_aside", "lower signature found in the reasons"


def test_appeal_cannot_predate_the_judgment():
    cases = [first(1, "V GC 657/14", "Sąd Rejonowy w Radomiu", date="2017-01-01"), CASES[4]]
    assert link_appeals(cases) == {}


def test_reversal_rate_counts_only_appealed_cases():
    stats = reversal_stats(CASES, link_appeals(CASES))["all"]
    assert stats["appealed"] == 3 and stats["not_found"] == 1
    assert abs(stats["reversal_rate"] - 2 / 3) < 1e-9


def test_reversed_judgments_leave_examples_and_overviews():
    links = link_appeals(CASES)
    kept = P.build_examples(CASES, {}, None, links)
    assert {e.id for e in kept} == {1, 2}
    assert next(e for e in kept if e.id == 1).appeal == "upheld"
    assert len(P.build_examples(CASES, {}, None, links, drop_reversed=False)) == 4
    cases = {c["id"]: c for c in CASES}
    notes = [
        {"id": i, "ok": True, "teza": "fracht", "argumenty": [], "przepisy": [], "zagadnienia": []}
        for i in (1, 3, 4)
    ]
    assert [c["id"] for c, _ in NoteIndex(cases, notes, links).items] == [1]


def test_cli_appeals(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in CASES))
    cli.main(["appeals", "--min-n", "1"])
    out = capsys.readouterr().out
    assert "3 linked" in out and "67%" in out and "changed by XIII Ga 1/16" in out
    assert "ALL 3 TSL appellate judgments" in out and "changed or set aside in 67%" in out
    saved = json.loads((tmp_path / "data/appeals.json").read_text())
    assert saved["1"]["status"] == "upheld"
