import json

from tsl_case_law import cli
from tsl_case_law import predict as P
from tsl_case_law.commentary import (
    NoteIndex,
    annotate,
    annotate_all,
    check_citations,
    grounded_provisions,
    overview,
    validate_note,
)

from .test_predict import FakeLLM as PredictLLM
from .test_predict import make_cases

ASSESSMENT = (
    "Sąd zważył, co następuje: roszczenie o zapłatę przewoźnego przedawnia się z upływem roku "
    "(art. 77 ust. 1 Prawa przewozowego). Pozwany podniósł zarzut przedawnienia skutecznie."
)


class NoteLLM:
    provider = model = "fake"

    def __init__(self, answer=None):
        self.calls = 0
        self.answer = answer

    def complete(self, system, user, max_tokens=2048):
        self.calls += 1
        if "przegląd orzecznictwa" in system:
            return "1. Sądy oddalają spóźnione roszczenia [V GC 1/15] [I C 999/99].\n\n2. Wnioski."
        return self.answer or json.dumps(
            {
                "teza": "Roszczenie o fracht przedawnia się po roku.",
                "argumenty": ["upływ terminu", "zarzut podniesiony skutecznie"],
                "przepisy": ["art. 77 ust. 1 Prawa przewozowego", "art. 123 k.c."],
                "wskazowka": "Dochodź frachtu przed upływem roku.",
                "zagadnienia": ["przedawnienie roszczeń przewozowych"],
            }
        )


def test_grounding_flags_provisions_missing_from_the_judgment():
    assert grounded_provisions(
        ["art. 77 ust. 1 Prawa przewozowego", "art. 123 k.c."], ASSESSMENT
    ) == [
        True,
        False,
    ]
    assert grounded_provisions(["Prawa przewozowego"], ASSESSMENT) == [True]


def test_annotate_validates_and_marks_grounding():
    note = annotate(NoteLLM(), {"id": 7, "assessment": ASSESSMENT, "sentencja": "oddala powództwo"})
    assert note["ok"] and note["teza"].startswith("Roszczenie")
    assert note["przepisy_grounded"] == [True, False]
    assert annotate(NoteLLM(answer="brak"), {"id": 8, "assessment": ASSESSMENT})["ok"] is False


def test_validate_note_tolerates_bad_types():
    v = validate_note({"teza": "x", "argumenty": "nie lista", "zagadnienia": ["a"] * 9})
    assert v["argumenty"] == [] and len(v["zagadnienia"]) == 5


def test_annotate_all_caches(tmp_path):
    llm = NoteLLM()
    cases = [{"id": i, "assessment": ASSESSMENT, "sentencja": ""} for i in range(4)]
    annotate_all(llm, cases, cache_dir=tmp_path, workers=2, progress=lambda _: None)
    n = llm.calls
    annotate_all(llm, cases, cache_dir=tmp_path, workers=2, progress=lambda _: None)
    assert llm.calls == n


def test_citation_check_catches_case_numbers_outside_the_list():
    text = (
        "**1. Jak sądy rozstrzygają**\n"
        "- Roczny termin z CMR [V GC 1/15; V  GC 2/15], także w apelacji [V GC 1/15, V GC 2 / 15].\n"
        "- Inaczej przy OC [I C 999/99].\n"
        "- Twierdzenie bez źródła.\n"
        "*To nie jest porada prawna.*"
    )
    chk = check_citations(text, {"V GC 1/15", "V GC 2/15"})
    assert (chk["citations"], chk["valid"], chk["distinct_cases"]) == (5, 4, 2)
    assert chk["invalid"] == ["I C 999/99"]
    assert chk["claims_with_citation"] == "2/3", "heading and disclaimer are not claims"


def test_real_overview_format_has_no_false_alarms():
    text = "- Roczny termin [V GC 388/14; V GC 1332/17; XII Ga 246/17].\n- Przypis [1]."
    chk = check_citations(text, {"V GC 388/14", "V GC 1332/17", "XII Ga 246/17"})
    assert chk["invalid"] == [] and chk["valid"] == 3


def test_overview_reports_invented_citations():
    cases = {1: {"id": 1, "case_number": "V GC 1/15", "date": "2015-01-01"}}
    notes = [
        {
            "id": 1,
            "ok": True,
            "teza": "przedawnienie roszczeń o fracht",
            "argumenty": [],
            "przepisy": [],
            "zagadnienia": ["przedawnienie"],
        }
    ]
    res = overview(NoteLLM(), "przedawnienie", NoteIndex(cases, notes))
    assert res["check"]["invalid"] == ["I C 999/99"]


def test_issue_retrieval_uses_notes_and_still_only_earlier_cases(tmp_path):
    cases, extracted = make_cases()
    notes = {
        c["id"]: {
            "id": c["id"],
            "ok": True,
            "teza": "przedawnienie" if "przedawn" in c["facts"] else "fracht",
            "argumenty": [],
            "przepisy": [],
            "zagadnienia": [],
            "wskazowka": "",
        }
        for c in cases
    }
    ex = P.build_examples(cases, extracted, notes)
    assert all(e.note for e in ex)
    train, test = P.temporal_split(ex)
    llm = PredictLLM()
    out = P.predict_llm(
        llm, train, test, k=3, by_issue=True, cache_dir=tmp_path, progress=lambda _: None
    )
    ids = {e.id for e in train}
    assert all(set(r["retrieved"]) <= ids for r in out)
    assert "Teza sądu:" in llm.prompts[0], "precedent holdings shown to the model"
    assert all(e.id not in r["retrieved"] for e, r in zip(test, out, strict=True))


def test_cli_annotate_and_commentary(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    cases, extracted = make_cases(n_per_year=2, years=range(2015, 2017))
    for c in cases:
        c["assessment"], c["sentencja"], c["court"] = ASSESSMENT, "oddala", "SR"
    cases[0]["case_number"] = "V GC 1/15"
    (tmp_path / "data/cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
    (tmp_path / "data/extracted.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in extracted.values())
    )
    monkeypatch.setattr("tsl_case_law.llm.LLM.from_env", classmethod(lambda cls: NoteLLM()))
    cli.main(["annotate"])
    out = capsys.readouterr().out
    assert "provisions found in the judgment text" in out
    cli.main(["commentary", "przedawnienie roszczeń przewozowych"])
    out = capsys.readouterr().out
    assert "PRZEGLĄD ORZECZNICTWA" in out and "Kontrola cytowań" in out and "I C 999/99" in out
