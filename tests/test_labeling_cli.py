import json

from tsl_case_law import cli
from tsl_case_law.labeling import export_for_labeling, read_labels, stratified_sample


def make_cases(n_per=(6, 2, 1)):
    out, i = [], 0
    for outcome, n in zip(("GRANTED", "DISMISSED", "SET_ASIDE"), n_per, strict=True):
        for _ in range(n):
            i += 1
            out.append(
                {
                    "id": i,
                    "case_number": f"I C {i}/24",
                    "court": "SR",
                    "date": "2024-01-01",
                    "instance": "first",
                    "outcome_rule": outcome,
                    "sentencja": "zasądza " * 400,
                    "facts": "przewoźnik " * 600,
                }
            )
    return out


def test_stratified_sample_covers_rare_classes():
    sample = stratified_sample(make_cases(), 3)
    assert {s["outcome_rule"] for s in sample} == {"GRANTED", "DISMISSED", "SET_ASIDE"}


def test_label_roundtrip_semicolon_and_comma(tmp_path):
    path = tmp_path / "s.csv"
    export_for_labeling(make_cases((2, 0, 0)), path)
    text = path.read_text(encoding="utf-8-sig")
    assert text.splitlines()[0].startswith("id;case_number")
    assert len(text.splitlines()[1]) < 3200, "excerpts must be truncated"
    assert ";… zasądza" in text.splitlines()[1], "sentencja shown from its end (operative part)"
    # Expert fills row 1 and re-saves with commas (e.g. Numbers in an English locale)
    lines = text.splitlines()
    header = lines[0].split(";")
    row = dict.fromkeys(header, "")
    row.update(
        id="1",
        is_tsl="tak",
        dispute_type="zaplata_frachtu",
        outcome_rule_ok="nie",
        outcome_gold="PARTIAL",
    )
    path.write_text(
        ",".join(header) + "\n" + ",".join(row[h] for h in header) + "\n", encoding="utf-8"
    )
    labels = read_labels(path)
    assert labels == {
        1: {
            "is_tsl": "tak",
            "dispute_type": "zaplata_frachtu",
            "outcome_rule_ok": "nie",
            "outcome_gold": "PARTIAL",
            "notes": "",
            "batch": 1,
        }
    }


def test_build_and_stats_end_to_end(tmp_path, monkeypatch, first_instance, appeal, capsys):
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    unrelated = dict(
        first_instance,
        id=303,
        textContent="<p>oddala powództwo</p><p>UZASADNIENIE</p><p>najem lokalu</p>",
    )
    criminal = dict(first_instance, id=404, courtCases=[{"caseNumber": "II K 7/15"}])
    for j in (first_instance, appeal, unrelated, criminal):
        (raw / f"{j['id']}.json").write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
    cli.main(["build", "--min-score", "4"])
    cases = cli.load_cases()
    assert [c["id"] for c in cases] == [101, 202]  # sorted by date, unrelated dropped
    cli.main(["stats"])
    out = capsys.readouterr().out
    assert "dropped     1: low TSL score" in out and "APPEAL_DISMISSED" in out
    assert "not a civil case (other: K)" in out
    cli.main(["inspect", "-n", "2"])
    out = capsys.readouterr().out
    assert "empty sentencja:" in out and "no assessment cut:" in out


def test_check_labels_compares_rules_and_llm(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/labels").mkdir(parents=True)
    cases = make_cases((2, 1, 0))  # ids 1,2 GRANTED; 3 DISMISSED
    (tmp_path / "data/cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
    export_for_labeling(cases, tmp_path / "data/labels/sample.csv")
    sheet = tmp_path / "data/labels/sample.csv"
    header = sheet.read_text(encoding="utf-8-sig").splitlines()[0].split(";")
    rows = [  # expert: case 2 is really PARTIAL, case 3 not TSL
        {"id": "1", "is_tsl": "tak", "dispute_type": "zaplata_frachtu", "outcome_rule_ok": "tak"},
        {
            "id": "2",
            "is_tsl": "tak",
            "dispute_type": "szkoda_w_ladunku",
            "outcome_rule_ok": "nie",
            "outcome_gold": "partial",
        },
        {"id": "3", "is_tsl": "nie"},
    ]
    blank = dict.fromkeys(header, "")
    lines = [";".join(header)] + [";".join((blank | r)[h] for h in header) for r in rows]
    sheet.write_text("\n".join(lines) + "\n", encoding="utf-8")
    llm_rows = [
        {
            "id": 1,
            "facts_llm": {"is_tsl": True, "dispute_type": "zaplata_frachtu"},
            "outcome_llm": {"outcome": "GRANTED"},
        },
        {
            "id": 2,
            "facts_llm": {"is_tsl": True, "dispute_type": "inne"},
            "outcome_llm": {"outcome": "PARTIAL"},
        },
        {
            "id": 3,
            "facts_llm": {"is_tsl": True, "dispute_type": "inne"},
            "outcome_llm": {"outcome": "DISMISSED"},
        },
    ]
    (tmp_path / "data/extracted.jsonl").write_text("".join(json.dumps(r) + "\n" for r in llm_rows))
    cli.main(["check-labels"])
    out = capsys.readouterr().out
    assert "relevant to TSL: 2 (67%)" in out
    assert "rules: 1/2 (50%)" in out and "LLM:   2/2 (100%)" in out
    assert "LLM relevance (is_tsl) agreement: 2/3 (67%)" in out
    assert "LLM dispute type agreement:       1/2 (50%)" in out
    assert "✗ I C 2/24: rule=GRANTED gold=PARTIAL llm=PARTIAL" in out


def test_interactive_labelling_saves_and_resumes(tmp_path):
    from tsl_case_law.labeling import label_interactively

    path = tmp_path / "s.csv"
    cases = make_cases((3, 0, 0))
    export_for_labeling(cases, path)
    # case 1: TSL, type 1 (freight), rule correct (Enter); case 2: not TSL; then quit
    answers = iter(["", "1", "", "n", "q"])
    assert label_interactively(path, ask=lambda _: next(answers), show=lambda *_: None) == 2
    labels = read_labels(path)
    assert labels[cases[0]["id"]] == {
        "is_tsl": "tak",
        "dispute_type": "zaplata_frachtu",
        "outcome_rule_ok": "tak",
        "outcome_gold": "",
        "notes": "",
        "batch": 1,
    }
    assert labels[cases[1]["id"]]["is_tsl"] == "nie"
    # resume: only case 3 is asked; rule wrong -> pick the correct outcome (3 = DISMISSED)
    answers = iter(["", "8", "n", "3"])
    assert label_interactively(path, ask=lambda _: next(answers), show=lambda *_: None) == 1
    assert read_labels(path)[cases[2]["id"]]["outcome_gold"] == "DISMISSED"


def test_review_mode_reasks_only_disputed_and_restores_on_quit(tmp_path):
    from tsl_case_law.labeling import label_interactively

    path = tmp_path / "s.csv"
    cases = make_cases((3, 0, 0))
    export_for_labeling(cases, path)
    answers = iter(["n", "n", "n"])  # label all three as not TSL
    label_interactively(path, ask=lambda _: next(answers), show=lambda *_: None)
    a, b = cases[0]["id"], cases[1]["id"]
    # review two disputed cases: flip the first to TSL, quit before the second
    answers = iter(["", "1", "", "q"])
    shown = []
    label_interactively(
        path, ask=lambda _: next(answers), show=shown.append, review={a: "fracht", b: "x"}
    )
    labels = read_labels(path)
    assert labels[a]["is_tsl"] == "tak" and labels[a]["dispute_type"] == "zaplata_frachtu"
    assert labels[b]["is_tsl"] == "nie", "unreviewed case keeps its earlier label"
    assert labels[cases[2]["id"]]["is_tsl"] == "nie", "undisputed case untouched"
    assert any("STRESZCZENIE LLM: fracht" in str(x) for x in shown)


def test_sample_add_appends_new_batch_and_keeps_labels(tmp_path, monkeypatch):
    from tsl_case_law.labeling import label_interactively

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/labels").mkdir(parents=True)
    cases = make_cases((6, 2, 1))
    (tmp_path / "data/cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
    cli.main(["sample", "-n", "3"])
    answers = iter(["n", "n", "n"])
    label_interactively(cli.LABEL_SHEET, ask=lambda _: next(answers), show=lambda *_: None)
    cli.main(["sample", "--add", "4"])
    from tsl_case_law.labeling import _read_sheet

    _, rows = _read_sheet(cli.LABEL_SHEET)
    assert len(rows) == 7 and len({r["id"] for r in rows}) == 7, "no case sampled twice"
    labels = read_labels(cli.LABEL_SHEET)
    assert len(labels) == 3 and {lab["batch"] for lab in labels.values()} == {1}
    assert sum(r["batch"] == "2" and not r["is_tsl"] for r in rows) == 4
