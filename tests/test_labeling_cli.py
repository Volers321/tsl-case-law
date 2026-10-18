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
    for j in (first_instance, appeal, unrelated):
        (raw / f"{j['id']}.json").write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
    cli.main(["build", "--min-score", "4"])
    cases = cli.load_cases()
    assert [c["id"] for c in cases] == [101, 202]  # sorted by date, unrelated dropped
    cli.main(["stats"])
    out = capsys.readouterr().out
    assert "dropped     1: low TSL score" in out and "APPEAL_DISMISSED" in out
