import pytest

from tsl_case_law.processing import (
    first_amount,
    html_to_text,
    reasons_outcome,
    repertorium,
    rule_outcome,
    split_sections,
    to_record,
    tsl_score,
)

from .conftest import FIRST_INSTANCE_HTML, saos_judgment


def test_html_to_text_keeps_paragraphs():
    text = html_to_text("<p>A&nbsp;b</p><p>c</p>")
    assert text == "A b\nc"
    assert html_to_text(None) == ""


def test_sections_never_leak_assessment_into_facts():
    sec = split_sections(html_to_text(FIRST_INSTANCE_HTML))
    assert "zasądza" in sec.sentencja and "UZASADNIENIE" not in sec.sentencja
    assert "stan faktyczny" in sec.facts
    assert "zasługiwało" not in sec.facts and "art. 23" not in sec.facts
    assert sec.assessment.startswith("Sąd zważył")


def test_spaced_reasons_header_and_appeal_assessment(appeal):
    sec = split_sections(html_to_text(appeal["textContent"]))
    assert "oddala apelację" in sec.sentencja
    assert "Sąd Okręgowy zasądził" in sec.facts
    assert sec.assessment.startswith("Apelacja nie zasługuje")


@pytest.mark.parametrize(
    "sentencja,expected",
    [
        (
            "zasądza od pozwanego kwotę 100 zł; w pozostałym zakresie powództwo oddala",
            ("first", "PARTIAL"),
        ),
        ("zasądza od pozwanego na rzecz powoda kwotę 100 zł", ("first", "GRANTED")),
        (
            "1. oddala powództwo; 2. zasądza od powoda na rzecz pozwanego kwotę 3 617 zł tytułem kosztów procesu",
            ("first", "DISMISSED"),
        ),
        (
            "I. zasądza od pozwanego kwotę 5 000 zł; II. zasądza od pozwanego kwotę 1 200 zł tytułem zwrotu kosztów procesu",
            ("first", "GRANTED"),
        ),
        (
            "oddala powództwo i zasądza od powoda kwotę 5417 zł tytułem zwrotu kosztów procesu",
            ("first", "DISMISSED"),
        ),
        (
            "oddala apelację i zasądza od pozwanego kwotę 4050 zł tytułem kosztów postępowania apelacyjnego",
            ("appeal", "APPEAL_DISMISSED"),
        ),
        ("umarza postępowanie", ("first", "DISCONTINUED")),
        ("oddala apelację", ("appeal", "APPEAL_DISMISSED")),
        ("zmienia zaskarżony wyrok i oddala apelację w pozostałej części", ("appeal", "CHANGED")),
        (
            "uchyla zaskarżony wyrok i przekazuje sprawę do ponownego rozpoznania; po rozpoznaniu apelacji",
            ("appeal", "SET_ASIDE"),
        ),
        ("oddala skargę kasacyjną", ("appeal", "APPEAL_DISMISSED")),
        ("postanawia odrzucić pozew", ("first", "UNKNOWN")),
    ],
)
def test_rule_outcome(sentencja, expected):
    assert rule_outcome(sentencja) == expected


def test_first_amount():
    assert first_amount("zasądza kwotę 4 500,00 zł z odsetkami") == (4500.0, "PLN")
    assert first_amount("kwotę 1.250 euro") == (1250.0, "EUR")
    assert first_amount("oddala powództwo") is None


def test_tsl_score_separates_transport_from_other_cases():
    transport = "przewoźnik, list przewozowy CMR, fracht, ładunek, spedytor"
    other = "umowa najmu lokalu mieszkalnego, czynsz, wypowiedzenie"
    assert tsl_score(transport) >= 10 > tsl_score(other) == 0


def test_to_record(first_instance):
    rec = to_record(first_instance)
    assert (rec.instance, rec.outcome_rule) == ("first", "PARTIAL")
    assert (rec.amount, rec.currency) == (4500.0, "PLN")
    assert rec.court == "Sąd Rejonowy w Przykładowie"
    assert rec.case_number == "X GC 123/24"
    assert rec.tsl_score > 6
    assert "CMR" in rec.regulations[0]


def test_to_record_tolerates_missing_fields():
    rec = to_record(
        {
            "id": 5,
            "courtType": "SUPREME",
            "textContent": "<p>oddala skargę kasacyjną</p>",
            "keywords": {"a": "przewóz"},
        }
    )
    assert rec.court == "Sąd Najwyższy" and rec.keywords == ["przewóz"] and rec.facts == ""


@pytest.mark.parametrize(
    "number,expected",
    [
        ("V GC 388/14", "GC"),
        ("III Ca 1264/14", "Ca"),
        ("I ACa 1/15", "ACa"),
        ("XVII AmC 2990/14", "AmC"),
        ("VIII U 4988/13", "U"),
        ("V CNP 27/14", "CNP"),
        ("", ""),
    ],
)
def test_repertorium(number, expected):
    assert repertorium(number) == expected


@pytest.mark.parametrize(
    "sentencja,appeal,expected",
    [
        ("I. utrzymuje w całości w mocy nakaz zapłaty z dnia 3 marca 2014 r.", None, "GRANTED"),
        ("1. z a s ą d z a od pozwanego kwotę 100 zł", None, "GRANTED"),
        ("1. o d d a l a powództwo; 2. zasądza od powoda koszty procesu", None, "DISMISSED"),
        ("odrzuca apelację", None, "APPEAL_DISMISSED"),
        # Register says appeal although the sentencja never says "apelacja"
        ("zmienia zaskarżony wyrok w ten sposób, że zasądza kwotę 900 zł", True, "CHANGED"),
    ],
)
def test_rule_outcome_real_data_patterns(sentencja, appeal, expected):
    assert rule_outcome(sentencja, appeal=appeal)[1] == expected


@pytest.mark.parametrize(
    "assessment,appeal,expected",
    [
        (
            "Apelacja nie zasługiwała na uwzględnienie. Dlatego ją oddalono.",
            True,
            "APPEAL_DISMISSED",
        ),
        ("Z tych względów Sąd Okręgowy zmienił zaskarżony wyrok.", True, "CHANGED"),
        ("Mając powyższe na uwadze, powództwo należało oddalić.", False, "DISMISSED"),
        ("Powództwo nie zasługiwało na uwzględnienie.", False, "DISMISSED"),
        ("Powództwo zasługiwało na uwzględnienie w całości.", False, "GRANTED"),
        ("Zasądzono 500 zł, a w pozostałym zakresie powództwo oddalono.", False, "PARTIAL"),
        ("O kosztach orzeczono na podstawie art. 98 k.p.c.", False, "UNKNOWN"),
    ],
)
def test_reasons_outcome(assessment, appeal, expected):
    assert reasons_outcome(assessment, appeal) == expected


def test_reasons_only_document_gets_outcome_from_reasons():
    html = (
        "<p>Sygn. akt III Ca 1264/14</p><p>UZASADNIENIE</p>"
        "<p>Sąd Rejonowy zasądził od pozwanego przewoźnika fracht.</p>"
        "<p>Apelacja pozwanego nie zasługiwała na uwzględnienie i podlegała oddaleniu.</p>"
    )
    rec = to_record(
        saos_judgment(html, judgmentType="REASONS", courtCases=[{"caseNumber": "III Ca 1264/14"}])
    )
    assert (rec.repertorium, rec.instance) == ("Ca", "appeal")
    assert (rec.outcome_rule, rec.outcome_source) == ("APPEAL_DISMISSED", "reasons")
    assert "nie zasługiwała" not in rec.facts


def test_new_assessment_markers_cut_facts():
    sec = split_sections(
        "WYROK\noddala powództwo\nUZASADNIENIE\nPowód żądał frachtu.\n"
        "Powództwo nie zasługiwało na uwzględnienie, bo roszczenie się przedawniło."
    )
    assert sec.facts == "Powód żądał frachtu."


@pytest.mark.parametrize(
    "sentencja,appeal,expected",
    [
        (
            "I. zasądza od pozwanego na rzecz powoda kwotę 45.009 zł z ustawowymi odsetkami od dnia "
            "29 stycznia 2013r. do dnia zapłaty, II.zasądza od pozwanego na rzecz powoda kwotę "
            "41.111,45 zł tytułem zwrotu kosztów procesu",
            None,
            "GRANTED",
        ),
        (
            "zasądza od pozwanego kwotę 5 000 zł z odsetkami od dnia 3 marca 2015 r. oraz kwotę 917 zł "
            "tytułem kosztów procesu",
            None,
            "GRANTED",
        ),
        (
            "1. zmienia zaskarżony wyrok w punkcie 1 w ten sposób, że powództwo oddala, - w punkcie 2 "
            "w ten sposób, że zasądza od powódki kwotę 3.617 złotych tytułem zwrotu kosztów procesu; "
            "2. zasądza od powódki kwotę 9.960 złotych tytułem kosztów postępowania apelacyjnego",
            True,
            "CHANGED",
        ),
        (
            "I. powództwo główne w całości oddala; II. zasądza od powoda koszty procesu",
            None,
            "DISMISSED",
        ),
        (
            "zasądza od pozwanego kwotę 2 700 zł tytułem kosztów procesu z odsetkami ustawowymi za "
            "opóźnienie od dnia uprawomocnienia się wyroku; oddala powództwo",
            None,
            "DISMISSED",
        ),
    ],
)
def test_rule_outcome_cost_orders_round_two(sentencja, appeal, expected):
    assert rule_outcome(sentencja, appeal=appeal)[1] == expected
