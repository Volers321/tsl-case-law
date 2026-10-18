import pytest

from tsl_case_law.processing import (
    first_amount,
    html_to_text,
    rule_outcome,
    split_sections,
    to_record,
    tsl_score,
)

from .conftest import FIRST_INSTANCE_HTML


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
