"""Synthetic judgments written for tests. They imitate the structure of SAOS records and
Polish civil judgments; parties, case numbers and amounts are invented."""

import pytest

FIRST_INSTANCE_HTML = """
<p>Sygn. akt X GC 123/24</p>
<p>WYROK</p>
<p>W IMIENIU RZECZYPOSPOLITEJ POLSKIEJ</p>
<p>Sąd Rejonowy w Przykładowie X Wydział Gospodarczy</p>
<p>1. zasądza od pozwanego Spedycja Beta sp. z o.o. na rzecz powoda Trans Alfa sp. z o.o. kwotę 4 500,00 zł z odsetkami;</p>
<p>2. w pozostałym zakresie powództwo oddala;</p>
<p>3. zasądza od pozwanego na rzecz powoda kwotę 917 zł tytułem kosztów procesu.</p>
<p>UZASADNIENIE</p>
<p>Powód wniósł o zasądzenie kwoty 6 000 zł tytułem frachtu za przewóz towaru na trasie Poznań – Lyon. Pozwany spedytor odmówił zapłaty, wskazując na opóźnienie w dostawie i uszkodzenie ładunku stwierdzone w liście przewozowym CMR.</p>
<p>Sąd ustalił następujący stan faktyczny: strony zawarły umowę przewozu na podstawie zlecenia transportowego. Przewoźnik dostarczył przesyłkę z jednodniowym opóźnieniem.</p>
<p>Sąd zważył, co następuje:</p>
<p>Powództwo zasługiwało na uwzględnienie w części. Zgodnie z art. 23 Konwencji CMR odszkodowanie za opóźnienie nie może przekraczać wysokości przewoźnego.</p>
"""

APPEAL_HTML = """
<p>WYROK</p>
<p>Sąd Apelacyjny po rozpoznaniu apelacji pozwanego od wyroku Sądu Okręgowego</p>
<p>1. oddala apelację;</p>
<p>2. zasądza od pozwanego na rzecz powoda kwotę 2 700 zł tytułem kosztów postępowania apelacyjnego.</p>
<p>U Z A S A D N I E N I E</p>
<p>Sąd Okręgowy zasądził od pozwanego przewoźnika odszkodowanie za utratę przesyłki. Apelację wniósł pozwany.</p>
<p>Apelacja nie zasługuje na uwzględnienie. Przewoźnik odpowiada za utratę towaru od przyjęcia do wydania.</p>
"""


def saos_judgment(html: str, **overrides) -> dict:
    j = {
        "id": 101,
        "courtType": "COMMON",
        "courtCases": [{"caseNumber": "X GC 123/24"}],
        "judgmentType": "SENTENCE",
        "judgmentDate": "2024-05-14",
        "division": {"court": {"name": "Sąd Rejonowy w Przykładowie", "type": "DISTRICT"}},
        "textContent": html,
        "referencedRegulations": [
            {"text": "Konwencja o umowie międzynarodowego przewozu drogowego towarów (CMR)"}
        ],
        "keywords": ["umowa przewozu"],
    }
    j.update(overrides)
    return j


@pytest.fixture
def first_instance():
    return saos_judgment(FIRST_INSTANCE_HTML)


@pytest.fixture
def appeal():
    return saos_judgment(
        APPEAL_HTML, id=202, courtCases=[{"caseNumber": "V ACa 9/25"}], judgmentDate="2025-02-03"
    )
