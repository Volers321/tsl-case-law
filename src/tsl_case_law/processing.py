"""Turn raw SAOS judgments into clean case records.

A Polish civil judgment has two parts: the *sentencja* (operative part — who wins what)
and the *uzasadnienie* (reasons), which usually describes the established facts first
and then the court's legal assessment, introduced by a phrase such as "Sąd zważył, co
następuje". For outcome prediction the model may only see the facts; the sentencja and
the legal assessment would leak the answer. This module splits the text accordingly and
derives a first, rule-based outcome label that the human-labelled gold set later checks.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field

from bs4 import BeautifulSoup

# ------------------------------------------------------------------------- text cleanup


def html_to_text(html: str | None) -> str:
    if not html:
        return ""
    text = BeautifulSoup(html, "html.parser").get_text("\n")
    text = unicodedata.normalize("NFC", text).replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ------------------------------------------------------------------------- relevance

# Weighted vocabulary of transport / forwarding disputes. Stems match inflected forms.
TSL_TERMS: dict[str, float] = {
    r"\bcmr\b": 3,
    r"prawo przewozowe|prawa przewozowego": 3,
    r"przewo[źz]ni[kc]": 2,
    r"spedy(tor|cj)": 2,
    r"\bfracht": 2,
    r"list(u|em)? przewozow": 2,
    r"umow\w* przewozu": 2,
    r"przesy[łl]k": 1,
    r"ładun(ek|ku|kiem)": 1,
    r"naczep|ciągnik siodłow|ciężarów": 1,
    r"zleceni\w* transportow": 2,
    r"towar\w* w transporcie|szkod\w* w towarze|uszkodzeni\w* towaru": 1,
}
_TSL_RE = [(re.compile(p, re.IGNORECASE), w) for p, w in TSL_TERMS.items()]


def tsl_score(text: str) -> float:
    """Weighted count of TSL vocabulary, capped per term so one repeated word can't win."""
    return sum(w * min(len(rx.findall(text)), 5) for rx, w in _TSL_RE)


# The court register ("repertorium") in the case number tells the kind of case:
# "V GC 388/14" is commercial first instance, "I ACa 12/15" a civil appeal. TSL vocabulary
# also appears in criminal (K, Ka), labour (P, Pa), social-insurance (U) and consumer-clause
# (AmC) cases, which are not contract disputes — only civil/commercial registers are kept.
CIVIL_FIRST = {"C", "GC", "Nc", "GNc", "Co", "GCo"}
CIVIL_APPEAL = {"Ca", "Ga", "Cz", "Gz", "ACa", "AGa", "ACz", "AGz", "CSK", "CSKP", "CZ", "CNP"}
CIVIL_REGISTERS = CIVIL_FIRST | CIVIL_APPEAL
_REPERTORIUM = re.compile(r"^\s*(?:[IVXL]+\s+)?([A-Za-z]+)\s*\d+\s*/\s*\d+")


def repertorium(case_number: str) -> str:
    m = _REPERTORIUM.match(case_number or "")
    return m.group(1) if m else ""


# ------------------------------------------------------------------------- sections

_REASONS_HEADER = re.compile(r"^\s*U\s*Z\s*A\s*S\s*A\s*D\s*N\s*I\s*E\s*N\s*I\s*E\s*$", re.M | re.I)
# Phrases that open the court's legal assessment — everything after them may reveal the
# outcome, so it never enters the prediction input.
_ASSESSMENT = re.compile(
    r"(Sąd\s+(?:\w+\s+){0,3}zważył|Sąd\s+(?:\w+\s+){0,3}(?:uznał|stwierdził),?\s+co\s+następuje"
    r"|Sąd\s+(?:\w+\s+){0,3}ocenił|Rozważania\s+prawne|W\s+tak\s+ustalonym\s+stanie\s+faktycznym"
    r"|(?:Apelacja|Skarga\s+kasacyjna|Zażalenie|Powództwo)\s+(?:\w+\s+){0,3}"
    r"(?:jest|okazał[ao]?\s+się|zasługuj\w+|nie\s+zasługiwał\w*|podlega\w*)"
    # Cutting earlier is always safe (less input, no leakage), so these fallbacks are broad:
    r"|Przechodząc\s+do\s+(?:oceny|rozważań)|Powyższy\s+stan\s+faktyczny|Ustalony\s+stan\s+faktyczny"
    r"|Mając\s+(?:powyższe|to)\s+na\s+(?:uwadze|względzie))",
    re.IGNORECASE,
)


@dataclass
class Sections:
    sentencja: str
    facts: str
    assessment: str


def split_sections(text: str) -> Sections:
    m = _REASONS_HEADER.search(text)
    sentencja, reasons = (text[: m.start()], text[m.end() :]) if m else (text, "")
    a = _ASSESSMENT.search(reasons)
    facts, assessment = (reasons[: a.start()], reasons[a.start() :]) if a else (reasons, "")
    return Sections(sentencja.strip(), facts.strip(), assessment.strip())


# ------------------------------------------------------------------------- outcome rules

FIRST_INSTANCE = ("GRANTED", "PARTIAL", "DISMISSED", "DISCONTINUED")
APPEAL = ("APPEAL_DISMISSED", "CHANGED", "SET_ASIDE")
UNKNOWN = "UNKNOWN"


def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE) is not None


_POINT_SPLIT = re.compile(
    # numbered points, also without a space after the number ("II.zasądza")
    r";|(?:^|\s)(?:\d{1,2}|[IVX]{1,4})[.)](?:\s|(?=[a-ząćęłńóśźż]))"
    r"|\s+i\s+(?=zasądza|nakazuje|obciąża|przyznaje|znosi)"  # "oddala … i zasądza koszty"
)


def _drop_cost_orders(s: str) -> str:
    """Remove operative points about court costs ("zasądza od powoda … kosztów procesu").

    Nearly every dismissal also orders the losing claimant to pay costs, so a naive search
    for "zasądza" would label dismissals as wins.
    """
    points = [p.strip() for p in _POINT_SPLIT.split(s) if p and p.strip()]
    kept = [p for p in points if not _is_cost_order(p)]
    return " ; ".join(kept)


def _is_cost_order(point: str) -> bool:
    if not re.search(r"koszt|opłat\w* sądow|wynagrodzeni\w* pełnomocnik", point, re.I):
        return False
    # A point that awards the claim and its costs together ("zasądza kwotę 5 000 zł z
    # odsetkami od dnia 29 stycznia 2013 r. oraz kwotę 917 zł tytułem kosztów") is kept:
    # interest from a calendar date belongs to the claim, costs run from finality.
    return not re.search(r"odset\w*[^;]{0,80}?od\s+dnia\s+\d", point, re.I)


def rule_outcome(sentencja: str, appeal: bool | None = None) -> tuple[str, str]:
    """Return (instance, outcome) inferred from the operative part.

    instance is "appeal" when the court decides on an appeal or cassation, else "first".
    Pass ``appeal`` when it is known from the case register; otherwise it is guessed from
    the wording.
    The rules are deliberately simple; their accuracy is measured against human labels.
    """
    full = _unspace(" ".join(sentencja.split()))
    s = _drop_cost_orders(full)
    if appeal is None:
        appeal = _has(r"apelacj|skarg\w* kasacyjn|zażaleni", s)
    if appeal:
        # "zmienia zaskarżony wyrok w punkcie 1 w ten sposób, że powództwo oddala, w punkcie 2
        # … kosztów procesu" is one long point that also mentions costs — look at it whole.
        if _has(r"\bzmienia\b", s) or _has(r"\bzmienia\b[^;]{0,400}powództw", full):
            return "appeal", "CHANGED"
        if _has(r"\buchyla\b", s):
            return "appeal", "SET_ASIDE"
        # "odrzuca apelację" is procedural, but for the parties it means the same: the
        # appealed judgment stands.
        if _has(r"\b(oddala|odrzuca)\b", s):
            return "appeal", "APPEAL_DISMISSED"
        return "appeal", UNKNOWN
    # Many commercial cases start with a payment order (nakaz zapłaty); after the defendant
    # objects, the court either upholds it (claimant wins) or sets it aside.
    upheld = _has(r"utrzymuje\s+(?:\w+\s+){0,3}w\s+mocy\s+(?:\w+\s+){0,2}nakaz", s)
    rest_dismissed = _has(
        r"(w\s+pozostał\w+\s+(zakresie|części)\s+(powództwo\s+)?oddala|oddala\s+powództwo\s+w\s+pozostał)",
        s,
    )
    # "umarza" alone means discontinued; next to "oddala" it is usually a partial
    # withdrawal (e.g. of a counterclaim) and the case was decided on the merits.
    if _has(r"\bumarza\b", s) and not _has(r"\b(zasądza|oddala)\b", s) and not upheld:
        return "first", "DISCONTINUED"
    if _has(r"\bzasądza\b", s) or upheld:
        partial = rest_dismissed or _has(r"utrzymuje\s+\w*\s*w\s+mocy[^;]{0,80}w\s+części", s)
        return "first", "PARTIAL" if partial else "GRANTED"
    if _has(r"\boddala\s+powództwo\b|\bpowództw\w*\s+(?:\w+\s+){0,3}oddala\b", s):
        return "first", "DISMISSED"
    return "first", UNKNOWN


_SPACED = re.compile(r"\b(?:[^\W\d_] ){3,}[^\W\d_]\b")


def _unspace(s: str) -> str:
    """Join letter-spaced words: "z a s ą d z a" -> "zasądza" (common in sentencje)."""
    return _SPACED.sub(lambda m: m.group(0).replace(" ", ""), s)


# The court's conclusion is near the end of the reasons, e.g. "apelacja podlegała oddaleniu"
# or "powództwo należało oddalić". Used only when the sentencja is missing (REASONS-only
# documents in SAOS) — less reliable than the sentencja, so records keep the source.
_NOT = r"(?<!nie )"


def reasons_outcome(assessment: str, appeal: bool) -> str:
    tail = _unspace(" ".join(assessment.split()))[-2000:]
    if appeal:
        if _has(r"zmieni\w*\s+(?:\w+\s+){0,2}(?:zaskarżon\w+\s+)?(?:wyrok|orzeczeni)", tail):
            return "CHANGED"
        if _has(r"uchyl\w*\s+(?:\w+\s+){0,2}(?:zaskarżon\w+\s+)?(?:wyrok|orzeczeni)", tail):
            return "SET_ASIDE"
        if _has(
            r"(apelacj|zażaleni|skarg)\w*[^.]{0,120}(oddal|odrzuc|bezzasadn|niezasadn|nie\s+zasługiw)"
            r"|(oddal|odrzuc)\w*\s+(?:\w+\s+){0,2}(apelacj|zażaleni|skarg)",
            tail,
        ):
            return "APPEAL_DISMISSED"
        return UNKNOWN
    if _has(r"umorzy\w*\s+postępowani|umorzeni\w*\s+postępowani", tail):
        return "DISCONTINUED"
    if _has(r"w\s+pozostał\w+\s+(?:zakresie|części)[^.]{0,80}oddal|częściowo\s+zasadn", tail):
        return "PARTIAL"
    if _has(r"powództw\w*[^.]{0,100}(oddal|bezzasadn|niezasadn|nie\s+zasługiw)", tail):
        return "DISMISSED"
    if _has(
        _NOT + r"zasługiwał\w*\s+na\s+uwzględnienie|powództw\w*[^.]{0,80}" + _NOT + r"zasadn"
        r"|utrzyma\w*\s+(?:\w+\s+){0,3}w\s+mocy\s+(?:\w+\s+){0,2}nakaz",
        tail,
    ):
        return "GRANTED"
    return UNKNOWN


_AMOUNT = re.compile(r"kwot[ęy]\s+([\d\s.]+(?:,\d{2})?)\s*(zł|złotych|euro|EUR)", re.IGNORECASE)


def first_amount(sentencja: str) -> tuple[float, str] | None:
    m = _AMOUNT.search(sentencja)
    if not m:
        return None
    raw = re.sub(r"[\s.]", "", m.group(1)).replace(",", ".")
    try:
        value = float(raw)
    except ValueError:
        return None
    currency = "EUR" if m.group(2).lower() in ("euro", "eur") else "PLN"
    return value, currency


# ------------------------------------------------------------------------- records


@dataclass
class CaseRecord:
    id: int
    case_number: str
    court: str
    court_type: str
    judgment_type: str
    date: str
    repertorium: str
    instance: str
    outcome_rule: str
    outcome_source: str  # "sentencja", "reasons" or "" when unknown
    amount: float | None
    currency: str | None
    tsl_score: float
    sentencja: str
    facts: str
    assessment: str
    regulations: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _court_name(j: dict) -> str:
    div = j.get("division") or {}
    court = div.get("court") or {}
    if court.get("name"):
        return court["name"]
    return "Sąd Najwyższy" if j.get("courtType") == "SUPREME" else ""


def to_record(j: dict) -> CaseRecord:
    text = html_to_text(j.get("textContent"))
    sec = split_sections(text)
    case_number = ", ".join(c.get("caseNumber", "") for c in j.get("courtCases") or [])
    rep = repertorium(case_number)
    # The register is more reliable than words in the sentencja.
    instance, outcome = rule_outcome(sec.sentencja, appeal=(rep in CIVIL_APPEAL) if rep else None)
    source = "sentencja" if outcome != UNKNOWN else ""
    if outcome == UNKNOWN and sec.assessment:
        outcome = reasons_outcome(sec.assessment, appeal=instance == "appeal")
        source = "reasons" if outcome != UNKNOWN else ""
    amount = first_amount(sec.sentencja)
    keywords = j.get("keywords") or []
    if isinstance(keywords, dict):  # the API documents "list or dictionary"
        keywords = list(keywords.values())
    return CaseRecord(
        id=int(j["id"]),
        case_number=case_number,
        court=_court_name(j),
        court_type=j.get("courtType", ""),
        judgment_type=j.get("judgmentType", ""),
        date=j.get("judgmentDate", ""),
        repertorium=rep,
        instance=instance,
        outcome_rule=outcome,
        outcome_source=source,
        amount=amount[0] if amount else None,
        currency=amount[1] if amount else None,
        tsl_score=tsl_score(text),
        sentencja=sec.sentencja,
        facts=sec.facts,
        assessment=sec.assessment,
        regulations=[r.get("text", "") for r in j.get("referencedRegulations") or []],
        keywords=[str(k) for k in keywords],
    )
