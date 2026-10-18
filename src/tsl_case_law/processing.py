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


# ------------------------------------------------------------------------- sections

_REASONS_HEADER = re.compile(r"^\s*U\s*Z\s*A\s*S\s*A\s*D\s*N\s*I\s*E\s*N\s*I\s*E\s*$", re.M | re.I)
# Phrases that open the court's legal assessment — everything after them may reveal the
# outcome, so it never enters the prediction input.
_ASSESSMENT = re.compile(
    r"(Sąd\s+(?:\w+\s+){0,3}zważył|Sąd\s+(?:\w+\s+){0,3}(?:uznał|stwierdził),?\s+co\s+następuje"
    r"|Sąd\s+(?:\w+\s+){0,3}ocenił|Rozważania\s+prawne|W\s+tak\s+ustalonym\s+stanie\s+faktycznym"
    r"|Apelacja\s+(?:\w+\s+){0,2}(?:jest|okazała\s+się|zasługuje|nie\s+zasługuje)"
    r"|Skarga\s+kasacyjna\s+(?:\w+\s+){0,2}(?:jest|okazała\s+się|zasługuje|nie\s+zasługuje))",
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
    r";|(?:^|\s)(?:\d{1,2}|[IVX]{1,4})[.)]\s"  # numbered points
    r"|\s+i\s+(?=zasądza|nakazuje|obciąża|przyznaje|znosi)"  # "oddala … i zasądza koszty"
)


def _drop_cost_orders(s: str) -> str:
    """Remove operative points about court costs ("zasądza od powoda … kosztów procesu").

    Nearly every dismissal also orders the losing claimant to pay costs, so a naive search
    for "zasądza" would label dismissals as wins.
    """
    points = [p.strip() for p in _POINT_SPLIT.split(s) if p and p.strip()]
    kept = [
        p
        for p in points
        if not re.search(r"koszt|opłat\w* sądow|wynagrodzeni\w* pełnomocnik", p, re.I)
    ]
    return " ; ".join(kept)


def rule_outcome(sentencja: str) -> tuple[str, str]:
    """Return (instance, outcome) inferred from the operative part.

    instance is "appeal" when the court decides on an appeal or cassation, else "first".
    The rules are deliberately simple; their accuracy is measured against human labels.
    """
    s = _drop_cost_orders(" ".join(sentencja.split()))
    appeal = _has(r"apelacj|skarg\w* kasacyjn|zażaleni", s)
    if appeal:
        if _has(r"\bzmienia\b", s):
            return "appeal", "CHANGED"
        if _has(r"\buchyla\b", s):
            return "appeal", "SET_ASIDE"
        if _has(r"\boddala\b", s):
            return "appeal", "APPEAL_DISMISSED"
        return "appeal", UNKNOWN
    if _has(r"\bumarza\b", s) and not _has(r"\bzasądza\b", s):
        return "first", "DISCONTINUED"
    if _has(r"\bzasądza\b", s):
        partial = _has(
            r"(w\s+pozostał\w+\s+(zakresie|części)\s+(powództwo\s+)?oddala|oddala\s+powództwo\s+w\s+pozostał)",
            s,
        )
        return "first", "PARTIAL" if partial else "GRANTED"
    if _has(r"\boddala\s+powództwo\b", s):
        return "first", "DISMISSED"
    return "first", UNKNOWN


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
    instance: str
    outcome_rule: str
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
    instance, outcome = rule_outcome(sec.sentencja)
    amount = first_amount(sec.sentencja)
    keywords = j.get("keywords") or []
    if isinstance(keywords, dict):  # the API documents "list or dictionary"
        keywords = list(keywords.values())
    return CaseRecord(
        id=int(j["id"]),
        case_number=", ".join(c.get("caseNumber", "") for c in j.get("courtCases") or []),
        court=_court_name(j),
        court_type=j.get("courtType", ""),
        judgment_type=j.get("judgmentType", ""),
        date=j.get("judgmentDate", ""),
        instance=instance,
        outcome_rule=outcome,
        amount=amount[0] if amount else None,
        currency=amount[1] if amount else None,
        tsl_score=tsl_score(text),
        sentencja=sec.sentencja,
        facts=sec.facts,
        assessment=sec.assessment,
        regulations=[r.get("text", "") for r in j.get("referencedRegulations") or []],
        keywords=[str(k) for k in keywords],
    )
