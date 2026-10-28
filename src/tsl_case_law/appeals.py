"""Link first-instance judgments to the appeals against them.

An appellate judgment names the judgment under appeal: "na skutek apelacji pozwanego od
wyroku Sądu Rejonowego w Radomiu z dnia 14 września 2015 r., sygn. akt V GC 657/14". With
that link every first-instance case in the dataset gets one of:

* ``upheld``      — appeal dismissed: the outcome stood (a confirmed, high-quality label)
* ``changed``     — the appellate court changed the judgment: the first-instance outcome is
                    not the final one
* ``set_aside``   — set aside and sent back: no final outcome yet
* ``not_found``   — no appeal in the dataset. This does **not** prove the judgment is final:
                    the appeal may exist but not be published in SAOS or not be TSL-tagged.

Changed and set-aside judgments are excluded from training, testing and the precedent pool
by default — they are exactly the rulings that "were not right in the end".
"""

from __future__ import annotations

import re
from collections import Counter

from .processing import CIVIL_FIRST

STATUS_OF_APPEAL = {
    "APPEAL_DISMISSED": "upheld",
    "CHANGED": "changed",
    "SET_ASIDE": "set_aside",
}
REVERSED = {"changed", "set_aside"}

_REGS = "|".join(sorted(CIVIL_FIRST, key=len, reverse=True))
# "V GC 657/14", "IX GNc 12/15", optionally preceded by "sygn. akt"
_SIG = re.compile(rf"\b([IVXL]{{1,6}}\s+(?:{_REGS})\s+\d{{1,6}}\s*/\s*\d{{2,4}})\b")
_LOWER_COURT = re.compile(
    r"Sąd\w*\s+(Rejonow\w+|Okręgow\w+)\s+(?:dla\s+)?(?:w\s+)?"
    r"([A-ZĄĆĘŁŃÓŚŹŻ][\wąćęłńóśźż]+(?:[- ][A-ZĄĆĘŁŃÓŚŹŻ][\wąćęłńóśźż]+)?)"
)


def norm_sig(sig: str) -> str:
    return re.sub(r"\s*/\s*", "/", " ".join(sig.split()))


def _city(text: str) -> str | None:
    m = _LOWER_COURT.search(text)
    return m.group(2).lower() if m else None


def lower_court_refs(appeal: dict) -> list[tuple[str, str | None]]:
    """(signature, city of the lower court) pairs mentioned in an appellate judgment.

    The sentencja is searched first (it names the judgment under appeal); the opening of
    the reasons is the fallback ("Wyrokiem z dnia … Sąd Rejonowy w … (sygn. akt …)").
    """
    own = norm_sig(appeal.get("case_number", ""))
    for text in (appeal.get("sentencja", ""), " ".join(appeal.get("facts", "").split()[:400])):
        refs = []
        for m in _SIG.finditer(text):
            sig = norm_sig(m.group(1))
            if sig != own:
                refs.append((sig, _city(text[max(0, m.start() - 250) : m.start()])))
        if refs:
            return refs
    return []


def link_appeals(cases: list[dict], outcome_of=None) -> dict[int, dict]:
    """{first-instance case id: {"status", "appeal_id", "appeal_case_number"}}.

    A signature such as "V GC 657/14" exists in many district courts, so a match needs the
    same city when the appeal names one, or a signature that is unique in the dataset.
    ``outcome_of(case)`` gives an appeal's outcome (default: the rule label).
    """
    outcome_of = outcome_of or (lambda c: c.get("outcome_rule", "UNKNOWN"))
    first = [c for c in cases if c.get("instance") == "first"]
    by_sig: dict[str, list[dict]] = {}
    for c in first:
        by_sig.setdefault(norm_sig(c.get("case_number", "")), []).append(c)

    links: dict[int, dict] = {}
    for appeal in (c for c in cases if c.get("instance") == "appeal"):
        status = STATUS_OF_APPEAL.get(outcome_of(appeal))
        if not status:
            continue
        for sig, city in lower_court_refs(appeal):
            candidates = by_sig.get(sig, [])
            if city:
                candidates = [c for c in candidates if city in c.get("court", "").lower()]
            if len(candidates) != 1:
                continue
            target = candidates[0]
            if appeal.get("date", "") < target.get("date", ""):
                continue  # an appeal cannot predate the judgment it reviews
            prev = links.get(target["id"])
            if prev is None or appeal.get("date", "") > prev["appeal_date"]:
                links[target["id"]] = {
                    "status": status,
                    "appeal_id": appeal["id"],
                    "appeal_case_number": appeal.get("case_number", ""),
                    "appeal_date": appeal.get("date", ""),
                }
    return links


def appeal_status(case_id: int, links: dict[int, dict]) -> str:
    return links.get(case_id, {}).get("status", "not_found")


def reversal_stats(cases: list[dict], links: dict[int, dict], key=None) -> dict:
    """How often first-instance judgments that were appealed got changed or set aside."""
    key = key or (lambda c: "all")
    groups: dict[str, Counter] = {}
    for c in cases:
        if c.get("instance") != "first":
            continue
        groups.setdefault(key(c), Counter())[appeal_status(c["id"], links)] += 1
    out = {}
    for g, cnt in groups.items():
        appealed = cnt["upheld"] + cnt["changed"] + cnt["set_aside"]
        out[g] = dict(cnt) | {
            "appealed": appealed,
            "reversal_rate": (cnt["changed"] + cnt["set_aside"]) / appealed if appealed else None,
        }
    return out
