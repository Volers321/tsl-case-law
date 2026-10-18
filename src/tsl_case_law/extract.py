"""LLM extraction of structured case features.

Two separate calls per case, on purpose:

* ``extract_facts`` reads **only the facts section** and returns party roles, dispute type,
  the claim and the defence arguments, plus a neutral summary. These are the inputs for
  outcome prediction, so the prompt forbids mentioning how the court ruled.
* ``extract_outcome`` reads **only the sentencja** and classifies the outcome. It is the
  LLM alternative to the regex rules in :mod:`processing`; both are scored against the
  expert's labels.

Results are cached per case and per prompt version, so re-running costs nothing and a
prompt change invalidates the cache automatically.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path

from .labeling import DISPUTE_TYPES
from .processing import APPEAL, FIRST_INSTANCE, UNKNOWN

ROLES = ("przewoznik", "spedytor", "zleceniodawca", "nadawca", "odbiorca", "ubezpieczyciel", "inny")
DEFENCES = (
    "przedawnienie",  # limitation period expired
    "brak_reklamacji",  # no (timely) complaint / reservation
    "limit_odpowiedzialnosci",  # liability capped, e.g. CMR art. 23
    "wina_nadawcy",  # sender's fault: packing, loading, documents
    "sila_wyzsza",  # circumstances the carrier could not avoid
    "brak_legitymacji",  # wrong party sued / no standing
    "potracenie",  # set-off against own claims
    "kwestionowanie_wysokosci",  # amount disputed
    "brak_dowodu_szkody",  # damage not proven
    "inne",
)

FACTS_SYSTEM = f"""Jesteś analitykiem sporów w branży transportowo-spedycyjnej (TSL).
Dostajesz WYŁĄCZNIE opis stanu faktycznego z uzasadnienia wyroku. Zwróć jeden obiekt JSON:
{{
  "is_tsl": true/false,                 // czy to spór z umowy przewozu lub spedycji
  "claimant_role": jedna z {list(ROLES)},
  "defendant_role": jedna z {list(ROLES)},
  "dispute_type": jedna z {list(DISPUTE_TYPES)},
  "international": true/false,          // przewóz międzynarodowy (np. CMR)
  "claim_amount": liczba lub null,
  "claim_currency": "PLN" / "EUR" / null,
  "defences": lista z {list(DEFENCES)},
  "summary": "neutralne streszczenie faktów, maks. 80 słów"
}}
ZASADY: nie opisuj, jak orzekł sąd w tej sprawie, i nie oceniaj, kto ma rację — streszczenie
ma zawierać tylko fakty i stanowiska stron. Jeśli czegoś nie da się ustalić, użyj null lub "inny"."""

OUTCOME_SYSTEM = f"""Klasyfikujesz rozstrzygnięcie polskiego wyroku cywilnego na podstawie SENTENCJI.
Pomiń punkty dotyczące wyłącznie kosztów procesu. Zwróć jeden obiekt JSON:
{{"instance": "first" lub "appeal", "outcome": jedna z {list(FIRST_INSTANCE + APPEAL + (UNKNOWN,))}}}
- first: GRANTED (powództwo uwzględnione w całości), PARTIAL (w części), DISMISSED (oddalone), DISCONTINUED (umorzone)
- appeal (apelacja / skarga kasacyjna): APPEAL_DISMISSED (oddalona), CHANGED (wyrok zmieniony), SET_ASIDE (uchylony)"""

PROMPT_VERSION = hashlib.sha1((FACTS_SYSTEM + OUTCOME_SYSTEM).encode()).hexdigest()[:8]
_JSON = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(raw: str) -> dict | None:
    """First JSON object in the reply; tolerates ```json fences and // comments."""
    m = _JSON.search(raw or "")
    if not m:
        return None
    text = re.sub(r"//[^\n]*", "", m.group(0))
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _clip(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words]) + (" […]" if len(words) > max_words else "")


def _choice(value, allowed: tuple[str, ...], default: str) -> str:
    return value if value in allowed else default


def validate_facts(d: dict) -> dict:
    """Coerce the model's answer into the schema; anything off-list becomes a default."""
    amount = d.get("claim_amount")
    try:
        amount = float(amount) if amount is not None else None
    except (TypeError, ValueError):
        amount = None
    return {
        "is_tsl": bool(d.get("is_tsl")),
        "claimant_role": _choice(d.get("claimant_role"), ROLES, "inny"),
        "defendant_role": _choice(d.get("defendant_role"), ROLES, "inny"),
        "dispute_type": _choice(d.get("dispute_type"), DISPUTE_TYPES, "inne"),
        "international": bool(d.get("international")),
        "claim_amount": amount,
        "claim_currency": d.get("claim_currency")
        if d.get("claim_currency") in ("PLN", "EUR")
        else None,
        "defences": [x for x in d.get("defences") or [] if x in DEFENCES],
        "summary": str(d.get("summary") or "")[:1200],
    }


def validate_outcome(d: dict) -> dict:
    instance = d.get("instance") if d.get("instance") in ("first", "appeal") else "first"
    allowed = APPEAL if instance == "appeal" else FIRST_INSTANCE
    return {"instance": instance, "outcome": _choice(d.get("outcome"), allowed, UNKNOWN)}


def _ask(llm, system: str, user: str, validate: Callable[[dict], dict], retries: int = 1) -> dict:
    for _ in range(retries + 1):
        parsed = parse_json(llm.complete(system, user, max_tokens=2048))
        if parsed is not None:
            return validate(parsed) | {"ok": True}
    return {"ok": False}


def extract_facts(llm, case: dict, max_words: int = 1500) -> dict:
    return _ask(
        llm, FACTS_SYSTEM, "STAN FAKTYCZNY:\n" + _clip(case["facts"], max_words), validate_facts
    )


def extract_outcome(llm, case: dict, max_words: int = 600) -> dict:
    return _ask(
        llm, OUTCOME_SYSTEM, "SENTENCJA:\n" + _clip(case["sentencja"], max_words), validate_outcome
    )


def extract_all(
    llm,
    cases: list[dict],
    cache_dir: str | Path = "data/cache/extract",
    *,
    progress: Callable[[str], None] = print,
) -> list[dict]:
    """Extract features for every case, reusing cached results for this prompt version."""
    cache_dir = Path(cache_dir) / PROMPT_VERSION
    cache_dir.mkdir(parents=True, exist_ok=True)
    out, fresh = [], 0
    for n, case in enumerate(cases, start=1):
        path = cache_dir / f"{case['id']}.json"
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
        else:
            row = {
                "id": case["id"],
                "facts_llm": extract_facts(llm, case),
                "outcome_llm": extract_outcome(llm, case),
            }
            path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
            fresh += 1
        out.append(row)
        if n % 20 == 0 or n == len(cases):
            progress(f"extracted {n}/{len(cases)} ({fresh} new calls)")
    return out
