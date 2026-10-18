import json

from tsl_case_law.extract import (
    PROMPT_VERSION,
    extract_all,
    extract_facts,
    extract_outcome,
    parse_json,
    validate_facts,
)


class FakeLLM:
    """Answers like a model would: fenced JSON with comments, sometimes off-schema values."""

    provider, model = "fake", "fake"

    def __init__(self):
        self.calls = []

    def complete(self, system, user, max_tokens=800):
        self.calls.append((system, user))
        if "SENTENCJI" in system:
            return '{"instance": "first", "outcome": "PARTIAL"}'
        return """```json
{
  "is_tsl": true,
  "claimant_role": "przewoznik",   // carrier sues
  "defendant_role": "spedytor",
  "dispute_type": "zaplata_frachtu",
  "international": true,
  "claim_amount": "6000",
  "claim_currency": "PLN",
  "defences": ["limit_odpowiedzialnosci", "made_up_defence"],
  "summary": "Przewoźnik żąda frachtu, spedytor powołuje się na opóźnienie."
}
```"""


CASE = {
    "id": 1,
    "facts": "przewoźnik " * 3000,
    "sentencja": "zasądza; w pozostałym zakresie oddala",
}


def test_parse_json_variants():
    assert parse_json('Odp: {"a": 1} koniec') == {"a": 1}
    assert parse_json("{'a': 1}") is None
    assert parse_json("") is None


def test_validate_facts_coerces_and_filters():
    v = validate_facts(
        {"claimant_role": "kierowca", "claim_amount": "abc", "defences": ["inne", "x"]}
    )
    assert v["claimant_role"] == "inny" and v["claim_amount"] is None and v["defences"] == ["inne"]
    assert v["dispute_type"] == "inne" and v["is_tsl"] is False


def test_facts_prompt_sees_only_facts_and_is_clipped():
    llm = FakeLLM()
    out = extract_facts(llm, CASE, max_words=100)
    system, user = llm.calls[0]
    assert "zasądza" not in user, "sentencja must never reach the facts extractor"
    assert len(user.split()) < 110
    assert out["ok"] and out["claim_amount"] == 6000.0
    assert out["defences"] == ["limit_odpowiedzialnosci"]


def test_outcome_from_sentencja():
    assert extract_outcome(FakeLLM(), CASE) == {
        "instance": "first",
        "outcome": "PARTIAL",
        "ok": True,
    }


def test_unparseable_answers_are_retried_then_flagged():
    class Broken(FakeLLM):
        def complete(self, system, user, max_tokens=800):
            self.calls.append(1)
            return "Nie wiem."

    llm = Broken()
    assert extract_outcome(llm, CASE) == {"ok": False}
    assert len(llm.calls) == 2


def test_extract_all_caches_per_prompt_version(tmp_path):
    llm = FakeLLM()
    rows = extract_all(llm, [CASE, dict(CASE, id=2)], tmp_path, progress=lambda m: None)
    assert len(rows) == 2 and len(llm.calls) == 4
    assert (tmp_path / PROMPT_VERSION / "1.json").exists()
    again = extract_all(llm, [CASE], tmp_path, progress=lambda m: None)
    assert len(llm.calls) == 4, "second run must come from cache"
    assert again[0] == json.loads((tmp_path / PROMPT_VERSION / "1.json").read_text())
