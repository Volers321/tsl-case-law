from tsl_case_law.llm import load_dotenv


def test_load_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# c\nFOO_KEY=abc  # comment\nEMPTY=\nQUOTED='x y'\nKEEP=fromfile\n")
    monkeypatch.delenv("FOO_KEY", raising=False)
    monkeypatch.delenv("EMPTY", raising=False)
    monkeypatch.setenv("KEEP", "fromenv")
    load_dotenv(env)
    import os

    assert os.environ["FOO_KEY"] == "abc"
    assert "EMPTY" not in os.environ
    assert os.environ["QUOTED"] == "x y"
    assert os.environ["KEEP"] == "fromenv"


def test_rejected_key_gives_a_short_message_and_stops_the_run(tmp_path):
    import pytest

    from tsl_case_law.extract import extract_all
    from tsl_case_law.llm import LLM

    class AuthenticationError(Exception):
        pass

    calls = []

    class Bad(LLM):
        def _call(self, system, user, max_tokens):
            calls.append(1)
            raise AuthenticationError("401 invalid x-api-key")

    cases = [{"id": i, "facts": "fracht", "sentencja": "oddala"} for i in range(40)]
    with pytest.raises(SystemExit) as e:
        extract_all(
            Bad("anthropic", "m"), cases, cache_dir=tmp_path, workers=1, progress=lambda _: None
        )
    assert "API key rejected" in str(e.value)
    assert len(calls) < 5, "queued cases are cancelled after the first rejection"
