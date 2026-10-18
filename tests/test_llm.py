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
