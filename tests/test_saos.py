import json

import pytest
import requests

from tsl_case_law.saos import Query, SaosClient, fetch_judgments, search_ids, year_ranges


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    """Serves 2 pages of search hits per court type, then an empty page."""

    def __init__(self, fail_first=0, timeout_first=0, years=False):
        self.headers, self.calls = {}, []
        self.fail_first, self.timeout_first, self.years = fail_first, timeout_first, years

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        if self.timeout_first:
            self.timeout_first -= 1
            raise requests.Timeout("slow")
        if self.fail_first:
            self.fail_first -= 1
            return FakeResponse({}, 503)
        if "/search/" in url:
            page = params["pageNumber"]
            base = 10_000 if params["courtType"] == "COMMON" else 20_000
            if self.years:  # different ids per year so chunks are distinguishable
                base += int(params["judgmentDateFrom"][:4]) * 100_000
            if page >= 2:
                return FakeResponse({"items": []})
            return FakeResponse(
                {"items": [{"id": base + page * 1000 + i} for i in range(params["pageSize"])]}
            )
        jid = int(url.rsplit("/", 1)[1])
        return FakeResponse({"data": {"id": jid, "textContent": "<p>x</p>"}})


def client(session):
    return SaosClient(session, delay=0, sleep=lambda s: None, log=lambda m: None)


def test_search_paginates_until_empty_page_and_unions_queries():
    s = FakeSession()
    c = client(s)
    found = search_ids(
        c, [Query("a", {"all": "CMR"}), Query("b", {"all": "fracht"})], progress=lambda m: None
    )
    # 2 court types x 2 pages x pageSize hits, identical for both queries -> union
    assert len(found) == 2 * 2 * 50  # court types x pages x pageSize
    assert all(v == ["a", "b"] for v in found.values())
    first = s.calls[0][1]
    assert first["judgmentTypes"] == ["SENTENCE", "REASONS"] and first["all"] == "CMR"
    assert "sortingField" not in first, "date sorting made SAOS time out"


def test_retries_on_server_errors():
    c = client(FakeSession(fail_first=2))
    assert c.judgment(7)["id"] == 7


def test_fetch_is_resumable(tmp_path):
    (tmp_path / "1.json").write_text("{}")  # already downloaded
    s = FakeSession()
    n = fetch_judgments(client(s), [1, 2, 3], tmp_path, progress=lambda m: None)
    assert n == 2
    assert [u for u, _ in s.calls] == [
        "https://www.saos.org.pl/api/judgments/2",
        "https://www.saos.org.pl/api/judgments/3",
    ]
    assert json.loads((tmp_path / "3.json").read_text())["id"] == 3
    assert not list(tmp_path.glob("*.part"))


def test_retries_on_timeouts():
    c = client(FakeSession(timeout_first=3))
    assert c.judgment(7)["id"] == 7


def test_gives_up_after_retries():
    c = SaosClient(
        FakeSession(timeout_first=99), delay=0, retries=2, sleep=lambda s: None, log=lambda m: None
    )
    with pytest.raises(requests.Timeout):
        c.judgment(7)


def test_year_ranges():
    assert year_ranges("2023-06-01", "2025-03-15") == [
        ("2023-06-01", "2023-12-31"),
        ("2024-01-01", "2024-12-31"),
        ("2025-01-01", "2025-03-15"),
    ]


def test_search_resumes_from_state_file(tmp_path):
    state = tmp_path / "state.json"
    q = [Query("a", {"all": "CMR"})]

    class Crashing(FakeSession):
        def get(self, url, params=None, timeout=None):
            if params and params.get("judgmentDateFrom", "").startswith("2025"):
                raise RuntimeError("network died")
            return super().get(url, params, timeout)

    with pytest.raises(RuntimeError):
        search_ids(
            client(Crashing(years=True)),
            q,
            date_from="2024-01-01",
            date_to="2025-12-31",
            state_path=state,
            progress=lambda m: None,
        )
    saved = json.loads(state.read_text())
    assert saved["done"] == ["a|2024-01-01|2024-12-31"] and len(saved["found"]) == 200

    s = FakeSession(years=True)
    found = search_ids(
        client(s),
        q,
        date_from="2024-01-01",
        date_to="2025-12-31",
        state_path=state,
        progress=lambda m: None,
    )
    assert len(found) == 400
    assert all(p["judgmentDateFrom"].startswith("2025") for _, p in s.calls), "2024 not re-run"


def test_fetch_skips_missing_judgments(tmp_path):
    class Missing(FakeSession):
        def get(self, url, params=None, timeout=None):
            if url.endswith("/2"):
                return FakeResponse({}, 404)
            return super().get(url, params, timeout)

    n = fetch_judgments(client(Missing()), [1, 2, 3], tmp_path, progress=lambda m: None)
    assert n == 2 and sorted(f.name for f in tmp_path.glob("*.json")) == ["1.json", "3.json"]
