import json

from tsl_case_law.saos import Query, SaosClient, fetch_judgments, search_ids


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

    def __init__(self, fail_first=0):
        self.headers, self.calls, self.fail_first = {}, [], fail_first

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        if self.fail_first:
            self.fail_first -= 1
            return FakeResponse({}, 503)
        if "/search/" in url:
            page = params["pageNumber"]
            base = 10_000 if params["courtType"] == "COMMON" else 20_000
            if page >= 2:
                return FakeResponse({"items": []})
            return FakeResponse(
                {"items": [{"id": base + page * 1000 + i} for i in range(params["pageSize"])]}
            )
        jid = int(url.rsplit("/", 1)[1])
        return FakeResponse({"data": {"id": jid, "textContent": "<p>x</p>"}})


def client(session):
    return SaosClient(session, delay=0, sleep=lambda s: None)


def test_search_paginates_until_empty_page_and_unions_queries():
    s = FakeSession()
    c = client(s)
    found = search_ids(
        c, [Query("a", {"all": "CMR"}), Query("b", {"all": "fracht"})], progress=lambda m: None
    )
    # 2 court types x 2 pages x pageSize hits, identical for both queries -> union
    assert len(found) == 2 * 2 * 100
    assert all(v == ["a", "b"] for v in found.values())
    first = s.calls[0][1]
    assert first["judgmentTypes"] == ["SENTENCE", "REASONS"] and first["all"] == "CMR"


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
