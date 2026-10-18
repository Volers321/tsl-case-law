"""Client for SAOS — System Analizy Orzeczeń Sądowych (https://www.saos.org.pl).

SAOS publishes anonymised Polish court judgments through a free REST API. Court rulings
are excluded from copyright in Poland (art. 4 of the Copyright Act), so they can be
processed freely; we still keep raw downloads out of git and fetch them on demand.

Two steps:
1. ``search_ids`` — run several topical queries against ``/api/search/judgments`` and
   collect judgment ids (the search API returns only excerpts).
2. ``fetch_judgments`` — download each full judgment from ``/api/judgments/{id}`` into
   ``data/raw/{id}.json``. Already-downloaded ids are skipped, so the step is resumable.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import requests

BASE = "https://www.saos.org.pl/api"
USER_AGENT = "tsl-case-law/0.1 (+https://github.com/Volers321/tsl-case-law)"


@dataclass(frozen=True)
class Query:
    """One topical search. ``params`` go straight to the SAOS search API."""

    name: str
    params: dict = field(default_factory=dict)


# Transport & forwarding disputes are found through the statutes they cite and through
# characteristic vocabulary. Each query is run separately and the ids are unioned.
TSL_QUERIES: tuple[Query, ...] = (
    Query("prawo_przewozowe", {"referencedRegulation": "Prawo przewozowe"}),
    Query("konwencja_cmr", {"referencedRegulation": "międzynarodowego przewozu drogowego towarów"}),
    Query("cmr_przewoznik", {"all": "CMR przewoźnik"}),
    Query("umowa_spedycji", {"all": "umowa spedycji spedytor"}),
    Query("fracht_list_przewozowy", {"all": "fracht list przewozowy"}),
)

JUDGMENT_TYPES = ("SENTENCE", "REASONS")  # wyroki i uzasadnienia; skip procedural orders


class SaosClient:
    def __init__(
        self,
        session: requests.Session | None = None,
        *,
        delay: float = 0.3,
        retries: int = 5,
        timeout: float = 120,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = print,
    ):
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self.delay, self.retries, self.timeout = delay, retries, timeout
        self._sleep, self._log = sleep, log

    def _get(self, url: str, params: dict | None = None) -> dict:
        """GET with retries on timeouts, dropped connections and 429/5xx (SAOS is slow
        and occasionally overloaded). Backs off 2, 4, 8, 16, 32 s."""
        for attempt in range(self.retries + 1):
            last = attempt == self.retries
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError) as e:
                if last:
                    raise
                self._log(f"  … {type(e).__name__}, retrying in {2 ** (attempt + 1)} s")
                self._sleep(2 ** (attempt + 1))
                continue
            if resp.status_code in (429, 500, 502, 503, 504) and not last:
                self._log(f"  … HTTP {resp.status_code}, retrying in {2 ** (attempt + 1)} s")
                self._sleep(2 ** (attempt + 1))
                continue
            resp.raise_for_status()
            self._sleep(self.delay)  # be polite to a free public service
            return resp.json()
        raise RuntimeError("unreachable")

    def search(
        self,
        query: Query,
        *,
        date_from: str | None = None,
        date_to: str | None = None,
        court_types: Iterable[str] = ("COMMON", "SUPREME"),
        page_size: int = 50,
        max_pages: int = 100,
    ) -> Iterator[dict]:
        """Yield search hits page by page until a page comes back short or empty.

        Uses the default (database-id) ordering: sorting by date made SAOS time out.
        """
        for court_type in court_types:
            for page in range(max_pages):
                params = {
                    **query.params,
                    "courtType": court_type,
                    "judgmentTypes": list(JUDGMENT_TYPES),
                    "pageSize": page_size,
                    "pageNumber": page,
                }
                if date_from:
                    params["judgmentDateFrom"] = date_from
                if date_to:
                    params["judgmentDateTo"] = date_to
                items = _items(self._get(f"{BASE}/search/judgments", params))
                if not items:
                    break
                yield from items
                if len(items) < page_size:
                    break

    def judgment(self, judgment_id: int) -> dict:
        data = self._get(f"{BASE}/judgments/{judgment_id}")
        return data.get("data", data)


def _items(payload: dict) -> list[dict]:
    # The documented field is "items"; tolerate other wrappers rather than fail silently.
    for key in ("items", "data", "results"):
        if isinstance(payload.get(key), list):
            return payload[key]
    return []


def year_ranges(date_from: str, date_to: str | None = None) -> list[tuple[str, str]]:
    """Split [date_from, date_to] into calendar years — smaller, faster SAOS queries."""
    start = int(date_from[:4])
    end_date = date_to or time.strftime("%Y-%m-%d")
    end = int(end_date[:4])
    ranges = []
    for year in range(start, end + 1):
        lo = date_from if year == start else f"{year}-01-01"
        hi = end_date if year == end else f"{year}-12-31"
        ranges.append((lo, hi))
    return ranges


def search_ids(
    client: SaosClient,
    queries: Iterable[Query] = TSL_QUERIES,
    *,
    date_from: str = "2015-01-01",
    date_to: str | None = None,
    state_path: str | Path | None = None,
    progress: Callable[[str], None] = print,
) -> dict[int, list[str]]:
    """Return {judgment_id: [names of queries that found it]}.

    Work is split into (query, year) chunks. With ``state_path`` the result is saved after
    every chunk and finished chunks are skipped on the next run, so an interrupted search
    resumes where it stopped instead of starting over.
    """
    state = {"done": [], "found": {}}
    if state_path and Path(state_path).exists():
        state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    found = {int(k): v for k, v in state["found"].items()}
    done = set(state["done"])
    for q in queries:
        n_before = len(found)
        for lo, hi in year_ranges(date_from, date_to):
            key = f"{q.name}|{lo}|{hi}"
            if key in done:
                continue
            for hit in client.search(q, date_from=lo, date_to=hi):
                names = found.setdefault(int(hit["id"]), [])
                if q.name not in names:
                    names.append(q.name)
            done.add(key)
            if state_path:
                tmp = Path(str(state_path) + ".part")
                tmp.write_text(json.dumps({"done": sorted(done), "found": found}), encoding="utf-8")
                tmp.rename(state_path)
            progress(f"{q.name:25s} {lo[:4]}  total ids: {len(found)}")
        progress(f"{q.name:25s} done: +{len(found) - n_before} new ids")
    return found


def fetch_judgments(
    client: SaosClient,
    ids: Iterable[int],
    out_dir: str | Path = "data/raw",
    *,
    limit: int | None = None,
    progress: Callable[[str], None] = print,
) -> int:
    """Download full judgments not yet on disk. Returns the number newly downloaded."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in ids if not (out_dir / f"{i}.json").exists()]
    if limit is not None:
        todo = todo[:limit]
    skipped = 0
    for n, judgment_id in enumerate(todo, start=1):
        try:
            data = client.judgment(judgment_id)
        except requests.HTTPError as e:  # e.g. a judgment withdrawn from SAOS (404)
            skipped += 1
            progress(f"  skip {judgment_id}: {e}")
            continue
        tmp = out_dir / f"{judgment_id}.json.part"
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.rename(out_dir / f"{judgment_id}.json")  # atomic: no half-written files
        if n % 25 == 0 or n == len(todo):
            progress(f"downloaded {n}/{len(todo)}")
    return len(todo) - skipped
