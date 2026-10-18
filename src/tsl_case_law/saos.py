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
        retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self.delay, self.retries, self._sleep = delay, retries, sleep

    def _get(self, url: str, params: dict | None = None) -> dict:
        for attempt in range(self.retries + 1):
            resp = self.session.get(url, params=params, timeout=60)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.retries:
                self._sleep(2**attempt)  # back off: 1, 2, 4, 8 s
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
        page_size: int = 100,
        max_pages: int = 50,
    ) -> Iterator[dict]:
        """Yield search hits page by page until a page comes back empty."""
        for court_type in court_types:
            for page in range(max_pages):
                params = {
                    **query.params,
                    "courtType": court_type,
                    "judgmentTypes": list(JUDGMENT_TYPES),
                    "pageSize": page_size,
                    "pageNumber": page,
                    "sortingField": "JUDGMENT_DATE",
                    "sortingDirection": "DESC",
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


def search_ids(
    client: SaosClient,
    queries: Iterable[Query] = TSL_QUERIES,
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    progress: Callable[[str], None] = print,
) -> dict[int, list[str]]:
    """Return {judgment_id: [names of queries that found it]}."""
    found: dict[int, list[str]] = {}
    for q in queries:
        n_before = len(found)
        for hit in client.search(q, date_from=date_from, date_to=date_to):
            found.setdefault(int(hit["id"]), []).append(q.name)
        progress(f"{q.name:25s} +{len(found) - n_before} new ids (total {len(found)})")
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
    for n, judgment_id in enumerate(todo, start=1):
        data = client.judgment(judgment_id)
        tmp = out_dir / f"{judgment_id}.json.part"
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.rename(out_dir / f"{judgment_id}.json")  # atomic: no half-written files
        if n % 25 == 0 or n == len(todo):
            progress(f"downloaded {n}/{len(todo)}")
    return len(todo)
