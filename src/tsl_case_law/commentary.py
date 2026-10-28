"""Case-law commentary written from the judgments themselves.

Published commentaries and glosses (Lex, Legalis, law journals) are copyrighted; court
judgments are not (art. 4 of the Polish Copyright Act). So the commentary here is built
from the courts' own reasoning, at two levels:

* ``annotate``  — one note per judgment: the holding (teza), key arguments, the provisions
  relied on, a practical takeaway for a TSL business, and the legal issues it touches.
* ``overview``  — a mini-commentary on a legal issue across many judgments, every claim
  tied to a case number.

Both are checked automatically: provisions must appear in the judgment's text and every
cited case number must be one of the judgments given to the model.

Leakage: notes are written from the court's assessment, so they are only ever shown for
*precedents* (earlier cases), never as input for predicting the same case.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path

from .extract import parse_json

ANNOTATE_SYSTEM = """Jesteś prawnikiem piszącym krótkie komentarze do orzeczeń w sprawach
transportowo-spedycyjnych (TSL). Dostajesz rozstrzygnięcie i rozważania sądu. Zwróć jeden JSON:
{
  "teza": "1-2 zdania: co sąd uznał za kluczowe i dlaczego (ogólna zasada, nie opis stron)",
  "argumenty": ["maks. 4 krótkie argumenty, na których sąd oparł rozstrzygnięcie"],
  "przepisy": ["przepisy, na które sąd się powołał, np. 'art. 77 ust. 1 Prawa przewozowego'"],
  "wskazowka": "1 zdanie: praktyczny wniosek dla firmy transportowej lub spedycyjnej",
  "zagadnienia": ["2-5 krótkich haseł, np. 'przedawnienie roszczeń przewozowych'"]
}
ZASADY: tylko to, co wynika z tekstu — nie dopisuj przepisów ani faktów spoza niego.
Pisz po polsku, rzeczowo, bez danych osobowych."""

OVERVIEW_SYSTEM = """Jesteś prawnikiem przygotowującym przegląd orzecznictwa dla firm TSL.
Dostajesz zagadnienie i komentarze do kilku wyroków (z sygnaturami). Napisz zwięzły komentarz
(150-300 słów) w punktach:
1. Jak sądy zwykle rozstrzygają to zagadnienie.
2. Od czego zależy wynik / gdzie orzeczenia się różnią.
3. Praktyczne wnioski dla przewoźnika, spedytora i zleceniodawcy.
ZASADY: każde twierdzenie poprzyj sygnaturą w nawiasie kwadratowym, np. [V GC 123/18].
Powołuj WYŁĄCZNIE sygnatury z dostarczonej listy. Jeśli wyroki nie wystarczają do wniosku,
napisz to wprost. To nie jest porada prawna."""

ANNOTATE_VERSION = hashlib.sha1(ANNOTATE_SYSTEM.encode()).hexdigest()[:8]


# ------------------------------------------------------------------ per-judgment notes


def _words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def _tail(text: str, n: int) -> str:
    return " ".join(text.split()[-n:])


def annotation_input(case: dict, max_words: int = 1500) -> str:
    return (
        f"ROZSTRZYGNIĘCIE (koniec sentencji): {_tail(case.get('sentencja', ''), 250)}\n\n"
        f"ROZWAŻANIA SĄDU: {_words(case.get('assessment', ''), max_words)}"
    )


_ART = re.compile(r"art\.?\s*(\d+[a-z]?)", re.IGNORECASE)


def grounded_provisions(provisions: list[str], source: str) -> list[bool]:
    """Is each cited article number actually present in the judgment? A provision without
    an article number (e.g. 'Konwencja CMR') counts as grounded if its name appears."""
    text = " ".join(source.split()).lower()
    out = []
    for p in provisions:
        nums = _ART.findall(p)
        if nums:
            out.append(all(re.search(rf"art\.?\s*{re.escape(n.lower())}\b", text) for n in nums))
        else:
            key = p.lower().strip()[:20]
            out.append(bool(key) and key in text)
    return out


def validate_note(d: dict) -> dict:
    def strs(x, n):
        return [str(s).strip() for s in (x if isinstance(x, list) else []) if str(s).strip()][:n]

    return {
        "teza": str(d.get("teza") or "").strip()[:600],
        "argumenty": strs(d.get("argumenty"), 4),
        "przepisy": strs(d.get("przepisy"), 8),
        "wskazowka": str(d.get("wskazowka") or "").strip()[:300],
        "zagadnienia": strs(d.get("zagadnienia"), 5),
    }


def annotate(llm, case: dict) -> dict:
    for _ in range(2):
        parsed = parse_json(llm.complete(ANNOTATE_SYSTEM, annotation_input(case), max_tokens=2048))
        if parsed and parsed.get("teza"):
            note = validate_note(parsed)
            source = case.get("assessment", "") + " " + case.get("sentencja", "")
            note["przepisy_grounded"] = grounded_provisions(note["przepisy"], source)
            return {"id": case["id"], "ok": True} | note
    return {"id": case["id"], "ok": False}


def annotate_all(
    llm,
    cases: list[dict],
    cache_dir: str | Path = "data/cache/annotate",
    *,
    workers: int = 6,
    progress: Callable[[str], None] = print,
) -> list[dict]:
    """One note per case, cached per prompt version; parallel; Ctrl+C stops at once."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache = Path(cache_dir) / ANNOTATE_VERSION
    cache.mkdir(parents=True, exist_ok=True)
    notes: dict[int, dict] = {}
    todo = []
    for c in cases:
        path = cache / f"{c['id']}.json"
        if path.exists():
            notes[c["id"]] = json.loads(path.read_text(encoding="utf-8"))
        else:
            todo.append(c)
    progress(f"{len(notes)} cached, {len(todo)} to annotate")

    def work(c: dict) -> dict:
        note = annotate(llm, c)
        (cache / f"{c['id']}.json").write_text(json.dumps(note, ensure_ascii=False), "utf-8")
        return note

    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        futures = [pool.submit(work, c) for c in todo]
        for n, fut in enumerate(as_completed(futures), start=1):
            note = fut.result()
            notes[note["id"]] = note
            if n % 25 == 0 or n == len(todo):
                progress(f"annotated {n}/{len(todo)}")
    except SystemExit:  # e.g. rejected API key: stop at once instead of failing every case
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        progress("\nStopped. Finished notes are cached; run the same command to resume.")
        raise SystemExit(130) from None
    pool.shutdown()
    return [notes[c["id"]] for c in cases]


def note_text(note: dict) -> str:
    """What a note contributes to retrieval: the legal issue, not the parties' wording."""
    return " ".join(
        [note.get("teza", ""), *note.get("zagadnienia", []), *note.get("argumenty", [])]
    )


# ------------------------------------------------------------------ issue overviews


class NoteIndex:
    """TF-IDF over notes, to find the judgments that deal with a given legal issue."""

    def __init__(self, cases: dict[int, dict], notes: list[dict]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.items = [(cases[n["id"]], n) for n in notes if n.get("ok") and n["id"] in cases]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform([note_text(n) for _, n in self.items])

    def search(self, query: str, k: int = 12, min_score: float = 0.05):
        sims = (self.matrix @ self.vec.transform([query]).T).toarray().ravel()
        order = sims.argsort()[::-1][:k]
        return [(*self.items[i], float(sims[i])) for i in order if sims[i] >= min_score]


def overview_input(issue: str, hits) -> str:
    blocks = []
    for case, note, _ in hits:
        blocks.append(
            f"[{case['case_number']}] {case.get('court', '')}, {case.get('date', '')[:10]}\n"
            f"Teza: {note['teza']}\nArgumenty: {'; '.join(note['argumenty'])}\n"
            f"Przepisy: {', '.join(note['przepisy'])}"
        )
    return f"ZAGADNIENIE: {issue}\n\nWYROKI:\n\n" + "\n\n".join(blocks)


_CITE = re.compile(r"\[([^\[\]]{3,40}?)\]")


def check_citations(text: str, allowed: set[str]) -> dict:
    """Every [case number] in the overview must be one of the judgments supplied."""
    norm = {" ".join(a.split()): a for a in allowed}
    cited = [" ".join(c.split()) for c in _CITE.findall(text)]
    valid = [c for c in cited if c in norm]
    invalid = sorted({c for c in cited if c not in norm})
    paragraphs = [p for p in re.split(r"\n\s*\n|\n(?=\s*(?:\d+\.|-|•))", text) if p.strip()]
    supported = sum(bool(_CITE.search(p)) for p in paragraphs)
    return {
        "citations": len(cited),
        "valid": len(valid),
        "invalid": invalid,
        "distinct_cases": len(set(valid)),
        "paragraphs_with_citation": f"{supported}/{len(paragraphs)}",
    }


def overview(llm, issue: str, index: NoteIndex, k: int = 12) -> dict:
    hits = index.search(issue, k)
    if not hits:
        return {"text": "", "hits": [], "check": None}
    text = llm.complete(OVERVIEW_SYSTEM, overview_input(issue, hits), max_tokens=4096).strip()
    check = check_citations(text, {c["case_number"] for c, _, _ in hits})
    return {"text": text, "hits": hits, "check": check}
