"""Outcome prediction from the facts, with similar past cases as cited precedents.

Task: given the facts of a first-instance TSL dispute, how likely is the claimant to win in
full, win in part, or lose? Four methods are compared on the same **temporal split** —
trained / retrieved on older judgments, tested on the newest — so no method ever sees the
future and every precedent predates the case it is used for:

* ``majority``   — always the most common outcome (the bar to beat)
* ``tfidf``      — logistic regression on TF-IDF of the facts + extracted features
* ``llm``        — an LLM reads a neutral summary of the facts and gives probabilities
* ``llm-rag``    — the same, plus the k most similar earlier cases and how they ended

Inputs never include the sentencja or the court's assessment (see :mod:`processing`).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .extract import CONDUCT, parse_json

LABELS = ("win", "partial", "lose")
OUTCOME_TO_LABEL = {"GRANTED": "win", "PARTIAL": "partial", "DISMISSED": "lose"}
LABEL_PL = {"win": "wygrana w całości", "partial": "wygrana w części", "lose": "przegrana"}


# ------------------------------------------------------------------------- examples


@dataclass
class Example:
    id: int
    case_number: str
    date: str
    facts: str
    summary: str
    label: str
    features: dict = field(default_factory=dict)
    note: dict | None = None  # commentary written from the court's reasoning (precedents only)
    appeal: str = "not_found"  # upheld / changed / set_aside / not_found (see appeals.py)

    def feature_tokens(self) -> str:
        """Structured features as pseudo-words, so one TF-IDF model can use both."""
        toks = [f"F_type_{self.features.get('dispute_type', 'inne')}"]
        toks += [f"F_def_{d}" for d in self.features.get("defences", [])]
        for k, v in (self.features.get("conduct") or {}).items():
            if v is not None:
                toks.append(f"F_{k}_{'yes' if v else 'no'}")
        if self.features.get("international"):
            toks.append("F_international")
        return " ".join(toks)


def gold_outcome(case: dict, extracted: dict | None) -> str:
    """Court outcome used as ground truth: the rule label, or the LLM's reading of the
    sentencja when the rules could not tell (both were checked against the expert)."""
    if case.get("outcome_rule") not in (None, "", "UNKNOWN"):
        return case["outcome_rule"]
    return ((extracted or {}).get("outcome_llm") or {}).get("outcome") or "UNKNOWN"


def build_examples(
    cases: list[dict],
    extracted: dict[int, dict],
    notes: dict[int, dict] | None = None,
    links: dict[int, dict] | None = None,
    *,
    drop_reversed: bool = True,
) -> list[Example]:
    """First-instance TSL cases with a known win / partial / lose outcome, oldest first.

    Without LLM extraction every case is kept (keyword filter only); with it, only cases
    the LLM classed as TSL. With appeal ``links``, judgments changed or set aside on appeal
    are dropped (their outcome is not the final one) unless ``drop_reversed`` is False.
    """
    from .appeals import REVERSED, appeal_status

    out = []
    for c in cases:
        status = appeal_status(c["id"], links or {})
        if drop_reversed and status in REVERSED:
            continue
        ext = extracted.get(c["id"])
        facts_llm = (ext or {}).get("facts_llm") or {}
        if extracted and not facts_llm.get("is_tsl"):
            continue
        if c.get("instance") != "first":
            continue
        label = OUTCOME_TO_LABEL.get(gold_outcome(c, ext))
        if not label:
            continue
        out.append(
            Example(
                id=c["id"],
                case_number=c.get("case_number", ""),
                date=c.get("date", ""),
                facts=c["facts"],
                summary=facts_llm.get("summary") or " ".join(c["facts"].split()[:120]),
                label=label,
                features={k: facts_llm.get(k) for k in ("dispute_type", "defences", "conduct")}
                | {"international": facts_llm.get("international")},
                note=(notes or {}).get(c["id"])
                if (notes or {}).get(c["id"], {}).get("ok")
                else None,
                appeal=status,
            )
        )
    return sorted(out, key=lambda e: e.date)


def temporal_split(examples: list[Example], test_frac: float = 0.2):
    """Oldest (1 - test_frac) for training / retrieval, newest test_frac for testing."""
    n_test = max(1, round(len(examples) * test_frac))
    return examples[:-n_test], examples[-n_test:]


# ------------------------------------------------------------------------- retrieval


class SimilarCases:
    """TF-IDF cosine similarity over past cases.

    ``by_issue=False`` compares the wording of the facts. ``by_issue=True`` indexes each
    past case by its facts summary *plus* the commentary note (holding, issues, arguments),
    so a query matches cases about the same legal question rather than similar words.
    Queries are always facts only.
    """

    def __init__(self, examples: list[Example], by_issue: bool = False):
        from sklearn.feature_extraction.text import TfidfVectorizer

        from .commentary import note_text

        self.examples = examples
        self.vec = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=1, max_df=0.9, sublinear_tf=True
        )
        if by_issue:
            docs = [e.summary + " " + (note_text(e.note) if e.note else "") for e in examples]
        else:
            docs = [_clip(e.facts, 1500) for e in examples]
        self.matrix = self.vec.fit_transform(docs)

    def search(self, text: str, k: int = 5, exclude_id: int | None = None):
        sims = (self.matrix @ self.vec.transform([_clip(text, 1500)]).T).toarray().ravel()
        order = sims.argsort()[::-1]
        hits = [(self.examples[i], float(sims[i])) for i in order if sims[i] > 0]
        return [h for h in hits if h[0].id != exclude_id][:k]


def _clip(text: str, max_words: int) -> str:
    return " ".join(text.split()[:max_words])


# ------------------------------------------------------------------------- predictors


def predict_majority(train: list[Example], test: list[Example]) -> list[dict]:
    counts = Counter(e.label for e in train)
    probs = {lab: counts[lab] / len(train) for lab in LABELS}
    return [dict(probs) for _ in test]


def predict_tfidf(train: list[Example], test: list[Example]) -> list[dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    def doc(e: Example) -> str:
        return _clip(e.facts, 1500) + " " + e.feature_tokens()

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)
    x_train = vec.fit_transform([doc(e) for e in train])
    model = LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0)
    model.fit(x_train, [e.label for e in train])
    proba = model.predict_proba(vec.transform([doc(e) for e in test]))
    return [
        {
            lab: float(p[list(model.classes_).index(lab)]) if lab in model.classes_ else 0.0
            for lab in LABELS
        }
        for p in proba
    ]


PREDICT_SYSTEM = """Jesteś doświadczonym prawnikiem od sporów transportowo-spedycyjnych (TSL).
Na podstawie opisu faktów sprawy w pierwszej instancji oceń szanse POWODA. Zwróć jeden JSON:
{"win": p, "partial": p, "lose": p, "reason": "2-3 zdania", "precedents": [numery użytych spraw]}
- win = powództwo uwzględnione w całości, partial = w części, lose = oddalone; p sumują się do 1
- podaj prawdopodobieństwa, nie pewność: sprawy sądowe są niepewne
- jeśli dostajesz PODOBNE SPRAWY, opieraj się na nich tylko wtedy, gdy są naprawdę podobne,
  i wymień w "precedents" numery tych, z których skorzystałeś"""

PREDICT_VERSION = hashlib.sha1(PREDICT_SYSTEM.encode()).hexdigest()[:8]


def case_prompt(summary: str, features: dict | None = None) -> str:
    f = features or {}
    lines = [f"OPIS SPRAWY: {summary}"]
    if f.get("dispute_type"):
        lines.append(f"Rodzaj sporu: {f['dispute_type']}")
    if f.get("defences"):
        lines.append("Zarzuty pozwanego: " + ", ".join(f["defences"]))
    conduct = {k: v for k, v in (f.get("conduct") or {}).items() if v is not None}
    if conduct:
        lines.append(
            "Zachowanie stron przed procesem: "
            + ", ".join(f"{k}={'tak' if v else 'nie'}" for k, v in conduct.items())
        )
    return "\n".join(lines)


def precedents_prompt(hits: list[tuple[Example, float]], with_notes: bool = False) -> str:
    if not hits:
        return ""
    blocks = []
    for n, (e, _) in enumerate(hits, start=1):
        upheld = ", utrzymany w apelacji" if e.appeal == "upheld" else ""
        block = (
            f"[{n}] {e.case_number} ({e.date[:4]}{upheld}) — wynik: {LABEL_PL[e.label]}\n"
            f"{e.summary}"
        )
        if with_notes and e.note:
            block += f"\nTeza sądu: {e.note['teza']}"
        blocks.append(block)
    return "PODOBNE SPRAWY Z PRZESZŁOŚCI:\n" + "\n\n".join(blocks) + "\n\n"


def parse_probs(raw: str) -> dict | None:
    d = parse_json(raw)
    if not d:
        return None
    try:
        p = {lab: max(0.0, float(d.get(lab, 0))) for lab in LABELS}
    except (TypeError, ValueError):
        return None
    total = sum(p.values())
    if total <= 0:
        return None
    out = {lab: v / total for lab, v in p.items()}
    out["reason"] = str(d.get("reason") or "")
    out["precedents"] = [x for x in d.get("precedents") or [] if isinstance(x, int | str)]
    return out


def predict_llm(
    llm,
    train: list[Example],
    test: list[Example],
    *,
    k: int = 0,
    by_issue: bool = False,
    repeat: int = 0,
    cache_dir: str | Path = "data/cache/predict",
    workers: int = 6,
    progress: Callable[[str], None] = print,
) -> list[dict]:
    """LLM probabilities; with k > 0 the k most similar *training* cases are included.

    ``repeat`` = n runs the same prompts again into a separate cache: LLM answers vary
    between calls, and that run-to-run spread has to be measured, not ignored.
    """
    from concurrent.futures import ThreadPoolExecutor

    retriever = SimilarCases(train, by_issue=by_issue) if k else None
    # Precedents depend on which cases are in the training pool, so it is part of the key.
    pool_key = hashlib.sha1(",".join(str(e.id) for e in train).encode()).hexdigest()[:8]
    mode = "-issue" if by_issue else ""
    cache = Path(cache_dir) / f"{PREDICT_VERSION}-k{k}{mode}" / (pool_key if k else "no-pool")
    if repeat:
        cache = cache / f"r{repeat}"
    cache.mkdir(parents=True, exist_ok=True)
    uniform = {lab: 1 / len(LABELS) for lab in LABELS}

    def one(e: Example) -> dict:
        path = cache / f"{e.id}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        query = e.summary if by_issue else e.facts
        hits = retriever.search(query, k) if retriever else []
        user = precedents_prompt(hits, with_notes=by_issue) + case_prompt(e.summary, e.features)
        p = None
        for _ in range(2):
            p = parse_probs(llm.complete(PREDICT_SYSTEM, user, max_tokens=2048))
            if p:
                break
        p = (p or uniform | {"failed": True}) | {"retrieved": [h[0].id for h in hits]}
        path.write_text(json.dumps(p, ensure_ascii=False), encoding="utf-8")
        return p

    pool = ThreadPoolExecutor(max_workers=max(1, workers))  # no `with`: see extract_all
    try:
        results = list(pool.map(one, test))
    except SystemExit:  # e.g. rejected API key: stop at once instead of failing every case
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        progress("\nStopped. Finished predictions are cached; run again to resume.")
        raise SystemExit(130) from None
    pool.shutdown()
    progress(
        f"  llm k={k}{mode}: {len(results)} predictions, "
        f"{sum('failed' in r for r in results)} failed"
    )
    return results


# ------------------------------------------------------------------------- metrics


def evaluate(gold: list[str], probs: list[dict]) -> dict:
    """Accuracy, macro-F1 and Brier score (multi-class; 0 is perfect, lower is better)."""
    pred = [max(LABELS, key=lambda lab: p[lab]) for p in probs]
    acc = sum(p == g for p, g in zip(pred, gold, strict=True)) / len(gold)
    f1s = []
    for lab in LABELS:
        tp = sum(p == lab and g == lab for p, g in zip(pred, gold, strict=True))
        fp = sum(p == lab and g != lab for p, g in zip(pred, gold, strict=True))
        fn = sum(p != lab and g == lab for p, g in zip(pred, gold, strict=True))
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    brier = sum(
        sum((p[lab] - (g == lab)) ** 2 for lab in LABELS) for p, g in zip(probs, gold, strict=True)
    ) / len(gold)
    # The question a business usually asks: will the claimant get anything at all?
    p_any = [p["win"] + p["partial"] for p in probs]
    g_any = [g != "lose" for g in gold]
    acc_any = sum((p >= 0.5) == g for p, g in zip(p_any, g_any, strict=True)) / len(gold)
    brier_any = sum((p - g) ** 2 for p, g in zip(p_any, g_any, strict=True)) / len(gold)
    return {
        "n": len(gold),
        "accuracy": round(acc, 3),
        "macro_f1": round(sum(f1s) / len(f1s), 3) if f1s else 0.0,
        "brier": round(brier, 3),
        "any_win_accuracy": round(acc_any, 3),
        "any_win_brier": round(brier_any, 3),
    }


def bootstrap_ci(gold: list[str], probs: list[dict], n: int = 1000, seed: int = 7):
    """95% interval for accuracy — with ~100 test cases the uncertainty is large."""
    import random

    rng = random.Random(seed)
    pred = [max(LABELS, key=lambda lab: p[lab]) for p in probs]
    hits = [int(p == g) for p, g in zip(pred, gold, strict=True)]
    accs = sorted(sum(rng.choices(hits, k=len(hits))) / len(hits) for _ in range(n))
    return round(accs[int(0.025 * n)], 3), round(accs[int(0.975 * n) - 1], 3)


# ------------------------------------------------------------------------- conduct stats


def conduct_table(examples: list[Example], min_n: int = 5) -> list[dict]:
    """How often the claimant wins (fully or partly) depending on pre-trial conduct."""
    base = sum(e.label != "lose" for e in examples) / max(len(examples), 1)
    rows = []
    for k in CONDUCT:
        for value in (True, False):
            group = [e for e in examples if (e.features.get("conduct") or {}).get(k) is value]
            if len(group) >= min_n:
                rate = sum(e.label != "lose" for e in group) / len(group)
                rows.append(
                    {
                        "feature": k,
                        "value": value,
                        "n": len(group),
                        "claimant_wins": rate,
                        "vs_base": rate - base,
                    }
                )
    return rows


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion — honest error bars for small groups."""
    if n == 0:
        return 0.0, 1.0
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def average_probs(runs: list[list[dict]]) -> list[dict]:
    """Mean of several runs' probabilities per case — a cheap ensemble that usually
    calibrates better than any single run."""
    return [
        {lab: sum(r[i][lab] for r in runs) / len(runs) for lab in LABELS}
        for i in range(len(runs[0]))
    ]


# ------------------------------------------------------------------------- case memo

MEMO_SYSTEM = """Jesteś doświadczonym prawnikiem od sporów TSL (przewóz, spedycja, CMR). Przygotuj
dla firmy krótką, jasną ocenę sprawy. Dostajesz opis faktów, prognozę szans i podobne wyroki
(z numerami [1], [2]…, ich wynikami i tezami sądów). Zwróć jeden JSON:
{
  "najbardziej_prawdopodobny_wynik": "1-2 zdania, konkretnie: co powód prawdopodobnie dostanie",
  "pewnosc": "niska" / "średnia" / "wysoka",
  "kluczowe_pytanie": "1 zdanie: okoliczność, od której wynik zależy najbardziej",
  "za_powodem": ["maks. 4 argumenty"],
  "za_pozwanym": ["maks. 4 argumenty"],
  "przepisy": ["przepisy, które sąd najpewniej zastosuje, z krótkim wyjaśnieniem"],
  "scenariusze": [{"jesli": "warunek", "to": "wynik"}],
  "do_ustalenia": ["maks. 6 dowodów lub faktów do zebrania"],
  "ryzyka": ["maks. 3 największe ryzyka"],
  "precedensy": [numery wyroków, które naprawdę pasują i na których się opierasz]
}
ZASADY: pisz po polsku, rzeczowo. Powołuj tylko przepisy, których jesteś pewien. Nie wymyślaj
faktów spoza opisu — braki nazwij w "do_ustalenia". Wyroki o niskim podobieństwie traktuj
ostrożnie. To ocena szans dla strony sporu, nie wyrok i nie porada prawna."""

WEAK_SIMILARITY = 0.08


def memo_prompt(text: str, probs: dict, hits) -> str:
    chances = ", ".join(f"{LABEL_PL[lab]} {probs[lab]:.0%}" for lab in LABELS)
    lines = []
    for n, (e, score) in enumerate(hits, start=1):
        weak = " (SŁABO PODOBNY)" if score < WEAK_SIMILARITY else ""
        teza = f" Teza: {e.note['teza']}" if e.note else ""
        lines.append(f"[{n}] {e.case_number}, wynik: {LABEL_PL[e.label]}{weak}. {e.summary}{teza}")
    return (
        f"OPIS SPRAWY:\n{text.strip()}\n\nPROGNOZA SZANS POWODA: {chances}\n\n"
        "PODOBNE WYROKI:\n" + "\n".join(lines)
    )


def parse_memo(raw: str, n_hits: int) -> dict | None:
    d = parse_json(raw)
    if not d or not d.get("najbardziej_prawdopodobny_wynik"):
        return None

    def strs(key, n):
        v = d.get(key)
        return [str(x).strip() for x in v if str(x).strip()][:n] if isinstance(v, list) else []

    scen = [
        {"jesli": str(s.get("jesli", "")).strip(), "to": str(s.get("to", "")).strip()}
        for s in d.get("scenariusze") or []
        if isinstance(s, dict) and s.get("jesli")
    ][:4]
    prec = sorted({int(x) for x in d.get("precedensy") or [] if str(x).isdigit()})
    return {
        "wynik": str(d["najbardziej_prawdopodobny_wynik"]).strip(),
        "pewnosc": d.get("pewnosc")
        if d.get("pewnosc") in ("niska", "średnia", "wysoka")
        else "niska",
        "kluczowe_pytanie": str(d.get("kluczowe_pytanie") or "").strip(),
        "za_powodem": strs("za_powodem", 4),
        "za_pozwanym": strs("za_pozwanym", 4),
        "przepisy": strs("przepisy", 6),
        "scenariusze": scen,
        "do_ustalenia": strs("do_ustalenia", 6),
        "ryzyka": strs("ryzyka", 3),
        # a model citing a precedent that was not supplied is caught here
        "precedensy": [n for n in prec if 1 <= n <= n_hits],
        "precedensy_spoza_listy": [n for n in prec if not 1 <= n <= n_hits],
    }


def render_memo(memo: dict, probs: dict, hits) -> str:
    """Markdown memo; the same text is printed and can be saved with --out."""
    out = ["# Ocena sprawy (prognoza, nie porada prawna)", ""]
    out += [f"**Najbardziej prawdopodobny wynik:** {memo['wynik']}", ""]
    out += [
        f"**Pewność:** {memo['pewnosc']}  ·  szanse powoda: "
        + ", ".join(f"{LABEL_PL[lab]} {probs[lab]:.0%}" for lab in LABELS),
        "",
    ]
    if memo["kluczowe_pytanie"]:
        out += [f"**Kluczowe pytanie:** {memo['kluczowe_pytanie']}", ""]
    for title, key in (
        ("Za powodem", "za_powodem"),
        ("Za pozwanym", "za_pozwanym"),
        ("Przepisy", "przepisy"),
        ("Co ustalić", "do_ustalenia"),
        ("Największe ryzyka", "ryzyka"),
    ):
        if memo[key]:
            out += [f"## {title}"] + [f"- {x}" for x in memo[key]] + [""]
    if memo["scenariusze"]:
        out += ["## Scenariusze"] + [
            f"- Jeśli {s['jesli']} → {s['to']}" for s in memo["scenariusze"]
        ]
        out.append("")
    out.append("## Podobne wyroki")
    for n, (e, score) in enumerate(hits, start=1):
        used = "★ " if n in memo["precedensy"] else ""
        weak = " — słabo pasuje" if score < WEAK_SIMILARITY else ""
        upheld = ", utrzymany w apelacji" if e.appeal == "upheld" else ""
        teza = f"  \n  Teza: {e.note['teza']}" if e.note else ""
        out.append(
            f"- {used}[{n}] {e.case_number} ({e.date[:4]}{upheld}) — {LABEL_PL[e.label]}"
            f"{weak}{teza}"
        )
    if memo["precedensy_spoza_listy"]:
        out.append(
            f"\n⚠ Model powołał wyroki spoza listy: {memo['precedensy_spoza_listy']} (pominięte)"
        )
    out += [
        "",
        "★ = wyrok, na którym opiera się ocena. Trafność prognoz na teście: ok. 40% "
        "przy trzech możliwych wynikach — traktuj to jako punkt wyjścia, nie rozstrzygnięcie.",
    ]
    return "\n".join(out)
