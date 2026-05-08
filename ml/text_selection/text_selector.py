"""TF-IDF-based multi-column text selector for PetBERT reports."""

from __future__ import annotations

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

SOURCE_COLS = (
    "HISTOPATHOLOGICAL SUMMARY",
    "ANCILLARY TESTS",
    "COMMENT",
    "FINAL COMMENT",
    "ADDENDUM",
    "GROSS DESCRIPTION",
    "CLINICAL ABSTRACT",
)

_VET_ABBREVS = frozenset([
    "No.", "no.", "e.g.", "i.e.", "vs.", "Fig.", "fig.",
    "H.P.F.", "h.p.f.", "approx.", "dept.", "Dr.", "dr.",
    "mm.", "cm.", "mL.", "μL.",
])


class TextSelector:
    """Select and compress multi-column report text to a token budget."""

    def __init__(self) -> None:
        self._vectorizers: dict[str, TfidfVectorizer] = {}

    def fit(self, col_to_texts: dict[str, list[str]]) -> None:
        self._vectorizers = {}
        for col, texts in col_to_texts.items():
            v = TfidfVectorizer(max_features=20_000, sublinear_tf=True)
            v.fit(texts)
            self._vectorizers[col] = v

    def save(self, path: str) -> None:
        joblib.dump(self._vectorizers, path)

    def load(self, path: str) -> None:
        try:
            self._vectorizers = joblib.load(path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"TF-IDF vectorizers not found at {path!r}. "
                "Run ml/training/contrastive/fit_text_selector.py first."
            )

    def select(self, col_texts: dict[str, str], max_tokens: int = 512) -> str:
        assert self._vectorizers, "Call load() or fit() before select()."
        budget_chars = max_tokens * 4
        parts: list[str] = []
        used = 0

        for col in SOURCE_COLS:
            val = col_texts.get(col, "").strip()
            if len(val) < 10:
                continue
            sentences = _split_sentences(f"[{col}] {val}")
            col_chars = sum(len(s) + 1 for s in sentences)
            remaining = budget_chars - used
            if col_chars <= remaining:
                parts.extend(sentences)
                used += col_chars
            elif remaining >= 200:
                parts.append(_tfidf_select(self._vectorizers[col], sentences, remaining))
                used = budget_chars
            if used >= budget_chars:
                break

        return " ".join(parts)


def _tfidf_select(vectorizer: TfidfVectorizer, sentences: list[str], budget_chars: int) -> str:
    vec = vectorizer.transform(sentences)
    scores = vec.sum(axis=1).A1
    order = scores.argsort()[::-1]
    selected: set[int] = set()
    used = 0
    for idx in order:
        s = sentences[idx]
        if used + len(s) + 1 <= budget_chars:
            selected.add(idx)
            used += len(s) + 1
        if used >= budget_chars:
            break
    return " ".join(sentences[i] for i in sorted(selected))


def _split_sentences(text: str) -> list[str]:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    result: list[str] = []
    for line in lines:
        if len(line) <= 200:
            result.append(line)
        else:
            result.extend(_sentence_split(line))
    return result


def _sentence_split(text: str) -> list[str]:
    """Split on '. ' while protecting known veterinary abbreviations."""
    parts = text.split(". ")
    sentences: list[str] = []
    buf = parts[0]
    for part in parts[1:]:
        last_word = (buf.rsplit(None, 1)[-1] + ".") if buf.strip() else ""
        if last_word in _VET_ABBREVS:
            buf = buf + ". " + part
        else:
            sentences.append(buf.strip())
            buf = part
    if buf.strip():
        sentences.append(buf.strip())
    return sentences


_selector_cache: dict[str, TextSelector] = {}


def get_selector(vectorizer_path: str) -> TextSelector:
    if vectorizer_path not in _selector_cache:
        sel = TextSelector()
        sel.load(vectorizer_path)
        _selector_cache[vectorizer_path] = sel
    return _selector_cache[vectorizer_path]
