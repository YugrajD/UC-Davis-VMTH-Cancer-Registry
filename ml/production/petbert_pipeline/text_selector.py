"""TF-IDF-based multi-column text selector for PetBERT reports."""

from __future__ import annotations

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

SOURCE_COLS = (
    "HISTOPATHOLOGICAL SUMMARY",
    "FINAL COMMENT",
    "COMMENT",
)

_VET_ABBREVS = frozenset([
    "No.", "no.", "e.g.", "i.e.", "vs.", "Fig.", "fig.",
    "H.P.F.", "h.p.f.", "approx.", "dept.", "Dr.", "dr.",
    "mm.", "cm.", "mL.", "μL.",
])


class TextSelector:
    """Select and compress multi-column report text to a token budget."""

    def __init__(self) -> None:
        self._vectorizer: TfidfVectorizer | None = None

    def fit(self, texts: list[str]) -> None:
        self._vectorizer = TfidfVectorizer(max_features=20_000, sublinear_tf=True)
        self._vectorizer.fit(texts)

    def save(self, path: str) -> None:
        joblib.dump(self._vectorizer, path)

    def load(self, path: str) -> None:
        try:
            self._vectorizer = joblib.load(path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"TF-IDF vectorizer not found at {path!r}. "
                "Run ml/training/contrastive/fit_text_selector.py first."
            )

    def select(self, col_texts: dict[str, str], max_tokens: int = 512) -> str:
        assert self._vectorizer is not None, "Call load() or fit() before select()."

        parts: list[str] = []
        for col in SOURCE_COLS:
            val = col_texts.get(col, "").strip()
            if len(val) >= 10:
                parts.append(f"[{col}] {val}")

        if not parts:
            return ""

        combined = " ".join(parts)

        if len(combined) // 4 <= max_tokens:
            return combined

        sentences = _split_sentences(combined)
        if not sentences:
            return combined

        vec = self._vectorizer.transform(sentences)
        scores = vec.sum(axis=1).A1  # L1 norm per sentence

        budget_chars = max_tokens * 4
        order = scores.argsort()[::-1]
        selected: set[int] = set()
        chars_used = 0
        for idx in order:
            s = sentences[idx]
            if chars_used + len(s) + 1 <= budget_chars:
                selected.add(idx)
                chars_used += len(s) + 1
            if chars_used >= budget_chars:
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


_selector: TextSelector | None = None


def get_selector(vectorizer_path: str) -> TextSelector:
    global _selector
    if _selector is None:
        _selector = TextSelector()
        _selector.load(vectorizer_path)
    return _selector
