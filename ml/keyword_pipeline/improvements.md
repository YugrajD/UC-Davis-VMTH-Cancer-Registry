# keyword_scan pipeline — improvement strategy

---

## Issues

### I-1. Word-order variants not matched
The keyword index builds one regex per term in its canonical word order.
Diagnoses that use a different order for the same words produce no match.

**Example:** `KERATINIZING INFUNDIBULAR ACANTHOMA` vs taxonomy term
`Infundibular keratinizing acanthoma` — same three words, different order.

---

### I-2. Plural forms not matched
Pattern construction uses `\b...\b` with no allowance for a trailing `s`.
Single-word plurals and multi-word phrase plurals both fail.

**Example:** `PLASMACYTOMAS` vs core keyword `plasmacytoma`.

---

### I-3. Modifier-prefixed `-oma` terms not matched
When a diagnosis wraps an `-oma` term in qualifiers that are not part of any
taxonomy keyword (e.g. degree of differentiation, anatomical site, cell type),
no keyword in the index matches the full phrase.  The `-oma` word itself is
sufficient to identify the neoplasm class, but the pipeline currently returns
`no_match` rather than using it as a fallback.

**Examples:**

| Diagnosis | Blocking modifiers |
|---|---|
| `SPINDLE CELL HEPATOCELLULAR CARCINOMA` | "spindle cell" prefix; 4 words exceed the 3-word permutation cap |
| `POORLY DIFFERENTIATED ADENOCARCINOMA` | "poorly differentiated" not in taxonomy |
| `MULTIPLE NODULAR CUTANEOUS PLASMACYTOMAS` | site/count qualifiers prevent phrase match |

---

### I-4. `run_keyword_scan` summary will undercount once a fallback method exists
Lines 174 and 179 of `pipeline.py` hardcode `method == "keyword"` for both
`matched_df` and `match_rate_pct`.  Any result produced by a new fallback
method is silently excluded from the reported rate and top-term counts.

---

## Solutions

### S-1. Word-order permutations in `_build_keyword_index`
For every keyword candidate with 2–3 words, add all word-order permutations
to the index.  (3 words → 6 permutations; 4+ words are left to S-3.)
The `seen` set prevents duplicate entries across labels.

```python
from itertools import permutations as _permutations

# inside _build_keyword_index — replace `for kw in {norm, core}:` block:
candidates: set[str] = set()
for kw in {norm, core}:
    kw = kw.strip()
    candidates.add(kw)
    words = kw.split()
    if 2 <= len(words) <= 3:
        for perm in _permutations(words):
            candidates.add(" ".join(perm))

for kw in candidates:
    if len(kw) < 6 or kw in seen:
        continue
    seen.add(kw)
    pat = re.compile(r"\b" + re.escape(kw) + r"s?\b")   # s? from S-2
    entries.append((kw, pat, i))
```

---

### S-2. Optional trailing `s` on every pattern
Change the one pattern-construction line so every keyword implicitly accepts
its plural form.  This is already incorporated into the snippet in S-1.

```python
# standalone change if applying without S-1:
pat = re.compile(r"\b" + re.escape(kw) + r"s?\b")
```

---

### S-3. `-oma` fallback index (build-time) + fallback pass (runtime)
Build a dedicated `dict[str, int]` at index time that maps each single `-oma`
word found inside taxonomy term names to its label index.  At runtime, fire
this only when the main keyword scan returns `no_match`.

**Why a separate dict instead of re-scanning the keyword index:**  The main
index is sorted by keyword length descending, so a linear scan on a single
extracted word wastes work on every multi-word pattern that cannot possibly
match.  The dedicated dict is O(1) per lookup.

**Winner rule for multi-`-oma` diagnoses:** sort the extracted words by length
descending before looking them up — longer words are more specific (e.g.
`hepatocarcinoma` beats `carcinoma`).  Return the first hit.

```python
_OMA_RE = re.compile(r"\b(\w+oma)s?\b")  # module-level constant


def _build_oma_index(
    taxonomy_labels: list[TaxonomyLabel],
) -> dict[str, int]:
    """Map each single -oma word in any taxonomy term to its label index.
    First occurrence wins (Preferred terms appear before Synonyms in the CSV).
    """
    oma_index: dict[str, int] = {}
    for i, label in enumerate(taxonomy_labels):
        norm = _normalize(label.term)
        for word in norm.split():
            if word.endswith("oma") and word not in oma_index:
                oma_index[word] = i
    return oma_index


def _oma_fallback(
    norm_text: str,
    oma_index: dict[str, int],
    taxonomy_labels: list[TaxonomyLabel],
) -> _MatchResult | None:
    """Return the best -oma fallback match, or None."""
    raw_words = _OMA_RE.findall(norm_text)          # trailing s already stripped
    for word in sorted(set(raw_words), key=len, reverse=True):  # longest first
        if word in oma_index:
            label = taxonomy_labels[oma_index[word]]
            return _MatchResult(
                term=label.term, group=label.group, code=label.code,
                keyword=word, method="oma_fallback",
            )
    return None
```

Wire it into `_match_diagnosis` after the main loop:

```python
def _match_diagnosis(text, keyword_index, taxonomy_labels, oma_index):
    norm_text = _normalize(text)
    for kw, pattern, label_idx in keyword_index:
        if pattern.search(norm_text):
            label = taxonomy_labels[label_idx]
            return _MatchResult(
                term=label.term, group=label.group, code=label.code,
                keyword=kw, method="keyword",
            )
    result = _oma_fallback(norm_text, oma_index, taxonomy_labels)
    return result or _MatchResult(term="", group="", code="", keyword="", method="no_match")
```

`oma_index` is built once in `run_keyword_scan` alongside `keyword_index`:

```python
keyword_index = _build_keyword_index(taxonomy_labels)
oma_index = _build_oma_index(taxonomy_labels)
```

---

### S-4. Fix summary stats in `run_keyword_scan`
Replace the hardcoded `"keyword"` filter with `!= "no_match"` so all matched
methods (including future ones) are counted correctly.

```python
# before
matched_df = out_df[out_df["method"] == "keyword"]
"match_rate_pct": round(100 * method_counts.get("keyword", 0) / max(len(out_df), 1), 1),

# after
matched_df = out_df[out_df["method"] != "no_match"]
"match_rate_pct": round(100 * len(matched_df) / max(len(out_df), 1), 1),
```
