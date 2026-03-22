# Step 1 — Rename Packages

## Actions

**Rename directories:**
```
mv ml/petbert_scan   ml/petbert_pipeline
mv ml/keyword_scan   ml/keyword_pipeline
```

**`petbert_pipeline/io.py`** — rename output file string literals:
| Old | New |
|---|---|
| `petbert_scan_predictions.csv` | `petbert_predictions.csv` |
| `petbert_scan_provenance.csv` | `petbert_provenance.csv` |
| `petbert_scan_similarity_scores.csv` | `petbert_similarity_scores.csv` |
| `petbert_scan_visualization.csv` | `petbert_visualization.csv` |
| `petbert_scan_column_scores.csv` | `petbert_column_scores.csv` |
| `petbert_scan_neighbors.csv` | `petbert_neighbors.csv` |
| `petbert_scan_embeddings.npz` | `petbert_embeddings.npz` |
| `petbert_scan_summary.json` | `petbert_summary.json` |

**`petbert_pipeline/__main__.py`** — update docstring mention of `python -m petbert_scan` → `python -m petbert_pipeline`

**Add `keyword_pipeline/__main__.py`** (currently missing — needed for `python -m keyword_pipeline`):
```python
from .cli import main
import sys
sys.exit(main())
```

All intra-package imports use relative `.` imports — no changes needed inside either package.

## Verification
```bash
ml/.venv/bin/python3 -m petbert_pipeline --help
ml/.venv/bin/python3 -m keyword_pipeline --help
```
