# Run inference

How to produce predictions for the whole report corpus with `predict.py`, where they land and what
they contain. For maintainers. The model and the decision rule are in
[report-mapping.md](../concepts/report-mapping.md); every flag is in
[scripts-and-flags.md](../reference/scripts-and-flags.md).

## Predict with a generation

```bash
# the production generation
ml/.venv/Scripts/python.exe ml/scripts/predict.py --generation current --device cuda

# a trained and calibrated candidate
ml/.venv/Scripts/python.exe ml/scripts/predict.py --generation candidate --device cuda
```

`--generation` is `current` (the default), `candidate`, or a literal generation directory. The
script runs the spec check on the generation, reads `config.REPORT_CSV`, builds each report's text
from the eight text columns, scores every case that has text in one encoder pass, applies the
generation's own thresholds from `thresholds.json` and writes one predictions CSV. It prints the
number of prediction rows and of cases that got one. There is no embedding step and no cache: nothing
is stored between runs except the CSV.

The spec check refuses a generation whose file hashes do not match its manifest, whose architecture
(including the text spec) is not the code's, or whose `calibration.status` is not `calibrated` for
its own weights ([generations.md](../concepts/generations.md#the-spec-check)). A refusal prints
`REFUSED: ...` and exits 1.

| Flag | Use |
|---|---|
| `--generation` | `current` (default), `candidate` or a generation directory. |
| `--device` | `auto` (default: cuda when available, else cpu), `cpu` or `cuda`. |
| `--out` | Write the predictions CSV to this path instead of the default. |

## Where predictions land

By default `output/predictions/<generation_id>_predictions.csv` (`config.PREDICTIONS_DIR`), UTF-8,
one row per predicted diagnosis:

`case_id, diagnosis_index, predicted_term, predicted_group, predicted_code, case_presence_prob,
confidence, group_prob, method, generation_id`

`generation_id` stamps which generation produced the row. `case_presence_prob` is the gate
probability (four decimals) and `confidence` the term's probability (two decimals); `group_prob` is
always blank. `method` is `term_head` (the gate accepted the case: the rank-1 term, plus a rank-2 row
when both `tau` and `rho` allow it) or `rejected_by_case_presence` (the gate said non-cancer: one row,
term and group `Non-Cancer`, no code, confidence `0.00`). A case with no text in any of the eight
columns gets no row. The file holds report-derived predictions, not report text. It is the input to
`evaluate.py`, `promote.py` and `code_cases.py`; see [evaluation.md](../concepts/evaluation.md) and
[coding.md](../concepts/coding.md). `code_cases.py combine` and `queue` need the generation's
`thresholds.json` as well as the CSV (they take `--generation`). The `predictions` set is synced to
S3 ([sync-with-s3.md](sync-with-s3.md)).

## Other ways to score

The same scoring runs in memory for the cloud worker: `predict_frame` takes a frame of reports (a
`case_id` column plus any subset of the eight columns) and `predict_texts` takes already-built texts.
A test pins `predict_frame` to the rows `run_predict` writes. The worker's input and response are in
[handoff-contracts.md](../reference/handoff-contracts.md).

On CPU the model runs in fp32; on CUDA the encoder autocasts to bf16 and the heads stay fp32. On the
whole calibration partition the two differ by at most 0.07 pp on each of Good, Slight, CO, FP and FN,
and 21 of 5,828 cases change decision. CPU timing with the worker's pins (torch 2.9.1, 8 threads) was
0.72 s per report in chunked calls and 0.41 s in a single call. A CUDA device is still the way to
predict the whole corpus.

_Last verified against code: 2026-10-09_
