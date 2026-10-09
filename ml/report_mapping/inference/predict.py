"""Load a generation and predict: ``predict_frame`` (reports in memory; the worker) and ``run_predict``
(report.csv -> predictions CSV).

The taxonomy is never read here: term -> (group, code) comes from the generation's own term_info.json.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import config
import io_utils
from report_mapping import generation, text
from report_mapping.inference import decide, scoring
from report_mapping.model import JointClassifier, load_joint


@dataclass(frozen=True)
class LoadedGeneration:
    model: JointClassifier
    tokenizer: object
    thresholds: decide.Thresholds
    terms: list[str]
    term_info: dict[str, tuple[str, str]]
    max_length: int
    generation_id: str


def load_generation(directory: Path, device) -> LoadedGeneration:
    """Spec-check ``directory`` (files, architecture, calibration), then load the model onto ``device``."""
    directory = Path(directory)
    manifest = generation.verify_spec(directory)
    model, tokenizer = load_joint(directory)
    values = generation.read_thresholds(directory)
    return LoadedGeneration(
        model=model.to(device).eval(), tokenizer=tokenizer,
        thresholds=decide.Thresholds(values["gate_t"], values["tau"], values["rho"]), terms=model.terms,
        term_info=generation.read_term_info(directory), max_length=manifest["model"]["max_length"],
        generation_id=manifest["generation_id"])


def predict_frame(reports: pd.DataFrame, loaded: LoadedGeneration, device) -> pd.DataFrame:
    """Prediction rows (``decide.COLUMNS``) for ``reports``: ``case_id`` plus any subset of the 8 report columns.
    A case with no text in any column gets no rows."""
    texts = text.build_report_texts(reports)
    keep = [i for i, t in enumerate(texts) if t]
    case_ids = reports["case_id"].map(text.clean_text).tolist()
    if not keep:
        return pd.DataFrame(columns=decide.COLUMNS)
    term_logits, gate_logits, _ = scoring.logits_for(loaded.model, loaded.tokenizer, [texts[i] for i in keep],
                                                     loaded.max_length, device)
    ranked = scoring.rank([case_ids[i] for i in keep], loaded.terms, term_logits, gate_logits)
    return decide.decide_all(ranked, loaded.thresholds, loaded.term_info, loaded.generation_id)


def run_predict(generation_dir: Path, device, out: Path | None = None) -> Path:
    """Predict every case in report.csv; write ``<PREDICTIONS_DIR>/<generation_id>_predictions.csv`` (or ``out``)."""
    loaded = load_generation(generation_dir, device)
    reports = text.read_reports()
    predictions = predict_frame(reports, loaded, device)
    if out is None:
        out = config.PREDICTIONS_DIR / f"{loaded.generation_id}_predictions.csv"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(predictions, out)
    print(f"{len(predictions)} prediction rows for {predictions['case_id'].nunique()} of {len(reports)} cases -> {out}")
    return out
