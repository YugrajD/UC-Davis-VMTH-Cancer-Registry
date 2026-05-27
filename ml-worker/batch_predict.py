"""Standalone PetBERT batch prediction script for GCP Batch.

Reads env vars for paths, runs the 4-stage PetBERT pipeline, and writes
predictions.json to the output directory. No web server — runs once and exits.
"""

import csv
import json
import os
import re
import sys
import tempfile
from collections import defaultdict

import pandas as pd

sys.path.insert(0, "/ml")

from production.petbert_pipeline.pipeline import run_scan
from production.petbert_pipeline.types import ScanConfig

# Matches |H|SECTION NAME:...|| headers — captures section name (stops at : or |).
_H_HEADER_RE = re.compile(r"\|H\|([^|:]+)[^|]*\|\|", re.IGNORECASE)
# Matches |U|..|| sub-section markers (stripped from section body content).
_U_HEADER_RE = re.compile(r"\|U\|[^|]*\|\|")


def _extract_sections(merged_text: str) -> dict[str, str]:
    """Parse a space-merged pathology report into named sections.

    The upload router concatenates continuation rows into a single text string,
    preserving |H|NAME:|| section headers inline. This function splits on those
    markers and returns a dict keyed by uppercased section name.
    """
    matches = list(_H_HEADER_RE.finditer(merged_text))
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1).strip().upper()
        content_start = m.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(merged_text)
        content = _U_HEADER_RE.sub(" ", merged_text[content_start:content_end]).strip()
        sections[name] = content
    return sections


def _join_numbered(rows: list[dict], field: str) -> str:
    return " ".join(f"{i + 1}) {r.get(field, '')}" for i, r in enumerate(rows))


def main() -> None:
    job_id = os.environ["JOB_ID"]
    input_csv = os.environ["INPUT_CSV_PATH"]
    output_dir = os.environ["OUTPUT_DIR"]
    model_path = os.environ.get("MODEL_PATH", "/tmp/batch_data/models/petbert")
    labels_csv = os.environ.get("LABELS_CSV_PATH", "/tmp/batch_data/models/labels/labels.csv")
    group_classifier = os.environ.get("GROUP_CLASSIFIER_PATH") or None

    # Optional — set to None/empty if the file wasn't downloaded for this bundle.
    _cp = os.environ.get("CASE_PRESENCE_CLASSIFIER_PATH") or ""
    case_presence_classifier = _cp if (_cp and os.path.exists(_cp)) else None
    _lp = os.environ.get("LP_THRESHOLDS_JSON_PATH") or ""
    lp_thresholds_json = _lp if (_lp and os.path.exists(_lp)) else None
    _ug = os.environ.get("UNCOMMON_GROUPS_PATH") or ""
    uncommon_groups = _ug if (_ug and os.path.exists(_ug)) else ""

    if not lp_thresholds_json:
        print("[batch_predict] No lp_thresholds.json — using global LP threshold.")
    if not uncommon_groups:
        print("[batch_predict] No uncommon_groups.txt — treating all groups as common.")

    print(f"[batch_predict] job={job_id} input={input_csv} output={output_dir}")

    # Dataset A has a single 'Text' column whose value is the full pathology
    # report with embedded |H|SECTION NAME:|| markers (produced by the upload
    # router's continuation-row merge step). Parse those markers to populate the
    # three section columns the pipeline's CONCAT_3 embedding expects:
    #   HISTOPATHOLOGICAL SUMMARY / FINAL COMMENT (or COMMENT) / ANCILLARY TESTS
    # Using distinct per-section content avoids the [e, e, e] degenerate
    # embedding that caused the CasePresenceClassifier to reject every patient.
    df = pd.read_csv(input_csv, encoding="latin-1")

    if "Text" in df.columns:
        hist_col, final_col, comment_col, anc_col = [], [], [], []
        for text_val in df["Text"].fillna(""):
            secs = _extract_sections(str(text_val))
            full = str(text_val)
            hist_col.append(secs.get("HISTOPATHOLOGICAL SUMMARY", full))
            final = secs.get("FINAL COMMENT", "")
            final_col.append(final)
            comment_col.append(secs.get("COMMENT") or final)
            anc_col.append(secs.get("ANCILLARY TESTS", ""))

        if "HISTOPATHOLOGICAL SUMMARY" not in df.columns:
            df["HISTOPATHOLOGICAL SUMMARY"] = hist_col
        if "FINAL COMMENT" not in df.columns:
            df["FINAL COMMENT"] = final_col
        if "COMMENT" not in df.columns:
            df["COMMENT"] = comment_col
        if "ANCILLARY TESTS" not in df.columns:
            df["ANCILLARY TESTS"] = anc_col
    else:
        for col in ("HISTOPATHOLOGICAL SUMMARY", "FINAL COMMENT", "COMMENT", "ANCILLARY TESTS"):
            if col not in df.columns:
                df[col] = ""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as tmp:
        df.to_csv(tmp, index=False)
        expanded_csv = tmp.name

    scan_out_dir = os.path.join(output_dir, "scan_output")
    os.makedirs(scan_out_dir, exist_ok=True)

    config_kwargs: dict = dict(
        csv_path=expanded_csv,
        id_col="anon_id",
        model_name=model_path,
        local_only=True,
        out_dir=scan_out_dir,
        max_rows=None,
        batch_size=16,
        max_length=256,
        neighbors_k=3,
        task="categorize",
        embedding_min_sim=0.6,
        device="auto",
        labels_csv_path=labels_csv,
        case_presence_classifier_path=case_presence_classifier,
        group_classifier_path=group_classifier,
        label_presence_thresholds_json=lp_thresholds_json,
    )
    if uncommon_groups:
        config_kwargs["uncommon_groups_path"] = uncommon_groups

    config = ScanConfig(**config_kwargs)

    print("[batch_predict] Starting PetBERT scan...")
    outputs = run_scan(config)
    print(f"[batch_predict] Scan complete: {outputs.predictions_csv}")

    # The new pipeline writes one row per (patient, diagnosis_index). Group by
    # anon_id and aggregate multiple predictions into the numbered format that
    # ingestion_service.parse_predictions expects (e.g. "1) Lymphoma 2) MCT").
    by_patient: dict[str, list[dict]] = defaultdict(list)
    with open(outputs.predictions_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            anon_id = row.get("anon_id", "").strip()
            if anon_id:
                by_patient[anon_id].append(row)

    orig_text_by_id: dict[str, str] = {}
    with open(expanded_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            aid = row.get("anon_id", "").strip()
            if aid and aid not in orig_text_by_id:
                orig_text_by_id[aid] = row.get("Text", "").strip()

    predictions = []
    for anon_id, rows in by_patient.items():
        rows.sort(key=lambda r: int(r.get("diagnosis_index", 1)))
        if len(rows) == 1:
            r = rows[0]
            predictions.append({
                "anon_id": anon_id,
                "original_text": orig_text_by_id.get(anon_id, ""),
                "predicted_term": r.get("predicted_term", ""),
                "predicted_group": r.get("predicted_group", ""),
                "predicted_code": r.get("predicted_code", ""),
                "confidence": r.get("confidence", ""),
                "method": r.get("method", ""),
            })
        else:
            predictions.append({
                "anon_id": anon_id,
                "original_text": orig_text_by_id.get(anon_id, ""),
                "predicted_term": _join_numbered(rows, "predicted_term"),
                "predicted_group": _join_numbered(rows, "predicted_group"),
                "predicted_code": _join_numbered(rows, "predicted_code"),
                "confidence": _join_numbered(rows, "confidence"),
                "method": _join_numbered(rows, "method"),
            })

    predictions_path = os.path.join(output_dir, "predictions.json")
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)

    print(f"[batch_predict] Wrote {len(predictions)} predictions to {predictions_path}")
    os.unlink(expanded_csv)


if __name__ == "__main__":
    main()
