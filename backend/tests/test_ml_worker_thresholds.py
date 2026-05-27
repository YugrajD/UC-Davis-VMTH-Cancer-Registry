"""Threshold plumbing tests for the local and GCP Batch ML worker entrypoints."""

from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_worker_module(filename: str, module_name: str):
    path = REPO_ROOT / "ml-worker" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_batch_predict_scan_config_reads_case_presence_threshold(monkeypatch):
    module = _load_worker_module("batch_predict.py", "batch_predict_under_test")
    monkeypatch.setenv("CASE_PRESENCE_THRESHOLD", "0.3")
    monkeypatch.setenv("GROUP_CLASSIFIER_THRESHOLD", "0.25")

    kwargs = module._build_scan_config_kwargs(
        expanded_csv="/tmp/input.csv",
        scan_out_dir="/tmp/scan_output",
        model_path="/tmp/model",
        labels_csv="/tmp/labels.csv",
        case_presence_classifier="/tmp/case_presence_classifier.pt",
        group_classifier="/tmp/group_classifier_best.pt",
        lp_thresholds_json=None,
        uncommon_groups=None,
    )

    assert kwargs["case_presence_threshold"] == 0.3
    assert kwargs["group_classifier_threshold"] == 0.25


def test_local_worker_scan_config_reads_case_presence_threshold(monkeypatch):
    module = _load_worker_module("app.py", "ml_worker_app_under_test")
    monkeypatch.setenv("CASE_PRESENCE_THRESHOLD", "0.4")
    monkeypatch.setenv("GROUP_CLASSIFIER_THRESHOLD", "0.2")

    kwargs = module._build_scan_config_kwargs(
        expanded_csv_path="/tmp/input.csv",
        out_dir="/tmp/output",
        model_path="/tmp/model",
        labels_csv="/tmp/labels.csv",
        case_presence_classifier="/tmp/case_presence_classifier.pt",
        group_classifier="/tmp/group_classifier_best.pt",
        lp_thresholds_json=None,
        uncommon_groups=None,
    )

    assert kwargs["case_presence_threshold"] == 0.4
    assert kwargs["group_classifier_threshold"] == 0.2
