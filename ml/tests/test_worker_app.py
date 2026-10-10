"""ml-worker/app.py over HTTP on a tiny generation. Skipped when fastapi (the worker's web stack, installed by
ml-worker/requirements.txt, not by ml/requirements.txt) is not importable."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="fastapi is a worker dependency, not installed in the ml venv")
pytest.importorskip("httpx", reason="fastapi's TestClient needs httpx")
pytest.importorskip("multipart", reason="UploadFile needs python-multipart")
from fastapi.testclient import TestClient  # noqa: E402

from report_mapping.generation import verify_spec  # noqa: E402

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ml-worker").is_dir())


@pytest.fixture
def client(monkeypatch, tiny_generation_dir):
    import torch

    threads = torch.get_num_threads()
    monkeypatch.setenv("PETBERT_MODEL_PATH", str(tiny_generation_dir / "petbert"))
    monkeypatch.setenv("TORCH_NUM_THREADS", "1")
    spec = importlib.util.spec_from_file_location("ml_worker_app", _REPO_ROOT / "ml-worker" / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield TestClient(module.app)
    torch.set_num_threads(threads)


def _post(client, csv: bytes, name="dataset_a.csv"):
    return client.post("/predict", files={"file": (name, io.BytesIO(csv), "text/csv")})


def test_health_reports_the_generation_id(client, tiny_generation_dir):
    generation_id = verify_spec(tiny_generation_dir)["generation_id"]
    assert client.get("/health").json() == {"status": "ok", "source_version": generation_id}


def test_predict_returns_the_backend_payload(client):
    response = _post(client, b"anon_id,Text\nA,mast cell tumor present\nB,\nC,carcinoma malignant\n")
    assert response.status_code == 200
    body = response.json()
    assert [p["anon_id"] for p in body["predictions"]] == ["A", "C"]
    assert set(body["predictions"][0]) == {"anon_id", "original_text", "predicted_term", "predicted_group",
                                            "predicted_code", "confidence", "method", "source_version"}
    assert body["input_rows"] >= 2


def test_predict_refuses_bad_uploads(client):
    assert _post(client, b"anon_id,Text\nA,x\n", name="dataset_a.txt").status_code == 400
    assert _post(client, b"").status_code == 400
    assert _post(client, b"id,Text\nA,x\n").status_code == 400
