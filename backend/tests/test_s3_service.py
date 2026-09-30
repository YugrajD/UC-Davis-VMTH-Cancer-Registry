"""Unit tests for s3_service helpers (boto3 client is mocked; no network)."""

import io
import json
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from app.services.s3_service import (
    _LEGACY_MODEL_DIRS,
    cleanup_job_files,
    csv_exists,
    download_petbert_summary,
    download_predictions,
    download_report_text,
    job_prefix,
    list_model_folders,
    upload_csv,
    upload_report_text,
)


def _body(data: bytes) -> dict:
    return {"Body": io.BytesIO(data)}


def _paginator(pages: list[dict]) -> MagicMock:
    client = MagicMock()
    client.get_paginator.return_value.paginate.return_value = pages
    return client


# --- list_model_folders -----------------------------------------------------


@patch("app.services.s3_service._get_client")
def test_list_model_folders_returns_sorted_versioned_bundles(mock_get_client):
    mock_get_client.return_value = _paginator([
        {"CommonPrefixes": [{"Prefix": "models/production/"}, {"Prefix": "models/alpha/"}]},
    ])
    assert list_model_folders() == ["alpha", "production"]


@patch("app.services.s3_service._get_client")
def test_list_model_folders_excludes_legacy_dirs(mock_get_client):
    prefixes = [{"Prefix": f"models/{d}/"} for d in _LEGACY_MODEL_DIRS]
    prefixes.append({"Prefix": "models/production/"})
    mock_get_client.return_value = _paginator([{"CommonPrefixes": prefixes}])
    assert list_model_folders() == ["production"]


@patch("app.services.s3_service._get_client")
def test_list_model_folders_empty_bucket(mock_get_client):
    mock_get_client.return_value = _paginator([{}])
    assert list_model_folders() == []


# --- upload / download ------------------------------------------------------


@patch("app.services.s3_service._get_client")
def test_upload_csv_uses_job_prefix_key(mock_get_client, monkeypatch):
    monkeypatch.setattr("app.services.s3_service.settings.S3_BUCKET", "bucket")
    key = upload_csv(7, "dataset_a.csv", b"a,b")
    assert key == "uploads/7/dataset_a.csv" == f"{job_prefix(7)}/dataset_a.csv"
    mock_get_client.return_value.put_object.assert_called_once_with(
        Bucket="bucket", Key=key, Body=b"a,b", ContentType="text/csv"
    )


@patch("app.services.s3_service._get_client")
def test_upload_report_text_returns_key_and_encodes_utf8(mock_get_client, monkeypatch):
    monkeypatch.setattr("app.services.s3_service.settings.S3_BUCKET", "bucket")
    key = upload_report_text(job_id=7, anon_id="ID_42", text="héllo")
    assert key == "reports/7/ID_42.txt"
    kwargs = mock_get_client.return_value.put_object.call_args.kwargs
    assert kwargs["Body"] == "héllo".encode("utf-8")
    assert kwargs["ContentType"] == "text/plain; charset=utf-8"


@patch("app.services.s3_service._get_client")
def test_download_report_text_decodes_utf8(mock_get_client, monkeypatch):
    monkeypatch.setattr("app.services.s3_service.settings.S3_BUCKET", "bucket")
    mock_get_client.return_value.get_object.return_value = _body("héllo".encode("utf-8"))
    assert download_report_text("reports/1/ID_1.txt") == "héllo"
    mock_get_client.return_value.get_object.assert_called_once_with(
        Bucket="bucket", Key="reports/1/ID_1.txt"
    )


@patch("app.services.s3_service._get_client")
def test_download_predictions_parses_json(mock_get_client):
    mock_get_client.return_value.get_object.return_value = _body(json.dumps([{"anon_id": "A"}]).encode())
    assert download_predictions(3) == [{"anon_id": "A"}]


@patch("app.services.s3_service._get_client")
def test_download_petbert_summary_reads_scan_output_path(mock_get_client, monkeypatch):
    monkeypatch.setattr("app.services.s3_service.settings.S3_BUCKET", "bucket")
    mock_get_client.return_value.get_object.return_value = _body(
        b'{"prediction_method_counts": {"low_confidence": 2}}'
    )
    summary = download_petbert_summary(9)
    assert summary["prediction_method_counts"] == {"low_confidence": 2}
    assert mock_get_client.return_value.get_object.call_args.kwargs["Key"] == (
        "uploads/9/scan_output/petbert_summary.json"
    )


@patch("app.services.s3_service._get_client")
def test_download_petbert_summary_returns_empty_when_missing(mock_get_client):
    mock_get_client.return_value.get_object.side_effect = Exception("NoSuchKey")
    assert download_petbert_summary(9) == {}


@patch("app.services.s3_service._get_client")
def test_download_petbert_summary_returns_empty_on_invalid_json(mock_get_client):
    mock_get_client.return_value.get_object.return_value = _body(b"not json")
    assert download_petbert_summary(9) == {}


# --- csv_exists / cleanup ---------------------------------------------------


@patch("app.services.s3_service._get_client")
def test_csv_exists_false_on_404(mock_get_client):
    mock_get_client.return_value.head_object.side_effect = ClientError(
        {"Error": {"Code": "404"}}, "HeadObject"
    )
    assert csv_exists(1) is False


@patch("app.services.s3_service._get_client")
def test_csv_exists_reraises_other_errors(mock_get_client):
    mock_get_client.return_value.head_object.side_effect = ClientError(
        {"Error": {"Code": "403"}}, "HeadObject"
    )
    with pytest.raises(ClientError):
        csv_exists(1)


@patch("app.services.s3_service._get_client")
def test_cleanup_job_files_deletes_listed_objects(mock_get_client, monkeypatch):
    monkeypatch.setattr("app.services.s3_service.settings.S3_BUCKET", "bucket")
    client = _paginator([{"Contents": [{"Key": "uploads/5/a"}, {"Key": "uploads/5/b"}]}])
    mock_get_client.return_value = client
    cleanup_job_files(5)
    client.delete_objects.assert_called_once_with(
        Bucket="bucket",
        Delete={"Objects": [{"Key": "uploads/5/a"}, {"Key": "uploads/5/b"}], "Quiet": True},
    )


@patch("app.services.s3_service._get_client")
def test_cleanup_job_files_noop_when_empty(mock_get_client):
    client = _paginator([{}])
    mock_get_client.return_value = client
    cleanup_job_files(5)
    client.delete_objects.assert_not_called()
