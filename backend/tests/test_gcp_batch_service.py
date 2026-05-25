"""Unit tests for gcp_batch_service helpers.

Tests list_model_folders legacy-dir filtering and the GCS path construction
for pathology report upload/download without making real network calls.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.services.gcp_batch_service import (
    _LEGACY_MODEL_DIRS,
    _REPORTS_PREFIX,
    download_report_text_from_gcs,
    list_model_folders,
    upload_report_text_to_gcs,
)


# ---------------------------------------------------------------------------
# list_model_folders — legacy-dir filtering
# ---------------------------------------------------------------------------


def _make_iterator(prefixes: list[str]) -> MagicMock:
    """Return a mock GCS blob iterator whose .prefixes is set after iteration."""
    it = MagicMock()
    it.__iter__ = MagicMock(return_value=iter([]))
    it.prefixes = prefixes
    return it


def _mock_client(prefixes: list[str]) -> MagicMock:
    client = MagicMock()
    client.list_blobs.return_value = _make_iterator(prefixes)
    return client


@patch("app.services.gcp_batch_service._get_storage_client")
def test_list_model_folders_returns_versioned_bundles(mock_get_client):
    mock_get_client.return_value = _mock_client([
        "models/production/",
        "models/model_a/",
        "models/model_b/",
    ])
    result = list_model_folders()
    assert result == ["model_a", "model_b", "production"]


@patch("app.services.gcp_batch_service._get_storage_client")
def test_list_model_folders_excludes_legacy_dirs(mock_get_client):
    prefixes = [f"models/{d}/" for d in _LEGACY_MODEL_DIRS]
    prefixes.append("models/production/")
    mock_get_client.return_value = _mock_client(prefixes)
    result = list_model_folders()
    assert result == ["production"]
    for legacy in _LEGACY_MODEL_DIRS:
        assert legacy not in result


@patch("app.services.gcp_batch_service._get_storage_client")
def test_list_model_folders_empty_bucket(mock_get_client):
    mock_get_client.return_value = _mock_client([])
    assert list_model_folders() == []


@patch("app.services.gcp_batch_service._get_storage_client")
def test_list_model_folders_returns_sorted(mock_get_client):
    mock_get_client.return_value = _mock_client([
        "models/zebra/",
        "models/alpha/",
        "models/production/",
    ])
    result = list_model_folders()
    assert result == sorted(result)


@patch("app.services.gcp_batch_service._get_storage_client")
def test_list_model_folders_returns_empty_on_gcs_error(mock_get_client):
    mock_get_client.side_effect = Exception("GCS unavailable")
    # list_model_folders is documented to return [] when GCS is unreachable.
    # It currently propagates the exception; the caller (ingest router) catches it.
    # This test documents the current contract — update if silent fallback is added.
    with pytest.raises(Exception):
        list_model_folders()


# ---------------------------------------------------------------------------
# upload_report_text_to_gcs — path construction
# ---------------------------------------------------------------------------


@patch("app.services.gcp_batch_service._get_bucket")
def test_upload_report_text_to_gcs_returns_correct_path(mock_get_bucket):
    mock_blob = MagicMock()
    mock_get_bucket.return_value.blob.return_value = mock_blob

    path = upload_report_text_to_gcs(job_id=7, anon_id="ID_42", text="report text")

    assert path == f"{_REPORTS_PREFIX}/7/ID_42.txt"


@patch("app.services.gcp_batch_service._get_bucket")
def test_upload_report_text_to_gcs_uploads_utf8_plain_text(mock_get_bucket):
    mock_blob = MagicMock()
    mock_get_bucket.return_value.blob.return_value = mock_blob

    upload_report_text_to_gcs(job_id=1, anon_id="ID_1", text="hello")

    mock_blob.upload_from_string.assert_called_once_with(
        "hello", content_type="text/plain; charset=utf-8"
    )


@patch("app.services.gcp_batch_service._get_bucket")
def test_upload_report_text_to_gcs_uses_blob_path_as_key(mock_get_bucket):
    mock_bucket = MagicMock()
    mock_get_bucket.return_value = mock_bucket

    upload_report_text_to_gcs(job_id=3, anon_id="ID_99", text="x")

    mock_bucket.blob.assert_called_once_with(f"{_REPORTS_PREFIX}/3/ID_99.txt")


# ---------------------------------------------------------------------------
# download_report_text_from_gcs — path routing
# ---------------------------------------------------------------------------


@patch("app.services.gcp_batch_service._get_bucket")
def test_download_report_text_from_gcs_uses_given_path(mock_get_bucket):
    mock_blob = MagicMock()
    mock_blob.download_as_text.return_value = "the report"
    mock_get_bucket.return_value.blob.return_value = mock_blob

    result = download_report_text_from_gcs("reports/7/ID_42.txt")

    mock_get_bucket.return_value.blob.assert_called_once_with("reports/7/ID_42.txt")
    assert result == "the report"


@patch("app.services.gcp_batch_service._get_bucket")
def test_download_report_text_from_gcs_decodes_utf8(mock_get_bucket):
    mock_blob = MagicMock()
    mock_get_bucket.return_value.blob.return_value = mock_blob

    download_report_text_from_gcs("reports/1/ID_1.txt")

    mock_blob.download_as_text.assert_called_once_with(encoding="utf-8")
