"""Unit tests for ml_task_service (ECS client is mocked; no network)."""

from unittest.mock import patch

import pytest

from app.services import ml_task_service as svc


@pytest.fixture(autouse=True)
def _ml_settings(monkeypatch):
    monkeypatch.setattr(svc.settings, "ECS_CLUSTER_ARN", "arn:cluster")
    monkeypatch.setattr(svc.settings, "ML_TASK_DEFINITION_ARN", "arn:taskdef")
    monkeypatch.setattr(svc.settings, "ML_TASK_SUBNET_IDS", "subnet-a, subnet-b")
    monkeypatch.setattr(svc.settings, "ML_TASK_SECURITY_GROUP_ID", "sg-1")
    monkeypatch.setattr(svc.settings, "ML_TASK_CONTAINER_NAME", "ml-worker")


@patch("app.services.ml_task_service._get_client")
def test_submit_ml_task_passes_network_and_env(mock_get_client, monkeypatch):
    monkeypatch.setattr(svc.settings, "CASE_PRESENCE_THRESHOLD", 0.3)
    monkeypatch.setattr(svc.settings, "GROUP_CLASSIFIER_THRESHOLD", 0.25)
    mock_get_client.return_value.run_task.return_value = {"tasks": [{"taskArn": "arn:task/1"}]}

    arn = svc.submit_ml_task(42, "model_a")

    assert arn == "arn:task/1"
    kwargs = mock_get_client.return_value.run_task.call_args.kwargs
    assert kwargs["cluster"] == "arn:cluster"
    assert kwargs["taskDefinition"] == "arn:taskdef"
    vpc = kwargs["networkConfiguration"]["awsvpcConfiguration"]
    assert vpc["subnets"] == ["subnet-a", "subnet-b"]
    assert vpc["securityGroups"] == ["sg-1"]
    override = kwargs["overrides"]["containerOverrides"][0]
    assert override["name"] == "ml-worker"
    env = {e["name"]: e["value"] for e in override["environment"]}
    assert env == {
        "JOB_ID": "42",
        "MODEL_FOLDER": "model_a",
        "CASE_PRESENCE_THRESHOLD": "0.3",
        "GROUP_CLASSIFIER_THRESHOLD": "0.25",
    }


@patch("app.services.ml_task_service._get_client")
def test_submit_ml_task_raises_on_failures(mock_get_client):
    mock_get_client.return_value.run_task.return_value = {
        "tasks": [],
        "failures": [{"reason": "RESOURCE:CPU"}],
    }
    with pytest.raises(RuntimeError, match="RESOURCE:CPU"):
        svc.submit_ml_task(1)


def _describe(mock_get_client, task: dict) -> None:
    mock_get_client.return_value.describe_tasks.return_value = {"tasks": [task]}


@pytest.mark.parametrize("status", ["PROVISIONING", "PENDING", "ACTIVATING"])
@patch("app.services.ml_task_service._get_client")
def test_get_task_status_pending(mock_get_client, status):
    _describe(mock_get_client, {"lastStatus": status})
    assert svc.get_task_status("arn") == (svc.STATE_PENDING, None)


@patch("app.services.ml_task_service._get_client")
def test_get_task_status_running(mock_get_client):
    _describe(mock_get_client, {"lastStatus": "RUNNING"})
    assert svc.get_task_status("arn") == (svc.STATE_RUNNING, None)


@patch("app.services.ml_task_service._get_client")
def test_get_task_status_succeeded_on_exit_zero(mock_get_client):
    _describe(mock_get_client, {
        "lastStatus": "STOPPED",
        "containers": [{"name": "ml-worker", "exitCode": 0}],
    })
    assert svc.get_task_status("arn") == (svc.STATE_SUCCEEDED, None)


@patch("app.services.ml_task_service._get_client")
def test_get_task_status_failed_reports_reason(mock_get_client):
    _describe(mock_get_client, {
        "lastStatus": "STOPPED",
        "stoppedReason": "Essential container exited",
        "containers": [{"name": "ml-worker", "exitCode": 137, "reason": "OutOfMemoryError"}],
    })
    state, err = svc.get_task_status("arn")
    assert state == svc.STATE_FAILED
    assert "137" in err and "OutOfMemoryError" in err


@patch("app.services.ml_task_service._get_client")
def test_get_task_status_failed_when_task_missing(mock_get_client):
    mock_get_client.return_value.describe_tasks.return_value = {
        "tasks": [],
        "failures": [{"reason": "MISSING"}],
    }
    assert svc.get_task_status("arn") == (svc.STATE_FAILED, "MISSING")


@patch("app.services.ml_task_service._get_client")
def test_stop_ml_task_swallows_errors(mock_get_client):
    mock_get_client.return_value.stop_task.side_effect = Exception("already stopped")
    svc.stop_ml_task("arn")  # must not raise
