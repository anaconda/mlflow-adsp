import os
from unittest.mock import MagicMock

import mlflow

from mlflow_adsp import create_unique_name
from mlflow_adsp.common.tracking import _resolve_experiment_name, upsert_experiment


def test_create_unique_name():
    name: str = "mock-name"
    result = create_unique_name(name=name)
    assert name in result
    assert name != result


def test_resolve_experiment_name():
    # Scenario 1: No name supplied and no environment variable.
    result: str = _resolve_experiment_name()
    assert result == "Default"

    # Scenario 2: No name supplied, but environment variable defined.
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "MOCK_MLFLOW_EXPERIMENT_NAME"
    result: str = _resolve_experiment_name()
    del os.environ["MLFLOW_EXPERIMENT_NAME"]
    assert result == "MOCK_MLFLOW_EXPERIMENT_NAME"

    # Scenario 3: Name is supplied
    result = _resolve_experiment_name(name="Mock-Name")
    assert result == "Mock-Name"


def test_upsert_experiment(monkeypatch):
    # Scenario 1: Experiment exists
    def get_experiment_by_name_mock(name: str):
        assert name == "Default"
        mock = MagicMock()
        mock.experiment_id = "MOCK-ID"
        return mock

    monkeypatch.setattr(mlflow, "get_experiment_by_name", get_experiment_by_name_mock)
    id: str = upsert_experiment()
    assert id == "MOCK-ID"

    # Scenario 2: Experiment does not exist
    def get_experiment_by_name_none_mock(name: str):
        assert name == "Default"
        return None

    def create_experiment_mock(name: str):
        assert name == "Default"
        return "MOCK-ID-TWO"

    monkeypatch.setattr(mlflow, "get_experiment_by_name", get_experiment_by_name_none_mock)
    monkeypatch.setattr(mlflow, "create_experiment", create_experiment_mock)
    id = upsert_experiment()
    assert id == "MOCK-ID-TWO"
