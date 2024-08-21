import logging
import os
import subprocess
import unittest
import warnings
from copy import deepcopy
from unittest.mock import MagicMock

import mlflow
import pytest
import requests

import mlflow_adsp
from mlflow_adsp import ADSPMLFlowPluginError, EndpointManager, EndpointManagerParameters

warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class MockPOpen:
    pid: int = 0

    def __init__(self, *args, **kwargs):
        self.terminated: bool = False

        self.stdout = MagicMock()
        self.stdout.readline = MagicMock(return_value=b"")

        self.stderr = MagicMock()
        self.stderr.readline = MagicMock(return_value=b"")

    def communicate(self, *args, **kwargs):
        return self.stdout, self.stderr

    def poll(self, *args, **kwargs):
        if self.terminated:
            return 1
        return None


class GetMock:
    status_code: int = 200

    def __init__(self, *args, **kwargs):
        pass


def test_non_reloadable_uri(monkeypatch):
    model_uri: str = "runs:/some_run/model"
    params: EndpointManagerParameters = EndpointManagerParameters(model_uri=model_uri)
    client: MagicMock = MagicMock()

    mock_popen = MockPOpen()

    def mock_it(args, cwd):
        results = deepcopy(args)
        mock_popen.results = results
        return mock_popen

    monkeypatch.setattr(subprocess, "Popen", mock_it)
    monkeypatch.setattr(requests, "get", GetMock)

    # execute the test
    EndpointManager(client=client, params=params)

    assert mock_popen.results == [
        "mlflow",
        "models",
        "serve",
        "--env-manager",
        "local",
        "--host",
        "0.0.0.0",
        "--port",
        "8086",
        "--model-uri",
        "runs:/some_run/model",
    ]


def test_non_reloadable_uri_mlserver(monkeypatch):
    # set up the test
    model_uri: str = "runs:/some_run/model"
    params: EndpointManagerParameters = EndpointManagerParameters(model_uri=model_uri, enable_mlserver=True)
    client: MagicMock = MagicMock()

    mock_popen = MockPOpen()

    def mock_it(args, cwd):
        results = deepcopy(args)
        mock_popen.results = results
        return mock_popen

    monkeypatch.setattr(subprocess, "Popen", mock_it)
    monkeypatch.setattr(requests, "get", GetMock)

    # execute the test
    EndpointManager(client=client, params=params)

    # review the results
    assert mock_popen.results == [
        "mlflow",
        "models",
        "serve",
        "--env-manager",
        "local",
        "--host",
        "0.0.0.0",
        "--port",
        "8086",
        "--model-uri",
        "runs:/some_run/model",
        "--enable-mlserver",
    ]


def test_update_is_not_reloadable(monkeypatch):
    # test setup
    model_uri: str = "runs:/some_run/model"
    params: EndpointManagerParameters = EndpointManagerParameters(model_uri=model_uri)
    client: MagicMock = MagicMock()
    monkeypatch.setattr(subprocess, "Popen", MockPOpen)
    monkeypatch.setattr(requests, "get", GetMock)
    manager = EndpointManager(client=client, params=params)
    manager._launch = MagicMock()

    # Execute the test
    manager.update()

    # Review the results
    manager._launch.assert_not_called()


def test_update_is_reloadable_does_not_need_to_reload(monkeypatch):
    model_uri: str = "models:/registry/Production"
    params: EndpointManagerParameters = EndpointManagerParameters(model_uri=model_uri)
    client: MagicMock = MagicMock()

    class MockVersion:
        version: str = "mock-version"

    client.get_latest_versions = MagicMock(return_value=[MockVersion()])
    mock_popen: MagicMock = MagicMock()
    mock_launch: MagicMock = MagicMock()

    monkeypatch.setattr(subprocess, "Popen", mock_popen)
    monkeypatch.setattr(mlflow_adsp.EndpointManager, "_launch", mock_launch)

    manager = EndpointManager(client=client, params=params)
    manager._launch = MagicMock()

    manager.update()
    manager._launch.assert_not_called()
    assert client.get_latest_versions.call_count == 2


def test_update_is_reloadable_needs_to_reload(monkeypatch):
    model_uri: str = "models:/registry/Production"
    params: EndpointManagerParameters = EndpointManagerParameters(model_uri=model_uri)

    client: MagicMock = MagicMock()

    class MockVersion:
        def __init__(self, version_str: str):
            self.version: str = version_str

    client.get_latest_versions = MagicMock(
        side_effect=[[MockVersion(version_str="mock-version-2")], [MockVersion(version_str="mock-version-3")]]
    )

    mock_popen: MagicMock = MagicMock()
    monkeypatch.setattr(subprocess, "Popen", mock_popen)

    mock_launch: MagicMock = MagicMock()
    monkeypatch.setattr(mlflow_adsp.EndpointManager, "_launch", mock_launch)

    mock_stop: MagicMock = MagicMock()
    monkeypatch.setattr(mlflow_adsp.EndpointManager, "_stop", mock_stop)

    manager = EndpointManager(client=client, params=params)

    manager.update()
    mock_stop.assert_called_once()
    mock_launch.call_count = 2
    assert client.get_latest_versions.call_count == 2


def test_get_latest_version_with_stage(monkeypatch):
    model_uri: str = "models:/registry/Production"
    params: EndpointManagerParameters = EndpointManagerParameters(model_uri=model_uri)
    client: MagicMock = MagicMock()

    class MockVersion:
        version: str = "mock-version-stage"

    client.get_latest_versions = MagicMock(return_value=[MockVersion()])
    mock_popen: MagicMock = MagicMock()
    mock_launch: MagicMock = MagicMock()

    monkeypatch.setattr(subprocess, "Popen", mock_popen)
    monkeypatch.setattr(mlflow_adsp.EndpointManager, "_launch", mock_launch)

    manager = EndpointManager(client=client, params=params)

    assert manager._get_latest_version() == "mock-version-stage"


def test_get_latest_version_with_stage_gracefully_fails_assertion(monkeypatch):
    model_uri: str = "models:/registry/Production"
    params: EndpointManagerParameters = EndpointManagerParameters(model_uri=model_uri)
    client: MagicMock = MagicMock()

    client.get_latest_versions = MagicMock(side_effect=[AssertionError()])
    mock_popen: MagicMock = MagicMock()
    mock_launch: MagicMock = MagicMock()

    monkeypatch.setattr(subprocess, "Popen", mock_popen)
    monkeypatch.setattr(mlflow_adsp.EndpointManager, "_launch", mock_launch)

    with pytest.raises(ADSPMLFlowPluginError):
        EndpointManager(client=client, params=params)


def test_get_latest_version_with_stage_gracefully_fails_mlflow(monkeypatch):
    model_uri: str = "models:/registry/Production"
    params: EndpointManagerParameters = EndpointManagerParameters(model_uri=model_uri)
    client: MagicMock = MagicMock()

    client.get_latest_versions = MagicMock(side_effect=[mlflow.exceptions.MlflowException("Boom!")])
    mock_popen: MagicMock = MagicMock()
    mock_launch: MagicMock = MagicMock()

    monkeypatch.setattr(subprocess, "Popen", mock_popen)
    monkeypatch.setattr(mlflow_adsp.EndpointManager, "_launch", mock_launch)

    with pytest.raises(ADSPMLFlowPluginError):
        EndpointManager(client=client, params=params)


def test_get_latest_version_with_alias(monkeypatch):
    model_uri: str = "models:/registry@winner"
    params: EndpointManagerParameters = EndpointManagerParameters(model_uri=model_uri)
    client: MagicMock = MagicMock()

    class MockVersion:
        version: str = "mock-version-alias"

    client.get_model_version_by_alias = MagicMock(return_value=MockVersion())
    mock_popen: MagicMock = MagicMock()
    mock_launch: MagicMock = MagicMock()

    monkeypatch.setattr(subprocess, "Popen", mock_popen)
    monkeypatch.setattr(mlflow_adsp.EndpointManager, "_launch", mock_launch)

    manager = EndpointManager(client=client, params=params)
    assert manager._get_latest_version() == "mock-version-alias"


def test_get_latest_version_fails_gracefully(monkeypatch):
    model_uri: str = "models:/registry/Production"
    params: EndpointManagerParameters = EndpointManagerParameters(model_uri=model_uri)
    client: MagicMock = MagicMock()

    class MockVersion:
        version: str = "mock-version-stage"

    client.get_latest_versions = MagicMock(return_value=[MockVersion()])
    mock_popen: MagicMock = MagicMock()
    mock_launch: MagicMock = MagicMock()

    monkeypatch.setattr(subprocess, "Popen", mock_popen)
    monkeypatch.setattr(mlflow_adsp.EndpointManager, "_launch", mock_launch)

    manager = EndpointManager(client=client, params=params)
    manager.metadata.type = "MOCK-TYPE"
    with pytest.raises(ADSPMLFlowPluginError) as context:
        manager._get_latest_version()
        assert context.value == "Unknown designation type encountered: (MOCK-TYPE)"


def test_get_latest_version_fails_gracefully_mlflow(monkeypatch):
    model_uri: str = "models:/registry/Production"
    params: EndpointManagerParameters = EndpointManagerParameters(model_uri=model_uri)
    client: MagicMock = MagicMock()

    client.get_latest_versions = MagicMock(side_effect=[mlflow.exceptions.MlflowException("Boom!")])
    mock_popen: MagicMock = MagicMock()
    mock_launch: MagicMock = MagicMock()

    monkeypatch.setattr(subprocess, "Popen", mock_popen)
    monkeypatch.setattr(mlflow_adsp.EndpointManager, "_launch", mock_launch)

    with pytest.raises(ADSPMLFlowPluginError):
        EndpointManager(client=client, params=params)


def test_replace(monkeypatch):
    manager = MagicMock()
    EndpointManager._replace(manager=manager)

    call: unittest.mock._Call = manager.mock_calls[0]
    func_name = list(call)[0]
    assert func_name == "update"


def test_replace_loop(monkeypatch):
    manager = MagicMock()
    manager.update = MagicMock(side_effect=[Exception("Boom!"), ""])

    mock_exception_handler = MagicMock()
    monkeypatch.setattr(mlflow_adsp.EndpointManager, "_exception_handler", mock_exception_handler)

    EndpointManager._replace(manager=manager)

    assert manager.update.call_count == 2
    mock_exception_handler.assert_called_once()


def test_exception_handler():
    error: Exception = Exception("Boom!")
    attempt: int = 1

    updated_attempt = EndpointManager._exception_handler(error=error, attempt=attempt)
    assert updated_attempt == 2


def test_process_launch_gracefully_fails(monkeypatch):
    model_uri: str = "runs:/some_run/model"
    params: EndpointManagerParameters = EndpointManagerParameters(model_uri=model_uri)
    client: MagicMock = MagicMock()

    def mock_it(args, cwd):
        raise ADSPMLFlowPluginError("Boom!")

    monkeypatch.setattr(subprocess, "Popen", mock_it)

    with pytest.raises(ADSPMLFlowPluginError) as context:
        # execute the test
        EndpointManager(client=client, params=params)


def test_process_launch_gracefully_fails_with_other_exceptions(monkeypatch):
    model_uri: str = "runs:/some_run/model"
    params: EndpointManagerParameters = EndpointManagerParameters(model_uri=model_uri)
    client: MagicMock = MagicMock()

    def mock_it(args, cwd):
        raise Exception("Boom!")

    monkeypatch.setattr(subprocess, "Popen", mock_it)

    with pytest.raises(ADSPMLFlowPluginError) as context:
        # execute the test
        EndpointManager(client=client, params=params)


def test_poll_network_service_several_codes(monkeypatch):
    # test setup
    model_uri: str = "runs:/some_run/model"
    params: EndpointManagerParameters = EndpointManagerParameters(model_uri=model_uri)
    client: MagicMock = MagicMock()
    monkeypatch.setattr(subprocess, "Popen", MockPOpen)

    class LocalGetMock:
        status_code_mock = [500, 403, 200]

        @property
        def status_code(self):
            """ """
            return LocalGetMock.status_code_mock.pop()

        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(requests, "get", LocalGetMock)

    EndpointManager(client=client, params=params)


def test_poll_network_service_gracefully_fails(monkeypatch):
    # test setup
    model_uri: str = "runs:/some_run/model"
    params: EndpointManagerParameters = EndpointManagerParameters(model_uri=model_uri)
    client: MagicMock = MagicMock()
    monkeypatch.setattr(subprocess, "Popen", MockPOpen)

    def mock_get(url, timeout):
        raise requests.exceptions.ConnectionError("Boom!")

    monkeypatch.setattr(requests, "get", mock_get)
    monkeypatch.setattr(os, "kill", MagicMock)

    with pytest.raises(ADSPMLFlowPluginError):
        EndpointManager(client=client, params=params)


def test_process_launch_wrapper_gracefully_fails(monkeypatch):
    model_uri: str = "runs:/some_run/model"
    params: EndpointManagerParameters = EndpointManagerParameters(model_uri=model_uri)
    client: MagicMock = MagicMock()

    mock_popen = MockPOpen()
    mock_popen.terminated = True

    def mock_it(args, cwd):
        results = deepcopy(args)
        mock_popen.results = results
        return mock_popen

    monkeypatch.setattr(subprocess, "Popen", mock_it)
    monkeypatch.setattr(requests, "get", GetMock)

    # execute the test
    with pytest.raises(ADSPMLFlowPluginError):
        EndpointManager(client=client, params=params)


def test_process_launch_wrapper_should_timeout(monkeypatch):
    # test setup
    model_uri: str = "runs:/some_run/model"
    params: EndpointManagerParameters = EndpointManagerParameters(model_uri=model_uri)
    client: MagicMock = MagicMock()
    monkeypatch.setattr(subprocess, "Popen", MockPOpen)

    class LocalGetMock:
        @property
        def status_code(self):
            """ """
            return 400

        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(requests, "get", LocalGetMock)

    with pytest.raises(ADSPMLFlowPluginError):
        EndpointManager(client=client, params=params)
