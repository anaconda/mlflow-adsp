import uuid
from typing import Dict, List
from unittest.mock import MagicMock

import pytest
from mlflow.entities import RunStatus

from ae5_tools.api import AEUserSession
from mlflow_adsp import ADSPMLFlowPluginError, ADSPSubmittedRun, AEProjectJobRunStateType


@pytest.fixture(scope="function")
def get_token_fixture():
    return {
        "access_token": str(uuid.uuid4()),
        "refresh_token": str(uuid.uuid4()),
    }


@pytest.fixture(scope="function")
def get_ae_user_session(get_token_fixture) -> AEUserSession:
    user_session = AEUserSession(
        hostname="MOCK-HOSTNAME", username="MOCK-AE-USERNAME", password="MOCK-AE-USER-PASSWORD"
    )
    user_session._load = MagicMock()
    user_session._sdata = get_token_fixture
    return user_session


@pytest.fixture(scope="function")
def submitted_run(get_ae_user_session) -> ADSPSubmittedRun:
    mock_mlflow_run_id: str = str(uuid.uuid4())
    mock_adsp_job_id: str = str(uuid.uuid4())
    mock_response: Dict = {}

    submitted_run: ADSPSubmittedRun = ADSPSubmittedRun(
        ae_session=get_ae_user_session,
        mlflow_run_id=mock_mlflow_run_id,
        adsp_job_id=mock_adsp_job_id,
        response=mock_response,
    )

    return submitted_run


def test_get_log_missing_run(submitted_run):
    """ """
    # Set up the scenario
    submitted_run.ae_session.job_runs = MagicMock(return_value=[])

    # Execute the test
    with pytest.raises(ADSPMLFlowPluginError):
        submitted_run.get_log()


def test_get_log(submitted_run):
    # Set up the scenario
    mock_job_runs: List[Dict] = [{"id": str(uuid.uuid4())}]
    mock_job_log: str = "mock logs"
    submitted_run.ae_session.job_runs = MagicMock(return_value=mock_job_runs)
    submitted_run.ae_session.run_log = MagicMock(return_value=mock_job_log)

    # Execute the test
    generated_log: str = submitted_run.get_log()

    # Review the results
    assert generated_log == mock_job_log
    submitted_run.ae_session.job_runs.assert_called_once_with(ident=submitted_run.adsp_job_id)
    submitted_run.ae_session.run_log.assert_called_once_with(ident=mock_job_runs[0]["id"])


def test_wait_on_job_run(submitted_run):
    # Set up the scenario
    mock_job_runs: List[Dict] = [{"id": str(uuid.uuid4()), "state": AEProjectJobRunStateType.FAILED}]
    submitted_run.ae_session.job_runs = MagicMock(return_value=mock_job_runs)

    # Execute the test
    submitted_run._wait_on_job_run()

    # Review the results
    submitted_run.ae_session.job_runs.assert_called_once_with(ident=submitted_run.adsp_job_id)


def test_wait_on_job_run_with_a_wait(submitted_run):
    # Set up the scenario
    submitted_run.wait_interval = 1
    mock_job_runs_response_one: List[Dict] = [{"id": str(uuid.uuid4()), "state": AEProjectJobRunStateType.RUNNING}]
    mock_job_runs_response_two: List[Dict] = [{"id": str(uuid.uuid4()), "state": AEProjectJobRunStateType.COMPLETED}]
    submitted_run.ae_session.job_runs = MagicMock(side_effect=[mock_job_runs_response_one, mock_job_runs_response_two])

    # Execute the test
    submitted_run._wait_on_job_run()

    # Review the results
    assert submitted_run.ae_session.job_runs.call_count == 2


def test_wait_success(submitted_run):
    # Set up the scenario
    mock_job_runs: List[Dict] = [{"id": str(uuid.uuid4()), "state": AEProjectJobRunStateType.COMPLETED}]
    submitted_run.ae_session.job_runs = MagicMock(return_value=mock_job_runs)

    # Execute the test
    result: bool = submitted_run.wait()

    # Review the results
    assert result is True


def test_wait_failure(submitted_run):
    # Set up the scenario
    mock_job_runs: List[Dict] = [{"id": str(uuid.uuid4()), "state": AEProjectJobRunStateType.FAILED}]
    submitted_run.ae_session.job_runs = MagicMock(return_value=mock_job_runs)

    # Execute the test
    result: bool = submitted_run.wait()

    # Review the results
    assert result is False


def test_get_status(submitted_run):
    id: str = str(uuid.uuid4())
    test_cases = [
        {"id": id, "state": AEProjectJobRunStateType.INITIAL, "expected_result": RunStatus.RUNNING},
        {"id": id, "state": AEProjectJobRunStateType.RUNNING, "expected_result": RunStatus.RUNNING},
        {"id": id, "state": AEProjectJobRunStateType.FAILED, "expected_result": RunStatus.FAILED},
        {"id": id, "state": AEProjectJobRunStateType.STOPPED, "expected_result": RunStatus.KILLED},
        {"id": id, "state": AEProjectJobRunStateType.COMPLETED, "expected_result": RunStatus.FINISHED},
    ]

    for test_case in test_cases:
        # Set up the scenario
        mock_job_runs: List[Dict] = [{"id": test_case["id"], "state": test_case["state"]}]
        submitted_run.ae_session.job_runs = MagicMock(return_value=mock_job_runs)

        # Execute the test
        status = submitted_run.get_status()

        # Review the results
        assert status == test_case["expected_result"]


def test_get_status_gracefully_fails(submitted_run):
    # Set up the scenario
    mock_job_runs: List[Dict] = [{"id": str(uuid.uuid4()), "state": "MOCK-STATE"}]
    submitted_run.ae_session.job_runs = MagicMock(return_value=mock_job_runs)

    # Execute the test
    with pytest.raises(ADSPMLFlowPluginError):
        submitted_run.get_status()


def test_cancel(submitted_run):
    # Set up the scenario
    submitted_run.ae_session.run_stop = MagicMock()

    # Execute the test
    submitted_run.cancel()

    # Review the results
    mock: MagicMock = submitted_run.ae_session.run_stop
    mock.assert_called_once_with(ident=submitted_run.adsp_job_id)


def test_run_id_property(submitted_run):
    assert submitted_run.run_id == submitted_run.mlflow_run_id
